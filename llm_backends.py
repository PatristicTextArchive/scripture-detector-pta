"""Local language-model backends (no Ollama needed).

``openai``    any OpenAI-compatible chat endpoint: llama.cpp ``llama-server``,
              vLLM, LM Studio, SGLang, LocalAI, text-generation-webui, ...
``llamacpp``  a GGUF model loaded in-process with ``llama-cpp-python``
              (no server at all; Qwen, Gemma, Llama, Mistral, ... as GGUF).

Both return the model's raw text; ``parse_quotes`` turns it into the list of
quotations, tolerating markdown fences, <think> blocks and truncated JSON.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("scripture-detector")

QUOTE_TYPES = ("full", "partial", "paraphrase", "allusion")

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "exact text of the passage as it appears in the document"},
        "resolved_references": {"type": "array", "items": {"type": "string"},
                                "description": "Bible verses in the format book_chapter:verse"},
        "quote_type": {"type": "string", "enum": list(QUOTE_TYPES)},
        "marked": {"type": "boolean", "description": "explicitly introduced as a quotation"},
    },
    "required": ["text", "resolved_references", "quote_type", "marked"],
}
# An object at the top level is accepted by every server (arrays are not).
QUOTES_SCHEMA = {
    "type": "object",
    "properties": {"quotes": {"type": "array", "items": ITEM_SCHEMA}},
    "required": ["quotes"],
}

FORMAT_HINT = """Answer with JSON only - no explanations, no markdown, no comments. Use exactly this format:
{"quotes": [{"text": "<verbatim passage>", "resolved_references": ["<book_chapter:verse>", "..."], "quote_type": "full|partial|paraphrase|allusion", "marked": true}]}
If the text contains no biblical passage, answer {"quotes": []}.
"""


class BackendError(RuntimeError):
    """A local backend could not produce an answer (message is shown to the user)."""


@dataclass
class Result:
    text: str
    truncated: bool = False          # the model hit its output limit


# ── parsing ──────────────────────────────────────────────────────────────────

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_LIST_KEYS = ("quotes", "passages", "results", "items", "references")


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def normalize_item(item) -> dict | None:
    """One model item -> {text, resolved_references, quote_type, marked}; None if unusable."""
    if not isinstance(item, dict):
        return None
    text = next((item[k] for k in ("text", "quote", "quote_text") if isinstance(item.get(k), str)), None)
    refs = next((item[k] for k in ("resolved_references", "references", "refs")
                 if item.get(k) not in (None, "", [])), [])
    if isinstance(refs, str):
        refs = [r for r in re.split(r"[,;\s]+", refs) if r]
    if not isinstance(refs, list):
        refs = []
    refs = [str(r).strip() for r in refs if str(r).strip()]
    if not text or not text.strip() or not refs:
        return None
    qtype = str(item.get("quote_type") or item.get("type") or "allusion").strip().lower()
    if qtype == "paraphrasis":
        qtype = "paraphrase"
    if qtype not in QUOTE_TYPES:
        qtype = "allusion"
    return {"text": text, "resolved_references": refs, "quote_type": qtype, "marked": _coerce_bool(item.get("marked"))}


def _salvage(t: str) -> list:
    """Complete objects of a (possibly truncated) JSON array."""
    start = t.find("[")
    if start == -1:
        return []
    dec, pos, out = json.JSONDecoder(), start + 1, []
    while pos < len(t):
        while pos < len(t) and t[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(t) or t[pos] == "]":
            break
        try:
            obj, pos = dec.raw_decode(t, pos)
        except ValueError:
            break
        out.append(obj)
    return out


def parse_quotes(text: str) -> list[dict]:
    """Model answer -> list of normalised quotes (see normalize_item)."""
    t = _THINK.sub("", text or "")
    if "</think>" in t:                                    # opening tag was swallowed by the template
        t = t.split("</think>")[-1]
    t = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", t.strip(), flags=re.I)
    try:
        data = json.loads(t)
    except ValueError:
        data = _salvage(t)
    if isinstance(data, dict):
        lst = next((data[k] for k in _LIST_KEYS if isinstance(data.get(k), list)), None)
        if lst is None:
            lst = next((v for v in data.values() if isinstance(v, list)), None)
        data = lst if lst is not None else [data]
    if not isinstance(data, list):
        return []
    return [q for q in (normalize_item(i) for i in data) if q]


# ── OpenAI-compatible servers ────────────────────────────────────────────────

_NO_SCHEMA: set[str] = set()          # base URLs whose server rejected response_format
_SCHEMA_WORDS = ("response_format", "json_schema", "schema", "grammar", "guided")


class _SchemaRejected(Exception):
    pass


class OpenAICompat:
    def __init__(self, base_url: str, model: str = "", api_key: str = "", max_tokens: int = 4096,
                 timeout: float = 600, structured: str = "auto"):
        self.base_url = (base_url or "").rstrip("/")
        self.model, self.api_key = model, api_key
        self.max_tokens, self.timeout, self.structured = max_tokens, timeout, structured

    def _post(self, body: dict, with_schema: bool) -> dict:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise BackendError(
                f"Cannot connect to the local server at {self.base_url}. Is it running? "
                f"(e.g. `llama-server -m model.gguf --port 8080` -> http://127.0.0.1:8080/v1)") from exc
        except requests.Timeout as exc:
            raise BackendError(f"The local server at {self.base_url} did not answer within {self.timeout:.0f} s. "
                               f"Use a smaller chunk size or a faster model, or raise the timeout in Settings.") from exc
        if r.status_code in (400, 422) and with_schema and any(w in r.text.lower() for w in _SCHEMA_WORDS):
            raise _SchemaRejected()
        if r.status_code in (401, 403):
            raise BackendError(f"The server at {self.base_url} rejected the API key (HTTP {r.status_code}).")
        if r.status_code != 200:
            raise BackendError(f"The local server returned HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError as exc:
            raise BackendError(f"The server at {self.base_url} did not return JSON: {r.text[:200]}") from exc

    def generate(self, prompt: str) -> Result:
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": self.max_tokens, "stream": False}
        if self.model:
            body["model"] = self.model
        use_schema = self.structured != "off" and self.base_url not in _NO_SCHEMA
        if use_schema:
            try:
                data = self._post({**body, "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "quotes", "strict": True, "schema": QUOTES_SCHEMA}}}, True)
            except _SchemaRejected:
                log.warning("Server %s does not accept response_format; continuing without it", self.base_url)
                _NO_SCHEMA.add(self.base_url)
                data = self._post(body, False)
        else:
            data = self._post(body, False)
        try:
            choice = data["choices"][0]
            return Result((choice.get("message") or {}).get("content") or "", choice.get("finish_reason") == "length")
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"Unexpected answer from {self.base_url}: {str(data)[:300]}") from exc


# ── in-process GGUF (llama-cpp-python) ───────────────────────────────────────

_LLAMAS: dict[tuple, object] = {}
_LLAMA_LOCK = threading.Lock()


class LlamaCpp:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1, max_tokens: int = 4096):
        self.model_path, self.n_ctx = model_path, n_ctx
        self.n_gpu_layers, self.max_tokens = n_gpu_layers, max_tokens

    def _load(self):
        key = (self.model_path, self.n_ctx, self.n_gpu_layers)
        if key not in _LLAMAS:
            if not self.model_path or not Path(self.model_path).is_file():
                raise BackendError(f"GGUF model file not found: '{self.model_path}'. Set the path in Settings.")
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise BackendError("llama-cpp-python is not installed. Install it with `pip install llama-cpp-python` "
                                   "(see the README for GPU builds), or use a local server instead.") from exc
            log.info("Loading %s (n_ctx=%s, n_gpu_layers=%s) ...", self.model_path, self.n_ctx, self.n_gpu_layers)
            _LLAMAS[key] = Llama(model_path=self.model_path, n_ctx=self.n_ctx,
                                 n_gpu_layers=self.n_gpu_layers, verbose=False)
        return _LLAMAS[key]

    def generate(self, prompt: str) -> Result:
        with _LLAMA_LOCK:                      # llama.cpp contexts are not thread-safe
            llm = self._load()
            try:
                out = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=self.max_tokens,
                    response_format={"type": "json_object", "schema": QUOTES_SCHEMA})
            except ValueError as exc:          # typically: prompt longer than n_ctx
                raise BackendError(f"The model could not process the request ({exc}). "
                                   f"Use a smaller chunk size or a larger context size (n_ctx).") from exc
        choice = out["choices"][0]
        return Result((choice.get("message") or {}).get("content") or "", choice.get("finish_reason") == "length")


def _int(v, default):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def make_backend(provider: str, s: dict):
    """Backend for the provider selected in Settings (*s* = all settings)."""
    max_tokens = _int(s.get("local_max_tokens"), 4096)
    if provider == "openai":
        return OpenAICompat(s.get("local_base_url") or "http://127.0.0.1:8080/v1", s.get("local_model", ""),
                            s.get("local_api_key", ""), max_tokens, _int(s.get("local_timeout"), 600),
                            s.get("local_structured", "auto"))
    if provider == "llamacpp":
        return LlamaCpp(s.get("gguf_path", ""), _int(s.get("n_ctx"), 8192), _int(s.get("n_gpu_layers"), -1), max_tokens)
    raise BackendError(f"Unknown local provider '{provider}'")
