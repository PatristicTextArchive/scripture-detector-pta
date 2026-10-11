import io
import json
import os
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SD_DB_DIR", tempfile.mkdtemp())

import llm_backends as lb  # noqa: E402

Q = {"text": "ἐκολλήθη ἡ ψυχή μου ὀπίσω σου", "resolved_references": ["ps_62:9"], "quote_type": "full", "marked": True}
MT = "μακάριοι οἱ εἰρηνοποιοί, ὅτι αὐτοὶ υἱοὶ θεοῦ κληθήσονται"


# ── parsing ──────────────────────────────────────────────────────────────────

def test_parse_plain_wrapped_and_fenced():
    arr = json.dumps([Q])
    assert lb.parse_quotes(arr)[0]["resolved_references"] == ["ps_62:9"]
    assert lb.parse_quotes(json.dumps({"quotes": [Q]}))[0]["text"] == Q["text"]
    assert lb.parse_quotes("```json\n" + arr + "\n```")[0]["quote_type"] == "full"
    assert lb.parse_quotes("Here you go:\n" + arr)[0]["text"] == Q["text"]          # prose before the array
    assert lb.parse_quotes("Here you go:\n" + arr + " done")[0]["text"] == Q["text"]


def test_parse_strips_think_blocks_and_normalises():
    raw = "<think>Let me see... [not json]</think>\n" + json.dumps({"results": [
        {"quote": "Φ", "references": "mt_5:9, mt_5:10", "type": "paraphrasis", "marked": "true"},
        {"text": "no refs", "resolved_references": []},
        {"text": "bad type", "resolved_references": ["a_1:1"], "quote_type": "weird"},
        "junk"]})
    out = lb.parse_quotes(raw)
    assert [o["text"] for o in out] == ["Φ", "bad type"]
    assert out[0]["resolved_references"] == ["mt_5:9", "mt_5:10"] and out[0]["quote_type"] == "paraphrase"
    assert out[0]["marked"] is True and out[1]["quote_type"] == "allusion" and out[1]["marked"] is None
    assert lb.parse_quotes("template ate the opening tag... </think>" + json.dumps([Q]))[0]["text"] == Q["text"]


def test_parse_salvages_truncated_output():
    full = json.dumps({"quotes": [Q, dict(Q, text="zweites")]})
    cut = full[: full.index("zweites") + 5]             # second object cut in the middle
    out = lb.parse_quotes(cut)
    assert [o["text"] for o in out] == [Q["text"]]
    assert lb.parse_quotes("") == [] and lb.parse_quotes("no json at all") == []


# ── OpenAI-compatible server ─────────────────────────────────────────────────

class Fake:
    def __init__(self, behave):
        self.requests, self.behave, self.props = [], behave, None
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                fake.requests.append({"path": self.path, "body": None, "auth": None})
                if self.path == "/props" and fake.props is not None:
                    raw, status = json.dumps(fake.props).encode(), 200
                else:
                    raw, status = b"{}", 404
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                status, payload = fake.behave(body)
                raw = json.dumps(payload).encode() if not isinstance(payload, str) else payload.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def reply(content, finish="stop"):
    return 200, {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


@pytest.fixture()
def server():
    made = []

    def make(behave):
        f = Fake(behave)
        made.append(f)
        return f
    lb._NO_SCHEMA.clear()
    lb._CTX_CACHE.clear()
    yield make
    for f in made:
        f.close()


def test_openai_compat_sends_schema_model_and_key(server):
    f = server(lambda b: reply(json.dumps({"quotes": [Q]})))
    res = lb.OpenAICompat(f.url, "qwen3", "secret", max_tokens=123).generate("PROMPT")
    r = f.requests[0]
    assert r["path"] == "/v1/chat/completions" and r["auth"] == "Bearer secret"
    b = r["body"]
    assert b["model"] == "qwen3" and b["temperature"] == 0 and b["max_tokens"] == 123
    assert b["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert b["response_format"]["type"] == "json_schema" and "quotes" in b["response_format"]["json_schema"]["schema"]["properties"]
    assert not res.truncated and lb.parse_quotes(res.text)[0]["text"] == Q["text"]


def test_openai_compat_falls_back_when_schema_is_rejected(server):
    def behave(b):
        if "response_format" in b:
            return 400, {"error": {"message": "unsupported response_format json_schema"}}
        return reply(json.dumps([Q]))
    f = server(behave)
    be = lb.OpenAICompat(f.url)
    assert lb.parse_quotes(be.generate("x").text)
    assert len(f.requests) == 2 and "response_format" not in f.requests[1]["body"] and "model" not in f.requests[1]["body"]
    be.generate("y")                                    # remembered: no second attempt with the schema
    assert len(f.requests) == 3 and "response_format" not in f.requests[2]["body"]


def test_openai_compat_flags_truncation_and_reports_errors(server):
    f = server(lambda b: reply('{"quotes": [', "length"))
    assert lb.OpenAICompat(f.url).generate("x").truncated is True
    f2 = server(lambda b: (500, "boom"))
    with pytest.raises(lb.BackendError, match="HTTP 500"):
        lb.OpenAICompat(f2.url).generate("x")
    f3 = server(lambda b: (401, {"error": "no"}))
    with pytest.raises(lb.BackendError, match="rejected the API key"):
        lb.OpenAICompat(f3.url).generate("x")
    with pytest.raises(lb.BackendError, match="Cannot connect.*llama-server"):
        lb.OpenAICompat("http://127.0.0.1:9/v1", timeout=2).generate("x")


# ── llama-cpp-python (in-process) ────────────────────────────────────────────

def test_llamacpp_loads_once_and_uses_a_json_schema(tmp_path, monkeypatch):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    calls = {"init": [], "chat": []}

    class FakeLlama:
        def __init__(self, **kw):
            calls["init"].append(kw)

        def create_chat_completion(self, **kw):
            calls["chat"].append(kw)
            return {"choices": [{"message": {"content": json.dumps({"quotes": [Q]})}, "finish_reason": "stop"}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    lb._LLAMAS.clear()
    be = lb.LlamaCpp(str(gguf), n_ctx=4096, n_gpu_layers=0, max_tokens=999)
    be.generate("a")
    res = be.generate("b")
    assert len(calls["init"]) == 1 and calls["init"][0]["n_ctx"] == 4096 and calls["init"][0]["n_gpu_layers"] == 0
    assert calls["chat"][0]["response_format"]["type"] == "json_object" and calls["chat"][0]["max_tokens"] == 999
    assert lb.parse_quotes(res.text)[0]["text"] == Q["text"]


def test_llamacpp_clear_errors(tmp_path, monkeypatch):
    with pytest.raises(lb.BackendError, match="not found"):
        lb.LlamaCpp(str(tmp_path / "missing.gguf")).generate("x")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)      # import fails
    lb._LLAMAS.clear()
    with pytest.raises(lb.BackendError, match="pip install llama-cpp-python"):
        lb.LlamaCpp(str(gguf)).generate("x")


# ── whole app with a local server ────────────────────────────────────────────

def test_app_end_to_end_with_local_server(server, monkeypatch):
    import app as sd
    TEI = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>Lokal</title></titleStmt>
<publicationStmt><p>x</p></publicationStmt><sourceDesc><p>y</p></sourceDesc></fileDesc></teiHeader>
<text><body><div type="edition" xml:lang="grc" n="urn:cts:pta:pta0001.pta001.pta-grc1"><p>Der Herr sagt:
<quote>{MT}</quote>. Und die Seele ruft: {Q['text']}.</p></div></body></text></TEI>""".encode()

    seen = {}

    def behave(body):
        prompt = body["messages"][0]["content"]
        seen["prompt"] = prompt
        return reply(json.dumps({"quotes": [
            {"text": MT, "resolved_references": ["mt_5:9"], "quote_type": "full", "marked": True},
            dict(Q)]}))
    f = server(behave)
    sd.app.config["TESTING"] = True
    c = sd.app.test_client()
    c.post("/api/settings", json={"api_provider": "openai", "local_base_url": f.url, "local_model": "qwen3"})
    assert sd.is_local() and sd.get_chunk_chars() == 3500
    sid = c.post("/api/import/tei", data={"files": (io.BytesIO(TEI), "l.xml")},
                 content_type="multipart/form-data").get_json()["imported"][0]["id"]
    assert c.post(f"/api/sources/{sid}/process").get_json()["count"] == 2
    assert "Answer with JSON only" in seen["prompt"] and MT in seen["prompt"]       # format hint + <quote> hint
    out = c.get(f"/api/sources/{sid}/export/tei").data.decode()
    assert out.count("<quote") == 2                                                 # existing quote enriched, one new
    assert 'cRef="NA:Mt:5:9"' in out and 'cRef="LXX:Ps:62:9"' in out
    assert b"qwen3" in c.get(f"/viewer/{sid}").data and b'id="model-select"' not in c.get(f"/viewer/{sid}").data

    t = c.post("/api/test-backend").get_json()
    assert t["status"] == "ok" and t["provider"] == "openai"
    c.post("/api/settings", json={"api_provider": "gemini", "local_model": ""})      # empty value clears
    assert c.get("/api/settings").get_json()["local_model"] == ""
    assert b'id="model-select"' in c.get(f"/viewer/{sid}").data


def test_truncated_local_answer_splits_the_text_and_keeps_all_quotes(server, monkeypatch):
    import app as sd
    para_a = "α " * 450 + Q["text"]
    para_b = "β " * 450 + MT
    text = para_a + "\n" + para_b
    sizes = []

    def behave(body):
        t = body["messages"][0]["content"].split("TEXT:\n", 1)[1]
        sizes.append(len(t))
        found = [dict(Q) if Q["text"] in t else None,
                 {"text": MT, "resolved_references": ["mt_5:9"], "quote_type": "full", "marked": False} if MT in t else None]
        found = [x for x in found if x]
        if len(t) > 1200:                                # too much: answer is cut off after the first item
            cut = json.dumps({"quotes": found})
            return reply(cut[: cut.rindex("{") - 2] if len(found) > 1 else cut[:30], "length")
        return reply(json.dumps({"quotes": found}))
    f = server(behave)
    sd.app.config["TESTING"] = True
    c = sd.app.test_client()
    c.post("/api/settings", json={"api_provider": "openai", "local_base_url": f.url, "chunk_chars": "5000"})
    sid = c.post("/api/sources", json={"name": "Lang", "text": text}).get_json()["id"]
    r = c.post(f"/api/sources/{sid}/process")
    assert r.get_json()["count"] == 2, r.get_json()
    assert len(sizes) == 3 and sizes[0] > 1200 and max(sizes[1:]) < 1200            # one call, then two halves
    c.post("/api/settings", json={"api_provider": "gemini"})


def test_default_context_size_is_16384():
    assert lb.LlamaCpp("x.gguf").n_ctx == 16384
    assert lb.make_backend("llamacpp", {"gguf_path": "x.gguf"}).n_ctx == 16384
    assert lb.make_backend("llamacpp", {"gguf_path": "x.gguf", "n_ctx": "4096"}).n_ctx == 4096      # explicit value wins
    import app as sd
    c = sd.app.test_client()
    assert 'id="n-ctx" min="2048" step="1024" value="16384"' in c.get("/settings").data.decode()


# ── context window ───────────────────────────────────────────────────────────

LLAMA_OVERFLOW = {"error": {"code": 400, "type": "exceed_context_size_error",
                            "message": "the request exceeds the available context size, try increasing it",
                            "n_prompt_tokens": 5100, "n_ctx": 4096}}
VLLM_OVERFLOW = {"object": "error", "code": 400, "message":
                 "This model's maximum context length is 8192 tokens. However, you requested 12000 tokens "
                 "(4000 in the messages, 8000 in the completion). Please reduce the length of the messages or completion."}


def test_overflow_is_recognised_in_llamacpp_and_vllm_format(server):
    f = server(lambda b: (400, LLAMA_OVERFLOW))
    with pytest.raises(lb.ContextOverflow) as e:
        lb.OpenAICompat(f.url).generate("x")
    assert (e.value.n_prompt, e.value.n_ctx) == (5100, 4096)
    assert "-np 1" in e.value.user_message() and "5100" in e.value.user_message()
    assert lb._overflow_from(400, json.dumps(VLLM_OVERFLOW)).n_ctx == 8192
    assert lb._overflow_from(400, "unsupported response_format json_schema") is None      # not an overflow
    assert lb._overflow_from(200, "exceeds the context") is None


def test_answer_limit_is_reduced_when_prompt_plus_answer_do_not_fit(server):
    def behave(b):
        if b["max_tokens"] > 5000:
            return 400, {"message": "maximum context length is 8192 tokens. However, you requested 12000 tokens "
                                    "(4000 in the messages, 8000 in the completion)", "n_prompt_tokens": 4000, "n_ctx": 8192}
        return reply(json.dumps({"quotes": []}))
    f = server(behave)
    res = lb.OpenAICompat(f.url, max_tokens=8000, structured="off").generate("x")
    assert lb.parse_quotes(res.text) == []
    assert [r["body"]["max_tokens"] for r in f.requests] == [8000, 8192 - 4000 - 64]


def test_context_info_from_props(server):
    f = server(lambda b: reply("{}"))
    f.props = {"default_generation_settings": {"n_ctx": 4096}, "total_slots": 4}
    assert lb.OpenAICompat(f.url).context_info() == {"n_ctx": 4096, "slots": 4}
    f2 = server(lambda b: reply("{}"))                 # no /props (vLLM, LM Studio): unknown, not an error
    assert lb.OpenAICompat(f2.url).context_info() is None


def test_app_shrinks_the_request_on_context_overflow_and_reports_small_contexts(server):
    import app as sd
    seen = []

    def behave(body):
        prompt = body["messages"][0]["content"]
        seen.append(len(prompt))
        if len(prompt) > 9000:                                   # a window that is too small for the full prompt
            return 400, LLAMA_OVERFLOW
        if "TEXT:\n" not in prompt:                              # the connection test
            return reply('{"quotes": []}')
        t = prompt.split("TEXT:\n", 1)[1]
        found = [dict(Q)] if Q["text"] in t else []
        return reply(json.dumps({"quotes": found}))
    f = server(behave)
    f.props = {"default_generation_settings": {"n_ctx": 4096}, "total_slots": 4}
    sd.app.config["TESTING"] = True
    c = sd.app.test_client()
    c.post("/api/settings", json={"api_provider": "openai", "local_base_url": f.url, "chunk_chars": "5000", "rag": "on"})
    text = "ἄλλος λόγος " * 150 + Q["text"] + " καὶ " + "τέλος λόγου " * 120
    sid = c.post("/api/sources", json={"name": "Eng", "text": text}).get_json()["id"]
    r = c.post(f"/api/sources/{sid}/process")
    assert r.status_code == 200 and r.get_json()["count"] == 1, r.get_json()
    assert max(seen) > 9000 and seen[-1] <= 9000 and len(seen) > 2          # tried, shrank, succeeded
    posts = [r for r in f.requests if r["body"]]
    assert posts[-1]["body"]["max_tokens"] == 4096 // 3                    # capped by the context the server reports

    t = c.post("/api/test-backend").get_json()
    assert t["context"] == 4096 and t["slots"] == 4 and "-np 1" in t["warning"]

    f.behave = lambda body: (400, LLAMA_OVERFLOW)                         # nothing fits at all
    sid2 = c.post("/api/sources", json={"name": "Tiny", "text": "κείμενον " * 30}).get_json()["id"]
    r = c.post(f"/api/sources/{sid2}/process")
    assert r.status_code == 500 and "-np 1" in r.get_json()["error"]
    c.post("/api/settings", json={"api_provider": "gemini", "chunk_chars": ""})
