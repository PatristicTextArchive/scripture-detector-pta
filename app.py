import os
import sys

# ── parse custom flags BEFORE importing database (which reads env vars) ───────
# The --cache-db CLI flag sets the env var so gunicorn deployments can instead
# set SCRIPTURE_DETECTOR_CACHE_DB=1 in their environment directly.
_argv = sys.argv[1:]
if "--cache-db" in _argv:
    os.environ["SCRIPTURE_DETECTOR_CACHE_DB"] = "1"
sys.argv = [sys.argv[0]] + [a for a in _argv if a != "--cache-db"]

import io
import json
import re
import uuid
import zipfile
from datetime import date
from pathlib import Path

from flask import Flask, render_template, jsonify, request, redirect, url_for, Response, session
from werkzeug.middleware.proxy_fix import ProxyFix
from google import genai

import database   # imported as module so we can write to database.session_local
from database import (
    init_db,
    create_source, get_source, get_all_sources, delete_source,
    add_quote, update_quote, delete_quote, get_quotes_for_source,
    delete_quotes_for_source, delete_quotes_in_range,
    get_setting, set_setting, get_all_settings,
    get_book_distribution, get_quote_type_distribution, get_dashboard_data,
    search_sources, get_quote_source_id,
)
import corpora
import greek
import tei_doc

app = Flask(__name__)

# Trust the X-Forwarded-* headers from reverse proxies (HuggingFace, nginx…).
# This lets Flask see the real HTTPS scheme so secure cookies work correctly.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# ── Secret key ────────────────────────────────────────────────────────────────
# Set SD_SECRET_KEY in the environment (HF Spaces → Settings → Secrets) so
# sessions survive server restarts.  A random key is used as a safe fallback
# (sessions reset whenever the server restarts).
app.secret_key = os.environ.get("SD_SECRET_KEY") or os.urandom(32)

# ── Session cookie settings ───────────────────────────────────────────────────
# HuggingFace Spaces embeds the app inside an <iframe>.  Browsers block
# SameSite=Lax cookies in cross-site iframes, which would create a new session
# on every request and make the per-user database invisible.
# SameSite=None + Secure=True is the correct fix for iframe deployments.
# We detect HTTPS via the ProxyFix-corrected request scheme at runtime so
# that local HTTP development still works without secure cookies.
_BEHIND_PROXY = bool(os.environ.get("SD_BEHIND_PROXY") or
                     os.environ.get("SCRIPTURE_DETECTOR_CACHE_DB"))

if _BEHIND_PROXY:
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
    app.config["SESSION_COOKIE_SECURE"]   = True

_CACHE_MODE = bool(os.environ.get("SCRIPTURE_DETECTOR_CACHE_DB"))


@app.before_request
def _bind_session_db():
    """Assign (and persist) a unique DB session ID for this browser session."""
    if not _CACHE_MODE:
        return
    if "_db_sid" not in session:
        session["_db_sid"] = str(uuid.uuid4())
        session.permanent = True
    database.session_local.session_id = session["_db_sid"]

PROJECT_ROOT = Path(__file__).resolve().parent
CHUNK_CHARS = int(os.environ.get("SD_CHUNK_CHARS", "12000"))

MODELS = [
    {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash"},
    {"id": "gemini-3.7-flash", "name": "Gemini 3.7 Flash"},
    {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash"},
    {"id": "gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash Lite"},
]


def source_lang(source: dict | None) -> str:
    """Language of a source: stored (xml:lang of a TEI file / chosen) or guessed from the script."""
    if not source:
        return corpora.DEFAULT_LANG
    return source.get("lang") or corpora.guess_lang(source["text"])


def lang_arg() -> str:
    return corpora.normalize_lang(request.args.get("lang")) or corpora.DEFAULT_LANG


# ── AI Integration ───────────────────────────────────────────────────────────

def get_genai_client():
    settings = get_all_settings()
    provider = settings.get("api_provider", "gemini")
    if provider == "vertex":
        project_id = settings.get("vertex_project_id", "")
        location = settings.get("vertex_location", "global")
        if not project_id:
            raise ValueError("Vertex AI Project ID not configured. Go to Settings.")
        return genai.Client(vertexai=True, project=project_id, location=location)
    else:
        api_key = settings.get("gemini_api_key", "")
        if not api_key:
            raise ValueError("Gemini API Key not configured. Go to Settings.")
        return genai.Client(api_key=api_key)


def get_model() -> str:
    return get_setting("model", "gemini-3.8-flash")


def build_prompt(text: str, corpus: "corpora.Corpus", hints: list[str] | None = None) -> str:
    meta = corpus.meta
    def label(v):
        extra = f" / {v['greek_name']}" if v.get("greek_name") else ""
        return f"{v['name']}{extra}"
    mapping_lines = "\n".join(f"  {label(v)} -> {k}" for k, v in sorted(corpus.mapping.items()))
    examples = meta.get("example_refs", "mt_5:9, ps_22:1, 1cor_15:33")
    ex_book = examples.split("_")[0]

    formulas = meta.get("intro_formulas", "")
    intro = ""
    if formulas:
        intro = f"""
Biblical quotations are often introduced by formulas such as: {formulas}.
The introductory words are NOT part of the quotation: extract only the quoted words. Treat such a formula as a signal that a quotation follows (or precedes), and set marked = true when one introduces the passage.
"""
    hint_block = ""
    if hints:
        items = "\n".join(f"  {i}. {h[:600]}" for i, h in enumerate(hints[:200], 1))
        hint_block = f"""
The document already marks the following passages as quotations (quote elements). Check each of them: if it is a biblical quotation, paraphrase or allusion, report it with exactly this text (verbatim, same boundaries) and its references; ignore those that are not biblical.
{items}
"""
    return f"""You are an expert in biblical texts and scripture detection.

The Bible text you work with is {meta.get('corpus', 'the Bible')}. Given the following {meta.get('language', '')} text, identify ALL scriptural (Biblical) quotations, partial quotations, paraphrases, and clear allusions to specific Bible verses.

For each identified passage:
1. Extract the EXACT text as it appears in the document — preserve the original spelling, accents, breathings, punctuation, and word order verbatim. Never normalise, translate or complete the text.
2. Identify the specific Bible verse(s) being quoted or referenced.
3. Classify the type of reuse as one of:
   - "full"       — a complete or near-complete verse quoted verbatim.
   - "partial"    — a recognisable portion of a verse, quoted with minor variation or truncation.
   - "paraphrase" — the biblical content is clearly restated in different words while preserving the meaning.
   - "allusion"   — a brief phrase, thematic echo, or indirect reference to a specific verse.
4. Say whether the passage is explicitly marked as a quotation in the document (introduced by a formula such as "it is written", "the prophet says", "as the apostle says", or an attribution) -> marked = true; otherwise marked = false.
{intro}{hint_block}
Reference format: book_chapter:verse  (e.g. {examples})
CRITICAL: Each reference must be a SINGLE verse with a plain verse number. Never use ranges like {ex_book}_15:1-2.
Instead, list each verse separately: {ex_book}_15:1, {ex_book}_15:2.

Valid book codes: {", ".join(corpus.codes)}

Book name to code mapping:
{mapping_lines}

{meta.get('prompt_notes', '')}

Important:
- Include both direct quotes and partial quotes / paraphrases / allusions.
- A single passage may reference multiple Bible verses — list all of them.
- Be thorough — identify even brief allusions to specific verses, but do not invent references: only report passages you can tie to specific verses.
- The extracted text must be a verbatim substring of the input document.

TEXT:
{text}"""


def extract_quotes_with_gemini(text: str, lang: str = corpora.DEFAULT_LANG,
                               hints: list[str] | None = None) -> list[dict]:
    client = get_genai_client()
    model = get_model()
    prompt = build_prompt(text, corpora.get_corpus(lang), hints)

    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "text": {
                    "type": "STRING",
                    "description": "The exact text of the scriptural quote as it appears verbatim in the document",
                },
                "resolved_references": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "List of Bible verse references in format book_chapter:verse",
                },
                "quote_type": {
                    "type": "STRING",
                    "enum": ["full", "partial", "paraphrase", "allusion"],
                    "description": "Type of biblical reuse",
                },
                "marked": {
                    "type": "BOOLEAN",
                    "description": "True if the passage is explicitly marked/introduced as a quotation",
                },
            },
            "required": ["text", "resolved_references", "quote_type"],
        },
    }

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    )

    quotes = json.loads(response.text)
    for q in quotes:
        q["resolved_references"] = greek.expand_ranges(q.get("resolved_references", []))
    return quotes


def find_spans(text: str, quotes: list[dict], corpus: "corpora.Corpus", offset: int = 0) -> list[dict]:
    """Locate quotes in *text* (accent/case/punctuation-insensitive) and shift
    the spans by *offset*; refs are validated and each quote gets a score."""
    index = greek.SpanIndex(text)
    results = []
    for quote in quotes:
        qt = quote["text"]
        refs, _dropped = corpus.validate_refs(quote.get("resolved_references", []))
        if not refs:
            continue
        hit = index.find(qt)
        found_text = text[hit[0]:hit[1]] if hit else qt
        results.append({
            "text": found_text,
            "span_start": hit[0] + offset if hit else None,
            "span_end": hit[1] + offset if hit else None,
            "resolved_references": refs,
            "quote_type": quote.get("quote_type", "allusion"),
            "marked": quote.get("marked"),
            "score": corpus.score_for(found_text, refs),
        })
    return results


def detect_quotes(text: str, lang: str, offset: int = 0, hints: list[dict] | None = None) -> list[dict]:
    """Run detection chunk-wise over *text* (which starts at *offset* in the
    source).  *hints*: passages already marked as <quote> in the source, with
    absolute positions; they are passed to the model chunk by chunk."""
    corpus = corpora.get_corpus(lang)
    found: list[dict] = []
    for start, end in greek.chunk_text(text, CHUNK_CHARS):
        chunk = text[start:end]
        if not chunk.strip():
            continue
        lo, hi = offset + start, offset + end
        chunk_hints = [h["text"] for h in (hints or []) if h["start"] >= lo and h["end"] <= hi]
        found.extend(find_spans(chunk, extract_quotes_with_gemini(chunk, lang, chunk_hints), corpus, lo))
    return found


def hints_for(source: dict) -> list[dict]:
    if source.get("tei_xml"):
        try:
            return tei_doc.existing_quotes(bytes(source["tei_xml"]))
        except tei_doc.TeiError:
            return []
    return []


def compute_segments(text: str, annotations: list[dict]) -> list[dict]:
    boundaries: set[int] = {0, len(text)}
    for a in annotations:
        if a["span_start"] is not None:
            boundaries.add(a["span_start"])
            boundaries.add(a["span_end"])
    ordered = sorted(boundaries)

    segments = []
    for i in range(len(ordered) - 1):
        start, end = ordered[i], ordered[i + 1]
        ann_ids = [
            j for j, a in enumerate(annotations)
            if a["span_start"] is not None
            and a["span_start"] <= start and end <= a["span_end"]
        ]
        segments.append({
            "text": text[start:end],
            "start": start,
            "end": end,
            "annotation_ids": ann_ids,
        })
    return segments


# ── Page Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def sources_page():
    return render_template("sources.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/viewer/<int:source_id>")
def viewer(source_id: int):
    source = get_source(source_id)
    if not source:
        return redirect(url_for("sources_page"))
    settings = get_all_settings()
    current_model = settings.get("model", "gemini-3-pro-preview")
    return render_template("viewer.html", source=source, models=MODELS, current_model=current_model,
                           source_lang=source_lang(source))


@app.route("/settings")
def settings_page():
    return render_template("settings.html", models=MODELS)


@app.route("/about")
def about_page():
    return render_template("about.html")


# ── API: Search ───────────────────────────────────────────────────────────────

@app.route("/api/search", methods=["POST"])
def api_search():
    body    = request.get_json() or {}
    filters = body.get("filters", [])
    logic   = body.get("logic", "AND")
    if logic not in ("AND", "OR"):
        logic = "AND"

    # Normalise filter values to lowercase; reject empties
    clean = [
        {"type": f.get("type", "text"), "value": str(f.get("value", "")).strip().lower()}
        for f in filters
        if str(f.get("value", "")).strip()
    ]

    result = search_sources(clean, logic)

    # Enrich book_distribution entries with human-readable names
    bm = corpora.merged_mapping()
    for src in result["results"]:
        for item in src.get("book_distribution", []):
            info = bm.get(item["book_code"], {})
            item["book_name"]  = info.get("name",      item["book_code"])
            item["testament"]  = info.get("testament", "")

    return jsonify(result)


# ── API: Sources ─────────────────────────────────────────────────────────────

@app.route("/api/sources", methods=["POST"])
def api_create_source():
    body = request.get_json()
    name = body.get("name", "").strip()
    text = body.get("text", "").strip()
    if not name or not text:
        return jsonify({"error": "Name and text are required"}), 400
    lang = corpora.normalize_lang(body.get("lang")) or corpora.guess_lang(text)
    source_id = create_source(name, text, lang=lang)
    return jsonify({"id": source_id, "name": name, "lang": lang})


@app.route("/api/sources/<int:source_id>", methods=["GET"])
def api_get_source(source_id: int):
    source = get_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404

    quotes = get_quotes_for_source(source_id)
    lang = source_lang(source)
    corpus = corpora.get_corpus(lang)
    verses = corpus.verses
    bm = corpus.mapping

    annotations = []
    for q in quotes:
        refs = [r["reference"] for r in q["references"]]
        verse_lookup = []
        for ref in refs:
            ref_lower = ref.strip().lower()
            book_code = ref_lower.split("_")[0] if "_" in ref_lower else ""
            keys = corpus.resolve_ref(ref_lower)
            verse_lookup.append({
                "ref": ref_lower,
                "text": " ".join(verses.get(k, "") for k in keys),
                "book_name": bm.get(book_code, {}).get("name", ""),
            })
        annotations.append({
            "id": q["id"],
            "span_start": q["span_start"],
            "span_end": q["span_end"],
            "quote_text": q["quote_text"],
            "quote_type": q["quote_type"],
            "score": q.get("score"),
            "marked": q.get("marked"),
            "refs": refs,
            "verses": verse_lookup,
        })

    segments = compute_segments(source["text"], annotations)

    return jsonify({
        "source": {"id": source["id"], "name": source["name"], "is_tei": bool(source.get("tei_xml")),
                   "lang": lang},
        "segments": segments,
        "annotations": annotations,
    })


@app.route("/api/sources/<int:source_id>", methods=["DELETE"])
def api_delete_source(source_id: int):
    delete_source(source_id)
    return jsonify({"status": "ok"})


# ── API: TEI Export (single source) ──────────────────────────────────────────

def tidy_warnings(warnings: list[str], limit: int = 300) -> list[str]:
    """Merge identical messages ("… (×3)") and cap the list."""
    counts: dict[str, int] = {}
    for w in warnings:
        counts[w] = counts.get(w, 0) + 1
    out = [f"{w} (×{n})" if n > 1 else w for w, n in counts.items()]
    if len(out) > limit:
        out = out[:limit] + [f"… and {len(out) - limit} more"]
    return out


def export_source_tei(source: dict) -> tuple[bytes, list[str]]:
    """TEI (PTA encoding) for a source: written into the original file for
    imported TEI, into a generated PTA skeleton for pasted text."""
    lang = source_lang(source)
    corpus = corpora.get_corpus(lang)
    quotes = get_quotes_for_source(source["id"])
    anns = [{
        "id": q["id"],
        "span_start": q["span_start"],
        "span_end": q["span_end"],
        "quote_type": q["quote_type"],
        "marked": None if q.get("marked") is None else bool(q["marked"]),
        "refs": [r["reference"] for r in q["references"]],
    } for q in quotes]
    info = corpus.book_info()
    if source.get("tei_xml"):
        return tei_doc.annotate_tei(bytes(source["tei_xml"]), anns, info, expected_text=source["text"])
    xml, omap = tei_doc.plain_to_tei(source["name"], source["text"], lang)
    for a in anns:
        if a["span_start"] is not None and a["span_end"] is not None:
            a["span_start"], a["span_end"] = omap[a["span_start"]], omap[a["span_end"]]
    return tei_doc.annotate_tei(xml, anns, info)


@app.route("/api/sources/<int:source_id>/export/tei")
def api_export_tei(source_id: int):
    source = get_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    try:
        xml_bytes, warnings = export_source_tei(source)
    except tei_doc.TeiError as exc:
        return jsonify({"error": str(exc)}), 409

    safe_name = re.sub(r"[^\w\-]", "_", source["name"])[:60]
    warnings = tidy_warnings(warnings)
    if request.args.get("json"):
        # the viewer wants the complete list of notes, which does not fit into a header
        return jsonify({"filename": f"{safe_name}.xml", "xml": xml_bytes.decode("utf-8"), "warnings": warnings})
    return Response(
        xml_bytes,
        mimetype="application/xml",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.xml"',
            "X-SD-Warning-Count": str(len(warnings)),
        },
    )


# ── API: ZIP Export (all sources) ────────────────────────────────────────────

@app.route("/api/export/zip")
def api_export_zip():
    sources = get_all_sources()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "version":      "2.0",
            "app":          "Scripture Detector",
            "format":       "TEI, PTA encoding guidelines",
            "exported":     date.today().isoformat(),
            "source_count": len(sources),
            "sources":      [],
        }
        for idx, src in enumerate(sources, start=1):
            full_source = get_source(src["id"])
            safe_name = re.sub(r"[^\w\-]", "_", src["name"])[:60]
            fname = f"sources/{idx:04d}_{safe_name}.xml"
            entry = {"filename": fname, "name": src["name"]}
            try:
                xml_bytes, warnings = export_source_tei(full_source)
                zf.writestr(fname, xml_bytes)
                if warnings:
                    entry["warnings"] = tidy_warnings(warnings)
            except tei_doc.TeiError as exc:
                entry["error"] = str(exc)
            manifest["sources"].append(entry)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

    buf.seek(0)
    today = date.today().strftime("%Y%m%d")
    return Response(
        buf.read(),
        mimetype="application/zip",
        headers={
            "Content-Disposition":
                f'attachment; filename="scripture_detector_export_{today}.zip"'
        },
    )


# ── API: ZIP Import ───────────────────────────────────────────────────────────

def import_tei_bytes(xml_bytes: bytes, fallback_name: str) -> dict:
    """Create a source from a TEI file (keeping the original XML for export).
    The language comes from xml:lang of the first <div> in <text> (grc / lat);
    annotations written by an earlier export are restored."""
    data = tei_doc.read_tei(xml_bytes, None, fallback_name)
    if not data.text.strip():
        raise tei_doc.TeiError("The <text> of this TEI file contains no text")
    raw = (data.lang or "").strip()
    if raw:
        lang = corpora.normalize_lang(raw)
        if lang is None:
            raise tei_doc.TeiError(
                f"Unsupported language '{raw}' (xml:lang of the first <div>); supported: grc, lat")
    else:
        lang = corpora.guess_lang(data.text)
    corpus = corpora.get_corpus(lang)
    source_id = create_source(data.name, data.text, tei_xml=xml_bytes, lang=lang)
    restored = 0
    for ann in data.annotations:
        refs = corpus.refs_from_crefs(ann["crefs"])
        if not refs:
            continue
        add_quote(
            source_id=source_id, span_start=ann["span_start"], span_end=ann["span_end"],
            quote_text=ann["quote_text"], quote_type=ann["quote_type"], references=refs,
            score=corpus.score_for(ann["quote_text"], refs), marked=ann["marked"],
        )
        restored += 1
    return {"id": source_id, "name": data.name, "annotations": restored, "lang": lang}


@app.route("/api/import/tei", methods=["POST"])
def api_import_tei():
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        return jsonify({"error": "No file uploaded"}), 400
    imported, errors = [], []
    for f in files:
        try:
            imported.append(import_tei_bytes(f.read(), Path(f.filename or "TEI document").stem))
        except tei_doc.TeiError as exc:
            errors.append({"file": f.filename, "error": str(exc)})
    status = 200 if imported or not errors else 400
    return jsonify({"status": "ok" if imported else "error", "imported": imported, "errors": errors}), status


@app.route("/api/import/zip", methods=["POST"])
def api_import_zip():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".zip"):
        return jsonify({"error": "File must be a .zip archive"}), 400

    imported = 0
    errors   = []

    try:
        with zipfile.ZipFile(io.BytesIO(f.read()), "r") as zf:
            tei_names = sorted(n for n in zf.namelist() if n.lower().endswith(".xml"))
            for name in tei_names:
                try:
                    import_tei_bytes(zf.read(name), Path(name).stem)
                    imported += 1
                except Exception as exc:
                    errors.append({"file": name, "error": str(exc)})
    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid or corrupt ZIP file"}), 400

    return jsonify({"status": "ok", "imported": imported, "errors": errors})


# ── API: Processing ──────────────────────────────────────────────────────────

def _store_detected(source_id: int, items: list[dict]) -> int:
    for q in items:
        add_quote(
            source_id=source_id,
            span_start=q["span_start"],
            span_end=q["span_end"],
            quote_text=q["text"],
            quote_type=q["quote_type"],
            references=q["resolved_references"],
            score=q.get("score"),
            marked=q.get("marked"),
        )
    return len(items)


@app.route("/api/sources/<int:source_id>/process", methods=["POST"])
def api_process_source(source_id: int):
    source = get_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    try:
        items = detect_quotes(source["text"], source_lang(source), hints=hints_for(source))
        delete_quotes_for_source(source_id)
        return jsonify({"status": "ok", "count": _store_detected(source_id, items)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sources/<int:source_id>/process-selection", methods=["POST"])
def api_process_selection(source_id: int):
    source = get_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    body = request.get_json()
    start = body.get("start")
    end = body.get("end")
    if start is None or end is None:
        return jsonify({"error": "start and end are required"}), 400

    selection_text = source["text"][start:end]
    if not selection_text.strip():
        return jsonify({"error": "Empty selection"}), 400

    try:
        items = detect_quotes(selection_text, source_lang(source), offset=start, hints=hints_for(source))
        delete_quotes_in_range(source_id, start, end)
        return jsonify({"status": "ok", "count": _store_detected(source_id, items)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Quotes ──────────────────────────────────────────────────────────────

@app.route("/api/quotes", methods=["POST"])
def api_add_quote():
    body = request.get_json()
    source_id = body.get("source_id")
    if not source_id:
        return jsonify({"error": "source_id is required"}), 400
    refs = body.get("references", [])
    corpus = corpora.get_corpus(source_lang(get_source(source_id)))
    quote_id = add_quote(
        source_id=source_id,
        span_start=body.get("span_start"),
        span_end=body.get("span_end"),
        quote_text=body.get("quote_text", ""),
        quote_type=body.get("quote_type", "allusion"),
        references=refs,
        score=corpus.score_for(body.get("quote_text", ""), refs),
        marked=body.get("marked"),
    )
    return jsonify({"id": quote_id})


@app.route("/api/quotes/<int:quote_id>", methods=["PUT"])
def api_update_quote(quote_id: int):
    body = request.get_json()
    score = None
    if body.get("references") is not None and body.get("quote_text"):
        sid = get_quote_source_id(quote_id)
        corpus = corpora.get_corpus(source_lang(get_source(sid)) if sid else None)
        score = corpus.score_for(body["quote_text"], body["references"])
    update_quote(
        quote_id=quote_id,
        quote_text=body.get("quote_text"),
        quote_type=body.get("quote_type"),
        span_start=body.get("span_start"),
        span_end=body.get("span_end"),
        references=body.get("references"),
        score=score,
        marked=body.get("marked"),
    )
    return jsonify({"status": "ok"})


@app.route("/api/quotes/<int:quote_id>", methods=["DELETE"])
def api_delete_quote(quote_id: int):
    delete_quote(quote_id)
    return jsonify({"status": "ok"})


# ── API: Dashboard ───────────────────────────────────────────────────────────

@app.route("/api/dashboard")
def api_dashboard():
    data = get_dashboard_data()
    data["book_distribution"] = get_book_distribution()
    data["type_distribution"] = get_quote_type_distribution()

    bm = corpora.merged_mapping()
    for item in data["book_distribution"]:
        info = bm.get(item["book_code"], {})
        item["book_name"] = info.get("name", item["book_code"])
        item["testament"] = info.get("testament", "")

    for source in data["sources"]:
        source["type_distribution"] = get_quote_type_distribution(source["id"])
        source["book_distribution"] = get_book_distribution(source["id"])

    return jsonify(data)


@app.route("/api/sources/<int:source_id>/distribution")
def api_source_distribution(source_id: int):
    bm = corpora.get_corpus(source_lang(get_source(source_id))).mapping
    book_dist = get_book_distribution(source_id)
    for item in book_dist:
        info = bm.get(item["book_code"], {})
        item["book_name"] = info.get("name", item["book_code"])
        item["testament"] = info.get("testament", "")
    type_dist = get_quote_type_distribution(source_id)
    return jsonify({"book_distribution": book_dist, "type_distribution": type_dist})


# ── API: Settings ────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    settings = get_all_settings()
    if "gemini_api_key" in settings:
        key = settings["gemini_api_key"]
        settings["gemini_api_key_masked"] = (
            key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
        )
    return jsonify(settings)


@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    body = request.get_json()
    for key, value in body.items():
        if value is not None and str(value).strip() != "":
            set_setting(key, str(value))
    return jsonify({"status": "ok"})


@app.route("/api/book-mapping")
def api_book_mapping():
    lang = request.args.get("lang")
    return jsonify(corpora.get_corpus(lang_arg()).mapping if lang else corpora.merged_mapping())


@app.route("/api/bible/books")
def api_bible_books():
    bm = corpora.get_corpus(lang_arg()).mapping
    books = [
        {"code": code, "name": info["name"], "testament": info["testament"]}
        for code, info in sorted(bm.items(), key=lambda x: x[1]["name"])
    ]
    return jsonify(books)


@app.route("/api/bible/<book_code>/chapters")
def api_bible_chapters(book_code: str):
    bs = corpora.get_corpus(lang_arg()).structure
    book = book_code.strip().lower()
    if book not in bs:
        return jsonify([])
    return jsonify(sorted(bs[book].keys(), key=greek.verse_sort_key))


@app.route("/api/bible/<book_code>/<chapter>/verses")
def api_bible_verses_list(book_code: str, chapter: str):
    bs = corpora.get_corpus(lang_arg()).structure
    book, chapter = book_code.strip().lower(), chapter.strip().lower()
    if book not in bs or chapter not in bs[book]:
        return jsonify([])
    return jsonify(bs[book][chapter])


init_db()

if __name__ == "__main__":
    _cache_db = bool(os.environ.get("SCRIPTURE_DETECTOR_CACHE_DB"))
    _host  = os.environ.get("SD_HOST", "127.0.0.1")
    _port  = int(os.environ.get("SD_PORT", "5001"))
    # Disable the reloader in cache-db mode: the reloader forks the process,
    # which would create a fresh in-memory database and lose all data.
    _debug = not _cache_db
    app.run(debug=_debug, port=_port, host=_host, use_reloader=_debug)
