import os
import sys

# ── parse custom flags BEFORE importing database (which reads env vars) ───────
# The --cache-db CLI flag sets the env var so gunicorn deployments can instead
# set SCRIPTURE_DETECTOR_CACHE_DB=1 in their environment directly.
_argv = sys.argv[1:]
if "--cache-db" in _argv:
    os.environ["SCRIPTURE_DETECTOR_CACHE_DB"] = "1"
sys.argv = [sys.argv[0]] + [a for a in _argv if a != "--cache-db"]

import csv
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
    search_sources,
)
from tei import source_to_tei, tei_to_source_data

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
BIBLE_TSV_PATH = PROJECT_ROOT / "data" / "bible.tsv"
BOOK_MAPPING_PATH = PROJECT_ROOT / "data" / "book_mapping.tsv"

MODELS = [
    {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro Preview"},
    {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash Preview"},
    {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro Preview"},
    {"id": "gemini-3.1-flash-lite-preview", "name": "Gemini 3.1 Flash Lite Preview"},
]

_bible_cache: dict[str, str] | None = None
_book_mapping_cache: dict[str, dict] | None = None
_bible_structure_cache: dict[str, dict[int, list[dict]]] | None = None


def bible_verses() -> dict[str, str]:
    global _bible_cache
    if _bible_cache is not None:
        return _bible_cache
    verses: dict[str, str] = {}
    with open(BIBLE_TSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            book = row["book_code"].strip().lower()
            chapter = str(int(row["chapter_number"]))
            verse = str(int(row["verse_index"]))
            verses[f"{book}_{chapter}:{verse}"] = row["text"]
    _bible_cache = verses
    return verses


def book_mapping() -> dict[str, dict]:
    global _book_mapping_cache
    if _book_mapping_cache is not None:
        return _book_mapping_cache
    mapping: dict[str, dict] = {}
    with open(BOOK_MAPPING_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            code = row["book_code"].strip().lower()
            mapping[code] = {"name": row["work_name"], "testament": row["testament"]}
    _book_mapping_cache = mapping
    return mapping


def bible_structure() -> dict[str, dict[int, list[dict]]]:
    global _bible_structure_cache
    if _bible_structure_cache is not None:
        return _bible_structure_cache
    structure: dict[str, dict[int, list[dict]]] = {}
    with open(BIBLE_TSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            book = row["book_code"].strip().lower()
            chapter = int(row["chapter_number"])
            verse = int(row["verse_index"])
            text = row["text"]
            if book not in structure:
                structure[book] = {}
            if chapter not in structure[book]:
                structure[book][chapter] = []
            structure[book][chapter].append({"verse": verse, "text": text})
    for book in structure:
        for chapter in structure[book]:
            structure[book][chapter].sort(key=lambda v: v["verse"])
    _bible_structure_cache = structure
    return structure


def get_valid_book_codes() -> list[str]:
    return sorted(book_mapping().keys())


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
    return get_setting("model", "gemini-3-pro-preview")


def build_prompt(text: str) -> str:
    codes_str = ", ".join(get_valid_book_codes())
    bm = book_mapping()
    mapping_lines = "\n".join(f"  {v['name']} -> {k}" for k, v in sorted(bm.items()))
    return f"""You are an expert in biblical texts and scripture detection.

Given the following text, identify ALL scriptural (Biblical) quotations, partial quotations, paraphrases, and clear allusions to specific Bible verses.

For each identified passage:
1. Extract the EXACT text as it appears in the document — preserve the original spelling, punctuation, and word order verbatim.
2. Identify the specific Bible verse(s) being quoted or referenced.
3. Classify the type of reuse as one of:
   - "full"       — a complete or near-complete verse quoted verbatim.
   - "partial"    — a recognisable portion of a verse, quoted with minor variation or truncation.
   - "paraphrase" — the biblical content is clearly restated in different words while preserving the meaning.
   - "allusion"   — a brief phrase, thematic echo, or indirect reference to a specific verse.

Reference format: book_chapter:verse  (e.g. matt_5:9, ps_82:14, 1cor_15:33)
CRITICAL: Each reference must be a SINGLE verse. Never use ranges like matt_15:1-2.
Instead, list each verse separately: matt_15:1, matt_15:2.

Valid book codes: {codes_str}

Book name to code mapping:
{mapping_lines}

Important:
- Include both direct quotes and partial quotes / paraphrases / allusions.
- A single passage may reference multiple Bible verses — list all of them.
- Be thorough — identify even brief allusions to specific verses.
- The extracted text must be a verbatim substring of the input document.

TEXT:
{text}"""


_RANGE_RE = re.compile(r"^(.+_\d+):(\d+)-(\d+)$")


def expand_range_references(refs: list[str]) -> list[str]:
    expanded: list[str] = []
    for ref in refs:
        m = _RANGE_RE.match(ref.strip())
        if m:
            prefix, start, end = m.group(1), int(m.group(2)), int(m.group(3))
            for v in range(start, end + 1):
                expanded.append(f"{prefix}:{v}")
        else:
            expanded.append(ref.strip())
    return expanded


def extract_quotes_with_gemini(text: str) -> list[dict]:
    client = get_genai_client()
    model = get_model()
    prompt = build_prompt(text)

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
        q["resolved_references"] = expand_range_references(q.get("resolved_references", []))
    return quotes


def find_spans(text: str, quotes: list[dict]) -> list[dict]:
    results = []
    for quote in quotes:
        qt = quote["text"]
        idx = text.find(qt)
        if idx == -1:
            idx = text.lower().find(qt.lower())
        span_start = idx if idx != -1 else None
        span_end = (idx + len(qt)) if idx != -1 else None
        results.append({
            "text": qt,
            "span_start": span_start,
            "span_end": span_end,
            "resolved_references": quote["resolved_references"],
            "quote_type": quote.get("quote_type", "allusion"),
        })
    return results


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
    return render_template("viewer.html", source=source, models=MODELS, current_model=current_model)


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
    bm = book_mapping()
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
    source_id = create_source(name, text)
    return jsonify({"id": source_id, "name": name})


@app.route("/api/sources/<int:source_id>", methods=["GET"])
def api_get_source(source_id: int):
    source = get_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404

    quotes = get_quotes_for_source(source_id)
    verses = bible_verses()
    bm = book_mapping()

    annotations = []
    for q in quotes:
        refs = [r["reference"] for r in q["references"]]
        verse_lookup = []
        for ref in refs:
            ref_lower = ref.strip().lower()
            book_code = ref_lower.split("_")[0] if "_" in ref_lower else ""
            verse_lookup.append({
                "ref": ref_lower,
                "text": verses.get(ref_lower, ""),
                "book_name": bm.get(book_code, {}).get("name", ""),
            })
        annotations.append({
            "id": q["id"],
            "span_start": q["span_start"],
            "span_end": q["span_end"],
            "quote_text": q["quote_text"],
            "quote_type": q["quote_type"],
            "refs": refs,
            "verses": verse_lookup,
        })

    segments = compute_segments(source["text"], annotations)

    return jsonify({
        "source": {"id": source["id"], "name": source["name"]},
        "segments": segments,
        "annotations": annotations,
    })


@app.route("/api/sources/<int:source_id>", methods=["DELETE"])
def api_delete_source(source_id: int):
    delete_source(source_id)
    return jsonify({"status": "ok"})


# ── API: TEI Export (single source) ──────────────────────────────────────────

@app.route("/api/sources/<int:source_id>/export/tei")
def api_export_tei(source_id: int):
    source = get_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404

    quotes      = get_quotes_for_source(source_id)
    bm          = book_mapping()
    book_names  = {code: info["name"] for code, info in bm.items()}
    verses      = bible_verses()

    annotations = []
    for q in quotes:
        refs = [r["reference"] for r in q["references"]]
        annotations.append({
            "id":          q["id"],
            "span_start":  q["span_start"],
            "span_end":    q["span_end"],
            "quote_text":  q["quote_text"],
            "quote_type":  q["quote_type"],
            "refs":        refs,
        })

    xml_bytes = source_to_tei(source, annotations, book_names)
    safe_name = re.sub(r"[^\w\-]", "_", source["name"])[:60]
    filename  = f"{safe_name}.tei.xml"

    return Response(
        xml_bytes,
        mimetype="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── API: ZIP Export (all sources) ────────────────────────────────────────────

@app.route("/api/export/zip")
def api_export_zip():
    sources    = get_all_sources()
    bm         = book_mapping()
    book_names = {code: info["name"] for code, info in bm.items()}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "version":      "1.0",
            "app":          "Scripture Detector",
            "exported":     date.today().isoformat(),
            "source_count": len(sources),
            "sources":      [],
        }

        for idx, src in enumerate(sources, start=1):
            full_source = get_source(src["id"])
            quotes      = get_quotes_for_source(src["id"])

            annotations = []
            for q in quotes:
                annotations.append({
                    "id":         q["id"],
                    "span_start": q["span_start"],
                    "span_end":   q["span_end"],
                    "quote_text": q["quote_text"],
                    "quote_type": q["quote_type"],
                    "refs":       [r["reference"] for r in q["references"]],
                })

            xml_bytes = source_to_tei(full_source, annotations, book_names)
            safe_name = re.sub(r"[^\w\-]", "_", src["name"])[:60]
            fname = f"sources/{idx:04d}_{safe_name}.tei.xml"
            zf.writestr(fname, xml_bytes)
            manifest["sources"].append({"filename": fname, "name": src["name"]})

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
            # Collect TEI files (from sources/ sub-directory or root)
            tei_names = sorted(
                name for name in zf.namelist()
                if name.lower().endswith(".tei.xml") or name.lower().endswith(".xml")
            )

            for name in tei_names:
                try:
                    xml_bytes   = zf.read(name)
                    src_data    = tei_to_source_data(xml_bytes)
                    source_id   = create_source(src_data["name"], src_data["text"])
                    for ann in src_data["annotations"]:
                        if ann.get("refs"):
                            add_quote(
                                source_id  = source_id,
                                span_start = ann["span_start"],
                                span_end   = ann["span_end"],
                                quote_text = ann["quote_text"],
                                quote_type = ann["quote_type"],
                                references = ann["refs"],
                            )
                    imported += 1
                except Exception as exc:
                    errors.append({"file": name, "error": str(exc)})

    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid or corrupt ZIP file"}), 400

    return jsonify({
        "status":   "ok",
        "imported": imported,
        "errors":   errors,
    })


# ── API: Processing ──────────────────────────────────────────────────────────

@app.route("/api/sources/<int:source_id>/process", methods=["POST"])
def api_process_source(source_id: int):
    source = get_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    try:
        quotes = extract_quotes_with_gemini(source["text"])
        quotes_with_spans = find_spans(source["text"], quotes)
        delete_quotes_for_source(source_id)
        for q in quotes_with_spans:
            add_quote(
                source_id=source_id,
                span_start=q["span_start"],
                span_end=q["span_end"],
                quote_text=q["text"],
                quote_type=q["quote_type"],
                references=q["resolved_references"],
            )
        return jsonify({"status": "ok", "count": len(quotes_with_spans)})
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
        quotes = extract_quotes_with_gemini(selection_text)
        quotes_with_spans = find_spans(selection_text, quotes)
        delete_quotes_in_range(source_id, start, end)
        count = 0
        for q in quotes_with_spans:
            adj_start = (q["span_start"] + start) if q["span_start"] is not None else None
            adj_end = (q["span_end"] + start) if q["span_end"] is not None else None
            add_quote(
                source_id=source_id,
                span_start=adj_start,
                span_end=adj_end,
                quote_text=q["text"],
                quote_type=q["quote_type"],
                references=q["resolved_references"],
            )
            count += 1
        return jsonify({"status": "ok", "count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Quotes ──────────────────────────────────────────────────────────────

@app.route("/api/quotes", methods=["POST"])
def api_add_quote():
    body = request.get_json()
    source_id = body.get("source_id")
    if not source_id:
        return jsonify({"error": "source_id is required"}), 400
    quote_id = add_quote(
        source_id=source_id,
        span_start=body.get("span_start"),
        span_end=body.get("span_end"),
        quote_text=body.get("quote_text", ""),
        quote_type=body.get("quote_type", "allusion"),
        references=body.get("references", []),
    )
    return jsonify({"id": quote_id})


@app.route("/api/quotes/<int:quote_id>", methods=["PUT"])
def api_update_quote(quote_id: int):
    body = request.get_json()
    update_quote(
        quote_id=quote_id,
        quote_text=body.get("quote_text"),
        quote_type=body.get("quote_type"),
        span_start=body.get("span_start"),
        span_end=body.get("span_end"),
        references=body.get("references"),
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

    bm = book_mapping()
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
    bm = book_mapping()
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
    return jsonify(book_mapping())


@app.route("/api/bible/books")
def api_bible_books():
    bm = book_mapping()
    books = [
        {"code": code, "name": info["name"], "testament": info["testament"]}
        for code, info in sorted(bm.items(), key=lambda x: x[1]["name"])
    ]
    return jsonify(books)


@app.route("/api/bible/<book_code>/chapters")
def api_bible_chapters(book_code: str):
    bs = bible_structure()
    book = book_code.strip().lower()
    if book not in bs:
        return jsonify([])
    return jsonify(sorted(bs[book].keys()))


@app.route("/api/bible/<book_code>/<int:chapter>/verses")
def api_bible_verses_list(book_code: str, chapter: int):
    bs = bible_structure()
    book = book_code.strip().lower()
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
