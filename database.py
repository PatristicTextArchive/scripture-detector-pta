import os
import sqlite3
import re
import threading
import time
from pathlib import Path
from contextlib import contextmanager

# SD_DB_DIR lets Docker users put the database in a mounted volume.
_db_dir = os.environ.get("SD_DB_DIR")
DB_PATH = (
    Path(_db_dir) / "scripture_detector.db"
    if _db_dir
    else Path(__file__).resolve().parent / "scripture_detector.db"
)

# ── cache-db (per-session in-memory) mode ────────────────────────────────────
# Enabled by SCRIPTURE_DETECTOR_CACHE_DB=1 (set by app.py when --cache-db is
# passed).  Each browser session gets its own isolated SQLite :memory: database.
# Sessions are keyed by a UUID stored in the signed Flask session cookie and
# cleaned up after SESSION_TTL seconds of inactivity.

_CACHE_MODE: bool = os.environ.get("SCRIPTURE_DETECTOR_CACHE_DB", "") in ("1", "true", "yes")
SESSION_TTL: int  = int(os.environ.get("SD_SESSION_TTL", "3600"))  # default 1 hour

# Maps session_id → {conn, lock, last_used}
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()

# Thread-local holds the current session_id (set by app.py before_request hook)
session_local = threading.local()


_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        text TEXT NOT NULL,
        tei_xml BLOB,
        lang TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS quotes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL,
        span_start INTEGER,
        span_end INTEGER,
        quote_text TEXT NOT NULL,
        quote_type TEXT NOT NULL DEFAULT 'allusion'
            CHECK(quote_type IN ('full','partial','paraphrase','allusion')),
        score REAL,
        marked INTEGER,
        origin TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS quote_references (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        quote_id INTEGER NOT NULL,
        reference TEXT NOT NULL,
        book_code TEXT,
        FOREIGN KEY (quote_id) REFERENCES quotes(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first release to existing databases."""
    def cols(table: str) -> set[str]:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if "tei_xml" not in cols("sources"):
        conn.execute("ALTER TABLE sources ADD COLUMN tei_xml BLOB")
    if "lang" not in cols("sources"):
        conn.execute("ALTER TABLE sources ADD COLUMN lang TEXT")
    q = cols("quotes")
    if "score" not in q:
        conn.execute("ALTER TABLE quotes ADD COLUMN score REAL")
    if "marked" not in q:
        conn.execute("ALTER TABLE quotes ADD COLUMN marked INTEGER")
    if "origin" not in q:
        conn.execute("ALTER TABLE quotes ADD COLUMN origin TEXT")


def _new_session_conn() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite connection with the full schema."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA_SQL)
    _migrate(conn)
    conn.commit()
    return conn


def _get_session_entry(session_id: str) -> dict:
    """Return (and lazily create) the session entry for *session_id*."""
    with _sessions_lock:
        # Prune idle sessions
        now = time.monotonic()
        stale = [sid for sid, s in _sessions.items()
                 if now - s["last_used"] > SESSION_TTL]
        for sid in stale:
            try:
                _sessions[sid]["conn"].close()
            except Exception:
                pass
            del _sessions[sid]

        if session_id not in _sessions:
            _sessions[session_id] = {
                "conn":      _new_session_conn(),
                "lock":      threading.Lock(),
                "last_used": now,
            }
        else:
            _sessions[session_id]["last_used"] = now

        return _sessions[session_id]


@contextmanager
def get_db():
    if _CACHE_MODE:
        sid   = getattr(session_local, "session_id", None) or "anonymous"
        entry = _get_session_entry(sid)
        with entry["lock"]:
            conn = entry["conn"]
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    else:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db():
    if _CACHE_MODE:
        # In cache mode each session's schema is initialised when its
        # connection is first created (_new_session_conn).  Nothing to do here.
        return
    with get_db() as conn:
        conn.executescript(_SCHEMA_SQL)
        _migrate(conn)


def extract_book_code(reference: str) -> str:
    match = re.match(r"^([a-z0-9]+)_", reference.strip().lower())
    return match.group(1) if match else ""


# ── Sources ──────────────────────────────────────────────────────────────────

def create_source(name: str, text: str, tei_xml: bytes | None = None, lang: str | None = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO sources (name, text, tei_xml, lang) VALUES (?, ?, ?, ?)",
            (name, text, tei_xml, lang),
        )
        return cur.lastrowid


def get_source(source_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        return dict(row) if row else None


def get_all_sources() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT s.id, s.name, s.created_at, s.updated_at,
                      LENGTH(s.text) as text_length,
                      (s.tei_xml IS NOT NULL) as is_tei,
                      s.lang as lang,
                      COUNT(q.id) as quote_count
               FROM sources s
               LEFT JOIN quotes q ON q.source_id = s.id
               GROUP BY s.id
               ORDER BY s.created_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def delete_source(source_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))


# ── Quotes ───────────────────────────────────────────────────────────────────

def add_quote(
    source_id: int,
    span_start: int | None,
    span_end: int | None,
    quote_text: str,
    quote_type: str,
    references: list[str],
    score: float | None = None,
    marked: bool | None = None,
    origin: str | None = None,
) -> int:
    """*origin* "source": the passage was already annotated in the source file (read-only)."""
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO quotes (source_id, span_start, span_end, quote_text,
                                   quote_type, score, marked, origin)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_id, span_start, span_end, quote_text, quote_type, score,
             None if marked is None else int(bool(marked)), origin),
        )
        quote_id = cur.lastrowid
        for ref in references:
            ref_clean = ref.strip().lower()
            book = extract_book_code(ref_clean)
            conn.execute(
                "INSERT INTO quote_references (quote_id, reference, book_code) VALUES (?, ?, ?)",
                (quote_id, ref_clean, book),
            )
        return quote_id


def update_quote(
    quote_id: int,
    quote_text: str = None,
    quote_type: str = None,
    span_start: int = None,
    span_end: int = None,
    references: list[str] = None,
    score: float = None,
    marked: bool = None,
):
    with get_db() as conn:
        if score is not None:
            conn.execute("UPDATE quotes SET score=? WHERE id=?", (score, quote_id))
        if marked is not None:
            conn.execute("UPDATE quotes SET marked=? WHERE id=?", (int(bool(marked)), quote_id))
        if quote_text is not None:
            conn.execute(
                "UPDATE quotes SET quote_text=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (quote_text, quote_id),
            )
        if quote_type is not None:
            conn.execute(
                "UPDATE quotes SET quote_type=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (quote_type, quote_id),
            )
        if span_start is not None:
            conn.execute(
                "UPDATE quotes SET span_start=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (span_start, quote_id),
            )
        if span_end is not None:
            conn.execute(
                "UPDATE quotes SET span_end=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (span_end, quote_id),
            )
        if references is not None:
            conn.execute("DELETE FROM quote_references WHERE quote_id=?", (quote_id,))
            for ref in references:
                ref_clean = ref.strip().lower()
                book = extract_book_code(ref_clean)
                conn.execute(
                    "INSERT INTO quote_references (quote_id, reference, book_code) VALUES (?, ?, ?)",
                    (quote_id, ref_clean, book),
                )


def delete_quote(quote_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM quotes WHERE id = ?", (quote_id,))


def get_quotes_for_source(source_id: int) -> list[dict]:
    with get_db() as conn:
        quotes = conn.execute(
            "SELECT * FROM quotes WHERE source_id = ? ORDER BY span_start",
            (source_id,),
        ).fetchall()
        result = []
        for q in quotes:
            qd = dict(q)
            refs = conn.execute(
                "SELECT reference, book_code FROM quote_references WHERE quote_id = ?",
                (q["id"],),
            ).fetchall()
            qd["references"] = [dict(r) for r in refs]
            result.append(qd)
        return result


def delete_quotes_for_source(source_id: int):
    with get_db() as conn:
        # annotations that come from the source file are kept: they are part of the file
        conn.execute("DELETE FROM quotes WHERE source_id = ? AND COALESCE(origin, '') != 'source'", (source_id,))


def delete_quotes_in_range(source_id: int, start: int, end: int):
    with get_db() as conn:
        conn.execute(
            """DELETE FROM quotes WHERE source_id = ?
               AND span_start IS NOT NULL AND span_end IS NOT NULL
               AND COALESCE(origin, '') != 'source'
               AND NOT (span_end <= ? OR span_start >= ?)""",
            (source_id, start, end),
        )


# ── Settings ─────────────────────────────────────────────────────────────────

def get_setting(key: str, default: str = None) -> str | None:
    with get_db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def get_all_settings() -> dict:
    with get_db() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}


# ── Analytics ────────────────────────────────────────────────────────────────

def get_book_distribution(source_id: int = None) -> list[dict]:
    with get_db() as conn:
        if source_id:
            rows = conn.execute(
                """SELECT qr.book_code, COUNT(*) as count
                   FROM quote_references qr
                   JOIN quotes q ON qr.quote_id = q.id
                   WHERE q.source_id = ? AND qr.book_code != ''
                   GROUP BY qr.book_code ORDER BY count DESC""",
                (source_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT qr.book_code, COUNT(*) as count
                   FROM quote_references qr
                   JOIN quotes q ON qr.quote_id = q.id
                   WHERE qr.book_code != ''
                   GROUP BY qr.book_code ORDER BY count DESC"""
            ).fetchall()
        return [dict(r) for r in rows]


def get_quote_type_distribution(source_id: int = None) -> dict:
    with get_db() as conn:
        if source_id:
            rows = conn.execute(
                "SELECT quote_type, COUNT(*) as count FROM quotes WHERE source_id=? GROUP BY quote_type",
                (source_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT quote_type, COUNT(*) as count FROM quotes GROUP BY quote_type"
            ).fetchall()
        return {r["quote_type"]: r["count"] for r in rows}


def search_sources(filters: list[dict], logic: str = "AND") -> dict:
    """
    Search sources by text content and/or scripture references.

    filters: list of dicts, each with:
      - type: 'text' | 'book' | 'chapter' | 'verse'
      - value: the query string (already lowercased by caller)
    logic: 'AND' (source must match all filters) |
           'OR'  (source must match any filter)

    Returns {'total': int, 'results': [enriched source dicts]}
    """
    valid = [f for f in filters if f.get("value", "").strip()]
    if not valid:
        return {"total": 0, "results": []}

    with get_db() as conn:
        # --- per-filter: map source_id → list of evidence dicts ---------------
        filter_evidence: list[dict[int, list[dict]]] = []

        for f in valid:
            ftype = f["type"]
            fval  = f["value"].strip().lower()
            ev: dict[int, list[dict]] = {}

            if ftype == "text":
                rows = conn.execute(
                    "SELECT id, name, text FROM sources "
                    "WHERE lower(text) LIKE ? OR lower(name) LIKE ?",
                    [f"%{fval}%", f"%{fval}%"],
                ).fetchall()
                for r in rows:
                    full = r["text"]
                    idx  = full.lower().find(fval)
                    if idx >= 0:
                        s0 = max(0, idx - 100)
                        s1 = min(len(full), idx + len(fval) + 100)
                        pre  = ("…" if s0 > 0 else "")
                        post = ("…" if s1 < len(full) else "")
                        snippet = pre + full[s0:s1] + post
                    else:
                        # Query matched the source name, not the text body
                        snippet = r["name"]
                    ev[r["id"]] = ev.get(r["id"], [])
                    ev[r["id"]].append({
                        "kind": "text", "snippet": snippet,
                        "query": f["value"], "offset": idx,
                    })

            elif ftype == "book":
                rows = conn.execute(
                    """SELECT DISTINCT q.source_id,
                              qr.reference, q.quote_text, q.quote_type
                       FROM quotes q
                       JOIN quote_references qr ON qr.quote_id = q.id
                       WHERE qr.book_code = ?""",
                    [fval],
                ).fetchall()
                for r in rows:
                    sid = r["source_id"]
                    ev.setdefault(sid, [])
                    ev[sid].append({
                        "kind": "ref", "reference": r["reference"],
                        "quote_text": r["quote_text"][:120],
                        "quote_type": r["quote_type"],
                    })

            elif ftype == "chapter":
                # fval is e.g. "gen_1"
                rows = conn.execute(
                    """SELECT DISTINCT q.source_id,
                              qr.reference, q.quote_text, q.quote_type
                       FROM quotes q
                       JOIN quote_references qr ON qr.quote_id = q.id
                       WHERE qr.reference LIKE ?""",
                    [f"{fval}:%"],
                ).fetchall()
                for r in rows:
                    sid = r["source_id"]
                    ev.setdefault(sid, [])
                    ev[sid].append({
                        "kind": "ref", "reference": r["reference"],
                        "quote_text": r["quote_text"][:120],
                        "quote_type": r["quote_type"],
                    })

            elif ftype == "verse":
                # fval is e.g. "gen_1:1"
                rows = conn.execute(
                    """SELECT DISTINCT q.source_id,
                              qr.reference, q.quote_text, q.quote_type
                       FROM quotes q
                       JOIN quote_references qr ON qr.quote_id = q.id
                       WHERE qr.reference = ?""",
                    [fval],
                ).fetchall()
                for r in rows:
                    sid = r["source_id"]
                    ev.setdefault(sid, [])
                    ev[sid].append({
                        "kind": "ref", "reference": r["reference"],
                        "quote_text": r["quote_text"][:120],
                        "quote_type": r["quote_type"],
                    })

            filter_evidence.append(ev)

        # --- combine filter result sets ---------------------------------------
        id_sets = [set(ev.keys()) for ev in filter_evidence]
        if logic == "AND":
            matching: set[int] = id_sets[0]
            for s in id_sets[1:]:
                matching &= s
        else:
            matching = set()
            for s in id_sets:
                matching |= s

        if not matching:
            return {"total": 0, "results": []}

        # --- fetch source rows with aggregate stats --------------------------
        placeholders = ",".join("?" * len(matching))
        src_rows = conn.execute(
            f"""SELECT s.id, s.name, s.lang, s.created_at,
                       COUNT(q.id) as quote_count
                FROM sources s
                LEFT JOIN quotes q ON q.source_id = s.id
                WHERE s.id IN ({placeholders})
                GROUP BY s.id ORDER BY s.created_at DESC""",
            list(matching),
        ).fetchall()

        results = []
        for src in src_rows:
            sd = dict(src)

            # type distribution
            td_rows = conn.execute(
                "SELECT quote_type, COUNT(*) as c FROM quotes "
                "WHERE source_id=? GROUP BY quote_type",
                [sd["id"]],
            ).fetchall()
            sd["type_distribution"] = {r["quote_type"]: r["c"] for r in td_rows}

            # book distribution (top 5)
            bd_rows = conn.execute(
                """SELECT qr.book_code, COUNT(*) as count
                   FROM quote_references qr
                   JOIN quotes q ON qr.quote_id = q.id
                   WHERE q.source_id = ? AND qr.book_code != ''
                   GROUP BY qr.book_code ORDER BY count DESC LIMIT 5""",
                [sd["id"]],
            ).fetchall()
            sd["book_distribution"] = [dict(r) for r in bd_rows]

            # aggregate evidence from all matching filters, deduplicate refs
            all_ev: list[dict] = []
            seen_refs: set[str] = set()
            for ev in filter_evidence:
                if sd["id"] in ev:
                    for item in ev[sd["id"]]:
                        if item["kind"] == "ref":
                            key = item["reference"]
                            if key in seen_refs:
                                continue
                            seen_refs.add(key)
                        all_ev.append(item)
            sd["match_evidence"] = all_ev[:10]

            results.append(sd)

        return {"total": len(results), "results": results}


def get_dashboard_data() -> dict:
    with get_db() as conn:
        source_count = conn.execute("SELECT COUNT(*) as c FROM sources").fetchone()["c"]
        quote_count = conn.execute("SELECT COUNT(*) as c FROM quotes").fetchone()["c"]
        ref_count = conn.execute("SELECT COUNT(*) as c FROM quote_references").fetchone()["c"]

        sources = conn.execute(
            """SELECT s.id, s.name, s.lang, s.created_at,
                      COUNT(q.id) as quote_count
               FROM sources s
               LEFT JOIN quotes q ON q.source_id = s.id
               GROUP BY s.id ORDER BY s.created_at DESC"""
        ).fetchall()

        return {
            "source_count": source_count,
            "quote_count": quote_count,
            "reference_count": ref_count,
            "sources": [dict(s) for s in sources],
        }


def get_quote_source_id(quote_id: int) -> int | None:
    with get_db() as conn:
        row = conn.execute("SELECT source_id FROM quotes WHERE id = ?", (quote_id,)).fetchone()
        return row["source_id"] if row else None


def get_quote_origin(quote_id: int) -> str | None:
    with get_db() as conn:
        row = conn.execute("SELECT origin FROM quotes WHERE id = ?", (quote_id,)).fetchone()
        return row["origin"] if row else None
