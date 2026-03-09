import sqlite3
import re
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent / "scripture_detector.db"


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                text TEXT NOT NULL,
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
        """)


def extract_book_code(reference: str) -> str:
    match = re.match(r"^([a-z0-9]+)_", reference.strip().lower())
    return match.group(1) if match else ""


# ── Sources ──────────────────────────────────────────────────────────────────

def create_source(name: str, text: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO sources (name, text) VALUES (?, ?)", (name, text)
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
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO quotes (source_id, span_start, span_end, quote_text, quote_type)
               VALUES (?, ?, ?, ?, ?)""",
            (source_id, span_start, span_end, quote_text, quote_type),
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
):
    with get_db() as conn:
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
        conn.execute("DELETE FROM quotes WHERE source_id = ?", (source_id,))


def delete_quotes_in_range(source_id: int, start: int, end: int):
    with get_db() as conn:
        conn.execute(
            """DELETE FROM quotes WHERE source_id = ?
               AND span_start IS NOT NULL AND span_end IS NOT NULL
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


def get_dashboard_data() -> dict:
    with get_db() as conn:
        source_count = conn.execute("SELECT COUNT(*) as c FROM sources").fetchone()["c"]
        quote_count = conn.execute("SELECT COUNT(*) as c FROM quotes").fetchone()["c"]
        ref_count = conn.execute("SELECT COUNT(*) as c FROM quote_references").fetchone()["c"]

        sources = conn.execute(
            """SELECT s.id, s.name, s.created_at,
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
