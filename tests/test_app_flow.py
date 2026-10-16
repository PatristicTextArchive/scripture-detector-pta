import io
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["SD_DB_DIR"] = tempfile.mkdtemp()

import app as sd  # noqa: E402
import corpora  # noqa: E402
import tei_doc  # noqa: E402
from lxml import etree  # noqa: E402

MT_5_9 = "μακάριοι οἱ εἰρηνοποιοί, ὅτι αὐτοὶ υἱοὶ θεοῦ κληθήσονται"
PS_62_9 = "ἐκολλήθη ἡ ψυχή μου ὀπίσω σου"

TEI = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>
<titleStmt><title>Homilie</title><author>Pseudo-Test</author></titleStmt><publicationStmt><p>x</p></publicationStmt>
<sourceDesc><p>y</p></sourceDesc></fileDesc></teiHeader>
<text><body><div type="edition" xml:lang="grc" n="urn:cts:pta:pta0001.pta001.pta-grc1">
<div type="textpart" subtype="section" n="1"><p>Der Herr sagt: Μακάριοι οἱ εἰρη<lb break="no"/>νοποιοί, ὅτι αὐτοὶ υἱοὶ
θεοῦ κληθήσονται. Und die Seele ruft: ἐκολλήθη ἡ ψυχή μου ὀπίσω σου.</p></div>
</div></body></text></TEI>""".encode("utf-8")


@pytest.fixture()
def client(monkeypatch):
    def fake_gemini(text, lang="grc", hints=None):
        sd.LAST_CALL = {"lang": lang, "hints": hints, "text": text}
        # model output as it would arrive: normalised, partly wrong references
        return [
            {"text": "μακαριοι οι ειρηνοποιοι οτι αυτοι υιοι θεου κληθησονται",
             "resolved_references": ["mt_5:9", "nonexistent_1:1", "mt_5:999"],
             "quote_type": "full", "marked": True},
            {"text": PS_62_9, "resolved_references": ["ps_62:9"], "quote_type": "partial", "marked": False},
            {"text": "nur erfunden", "resolved_references": ["mt_999:1"], "quote_type": "allusion"},
        ]
    monkeypatch.setattr(sd, "extract_quotes", fake_gemini)
    sd.app.config["TESTING"] = True
    return sd.app.test_client()


def test_bible_data_loaded():
    g, l = corpora.get_corpus("grc"), corpora.get_corpus("lat")
    assert len(g.verses) > 36000 and len(l.verses) > 30000
    assert g.verses["mt_5:9"].startswith("μακάριοι")
    assert l.verses["gen_1:1"].startswith("In principio")            # zero-padded ids are normalised
    assert g.mapping["ps"]["pta_abbr"] == "Ps" and l.mapping["gen"]["pta_corpus"] == "Vg"
    assert ("na", "mt") in g.abbr_map() and ("vg", "gn") in l.abbr_map()


def test_prompt_is_greek_specific():
    p = sd.build_prompt("κείμενον", corpora.get_corpus("grc"), ["ὁ λόγος"])
    assert "Septuagint" in p and "3esr" in p and "2esd" not in p
    assert "γέγραπται, φησί" in p and "διαμαρτύρεται" in p and "already marks" in p and "1. ὁ λόγος" in p
    assert "Κατὰ Ματθαῖον -> mt" in p and "Psalms (LXX numbering, incl. Ps 151) / Ψαλμοί -> ps" in p


def test_tei_import_detect_export(client):
    r = client.post("/api/import/tei", data={"files": (io.BytesIO(TEI), "hom.xml")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    sid = r.get_json()["imported"][0]["id"]
    assert "Pseudo-Test: Homilie" == r.get_json()["imported"][0]["name"]

    r = client.post(f"/api/sources/{sid}/process")
    assert r.get_json()["count"] == 2                       # invented reference dropped, bad refs filtered

    info = client.get(f"/api/sources/{sid}").get_json()
    assert info["source"]["is_tei"]
    by_ref = {a["refs"][0]: a for a in info["annotations"]}
    assert set(by_ref) == {"mt_5:9", "ps_62:9"}
    assert by_ref["mt_5:9"]["score"] == 1.0 and by_ref["mt_5:9"]["marked"] == 1
    text = info["segments"]
    full = "".join(s["text"] for s in text)
    a = by_ref["mt_5:9"]
    assert full[a["span_start"]:a["span_end"]].startswith("Μακάριοι οἱ εἰρηνοποιοί")   # lb break=no joined

    r = client.get(f"/api/sources/{sid}/export/tei")
    assert r.status_code == 200
    root = etree.fromstring(r.data)
    quotes = list(root.iter(tei_doc.T + "quote"))
    crefs = sorted(x.get("cRef") for q in quotes for x in q.iter(tei_doc.T + "ref"))
    assert crefs == ["LXX:Ps:62:9", "NA:Mt:5:9"]
    assert {q.get("type") for q in quotes} == {"marked", "unmarked"}
    # <lb break="no"/> of the original is still there, inside the quotation
    assert root.find(f".//{tei_doc.T}quote/{tei_doc.T}lb") is not None

    # export -> import round trip restores the annotations
    r2 = client.post("/api/import/tei", data={"files": (io.BytesIO(r.data), "again.xml")}, content_type="multipart/form-data")
    assert r2.get_json()["imported"][0]["annotations"] == 2


def test_pasted_text_exports_pta_skeleton(client):
    text = "Ὁ κύριος λέγει·\nἐκολλήθη ἡ ψυχή μου ὀπίσω σου. Ende"
    sid = client.post("/api/sources", json={"name": "Paste", "text": text}).get_json()["id"]
    assert client.post(f"/api/sources/{sid}/process").get_json()["count"] == 2
    r = client.get(f"/api/sources/{sid}/export/tei")
    s = r.data.decode()
    assert 'type="edition"' in s and 'xml:lang="grc"' in s and 'cRef="LXX:Ps:62:9"' in s


def test_zip_export_and_reimport(client):
    r = client.get("/api/export/zip")
    assert r.status_code == 200 and r.data[:2] == b"PK"
    r2 = client.post("/api/import/zip", data={"file": (io.BytesIO(r.data), "x.zip")}, content_type="multipart/form-data")
    assert r2.get_json()["status"] == "ok" and not r2.get_json()["errors"]


def test_pages_and_bible_endpoints_render(client):
    sid = client.post("/api/sources", json={"name": "P", "text": "Ἐν ἀρχῇ"}).get_json()["id"]
    for url in ("/", "/dashboard", f"/viewer/{sid}", "/settings", "/about"):
        assert client.get(url).status_code == 200, url
    assert "24a" in client.get("/api/bible/prov/chapters").get_json()
    assert client.get("/api/bible/mt/5/verses").get_json()[8]["verse"] == "9"
    assert client.get("/api/bible/sir/p/verses").get_json()
    assert corpora.get_corpus("grc").resolve_ref("prov_24:23") == ["prov_24a:23"]


def test_existing_database_is_migrated(tmp_path, monkeypatch):
    import sqlite3, database
    con = sqlite3.connect(tmp_path / "old.db")
    con.executescript("""CREATE TABLE sources (id INTEGER PRIMARY KEY, name TEXT, text TEXT);
                         CREATE TABLE quotes (id INTEGER PRIMARY KEY, source_id INTEGER, quote_text TEXT);""")
    database._migrate(con)
    assert {"tei_xml"} <= {r[1] for r in con.execute("PRAGMA table_info(sources)")}
    assert {"score", "marked"} <= {r[1] for r in con.execute("PRAGMA table_info(quotes)")}


# ── existing <quote> elements ────────────────────────────────────────────────

TEI_QUOTE = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>Mit Quote</title></titleStmt>
<publicationStmt><p>x</p></publicationStmt><sourceDesc><p>y</p></sourceDesc></fileDesc></teiHeader>
<text><body><div type="edition" xml:lang="grc" n="urn:cts:pta:pta0001.pta001.pta-grc1"><p>Der Apostel φησίν: <quote>{MT_5_9}</quote>.
Ein anderer Autor sagt <quote>τοῦτο οὐκ ἔστι βιβλικόν ἀλλὰ ἀνθρώπινον</quote> und so weiter.</p></div></body></text></TEI>""".encode("utf-8")


def test_existing_quotes_are_hints_and_get_only_refs(client, monkeypatch):
    seen = {}

    def fake(text, lang="grc", hints=None):
        seen.update(lang=lang, hints=hints)
        return [{"text": MT_5_9, "resolved_references": ["mt_5:9"], "quote_type": "full", "marked": True}]

    monkeypatch.setattr(sd, "extract_quotes", fake)
    sid = client.post("/api/import/tei", data={"files": (io.BytesIO(TEI_QUOTE), "q.xml")},
                      content_type="multipart/form-data").get_json()["imported"][0]["id"]
    assert client.post(f"/api/sources/{sid}/process").get_json()["count"] == 1
    assert seen["lang"] == "grc" and seen["hints"] == [MT_5_9, "τοῦτο οὐκ ἔστι βιβλικόν ἀλλὰ ἀνθρώπινον"]

    out = client.get(f"/api/sources/{sid}/export/tei").data.decode()
    assert out.count("<quote") == 2 and "<seg" not in out.split("<text>")[1]       # no further quote element
    assert f'<quote type="marked">{MT_5_9}<ref decls="#biblical" cRef="NA:Mt:5:9" xml:id="sd-r1"/></quote>' in out


# ── Latin ────────────────────────────────────────────────────────────────────

LAT = "In principio creavit Deus caelum et terram"
TEI_LAT = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>Hexaemeron</title></titleStmt>
<publicationStmt><p>x</p></publicationStmt><sourceDesc><p>y</p></sourceDesc></fileDesc></teiHeader>
<text><body><div type="edition" xml:lang="lat" n="urn:cts:pta:pta0002.pta001.pta-lat1"><p>Scriptum est: {LAT}. Amen.</p></div></body></text></TEI>""".encode("utf-8")


def test_latin_tei_uses_vulgate_and_vg_prefix(client, monkeypatch):
    seen = {}

    def fake(text, lang="grc", hints=None):
        seen["lang"] = lang
        seen["prompt"] = sd.build_prompt(text, corpora.get_corpus(lang), hints)
        return [{"text": LAT, "resolved_references": ["gen_1:1", "mt_5:9", "gen_1:999"],
                 "quote_type": "full", "marked": True}]

    monkeypatch.setattr(sd, "extract_quotes", fake)
    r = client.post("/api/import/tei", data={"files": (io.BytesIO(TEI_LAT), "lat.xml")},
                    content_type="multipart/form-data").get_json()
    assert r["imported"][0]["lang"] == "lat"
    sid = r["imported"][0]["id"]
    assert client.post(f"/api/sources/{sid}/process").get_json()["count"] == 1
    assert seen["lang"] == "lat" and "Vulgate" in seen["prompt"] and "scriptum est" in seen["prompt"]
    assert "Valid book codes" in seen["prompt"] and " gen," in seen["prompt"] and " mt," not in seen["prompt"]

    info = client.get(f"/api/sources/{sid}").get_json()
    assert info["source"]["lang"] == "lat"
    assert [a["refs"] for a in info["annotations"]] == [["gen_1:1"]]        # Greek code / bad verse dropped
    assert info["annotations"][0]["score"] == 1.0

    out = client.get(f"/api/sources/{sid}/export/tei").data.decode()
    assert 'cRef="Vg:Gn:1:1"' in out and "LXX" not in out.split("<text>")[1]

    again = client.post("/api/import/tei", data={"files": (io.BytesIO(out.encode()), "again.xml")},
                        content_type="multipart/form-data").get_json()
    assert again["imported"][0]["annotations"] == 1 and again["imported"][0]["lang"] == "lat"


def test_unsupported_language_is_rejected_and_paste_is_detected(client):
    syr = TEI_LAT.replace(b'xml:lang="lat"', b'xml:lang="cop"')
    r = client.post("/api/import/tei", data={"files": (io.BytesIO(syr), "syr.xml")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "Unsupported language 'cop'" in r.get_json()["errors"][0]["error"]
    assert client.post("/api/sources", json={"name": "L", "text": LAT}).get_json()["lang"] == "lat"
    assert client.post("/api/sources", json={"name": "G", "text": MT_5_9}).get_json()["lang"] == "grc"
    lat_books = client.get("/api/bible/books?lang=lat").get_json()
    assert any(b["code"] == "matt" for b in lat_books) and not any(b["code"] == "mt" for b in lat_books)
    assert client.get("/api/bible/gen/1/verses?lang=lat").get_json()[0]["verse"] == "1"
    assert client.get("/api/bible/gen/1/verses").get_json() == []            # default corpus is Greek


def test_latin_paste_exports_skeleton_with_lat(client, monkeypatch):
    def fake(text, lang="grc", hints=None):
        return [{"text": LAT, "resolved_references": ["gen_1:1"], "quote_type": "full"}]
    monkeypatch.setattr(sd, "extract_quotes", fake)
    sid = client.post("/api/sources", json={"name": "P", "text": f"Dicit: {LAT}. Amen."}).get_json()["id"]
    client.post(f"/api/sources/{sid}/process")
    s = client.get(f"/api/sources/{sid}/export/tei").data.decode()
    assert 'xml:lang="lat"' in s and 'cRef="Vg:Gn:1:1"' in s


def test_tidy_warnings_merges_duplicates_and_caps():
    w = ["a", "b", "a", "a"] + [f"x{i}" for i in range(5)]
    out = sd.tidy_warnings(w, limit=4)
    assert out[:2] == ["a (×3)", "b"] and out[-1] == "… and 3 more"


def test_export_json_returns_all_warnings_and_xml(client, monkeypatch):
    # a transcription (-Ms…) triggers a warning; many skipped duplicates must be merged, not truncated
    tei = TEI.replace(b"pta0001.pta001.pta-grc1", b"pta0013.pta003.pta-MsPs")
    sid = client.post("/api/import/tei", data={"files": (io.BytesIO(tei), "ms.xml")},
                      content_type="multipart/form-data").get_json()["imported"][0]["id"]
    monkeypatch.setattr(sd, "extract_quotes", lambda t, lang="grc", hints=None: [
        {"text": PS_62_9, "resolved_references": ["ps_62:9"], "quote_type": "full"}])
    client.post(f"/api/sources/{sid}/process")
    r = client.get(f"/api/sources/{sid}/export/tei?json=1")
    d = r.get_json()
    assert d["filename"].endswith(".xml") and d["xml"].startswith("<?xml") and "cRef=" in d["xml"]
    assert len(d["warnings"]) == 1 and "transcription" in d["warnings"][0]
    plain = client.get(f"/api/sources/{sid}/export/tei")
    assert plain.headers["X-SD-Warning-Count"] == "1" and plain.data.startswith(b"<?xml")


def test_notification_script_is_served_and_used(client):
    assert b"showNotice" in client.get("/static/ui.js").data
    sid = client.post("/api/sources", json={"name": "U", "text": "Ἐν ἀρχῇ"}).get_json()["id"]
    for url in ("/", f"/viewer/{sid}", "/settings"):
        assert b"/static/ui.js" in client.get(url).data, url


def test_about_page_links_to_pta_and_original(client):
    html = client.get("/about").data.decode()
    for url in ("https://pta.bbaw.de", "https://github.com/PatristicTextArchive/pta_data",
                "https://pta.bbaw.de/en/project/encoding-guidelines", "https://github.com/yale-ch/scripture-detector"):
        assert f'href="{url}' in html, url
    assert "Greek" in html and "Vg:Gn:1:1" in html and "CC BY-SA 4.0" in html


# ── rate limits ──────────────────────────────────────────────────────────────

def test_interrupted_run_resumes_without_repeating_requests(client, monkeypatch):
    import gemini_limits
    monkeypatch.setattr(sd, "CHUNK_CHARS", 100)
    text = "\n".join(f"Satz {i} " * 6 + PS_62_9 for i in range(3))
    sid = client.post("/api/sources", json={"name": "Lang", "text": text}).get_json()["id"]
    calls, fail = [], {"on": True}

    def fake(chunk, lang="grc", hints=None):
        calls.append(chunk)
        if len(calls) == 2 and fail["on"]:
            raise gemini_limits.RateLimited("429", retry_after=30, attempts=6, rpm=5)
        return [{"text": PS_62_9, "resolved_references": ["ps_62:9"], "quote_type": "full"}]

    monkeypatch.setattr(sd, "extract_quotes", fake)
    r = client.post(f"/api/sources/{sid}/process")
    assert r.status_code == 429
    body = r.get_json()
    assert body["retryable"] is True and "1 of 3 parts" in body["error"] and "Max requests per minute" in body["error"]
    assert len(calls) == 2
    assert client.get(f"/api/sources/{sid}").get_json()["annotations"] == []     # nothing half-written

    fail["on"] = False
    r = client.post(f"/api/sources/{sid}/process")
    assert r.status_code == 200 and r.get_json()["count"] == 3
    assert len(calls) == 2 + 2                      # part 1 was not asked again
    client.post(f"/api/sources/{sid}/process")      # a deliberate re-run asks the model again
    assert len(calls) == 2 + 2 + 3


def test_daily_quota_is_reported_as_not_retryable(client, monkeypatch):
    import gemini_limits
    sid = client.post("/api/sources", json={"name": "Q", "text": PS_62_9}).get_json()["id"]

    def fake(chunk, lang="grc", hints=None):
        raise gemini_limits.QuotaExhausted("daily")

    monkeypatch.setattr(sd, "extract_quotes", fake)
    r = client.post(f"/api/sources/{sid}/process")
    assert r.status_code == 429 and r.get_json()["retryable"] is False and "daily" in r.get_json()["error"]


def test_max_rpm_setting(client):
    assert sd.get_max_rpm() == 5
    client.post("/api/settings", json={"max_rpm": "12"})
    assert sd.get_max_rpm() == 12
    client.post("/api/settings", json={"max_rpm": "0"})
    assert sd.get_max_rpm() == 0
    client.post("/api/settings", json={"max_rpm": "abc"})
    assert sd.get_max_rpm() == 5
    assert b'id="max-rpm"' in client.get("/settings").data


def test_about_and_settings_mention_local_models(client):
    about = client.get("/about").data.decode()
    assert "llama-cpp-python" in about and "Ollama" in about
    st = client.get("/settings").data.decode()
    for needle in ('id="rc-openai"', 'id="rc-llamacpp"', 'id="gguf-path"', 'id="chunk-chars"', "testBackend()"):
        assert needle in st, needle
