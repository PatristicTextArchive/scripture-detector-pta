import io
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SD_DB_DIR", tempfile.mkdtemp())

import app as sd  # noqa: E402
import corpora  # noqa: E402
import llm_backends as lb  # noqa: E402
import tei_doc  # noqa: E402

G = corpora.get_corpus("grc")
MT = G.verses["mt_5:9"]
PS = G.verses["ps_62:9"].rstrip(".")
PHIL8 = G.verses["phil_2:8"].rstrip(",·")

TEI = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>Mit Annotation</title></titleStmt>
<publicationStmt><p>x</p></publicationStmt><sourceDesc><p>y</p></sourceDesc></fileDesc></teiHeader>
<text><body><div type="edition" xml:lang="grc" n="urn:cts:pta:pta0001.pta001.pta-grc1"><p>Der Herr sagt:
<quote type="marked">{MT}<ref decls="#biblical" cRef="NA:Mt:5:9"/></quote> Und die Seele ruft: {PS}. Anderswo
<quote>{PHIL8}<ref decls="#biblical" cRef="Peshitta:Phil:2:8"/></quote> Ende.</p></div></body></text></TEI>""".encode()


@pytest.fixture()
def client():
    sd.app.config["TESTING"] = True
    c = sd.app.test_client()
    c.post("/api/settings", json={"api_provider": "gemini", "rag": "on", "rag_add": "on", "rag_strictness": "normal"})
    return c


def import_source(c):
    r = c.post("/api/import/tei", data={"files": (io.BytesIO(TEI), "a.xml")}, content_type="multipart/form-data")
    return r.get_json()["imported"][0]


def annotations(c, sid):
    return c.get(f"/api/sources/{sid}").get_json()["annotations"]


def test_import_reports_and_stores_annotations_from_the_source(client):
    imp = import_source(client)
    assert imp["in_source"] == 2 and imp["annotations"] == 0
    a = {x["quote_text"][:10]: x for x in annotations(client, imp["id"])}
    assert a[MT[:10]]["origin"] == "source" and a[MT[:10]]["refs"] == ["mt_5:9"] and a[MT[:10]]["score"] == 1.0
    assert a[MT[:10]]["marked"] == 1
    assert a[PHIL8[:10]]["origin"] == "source" and a[PHIL8[:10]]["refs"] == ["peshitta:phil:2:8"]    # shown as written


def test_model_run_keeps_source_annotations_and_skips_duplicates(client, monkeypatch):
    sid = import_source(client)["id"]
    answer = [
        {"text": MT, "resolved_references": ["mt_5:9"], "quote_type": "full", "marked": True},       # already in the source
        {"text": PS, "resolved_references": ["ps_62:9"], "quote_type": "full", "marked": False},      # new
    ]
    monkeypatch.setattr(sd, "generate", lambda prompt: lb.Result(json.dumps({"quotes": answer})))
    r = client.post(f"/api/sources/{sid}/process").get_json()
    assert r["count"] == 1 and r["rag"]["in_source"] >= 1
    ann = annotations(client, sid)
    assert sorted(a["origin"] for a in ann) == ["source", "source", "tool"]            # nothing deleted, nothing doubled
    client.post(f"/api/sources/{sid}/process")                                          # running again changes nothing
    assert sorted(a["origin"] for a in annotations(client, sid)) == ["source", "source", "tool"]
    sel = client.post(f"/api/sources/{sid}/process-selection", json={"start": 0, "end": 10_000}).get_json()
    assert sel["status"] == "ok" and len([a for a in annotations(client, sid) if a["origin"] == "source"]) == 2


def test_source_annotations_cannot_be_changed_or_deleted(client):
    sid = import_source(client)["id"]
    qid = annotations(client, sid)[0]["id"]
    r = client.put(f"/api/quotes/{qid}", json={"quote_type": "allusion"})
    assert r.status_code == 409 and "source file" in r.get_json()["error"]
    assert client.delete(f"/api/quotes/{qid}").status_code == 409
    assert len(annotations(client, sid)) == 2
    new = client.post("/api/quotes", json={"source_id": sid, "span_start": 0, "span_end": 5, "quote_text": "Der H",
                                          "quote_type": "allusion", "references": ["mt_5:9"]}).get_json()["id"]
    assert client.delete(f"/api/quotes/{new}").status_code == 200                       # own annotations stay editable


def test_export_leaves_the_source_markup_untouched(client, monkeypatch):
    sid = import_source(client)["id"]
    answer = [{"text": MT, "resolved_references": ["mt_5:9"], "quote_type": "full", "marked": True},
              {"text": PS, "resolved_references": ["ps_62:9"], "quote_type": "full", "marked": False}]
    monkeypatch.setattr(sd, "generate", lambda prompt: lb.Result(json.dumps({"quotes": answer})))
    client.post(f"/api/sources/{sid}/process")
    out = client.get(f"/api/sources/{sid}/export/tei").data.decode()
    root = etree.fromstring(out.encode())
    quotes = list(root.find(tei_doc.T + "text").iter(tei_doc.T + "quote"))
    assert len(quotes) == 3                                                              # 2 existing + 1 new, none doubled
    assert f'<quote type="marked">{MT}<ref decls="#biblical" cRef="NA:Mt:5:9"/></quote>' in out
    assert 'cRef="Peshitta:Phil:2:8"' in out and out.count('cRef="NA:Mt:5:9"') == 1
    assert out.count('cRef="LXX:Ps:62:9"') == 1
    # round trip: the file read again shows the same two source annotations plus the one added by the tool
    again = client.post("/api/import/tei", data={"files": (io.BytesIO(out.encode()), "b.xml")},
                        content_type="multipart/form-data").get_json()["imported"][0]
    assert again["in_source"] == 2 and again["annotations"] == 1


def test_viewer_marks_source_annotations_differently(client):
    sid = import_source(client)["id"]
    page = client.get(f"/viewer/{sid}").data.decode()
    for needle in (".seg-src", ".ann-src", 'id="toggle-src"', "Already in source", "toggleSrc(", "in source"):
        assert needle in page, needle
    info = client.get(f"/api/sources/{sid}").get_json()
    assert [a["origin"] for a in info["annotations"]] == ["source", "source"]
    assert all(s["annotation_ids"] for s in info["segments"] if s["text"].startswith("μακάριοι"))


def test_old_databases_get_the_origin_column(tmp_path):
    import sqlite3
    import database
    con = sqlite3.connect(tmp_path / "old.db")
    con.executescript("""CREATE TABLE sources (id INTEGER PRIMARY KEY, name TEXT, text TEXT);
                         CREATE TABLE quotes (id INTEGER PRIMARY KEY, source_id INTEGER, quote_text TEXT);""")
    database._migrate(con)
    assert "origin" in {r[1] for r in con.execute("PRAGMA table_info(quotes)")}


def test_about_page_mentions_source_annotations(client):
    html = client.get("/about").data.decode()
    assert "in source" in html and "Annotations already in the source file" in html
