import io
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SD_DB_DIR", tempfile.mkdtemp())

import app as sd  # noqa: E402
import corpora  # noqa: E402
import llm_backends as lb  # noqa: E402
import retrieval  # noqa: E402
import tei_doc  # noqa: E402

S = corpora.get_corpus("syc")
MT = S.verses["mt_5:9"]                                   # New Testament: vocalised
PS = S.verses["ps_23:1"]                                  # Old Testament: consonantal


def bare(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


# ── corpus and language ──────────────────────────────────────────────────────

def test_language_codes_and_script_detection():
    for code in ("syc", "syr", "SYC", "syc-Syrc"):
        assert corpora.normalize_lang(code) == "syc"
    assert corpora.guess_lang(MT) == "syc" and corpora.guess_lang("In principio creavit") == "lat"
    assert corpora.guess_lang("ἐν ἀρχῇ ἐποίησεν ὁ θεός") == "grc"


def test_peshitta_corpus_is_loaded():
    assert len(S.verses) > 38000
    assert S.verses["gn_1:1"].startswith("ܒܪܫܝܬ ܒܪܐ")
    m = S.mapping
    assert m["gn"]["pta_corpus"] == "Peshitta" and m["gn"]["pta_abbr"] == "Gn" and m["mt"]["pta_abbr"] == "Mt"
    for code in ("neh", "prman", "4esr", "2bar", "3esr"):
        assert code in m, code
    assert ("pes", "gn") in S.abbr_map()
    assert S.resolve_ref("2bar_78:1") and S.resolve_ref("neh_13:1")          # chapters 78-86 of 2 Baruch are there
    assert "\\n" not in "".join(list(S.verses.values())[:5000])                # literal \n cleaned from the NT
    assert "Peshitta" in S.meta["corpus"] and "ܟܬܝܒ" in S.meta["intro_formulas"]


# ── retrieval ────────────────────────────────────────────────────────────────

def test_stems_ignore_vowel_points_and_attached_particles():
    st = lambda w: retrieval.tokenize(w, "syc")[0].stem
    assert st("ܡܠܟܐ") == st("ܘܡܠܟܐ") == st("ܕܡܠܟܐ") == st("ܘܕܡܠܟܐ")            # w-, d-, w+d- stripped
    assert st(MT.split()[0]) == st(bare(MT.split()[0]))                          # vocalised == consonantal
    assert st("ܒܪܐ") == "ܒܪܐ"                                                     # short words keep their letters


def test_vocalised_new_testament_is_found_in_unvocalised_text_with_changed_particles():
    words = bare(MT).split()
    words[2] = "ܘ" + words[2]                                  # an extra w-
    text = "ܐܡܪ ܡܪܢ ܒܡܬܠܐ ܘܡܠܦ ܠܬܠܡܝܕܘܗܝ ܘܐܡܪ ܠܗܘܢ: " + " ".join(words) + " ܘܗܟܢܐ ܕܝܢ ܐܡܪ ܡܪܢ"
    cands = retrieval.candidates(text, S, top=10)
    assert "mt_5:9" in [c.key for c in cands[:3]]
    (m,) = [x for x in retrieval.lexical_matches(text, S) if "mt_5:9" in x.refs] or [None]
    assert m is not None and m.words >= 6
    assert bare(MT).split()[0] in text[m.start:m.end]


def test_common_syriac_phrases_are_not_reported():
    assert retrieval.lexical_matches("ܘܗܫܐ ܘܠܥܠܡ ܥܠܡܝܢ ܐܡܝܢ", S, min_words=6) == []        # a doxology, too short anyway
    assert retrieval.lexical_matches("ܗܢܐ ܐܡܪ ܠܗ ܘܗܘ ܐܡܪ", S) == []


# ── whole app ────────────────────────────────────────────────────────────────

TEI = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>Syrische Homilie</title></titleStmt>
<publicationStmt><p>x</p></publicationStmt><sourceDesc><p>y</p></sourceDesc></fileDesc></teiHeader>
<text><body><div type="edition" xml:lang="syc" n="urn:cts:pta:pta0200.pta001.pta-syc1"><p>ܐܡܪ ܡܪܢ ܕܝܢ: {bare(MT)} ܘܗܟܢܐ ܡܠܦ.
ܘܐܡܪ ܕܘܝܕ: <quote type="marked">{bare(PS)}<ref decls="#biblical" cRef="Peshitta:Ps:23:1"/></quote></p></div></body></text></TEI>""".encode()


@pytest.fixture()
def client():
    sd.app.config["TESTING"] = True
    c = sd.app.test_client()
    c.post("/api/settings", json={"api_provider": "gemini", "rag": "on", "rag_add": "on", "rag_strictness": "normal"})
    return c


def import_syc(c):
    return c.post("/api/import/tei", data={"files": (io.BytesIO(TEI), "s.xml")},
                  content_type="multipart/form-data").get_json()["imported"][0]


def test_import_detect_export_with_pes_prefix(client, monkeypatch):
    imp = import_syc(client)
    assert imp["lang"] == "syc" and imp["in_source"] == 1                        # the hand-made Peshitta:Ps:23:1
    sid = imp["id"]
    seen = {}
    answer = [{"text": bare(MT), "resolved_references": ["mt_5:9"], "quote_type": "full", "marked": True}]
    monkeypatch.setattr(sd, "generate", lambda prompt: (seen.update(p=prompt), lb.Result(json.dumps({"quotes": answer})))[1])
    r = client.post(f"/api/sources/{sid}/process").get_json()
    assert r["count"] == 1
    p = seen["p"]
    assert "Peshitta" in p and "Hebrew numbering" in p and "ܟܬܝܒ" in p and "mt_5:9 | verse:" in p       # corpus, notes, formulas, RAG candidate
    ann = {a["quote_text"][:6]: a for a in client.get(f"/api/sources/{sid}").get_json()["annotations"]}
    assert ann[bare(MT)[:6]]["refs"] == ["mt_5:9"] and ann[bare(MT)[:6]]["score"] == 1.0
    assert ann[bare(PS)[:6]]["origin"] == "source" and ann[bare(PS)[:6]]["refs"] == ["ps_23:1"]
    out = client.get(f"/api/sources/{sid}/export/tei").data.decode()
    assert 'cRef="Peshitta:Mt:5:9"' in out and out.count('cRef="Peshitta:Ps:23:1"') == 1
    assert "Peshitta" in out and "Peshitta" in out.split("<text>")[0]                 # refsDecl mentions the corpus
    again = client.post("/api/import/tei", data={"files": (io.BytesIO(out.encode()), "again.xml")},
                        content_type="multipart/form-data").get_json()["imported"][0]
    assert again["lang"] == "syc" and again["annotations"] == 1 and again["in_source"] == 1


def test_text_search_only_finds_syriac_quotations(client, monkeypatch):
    monkeypatch.setattr(sd, "generate", lambda prompt: (_ for _ in ()).throw(AssertionError("no model")))
    client.post("/api/settings", json={"api_provider": "rag"})
    try:
        text = f"ܐܡܪ ܡܪܢ ܕܝܢ: {bare(MT)} ܘܗܟܢܐ ܡܠܦ ܠܬܠܡܝܕܘܗܝ ܒܡܬܠܐ ܐܚܪܢܐ ܕܠܐ ܐܝܬ ܒܗ ܟܬܒܐ."
        sid = client.post("/api/sources", json={"name": "Syr", "text": text}).get_json()
        assert sid["lang"] == "syc"
        r = client.post(f"/api/sources/{sid['id']}/process").get_json()
        assert r["count"] == 1
        a = client.get(f"/api/sources/{sid['id']}").get_json()["annotations"][0]
        assert a["refs"] == ["mt_5:9"] and a["marked"] == 1                       # ܐܡܪ stands just before: introduced by a formula
        out = client.get(f"/api/sources/{sid['id']}/export/tei").data.decode()
        assert 'xml:lang="syc"' in out and 'cRef="Peshitta:Mt:5:9"' in out
    finally:
        client.post("/api/settings", json={"api_provider": "gemini"})


def test_viewer_is_right_to_left_for_syriac_only(client):
    sid = import_syc(client)["id"]
    page = client.get(f"/viewer/{sid}").data.decode()
    assert 'id="text-panel" lang="syc" dir="rtl"' in page and ":lang(syc)" in page and 'dir="auto"' in page
    assert client.get(f"/api/bible/books?lang=syc").get_json()
    assert client.get("/api/bible/gn/1/verses?lang=syc").get_json()[0]["text"].startswith("ܒܪܫܝܬ")
    gid = client.post("/api/sources", json={"name": "G", "text": "ἐν ἀρχῇ ἐποίησεν ὁ θεός"}).get_json()["id"]
    assert ' dir="rtl"' not in client.get(f"/viewer/{gid}").data.decode().split('id="text-panel"')[1].split(">")[0]


def test_license_notice_for_the_syriac_data_exists():
    lic = (Path(__file__).resolve().parent.parent / "data" / "syriac" / "LICENSES.md").read_text(encoding="utf8")
    assert "non-commercial" in lic and "ETCBC/peshitta" in lic and "ShareAlike" in lic


def test_about_and_readme_describe_syriac_and_its_licence():
    client = sd.app.test_client()
    html = client.get("/about").data.decode()
    assert "Syriac" in html and "Peshitta:Gn:1:1" in html and "CC BY-NC 4.0" in html and "build_syriac_bible.py" in html
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf8")
    assert "CC BY-NC 4.0" in readme and "non-commercial" in readme and "scripts/build_syriac_bible.py" in readme
