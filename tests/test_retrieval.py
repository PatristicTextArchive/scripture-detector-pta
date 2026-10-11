import io
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

G = corpora.get_corpus("grc")
L = corpora.get_corpus("lat")
MT = G.verses["mt_5:9"]
PS = G.verses["ps_62:9"]


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


# ── retrieval ────────────────────────────────────────────────────────────────

def test_candidates_find_the_verse_despite_accents_and_inflection():
    text = "Er schreibt viel und erinnert sich: " + strip_accents(PS) + " und so fort."
    c = retrieval.candidates(text, G, top=10)
    assert "ps_62:9" in [x.key for x in c[:3]]
    best = next(x for x in c if x.key == "ps_62:9")
    assert strip_accents(PS)[:20] in text[best.start:best.end]          # the span is the matching passage


def test_candidates_latin_orthographic_variants():
    text = "Dicit scriptura: in principio creavit deus coelum et terram."      # coelum / caelum
    assert "gen_1:1" in [x.key for x in retrieval.candidates(text, L, top=5)]


def test_lexical_match_full_and_partial_and_span():
    text = f"Der Apostel sagt: {MT} Das ist deutlich."
    (m,) = retrieval.lexical_matches(text, G)
    assert "mt_5:9" in m.refs and m.quote_type == "full" and m.coverage >= 0.9
    assert text[m.start:m.end].startswith("μακάριοι") and text[m.start:m.end].endswith("κληθήσονται")
    part = "Er sagt: ὅτι αὐτοὶ υἱοὶ θεοῦ κληθήσονται und dann etwas anderes."
    (p,) = retrieval.lexical_matches(part, G, min_words=4, min_weight=15)
    assert "mt_5:9" in p.refs and p.quote_type == "partial"


def test_formulas_and_short_phrases_are_not_reported():
    assert retrieval.lexical_matches("εἰς τοὺς αἰῶνας τῶν αἰώνων. Ἀμήν", G) == []
    assert retrieval.lexical_matches("Der Herr sagte zu ihm: ἐπὶ τὴν γῆν und kam", G) == []
    assert retrieval.lexical_matches("Ein ganz anderer deutscher Satz ohne Griechisch.", G) == []


def test_consecutive_verses_form_one_match():
    text = "Wie geschrieben steht: " + G.verses["phil_2:8"] + " " + G.verses["phil_2:9"] + " Amen."
    ms = retrieval.lexical_matches(text, G)
    assert len(ms) == 1 and ms[0].refs == ["phil_2:8", "phil_2:9"]
    assert text[ms[0].start:ms[0].end].startswith("ἐταπείνωσεν") and text[ms[0].start:ms[0].end].endswith("ὄνομα")
    # verses with other words between them stay separate
    apart = G.verses["phil_2:8"] + " Und dann sagt er etwas ganz anderes: " + G.verses["phil_2:9"]
    assert len(retrieval.lexical_matches(apart, G)) == 2


def test_index_is_built_once():
    assert retrieval.get_index(G) is retrieval.get_index(G)


# ── prompt ───────────────────────────────────────────────────────────────────

def test_candidates_are_formatted_within_the_budget():
    text = "x " + PS + " y " + MT
    cands = retrieval.candidates(text, G, top=30)
    block = sd.format_candidates(cands, text, G, budget=500)
    assert "ps_62:9 | verse:" in block or "mt_5:9 | verse:" in block
    assert len(block) < 700 and "| in the text:" in block
    p = sd.build_prompt(text, G, None, json_hint=True, candidates=block, compact=True)
    assert "Candidate verses:" in p and block in p and "Κατὰ Ματθαῖον" not in p       # compact: no Greek names
    assert "Candidate verses:" not in sd.build_prompt(text, G)


# ── whole app ────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(monkeypatch):
    sd.app.config["TESTING"] = True
    c = sd.app.test_client()
    c.post("/api/settings", json={"api_provider": "gemini", "rag": "on", "rag_add": "on", "rag_top_k": ""})
    return c


def new_source(c, text):
    return c.post("/api/sources", json={"name": "T", "text": text}).get_json()["id"]


def refs_of(c, sid):
    return {a["quote_text"][:12]: a["refs"] for a in c.get(f"/api/sources/{sid}").get_json()["annotations"]}


TEXT = f"Der Autor sagt: γέγραπται· {MT} Und später: {PS}. Ende des Textes."


def test_prompt_contains_candidates_from_the_bible_texts(client, monkeypatch):
    seen = {}

    def fake(prompt):
        seen["p"] = prompt
        return lb.Result('{"quotes": []}')
    monkeypatch.setattr(sd, "generate", fake)
    sid = new_source(client, TEXT)
    client.post(f"/api/sources/{sid}/process")
    assert "Candidate verses:" in seen["p"] and "mt_5:9 | verse:" in seen["p"] and "ps_62:9 | verse:" in seen["p"]
    client.post("/api/settings", json={"rag": "off"})
    client.post(f"/api/sources/{sid}/process")
    assert "Candidate verses:" not in seen["p"]


def test_wrong_reference_is_repaired_and_missed_quote_is_added(client, monkeypatch):
    wrong = [{"text": PS.rstrip("."), "resolved_references": ["ps_33:5"], "quote_type": "full", "marked": False}]
    monkeypatch.setattr(sd, "generate", lambda prompt: lb.Result(__import__("json").dumps({"quotes": wrong})))
    sid = new_source(client, TEXT)
    assert client.post(f"/api/sources/{sid}/process").get_json()["count"] == 2
    r = refs_of(client, sid)
    assert r[PS[:12]] == ["ps_62:9"]                                       # repaired (ps_33:5 was a valid verse, but wrong)
    assert r[MT[:12]] == ["mt_5:9"]                                        # added although the model did not report it
    ann = {a["quote_text"][:12]: a for a in client.get(f"/api/sources/{sid}").get_json()["annotations"]}
    assert ann[MT[:12]]["marked"] == 1 and ann[MT[:12]]["score"] == 1.0    # γέγραπται before the passage


def test_matches_are_not_added_when_switched_off(client, monkeypatch):
    monkeypatch.setattr(sd, "generate", lambda prompt: lb.Result('{"quotes": []}'))
    client.post("/api/settings", json={"rag_add": "off"})
    sid = new_source(client, TEXT)
    assert client.post(f"/api/sources/{sid}/process").get_json()["count"] == 0


def test_good_model_results_are_left_alone(client, monkeypatch):
    good = [{"text": MT, "resolved_references": ["mt_5:9"], "quote_type": "partial", "marked": True}]
    monkeypatch.setattr(sd, "generate", lambda prompt: lb.Result(__import__("json").dumps({"quotes": good})))
    sid = new_source(client, f"Er sagt {MT} fertig.")
    client.post(f"/api/sources/{sid}/process")
    a = client.get(f"/api/sources/{sid}").get_json()["annotations"]
    assert len(a) == 1 and a[0]["quote_type"] == "partial" and a[0]["marked"] == 1      # the model's judgement is kept


def test_text_search_only_needs_no_model(client, monkeypatch):
    def boom(prompt):
        raise AssertionError("no model must be called")
    monkeypatch.setattr(sd, "generate", boom)
    client.post("/api/settings", json={"api_provider": "rag"})
    try:
        assert sd.model_label() == "text search (no model)"
        sid = new_source(client, TEXT)
        assert client.post(f"/api/sources/{sid}/process").get_json()["count"] == 2
        assert set(sum(refs_of(client, sid).values(), [])) == {"mt_5:9", "ps_62:9"}
        assert client.post("/api/test-backend").get_json()["status"] == "ok"
        out = client.get(f"/api/sources/{sid}/export/tei").data.decode()
        assert 'cRef="NA:Mt:5:9"' in out and 'cRef="LXX:Ps:62:9"' in out
        page = client.get(f"/viewer/{sid}").data
        assert b"text search (no model)" in page and b'id="model-select"' not in page
    finally:
        client.post("/api/settings", json={"api_provider": "gemini"})


def test_latin_text_search(client, monkeypatch):
    monkeypatch.setattr(sd, "generate", lambda prompt: lb.Result('{"quotes": []}'))
    sid = new_source(client, "Scriptum est: In principio creavit Deus caelum et terram. Amen.")
    assert client.post(f"/api/sources/{sid}/process").get_json()["count"] == 1
    assert refs_of(client, sid)["In principio"] == ["gen_1:1"]


def test_settings_page_has_rag_options(client):
    page = client.get("/settings").data.decode()
    for needle in ('id="rag-on"', 'id="rag-add"', 'id="rag-top-k"', 'id="rc-rag"'):
        assert needle in page, needle
    client.post("/api/settings", json={"rag": "off", "rag_top_k": "7"})
    s = client.get("/api/settings").get_json()
    assert s["rag"] == "off" and s["rag_top_k"] == "7"
    client.post("/api/settings", json={"rag": "on", "rag_top_k": ""})
    assert sd.get_rag_top_k() == 50


def test_about_page_describes_rag(client):
    html = client.get("/about").data.decode()
    assert "Finding Quotations with the Bible Texts (RAG)" in html and "Text search only" in html
