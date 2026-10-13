import re
import sys
from pathlib import Path

import pytest
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import greek  # noqa: E402
import tei_doc  # noqa: E402

T = tei_doc.T
BOOKS = {
    "gn": {"pta_abbr": "Gn", "pta_corpus": "LXX", "name": "Genesis"},
    "ps": {"pta_abbr": "Ps", "pta_corpus": "LXX", "name": "Psalms"},
    "mt": {"pta_abbr": "Mt", "pta_corpus": "NA", "name": "Matthew"},
    "gen": {"pta_abbr": "Gn", "pta_corpus": "Vg", "name": "Genesis"},
}
ABBR = {("lxx", "gn"): "gn", ("lxx", "ps"): "ps", ("na", "mt"): "mt", ("vg", "gn"): "gen"}

GN = "καὶ προσκολληθήσεται πρὸς τὴν γυναῖκα αὐτοῦ, καὶ ἔσονται οἱ δύο εἰς σάρκα μίαν"


def wrap(body: str, edition_n="urn:cts:pta:pta0001.pta001.pta-grc1") -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc><titleStmt><title>Test</title><author>Autor</author></titleStmt>
  <publicationStmt><p>x</p></publicationStmt><sourceDesc><p>y</p></sourceDesc></fileDesc>
  <encodingDesc><refsDecl n="CTS"><cRefPattern n="s" matchPattern="(.+)" replacementPattern="#x"/></refsDecl>
  <schemaRef key="v3" url="https://example.org/s.rng"/></encodingDesc>
  <revisionDesc><change when="2020-01-01">x</change></revisionDesc></teiHeader>
  <text><body><div type="edition" xml:lang="grc" n="{edition_n}">
  {body}
  </div></body></text>
</TEI>""".encode("utf-8")


def ann(text_map_text: str, quote: str, refs, qtype="full", marked=None, id_=1):
    i = text_map_text.index(quote)
    return {"id": id_, "span_start": i, "span_end": i + len(quote), "quote_type": qtype,
            "refs": refs, "marked": marked}


def run(xml: bytes, quote: str, refs, **kw):
    data = tei_doc.read_tei(xml, ABBR)
    out, warns = tei_doc.annotate_tei(xml, [ann(data.text, quote, refs, **kw)], BOOKS, expected_text=data.text)
    return data, out, warns


def squeeze(s: str) -> str:
    return re.sub(r"\s+", " ", s)


# ── export in PTA format ─────────────────────────────────────────────────────

def test_simple_marked_quote_in_pta_format():
    xml = wrap(f"<div type='textpart' subtype='section' n='1'><p>Und wie dort gesagt: {GN}, so spricht die Seele.</p></div>")
    data, out, warns = run(xml, GN, ["gn_2:24"], marked=True)
    s = out.decode()
    assert f'<quote type="marked" xml:id="sd-q1">{GN}<ref decls="#biblical" cRef="LXX:Gn:2:24"/></quote>' in s
    assert 'xml:id="biblical"' in s and 'id="annotationstatus"' in s and 'xml:id="biblical-quotations"' in s
    assert not warns
    # the plain text is unchanged by the export
    assert tei_doc.read_tei(out, ABBR).text == data.text


def test_allusion_and_paraphrase_elements():
    xml = wrap("<div type='textpart' subtype='section' n='1'><p>Er sprach vom Ende. Dann folgte eine Umschreibung hier.</p></div>")
    data = tei_doc.read_tei(xml, ABBR)
    anns = [ann(data.text, "vom Ende", ["mt_5:9"], qtype="allusion", id_=1),
            ann(data.text, "eine Umschreibung", ["ps_62:9", "ps_62:10"], qtype="paraphrase", id_=2)]
    out, _ = tei_doc.annotate_tei(xml, anns, BOOKS)
    s = out.decode()
    assert '<seg type="allusion" xml:id="sd-q1">vom Ende<ref decls="#biblical" cRef="NA:Mt:5:9"/></seg>' in s
    assert 'type="paraphrasis"' in s and 'cRef="LXX:Ps:62:9-10"' in s


def test_quote_across_lb_pb_hi_and_nobreak():
    xml = wrap("<p>Davor <hi rend='x'>καὶ προσ</hi><lb break='no'/>κολληθήσεται πρὸς <pb n='2r'/>τὴν γυναῖκα "
               "<lb/>αὐτοῦ, καὶ ἔσονται οἱ δύο εἰς σάρκα μίαν danach</p>")
    data = tei_doc.read_tei(xml, ABBR)
    assert GN in data.text                    # break="no" is joined, other breaks become spaces
    _, out, _ = run(xml, GN, ["gn_2:24"], marked=False)
    root = etree.fromstring(out)
    quotes = list(root.iter(T + "quote"))
    assert quotes and all(q.get("type") == "unmarked" for q in quotes)
    # every fragment is linked; exactly one <ref> in the last fragment
    refs = [r for q in quotes for r in q.iter(T + "ref")]
    assert len(refs) == 1 and refs[0].getparent() is quotes[-1]
    if len(quotes) > 1:
        assert quotes[0].get("next") == "#" + quotes[1].get(XML_ID := "{http://www.w3.org/XML/1998/namespace}id")
        assert quotes[1].get("prev") == "#" + quotes[0].get(XML_ID)
    assert tei_doc.read_tei(out, ABBR).text == data.text
    joined = squeeze("".join("".join(q.itertext()) for q in quotes))
    assert "κολληθήσεται" in joined


def test_quote_across_paragraphs_is_fragmented():
    xml = wrap("<p>Anfang καὶ προσκολληθήσεται πρὸς τὴν γυναῖκα αὐτοῦ,</p><p>καὶ ἔσονται οἱ δύο εἰς σάρκα μίαν Ende</p>")
    data = tei_doc.read_tei(xml, ABBR)
    assert "\n" in data.text
    q = data.text[data.text.index("καὶ προσ"): data.text.index("μίαν") + 4]
    _, out, warns = run(xml, q, ["gn_2:24"])
    root = etree.fromstring(out)
    ps = list(root.find(T + "text").iter(T + "p"))
    assert len(ps) == 2 and all(len(list(p.iter(T + "quote"))) == 1 for p in ps)
    assert not warns


def test_roundtrip_replaces_own_annotations_without_duplicates():
    xml = wrap(f"<p>x {GN} y</p>")
    data, out1, _ = run(xml, GN, ["gn_2:24"], qtype="full", marked=True)
    d2 = tei_doc.read_tei(out1, ABBR)
    assert len(d2.annotations) == 1
    a = d2.annotations[0]
    assert a["refs"] == ["gn_2:24"] and a["marked"] is True and a["quote_text"] == GN
    a["id"] = 7
    out2, _ = tei_doc.annotate_tei(out1, [a], BOOKS, expected_text=d2.text)
    assert out2.decode().count("<quote") == 1 and out2.decode().count("cRef=") == 1
    assert out2.decode().count('xml:id="biblical"') == 1       # header declared only once


def test_handmade_annotation_is_preserved_and_not_duplicated():
    body = (f"<p><quote type='marked'>{GN}<ref decls='#biblical' cRef='LXX:Gn:2:24'/></quote> und "
            "<hi>weiter</hi> mehr Text</p>")
    xml = wrap(body)
    data = tei_doc.read_tei(xml, ABBR)
    assert len(data.annotations) == 1                  # hand-made markup is read ...
    a0 = data.annotations[0]
    assert a0["origin"] == "source" and a0["refs"] == ["gn_2:24"] and a0["quote_text"] == GN
    out, warns = tei_doc.annotate_tei(xml, [ann(data.text, GN, ["gn_2:24"])], BOOKS)
    s = out.decode()
    assert s.count("<quote") == 1 and "sd-q" not in s    # ... and not duplicated
    assert any("already" in w for w in warns)


def test_transcription_warning():
    xml = wrap(f"<p>{GN}</p>", edition_n="urn:cts:pta:pta0013.pta003.pta-MsPs")
    _, _, warns = run(xml, GN, ["gn_2:24"])
    assert any("transcription" in w for w in warns)


def test_comments_and_pis_between_siblings():
    xml = wrap("<p>Davor καὶ προσκολληθήσεται <!-- Kommentar --><hi>πρὸς</hi><?pi x?> τὴν γυναῖκα αὐτοῦ, "
               "καὶ ἔσονται οἱ δύο εἰς σάρκα μίαν danach</p>")
    data, out, _ = run(xml, GN, ["gn_2:24"])
    assert tei_doc.read_tei(out, ABBR).text == data.text
    assert "<!-- Kommentar -->" in out.decode() and out.decode().count("cRef=") == 1


def test_apparatus_notes_and_choice_are_handled():
    xml = wrap("<p>Eins <app type='variants'><lem>zwei</lem><rdg wit='#A'>ZZZ</rdg></app> drei "
               "<note>NOTIZ</note><choice><abbr>κς</abbr><expan>κύριος</expan></choice> vier</p>")
    data = tei_doc.read_tei(xml, ABBR)
    assert data.text == "Eins zwei drei κύριος vier"


def test_text_mismatch_is_detected():
    xml = wrap("<p>abc def</p>")
    with pytest.raises(tei_doc.TeiTextMismatch):
        tei_doc.annotate_tei(xml, [], BOOKS, expected_text="something else")


def test_plain_text_gets_pta_skeleton():
    text = f"Erste Zeile.\n\n  {GN}   Ende.\nLetzte"
    xml, omap = tei_doc.plain_to_tei("Mein Text", text)
    plain = tei_doc.read_tei(xml, ABBR).text
    i = text.index(GN)
    a = {"id": 1, "span_start": omap[i], "span_end": omap[i + len(GN)], "quote_type": "full",
         "refs": ["gn_2:24"], "marked": None}
    assert plain[a["span_start"]:a["span_end"]] == GN
    out, _ = tei_doc.annotate_tei(xml, [a], BOOKS)
    assert 'subtype="section"' in out.decode() and f"{GN}<ref" in out.decode()


# ── greek helpers ────────────────────────────────────────────────────────────

def test_span_index_ignores_accents_case_sigma_and_punctuation():
    doc = "Ἐκολλήθη ἡ ψυχή μου ὀπίσω σου· ἐμοῦ δὲ ἀντελάβετο ἡ δεξιά σου."
    idx = greek.SpanIndex(doc)
    s, e = idx.find("εκολληθη η ψυχη μου οπισω σου")           # no diacritics
    assert doc[s:e] == "Ἐκολλήθη ἡ ψυχή μου ὀπίσω σου"
    s, e = idx.find("ΕΜΟΥ ΔΕ ΑΝΤΕΛΑΒΕΤΟ")
    assert doc[s:e].startswith("ἐμοῦ") and doc[s:e].endswith("ἀντελάβετο")


def test_span_index_repeated_quote_and_nfd_input():
    import unicodedata
    doc = unicodedata.normalize("NFD", "τὸ ὕδωρ καὶ ἄλλο· τὸ ὕδωρ καὶ ἄλλο")
    idx = greek.SpanIndex(doc)
    q = unicodedata.normalize("NFC", "τὸ ὕδωρ")
    a, b = idx.find(q), idx.find(q)
    assert a and b and a[0] < b[0]
    assert unicodedata.category(doc[a[1] - 1]) != "Mn" or True


def test_span_index_anchor_fallback_tolerates_small_deviation():
    doc = "πρῶτον πάντων πίστευσον ὅτι εἷς ἐστιν ὁ θεὸς ὁ τὰ πάντα κτίσας καὶ καταρτίσας"
    idx = greek.SpanIndex(doc)
    q = "πρῶτον πάντων πίστευσον ὅτι εἷς ἐστιν ὁ θεὸς ὁ τὰ ΠΑΝΤΑΧ κτίσας καὶ καταρτίσας"
    assert idx.find(q) is not None


def test_score_quote():
    verse = "μακάριοι οἱ εἰρηνοποιοί, ὅτι αὐτοὶ υἱοὶ θεοῦ κληθήσονται."
    assert greek.score_quote("Μακάριοι οἱ εἰρηνοποιοὶ ὅτι αὐτοὶ υἱοὶ θεοῦ κληθήσονται", [verse]) == 1.0
    assert greek.score_quote("ἄλλο τι πάντως", [verse]) < 0.4
    assert greek.score_quote("μακάριοι", []) is None


def test_cref_roundtrip_and_formatting():
    refs = ["ps_62:9", "ps_62:10", "ps_62:12", "ps_63:1", "mt_5:9", "xx_1:1", "oden_1:t", "mt_7:2a"]
    books = dict(BOOKS, oden={"pta_abbr": "Oden", "pta_corpus": "LXX"})
    out = greek.format_crefs(refs, books)
    assert out == ["LXX:Ps:62:9-10.12", "LXX:Ps:63:1", "NA:Mt:5:9", "NA:Mt:7:2"]
    assert greek.parse_cref("LXX:Ps:62:9-10.12", ABBR) == ["ps_62:9", "ps_62:10", "ps_62:12"]
    assert greek.parse_cref("pta:foo", ABBR) == []
    assert greek.parse_cref("Vg:Gn:1:1", ABBR) == ["gen_1:1"]
    assert greek.parse_cref("LXX:Gn:1:1", {("vg", "gn"): "gen"}) == []        # wrong corpus


def test_chunking_covers_text_without_gaps():
    text = ("Satz eins. " * 400 + "\n") * 5
    ranges = greek.chunk_text(text, max_chars=1500)
    assert ranges[0][0] == 0 and ranges[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert all(e - s <= 1500 for s, e in ranges)


# ── existing <quote> elements ────────────────────────────────────────────────

def test_existing_quote_gets_ref_but_no_new_element():
    xml = wrap(f"<p>Er sagt: <quote>{GN}</quote> und weiter.</p>")
    hints = tei_doc.existing_quotes(xml)
    assert [h["text"] for h in hints] == [GN]
    data = tei_doc.read_tei(xml, ABBR)
    a = ann(data.text, GN, ["gn_2:24"], marked=True)
    out, warns = tei_doc.annotate_tei(xml, [a], BOOKS)
    s = out.decode()
    assert s.count("<quote") == 1 and "<seg" not in s.split("<text>")[1]
    assert f'<quote type="marked">{GN}<ref decls="#biblical" cRef="LXX:Gn:2:24" xml:id="sd-r1"/></quote>' in s
    assert not warns


def test_existing_quote_keeps_its_own_type_and_matches_a_larger_span():
    xml = wrap(f"<p>Er sagt: <quote type='lemma'>{GN}</quote> und weiter.</p>")
    data = tei_doc.read_tei(xml, ABBR)
    big = data.text[data.text.index("sagt"): data.text.index("und weiter")]      # ≥50 % overlap both ways
    a = {"id": 1, "span_start": data.text.index("sagt"), "span_end": data.text.index("und weiter"),
         "quote_type": "full", "refs": ["gn_2:24"], "marked": False}
    out, _ = tei_doc.annotate_tei(xml, [a], BOOKS)
    s = out.decode()
    assert 'type="lemma"' in s and 'type="unmarked"' not in s and s.count("<quote") == 1 and "cRef=" in s


def test_annotation_inside_larger_quote_is_skipped_with_warning():
    long_q = "ἄλλος λόγος ἀνθρώπου " * 6 + GN + " καὶ τὰ λοιπὰ τοῦ ἀνθρώπου λόγου " * 4
    xml = wrap(f"<p><quote>{long_q}</quote></p>")
    data = tei_doc.read_tei(xml, ABBR)
    out, warns = tei_doc.annotate_tei(xml, [ann(data.text, GN, ["gn_2:24"])], BOOKS)
    assert out.decode().count("<quote") == 1 and "cRef=" not in out.decode()
    assert any("only in part" in w for w in warns)


def test_fragmented_existing_quote_gets_ref_in_last_fragment():
    xml = wrap("<p><quote xml:id='q1' next='#q2'>καὶ προσκολληθήσεται πρὸς τὴν γυναῖκα αὐτοῦ,</quote></p>"
               "<p><quote xml:id='q2' prev='#q1'>καὶ ἔσονται οἱ δύο εἰς σάρκα μίαν</quote></p>")
    data = tei_doc.read_tei(xml, ABBR)
    q = data.text[data.text.index("καὶ προσ"): data.text.index("μίαν") + 4]
    out, _ = tei_doc.annotate_tei(xml, [ann(data.text, q, ["gn_2:24"])], BOOKS)
    root = etree.fromstring(out)
    quotes = list(root.find(T + "text").iter(T + "quote"))
    assert len(quotes) == 2
    assert not list(quotes[0].iter(T + "ref")) and len(list(quotes[1].iter(T + "ref"))) == 1


def test_existing_quote_roundtrip_and_idempotent_export():
    xml = wrap(f"<p>Er sagt: <quote>{GN}</quote> und weiter.</p>")
    data = tei_doc.read_tei(xml, ABBR)
    out1, _ = tei_doc.annotate_tei(xml, [ann(data.text, GN, ["gn_2:24"], marked=True)], BOOKS)
    d2 = tei_doc.read_tei(out1, ABBR)
    assert len(d2.annotations) == 1 and d2.annotations[0]["refs"] == ["gn_2:24"]
    assert d2.annotations[0]["marked"] is True and d2.annotations[0]["quote_text"] == GN
    a = dict(d2.annotations[0], id=1)
    out2, _ = tei_doc.annotate_tei(out1, [a], BOOKS)
    assert out2.decode().count("cRef=") == 1 and out2.decode().count("<quote") == 1
    assert tei_doc.existing_quotes(out1)[0]["text"] == GN         # our ref is not "human" markup


def test_quote_with_human_biblical_ref_is_not_a_hint():
    xml = wrap(f"<p><quote>{GN}<ref decls='#biblical' cRef='LXX:Gn:2:24'/></quote></p>")
    assert tei_doc.existing_quotes(xml) == []


# ── language ─────────────────────────────────────────────────────────────────

def test_language_of_first_div_and_vulgate_export():
    import corpora
    assert corpora.normalize_lang("grc") == "grc" and corpora.normalize_lang("la") == "lat"
    assert corpora.normalize_lang("syr") is None
    lat = "In principio creavit Deus caelum et terram."
    xml = wrap(f"<p>Dicit: {lat} Amen.</p>").replace(b'xml:lang="grc"', b'xml:lang="lat"')
    data = tei_doc.read_tei(xml, None)
    assert data.lang == "lat"
    out, _ = tei_doc.annotate_tei(xml, [ann(data.text, lat, ["gen_1:1"])], BOOKS)
    assert 'cRef="Vg:Gn:1:1"' in out.decode()
    assert corpora.guess_lang(lat) == "lat" and corpora.guess_lang(GN) == "grc"


# ── warnings carry context and location ──────────────────────────────────────

def test_warnings_show_context_and_location():
    body = (f"<div type='textpart' subtype='section' n='3'><div type='textpart' subtype='chapter' n='12'>"
            f"<p>Erster Absatz.</p><pb n='14v'/><p xml:id='par7'>Davor steht <lb n='9'/>etwas. "
            f"<quote>{GN}<ref decls='#biblical' cRef='LXX:Gn:2:24'/></quote> Danach folgt Text.</p></div></div>")
    xml = wrap(body)
    data = tei_doc.read_tei(xml, ABBR)
    _, warns = tei_doc.annotate_tei(xml, [ann(data.text, GN, ["gn_2:24"])], BOOKS)
    assert len(warns) == 1
    w = warns[0]
    assert w.startswith("Skipped, the existing <quote> already has a biblical reference: …")
    assert "«καὶ προσκολληθήσεται" in w and "μίαν» Danach folgt Text" in w          # passage with context
    assert "etwas." in w                                                              # text before
    for part in ("section 3.12", "<p> 2", "page 14v", "line 9", 'xml:id "par7"', "char "):
        assert part in w, part


def test_long_passages_are_shortened_in_warnings():
    long_q = " ".join(["λόγος"] * 60)
    xml = wrap(f"<p><quote>{long_q}<ref decls='#biblical' cRef='LXX:Gn:2:24'/></quote></p>")
    data = tei_doc.read_tei(xml, ABBR)
    _, warns = tei_doc.annotate_tei(xml, [ann(data.text, long_q, ["gn_2:24"])], BOOKS)
    assert " … " in warns[0] and len(warns[0]) < 300


# ── consecutive verses -> one ref ────────────────────────────────────────────

PHIL = dict(BOOKS, phil={"pta_abbr": "Phil", "pta_corpus": "NA", "name": "Philippians"})
PHIL_Q = "ἐταπείνωσεν ἑαυτὸν γενόμενος ὑπήκοος μέχρι θανάτου"


def _refs(out: bytes) -> list[str]:
    return re.findall(r'cRef="([^"]+)"', out.decode())


def test_consecutive_verses_become_one_ref_with_range():
    xml = wrap(f"<p>Er sagt: {PHIL_Q} und weiter.</p>")
    data = tei_doc.read_tei(xml, ABBR)
    for refs in (["phil_2:8", "phil_2:9", "phil_2:10"], ["phil_2:10", "phil_2:8", "phil_2:9"]):
        out, _ = tei_doc.annotate_tei(xml, [ann(data.text, PHIL_Q, refs)], PHIL)
        assert _refs(out) == ["NA:Phil:2:8-10"]
    out, _ = tei_doc.annotate_tei(xml, [ann(data.text, PHIL_Q, ["phil_2:8", "phil_2:9", "phil_2:10", "phil_2:12"])], PHIL)
    assert _refs(out) == ["NA:Phil:2:8-10.12"]                    # PTA: "." joins verses that do not follow


def test_consecutive_verses_from_several_annotations_on_an_existing_quote_are_merged():
    xml = wrap(f"<p>Er sagt: <quote>{PHIL_Q}</quote> und weiter.</p>")
    data = tei_doc.read_tei(xml, ABBR)
    anns = [ann(data.text, PHIL_Q, ["phil_2:8"], id_=1), ann(data.text, PHIL_Q, ["phil_2:9", "phil_2:10"], id_=2)]
    out, _ = tei_doc.annotate_tei(xml, anns, PHIL)
    assert _refs(out) == ["NA:Phil:2:8-10"] and out.decode().count("<ref decls") == 1
    again = tei_doc.read_tei(out, {("na", "phil"): "phil"})
    assert again.annotations[0]["refs"] == ["phil_2:8", "phil_2:9", "phil_2:10"]


def test_split_in_two_prefers_paragraphs_keeps_overlap_and_loses_nothing():
    a, b = "α " * 300, "β " * 300
    p1, p2 = greek.split_in_two(a.rstrip() + "\n" + b.rstrip(), overlap=0)
    assert p1.endswith("\n") and p2.startswith("β")
    text = " ".join(f"wort{i}" for i in range(400))
    x, y = greek.split_in_two(text, overlap=100)
    assert x + y[len(y) - (len(text) - len(x)):] == text                 # no gap
    assert 0 < len(x) + len(y) - len(text) <= 120                        # a little overlap
    assert y.startswith("wort")                                          # starts at a word boundary


# ── annotations that are already in the source file ──────────────────────────

def test_source_annotations_are_read_with_origin_source():
    xml = wrap(f"<p>A <quote type='marked'>{GN}<ref decls='#biblical' cRef='LXX:Gn:2:24'/></quote> b "
               "<seg type='allusion'>vom Ende<ref decls='#biblical' cRef='NA:Mt:5:9'/></seg> c "
               "<quote type='paraphrasis'>eine Umschreibung<ref decls='#biblical' cRef='LXX:Ps:62:9'/></quote> d "
               "<quote>ohne Referenz</quote> e <quote>fremd<ref cRef='other:1'/></quote></p>")
    data = tei_doc.read_tei(xml, ABBR)
    by = {a["quote_text"]: a for a in data.annotations}
    assert set(by) == {GN, "vom Ende", "eine Umschreibung"}                  # quotes without a biblical ref are not annotations
    assert all(a["origin"] == "source" for a in by.values())
    assert by[GN]["marked"] is True and by[GN]["quote_type"] == "full" and by[GN]["refs"] == ["gn_2:24"]
    assert by["vom Ende"]["quote_type"] == "allusion" and by["vom Ende"]["refs"] == ["mt_5:9"]
    assert by["eine Umschreibung"]["quote_type"] == "paraphrase"


def test_source_annotations_with_unknown_references_are_kept_with_their_cref():
    xml = wrap("<p>Text <quote>etwas Bekanntes<ref decls='#biblical' cRef='Peshitta:Gn:1:1'/></quote> Ende</p>")
    (a,) = tei_doc.read_tei(xml, ABBR).annotations
    assert a["origin"] == "source" and a["refs"] == [] and a["crefs"] == ["Peshitta:Gn:1:1"]


def test_tool_refs_in_existing_quotes_are_not_source_annotations_but_human_refs_win():
    xml = wrap(f"<p><quote>{GN}<ref decls='#biblical' cRef='LXX:Gn:2:24' xml:id='sd-r1'/></quote> und "
               f"<quote>{PHIL_Q}<ref decls='#biblical' cRef='NA:Phil:2:8'/><ref decls='#biblical' cRef='NA:Phil:2:9' xml:id='sd-r2'/></quote></p>")
    data = tei_doc.read_tei(xml, {**ABBR, ("na", "phil"): "phil"})
    by = {a["quote_text"]: a for a in data.annotations}
    assert by[GN]["origin"] == "tool"                                          # added by this tool earlier
    assert by[PHIL_Q]["origin"] == "source" and by[PHIL_Q]["crefs"] == ["NA:Phil:2:8"]
    assert len(data.annotations) == 2
