"""Read arbitrary TEI files and write detected biblical quotations back into
them, following the encoding guidelines of the Patristic Text Archive (PTA):
https://pta.bbaw.de/en/project/encoding-guidelines

Reading
    ``read_tei`` extracts the running text of ``<text>`` (lemma readings only,
    notes/variants skipped, ``<lb break="no"/>`` joined) together with a map
    from every character of the plain text back to its XML text node.

Writing
    ``annotate_tei`` inserts, for each annotation,
        <quote type="marked|unmarked|paraphrasis"> ... <ref decls="#biblical" cRef="LXX:Ps:62:9"/></quote>
        <seg type="allusion"> ... <ref decls="#biblical" cRef="NA:Jn:1:1"/></seg>
    Quotations that cross element boundaries are fragmented and linked with
    ``@prev``/``@next`` (PTA "Overlaps"), and the header receives the
    ``<refsDecl xml:id="biblical">`` and the annotation-status declarations.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date

from lxml import etree

import greek

TEI_NS = "http://www.tei-c.org/ns/1.0"
T = "{%s}" % TEI_NS
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_MARK = "{urn:scripture-detector}m"
_OLD = "{urn:scripture-detector}old"

ID_PREFIX = "sd-"
REF_PREFIX = "sd-r"      # xml:id prefix of <ref> elements we add to existing <quote>s

BLOCK = {
    "p", "ab", "l", "lg", "div", "head", "item", "list", "sp", "speaker", "table", "row",
    "cell", "titlePart", "trailer", "opener", "closer", "argument", "front", "back", "body",
    "floatingText", "stage", "byline", "dateline", "salute", "signed",
}
BREAKS = {"lb", "pb", "cb", "gb"}
SKIP = {"note", "rdg", "witDetail", "del", "surplus", "figDesc", "teiHeader"}

QUOTE_TYPES = {"full": "quote", "partial": "quote", "paraphrase": "quote", "allusion": "seg"}


class TeiError(ValueError):
    pass


class TeiTextMismatch(TeiError):
    pass


def _ln(el) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


# ── plain-text extraction with offset map ────────────────────────────────────

class TextMap:
    def __init__(self):
        self.text = ""
        self.runs: list[list[int]] = []         # [start, end, node_idx, node_off]; node_idx -1 = virtual
        self.nodes: list[tuple[etree._Element, str]] = []
        self.spans: dict[etree._Element, tuple[int, int]] = {}
        self._starts: list[int] = []

    # lookups -------------------------------------------------------------
    def _run_at(self, pos: int) -> int:
        return bisect_right(self._starts, pos) - 1

    def locate_start(self, pos: int):
        """(node_idx, node_off) of the first real character at or after pos."""
        i = self._run_at(pos)
        while i < len(self.runs):
            s, e, n, off = self.runs[i]
            p = max(pos, s)
            if n >= 0 and p < e:
                return n, off + (p - s)
            i += 1
        return None

    def locate_end(self, pos: int):
        """(node_idx, node_off) just after the last real character before pos."""
        i = self._run_at(pos - 1)
        while i >= 0:
            s, e, n, off = self.runs[i]
            p = min(pos, e)
            if n >= 0 and p > s:
                return n, off + (p - s)
            i -= 1
        return None


class _Builder:
    def __init__(self):
        self.chars: list[str] = []
        self.tm = TextMap()
        self.node_ids: dict[tuple[int, str], int] = {}
        self.last_ws = True
        self.nobreak = False

    def node(self, el, kind: str) -> int:
        key = (id(el), kind)
        if key not in self.node_ids:
            self.node_ids[key] = len(self.tm.nodes)
            self.tm.nodes.append((el, kind))
        return self.node_ids[key]

    @property
    def pos(self) -> int:
        return len(self.chars)

    def _emit(self, ch: str, node_idx: int, node_off: int):
        runs = self.tm.runs
        if runs and node_idx >= 0 and runs[-1][2] == node_idx and runs[-1][3] + runs[-1][1] - runs[-1][0] == node_off:
            runs[-1][1] += 1
        else:
            runs.append([self.pos, self.pos + 1, node_idx, node_off])
        self.chars.append(ch)
        self.last_ws = ch in " \n"

    def _pop(self):
        self.chars.pop()
        run = self.tm.runs[-1]
        run[1] -= 1
        if run[1] == run[0]:
            self.tm.runs.pop()
        self.last_ws = (not self.chars) or self.chars[-1] in " \n"

    def add_text(self, s: str, node_idx: int):
        for i, c in enumerate(s):
            if c.isspace():
                if self.nobreak or self.last_ws:
                    continue
                self._emit(" ", node_idx, i)
            else:
                self.nobreak = False
                self._emit(c, node_idx, i)

    def soft(self):
        if not self.last_ws:
            self._emit(" ", -1, 0)

    def hard(self):
        if self.chars and self.chars[-1] == " ":
            self._pop()
        if self.chars and self.chars[-1] != "\n":
            self._emit("\n", -1, 0)

    def join(self):
        if self.chars and self.chars[-1] == " ":
            self._pop()
        self.nobreak = True


def _pick_choice(children):
    """Prefer the normalised/expanded reading inside <choice>."""
    for want in ("expan", "reg", "corr"):
        for c in children:
            if _ln(c) == want:
                return c
    return next((c for c in children if isinstance(c.tag, str)), None)


def _walk(el, b: _Builder):
    name = _ln(el)
    start = b.pos
    if name in BLOCK:
        b.hard()
    if name in BREAKS:
        b.join() if el.get("break") == "no" else b.soft()
    if el.text:
        b.add_text(el.text, b.node(el, "text"))
    children = list(el)
    if name == "choice":
        pick = _pick_choice(children)
        children = [pick] if pick is not None else []
    for ch in children:
        if isinstance(ch.tag, str):
            cname = _ln(ch)
            skip = cname in SKIP or (cname == "div" and ch.get("type") == "praefatio")
            if not skip:
                _walk(ch, b)
        if ch.tail:
            b.add_text(ch.tail, b.node(ch, "tail"))
    if name in BLOCK:
        b.hard()
    b.tm.spans[el] = (start, b.pos)


def extract_text(root) -> TextMap:
    text_el = root.find(T + "text")
    if text_el is None:
        raise TeiError("No <text> element found - is this a TEI document?")
    b = _Builder()
    _walk(text_el, b)
    while b.chars and b.chars[-1] in " \n":
        b._pop()
    tm = b.tm
    tm.text = "".join(b.chars)
    tm._starts = [r[0] for r in tm.runs]
    return tm


# ── parsing helpers ──────────────────────────────────────────────────────────

def _parse(xml_bytes: bytes):
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)
    try:
        root = etree.fromstring(xml_bytes, parser)
    except etree.XMLSyntaxError as exc:
        raise TeiError(f"Not well-formed XML: {exc}") from exc
    if root.tag != T + "TEI":
        raise TeiError("Root element is not <TEI> in the TEI namespace")
    return root


def _title(root, fallback: str) -> str:
    t = root.find(f"{T}teiHeader/{T}fileDesc/{T}titleStmt/{T}title")
    a = root.find(f"{T}teiHeader/{T}fileDesc/{T}titleStmt/{T}author")
    title = re.sub(r"\s+", " ", "".join(t.itertext())).strip() if t is not None else ""
    author = re.sub(r"\s+", " ", "".join(a.itertext())).strip() if a is not None else ""
    if title and author:
        return f"{author}: {title}"
    return title or fallback


def _is_biblical_ref(r) -> bool:
    return _ln(r) == "ref" and (r.get("decls") or "").lstrip("#") == "biblical" and bool(r.get("cRef"))


def _sd_elements(root):
    return [e for e in root.iter(T + "quote", T + "seg")
            if (e.get(XML_ID) or "").startswith(ID_PREFIX)]


def _frag_base(xml_id: str) -> str:
    return re.sub(r"-\d+$", "", xml_id)


@dataclass
class TeiData:
    name: str
    text: str
    lang: str = ""
    annotations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _own(el) -> bool:
    return (el.get(XML_ID) or "").startswith(ID_PREFIX)


def _is_own_ref(r) -> bool:
    return (r.get(XML_ID) or "").startswith(REF_PREFIX)


def _first_lang(root) -> str:
    """xml:lang of the first <div> in <text> (falling back to its ancestors,
    then to any later <div> that has one)."""
    text_el = root.find(T + "text")
    if text_el is None:
        return ""
    divs = list(text_el.iter(T + "div"))
    if divs:
        for el in [divs[0], *divs[0].iterancestors()]:
            if el.get(XML_LANG):
                return el.get(XML_LANG)
        for d in divs[1:]:
            if d.get(XML_LANG):
                return d.get(XML_LANG)
    return text_el.get(XML_LANG) or root.get(XML_LANG) or ""


def _trim(tm: TextMap, s: int, e: int) -> tuple[int, int]:
    e = min(e, len(tm.text))
    while s < e and tm.text[s] in " \n":
        s += 1
    while e > s and tm.text[e - 1] in " \n":
        e -= 1
    return s, e


def _foreign_quote_chains(root, tm: TextMap) -> list[dict]:
    """Existing (hand-made) <quote> elements; fragments linked by @prev/@next
    form one chain.  ``biblical`` is True if a hand-made biblical <ref> exists."""
    quotes = [q for q in root.iter(T + "quote") if not _own(q) and q in tm.spans]
    by_id = {q.get(XML_ID): q for q in quotes if q.get(XML_ID)}

    def link(q, attr):
        v = (q.get(attr) or "").split()
        return by_id.get(v[0].lstrip("#")) if v else None

    seen: set = set()
    chains = []
    for q in quotes:
        if q in seen:
            continue
        head = q
        for _ in range(1000):
            p = link(head, "prev")
            if p is None or p is head:
                break
            head = p
        chain, cur = [], head
        while cur is not None and cur not in seen and len(chain) < 1000:
            chain.append(cur)
            seen.add(cur)
            cur = link(cur, "next")
        if q not in chain:
            chain.append(q)
            seen.add(q)
        chains.append(chain)

    out = []
    for ch in chains:
        spans = [tm.spans[e] for e in ch]
        s, e = _trim(tm, min(a for a, _ in spans), max(b for _, b in spans))
        refs = [c for el in ch for c in el if _is_biblical_ref(c)]
        out.append({"els": ch, "start": s, "end": e,
                    "biblical": any(not _is_own_ref(r) for r in refs),
                    "own_refs": [r for r in refs if _is_own_ref(r)]})
    return out


def existing_quotes(xml_bytes: bytes) -> list[dict]:
    """Passages already marked as <quote> that carry no biblical reference yet
    (start, end, text) - used as hints for the detection prompt."""
    root = _parse(xml_bytes)
    _strip_own(root)
    tm = extract_text(root)
    return [{"start": c["start"], "end": c["end"], "text": tm.text[c["start"]:c["end"]]}
            for c in _foreign_quote_chains(root, tm) if not c["biblical"] and c["end"] > c["start"]]


def _annotation(tm, s, e, crefs, qtype, marked, abbr_to_code, origin="tool"):
    if s >= e or not crefs:
        return None
    refs: list[str] = []
    if abbr_to_code is not None:
        for c in crefs:
            refs.extend(greek.parse_cref(c, abbr_to_code))
        if not refs and origin != "source":
            return None
    return {"span_start": s, "span_end": e, "quote_text": tm.text[s:e], "quote_type": qtype,
            "crefs": crefs, "refs": refs, "marked": marked, "origin": origin}


def read_tei(xml_bytes: bytes, abbr_to_code: dict | None = None,
             fallback_name: str = "TEI document") -> TeiData:
    """Extract plain text, the language (xml:lang of the first <div>) and the
    annotations written earlier by this tool: elements with xml:id ``sd-…`` and
    ``<ref xml:id="sd-r…">`` inside existing <quote>s.  Each annotation carries
    its ``crefs``; ``refs`` (internal codes) only if *abbr_to_code* is given."""
    root = _parse(xml_bytes)
    tm = extract_text(root)
    data = TeiData(name=_title(root, fallback_name), text=tm.text, lang=_first_lang(root))

    groups: dict[str, list] = {}
    for el in _sd_elements(root):
        groups.setdefault(_frag_base(el.get(XML_ID)), []).append(el)
    for base, els in groups.items():
        spans = [tm.spans[e] for e in els if e in tm.spans]
        if not spans:
            continue
        s, e = _trim(tm, min(a for a, _ in spans), max(b for _, b in spans))
        crefs = [r.get("cRef") for el in els for r in el if _is_biblical_ref(r)]
        first = els[0]
        typ = first.get("type")
        qtype = "allusion" if _ln(first) == "seg" else ("paraphrase" if typ == "paraphrasis" else "full")
        ann = _annotation(tm, s, e, crefs, qtype, {"marked": True, "unmarked": False}.get(typ), abbr_to_code)
        if ann:
            data.annotations.append(ann)

    for ch in _foreign_quote_chains(root, tm):
        if not ch["own_refs"] or ch["biblical"]:
            continue
        typ = ch["els"][0].get("type")
        ann = _annotation(tm, ch["start"], ch["end"], [r.get("cRef") for r in ch["own_refs"]],
                          "paraphrase" if typ == "paraphrasis" else "full",
                          {"marked": True, "unmarked": False}.get(typ), abbr_to_code)
        if ann:
            data.annotations.append(ann)

    # Biblical markup that is already in the source file (made by hand, e.g. in a PTA edition):
    # shown in the viewer, never changed by the export.  origin = "source".
    for ch in _foreign_quote_chains(root, tm):
        human = [r.get("cRef") for el in ch["els"] for r in el if _is_biblical_ref(r) and not _is_own_ref(r)]
        typ = ch["els"][0].get("type")
        ann = _annotation(tm, ch["start"], ch["end"], human, "paraphrase" if typ == "paraphrasis" else "full",
                          {"marked": True, "unmarked": False}.get(typ), abbr_to_code, origin="source")
        if ann:
            data.annotations.append(ann)
    for el in root.iter(T + "seg"):
        if _own(el) or el not in tm.spans or el.get("type") != "allusion":
            continue
        human = [r.get("cRef") for r in el if _is_biblical_ref(r) and not _is_own_ref(r)]
        s0, e0 = _trim(tm, *tm.spans[el])
        ann = _annotation(tm, s0, e0, human, "allusion", None, abbr_to_code, origin="source")
        if ann:
            data.annotations.append(ann)

    data.annotations.sort(key=lambda a: a["span_start"])
    return data


# ── export ───────────────────────────────────────────────────────────────────

_REFSDECL_PATTERNS = [
    (r"(.+):(.+):([0-9]+):([0-9]+)[\-\.]*[0-9]*", "https://www.bibelwissenschaft.de/bibelstelle/$2$3,$4/$1",
     "Noted are references to a biblical book, chapter, verse, sc. LXX:Gn:1:1; also possible: LXX:Gn:1:1-3 "
     "for a sequence of verses or LXX:Gn:1:1.3 for two verses not following each other."),
    (r"(.+):(.+):([0-9]+)[\-\.]*[0-9]*", "https://www.bibelwissenschaft.de/bibelstelle/$2$3/$1",
     "Noted are references to a biblical book and chapter, sc. LXX:Gn:1; also possible: LXX:Gn:1-3 for a "
     "sequence of chapters or LXX:Gn:1.3 for two chapters not following each other."),
    (r"(.+):(.+)", "https://www.bibelwissenschaft.de/bibelstelle/$2/$1",
     "Noted are references to a biblical book, sc. LXX:Gn."),
]

_ABBR_LIST = (
    "Gn, Ex, Lv, Num, Dt, Jos, Judg, Rt, 1Sa, 2Sa, 1Ko, 2Ko, 1Chr, 2Chr, 3Esr, Esr, Est, Jdt, Tob, 1Mak, "
    "2Mak, 3Mak, 4Mak, Ps, Oden, Prov, Eccl, Song, Job, Wis, Sir, PsSal, Hos, Am, Mi, Joel, Ob, Jon, Nah, "
    "Hab, Zeph, Hag, Sach, Mal, Is, Jr, Bar, Lam, EpistJer, Hes, Sus (Sus-LXX), Dn (Dn-LXX), Bel (Bel-LXX); "
    "Mt, Mk, Lk, Jn, Act, Rom, 1Cor, 2Cor, Gal, Eph, Phil, Col, 1Th, 2Th, 1Tim, 2Tim, Tt, Phm, Heb, Jak, "
    "1P, 2P, 1Jn, 2Jn, 3Jn, Jud, Rev; Neh (Vulgate, Peshitta); PrMan, 4Esr, 2Bar (Peshitta only)."
)

_STATUS_CATEGORIES = [
    ("fully-annotated", "All instances of a phenomenon (e.g. biblical quotations, persons, places) are encoded with the respective element."),
    ("partially-annotated", "Some instances of a phenomenon (e.g. biblical quotations, persons, places) are encoded with the respective element."),
    ("fully-identified", "All encoded instances of a phenomenon (e.g. biblical quotations, persons, places) are identified with the respective norm data."),
    ("partially-identified", "Some encoded instances of a phenomenon (e.g. biblical quotations, persons, places) are identified with the respective norm data."),
    ("not-identified", "No encoded instances of a phenomenon (e.g. biblical quotations, persons, places) are identified with the respective norm data."),
]


def _sub(parent, tag, text=None, **attrs):
    el = etree.SubElement(parent, T + tag)
    for k, v in attrs.items():
        el.set(XML_ID if k == "xml_id" else k, v)
    if text:
        el.text = text
    return el


def _child(parent, name):
    return parent.find(T + name)


def _ensure_header(root):
    header = root.find(T + "teiHeader")
    if header is None:
        raise TeiError("The document has no <teiHeader>")
    enc = _child(header, "encodingDesc")
    if enc is None:
        enc = etree.Element(T + "encodingDesc")
        rev = _child(header, "revisionDesc")
        (rev.addprevious(enc) if rev is not None else header.append(enc))
    return header, enc


def _insert_in_enc(enc, el, after_names=()):
    """Insert *el* into encodingDesc, keeping <variantEncoding> last and
    <schemaRef> after the declarations."""
    last_after = None
    for name in after_names:
        found = enc.findall(T + name)
        if found:
            last_after = found[-1]
    if last_after is not None:
        last_after.addnext(el)
        return
    for name in ("variantEncoding", "schemaRef"):
        anchor = _child(enc, name)
        if anchor is not None:
            anchor.addprevious(el)
            return
    enc.append(el)


def _declare_in_header(root, n_annotations: int, note: str) -> None:
    header, enc = _ensure_header(root)

    # refsDecl xml:id="biblical"
    if not any(r.get(XML_ID) == "biblical" for r in enc.findall(T + "refsDecl")):
        rd = etree.Element(T + "refsDecl")
        rd.set(XML_ID, "biblical")
        for i, (mp, rp, desc) in enumerate(_REFSDECL_PATTERNS):
            c = _sub(rd, "cRefPattern", matchPattern=mp, replacementPattern=rp)
            if i == 0:
                _sub(c, "p", "For references to the Septuagint LXX is used as abbreviation, for references "
                             "to the New Testament NA is used as abbreviation for the version, for "
                             "references to the Latin Vulgate Vg, for references to the Syriac "
                             "Peshitta Peshitta.")
                _sub(c, "p", desc)
                _sub(c, "p", "The following abbreviations are used: " + _ABBR_LIST)
            else:
                _sub(c, "p", desc)
        _insert_in_enc(enc, rd, after_names=("refsDecl",))

    # editorialDecl/interpretation/p[@xml:id='biblical-quotations']
    ed = _child(enc, "editorialDecl")
    if ed is None:
        ed = etree.Element(T + "editorialDecl")
        _insert_in_enc(enc, ed, after_names=("refsDecl", "classDecl"))
    interp = _child(ed, "interpretation")
    if interp is None:
        interp = etree.Element(T + "interpretation")
        ed.insert(0, interp)
    if not any(p.get(XML_ID) == "biblical-quotations" for p in interp.findall(T + "p")):
        p = _sub(interp, "p", None, xml_id="biblical-quotations", ana="partially-annotated partially-identified")
        p.text = note + " References follow the scheme declared "
        r = _sub(p, "ref", "above", target="#biblical")
        r.tail = "."

    # classDecl/taxonomy[@xml:id='annotationstatus']
    cd = _child(enc, "classDecl")
    if cd is None:
        cd = etree.Element(T + "classDecl")
        _insert_in_enc(enc, cd, after_names=("refsDecl",))
    if not any(t.get(XML_ID) == "annotationstatus" for t in cd.findall(T + "taxonomy")):
        tax = _sub(cd, "taxonomy", xml_id="annotationstatus")
        _sub(tax, "bibl", "Patristic Text Archive Status of annotation")
        for cid, desc in _STATUS_CATEGORIES:
            cat = _sub(tax, "category", xml_id=cid)
            _sub(cat, "catDesc", desc)

    # revisionDesc/change
    rev = _child(header, "revisionDesc")
    if rev is None:
        rev = etree.SubElement(header, T + "revisionDesc")
    _sub(rev, "change", f"Biblical quotations and allusions annotated automatically with Scripture Detector "
                        f"({n_annotations} annotations, not manually verified).", when=date.today().isoformat())


def _remove_keep_tail(el):
    parent = el.getparent()
    prev = el.getprevious()
    tail = el.tail or ""
    if prev is not None:
        prev.tail = (prev.tail or "") + tail
    else:
        parent.text = (parent.text or "") + tail
    parent.remove(el)


def _strip_own(root):
    for r in [r for r in root.iter(T + "ref") if _is_own_ref(r)]:
        _remove_keep_tail(r)             # refs we added to existing <quote>s
    own = _sd_elements(root)
    for el in own:
        for r in [c for c in el if _is_biblical_ref(c)]:
            _remove_keep_tail(r)
        el.tag = _OLD
    etree.strip_tags(root, _OLD)
    return len(own)


# -- locating a passage for messages ------------------------------------------

_NS = {"t": TEI_NS}


def _where(tm: TextMap, pos: int) -> str:
    """Human-readable location of a text position: character offset, section
    (div/@n), paragraph, page (pb/@n), line (lb/@n) and nearest xml:id."""
    bits = [f"char {pos}"]
    loc = tm.locate_start(pos)
    if loc is None:
        return bits[0]
    el, _ = tm.nodes[loc[0]]
    chain = [el, *el.iterancestors()]
    sections = [a.get("n") for a in chain
                if _ln(a) == "div" and a.get("n") and not a.get("n").startswith("urn:")]
    if sections:
        bits.append("section " + ".".join(reversed(sections)))
    for a in chain:
        name = _ln(a)
        if name in ("p", "l", "ab", "head", "item"):
            idx = 1 + sum(1 for sib in a.itersiblings(preceding=True) if _ln(sib) == name)
            bits.append(f"<{name}> {idx}")
            break
    for tag, label in (("pb", "page"), ("lb", "line")):
        n = el.xpath(f"preceding::t:{tag}[1]/@n", namespaces=_NS)
        if n:
            bits.append(f"{label} {n[0]}")
    xid = next((a.get(XML_ID) for a in chain
                if a.get(XML_ID) and not a.get(XML_ID).startswith(ID_PREFIX)), None)
    if xid:
        bits.append(f'xml:id "{xid}"')
    return ", ".join(bits)


def _passage(tm: TextMap, s: int, e: int, width: int = 25) -> str:
    """The passage with a little context and its location:
    ``…before «passage» after… [char 12, section 3, <p> 2]``."""
    flat = lambda t: t.replace("\n", " ")
    mid = flat(tm.text[s:e])
    if len(mid) > 80:
        mid = mid[:50] + " … " + mid[-25:]
    pre = flat(tm.text[max(0, s - width):s]).lstrip()
    post = flat(tm.text[e:e + width]).rstrip()
    return f"…{pre}«{mid}»{post}… [{_where(tm, s)}]"


# -- DOM range wrapping -------------------------------------------------------

def _has_text(el_text, sibs) -> bool:
    if (el_text or "").strip():
        return True
    for s in sibs:
        if isinstance(s.tag, str) and "".join(s.itertext()).strip():
            return True
        if (s.tail or "").strip():          # also covers comments / processing instructions
            return True
    return False


def _frag_after(parent, x, tag):
    sibs = list(x.itersiblings())
    if not _has_text(x.tail, sibs):
        return None
    new = etree.Element(tag)
    new.text, x.tail = x.tail, None
    for s in sibs:
        new.append(s)
    parent.insert(parent.index(x) + 1, new)
    return new


def _frag_before(parent, x, tag):
    sibs = list(reversed(list(x.itersiblings(preceding=True))))
    if not _has_text(parent.text, sibs):
        return None
    new = etree.Element(tag)
    new.text, parent.text = parent.text, None
    for s in sibs:
        new.append(s)
    parent.insert(0, new)
    return new


def _frag_between(parent, a, b, tag):
    sibs = []
    for s in a.itersiblings():
        if s is b:
            break
        sibs.append(s)
    if not _has_text(a.tail, sibs):
        return None
    new = etree.Element(tag)
    new.text, a.tail = a.tail, None
    for s in sibs:
        new.append(s)
    parent.insert(parent.index(a) + 1, new)
    return new


def _wrap_between_markers(S, E, tag, warnings: list[str], where=lambda: ""):
    anc_s = [S] + list(S.iterancestors())
    anc_e = [E] + list(E.iterancestors())
    set_e = set(anc_e[1:])
    ci = next(i for i, a in enumerate(anc_s) if i > 0 and a in set_e)
    C = anc_s[ci]
    path_s = anc_s[: ci + 1]
    path_e = anc_e[: anc_e.index(C) + 1]

    left = []
    for i in range(len(path_s) - 2):
        f = _frag_after(path_s[i + 1], path_s[i], tag)
        if f is not None:
            left.append(f)

    s_top, e_top = path_s[-2], path_e[-2]
    between = []
    for s in s_top.itersiblings():
        if s is e_top:
            break
        between.append(s)
    if any(_ln(s) in BLOCK for s in between):
        warnings.append("A quotation spans whole structural blocks; the blocks in between were not wrapped: "
                        + where())
        mid = None
    else:
        mid = _frag_between(C, s_top, e_top, tag)

    right = []
    for i in range(len(path_e) - 3, -1, -1):
        f = _frag_before(path_e[i + 1], path_e[i], tag)
        if f is not None:
            right.append(f)

    return left + ([mid] if mid is not None else []) + right


# -- main entry ---------------------------------------------------------------

def annotate_tei(
    xml_bytes: bytes,
    annotations: list[dict],
    book_info: dict[str, dict],
    expected_text: str | None = None,
) -> tuple[bytes, list[str]]:
    """Write *annotations* into the TEI document and return (xml, warnings).

    Each annotation: id, span_start, span_end, quote_type, refs, marked.
    Spans refer to the plain text produced by ``extract_text``/``read_tei``.
    Elements written by an earlier export (xml:id ``sd-…``) are replaced;
    biblical markup that was made by hand is left untouched.  Passages that are
    already marked up as <quote> get only a <ref> (and @type if missing); no
    further quote element is created for them.
    """
    warnings: list[str] = []
    root = _parse(xml_bytes)
    _strip_own(root)
    tm = extract_text(root)
    if expected_text is not None and tm.text != expected_text:
        raise TeiTextMismatch("The text of the stored TEI file no longer matches the annotated text.")

    for ed in root.iter(T + "div"):
        if ed.get("type") == "edition" and re.search(r"-Ms[A-Za-z0-9]+$", ed.get("n") or ""):
            warnings.append("This looks like a manuscript transcription. The PTA guidelines do not use <quote> "
                            "in transcriptions (use <g type=\"diple\"/>); check before publishing.")
            break

    # hand-made markup: existing <quote>s are enriched instead of duplicated
    chains = _foreign_quote_chains(root, tm)
    foreign_segs = []
    for el in root.iter(T + "seg"):
        refs = [c.get("cRef") for c in el if _is_biblical_ref(c)]
        if refs and not _own(el) and el in tm.spans:
            foreign_segs.append((tm.spans[el], set(refs)))

    prepared = []
    matched: dict[int, dict] = {}
    used_ids = {e.get(XML_ID) for e in root.iter() if isinstance(e.tag, str) and e.get(XML_ID)}
    for a in annotations:
        s, e = a.get("span_start"), a.get("span_end")
        if s is None or e is None:
            continue
        s, e = _trim(tm, max(0, s), min(len(tm.text), e))
        crefs = greek.format_crefs(a.get("refs", []), book_info)
        if s >= e or not crefs:
            continue
        label = lambda s=s, e=e: _passage(tm, s, e)      # only built when a warning is needed
        if any(fs < e and s < fe and (set(crefs) & frefs) for (fs, fe), frefs in foreign_segs):
            warnings.append(f"Skipped, already annotated in the file: {label()}")
            continue
        best, best_ov = None, 0
        for ci, ch in enumerate(chains):
            ov = min(e, ch["end"]) - max(s, ch["start"])
            if ov > best_ov:
                best, best_ov = ci, ov
        if best is not None:
            ch = chains[best]
            if ch["biblical"]:
                warnings.append(f"Skipped, the existing <quote> already has a biblical reference: {label()}")
            elif best_ov >= 0.5 * (e - s) and best_ov >= 0.5 * (ch["end"] - ch["start"]):
                m = matched.setdefault(best, {"anns": [], "refs": []})
                m["anns"].append(a)
                m["refs"].extend(r for r in a.get("refs", []) if r not in m["refs"])
            else:
                warnings.append("Skipped, it overlaps an existing <quote> only in part "
                                f"(no further quote element is added): {label()}")
            continue
        st, en = tm.locate_start(s), tm.locate_end(e)
        if st is None or en is None:
            continue
        prepared.append({"a": a, "s": s, "e": e, "crefs": crefs, "st": st, "en": en})
    prepared.sort(key=lambda p: (p["s"], -p["e"]))

    # 1) place milestone markers at the span boundaries
    per_node: dict[int, dict[int, list]] = {}
    for n, p in enumerate(prepared):
        p["S"], p["E"] = etree.Element(_MARK), etree.Element(_MARK)
        per_node.setdefault(p["st"][0], {}).setdefault(p["st"][1], []).append(("S", p))
        per_node.setdefault(p["en"][0], {}).setdefault(p["en"][1], []).append(("E", p))
    for node_idx, by_off in per_node.items():
        el, kind = tm.nodes[node_idx]
        for off in sorted(by_off, reverse=True):
            items = by_off[off]
            ends = sorted([p for k, p in items if k == "E"], key=lambda p: -p["s"])
            starts = sorted([p for k, p in items if k == "S"], key=lambda p: -p["e"])
            markers = [p["E"] for p in ends] + [p["S"] for p in starts]
            cur = (el.text if kind == "text" else el.tail) or ""
            head, rest = cur[:off], cur[off:]
            if kind == "text":
                el.text = head
                for k, m in enumerate(markers):
                    m.tail = rest if k == len(markers) - 1 else None
                    el.insert(k, m)
            else:
                el.tail = head
                parent = el.getparent()
                idx = parent.index(el) + 1
                for k, m in enumerate(markers):
                    m.tail = rest if k == len(markers) - 1 else None
                    parent.insert(idx + k, m)

    # 2) wrap the content between the marker pairs
    counter = 0
    written = 0
    for p in prepared:
        a = p["a"]
        qtype = a.get("quote_type", "allusion")
        tag = T + QUOTE_TYPES.get(qtype, "seg")
        frags = _wrap_between_markers(p["S"], p["E"], tag, warnings,
                                      lambda p=p: _passage(tm, p["s"], p["e"]))
        if not frags:
            warnings.append("Could not annotate this passage (it contains no markable text): "
                            + _passage(tm, p["s"], p["e"]))
            continue
        counter += 1
        base = f"{ID_PREFIX}q{a.get('id') or counter}"
        while base in used_ids:
            base += "x"
        used_ids.add(base)
        for k, f in enumerate(frags, 1):
            if _ln(f) == "quote":
                if qtype == "paraphrase":
                    f.set("type", "paraphrasis")
                elif a.get("marked") is True:
                    f.set("type", "marked")
                elif a.get("marked") is False:
                    f.set("type", "unmarked")
            else:
                f.set("type", "allusion")
            fid = base if len(frags) == 1 else f"{base}-{k}"
            f.set(XML_ID, fid)
            if len(frags) > 1:
                if k > 1:
                    f.set("prev", f"#{base}-{k - 1}")
                if k < len(frags):
                    f.set("next", f"#{base}-{k + 1}")
        for cref in p["crefs"]:
            r = etree.SubElement(frags[-1], T + "ref")
            r.set("decls", "#biblical")
            r.set("cRef", cref)
        written += 1

    etree.strip_tags(root, _MARK)

    # 2b) existing <quote>s: add the reference (and @type if missing), no new element
    rn = 0
    for ci, m in matched.items():
        els = chains[ci]["els"]
        first = m["anns"][0]
        typ = ("paraphrasis" if first.get("quote_type") == "paraphrase"
               else {True: "marked", False: "unmarked"}.get(first.get("marked")))
        if typ:
            for el in els:
                if el.get("type") is None:
                    el.set("type", typ)
        # merge the verses of all annotations first, so that consecutive verses
        # become ONE ref (phil_2:8 + phil_2:9-10 -> NA:Phil:2:8-10)
        for cref in greek.format_crefs(m["refs"], book_info):
            rn += 1
            rid = f"{REF_PREFIX}{rn}"
            while rid in used_ids:
                rid += "x"
            used_ids.add(rid)
            r = etree.SubElement(els[-1], T + "ref")
            r.set("decls", "#biblical")
            r.set("cRef", cref)
            r.set(XML_ID, rid)
        written += len(m["anns"])

    # 3) header declarations
    if written or _sd_elements(root):
        _declare_in_header(
            root, written,
            "Biblical quotations are partially annotated (automatic detection) and partially referenced.")
    return etree.tostring(root.getroottree(), xml_declaration=True, encoding="UTF-8"), warnings


# ── plain text -> PTA skeleton ───────────────────────────────────────────────

def plain_to_tei(name: str, text: str, lang: str = "grc") -> tuple[bytes, list[int]]:
    """Wrap pasted plain text into a minimal PTA-style TEI document.

    Returns (xml, offset_map) where ``offset_map[i]`` is the position in the
    text extracted from the new document that corresponds to ``text[i]``
    (``len(text)`` maps to the end).
    """
    norm: list[str] = []
    omap: list[int] = [0] * (len(text) + 1)
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if norm:
                sep = "\n" if "\n" in text[i:j] else " "
                norm.append(sep)
                for k in range(i, j):
                    omap[k] = len(norm) - 1
            else:
                for k in range(i, j):
                    omap[k] = 0
            i = j
        else:
            norm.append(text[i])
            omap[i] = len(norm) - 1
            i += 1
    omap[n] = len(norm)
    while norm and norm[-1] in " \n":
        norm.pop()
    paragraphs = "".join(norm).split("\n")

    tei = etree.Element(T + "TEI", nsmap={None: TEI_NS})
    header = etree.SubElement(tei, T + "teiHeader")
    fd = etree.SubElement(header, T + "fileDesc")
    ts = etree.SubElement(fd, T + "titleStmt")
    _sub(ts, "title", name)
    ps = etree.SubElement(fd, T + "publicationStmt")
    _sub(ps, "p", "Created with Scripture Detector from plain text.")
    sd = etree.SubElement(fd, T + "sourceDesc")
    _sub(sd, "p", "Plain text import.")
    etree.SubElement(header, T + "encodingDesc")
    body = etree.SubElement(etree.SubElement(tei, T + "text"), T + "body")
    ed = _sub(body, "div", type="edition", n="urn:cts:pta:pta0000.pta000.pta-grc1")
    ed.set(XML_LANG, lang)
    ed.insert(0, etree.Comment(" TODO: replace the placeholder CTS URN in @n "))
    part = _sub(ed, "div", type="textpart", subtype="section", n="1")
    for para in paragraphs:
        if para.strip():
            _sub(part, "p", para)
    xml = etree.tostring(tei, xml_declaration=True, encoding="UTF-8")
    return xml, omap
