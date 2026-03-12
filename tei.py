"""TEI-XML export and import for Scripture Detector.

Export schema
─────────────
  TEI
  ├── teiHeader / fileDesc, encodingDesc
  ├── text / body / ab   ← source text with inline <seg> for annotated spans
  └── standOff / listAnnotation   ← one <annotation> per quote

Import
──────
  Reads a TEI file produced by this module and reconstructs source name,
  full text, and annotations (with character-offset spans).
"""

from __future__ import annotations

import re
from datetime import date

from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"

_T  = f"{{{TEI_NS}}}"   # prefix shortcut
_X  = f"{{{XML_NS}}}"   # xml: namespace prefix


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_segments(text: str, annotations: list[dict]) -> list[dict]:
    """Split *text* at annotation boundaries (same logic as app.compute_segments)."""
    boundaries: set[int] = {0, len(text)}
    for a in annotations:
        if a.get("span_start") is not None and a.get("span_end") is not None:
            boundaries.add(a["span_start"])
            boundaries.add(a["span_end"])
    ordered = sorted(boundaries)
    segments = []
    for i in range(len(ordered) - 1):
        start, end = ordered[i], ordered[i + 1]
        ann_ids = [
            j for j, a in enumerate(annotations)
            if a.get("span_start") is not None
            and a["span_start"] <= start and end <= a["span_end"]
        ]
        segments.append({"text": text[start:end], "start": start, "end": end,
                          "annotation_ids": ann_ids})
    return segments


def _ref_label(ref: str, book_names: dict[str, str]) -> str:
    """'gen_1:5'  →  'Genesis 1:5'"""
    ref = ref.strip().lower()
    m = re.match(r"^([a-z0-9]+)_(\d+):(\d+)$", ref)
    if m:
        book_code, ch, vs = m.groups()
        book = book_names.get(book_code, book_code.capitalize())
        return f"{book} {ch}:{vs}"
    m2 = re.match(r"^([a-z0-9]+)_(\d+)$", ref)
    if m2:
        book_code, ch = m2.groups()
        book = book_names.get(book_code, book_code.capitalize())
        return f"{book} {ch}"
    return ref


# ── export ────────────────────────────────────────────────────────────────────

def source_to_tei(
    source: dict,
    annotations: list[dict],
    book_names: dict[str, str] | None = None,
) -> bytes:
    """
    Serialise *source* + *annotations* as UTF-8 TEI XML bytes.

    source:      dict with keys id, name, text, created_at
    annotations: list of dicts with keys id, span_start, span_end,
                 quote_text, quote_type, refs
    book_names:  {book_code: human_name} — used for human-readable <ref> labels
    """
    book_names = book_names or {}

    NSMAP = {None: TEI_NS}
    root = etree.Element(f"{_T}TEI", nsmap=NSMAP)

    # ── teiHeader ────────────────────────────────────────────────────────────
    header     = etree.SubElement(root,   f"{_T}teiHeader")
    fileDesc   = etree.SubElement(header, f"{_T}fileDesc")
    titleStmt  = etree.SubElement(fileDesc, f"{_T}titleStmt")
    title_el   = etree.SubElement(titleStmt, f"{_T}title")
    title_el.text = source["name"]
    resp       = etree.SubElement(titleStmt, f"{_T}respStmt")
    resp_resp  = etree.SubElement(resp, f"{_T}resp")
    resp_resp.text = "Analyzed by"
    resp_name  = etree.SubElement(resp, f"{_T}name")
    resp_name.text = "Scripture Detector (Dr. William J.B. Mattingly, Yale University)"
    pubStmt    = etree.SubElement(fileDesc, f"{_T}publicationStmt")
    pub_p      = etree.SubElement(pubStmt, f"{_T}p")
    pub_p.text = (
        f"Exported from Scripture Detector on {date.today().isoformat()}. "
        "Scripture Detector is developed by Dr. William J.B. Mattingly, "
        "Cultural Heritage Data Scientist, Yale University."
    )
    srcDesc    = etree.SubElement(fileDesc, f"{_T}sourceDesc")
    src_p      = etree.SubElement(srcDesc,  f"{_T}p")
    src_p.text = "AI-assisted detection of biblical quotations, paraphrases, and allusions."

    encDesc    = etree.SubElement(header, f"{_T}encodingDesc")
    projDesc   = etree.SubElement(encDesc, f"{_T}projectDesc")
    proj_p     = etree.SubElement(projDesc, f"{_T}p")
    proj_p.text = (
        "Scripture Detector uses Google Gemini to identify and classify biblical "
        "references in historical texts. Reference types follow a four-level taxonomy."
    )
    clasDecl   = etree.SubElement(encDesc, f"{_T}classDecl")
    taxonomy   = etree.SubElement(clasDecl, f"{_T}taxonomy")
    taxonomy.set(f"{_X}id", "sd-types")
    for cat_id, desc in [
        ("sd-full",       "Full quotation: verbatim or near-verbatim citation of a biblical verse"),
        ("sd-partial",    "Partial quotation: a recognisable portion of a verse"),
        ("sd-paraphrase", "Paraphrase: biblical content restated in different words"),
        ("sd-allusion",   "Allusion: brief thematic or verbal echo of a scriptural passage"),
    ]:
        cat = etree.SubElement(taxonomy, f"{_T}category")
        cat.set(f"{_X}id", cat_id)
        catDesc = etree.SubElement(cat, f"{_T}catDesc")
        catDesc.text = desc

    # ── text / body / ab ─────────────────────────────────────────────────────
    text_el = etree.SubElement(root,    f"{_T}text")
    body    = etree.SubElement(text_el, f"{_T}body")
    ab      = etree.SubElement(body,    f"{_T}ab")
    ab.set(f"{_X}id", "source-text")

    segments = _compute_segments(source["text"], annotations)
    last_el  = None  # most-recently appended child element

    for seg in segments:
        raw = seg["text"]
        if not seg["annotation_ids"]:
            # plain text: append to .text of <ab> or .tail of last element
            if last_el is None:
                ab.text = (ab.text or "") + raw
            else:
                last_el.tail = (last_el.tail or "") + raw
        else:
            ann_refs = " ".join(
                f"#ann{annotations[i]['id']}" for i in seg["annotation_ids"]
            )
            subtypes = {annotations[i]["quote_type"] for i in seg["annotation_ids"]}
            subtype  = next(iter(subtypes)) if len(subtypes) == 1 else "mixed"

            seg_el = etree.SubElement(ab, f"{_T}seg")
            seg_el.set(f"{_X}id",  f"seg{seg['start']}x{seg['end']}")
            seg_el.set("ana",      ann_refs)
            seg_el.set("type",     "biblical-reference")
            seg_el.set("subtype",  subtype)
            seg_el.text = raw
            last_el = seg_el

    # ── standOff ─────────────────────────────────────────────────────────────
    stand_off = etree.SubElement(root,       f"{_T}standOff")
    list_ann  = etree.SubElement(stand_off,  f"{_T}listAnnotation")

    for a in annotations:
        ann_el = etree.SubElement(list_ann, f"{_T}annotation")
        ann_el.set(f"{_X}id", f"ann{a['id']}")
        ann_el.set("type",    "biblical-reference")
        ann_el.set("subtype", a.get("quote_type", "allusion"))
        ann_el.set("ana",     f"#sd-{a.get('quote_type','allusion')}")

        note_el = etree.SubElement(ann_el, f"{_T}note")
        note_el.set("type", "quotedText")
        note_el.text = a.get("quote_text", "")

        refs_el = etree.SubElement(ann_el, f"{_T}listRef")
        for ref in (a.get("refs") or []):
            ref_clean = ref.strip().lower()
            ref_el = etree.SubElement(refs_el, f"{_T}ref")
            ref_el.set("target", f"bible:{ref_clean}")
            ref_el.text = _ref_label(ref_clean, book_names)

    return etree.tostring(
        root,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
    )


# ── import ────────────────────────────────────────────────────────────────────

def tei_to_source_data(xml_bytes: bytes) -> dict:
    """
    Parse a TEI file produced by :func:`source_to_tei`.

    Returns a dict::

        {
          "name":  str,
          "text":  str,
          "annotations": [
              {
                  "quote_text":  str,
                  "quote_type":  str,
                  "refs":        [str, ...],
                  "span_start":  int | None,
                  "span_end":    int | None,
              },
              ...
          ]
        }
    """
    root = etree.fromstring(xml_bytes)

    # ── source name ──────────────────────────────────────────────────────────
    title_el = root.find(f".//{_T}teiHeader//{_T}titleStmt/{_T}title")
    name = (title_el.text or "Untitled").strip() if title_el is not None else "Untitled"

    # ── reconstruct plain text + offset map for <seg> ids ────────────────────
    ab = root.find(f".//{_T}body//{_T}ab")
    if ab is None:
        ab = root.find(f".//{_T}body")

    text_parts: list[str] = []
    # Maps xml:id → (start_char, end_char) offsets within the joined text
    offset_map: dict[str, tuple[int, int]] = {}

    def _walk(el: etree._Element) -> None:
        if el.text:
            text_parts.append(el.text)
        for child in el:
            child_start = sum(len(p) for p in text_parts)
            _walk(child)
            child_end = sum(len(p) for p in text_parts)
            xml_id = child.get(f"{_X}id")
            if xml_id:
                offset_map[xml_id] = (child_start, child_end)
            if child.tail:
                text_parts.append(child.tail)

    if ab is not None:
        _walk(ab)

    full_text = "".join(text_parts)

    # ── parse standOff annotations ────────────────────────────────────────────
    annotations: list[dict] = []

    for ann_el in root.findall(f".//{_T}standOff//{_T}annotation"):
        ann_xml_id = ann_el.get(f"{_X}id", "")
        subtype    = ann_el.get("subtype", "allusion")

        note_el    = ann_el.find(f"{_T}note[@type='quotedText']")
        quote_text = (note_el.text or "").strip() if note_el is not None else ""

        refs: list[str] = []
        for ref_el in ann_el.findall(f".//{_T}ref"):
            target = ref_el.get("target", "")
            if target.startswith("bible:"):
                refs.append(target[6:])

        # Determine character span from the seg elements referencing this annotation
        span_start = span_end = None
        if ab is not None and ann_xml_id:
            ref_key = f"#{ann_xml_id}"
            seg_offsets = []
            for seg_el in ab.iter(f"{_T}seg"):
                ana_val = seg_el.get("ana", "")
                if ref_key in ana_val.split():
                    seg_id = seg_el.get(f"{_X}id")
                    if seg_id and seg_id in offset_map:
                        seg_offsets.append(offset_map[seg_id])
            if seg_offsets:
                span_start = min(s for s, _ in seg_offsets)
                span_end   = max(e for _, e in seg_offsets)

        annotations.append({
            "quote_text":  quote_text,
            "quote_type":  subtype,
            "refs":        refs,
            "span_start":  span_start,
            "span_end":    span_end,
        })

    return {"name": name, "text": full_text, "annotations": annotations}
