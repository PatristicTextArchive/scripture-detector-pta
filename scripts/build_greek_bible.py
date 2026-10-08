#!/usr/bin/env python3
"""Build data/greek/bible.tsv and data/greek/book_mapping.tsv from the Greek
Bible texts of the Patristic Text Archive (PTA).

Source: https://github.com/PatristicTextArchive/pta_data  (data/pta9999/*/*-grc1.xml)
        Licence of the source texts: CC BY-SA 4.0 (see the TEI headers there).

Usage:
    python scripts/build_greek_bible.py --pta-data /path/to/pta_data
    python scripts/build_greek_bible.py --download      # shallow-clones pta_data to a temp dir

Internal book codes are the lower-cased PTA abbreviations (gn, ps, mt, 1cor, ...).
The PTA abbreviation (correct case) and corpus (LXX / NA) are stored in
book_mapping.tsv so that the TEI export can write cRef values like LXX:Ps:62:9.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from lxml import etree

TEI = "{http://www.tei-c.org/ns/1.0}"

# pta9999 work id -> (PTA abbreviation, corpus, testament, English name, Greek name)
BOOKS = [
    ("001", "Gn", "LXX", "ot", "Genesis", "Γένεσις"),
    ("002", "Ex", "LXX", "ot", "Exodus", "Ἔξοδος"),
    ("003", "Lv", "LXX", "ot", "Leviticus", "Λευιτικόν"),
    ("004", "Num", "LXX", "ot", "Numbers", "Ἀριθμοί"),
    ("005", "Dt", "LXX", "ot", "Deuteronomy", "Δευτερονόμιον"),
    ("006", "Jos", "LXX", "ot", "Joshua", "Ἰησοῦς"),
    ("007", "Judg", "LXX", "ot", "Judges", "Κριταί"),
    ("008", "Rt", "LXX", "ot", "Ruth", "Ῥούθ"),
    ("009", "1Sa", "LXX", "ot", "1 Samuel (1 Kingdoms)", "Βασιλειῶν Α"),
    ("010", "2Sa", "LXX", "ot", "2 Samuel (2 Kingdoms)", "Βασιλειῶν Β"),
    ("011", "1Ko", "LXX", "ot", "1 Kings (3 Kingdoms)", "Βασιλειῶν Γ"),
    ("012", "2Ko", "LXX", "ot", "2 Kings (4 Kingdoms)", "Βασιλειῶν Δ"),
    ("013", "1Chr", "LXX", "ot", "1 Chronicles (1 Paralipomena)", "Παραλειπομένων Α"),
    ("014", "2Chr", "LXX", "ot", "2 Chronicles (2 Paralipomena)", "Παραλειπομένων Β"),
    ("016", "3Esr", "LXX", "ap", "1 Esdras", "Ἔσδρας Α"),
    ("017", "Esr", "LXX", "ot", "Ezra-Nehemiah (2 Esdras; 1-10 Ezra, 11-23 Nehemiah)", "Ἔσδρας Β"),
    ("020", "Est", "LXX", "ot", "Esther", "Ἐσθήρ"),
    ("021", "Jdt", "LXX", "ap", "Judith", "Ἰουδίθ"),
    ("022", "Tob", "LXX", "ap", "Tobit", "Τωβίτ"),
    ("023", "1Mak", "LXX", "ap", "1 Maccabees", "Μακκαβαίων Α"),
    ("024", "2Mak", "LXX", "ap", "2 Maccabees", "Μακκαβαίων Β"),
    ("025", "3Mak", "LXX", "ap", "3 Maccabees", "Μακκαβαίων Γ"),
    ("026", "4Mak", "LXX", "ap", "4 Maccabees", "Μακκαβαίων Δ"),
    ("031", "Ps", "LXX", "ot", "Psalms (LXX numbering, incl. Ps 151)", "Ψαλμοί"),
    ("032", "Oden", "LXX", "ap", "Odes", "Ὠδαί"),
    ("033", "Prov", "LXX", "ot", "Proverbs", "Παροιμίαι"),
    ("034", "Eccl", "LXX", "ot", "Ecclesiastes", "Ἐκκλησιαστής"),
    ("035", "Song", "LXX", "ot", "Song of Songs", "Ἆισμα Ἀσμάτων"),
    ("036", "Job", "LXX", "ot", "Job", "Ἰώβ"),
    ("037", "Wis", "LXX", "ap", "Wisdom of Solomon", "Σοφία Σαλωμῶντος"),
    ("038", "Sir", "LXX", "ap", "Sirach", "Σοφία Σιραχ"),
    ("039", "PsSal", "LXX", "ap", "Psalms of Solomon", "Ψαλμοὶ Σαλωμῶντος"),
    ("040", "Hos", "LXX", "ot", "Hosea", "Ὡσηέ"),
    ("041", "Am", "LXX", "ot", "Amos", "Ἀμώς"),
    ("042", "Mi", "LXX", "ot", "Micah", "Μιχαίας"),
    ("043", "Joel", "LXX", "ot", "Joel", "Ἰωήλ"),
    ("044", "Ob", "LXX", "ot", "Obadiah", "Ὀβδιού"),
    ("045", "Jon", "LXX", "ot", "Jonah", "Ἰωνᾶς"),
    ("046", "Nah", "LXX", "ot", "Nahum", "Ναούμ"),
    ("047", "Hab", "LXX", "ot", "Habakkuk", "Ἀμβακούμ"),
    ("048", "Zeph", "LXX", "ot", "Zephaniah", "Σοφονίας"),
    ("049", "Hag", "LXX", "ot", "Haggai", "Ἀγγαῖος"),
    ("050", "Sach", "LXX", "ot", "Zechariah", "Ζαχαρίας"),
    ("051", "Mal", "LXX", "ot", "Malachi", "Μαλαχίας"),
    ("052", "Is", "LXX", "ot", "Isaiah", "Ἠσαΐας"),
    ("053", "Jr", "LXX", "ot", "Jeremiah", "Ἱερεμίας"),
    ("054", "Bar", "LXX", "ap", "Baruch", "Βαρούχ"),
    ("055", "Lam", "LXX", "ot", "Lamentations", "Θρῆνοι"),
    ("056", "EpistJer", "LXX", "ap", "Letter of Jeremiah", "Ἐπιστολὴ Ἱερεμίου"),
    ("059", "Hes", "LXX", "ot", "Ezekiel", "Ἰεζεκιήλ"),
    ("060", "Sus", "LXX", "ap", "Susanna", "Σουσάννα"),
    ("061", "Dn", "LXX", "ot", "Daniel", "Δανιήλ"),
    ("062", "Bel", "LXX", "ap", "Bel and the Dragon", "Βὴλ καὶ Δράκων"),
    ("063", "Mt", "NA", "nt", "Matthew", "Κατὰ Ματθαῖον"),
    ("064", "Mk", "NA", "nt", "Mark", "Κατὰ Μᾶρκον"),
    ("065", "Lk", "NA", "nt", "Luke", "Κατὰ Λουκᾶν"),
    ("066", "Jn", "NA", "nt", "John", "Κατὰ Ἰωάννην"),
    ("067", "Act", "NA", "nt", "Acts", "Πράξεις Ἀποστόλων"),
    ("068", "Rom", "NA", "nt", "Romans", "Πρὸς Ῥωμαίους"),
    ("069", "1Cor", "NA", "nt", "1 Corinthians", "Πρὸς Κορινθίους Α"),
    ("070", "2Cor", "NA", "nt", "2 Corinthians", "Πρὸς Κορινθίους Β"),
    ("071", "Gal", "NA", "nt", "Galatians", "Πρὸς Γαλάτας"),
    ("072", "Eph", "NA", "nt", "Ephesians", "Πρὸς Ἐφεσίους"),
    ("073", "Phil", "NA", "nt", "Philippians", "Πρὸς Φιλιππησίους"),
    ("074", "Col", "NA", "nt", "Colossians", "Πρὸς Κολοσσαεῖς"),
    ("075", "1Th", "NA", "nt", "1 Thessalonians", "Πρὸς Θεσσαλονικεῖς Α"),
    ("076", "2Th", "NA", "nt", "2 Thessalonians", "Πρὸς Θεσσαλονικεῖς Β"),
    ("077", "1Tim", "NA", "nt", "1 Timothy", "Πρὸς Τιμόθεον Α"),
    ("078", "2Tim", "NA", "nt", "2 Timothy", "Πρὸς Τιμόθεον Β"),
    ("079", "Tt", "NA", "nt", "Titus", "Πρὸς Τίτον"),
    ("080", "Phm", "NA", "nt", "Philemon", "Πρὸς Φιλήμονα"),
    ("081", "Heb", "NA", "nt", "Hebrews", "Πρὸς Ἑβραίους"),
    ("082", "Jak", "NA", "nt", "James", "Ἰακώβου"),
    ("083", "1P", "NA", "nt", "1 Peter", "Πέτρου Α"),
    ("084", "2P", "NA", "nt", "2 Peter", "Πέτρου Β"),
    ("085", "1Jn", "NA", "nt", "1 John", "Ἰωάννου Α"),
    ("086", "2Jn", "NA", "nt", "2 John", "Ἰωάννου Β"),
    ("087", "3Jn", "NA", "nt", "3 John", "Ἰωάννου Γ"),
    ("088", "Jud", "NA", "nt", "Jude", "Ἰούδα"),
    ("089", "Rev", "NA", "nt", "Revelation", "Ἀποκάλυψις Ἰωάννου"),
]

SKIP = {"rdg", "note", "witDetail", "surplus", "head", "title"}


def verse_text(div) -> str:
    """Plain text of a verse <div>: lemma readings only, punctuation attached."""
    out: list[str] = []

    def walk(el):
        if el.tag in (TEI + "pc",):
            txt = "".join(el.itertext()).strip()
            if out and out[-1].endswith(" "):
                out[-1] = out[-1].rstrip()
            out.append(txt + " ")
            return
        if el.tag == TEI + "w":
            out.append("".join(el.itertext()).strip() + " ")
            return
        if el.text:
            out.append(el.text)
        for ch in el:
            if isinstance(ch.tag, str) and etree.QName(ch).localname not in SKIP:
                walk(ch)
            if ch.tail:
                out.append(ch.tail)

    walk(div)
    text = re.sub(r"\s+", " ", "".join(out)).strip()
    # word-tagged files (<w>/<pc>) are pretty-printed: re-attach closing punctuation
    return re.sub(r"\s+([,.;:!?\u00b7\u0387\u037e])", r"\1", text)


def leaf_parts(div):
    """Leaf textpart divs (verses; 'section' parts are treated like verses)."""
    parts = [d for d in div.iter(TEI + "div") if d.get("type") == "textpart"]
    return [d for d in parts if not any(c.get("type") == "textpart" for c in d.iter(TEI + "div") if c is not d)]


def read_book(path: Path):
    root = etree.parse(str(path)).getroot()
    ed = root.find(f".//{TEI}div[@type='edition']")
    chapters = [d for d in ed if d.tag == TEI + "div" and d.get("subtype") == "chapter"]
    if not chapters:  # e.g. Susanna / Bel: verses directly under the edition
        chapters = [ed]
    for i, ch in enumerate(chapters, 1):
        chap_no = ch.get("n") if ch is not ed else "1"
        for part in leaf_parts(ch) if ch is not ed else [d for d in ed if d.tag == TEI + "div" and d.get("type") == "textpart"]:
            n = part.get("n")
            txt = verse_text(part)
            if n and txt:
                yield chap_no, n, txt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pta-data", type=Path, help="local clone of PatristicTextArchive/pta_data")
    ap.add_argument("--download", action="store_true", help="shallow-clone pta_data to a temp dir")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "greek")
    args = ap.parse_args()

    src = args.pta_data
    if args.download:
        tmp = Path(tempfile.mkdtemp())
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/PatristicTextArchive/pta_data.git", str(tmp)], check=True)
        src = tmp
    if not src:
        ap.error("give --pta-data PATH or --download")
    base = src / "data" / "pta9999"

    args.out.mkdir(parents=True, exist_ok=True)
    n_verses = 0
    with open(args.out / "bible.tsv", "w", encoding="utf-8") as fb, \
         open(args.out / "book_mapping.tsv", "w", encoding="utf-8") as fm:
        fb.write("testament\twork_name\tbook_code\tchapter_number\tverse_index\ttext\n")
        fm.write("testament\twork_name\tgreek_name\tbook_code\tpta_abbr\tpta_corpus\n")
        for wid, abbr, corpus, testament, name, gname in BOOKS:
            f = base / f"pta{wid}" / f"pta9999.pta{wid}.pta-grc1.xml"
            if not f.exists():
                print(f"missing: {f}", file=sys.stderr)
                continue
            code = abbr.lower()
            fm.write(f"{testament}\t{name}\t{gname}\t{code}\t{abbr}\t{corpus}\n")
            for chap, verse, txt in read_book(f):
                fb.write(f"{testament}\t{name}\t{code}\t{chap}\t{verse}\t{txt}\n")
                n_verses += 1
    print(f"wrote {n_verses} verses to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
