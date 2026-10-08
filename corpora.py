"""Bible corpora per language.

``grc`` -> data/greek   (LXX + NT, PTA texts; cRef prefixes LXX / NA)
``lat`` -> data/vulgate (Vulgate;             cRef prefix   Vg)

Every source document belongs to one language; verses, book codes, prompt and
PTA abbreviations are taken from the matching corpus.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import greek

DATA_DIR = Path(__file__).resolve().parent / "data"
DIRS = {"grc": DATA_DIR / "greek", "lat": DATA_DIR / "vulgate"}
LANGUAGES = tuple(DIRS)
DEFAULT_LANG = "grc"


def normalize_lang(code: str | None) -> str | None:
    """xml:lang value -> 'grc' / 'lat'; None if unsupported."""
    c = (code or "").strip().lower()
    if c == "grc" or c.startswith("grc-"):
        return "grc"
    if c in ("lat", "la") or c.startswith(("lat-", "la-")):
        return "lat"
    return None


def guess_lang(text: str) -> str:
    """Greek if a substantial share of the letters is in Greek script."""
    greek_n = latin_n = 0
    for ch in text[:20000]:
        o = ord(ch)
        if 0x370 <= o <= 0x3FF or 0x1F00 <= o <= 0x1FFF:
            greek_n += 1
        elif ch.isalpha():
            latin_n += 1
    return "grc" if greek_n > 0.3 * max(1, greek_n + latin_n) else "lat"


def _norm_id(value: str) -> str:
    v = value.strip().lower()
    return str(int(v)) if v.isdigit() else v


class Corpus:
    def __init__(self, lang: str, directory: Path):
        self.lang = lang
        self.dir = directory
        self._meta = self._verses = self._mapping = self._structure = self._sub = None

    # ── loading ──────────────────────────────────────────────────────────────
    @property
    def meta(self) -> dict:
        if self._meta is None:
            try:
                self._meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._meta = {"language": "", "corpus": "the Bible", "prompt_notes": ""}
        return self._meta

    def _rows(self):
        with open(self.dir / "bible.tsv", newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

    @property
    def verses(self) -> dict[str, str]:
        """``book_chapter:verse`` -> text (ids are strings: LXX has 2a, t, 24a ...)."""
        if self._verses is None:
            v: dict[str, str] = {}
            for row in self._rows():
                key = f"{row['book_code'].strip().lower()}_{_norm_id(row['chapter_number'])}:{_norm_id(row['verse_index'])}"
                v[key] = row["text"]
            self._verses = v
        return self._verses

    @property
    def mapping(self) -> dict[str, dict]:
        if self._mapping is None:
            m: dict[str, dict] = {}
            with open(self.dir / "book_mapping.tsv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    m[row["book_code"].strip().lower()] = {
                        "name": row["work_name"],
                        "testament": row["testament"],
                        "greek_name": row.get("greek_name") or "",
                        "pta_abbr": row.get("pta_abbr") or "",
                        "pta_corpus": row.get("pta_corpus") or "",
                    }
            self._mapping = m
        return self._mapping

    @property
    def structure(self) -> dict[str, dict[str, list[dict]]]:
        if self._structure is None:
            s: dict[str, dict[str, list[dict]]] = {}
            for row in self._rows():
                book = row["book_code"].strip().lower()
                s.setdefault(book, {}).setdefault(_norm_id(row["chapter_number"]), []).append(
                    {"verse": _norm_id(row["verse_index"]), "text": row["text"]})
            for book in s:
                for ch in s[book]:
                    s[book][ch].sort(key=lambda v: greek.verse_sort_key(v["verse"]))
            self._structure = s
        return self._structure

    @property
    def codes(self) -> list[str]:
        return sorted(self.mapping)

    def abbr_map(self) -> dict[tuple[str, str], str]:
        """(corpus, PTA abbreviation), both lower-case -> internal book code."""
        return {(i["pta_corpus"].lower(), i["pta_abbr"].lower()): c
                for c, i in self.mapping.items() if i.get("pta_abbr")}

    def book_info(self) -> dict[str, dict]:
        return {c: {"name": i["name"], "pta_abbr": i["pta_abbr"], "pta_corpus": i["pta_corpus"]}
                for c, i in self.mapping.items()}

    # ── references ───────────────────────────────────────────────────────────
    def _subverses(self) -> dict[str, list[str]]:
        if self._sub is None:
            idx: dict[str, list[str]] = {}
            for k in self.verses:
                m = re.match(r"^(.+:\d+)[a-z]+", k)
                if m:
                    idx.setdefault(m.group(1), []).append(k)
            self._sub = idx
        return self._sub

    def resolve_ref(self, ref: str) -> list[str]:
        """Existing verse keys for *ref* (``ps_9:3`` also matches ``3a``/``3b``;
        ``prov_24:23`` finds the LXX chapter ``24a``)."""
        if ref in self.verses:
            return [ref]
        if ref in self._subverses():
            return self._subverses()[ref]
        p = greek.parse_ref(ref)
        if p:
            alt = f"{p[0]}_{p[1]}a:{p[2]}"
            if alt in self.verses:
                return [alt]
        return []

    def verse_text(self, ref: str) -> str:
        return " ".join(self.verses.get(k, "") for k in self.resolve_ref(ref.strip().lower()))

    def score_for(self, quote_text: str, refs: list[str]) -> float | None:
        texts = []
        for r in refs:
            texts.extend(self.verses.get(k, "") for k in self.resolve_ref(r.strip().lower()))
        return greek.score_quote(quote_text, texts)

    def validate_refs(self, refs: list[str]) -> tuple[list[str], int]:
        """Drop references that cannot exist; (valid refs, number dropped)."""
        ok, dropped = [], 0
        for r in refs:
            r = r.strip().lower()
            p = greek.parse_ref(r)
            if not p or p[0] not in self.mapping:
                dropped += 1
            elif p[0] in self.structure and not self.resolve_ref(r):
                dropped += 1
            elif r not in ok:
                ok.append(r)
        return ok, dropped

    def refs_from_crefs(self, crefs: list[str]) -> list[str]:
        out: list[str] = []
        amap = self.abbr_map()
        for c in crefs:
            out.extend(greek.parse_cref(c, amap))
        return out


_CORPORA = {lang: Corpus(lang, d) for lang, d in DIRS.items()}


def get_corpus(lang: str | None) -> Corpus:
    return _CORPORA.get(lang or DEFAULT_LANG) or _CORPORA[DEFAULT_LANG]


def merged_mapping() -> dict[str, dict]:
    """Book names across all corpora (for dashboards): Latin names win for shared codes."""
    return {**get_corpus("grc").mapping, **get_corpus("lat").mapping}
