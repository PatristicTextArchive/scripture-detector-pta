"""Helpers for Greek (and other polytonic) text: folding, fuzzy span search,
quote scoring, chunking, and conversion between the app's internal verse
references (``ps_62:9``) and PTA ``cRef`` values (``LXX:Ps:62:9``).
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left

_LETTER_CATS = {"Lu", "Ll", "Lt", "Lo"}


# ── folding ──────────────────────────────────────────────────────────────────

def _fold_char(ch: str) -> str:
    """Base letter(s) of *ch*: no diacritics, lower case, final/lunate sigma -> σ."""
    out = []
    for c in unicodedata.normalize("NFD", ch):
        cat = unicodedata.category(c)
        if cat in _LETTER_CATS:
            c = c.lower()
            if c in ("ς", "ϲ"):
                c = "σ"
            out.append(c)
    return "".join(out)


def skeleton(text: str) -> tuple[str, list[int]]:
    """Fold *text* to bare lower-case letters (no spaces, accents, punctuation).

    Returns the folded string and, for every folded character, the index of the
    originating character in *text*.
    """
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        f = _fold_char(ch)
        for c in f:
            chars.append(c)
            index.append(i)
    return "".join(chars), index


def fold_tokens(text: str) -> list[str]:
    """Folded word tokens (used for scoring)."""
    toks = []
    for raw in re.split(r"\s+", text):
        f = "".join(_fold_char(c) for c in raw)
        if f:
            toks.append(f)
    return toks


# ── span search ──────────────────────────────────────────────────────────────

class SpanIndex:
    """Search quotes in a document ignoring accents, breathings, case, sigma
    variants, punctuation and whitespace differences."""

    def __init__(self, text: str):
        self.text = text
        self.skel, self.map = skeleton(text)
        self._used: dict[str, int] = {}

    def _to_original(self, a: int, b: int) -> tuple[int, int]:
        start = self.map[a]
        end = self.map[b - 1] + 1
        while end < len(self.text) and unicodedata.category(self.text[end]) == "Mn":
            end += 1
        return start, end

    def find(self, quote: str, min_anchor: int = 10) -> tuple[int, int] | None:
        """Return (start, end) in the original text, or None.

        Identical quotes are assigned to successive occurrences, so a quotation
        that appears twice in the document is located twice.
        """
        q, _ = skeleton(quote)
        if not q:
            return None
        n = self._used.get(q, 0)
        pos = -1
        start_from = 0
        for _ in range(n + 1):
            nxt = self.skel.find(q, start_from)
            if nxt == -1:
                break
            pos = nxt
            start_from = nxt + 1
        if pos == -1 and n > 0:                      # fewer occurrences than uses
            pos = self.skel.find(q)
        if pos != -1:
            self._used[q] = n + 1
            return self._to_original(pos, pos + len(q))
        return self._find_by_anchors(q, min_anchor)

    def _find_by_anchors(self, q: str, min_anchor: int) -> tuple[int, int] | None:
        """Fallback for quotes with small deviations: locate the beginning and
        the end of the quote separately."""
        if len(q) < 2 * min_anchor:
            return None
        for length in range(min(30, len(q) // 2), min_anchor - 1, -2):
            head, tail = q[:length], q[-length:]
            p = self.skel.find(head)
            while p != -1:
                window_end = p + int(len(q) * 1.6) + length
                t = self.skel.find(tail, p + length, window_end)
                if t != -1:
                    return self._to_original(p, t + length)
                p = self.skel.find(head, p + 1)
        return None


# ── scoring ──────────────────────────────────────────────────────────────────

def score_quote(quote: str, verse_texts: list[str]) -> float | None:
    """Share of the quote's words that occur (or share a >=4 letter stem) in
    the referenced verses.  Returns None if no verse text is available."""
    verse_texts = [v for v in verse_texts if v]
    if not verse_texts:
        return None
    qt = fold_tokens(quote)
    if not qt:
        return None
    vt = set()
    for v in verse_texts:
        vt.update(fold_tokens(v))
    stems = {t[:4] for t in vt if len(t) >= 4}
    hit = sum(1 for t in qt if t in vt or (len(t) >= 4 and t[:4] in stems))
    return round(hit / len(qt), 2)


# ── chunking ─────────────────────────────────────────────────────────────────

_SENT_END = re.compile(r"[.;·\u0387\u037e?!]\s")


def chunk_text(text: str, max_chars: int = 12000) -> list[tuple[int, int]]:
    """Split *text* into (start, end) ranges of at most *max_chars*, preferring
    paragraph breaks, then sentence ends, then whitespace."""
    n = len(text)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            cut = text.rfind("\n", start + max_chars // 2, end)
            if cut == -1:
                last = None
                for m in _SENT_END.finditer(text, start + max_chars // 2, end):
                    last = m
                cut = last.end() - 1 if last else text.rfind(" ", start + max_chars // 2, end)
            if cut != -1 and cut > start:
                end = cut + 1
        ranges.append((start, end))
        start = end
    return ranges


def split_in_two(text: str, overlap: int = 120) -> tuple[str, str]:
    """Split *text* near its middle at a paragraph, sentence or word boundary.
    The second part starts *overlap* characters early (at a word boundary), so
    that a quotation at the seam is complete in at least one part."""
    n = len(text)
    mid, lo, hi = n // 2, n // 4, 3 * n // 4
    cut = None
    nl = [i for i in range(lo, hi) if text[i] == "\n"]
    if nl:
        cut = min(nl, key=lambda i: abs(i - mid)) + 1
    else:
        ends = [m.end() for m in _SENT_END.finditer(text, lo, hi)]
        if ends:
            cut = min(ends, key=lambda i: abs(i - mid))
        else:
            sp = [i for i in range(lo, hi) if text[i] == " "]
            cut = (min(sp, key=lambda i: abs(i - mid)) + 1) if sp else mid
    start2 = max(0, cut - overlap)
    if start2 > 0:
        sp = text.find(" ", start2, cut)
        start2 = sp + 1 if sp != -1 else cut
    return text[:cut], text[start2:]


# ── verse references ─────────────────────────────────────────────────────────

_REF_RE = re.compile(r"^([a-z0-9]+)_([0-9a-z]+):([0-9a-z]+)$")
_RANGE_RE = re.compile(r"^([a-z0-9]+_[0-9a-z]+):(\d+)-(\d+)$")


def parse_ref(ref: str) -> tuple[str, str, str] | None:
    """``ps_62:9`` -> (book, chapter, verse); chapter/verse ids may carry letters."""
    m = _REF_RE.match(ref.strip().lower())
    return (m.group(1), m.group(2), m.group(3)) if m else None


def expand_ranges(refs: list[str]) -> list[str]:
    out: list[str] = []
    for ref in refs:
        ref = ref.strip().lower()
        m = _RANGE_RE.match(ref)
        if m:
            out.extend(f"{m.group(1)}:{v}" for v in range(int(m.group(2)), int(m.group(3)) + 1))
        else:
            out.append(ref)
    return out


def verse_sort_key(v: str) -> tuple:
    """Natural order for verse ids like 1, 2, 2a, 2b, 10, t (titles first)."""
    m = re.match(r"^(\d+)(.*)$", v)
    return (1, int(m.group(1)), m.group(2)) if m else (0, 0, v)


def format_crefs(refs: list[str], book_info: dict[str, dict]) -> list[str]:
    """Internal refs -> PTA cRef values, grouped per book and chapter.

    ``["ps_62:9", "ps_62:10", "ps_62:12"]`` -> ``["LXX:Ps:62:9-10.12"]``.
    Verse ids with letters (``2a``) are reduced to the verse number, as the PTA
    pattern only allows digits.  Unknown books are skipped.
    """
    groups: dict[tuple[str, str, int], set[int]] = {}
    order: list[tuple[str, str, int]] = []
    for ref in refs:
        p = parse_ref(ref)
        if not p:
            continue
        book, chapter, verse = p
        info = book_info.get(book)
        if not info or not info.get("pta_abbr"):
            continue
        nums = [int(x) for x in re.findall(r"\d+", verse)]
        chap = re.match(r"\d+", chapter)
        if not nums or not chap:          # e.g. Odes title 't', Sirach prologue 'p'
            continue
        key = (info["pta_corpus"], info["pta_abbr"], chap.group(0))
        if key not in groups:
            groups[key] = set()
            order.append(key)
        groups[key].update(range(nums[0], nums[-1] + 1) if len(nums) == 2 and nums[1] - nums[0] < 50 else nums)

    result = []
    for key in order:
        verses = sorted(groups[key])
        parts, i = [], 0
        while i < len(verses):
            j = i
            while j + 1 < len(verses) and verses[j + 1] == verses[j] + 1:
                j += 1
            parts.append(str(verses[i]) if i == j else f"{verses[i]}-{verses[j]}")
            i = j + 1
        corpus, abbr, chapter = key
        result.append(f"{corpus}:{abbr}:{chapter}:{'.'.join(parts)}")
    return result


def parse_cref(cref: str, abbr_to_code: dict[tuple[str, str], str]) -> list[str]:
    """PTA cRef (``LXX:Gn:1:1-3.5``) -> internal refs.  *abbr_to_code* maps
    (corpus, abbreviation) in lower case to a book code.  Empty list if the
    value is not a verse reference of this corpus or the book is unknown."""
    bits = cref.strip().split(":")
    if len(bits) != 4:
        return []
    corpus, abbr, chapter, verses = bits
    code = abbr_to_code.get((corpus.lower(), abbr.lower()))
    if not code or not chapter.isdigit():
        return []
    refs: list[str] = []
    for part in verses.split("."):
        m = re.match(r"^(\d+)(?:-(\d+))?", part)
        if not m:
            continue
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        refs.extend(f"{code}_{int(chapter)}:{v}" for v in range(a, min(b, a + 200) + 1))
    return refs


# ── misc ─────────────────────────────────────────────────────────────────────

def bisect_run(starts: list[int], pos: int) -> int:
    """Index of the last start <= pos."""
    return max(0, bisect_left(starts, pos + 1) - 1)
