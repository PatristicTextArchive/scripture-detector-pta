"""Retrieval over the Bible texts (the "R" of RAG).

For a piece of text, find the Bible verses that share wording with it.  The
search is lexical and tolerant of inflection and orthography: words are folded
(no accents/breathings, ς -> σ, Latin u/v and i/j) and cut to a stem of five
letters.  Rare stems and stem pairs of the text are looked up in an inverted
index of all verses; a sliding window over the text collects candidates.

``candidates``       best verses with the passage of the text they match
                     (they go into the prompt of the model)
``lexical_matches``  verse matches with an alignment (span, coverage) strong
                     enough to be reported without a model
"""
from __future__ import annotations

import heapq
import math
import re
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from operator import itemgetter

import greek

STEM = 5                # letters kept per word (Greek, Latin)
SYC_STEM = 4            # Syriac: roots are short, words carry prefixes
SYC_PROCLITICS = "ܘܒܕܠܟ"  # w- b- d- l- k-: attached conjunction / prepositions / relative particle
SYC_STRIP = 2           # at most this many proclitics are removed (while 3 letters remain)
SCALES = ((5, 2), (10, 3))     # (words per window, step): short windows find short quotations
PER_WINDOW = 20                # verses kept per window
MAX_DF_RATIO = 0.008   # stems occurring in more verses than this are not used to find candidates
MAX_DF_MIN = 120

def _mark_class() -> str:
    """Regex character class of all combining marks of the BMP (Greek accents, Hebrew and Syriac points, ...)."""
    import unicodedata
    ranges, start = [], None
    for cp in range(0x300, 0x10000):
        is_mark = unicodedata.category(chr(cp)).startswith("M")
        if is_mark and start is None:
            start = cp
        elif not is_mark and start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, 0xFFFF))
    return "".join(f"\\u{a:04x}" if a == b else f"\\u{a:04x}-\\u{b:04x}" for a, b in ranges)


# a word: letters and the combining marks that belong to them (also Syriac vowel points)
_TOKEN = re.compile(r"(?:[^\W\d_]|[" + _mark_class() + r"])+")
_LAT = str.maketrans({"v": "u", "j": "i", "æ": "e", "œ": "e", "ę": "e"})


@dataclass
class Token:
    start: int
    end: int
    stem: str


def _fold_word(word: str, lang: str) -> str:
    f = "".join(greek._fold_char(c) for c in word)
    if lang == "lat":
        f = f.translate(_LAT).replace("ae", "e").replace("oe", "e")
    return f


def _stem(f: str, lang: str) -> str:
    if lang == "syc":
        n = 0
        while n < SYC_STRIP and len(f) > 3 and f[0] in SYC_PROCLITICS:
            f, n = f[1:], n + 1
        return f[:SYC_STEM]
    return f[:STEM]


def tokenize(text: str, lang: str) -> list[Token]:
    out = []
    for m in _TOKEN.finditer(text):
        f = _fold_word(m.group(), lang)
        if f:
            out.append(Token(m.start(), m.end(), sys.intern(_stem(f, lang))))
    return out


@dataclass
class Candidate:
    key: str                  # verse key, e.g. ps_62:9
    score: float              # ranking score
    start: int                # character span of the matching passage in the searched text
    end: int
    words: int = 0            # aligned words
    weight: float = 0.0       # sum of idf of the aligned words
    parallels: int = 1        # verses containing this stretch of words (large = common phrase)


@dataclass
class Match:
    refs: list[str]
    start: int
    end: int
    words: int                # matched words
    coverage: float           # share of the verse's words found in the passage
    precision: float          # share of the passage's words that belong to the verse
    weight: float             # sum of idf of the matched words
    quote_type: str           # "full" | "partial"
    parallels: int = 1        # verses containing this stretch of words (1 = unique)
    keys: list[str] = field(default_factory=list)


class VerseIndex:
    def __init__(self, verses: dict[str, str], lang: str):
        self.lang = lang
        self.keys = list(verses)
        self.stems: list[list[str]] = []
        post: dict[str, list[int]] = defaultdict(list)
        for i, k in enumerate(self.keys):
            st = [t.stem for t in tokenize(verses[k], lang)]
            self.stems.append(st)
            for s in set(st):
                post[s].append(i)
        n = len(self.keys)
        self.n = n
        self.idf = {s: math.log((n + 1) / (len(p) + 0.5)) for s, p in post.items()}
        max_df = max(MAX_DF_MIN, int(n * MAX_DF_RATIO))
        self.rare = {s: p for s, p in post.items() if len(p) <= max_df}
        bi: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, st in enumerate(self.stems):
            for pair in set(zip(st, st[1:])):
                bi[pair].append(i)
        self.bigrams = {p: v for p, v in bi.items() if len(v) <= max_df}
        self.common_pairs = {p for p, v in bi.items() if len(v) > max_df}      # too common to be indexed
        self._verse_bigrams: dict[int, set] = {}

    def verse_bigrams(self, i: int) -> set:
        b = self._verse_bigrams.get(i)
        if b is None:
            st = self.stems[i]
            b = self._verse_bigrams[i] = set(zip(st, st[1:]))
        return b

    def phrase_count(self, pairs: list[tuple[str, str]]) -> int:
        """Number of verses that contain all the (indexed) word pairs: how often
        this stretch of words occurs in the Bible.  Formulas (doxologies,
        greetings) give large numbers; if no pair is rare enough to be indexed
        the stretch is treated as maximally common."""
        lists = sorted((self.bigrams[p] for p in set(pairs) if p in self.bigrams), key=len)[:5]
        if not lists:
            # no rare pair: a stretch made of common pairs is maximally common; pairs that do not occur
            # in the Bible at all say nothing (the text is not a verbatim stretch of it)
            return self.n if any(p in self.common_pairs for p in pairs) else 0
        common = set(lists[0])
        for lst in lists[1:]:
            common &= set(lst)
            if not common:
                break
        return len(common)

    def search(self, stems: list[str], top: int = 20) -> list[tuple[int, float]]:
        """Best verses for one window of stems."""
        acc: dict[int, float] = defaultdict(float)
        for s in set(stems):
            p = self.rare.get(s)
            if p:
                w = self.idf[s]
                for i in p:
                    acc[i] += w
        for a, b in set(zip(stems, stems[1:])):
            p = self.bigrams.get((a, b))
            if p:
                w = self.idf.get(a, 0) + self.idf.get(b, 0)
                for i in p:
                    acc[i] += w
        return heapq.nlargest(top, acc.items(), key=itemgetter(1)) if acc else []


_INDEXES: dict[str, VerseIndex] = {}


_LOCK = threading.Lock()


def get_index(corpus) -> VerseIndex:
    """The index of a corpus; built on first use (a few seconds), then cached."""
    idx = _INDEXES.get(corpus.lang)
    if idx is None:
        with _LOCK:
            idx = _INDEXES.get(corpus.lang)
            if idx is None:
                idx = _INDEXES[corpus.lang] = VerseIndex(corpus.verses, corpus.lang)
    return idx


def phrase_info(text: str, corpus) -> tuple[int, int]:
    """(number of words, number of verses containing this stretch of words).  A small
    second number means the stretch is specific; a large one that it is a common
    phrase; 0 that it does not occur verbatim in the Bible."""
    index = get_index(corpus)
    stems = [t.stem for t in tokenize(text, corpus.lang)]
    pairs = list(zip(stems, stems[1:]))
    return len(stems), (index.phrase_count(pairs) if pairs else 0)


# ── candidates ───────────────────────────────────────────────────────────────

def _windows(n_tokens: int, size: int, stride: int):
    if n_tokens <= size:
        yield 0, n_tokens
        return
    for s in range(0, n_tokens - size + stride, stride):
        yield s, min(s + size, n_tokens)


def scan(text: str, corpus, min_score: float = 12.0) -> tuple[list[Token], dict[int, tuple[float, int, int]]]:
    """Slide over *text*; best (score, first token, last token) per verse index."""
    index = get_index(corpus)
    toks = tokenize(text, corpus.lang)
    stems = [t.stem for t in toks]
    best: dict[int, tuple[float, int, int]] = {}
    for size, stride in SCALES:
        for a, b in _windows(len(toks), size, stride):
            for vi, sc in index.search(stems[a:b], PER_WINDOW):
                if sc >= min_score and (vi not in best or sc > best[vi][0]):
                    best[vi] = (sc, a, b)
    return toks, best


def candidates(text: str, corpus, top: int = 20, min_score: float = 8.0, pool: int = 400,
               per_region: int = 3, min_words: int = 0, min_weight: float = 0.0,
               max_parallels: int = 10**9) -> list[Candidate]:
    """Verses that share wording with *text*, best first.

    The *pool* best verses of the window search are re-ranked by an alignment of
    the verse with the text (weight of the matching words, weighted by how much
    of the aligned passage belongs to the verse); the span is the aligned passage."""
    index = get_index(corpus)
    toks, best = scan(text, corpus, min_score)
    stems = [t.stem for t in toks]
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:pool]
    out: list[Candidate] = []
    for vi, (sc, a, b) in ranked:
        v = index.stems[vi]
        lo, hi = max(0, a - len(v)), min(len(stems), b + len(v))
        al = _align(stems[lo:hi], v, index.idf)
        if not al:
            continue
        first, last, words, weight, pairs = al
        if words < min_words or weight < min_weight:
            continue
        parallels = index.phrase_count(pairs)
        if parallels > max_parallels:
            continue
        score = weight * (0.4 + 0.6 * words / (last - first))
        out.append(Candidate(index.keys[vi], score, toks[lo + first].start, toks[lo + last - 1].end,
                             words, round(weight, 1), parallels))
    out.sort(key=lambda c: -c.score)
    return _spread(out, top, per_region)


def _spread(cands: list[Candidate], top: int, per_region: int, bucket: int = 150) -> list[Candidate]:
    """Choose *top* candidates so that every part of the text gets some: at most
    *per_region* per stretch of *bucket* characters first, then the rest by score."""
    chosen: list[Candidate] = []
    used: dict[int, int] = defaultdict(int)
    rest: list[Candidate] = []
    for c in cands:
        b = c.start // bucket
        if used[b] < per_region and len(chosen) < top:
            used[b] += 1
            chosen.append(c)
        else:
            rest.append(c)
    chosen.extend(rest[: max(0, top - len(chosen))])
    chosen.sort(key=lambda c: -c.score)
    return chosen


# ── matches (alignment) ──────────────────────────────────────────────────────

def _align(text_stems: list[str], verse_stems: list[str], idf: dict[str, float]):
    """Matched stretches of *verse_stems* in *text_stems*: (first, last+1) in the
    text, matched word count, idf weight; blocks of one word are ignored."""
    sm = SequenceMatcher(None, text_stems, verse_stems, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 2]
    if not blocks:
        return None
    first, last = blocks[0].a, blocks[-1].a + blocks[-1].size
    words = sum(b.size for b in blocks)
    weight = sum(idf.get(text_stems[b.a + k], 0.0) for b in blocks for k in range(b.size))
    pairs = [(text_stems[b.a + k], text_stems[b.a + k + 1]) for b in blocks for k in range(b.size - 1)]
    return first, last, words, weight, pairs


def lexical_matches(text: str, corpus, min_words: int = 6, min_weight: float = 22.0,
                    min_precision: float = 0.6, min_score: float = 14.0, max_parallels: int = 5) -> list[Match]:
    """Passages of *text* that reuse a verse closely enough to be reported
    without a model (type full / partial), best alignment per passage."""
    index = get_index(corpus)
    toks, best = scan(text, corpus, min_score)
    stems = [t.stem for t in toks]
    found: list[Match] = []
    for vi, (sc, a, b) in best.items():
        v = index.stems[vi]
        if len(v) < 3:
            continue
        lo, hi = max(0, a - len(v)), min(len(stems), b + len(v))
        al = _align(stems[lo:hi], v, index.idf)
        if not al:
            continue
        first, last, words, weight, pairs = al
        region = last - first
        precision, coverage = words / region, words / len(v)
        if words < min_words or weight < min_weight or precision < min_precision:
            continue
        parallels = index.phrase_count(pairs)
        if parallels > max_parallels:          # a formula, not a quotation of this verse
            continue
        s_tok, e_tok = lo + first, lo + last
        qtype = "full" if coverage >= 0.75 and precision >= 0.75 else "partial"
        found.append(Match([index.keys[vi]], toks[s_tok].start, toks[e_tok - 1].end, words,
                           round(coverage, 2), round(precision, 2), round(weight, 1), qtype,
                           parallels, [index.keys[vi]]))
    return _merge(found, text)


_KEY = re.compile(r"^(.+):(\d+)$")


def _consecutive(k1: str, k2: str) -> bool:
    m1, m2 = _KEY.match(k1), _KEY.match(k2)
    return bool(m1 and m2 and m1.group(1) == m2.group(1) and abs(int(m1.group(2)) - int(m2.group(2))) == 1)


def _overlap(a: Match, b: Match) -> float:
    ov = min(a.end, b.end) - max(a.start, b.start)
    return ov / max(1, min(a.end - a.start, b.end - b.start))


def _merge(matches: list[Match], text: str, max_refs: int = 3) -> list[Match]:
    """One match per passage: parallels (same wording in several verses) share
    an annotation; then consecutive verses that follow each other in the text
    (only punctuation between them) are joined into one passage."""
    matches.sort(key=lambda m: (m.start, -m.weight))
    merged: list[Match] = []
    for m in matches:
        target = next((g for g in merged if _overlap(g, m) >= 0.6), None)
        if target is None:
            merged.append(m)
            continue
        joined = any(_consecutive(k, j) for k in target.keys for j in m.keys)
        if joined or m.weight >= 0.6 * target.weight:
            target.start, target.end = min(target.start, m.start), max(target.end, m.end)
            target.words = max(target.words, m.words)
            target.weight = max(target.weight, m.weight)
            for r in m.refs:
                if r not in target.refs:
                    target.refs.append(r)
            target.keys = target.refs
    for g in merged:
        g.refs = g.refs[:max_refs]
        g.keys = g.refs
    merged.sort(key=lambda m: m.start)

    chain: list[Match] = []
    for m in merged:
        prev = chain[-1] if chain else None
        if (prev is not None and m.start >= prev.end
                and not any(ch.isalpha() for ch in text[prev.end:m.start])
                and any(_consecutive(k, j) for k in prev.keys for j in m.keys)):
            prev.end = m.end
            prev.words += m.words
            prev.weight = round(prev.weight + m.weight, 1)
            prev.coverage = min(prev.coverage, m.coverage)
            prev.precision = min(prev.precision, m.precision)
            prev.quote_type = "full" if prev.quote_type == m.quote_type == "full" else "partial"
            prev.parallels = max(prev.parallels, m.parallels)
            prev.refs = prev.refs + [r for r in m.refs if r not in prev.refs]
            prev.keys = prev.refs
        else:
            chain.append(m)
    return chain
