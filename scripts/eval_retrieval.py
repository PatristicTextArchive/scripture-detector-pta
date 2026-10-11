#!/usr/bin/env python3
"""Measure the retrieval against hand-made biblical references in TEI files.

Gold standard: <quote> elements that carry <ref decls="#biblical" cRef="..."/>
(as in the PTA editions).  Reported:

* candidate recall - is a gold verse among the top-K candidates of the text
  part (what a model gets in its prompt)?
* lexical matches  - how many gold quotes are found by the alignment alone, and
  which share of the matches agrees with a gold annotation (a lower bound for
  precision: passages nobody annotated count as "unconfirmed").

Usage:
    python scripts/eval_retrieval.py FILE_OR_DIR [...] [--lang grc] [--chunk 3500] [--top 20]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import corpora  # noqa: E402
import greek  # noqa: E402
import retrieval  # noqa: E402
import tei_doc  # noqa: E402


def gold_from_tei(xml: bytes, corpus):
    root = tei_doc._parse(xml)
    tm = tei_doc.extract_text(root)
    amap = corpus.abbr_map()
    gold = []
    for ch in tei_doc._foreign_quote_chains(root, tm):
        crefs = [r.get("cRef") for el in ch["els"] for r in el if tei_doc._is_biblical_ref(r)]
        refs = []
        for c in crefs:
            refs.extend(greek.parse_cref(c, amap))
        keys = {k for r in refs for k in corpus.resolve_ref(r)}
        if keys and ch["end"] > ch["start"]:
            gold.append({"start": ch["start"], "end": ch["end"], "keys": keys,
                         "words": len(retrieval.tokenize(tm.text[ch["start"]:ch["end"]], corpus.lang))})
    return tm.text, gold


def files(paths):
    for p in map(Path, paths):
        yield from ([p] if p.is_file() else sorted(p.rglob("*.xml")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--lang", default="grc")
    ap.add_argument("--chunk", type=int, default=3500)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-words", type=int, default=6)
    ap.add_argument("--min-weight", type=float, default=22.0)
    ap.add_argument("--min-precision", type=float, default=0.6)
    a = ap.parse_args()

    corpus = corpora.get_corpus(a.lang)
    t = time.time()
    retrieval.get_index(corpus)
    print(f"index built in {time.time() - t:.1f}s ({retrieval.get_index(corpus).n} verses)")

    tot = {"gold": 0, "cand": 0, "cand_all": 0, "lex_found": 0, "m": 0, "m_ok": 0}
    by_len = {"3-5": [0, 0], "6-10": [0, 0], "11+": [0, 0]}
    t_cand = t_lex = 0.0
    chars = 0
    for f in files(a.paths):
        try:
            text, gold = gold_from_tei(f.read_bytes(), corpus)
        except tei_doc.TeiError:
            continue
        if not gold:
            continue
        chars += len(text)
        # candidates per text part
        t0 = time.time()
        for s, e in greek.chunk_text(text, a.chunk):
            inside = [g for g in gold if g["start"] >= s and g["end"] <= e]
            if not inside:
                continue
            cands = retrieval.candidates(text[s:e], corpus, top=a.top)
            ck = {c.key for c in cands}
            for g in inside:
                tot["gold"] += 1
                ok = bool(g["keys"] & ck)
                tot["cand"] += ok
                b = "3-5" if g["words"] <= 5 else "6-10" if g["words"] <= 10 else "11+"
                by_len[b][0] += ok
                by_len[b][1] += 1
        t_cand += time.time() - t0
        # lexical matches over the whole text, part by part
        t0 = time.time()
        matches = []
        for s, e in greek.chunk_text(text, 12000):
            for m in retrieval.lexical_matches(text[s:e], corpus, a.min_words, a.min_weight, a.min_precision):
                m.start += s
                m.end += s
                matches.append(m)
        t_lex += time.time() - t0
        for g in gold:
            if any(min(m.end, g["end"]) - max(m.start, g["start"]) >= 0.5 * (g["end"] - g["start"])
                   and {k for k in m.keys} & g["keys"] for m in matches):
                tot["lex_found"] += 1
        for m in matches:
            tot["m"] += 1
            if any(min(m.end, g["end"]) - max(m.start, g["start"]) > 0 and set(m.keys) & g["keys"] for g in gold):
                tot["m_ok"] += 1

    g = max(1, tot["gold"])
    print(f"\ngold quotes with a verse in the corpus: {tot['gold']}   ({chars / 1000:.0f}k characters)")
    print(f"candidate recall @top{a.top} (chunk {a.chunk}): {tot['cand'] / g:.1%}")
    for k, (ok, n) in by_len.items():
        if n:
            print(f"   quotes of {k:>4} words: {ok / n:.1%}  (n={n})")
    print(f"lexical matches: {tot['m']}   confirmed by gold: {tot['m_ok']} ({tot['m_ok'] / max(1, tot['m']):.1%})")
    print(f"gold quotes found by the alignment alone: {tot['lex_found'] / g:.1%}")
    print(f"time: candidates {t_cand:.1f}s, matches {t_lex:.1f}s")


if __name__ == "__main__":
    main()
