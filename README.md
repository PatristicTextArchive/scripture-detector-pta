# Scripture Detector PTA

> **Detection and analysis of biblical quotations, paraphrases and allusions in Greek, Latin and Syriac texts — with Gemini, with local models, or by a pure text search against the Bible texts. TEI import and export follow the encoding guidelines of the Patristic Text Archive.**

![Sources page](static/screenshots/sources.png)

This is a fork of [yale-ch/scripture-detector](https://github.com/yale-ch/scripture-detector) by **Annette von Stockhausen** (code mostly written by Claude Sonnet 5.5) for the **[Patristic Text Archive](https://pta.bbaw.de)**. The original application was developed by **Dr. William J.B. Mattingly**, Cultural Heritage Data Scientist at Yale University, for the international workshop **[Ruse of Reuse: Detecting Text-similarity with AI in Historical Sources](https://www.oeaw.ac.at/en/imafo/events/event-details/ruse-of-reuse)** (March 5–6, 2026, Austrian Academy of Sciences, Vienna). The Bible texts and the TEI encoding come from the **[Patristic Text Archive](https://pta.bbaw.de)** (PTA; data: [PatristicTextArchive/pta_data](https://github.com/PatristicTextArchive/pta_data)).

**What this fork adds**

- Bible corpora for **Greek** (Septuagint, New Testament), **Latin** (Vulgate) and **Syriac** (Peshitta); every source has a language.
- **TEI import and export**: detected quotations are written into the original TEI file following the PTA guidelines (`<quote>`, `<seg type="allusion">`, `<ref decls="#biblical" cRef="LXX:Ps:62:9"/>`).
- **Existing markup is respected**: `<quote>` elements already in the file are checked for references and only enriched, never duplicated; quotations that are already annotated are highlighted as *in source*.
- **Local models** instead of Gemini - any OpenAI-compatible server (llama.cpp, vLLM, LM Studio …) or a GGUF file loaded directly, no Ollama needed - and a **text-search-only mode** without any model.
- **Retrieval from the Bible texts (RAG)**: candidate verses in the prompt, correction of wrong references, addition of clear matches, a filter for very short and common phrases.
- Robustness: request rate limiting and retries for Gemini, resuming interrupted runs, handling of small context windows, readable notifications.

---

## Contents

[Overview](#overview) · [Features](#features) · [Quick start](#quick-start) · [Languages and Bible corpora](#languages-and-bible-corpora) · [TEI import and export](#tei-import-and-export) · [Finding quotations with the Bible texts (RAG)](#finding-quotations-with-the-bible-texts-rag) · [AI backends](#ai-backends) · [Configuration](#configuration) · [HTTP API](#http-api) · [Quote classification](#quote-classification) · [Advanced search](#advanced-search) · [Project structure](#project-structure) · [Tests and evaluation](#tests-and-evaluation) · [Known limits](#known-limits) · [Screenshots](#screenshots) · [About](#about) · [License](#license)

---

## Overview

Scripture Detector is a local web application that finds, classifies and annotates biblical references in a text you provide — from verbatim quotations to allusions. It renders an interactive, color-coded view of the source text with side-by-side Bible verse lookup, distribution charts and a manual annotation editor, and exports the result as TEI.

The detection can be done by a **model** (Google Gemini or Vertex AI, or a local model) supported by a **search in the Bible texts**, or by that **text search alone**. A free Gemini API key is enough to start; for work without any external service use a local model or the text-search-only mode (then your texts do not leave your machine). Settings, including an API key, are stored unencrypted in the local database (see [Configuration](#configuration)).

---

## Features

**Analysis**

- Full-document analysis with structured results: verse references, one of four types (full, partial, paraphrase, allusion), whether the passage is introduced as a quotation (*marked*), and a *match* score (share of the quoted words found in the referenced verses).
- Accent-, breathing-, case-, sigma-insensitive location of quotations in the text; invented references are checked against the verse database and dropped.
- Long texts are analysed in parts; interrupted runs continue where they stopped.
- Selection-based re-analysis of a part of the text.
- Support from the Bible texts (RAG): candidate verses in the prompt, repair and addition after the model's answer, strictness setting, or **text search only** without a model.

**Languages and TEI**

- Greek, Latin and Syriac corpora with the PTA book abbreviations and reference prefixes `LXX`/`NA`, `Vg`, `PES`; Syriac is shown right to left.
- TEI import (any TEI P5 file; language from `xml:lang`), export into the original file in the PTA encoding, notes on everything that could not be encoded (with passage, context and location).
- Existing `<quote>` elements are used as hints and enriched; annotations already in the source are highlighted (*in source*) and kept as they are.

**Models**

- Gemini API, Google Vertex AI, local OpenAI-compatible servers, GGUF files (llama-cpp-python), or no model.
- Rate limiting and retries (Gemini), context-window handling and connection test (local).

**Viewer, search and analytics** (from the original application)

- Color-coded in-text highlighting; click a passage to see its annotation card and the matched verses, and vice versa.
- Manual annotation editor with a Bible verse picker (book → chapter → verse).
- Advanced search across all sources by text, book, chapter or verse.
- Distribution charts per source and a global analytics dashboard.

---

## Quick start

### 1. Prerequisites

- Python 3.11.9 or newer
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- One of: a free **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey); a local model (see [AI backends](#ai-backends)); or nothing, for the text-search-only mode

### 2. Install and run

```bash
git clone <repo-url>
cd scripture-detector

# Using uv (recommended)
uv run python app.py

# Or using pip
pip install -e .
python app.py
```

The app starts at **http://127.0.0.1:5001** (`SD_HOST`, `SD_PORT`). The first analysis builds the search index of the language of the text, which takes a few seconds.

### 3. Choose the backend (Settings)

| Backend | What you need | In *Settings* |
|---|---|---|
| **Gemini API** | free key from Google AI Studio | provider *Gemini API*, paste the key |
| **Vertex AI** | Google Cloud project with Vertex AI | provider *Vertex AI*, project ID and location |
| **Local server** | a running `llama-server`, vLLM, LM Studio … | provider *Local server*, URL (e.g. `http://127.0.0.1:8080/v1`) and model name |
| **Local model file** | a GGUF file and `llama-cpp-python` | provider *Local model file*, file path, context size, GPU layers |
| **Text search only** | nothing | provider *Text search only* (finds full and partial quotations of about six words or more, no paraphrases) |

*Save & test connection* checks the chosen backend. Details: [AI backends](#ai-backends).

### 4. Analyze a text

1. **Sources → Add Source**: name and text (Greek, Latin or Syriac; the language is detected from the script) - or **Import TEI** for TEI files.
2. **View**, then **Process with AI** (the button has this name for every backend, also for *Text search only*). The notification tells how many passages were found and what the text search added, corrected or dropped.
3. Click a highlighted passage to see the matched verses; edit or add annotations by hand. Passages that were already annotated in a TEI source are shown hatched (*in source*).
4. **Export TEI** downloads the TEI file with the quotations encoded.

### Docker

```bash
docker build -t scripture-detector .
docker run -p 7860:7860 scripture-detector                        # in-memory database, resets with the container
docker run -p 7860:7860 -v "$(pwd)/data_volume:/app/db" -e SD_DB_DIR=/app/db \
    scripture-detector uv run python app.py                       # persistent database
```

The image listens on port 7860 (`SD_PORT`). An API key is entered in *Settings*; the app does not read a `GEMINI_API_KEY` environment variable. A local server on the host is reachable from the container as `http://host.docker.internal:8080/v1` (Docker Desktop; on Linux add `--add-host=host.docker.internal:host-gateway`).

---

## Languages and Bible corpora

Every source belongs to one language, and the verses, book codes, prompt and PTA abbreviations of the matching corpus are used.

| Language | `xml:lang` | Data | Text | Reference prefix | Licence |
|---|---|---|---|---|---|
| Greek | `grc` | `data/greek` | Septuagint (Rahlfs' numbering) and New Testament, 37,034 verses, 80 books (PTA) | `LXX`, `NA` | CC BY-SA 4.0 |
| Latin | `lat` | `data/vulgate` | Vulgate, 35,809 verses, 73 books (from the original application) | `Vg` | as the original application |
| Syriac | `syc` (`syr`) | `data/syriac` | Peshitta, 38,120 verses, 84 books (PTA) | `PES` | **Old Testament CC BY-NC 4.0 (non-commercial)**, New Testament CC BY-SA 4.0 |

Internal book codes are the lower-cased PTA abbreviations (`gn`, `ps`, `mt`, `1cor` …) in the Greek and Syriac corpora; the Latin corpus keeps the codes of the original application (`gen`, `matt` …) and has the PTA abbreviation in its `book_mapping.tsv`.

### Greek

`data/greek/` is generated by `scripts/build_greek_bible.py` from the Greek Bible texts of the
[Patristic Text Archive](https://github.com/PatristicTextArchive/pta_data) (`data/pta9999/*/*-grc1.xml`; 80 books, 37,034 verses,
LXX in Rahlfs' numbering, NT). These texts are licensed **CC BY-SA 4.0** (Berlin-Brandenburg Academy of Sciences and Humanities), so `data/greek/` is under the same licence; the code stays MIT. Internal book codes are the lower-cased PTA abbreviations (`gn`, `ps`, `mt`, `1cor` …).

```bash
git clone --depth 1 https://github.com/PatristicTextArchive/pta_data.git /tmp/pta_data
python scripts/build_greek_bible.py --pta-data /tmp/pta_data      # or: --download
```

Only the primary text form (`grc1`) of each book is loaded. Alternative forms (`grc2`: Judges B, Tobit S, Old Greek Daniel/Susanna/Bel) are not.

### Latin (Vulgate)

The Vulgate verse database comes from the original application; this fork added the PTA abbreviations and the corpus prefix `Vg` to its `book_mapping.tsv` (Nehemiah is `Neh`), a `meta.json` with the corpus description and a proposal of introductory formulas (*scriptum est*, *ait*, *inquit* …) that should be checked by someone who reads Latin. `main.py` (batch evaluation) still works on this corpus.

### Syriac (Peshitta)

`data/syriac/` is generated by `scripts/build_syriac_bible.py` from the `*-syc1.xml` files of the same PTA repository (`data/pta9999`): 84 books, 38,120 verses; the book codes are the PTA abbreviations of the Greek corpus plus `neh`, `prman`, `4esr` and `2bar` (Syriac Apocalypse of Baruch together with the Letter of Baruch, chapters 78-86); the corpus prefix `Peshitta` is the one the PTA guidelines use for the Peshitta. Chapters and verses are those of the Peshitta files (Psalms: Hebrew numbering). Variant text forms (`syc2` to `syc4`, e.g. Psalms 151-155) are not included.

> **Licence of the Syriac data: the Old Testament and deuterocanonical books come from the ETCBC Peshitta project and are licensed CC BY-NC 4.0 - for non-commercial use only. The New Testament (Gwilliam, Pinkerton and Gwynn; TEI edition by James E. Walters) is stated as CC BY-SA 4.0.** The licences of every book are listed in `data/syriac/LICENSES.md`. Please check that your use is covered before you pass the data on.

The Old Testament text is consonantal, the New Testament text is vocalised; the search ignores vowel points and removes the attached particles *w-, b-, d-, l-, k-* (at most two) before it cuts a word to four letters, so that vocalised and unvocalised texts and words with a different particle match. The viewer shows Syriac right to left with a Syriac font (Estrangelo Edessa, Noto Sans Syriac, Serto Jerusalem if installed). The introductory formulas for Syriac (and Latin) in `meta.json` are proposals and should be checked by someone who reads the language. *Evaluation is much thinner than for Greek:* the PTA contains hand-made `PES:` references in only two files, 25 passages in all (21 different verses or groups of verses). With these, the verse is among the 20 candidates for 72 % of the passages (the number is small); in a synthetic test with 300 Peshitta verses written without vowel points, with changed particles, dropped words and partial quotations, the verse was among the 25 candidates in 99.7 % and was reported by the text comparison in 85 % of the cases; and in a manual sample of 14 matches in Syriac patristic text, 12 were genuine near-verbatim quotations of the Peshitta, one a doxology/colophon and one only loosely related. The synthetic test uses variants of the Peshitta itself, real quotations deviate more.

### Language of a source

Every source belongs to one language, and verses, book codes, prompt and PTA abbreviations come from the matching corpus: `grc` -> `data/greek` (cRef prefixes `LXX` / `NA`), `lat` -> `data/vulgate` (prefix `Vg`, e.g. `Vg:Gn:1:1`), `syc` -> `data/syriac` (prefix `PES`, e.g. `PES:Gn:1:1`). For TEI files the language is the `xml:lang` of the **first `<div>` in `<text>`** (`grc`, `lat` or `syc`; `syr` is accepted as Syriac; other languages are rejected on import with a message). Pasted text is detected from its script, or can be set with `"lang"` in `POST /api/sources`. The language is shown as a small badge in the source list, and the reference picker in the viewer uses the corpus of the source.

---

## TEI import and export

**Import TEI.** *Sources → Import TEI* (or `POST /api/import/tei`, or a ZIP) accepts any TEI P5 file. The running text of `<text>` is extracted: `<lb break="no"/>` joins words, other `<lb/>`/`<pb/>` become spaces, `<choice>` uses `expan`/`reg`/`corr`, `<app>` uses `<lem>`, `<note>`, `<rdg>`, `<del>` and `<div type="praefatio">` are skipped. The original XML is kept in the database. Annotations written by an earlier export are restored; biblical markup that is already in the file is shown separately as *in source* (see *Annotations already in the source* below) and is left untouched.

**Export TEI (PTA encoding guidelines).** Results are written **into the original file** following the
[PTA guidelines](https://pta.bbaw.de/en/project/encoding-guidelines#encoding-of-biblical-and-other-quotations-and-allusions):

```xml
<quote type="marked" xml:id="sd-q12">Ἐκολλήθη ἡ ψυχή μου ὀπίσω σου<ref decls="#biblical" cRef="LXX:Ps:62:9"/></quote>
<seg type="allusion" xml:id="sd-q13">…<ref decls="#biblical" cRef="NA:Mt:5:9"/></seg>
```

* `full`/`partial` → `<quote>` (`@type` `marked`/`unmarked`, according to whether the model saw an introductory formula), `paraphrase` → `<quote type="paraphrasis">`, `allusion` → `<seg type="allusion">`.
* `cRef` is `LXX:`/`NA:`/`Vg:` + PTA book abbreviation + chapter + verses. Consecutive verses become **one** `<ref>` with a range (`phil_2:8`, `phil_2:9`, `phil_2:10` → `NA:Phil:2:8-10`), also when they come from several annotations of the same existing `<quote>`; verses that do not follow each other are joined with a dot (`LXX:Ps:62:9-10.12`); different chapters or books get separate `<ref>`s.
* Quotations crossing element boundaries (`<hi>`, `<lb/>`, `<pb/>`, paragraphs, …) are fragmented and linked with `@prev`/`@next`.
* The header receives `<refsDecl xml:id="biblical">`, the `annotationstatus` taxonomy, `<p xml:id="biblical-quotations">` and a `<change>` entry. The annotation status is declared as *partially annotated*; adjust it after review.
* Pasted plain text is exported in a minimal PTA skeleton (`<div type="edition">` with a placeholder CTS URN to replace).
* Per PTA rules `<quote>` is not used in manuscript transcriptions (`…-Ms…`); the export warns in that case.
* **Existing `<quote>` elements.** Passages already marked as `<quote>` in the file are passed to the model as hints ("check these for biblical references"). If the model finds a reference for one, **no further quote element is created**: the existing `<quote>` only receives the `<ref decls="#biblical" cRef="…" xml:id="sd-r…"/>` (as last child of the last fragment, if it is split with `@prev`/`@next`) and a `@type` (`marked`/`unmarked`/`paraphrasis`) if it has none; an existing `@type` is never changed. A detected passage is matched to an existing quote if they overlap by at least 50 % in both directions; if it only overlaps in part, it is skipped with a warning. Quotes that already carry a hand-made biblical `<ref>` are left untouched. Re-importing an exported file restores the references as annotations, and re-exporting replaces them.
* **Introductory formulas.** The prompt lists typical introductions (Greek: γέγραπται, φησί(ν), λέγει, εἶπεν, ἔφη, ἡ γραφὴ λέγει, κατὰ τὸ γεγραμμένον, ὥς φησι, βοᾷ, διδάσκει, παραινεῖ, ψάλλει, ψαλμῳδεῖ, μαρτυρεῖ, κηρύττει, ἀναφωνεῖ, διαμαρτύρεται; Latin: scriptum est, dicit scriptura, ait, inquit, psallit …). They are signals for a nearby quotation and for `marked`; the formula itself is not part of the quoted text. The lists, the corpus description and further hints are in `data/<corpus>/meta.json` and can be edited.
* **Annotations already in the source file.** `<quote>` elements with a biblical `<ref decls="#biblical" cRef="…"/>` (and `<seg type="allusion">` with one), made by hand as in the PTA editions, are read on import and shown in the viewer in a different style - grey hatching with a dashed underline, an *in source* badge on the card, and a tooltip with the reference; a checkbox in the legend (*Already in source*) hides them. They are **read-only**: they cannot be edited or deleted in the app (HTTP 409), are not deleted when a source is processed again, and are left untouched by the export. Detected passages that overlap one of them (at least half of the shorter span) are skipped, and the viewer says how many. References the corpus does not know (e.g. `Peshitta:Gn:1:1`) are shown as written. References added by this tool earlier (`xml:id="sd-r…"`) are not "in source": they stay editable.
* Elements written by the tool carry `xml:id="sd-…"`; re-exporting replaces them instead of duplicating them. If a quotation is already annotated by hand, the new one is skipped (with a warning).

**Notifications.** Messages are shown by `static/ui.js`: toasts stack, wrap, stay longer the longer they are (warnings and errors 1.6x), pause under the pointer and can be dismissed. Warnings of the TEI export and failed imports are additionally listed completely in a dialog that stays open until closed (with *Copy all*); identical messages are merged (`… (×3)`). Each export warning names the reason first, then the passage with context and its location: `…before «passage» after… [char 486, section 3.12, <p> 2, page 14v, line 9, xml:id "par7"]` (section = `div/@n` path, page/line = nearest preceding `pb/@n` / `lb/@n`). `GET /api/sources/<id>/export/tei?json=1` returns `{filename, xml, warnings}`.

---

## Finding quotations with the Bible texts (RAG)

The model does not have to guess references from memory: before each request the text part is compared with *all* verses of the corpus (`retrieval.py`). The comparison is lexical and ignores accents, breathings, case, sigma and Latin u/v, i/j; words are cut to five-letter stems, so inflected forms match. Rare stems and word pairs are looked up in an inverted index (built on first use, a few seconds and about 100 MB per corpus), a sliding window over the text collects candidate verses, and an alignment of each candidate with the text ranks them.

* **Master switch** (*Settings → Bible Text Search*, default on): with a model, uncheck *Use the Bible texts to support the model* and the model works entirely on its own - no candidate verses, nothing added, corrected or dropped.
* **Candidates in the prompt:** the best verses, each with the passage of the text it matches, are listed in the prompt (at most 25 for local models, 50 for Gemini, within a character budget; fewer if few verses match well enough). The prompt tells the model that short phrases and common wording are *not* quotations. This helps small local models most.
* **Strictness** (*strict / normal / lenient*, default *normal*) decides how closely a passage must follow a verse. Candidates must contain a minimum number of aligned words with sufficient weight (rare words count more) and must not be a phrase that occurs in many verses; *normal* needs four words, *strict* five, *lenient* has no minimum.
* **After the model has answered** (can be switched off separately): results that are only a very short phrase (two words; three in *strict*) or a common phrase (fewer than eight words that occur in more than 8 verses; 5 in *strict*) are **dropped** - this applies to what the model reports as a verbatim (full/partial) quotation, never to paraphrases and allusions; short results (up to four words) are kept only if the verse confirms them (match score of at least 0.75) and the phrase is rare. A reference the model got wrong (low *match* score, and the passage matches another verse) is **replaced** by the verse the passage actually matches. Passages that clearly reuse a verse (six or more matching words in *normal*, eight in *strict*; sufficient weight; not a formula that occurs in more than five verses) are **added** even if the model did not report them; consecutive verses become one annotation. The model's own judgement (type, `marked`) is kept; for added passages `marked` is set if one of the introductory formulas stands just before. After each run the viewer reports what was added, corrected and dropped.
* **Text search only** (*Settings → AI provider*): no model at all. Full and partial quotations of about six words or more are found by the comparison alone; paraphrases and allusions need a model.

*Measured* with `python scripts/eval_retrieval.py <TEI files>` on 18 PTA editions with 2,601 hand-annotated biblical quotations (1.28 million characters). Share of the annotated quotations whose verse is among the candidates of a text part (3,500 characters), and how many candidates that costs:

| strictness | recall | of 3-4 word quotations | candidates per text part | candidates that are annotated verses |
|---|---|---|---|---|
| lenient | 84 % | 69 % | 25 | 24 % |
| normal | 75 % | 55 % | 12 | 43 % |
| strict | 66 % | 47 % | 9 | 52 % |

Stricter settings thus remove many false candidates, but also miss short quotations; very short quotations cannot be told apart lexically anyway. The alignment alone finds about 28 % of the annotated quotations (the shorter ones are out of reach) and reports further passages: in a manual sample of 26 of them, 24 were genuine reuse that the editors had not marked up and 2 were formulas (these are now filtered); the "confirmed by gold" figure of the script (40 %) is therefore only a lower bound of the precision. None of this measures a model. Please check how a given model behaves on your own texts, and adjust the strictness.

---

## AI backends

### Gemini API and Vertex AI

The simplest option is a free key from [Google AI Studio](https://aistudio.google.com/apikey) (no billing required): select **Gemini API** in Settings and paste it. For enterprise use or higher rate limits select **Vertex AI** and enter the project ID and location of a Google Cloud project with Vertex AI enabled. The model is chosen in Settings (and in the viewer).

**Rate limits** (Gemini / Vertex only). Long texts are analysed in parts of about 12,000 characters (`SD_CHUNK_CHARS`), one Gemini request each. To stay within the per-minute quota of your model, requests are spaced by *Settings → Max requests per minute* (default 5, `SD_MAX_RPM` as fallback, 0 = unlimited). When the API still answers 429/503, the app waits for the delay the API suggests ("retry in 53s") or backs off exponentially (up to 6 attempts) and retries. A used-up *daily* quota is reported as such and not retried. If the limit does not clear, the parts that were already analysed are kept in memory: the next run continues with the remaining parts instead of repeating requests, and nothing is written to the database before the run is complete. The error message says how many parts were done. A deliberate re-run after a successful run asks the model again. The code is in `gemini_limits.py`.

### Local models

In *Settings* choose, instead of Gemini/Vertex:

* **Local server** - any OpenAI-compatible endpoint (`/v1/chat/completions`): llama.cpp's `llama-server` (`llama-server -m Qwen3-14B-Q4_K_M.gguf --port 8080 -c 16384` → `http://127.0.0.1:8080/v1`), vLLM (`vllm serve Qwen/Qwen3-14B` → `http://127.0.0.1:8000/v1`), LM Studio (`http://127.0.0.1:1234/v1`), SGLang, LocalAI … The model name is required by vLLM/LM Studio and optional for `llama-server`. The server is asked to follow a JSON schema; if it refuses `response_format`, the app continues without (and remembers that).
* **Local model file** - a GGUF file (Qwen, Gemma, Llama, Mistral …) loaded in-process with `llama-cpp-python`, no server: `pip install "scripture-detector[local]"` or `pip install llama-cpp-python` (GPU: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`, Metal: `-DGGML_METAL=on`). Set the file path, context size (`n_ctx`, default 16384 - the prompt with the candidate verses of the Bible text search plus the answer needs room; lower it if memory is short) and the number of GPU layers.

*Save & test connection* sends a tiny request with the saved settings. Notes: local models get a smaller default part size (about 3,500 characters, `SD_LOCAL_CHUNK_CHARS` or *Settings → Text Parts*) because of their context window; there is no request rate limit; the model's answer is parsed tolerantly (markdown fences, `<think>` blocks and truncated JSON are handled), and if an answer is cut off at the token limit, the text part is split in two (with a small overlap) and both halves are analysed. The model picker in the viewer is replaced by the name of the local model. Smaller models miss passages and invent references; invented verses are removed by the verse check, and the *match* score helps to spot doubtful ones. Please test a new model on a text you know before trusting its results.

**Context window of local models.** A request consists of the instructions and the book list (about 1,300 tokens), the candidate verses and the text part. For polytonic Greek this is roughly 4,500-6,000 tokens (Greek needs many tokens per word), plus the answer. Problems usually come from a context window that is smaller than expected: **many `llama-server` versions divide the context given with `-c` among their parallel slots (`-np`)**, so `-c 16384` with four slots leaves only 4,096 tokens per request. Start it with `-np 1` (e.g. `-c 16384 -np 1`). The app helps in three ways: *Save & test connection* reads the context per request from the server (`GET /props`, llama.cpp) and warns if it is below 8,192; the answer limit is capped to a third of that context; and if a request still does not fit (llama.cpp, vLLM and LM Studio error messages are recognised), the app first shortens the candidate list, then splits the text in halves, and only reports an error (naming `-np 1` / `-c`) if even a short request does not fit. The log lines of `llama-server` about the *prompt cache* ("selected slot by LCP similarity", "making room for prompt cache entry") are informational and do not indicate an error.

### Text parts

Long texts are sent in parts: about 12,000 characters for Gemini / Vertex (`SD_CHUNK_CHARS`) and about 3,500 for local models (`SD_LOCAL_CHUNK_CHARS`), or the value of *Settings → Text Parts*. If a local model's answer is cut off at the token limit, or a request does not fit into the context window, the part is split and analysed again.

---

## Configuration

**Settings** (stored in the SQLite database; *Settings* page):

| Setting | Key | Default |
|---|---|---|
| AI provider | `api_provider` | `gemini` (`vertex`, `openai` = local server, `llamacpp` = local model file, `rag` = text search only) |
| Gemini API key | `gemini_api_key` | - |
| Vertex AI project / location | `vertex_project_id`, `vertex_location` | - / `global` |
| Gemini model | `model` | `gemini-3-pro-preview` |
| Max requests per minute (Gemini / Vertex; 0 = unlimited) | `max_rpm` | 5 |
| Local server: URL, model name, key, JSON constraint, timeout | `local_base_url`, `local_model`, `local_api_key`, `local_structured`, `local_timeout` | `http://127.0.0.1:8080/v1`, -, -, `auto`, 600 s |
| Local model file: path, context size, GPU layers | `gguf_path`, `n_ctx`, `n_gpu_layers` | -, 16384, -1 |
| Max. answer length of local models (tokens) | `local_max_tokens` | 4096 |
| Characters per request | `chunk_chars` | 12,000 (Gemini / Vertex), 3,500 (local) |
| Bible text search: master switch, add/correct/drop, strictness, candidates per part | `rag`, `rag_add`, `rag_strictness`, `rag_top_k` | `on`, `on`, `normal`, 50 (Gemini) / 25 (local) |

**Environment variables**

| Variable | Meaning | Default |
|---|---|---|
| `SD_HOST`, `SD_PORT` | address and port of the app | `127.0.0.1`, `5001` (Docker image: `7860`) |
| `SD_DB_DIR` | directory of the database file (`scripture_detector.db`) | next to `app.py` |
| `--cache-db` / `SCRIPTURE_DETECTOR_CACHE_DB` | in-memory database per browser session (nothing is written to disk; the Docker default) | off |
| `SD_SECRET_KEY`, `SD_BEHIND_PROXY` | session key and proxy mode for deployments | random key |
| `SD_SESSION_TTL` | seconds of inactivity after which an in-memory session database (`--cache-db`) is discarded | `3600` |
| `SD_CHUNK_CHARS`, `SD_LOCAL_CHUNK_CHARS` | default characters per request | 12000, 3500 |
| `SD_MAX_RPM` | default for *Max requests per minute* | 5 |

The settings, **including an API key, are stored unencrypted** in the database, and `GET /api/settings` returns them to the browser. The app has no login: do not expose it on a public network. The database file is not part of the repository (`*.db` is ignored).

---

## HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /api/sources` | create a source `{name, text, lang?}`; the language is detected if not given |
| `GET /api/sources/<id>`, `DELETE` | source with text segments and annotations (including `origin`: `tool` or `source`), delete |
| `POST /api/import/tei` | import TEI files (multipart `files`); `POST /api/import/zip` for a ZIP |
| `GET /api/sources/<id>/export/tei` | TEI with the quotations encoded; `?json=1` returns `{filename, xml, warnings}`; `GET /api/export/zip` exports all sources |
| `POST /api/sources/<id>/process`, `.../process-selection` `{start, end}` | run the detection; answer `{count, rag: {added, repaired, dropped, in_source}}`; HTTP 429 `{error, retryable}` if a rate limit does not clear |
| `POST /api/quotes`, `PUT`/`DELETE /api/quotes/<id>` | manual annotations (HTTP 409 for annotations that come from the source file) |
| `GET /api/settings`, `POST /api/settings`, `POST /api/test-backend` | settings and connection test |
| `GET /api/bible/books`, `/api/bible/<book>/chapters`, `/api/bible/<book>/<chapter>/verses`, `/api/book-mapping` | Bible structure; `?lang=grc|lat|syc` selects the corpus |
| `POST /api/search`, `GET /api/dashboard`, `GET /api/sources/<id>/distribution` | search and statistics |

---

## Quote Classification

| Type | Description |
|---|---|
| **Full** | A complete or near-complete verse quoted verbatim |
| **Partial** | A recognizable portion of a verse with minor variation or truncation |
| **Paraphrase** | Biblical content restated in different words, preserving the meaning |
| **Allusion** | A brief phrase, thematic echo, or indirect reference to a specific verse |

---

Independently of the type, every passage carries a flag *marked* (introduced by a formula such as *γέγραπται*, *φησίν*, *scriptum est*, *ܟܬܝܒ*), which becomes `@type="marked"` / `"unmarked"` in the TEI export.

---

## Advanced Search

The Sources page includes a real-time multi-filter search system:

| Filter Type | What it matches |
|---|---|
| **Text Content** | Any source whose full text contains the search string |
| **Bible Book** | Any source with at least one reference to the chosen book |
| **Chapter** | Any source with a reference to the chosen chapter (e.g. Psalms 23) |
| **Verse** | Any source with a reference to the exact chosen verse (e.g. John 3:16) |

Filters can be stacked in any combination. The **AND** mode requires a source to satisfy *every* filter; **OR** mode returns sources satisfying *any* filter. Results update within ~350 ms of each keystroke or dropdown change. Matched text snippets and verse citations appear as inline evidence on each result card.

---

---

## Project structure

```
scripture-detector/
├── app.py                 # Flask application, API routes, detection pipeline
├── database.py            # SQLite layer (sources, quotes, settings)
├── corpora.py             # Bible corpora per language (grc / lat / syc), language detection
├── greek.py               # accent-insensitive span search, scoring, chunking, PTA cRef conversion
├── retrieval.py           # search in the Bible texts: candidates, text matches, strictness
├── tei_doc.py             # TEI reading and PTA-style export
├── llm_backends.py        # local backends (OpenAI-compatible server, llama-cpp-python), answer parsing
├── gemini_limits.py       # rate limiting and retries for Gemini
├── main.py                # CLI batch evaluation (Latin / Vulgate, from the original application)
├── scripts/
│   ├── build_greek_bible.py    # builds data/greek from the PTA repository
│   ├── build_syriac_bible.py   # builds data/syriac from the PTA repository
│   └── eval_retrieval.py       # measures the search against hand-made references in TEI files
├── tests/                 # pytest
├── data/
│   ├── greek/             # bible.tsv, book_mapping.tsv, meta.json
│   ├── vulgate/           # same files, Latin
│   └── syriac/            # same files, Syriac, plus LICENSES.md
├── templates/             # sources, viewer, dashboard, settings, about
└── static/                # style.css, ui.js (notifications), logo, favicon, screenshots
```

Each corpus directory has a `bible.tsv` (verses), a `book_mapping.tsv` (book codes, names, PTA abbreviation and corpus prefix) and a `meta.json` (corpus description for the prompt, introductory formulas, notes).

---

## Tests and evaluation

```bash
pip install pytest        # or: pip install -e ".[dev]"
pytest tests              # about 110 tests; no model or network needed (backends are simulated)
python scripts/eval_retrieval.py FILE_OR_DIR [...] [--lang grc|lat|syc]   # search against hand-made references in TEI files
```

The tests simulate Gemini and the local servers; no real model was used. The figures given in this README for the Bible text search come from `scripts/eval_retrieval.py` and from synthetic tests on the PTA data and describe the search, not a model.

---

## Known limits

- Detection quality depends on the model; the *match* score is a plausibility hint, not proof. Small local models miss passages and invent references (invented verses are removed, but not every wrong one). Try a new model on a text whose quotations you know.
- Very short quotations (two to four words) and common phrases cannot be told apart by wording; the text search is deliberately strict with them (*strictness* setting).
- Only the primary text form of each Greek book (`grc1`) and of each Syriac book (`syc1`) is loaded; variant forms (e.g. Judges B, Old Greek Daniel, Psalms 151-155 in Syriac) are not.
- Verses with letters (`2a`) are exported as the plain verse number; titles and prologues (Odes, Sirach `p`/`t`) cannot be expressed in the PTA `cRef` and are skipped. Quotations spanning several *whole* paragraphs are only marked at their boundaries (with a warning).
- The PTA guidelines do not use `<quote>` in manuscript transcriptions; the export warns in that case.
- The evaluation of the Syriac search rests on very little hand-made data (see above). The introductory formulas for Latin and Syriac are proposals.
- No login and no encryption of stored settings: for use on your own machine or in a trusted network.

---

## Screenshots

### Sources Page — with Advanced Search
Manage all your text sources. Use the search bar to search source content in real time, or open the **Advanced** panel to stack filters by Bible book, chapter, or verse with AND / OR logic. Match evidence appears directly on each result card.

![Sources page](static/screenshots/sources.png)

### Text Viewer
Color-coded annotations appear directly in the text. Click any highlighted passage to see the matched Bible verse(s) in the panel on the right.

![Viewer page](static/screenshots/viewer.png)

### Analytics Dashboard
Explore scripture distribution across all sources — broken down by Bible book, testament, and quote type.

![Dashboard](static/screenshots/dashboard.png)

### Settings
Configure your Gemini API key or Google Vertex AI credentials, and select which Gemini model to use.

![Settings page](static/screenshots/settings.png)

### About Page
Full documentation of the application, its features, and how it works.

![About page](static/screenshots/about.png)


---

## About

**Original Developer:** Dr. William J.B. Mattingly  
**Affiliation:** Yale University, Cultural Heritage Data Scientist  
**Workshop:** [Ruse of Reuse: Detecting Text-similarity with AI in Historical Sources](https://www.oeaw.ac.at/en/imafo/events/event-details/ruse-of-reuse)  
**Workshop Dates:** March 5–6, 2026  
**Workshop Venue:** Austrian Academy of Sciences, PSK Georg-Coch-Platz 2, 1010 Vienna  
**Organisers:** Digital Lab, Institute for Medieval Research, Austrian Academy of Sciences & [SOLEMNE](https://canones.org/), Radboud University  

---

---

## License

The code is licensed under the MIT license. The Bible data have their own licences: `data/greek` CC BY-SA 4.0 (Patristic Text Archive); `data/syriac` CC BY-NC 4.0 for the Old Testament and deuterocanonical books (ETCBC, **non-commercial use only**) and CC BY-SA 4.0 for the New Testament (see `data/syriac/LICENSES.md`); `data/vulgate` comes from the original Scripture Detector.
