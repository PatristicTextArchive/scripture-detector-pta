import json
import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

PROJECT_ID = "cultural-heritage-gemini"
LOCATION = "global"
MODEL = "gemini-3-pro-preview"

PROJECT_ROOT = Path(__file__).resolve().parent
PROBLEMS_DIR = PROJECT_ROOT / "data" / "task" / "problems"
SOLUTIONS_DIR = PROJECT_ROOT / "data" / "task" / "solutions"
REFERENCE_MAPPING_PATH = PROJECT_ROOT / "data" / "reference_mapping.json"
BIBLE_TSV_PATH = PROJECT_ROOT / "data" / "vulgate" / "bible.tsv"
OUTPUT_DIR = PROJECT_ROOT / "output"

console = Console()


def load_reference_mapping() -> dict[str, str]:
    with open(REFERENCE_MAPPING_PATH) as f:
        return json.load(f)


def load_problem(problem_id: str) -> str:
    return (PROBLEMS_DIR / f"{problem_id}.txt").read_text(encoding="utf-8")


def load_solution(problem_id: str) -> list[dict]:
    with open(SOLUTIONS_DIR / f"{problem_id}.json") as f:
        return json.load(f)


def get_valid_book_codes() -> list[str]:
    codes: set[str] = set()
    with open(BIBLE_TSV_PATH, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            codes.add(row["book_code"].strip().lower())
    return sorted(codes)


def build_prompt(text: str, valid_book_codes: list[str], ref_mapping: dict[str, str]) -> str:
    codes_str = ", ".join(valid_book_codes)
    mapping_lines = "\n".join(f"  {k} -> {v}" for k, v in sorted(ref_mapping.items()))
    return f"""You are an expert in medieval Latin texts and the Latin Vulgate Bible.

Given the following Latin text from a Carolingian-era ecclesiastical document, identify ALL scriptural (Biblical) quotations, partial quotations, paraphrases, and clear allusions to specific Bible verses.

For each identified passage:
1. Extract the EXACT text as it appears in the document — preserve the original spelling, punctuation, and word order verbatim.
2. Identify the specific Bible verse(s) being quoted or referenced.
3. Classify the type of reuse as one of:
   - "full"       — a complete or near-complete verse quoted verbatim from the Vulgate.
   - "partial"    — a recognisable portion of a verse, quoted with minor variation or truncation.
   - "paraphrase" — the biblical content is clearly restated in different words while preserving the meaning.
   - "allusion"   — a brief phrase, thematic echo, or indirect reference to a specific verse without quoting or restating it.

Reference format: book_chapter:verse  (e.g. matt_5:9, ps_82:14, 1cor_15:33, dan_4:14)
CRITICAL: Each reference must be a SINGLE verse. Never use ranges like matt_15:1-2.
Instead, list each verse separately: matt_15:1, matt_15:2.

Valid book codes: {codes_str}

Common abbreviation-to-code mapping (for your reference):
{mapping_lines}

Important:
- Include both direct quotes and partial quotes / paraphrases / allusions.
- A single passage may reference multiple Bible verses — list all of them.
- Use the Vulgate Latin text as your primary reference for identifying quotes.
- Be thorough — identify even brief allusions to specific verses.
- For Psalms, use the Vulgate / LXX numbering (which may differ from Hebrew numbering by 1).
- The extracted text must be a verbatim substring of the input document.

TEXT:
{text}"""


def extract_quotes_with_gemini(
    text: str,
    valid_book_codes: list[str],
    ref_mapping: dict[str, str],
) -> list[dict]:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    prompt = build_prompt(text, valid_book_codes, ref_mapping)

    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "text": {
                    "type": "STRING",
                    "description": (
                        "The exact text of the scriptural quote or allusion "
                        "as it appears verbatim in the document"
                    ),
                },
                "resolved_references": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": (
                        "List of Bible verse references in format "
                        "book_chapter:verse (e.g. matt_5:9)"
                    ),
                },
                "quote_type": {
                    "type": "STRING",
                    "enum": ["full", "partial", "paraphrase", "allusion"],
                    "description": (
                        "full = complete verse quoted verbatim, "
                        "partial = recognisable portion with minor variation, "
                        "paraphrase = biblical content restated in different words, "
                        "allusion = brief phrase or thematic echo"
                    ),
                },
            },
            "required": ["text", "resolved_references", "quote_type"],
        },
    }

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    )

    quotes = json.loads(response.text)
    for q in quotes:
        q["resolved_references"] = expand_range_references(q.get("resolved_references", []))
    return quotes


def find_spans(text: str, quotes: list[dict]) -> list[dict]:
    results = []
    for quote in quotes:
        qt = quote["text"]
        idx = text.find(qt)
        if idx == -1:
            idx = text.lower().find(qt.lower())
        span_start = idx if idx != -1 else None
        span_end = (idx + len(qt)) if idx != -1 else None
        results.append({
            "text": qt,
            "span_start": span_start,
            "span_end": span_end,
            "resolved_references": quote["resolved_references"],
            "quote_type": quote.get("quote_type", "allusion"),
        })
    return results


_RANGE_RE = re.compile(r"^(.+_\d+):(\d+)-(\d+)$")


def expand_range_references(refs: list[str]) -> list[str]:
    expanded: list[str] = []
    for ref in refs:
        m = _RANGE_RE.match(ref.strip())
        if m:
            prefix, start, end = m.group(1), int(m.group(2)), int(m.group(3))
            for v in range(start, end + 1):
                expanded.append(f"{prefix}:{v}")
        else:
            expanded.append(ref.strip())
    return expanded


def normalize_reference(ref: str) -> str:
    return ref.strip().lower()


def build_predictions(problem_id: str, quotes: list[dict]) -> list[dict]:
    predictions = []
    for quote in quotes:
        for ref in quote.get("resolved_references", []):
            predictions.append({
                "problem_id": problem_id,
                "reference": normalize_reference(ref),
                "text": quote.get("text", ""),
            })
    return predictions


def load_ground_truth(problem_id: str) -> dict[str, list[str]]:
    solution = load_solution(problem_id)
    refs: set[str] = set()
    for item in solution:
        for ref in item.get("resolved_references", []):
            refs.add(normalize_reference(ref))
    return {problem_id: sorted(refs)}


def score_predictions(
    predictions: list[dict],
    ground_truth_by_problem: dict[str, list[str]],
) -> dict:
    pred_pairs: set[tuple[str, str]] = set()
    for row in predictions:
        pid = str(row.get("problem_id", "")).strip()
        ref = normalize_reference(row.get("reference", ""))
        if pid and ref:
            pred_pairs.add((pid, ref))

    true_pairs: set[tuple[str, str]] = set()
    for problem_id, refs in ground_truth_by_problem.items():
        for ref in refs:
            true_pairs.add((problem_id, normalize_reference(ref)))

    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_pairs": pred_pairs,
        "true_pairs": true_pairs,
    }


def display_results(
    problem_id: str,
    quotes_with_spans: list[dict],
    metrics: dict,
    ground_truth: dict[str, list[str]],
) -> None:
    console.print()
    console.print(
        Panel(
            f"[bold]{TEAM_NAME}[/bold]  |  Problem: [cyan]{problem_id}[/cyan]  |  Model: [green]{MODEL}[/green]",
            title="Ruse of Reuse — Scriptural Quote Detection",
            border_style="blue",
        )
    )

    qt = Table(title="Extracted Quotes", box=box.ROUNDED, show_lines=True)
    qt.add_column("#", style="dim", width=3)
    qt.add_column("Text", style="white", max_width=70)
    qt.add_column("Type", style="magenta", width=8)
    qt.add_column("References", style="cyan")
    qt.add_column("Span", style="yellow")

    type_colors = {"full": "green", "partial": "yellow", "paraphrase": "cyan", "allusion": "red"}
    for i, q in enumerate(quotes_with_spans, 1):
        span = (
            f"{q['span_start']}–{q['span_end']}"
            if q["span_start"] is not None
            else "[red]NOT FOUND[/red]"
        )
        refs = ", ".join(q["resolved_references"])
        t = q["text"]
        display = (t[:67] + "...") if len(t) > 70 else t
        qtype = q.get("quote_type", "allusion")
        tc = type_colors.get(qtype, "white")
        qt.add_row(str(i), display, f"[{tc}]{qtype}[/{tc}]", refs, span)
    console.print(qt)

    mt = Table(title="Evaluation Metrics", box=box.DOUBLE_EDGE)
    mt.add_column("Metric", style="bold")
    mt.add_column("Value", justify="right")
    f1c = "green" if metrics["f1"] >= 0.7 else "yellow" if metrics["f1"] >= 0.4 else "red"
    mt.add_row("True Positives", f"[green]{metrics['true_positives']}[/green]")
    mt.add_row("False Positives", f"[red]{metrics['false_positives']}[/red]")
    mt.add_row("False Negatives", f"[red]{metrics['false_negatives']}[/red]")
    mt.add_row("Precision", f"{metrics['precision']:.4f}")
    mt.add_row("Recall", f"{metrics['recall']:.4f}")
    mt.add_row("F1 Score", f"[{f1c}]{metrics['f1']:.4f}[/{f1c}]")
    console.print(mt)

    pred_refs = {ref for _, ref in metrics["pred_pairs"]}
    true_refs = {ref for _, ref in metrics["true_pairs"]}

    ct = Table(title="Reference Comparison", box=box.ROUNDED, show_lines=True)
    ct.add_column("Reference", style="white")
    ct.add_column("Status", justify="center")

    for ref in sorted(pred_refs | true_refs):
        in_pred = ref in pred_refs
        in_true = ref in true_refs
        if in_pred and in_true:
            status = "[green]TP (correct)[/green]"
        elif in_pred:
            status = "[red]FP (spurious)[/red]"
        else:
            status = "[yellow]FN (missed)[/yellow]"
        ct.add_row(ref, status)
    console.print(ct)


def process_single(problem_id: str, valid_book_codes: list[str], ref_mapping: dict[str, str]) -> dict:
    text = load_problem(problem_id)
    quotes = extract_quotes_with_gemini(text, valid_book_codes, ref_mapping)
    quotes_with_spans = find_spans(text, quotes)
    predictions = build_predictions(problem_id, quotes)
    ground_truth = load_ground_truth(problem_id)
    metrics = score_predictions(predictions, ground_truth)

    OUTPUT_DIR.mkdir(exist_ok=True)
    serialisable_metrics = {
        k: v for k, v in metrics.items() if k not in ("pred_pairs", "true_pairs")
    }
    output_payload = {
        "problem_id": problem_id,
        "team_name": TEAM_NAME,
        "model": MODEL,
        "quotes": [
            {
                "text": q["text"],
                "span_start": q["span_start"],
                "span_end": q["span_end"],
                "resolved_references": q["resolved_references"],
                "quote_type": q.get("quote_type", "allusion"),
            }
            for q in quotes_with_spans
        ],
        "metrics": serialisable_metrics,
    }
    out_path = OUTPUT_DIR / f"{problem_id}.json"
    out_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"problem_id": problem_id, "num_quotes": len(quotes), **serialisable_metrics}


def all_problem_ids() -> list[str]:
    return sorted(p.stem for p in PROBLEMS_DIR.glob("*.txt"))


def main() -> None:
    threads = 20
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        problem_ids = [sys.argv[1]]
    else:
        problem_ids = all_problem_ids()

    console.print(
        Panel(
            f"[bold]{TEAM_NAME}[/bold]  |  Model: [green]{MODEL}[/green]  |  "
            f"Problems: [cyan]{len(problem_ids)}[/cyan]  |  Threads: [cyan]{threads}[/cyan]",
            title="Ruse of Reuse — Batch Extraction",
            border_style="blue",
        )
    )

    valid_book_codes = get_valid_book_codes()
    ref_mapping = load_reference_mapping()

    results: list[dict] = []
    errors: list[tuple[str, str]] = []
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing", total=len(problem_ids))

        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = {
                pool.submit(process_single, pid, valid_book_codes, ref_mapping): pid
                for pid in problem_ids
            }
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    errors.append((pid, str(exc)))
                progress.update(task, advance=1, description=f"Done: {pid}")

    elapsed = time.time() - t0
    console.print(f"\n[bold green]Completed in {elapsed:.1f}s[/bold green]")

    if errors:
        et = Table(title="Errors", box=box.ROUNDED, style="red")
        et.add_column("Problem")
        et.add_column("Error")
        for pid, err in errors:
            et.add_row(pid, err[:120])
        console.print(et)

    results.sort(key=lambda r: r["problem_id"])
    rt = Table(title="Results Summary", box=box.ROUNDED, show_lines=True)
    rt.add_column("Problem", style="cyan")
    rt.add_column("Quotes", justify="right")
    rt.add_column("TP", justify="right", style="green")
    rt.add_column("FP", justify="right", style="red")
    rt.add_column("FN", justify="right", style="red")
    rt.add_column("Prec", justify="right")
    rt.add_column("Rec", justify="right")
    rt.add_column("F1", justify="right")
    for r in results:
        f1v = r["f1"]
        f1c = "green" if f1v >= 0.7 else "yellow" if f1v >= 0.4 else "red"
        rt.add_row(
            r["problem_id"], str(r["num_quotes"]),
            str(r["true_positives"]), str(r["false_positives"]), str(r["false_negatives"]),
            f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"[{f1c}]{f1v:.3f}[/{f1c}]",
        )

    total_tp = sum(r["true_positives"] for r in results)
    total_fp = sum(r["false_positives"] for r in results)
    total_fn = sum(r["false_negatives"] for r in results)
    total_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    total_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    total_f1 = 2 * total_p * total_r / (total_p + total_r) if (total_p + total_r) else 0
    f1c = "green" if total_f1 >= 0.7 else "yellow" if total_f1 >= 0.4 else "red"
    rt.add_row(
        "[bold]TOTAL[/bold]", str(sum(r["num_quotes"] for r in results)),
        f"[bold]{total_tp}[/bold]", f"[bold]{total_fp}[/bold]", f"[bold]{total_fn}[/bold]",
        f"[bold]{total_p:.3f}[/bold]", f"[bold]{total_r:.3f}[/bold]",
        f"[bold][{f1c}]{total_f1:.3f}[/{f1c}][/bold]",
    )
    console.print(rt)


if __name__ == "__main__":
    main()
