from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = Path.home() / "Downloads" / "evaluation_200_queries_realistic.xlsx"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "final_dedup_by_actionplan_recovered.xlsx"
DEFAULT_GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "ollama")
DEFAULT_TOP_K = 5
REQUIRED_COLUMNS = {
    "query",
    "expected_ncid",
    "expected_nc",
    "expected_plan",
}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_text(value: Any, lookup: Any) -> str:
    return lookup.normalize_nc(_stringify(value))


def _normalize_plan(value: Any, lookup: Any) -> str:
    return lookup.normalize_plan(_stringify(value))


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def _format_metric_value(metric: str, value: Any) -> str:
    if metric == "Total Queries":
        return str(int(value))
    if isinstance(value, float):
        if "Latency" in metric:
            return f"{value:.3f} seconds"
        return f"{value:.2f}%"
    if metric != "Total Queries" and isinstance(value, int):
        return f"{value:.2f}%"
    return str(value)


def _candidate_ids(result: Any, max_count: int = 3) -> list[str]:
    ids: list[str] = []
    if result.matched_ncid:
        ids.append(str(result.matched_ncid).strip())
    for candidate in result.top_candidates:
        candidate_id = str(candidate.ncid).strip()
        if candidate_id and candidate_id not in ids:
            ids.append(candidate_id)
        if len(ids) >= max_count:
            break
    return ids[:max_count]


def _candidate_texts(result: Any, max_count: int = 3) -> list[str]:
    texts: list[str] = []
    if result.matched_nc:
        texts.append(str(result.matched_nc).strip())
    for candidate in result.top_candidates:
        candidate_text = str(candidate.nc).strip()
        if candidate_text and candidate_text not in texts:
            texts.append(candidate_text)
        if len(texts) >= max_count:
            break
    return texts[:max_count]


def _check_ollama_ready(
    base_url: str,
    model: str,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> None:
    import requests

    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/api/tags",
            timeout=timeout,
            headers=headers or {},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Ollama is not reachable. Start Ollama first, or rerun with "
            "--generation-backend template for a fast non-Ollama check. "
            f"Details: {exc}"
        ) from exc

    payload = response.json()
    available_models = [
        str(item.get("name", "")).strip()
        for item in payload.get("models", [])
        if item.get("name")
    ]
    if model and model not in available_models:
        raise RuntimeError(
            f"Ollama is reachable, but model '{model}' was not listed. "
            f"Available models: {', '.join(available_models) or 'none'}"
        )


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, ollama_headers
    from src.generator import Generator
    from src.pipeline_runtime import LegalRecommendationPipeline
    from src.truth_lookup import TruthLookup

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {input_path}")

    df = pd.read_excel(input_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    if (
        args.generation_backend == "ollama"
        and not args.no_generation
        and not args.skip_ollama_check
    ):
        _check_ollama_ready(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            timeout=args.ollama_check_timeout,
            headers=ollama_headers(),
        )

    lookup = TruthLookup(args.data_path)
    generator = Generator(backend=args.generation_backend)
    pipeline = LegalRecommendationPipeline(
        data_path=args.data_path,
        score_threshold=args.score_threshold,
        lookup=lookup,
        generator=generator,
    )

    detail_rows: list[dict[str, Any]] = []
    started_at = datetime.now()

    for row_number, (index, row) in enumerate(df.iterrows(), start=1):
        print(f"Evaluating {row_number}/{len(df)}: test_id={row.get('test_id', index + 1)}", flush=True)
        query = _stringify(row["query"])
        expected_ncid = _stringify(row["expected_ncid"])
        expected_nc = _stringify(row["expected_nc"])
        expected_plan = _stringify(row["expected_plan"])

        row_start = time.perf_counter()
        error = ""
        result = None

        try:
            result = pipeline.run(
                query=query,
                top_k=args.top_k,
                with_generation=not args.no_generation,
            )
        except Exception as exc:  # Keep the whole evaluation running.
            error = f"{type(exc).__name__}: {exc}"

        latency = time.perf_counter() - row_start

        top_ids: list[str] = []
        top_texts: list[str] = []
        top1_ncid = ""
        top1_nc = ""
        predicted_plan = ""
        generated_action_fr = ""
        generated_explanation_fr = ""
        mode = "error"
        decision_reason = ""

        if result is not None:
            top_ids = _candidate_ids(result, max_count=3)
            top_texts = _candidate_texts(result, max_count=3)
            top1_ncid = top_ids[0] if top_ids else ""
            top1_nc = top_texts[0] if top_texts else ""
            predicted_plan = _stringify(result.official_plan)
            generated_action_fr = _stringify(result.display_plan_fr)
            generated_explanation_fr = _stringify(result.explanation_fr)
            mode = result.mode
            decision_reason = result.decision_reason

        top1_ncid_hit = top1_ncid == expected_ncid
        top3_ncid_hit = expected_ncid in top_ids
        nc_text_hit = _normalize_text(top1_nc, lookup) == _normalize_text(expected_nc, lookup)
        plan_hit = _normalize_plan(predicted_plan, lookup) == _normalize_plan(expected_plan, lookup)

        detail_rows.append(
            {
                "test_id": row.get("test_id", index + 1),
                "query": query,
                "expected_ncid": expected_ncid,
                "expected_nc": expected_nc,
                "expected_plan": expected_plan,
                "predicted_ncid": top1_ncid,
                "predicted_nc": top1_nc,
                "predicted_plan": predicted_plan,
                "top3_predicted_ncids": ", ".join(top_ids),
                "top3_predicted_ncs": " | ".join(top_texts),
                "mode": mode,
                "decision_reason": decision_reason,
                "top1_ncid_match": top1_ncid_hit,
                "top3_ncid_match": top3_ncid_hit,
                "nc_text_match": nc_text_hit,
                "plan_exact_match": plan_hit,
                "latency_seconds": round(latency, 4),
                "generated_action_fr": generated_action_fr,
                "generated_explanation_fr": generated_explanation_fr,
                "error": error,
            }
        )

    details = pd.DataFrame(detail_rows)
    total = len(details)
    success = int((details["error"] == "").sum()) if total else 0
    verified = details[details["mode"] == "verified"] if total else details
    verified_correct = int(verified["top1_ncid_match"].sum()) if len(verified) else 0
    avg_latency = float(details.loc[details["error"] == "", "latency_seconds"].mean() or 0.0)

    metric_rows = [
        ("Total Queries", total),
        ("Pipeline Success Rate", _pct(success, total)),
        ("Top-1 NCid Accuracy", _pct(int(details["top1_ncid_match"].sum()), total)),
        ("Top-3 NCid Accuracy", _pct(int(details["top3_ncid_match"].sum()), total)),
        ("NC Text Accuracy", _pct(int(details["nc_text_match"].sum()), total)),
        ("Plan Exact Match Rate", _pct(int(details["plan_exact_match"].sum()), total)),
        ("Verified Coverage", _pct(len(verified), total)),
        ("Verified Precision", _pct(verified_correct, len(verified))),
        ("Ambiguous Rate", _pct(int((details["mode"] == "ambiguous").sum()), total)),
        ("Advisory Rate", _pct(int((details["mode"] == "advisory").sum()), total)),
        ("No-Match Rate", _pct(int((details["mode"] == "no_match").sum()), total)),
        ("Average Latency", round(avg_latency, 3)),
    ]

    metrics = pd.DataFrame(metric_rows, columns=["Metric", "Value"])
    metrics["Formatted Value"] = [
        _format_metric_value(metric, value)
        for metric, value in metrics[["Metric", "Value"]].itertuples(index=False)
    ]
    metrics.attrs["started_at"] = started_at.isoformat(timespec="seconds")
    return metrics, details


def write_report(metrics: pd.DataFrame, details: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, index=False, sheet_name="metrics")
        details.to_excel(writer, index=False, sheet_name="details")


def write_json_report(
    metrics: pd.DataFrame,
    details: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
) -> Path:
    json_path = output_path.with_suffix(".json")
    payload = {
        "run_info": {
            "started_at": metrics.attrs.get("started_at", ""),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "input": str(Path(args.input)),
            "output_xlsx": str(output_path),
            "output_json": str(json_path),
            "data_path": str(Path(args.data_path)),
            "generation_backend": args.generation_backend,
            "no_generation": args.no_generation,
            "top_k": args.top_k,
            "limit": args.limit,
            "score_threshold": args.score_threshold,
        },
        "metrics": metrics.to_dict(orient="records"),
        "details": details.to_dict(orient="records"),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return json_path


def print_metrics(metrics: pd.DataFrame) -> None:
    metric_width = max(len("Metric"), *(len(str(value)) for value in metrics["Metric"]))
    value_width = max(len("Value"), *(len(str(value)) for value in metrics["Formatted Value"]))

    print(f"{'Metric'.ljust(metric_width)}  {'Value'.ljust(value_width)}")
    print(f"{'-' * metric_width}  {'-' * value_width}")
    for _, row in metrics.iterrows():
        print(f"{str(row['Metric']).ljust(metric_width)}  {str(row['Formatted Value']).ljust(value_width)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate recommendation predictions and generated French output "
            "against an Excel benchmark."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the Excel file containing query and expected_* columns.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output .xlsx path. Defaults to data/evaluation_runs/<input>_results_<timestamp>.xlsx.",
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Path to the official recommendation dataset.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of candidates retrieved per query.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows.")
    parser.add_argument(
        "--generation-backend",
        choices=("ollama", "template"),
        default=DEFAULT_GENERATION_BACKEND,
        help="Generation backend to use for generated_action_fr and generated_explanation_fr.",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Skip generated French action/explanation and evaluate mapping only.",
    )
    parser.add_argument(
        "--skip-ollama-check",
        action="store_true",
        help="Do not preflight Ollama before an ollama-backed generation run.",
    )
    parser.add_argument(
        "--ollama-check-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the Ollama health check before failing early.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="Pipeline distance threshold used by the decision layer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics, details = evaluate(args)

    output_path = Path(args.output) if args.output else None
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_stem = Path(args.input).stem
        output_path = PROJECT_ROOT / "data" / "evaluation_runs" / f"{input_stem}_results_{timestamp}.xlsx"

    write_report(metrics, details, output_path)
    json_path = write_json_report(metrics, details, output_path, args)
    print_metrics(metrics)
    print(f"\nDetailed Excel report: {output_path}")
    print(f"Detailed JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
