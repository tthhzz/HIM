"""Summarize archived LoCoMo result JSON files.

The script reads the aggregate_metrics schema emitted by the evaluation
programs and prints F1 / BLEU-1 / ROUGE-L percentages by category.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
}
DEFAULT_RESULTS = [
    REPO_ROOT / "results" / "main" / "him_qwen2.5-3b_10pct.json",
]


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_mean(category_metrics: dict, metric: str) -> float:
    value = category_metrics[metric]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value) * 100


def summarize(path: Path) -> None:
    result = load_result(path)
    print(f"\n{path.relative_to(REPO_ROOT)}")
    print(f"model={result.get('model')}  questions={result.get('total_questions')}")
    print("category      F1     BLEU-1  ROUGE-L")
    for category_id, category_name in CATEGORIES.items():
        metrics = result["aggregate_metrics"][f"category_{category_id}"]
        print(
            f"{category_name:<12} "
            f"{metric_mean(metrics, 'f1'):>6.2f} "
            f"{metric_mean(metrics, 'bleu1'):>8.2f} "
            f"{metric_mean(metrics, 'rougeL_f'):>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="*",
        type=Path,
        help="Result JSON paths. Defaults to the archived HIM result file.",
    )
    args = parser.parse_args()
    paths = args.results or DEFAULT_RESULTS
    for path in paths:
        path = path if path.is_absolute() else REPO_ROOT / path
        summarize(path)


if __name__ == "__main__":
    main()
