"""Collect evaluation.json files from all experiments and produce a comparison table."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "main.py").is_file() and (path / "configs").is_dir():
            return path
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = Path("data/experiments")
OUTPUT_CSV = EXPERIMENTS_DIR / "results_comparison.csv"
OUTPUT_MD = EXPERIMENTS_DIR / "results_comparison.md"

METRIC_KEYS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy", "loss", "fpr_at_95_recall"]


def collect_results() -> list[dict]:
    rows = []
    for eval_path in sorted(EXPERIMENTS_DIR.glob("*/evaluation.json")):
        experiment_name = eval_path.parent.name
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})

        row = {"experiment": experiment_name}
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        row["ensemble_weight"] = data.get("ensemble_weight")
        rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> None:
    fieldnames = ["experiment"] + METRIC_KEYS + ["ensemble_weight"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written to {OUTPUT_CSV}")


def write_markdown(rows: list[dict]) -> None:
    cols = ["experiment"] + METRIC_KEYS
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, separator]
    for row in rows:
        values = []
        for col in cols:
            val = row.get(col)
            if val is None:
                values.append("--")
            elif isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    md = "\n".join(lines) + "\n"
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"Markdown written to {OUTPUT_MD}")


def main() -> None:
    rows = collect_results()
    if not rows:
        print("No evaluation.json files found in", EXPERIMENTS_DIR)
        return
    write_csv(rows)
    write_markdown(rows)
    print(f"\nCollected {len(rows)} experiment results:")
    for row in rows:
        pr_auc = row.get("pr_auc")
        pr_str = f"{pr_auc:.4f}" if pr_auc is not None else "--"
        print(f"  {row['experiment']:40s} PR-AUC={pr_str}")


if __name__ == "__main__":
    main()
