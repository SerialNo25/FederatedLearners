"""End-of-pipeline comparison report: tables + plots across all model families and banks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EXPERIMENTS_DIR = Path("data/experiments")
REPORT_DIR = EXPERIMENTS_DIR / "_report"

METRIC_KEYS = [
    "pr_auc",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "loss",
    "fpr_at_95_recall",
]
PRIMARY_METRIC = "pr_auc"
TIE_BREAK_METRIC = "f1"

FAMILY_ORDER = ["Local", "Global", "Fincl", "Fexcl", "Ens. L+Fexcl", "Ens. L+Fincl"]
FAMILY_PALETTE = {
    "Local": "#4C72B0",
    "Global": "#55A868",
    "Fincl": "#C44E52",
    "Fexcl": "#8172B2",
    "Ens. L+Fexcl": "#CCB974",
    "Ens. L+Fincl": "#937860",
}

NAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^eval_local_bank(?P<bank>\d+)$"), "Local"),
    (re.compile(r"^eval_global_bank(?P<bank>\d+)$"), "Global"),
    (re.compile(r"^eval_fincl_bank(?P<bank>\d+)$"), "Fincl"),
    (re.compile(r"^eval_fexcl(?P<bank>\d+)_bank(?P=bank)$"), "Fexcl"),
    (re.compile(r"^ensemble_L(?P<bank>\d+)_Fexcl(?P=bank)$"), "Ens. L+Fexcl"),
    (re.compile(r"^ensemble_L(?P<bank>\d+)_Fincl$"), "Ens. L+Fincl"),
]


def classify(name: str) -> tuple[str, int] | None:
    for pattern, family in NAME_PATTERNS:
        m = pattern.match(name)
        if m:
            return family, int(m.group("bank"))
    return None


def load_long_frame() -> pd.DataFrame:
    rows: list[dict] = []
    skipped: list[str] = []
    for eval_path in sorted(EXPERIMENTS_DIR.glob("*/evaluation.json")):
        name = eval_path.parent.name
        classified = classify(name)
        if classified is None:
            skipped.append(name)
            continue
        family, bank = classified
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        row = {
            "experiment": name,
            "family": family,
            "bank": bank,
            **{key: metrics.get(key) for key in METRIC_KEYS},
        }
        rows.append(row)

    if skipped:
        print(f"Skipped {len(skipped)} directory(ies) that don't match naming rules:")
        for name in skipped:
            print(f"  - {name}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)
        df = df.sort_values(["bank", "family"]).reset_index(drop=True)
    return df


def wide_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="bank", columns="family", values=metric, observed=True)
    pivot.loc["mean"] = pivot.mean(axis=0)
    pivot.index = pivot.index.astype(str)
    return pivot.round(4)


def best_per_bank(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bank, group in df.groupby("bank", observed=True):
        ranked = group.sort_values(
            [PRIMARY_METRIC, TIE_BREAK_METRIC], ascending=False
        )
        winner = ranked.iloc[0]
        rows.append(
            {
                "bank": bank,
                "winner_family": winner["family"],
                "winner_experiment": winner["experiment"],
                "pr_auc": round(winner[PRIMARY_METRIC], 4),
                "f1": round(winner[TIE_BREAK_METRIC], 4),
                "precision": round(winner["precision"], 4),
                "recall": round(winner["recall"], 4),
            }
        )
    return pd.DataFrame(rows)


def overall_ranking(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(
        [PRIMARY_METRIC, TIE_BREAK_METRIC], ascending=False
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    for key in METRIC_KEYS:
        ranked[key] = ranked[key].round(4)
    return ranked


def write_summary_markdown(
    df: pd.DataFrame,
    pivot_pr: pd.DataFrame,
    pivot_f1: pd.DataFrame,
    winners: pd.DataFrame,
    ranking: pd.DataFrame,
    path: Path,
) -> None:
    lines = ["# Federated Fraud Detection — Comparison Report", ""]
    lines.append(f"Total experiments compared: **{len(df)}**")
    lines.append(f"Primary metric: **{PRIMARY_METRIC}** (tie-break: {TIE_BREAK_METRIC})")
    lines.append("")

    lines.append("## Per-bank winners")
    lines.append("")
    for _, row in winners.iterrows():
        lines.append(
            f"- **Bank {row['bank']}**: `{row['winner_experiment']}` "
            f"({row['winner_family']}) — PR-AUC = **{row['pr_auc']:.4f}**, "
            f"F1 = {row['f1']:.4f}"
        )
    lines.append("")

    lines.append("## Overall top 3")
    lines.append("")
    lines.append(ranking.head(3).to_markdown(index=False))
    lines.append("")

    lines.append("## PR-AUC by bank × family")
    lines.append("")
    lines.append(pivot_pr.to_markdown())
    lines.append("")

    lines.append("## F1 by bank × family")
    lines.append("")
    lines.append(pivot_f1.to_markdown())
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_palette(families: list[str]) -> list[str]:
    return [FAMILY_PALETTE.get(f, "#333333") for f in families]


def plot_metric_by_bank(df: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=df,
        x="bank",
        y=metric,
        hue="family",
        hue_order=FAMILY_ORDER,
        palette=FAMILY_PALETTE,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Bank")
    ax.set_ylabel(metric)
    ax.legend(title="Model family", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ranked(df: pd.DataFrame, path: Path) -> None:
    ranked = df.sort_values(PRIMARY_METRIC, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(ranked))))
    colors = _apply_palette(list(ranked["family"].astype(str)))
    ax.barh(ranked["experiment"], ranked[PRIMARY_METRIC], color=colors, edgecolor="black")
    for i, (val, _) in enumerate(zip(ranked[PRIMARY_METRIC], ranked["experiment"])):
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel("PR-AUC")
    ax.set_title("All experiments ranked by PR-AUC")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_PALETTE[f]) for f in FAMILY_ORDER
    ]
    ax.legend(handles, FAMILY_ORDER, title="Family", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(df: pd.DataFrame, path: Path) -> None:
    heat_metrics = ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]
    df_sorted = df.sort_values(["bank", "family"])
    raw = df_sorted.set_index("experiment")[heat_metrics]
    normalized = (raw - raw.min()) / (raw.max() - raw.min()).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(raw))))
    sns.heatmap(
        normalized,
        annot=raw.round(3),
        fmt="",
        cmap="YlGnBu",
        cbar_kws={"label": "min–max normalized (per metric)"},
        ax=ax,
    )
    ax.set_title("All metrics, all experiments (color = normalized, cells show raw values)")
    ax.set_ylabel("")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="precision",
        y="recall",
        hue="family",
        hue_order=FAMILY_ORDER,
        palette=FAMILY_PALETTE,
        style="bank",
        s=140,
        ax=ax,
    )
    ax.set_title("Precision vs Recall (shape = bank, color = family)")
    ax.set_xlim(0, max(0.5, df["precision"].max() * 1.1))
    ax.set_ylim(0, 1.02)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_summary(df: pd.DataFrame, winners: pd.DataFrame, path: Path) -> None:
    banks = sorted(df["bank"].unique())
    fig, axes = plt.subplots(1, len(banks), figsize=(5 * len(banks), 5), sharey=True)
    if len(banks) == 1:
        axes = [axes]
    winner_by_bank = {row["bank"]: row["winner_family"] for _, row in winners.iterrows()}

    for ax, bank in zip(axes, banks):
        subset = df[df["bank"] == bank].sort_values("family")
        families = [str(f) for f in subset["family"]]
        values = subset[PRIMARY_METRIC].values
        colors = _apply_palette(families)
        edges = [
            "black" if f == str(winner_by_bank.get(bank)) else "none" for f in families
        ]
        linewidths = [2.5 if e == "black" else 0 for e in edges]
        bars = ax.bar(families, values, color=colors, edgecolor=edges, linewidth=linewidths)
        for bar, val, fam in zip(bars, values, families):
            label = f"{val:.3f}"
            if fam == str(winner_by_bank.get(bank)):
                label += " *"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                label,
                ha="center",
                fontsize=8,
                fontweight="bold" if fam == str(winner_by_bank.get(bank)) else "normal",
            )
        ax.set_title(f"Bank {bank} — winner: {winner_by_bank.get(bank)}")
        ax.set_ylabel(PRIMARY_METRIC if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=35)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")

    fig.suptitle("Best model per bank (starred + outlined)", y=1.02, fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview(image_paths: list[Path], path: Path) -> None:
    n = len(image_paths)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, img_path in zip(axes, image_paths):
        ax.imshow(plt.imread(img_path))
        ax.set_title(img_path.stem, fontsize=11)
        ax.axis("off")
    for ax in axes[len(image_paths):]:
        ax.axis("off")
    fig.suptitle("Comparison Report — Overview", fontsize=16, y=1.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_console_summary(winners: pd.DataFrame, ranking: pd.DataFrame) -> None:
    print("")
    print("=" * 60)
    print(f"  Report written to: {REPORT_DIR}")
    print("=" * 60)
    print("\nPer-bank winners:")
    for _, row in winners.iterrows():
        print(
            f"  Bank {row['bank']}: {row['winner_experiment']} "
            f"({row['winner_family']})  PR-AUC={row['pr_auc']:.4f}  F1={row['f1']:.4f}"
        )
    print("\nOverall top 3:")
    for _, row in ranking.head(3).iterrows():
        print(
            f"  #{row['rank']} {row['experiment']:35s} "
            f"PR-AUC={row['pr_auc']:.4f}  F1={row['f1']:.4f}"
        )


def main() -> None:
    sns.set_theme(style="whitegrid")

    if not EXPERIMENTS_DIR.exists():
        raise SystemExit(f"Experiments directory not found: {EXPERIMENTS_DIR}")

    df = load_long_frame()
    if df.empty:
        raise SystemExit("No classifiable evaluation.json files found; nothing to report.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values(["bank", "family"])
    df_sorted.to_csv(REPORT_DIR / "metrics_long.csv", index=False)

    pivot_pr = wide_pivot(df, PRIMARY_METRIC)
    pivot_f1 = wide_pivot(df, "f1")
    pivot_pr.to_csv(REPORT_DIR / "metrics_by_bank_pr_auc.csv")
    pivot_f1.to_csv(REPORT_DIR / "metrics_by_bank_f1.csv")

    winners = best_per_bank(df)
    winners.to_csv(REPORT_DIR / "best_per_bank.csv", index=False)

    ranking = overall_ranking(df)
    ranking.to_csv(REPORT_DIR / "overall_ranking.csv", index=False)

    write_summary_markdown(df, pivot_pr, pivot_f1, winners, ranking, REPORT_DIR / "summary.md")

    fig_paths = [
        REPORT_DIR / "fig_pr_auc_by_bank.png",
        REPORT_DIR / "fig_f1_by_bank.png",
        REPORT_DIR / "fig_pr_auc_ranked.png",
        REPORT_DIR / "fig_metric_heatmap.png",
        REPORT_DIR / "fig_precision_recall_scatter.png",
        REPORT_DIR / "fig_best_summary.png",
    ]
    plot_metric_by_bank(df, "pr_auc", "PR-AUC by bank × family", fig_paths[0])
    plot_metric_by_bank(df, "f1", "F1 by bank × family", fig_paths[1])
    plot_ranked(df, fig_paths[2])
    plot_metric_heatmap(df, fig_paths[3])
    plot_precision_recall(df, fig_paths[4])
    plot_best_summary(df, winners, fig_paths[5])
    plot_overview(fig_paths, REPORT_DIR / "fig_overview.png")

    print_console_summary(winners, ranking)


if __name__ == "__main__":
    main()
