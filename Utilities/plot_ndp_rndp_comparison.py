"""Compare NDP, 16-node R-NDP, and 32-node R-NDP on matched seeds.

Copy this beside plot_rndp_size_comparison.py in Utilities/, then run:
    python Utilities/plot_ndp_rndp_comparison.py RESULTS_16 RESULTS_32 \
        --output-dir Results/ndp_rndp_comparison

Saved .pkl results are loaded to calculate used graph sizes and node-state
diversity. Only run this on your own trusted experiment files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plot_rndp_size_comparison import TASKS, KEY, add_graph_metrics, read_logs
import plot_fitness_boxplots as fitness_plots


CONFIGS = ("NDP", "R-NDP 16", "R-NDP 32")
COLORS = {"NDP": "#4C78A8", "R-NDP 16": "#72B7B2", "R-NDP 32": "#F58518"}
SHORT = {"NDP": "NDP", "R-NDP 16": "R16", "R-NDP 32": "R32"}
FITNESS_MODEL_ORDER = (
    "NDP", "R-NDP 16", "R-NDP 32",
    "NDP + HL", "R-NDP 16 + HL", "R-NDP 32 + HL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_16", type=Path, help="Results containing NDP and 16-node R-NDP")
    parser.add_argument("results_32", type=Path, help="Results containing 32-node R-NDP")
    parser.add_argument("--output-dir", type=Path, default=Path("Results/ndp_rndp_comparison"))
    parser.add_argument("--tasks", nargs="+", help="Optional exact task IDs to include")
    parser.add_argument("--hide-outliers", action="store_true")
    return parser.parse_args()


def read_standard_ndp(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob("experiments_log.csv")):
        frame = pd.read_csv(path)
        required = {*KEY, "model", "filename", "best_score_mean", "best_score_test"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        frame = frame.loc[frame["model"] == "standard_ndp"].copy()
        if frame.empty:
            continue
        frame["source_log"] = str(path)
        frame["config"] = "NDP"
        frames.append(frame)
    if not frames:
        raise ValueError(f"No standard NDP logs below {root}")
    results = pd.concat(frames, ignore_index=True)
    results["hebbian"] = results["hebbian"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if results["hebbian"].isna().any():
        raise ValueError(f"Invalid Hebbian flag in {root}")
    results["seed"] = pd.to_numeric(results["seed"], errors="raise").astype(int)
    if results.duplicated(KEY).any():
        raise ValueError(f"Duplicate standard NDP task/Hebbian/seed rows in {root}")
    for metric in ("best_score_mean", "best_score_test"):
        results[metric] = pd.to_numeric(results[metric], errors="raise")
    return results


def match_three(standard: pd.DataFrame, rndp_16: pd.DataFrame,
                rndp_32: pd.DataFrame) -> pd.DataFrame:
    matched = standard[KEY].merge(rndp_16[KEY], on=KEY).merge(rndp_32[KEY], on=KEY)
    if matched.empty:
        raise ValueError("No task/Hebbian/seed combinations occur in all three configurations")
    rndp_16 = rndp_16.copy()
    rndp_32 = rndp_32.copy()
    rndp_16["config"] = "R-NDP 16"
    rndp_32["config"] = "R-NDP 32"
    return pd.concat(
        [frame.merge(matched, on=KEY) for frame in (standard, rndp_16, rndp_32)],
        ignore_index=True,
    ).sort_values(["task", "hebbian", "seed", "config"])


def draw_panel(axis: plt.Axes, results: pd.DataFrame, metric: str,
               show_outliers: bool) -> None:
    positions = (1, 2, 3, 5, 6, 7)
    conditions = [(hebbian, config) for hebbian in (False, True) for config in CONFIGS]
    groups = [
        results.loc[
            (results["hebbian"] == hebbian) & (results["config"] == config), metric
        ].dropna().to_numpy(dtype=float)
        for hebbian, config in conditions
    ]
    present = [(p, values, config) for p, values, (_, config) in zip(positions, groups, conditions)
               if values.size]
    if present:
        boxes = axis.boxplot(
            [values for _, values, _ in present],
            positions=[p for p, _, _ in present],
            widths=0.6,
            patch_artist=True,
            showfliers=show_outliers,
            medianprops={"color": "#202020", "linewidth": 1.3},
        )
        for box, (_, _, config) in zip(boxes["boxes"], present):
            box.set_facecolor(COLORS[config])
            box.set_edgecolor("#444444")
            box.set_alpha(0.88)
    axis.set_xlim(0.4, 7.6)
    axis.set_xticks(
        positions,
        [f"{SHORT[config]}\n(n={len(values)})" for (_, config), values in zip(conditions, groups)],
    )
    axis.axvline(4, color="#cccccc", linewidth=0.8)
    axis.text(2, 1.02, "No Hebbian", ha="center", transform=axis.get_xaxis_transform(), fontsize=9)
    axis.text(6, 1.02, "Hebbian", ha="center", transform=axis.get_xaxis_transform(), fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def plot_grid(results: pd.DataFrame, tasks: list[tuple[str, str]], metric: str,
              ylabel: str, title: str, destination: Path,
              show_outliers: bool) -> None:
    columns = min(2, len(tasks))
    rows = int(np.ceil(len(tasks) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 4.4 * rows),
                                squeeze=False, constrained_layout=True)
    for axis, (task_id, task_label) in zip(axes.flat, tasks):
        draw_panel(axis, results.loc[results["task"] == task_id], metric, show_outliers)
        axis.set_title(task_label, pad=27, fontsize=12)
        axis.set_ylabel(ylabel)
    for axis in list(axes.flat)[len(tasks):]:
        axis.axis("off")
    figure.legend(
        handles=[Patch(facecolor=COLORS[config], label=config) for config in CONFIGS],
        loc="outside lower center", ncol=3, frameon=False,
    )
    figure.suptitle(title, fontsize=15)
    figure.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_training_testing(results: pd.DataFrame, tasks: list[tuple[str, str]],
                          destination: Path, show_outliers: bool) -> None:
    """Reuse the original training/testing boxplot style with six conditions."""
    labelled = results.copy()
    labelled["model_label"] = [
        config + (" + HL" if hebbian else "")
        for config, hebbian in zip(labelled["config"], labelled["hebbian"])
    ]
    long_results = fitness_plots.reshape_results(labelled)
    columns = min(2, len(tasks))
    rows = int(np.ceil(len(tasks) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(9.5 * columns, 4.8 * rows),
                                squeeze=False, constrained_layout=True)
    previous_order = fitness_plots.MODEL_ORDER
    fitness_plots.MODEL_ORDER = list(FITNESS_MODEL_ORDER)
    try:
        for axis, (task_id, task_label) in zip(axes.flat, tasks):
            panel = long_results.loc[long_results["task"] == task_id]
            fitness_plots.draw_task_panel(axis, panel, task_id, show_outliers)
            counts = labelled.loc[labelled["task"] == task_id].groupby("model_label")["seed"].nunique()
            present = [label for label in FITNESS_MODEL_ORDER if label in set(panel["model_label"])]
            axis.set_xticklabels(
                [f"{label}\n(n={counts.get(label, 0)})" for label in present],
                rotation=24, ha="right",
            )
            axis.set_title(task_label)
            axis.set_ylabel("Fitness (lower is better)")
    finally:
        fitness_plots.MODEL_ORDER = previous_order
    for axis in list(axes.flat)[len(tasks):]:
        axis.axis("off")
    figure.legend(
        handles=[
            Patch(facecolor=fitness_plots.SPLIT_COLOURS[name], edgecolor="#333333", label=name)
            for name in ("Training", "Testing")
        ] + [Line2D([0], [0], color="#B22222", linestyle="--", label="Target")],
        loc="outside lower center", ncol=3, frameon=False,
    )
    figure.suptitle("NDP and R-NDP fitness: 16 vs 32 nodes", fontsize=15)
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results = match_three(
        read_standard_ndp(args.results_16),
        read_logs(args.results_16, 16),
        read_logs(args.results_32, 32),
    )
    selected = TASKS if not args.tasks else [(key, label) for key, label in TASKS if key in args.tasks]
    if not selected:
        raise ValueError("No selected tasks have a configured label")
    results = results.loc[results["task"].isin([key for key, _ in selected])].copy()
    if results.empty:
        raise ValueError("No matched runs for the selected tasks")
    print("Matched run counts:\n", results.groupby(["task", "hebbian", "config"]).size().to_string())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_testing(results, selected,
                          args.output_dir / "training_testing_fitness_boxplots",
                          not args.hide_outliers)
    results = add_graph_metrics(results)
    figures = [
        ("best_score_mean", "Training fitness (lower is better)", "Training fitness", "training_fitness"),
        ("best_score_test", "Testing fitness (lower is better)", "Held-out testing fitness", "testing_fitness"),
        ("used_nodes", "Functional nodes", "Used nodes", "used_nodes"),
        ("used_edges", "Functional edges", "Used edges", "used_edges"),
        ("state_diversity_all", "Mean feature-wise standard deviation", "All-node state diversity", "state_diversity_all"),
        ("state_diversity_functional", "Mean feature-wise standard deviation", "Functional-node state diversity", "state_diversity_functional"),
    ]
    for metric, ylabel, title, name in figures:
        plot_grid(results, selected, metric, ylabel, title,
                  args.output_dir / name, not args.hide_outliers)
    results.to_csv(args.output_dir / "matched_run_metrics.csv", index=False)
    print(f"Saved seven PNG/PDF boxplots and matched_run_metrics.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
