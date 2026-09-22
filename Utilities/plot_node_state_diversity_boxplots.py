"""Plot node-state diversity distributions from the run-level analysis CSV."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["NDP", "NDP + HL", "R-NDP", "R-NDP + HL"]
MODEL_COLOURS = {
    "NDP": "#4C78A8",
    "NDP + HL": "#72B7B2",
    "R-NDP": "#F58518",
    "R-NDP + HL": "#E45756",
}

TASK_ORDER = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "popgym-PositionOnlyCartPoleEasy-v0",
    "LunarLander-v3",
]
TASK_LABELS = {
    "CartPole-v1": "CartPole",
    "Acrobot-v1": "Acrobot",
    "MountainCar-v0": "MountainCar",
    "popgym-PositionOnlyCartPoleEasy-v0": "PositionOnlyCartPole",
    "LunarLander-v3": "LunarLander",
}

SCOPES = {
    "all": "All nodes",
    "functional": "Functional nodes",
    "functional_hidden": "Functional hidden nodes",
}

METRICS = {
    "mean_featurewise_std": {
        "label": "Node-state diversity",
        "suffix": "diversity",
    },
    "mean_pairwise_distance": {
        "label": "Mean pairwise node-state distance",
        "suffix": "pairwise_distance",
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create node-state diversity boxplots for each task and model."
    )
    parser.add_argument(
        "run_results",
        type=Path,
        help="Path to node_state_diversity_runs.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/node_state_diversity/figures"),
        help="Directory in which PNG and PDF figures are saved.",
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=list(SCOPES),
        default=["all", "functional"],
        help="Node scopes to plot. Defaults to all and functional nodes.",
    )
    parser.add_argument(
        "--metric",
        choices=list(METRICS),
        default="mean_featurewise_std",
        help="Diversity measure to plot.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional task names. The five study tasks are shown by default.",
    )
    parser.add_argument(
        "--show-outliers",
        action="store_true",
        help="Show points beyond the boxplot whiskers.",
    )
    return parser.parse_args()


def load_run_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Run-level diversity file not found: {path}")
    results = pd.read_csv(path)
    required = {
        "task",
        "seed",
        "model_label",
        "scope",
        "mean_featurewise_std",
        "mean_pairwise_distance",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"The input CSV is missing columns: {sorted(missing)}")
    for metric in METRICS:
        results[metric] = pd.to_numeric(results[metric], errors="coerce")
    return results


def draw_panel(
    axis: plt.Axes,
    task_results: pd.DataFrame,
    task_name: str,
    metric: str,
    show_outliers: bool,
) -> None:
    centres = np.arange(len(MODEL_ORDER), dtype=float)
    rng = np.random.default_rng(2026)

    for centre, model_label in zip(centres, MODEL_ORDER):
        values = task_results.loc[
            task_results["model_label"] == model_label,
            metric,
        ].dropna().to_numpy(dtype=float)
        if not values.size:
            continue

        boxes = axis.boxplot(
            [values],
            positions=[centre],
            widths=0.58,
            patch_artist=True,
            showfliers=show_outliers,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
            boxprops={"edgecolor": "#333333", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": MODEL_COLOURS[model_label],
                "markeredgecolor": "none",
                "alpha": 0.5,
            },
        )
        boxes["boxes"][0].set_facecolor(MODEL_COLOURS[model_label])
        boxes["boxes"][0].set_alpha(0.82)

        jitter = rng.uniform(-0.07, 0.07, size=len(values))
        axis.scatter(
            centre + jitter,
            values,
            s=12,
            color=MODEL_COLOURS[model_label],
            edgecolors="white",
            linewidths=0.25,
            alpha=0.5,
            zorder=3,
        )

    if task_results.empty or task_results[metric].dropna().empty:
        axis.text(
            0.5,
            0.5,
            "No results available",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#666666",
            fontsize=11,
            fontstyle="italic",
        )

    axis.set_title(TASK_LABELS.get(task_name, task_name))
    axis.set_xticks(centres, MODEL_ORDER, rotation=18, ha="right")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def create_scope_figure(
    results: pd.DataFrame,
    tasks: list[str],
    scope: str,
    metric: str,
    output_dir: Path,
    show_outliers: bool,
) -> tuple[Path, Path]:
    scoped_results = results[results["scope"] == scope]
    n_columns = min(2, len(tasks))
    n_rows = math.ceil(len(tasks) / n_columns)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7.2 * n_columns, 4.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for axis, task_name in zip(axes.flat, tasks):
        task_results = scoped_results[scoped_results["task"] == task_name]
        draw_panel(axis, task_results, task_name, metric, show_outliers)
        axis.set_ylabel(METRICS[metric]["label"] + " (lower = more converged)")
        axis.set_ylim(bottom=0)

    for axis in axes.flat[len(tasks):]:
        axis.set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"node_state_{METRICS[metric]['suffix']}_{scope}_boxplots"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    args = parse_arguments()
    results = load_run_results(args.run_results)

    available_tasks = set(results["task"])
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = TASK_ORDER.copy()
        tasks.extend(sorted(available_tasks.difference(tasks)))

    for scope in args.scopes:
        png_path, pdf_path = create_scope_figure(
            results,
            tasks,
            scope,
            args.metric,
            args.output_dir,
            args.show_outliers,
        )
        print(f"{SCOPES[scope]} PNG: {png_path}")
        print(f"{SCOPES[scope]} PDF: {pdf_path}")


if __name__ == "__main__":
    main()
