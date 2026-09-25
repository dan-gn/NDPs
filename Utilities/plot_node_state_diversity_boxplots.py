"""Plot node-state diversity distributions from the run-level analysis CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


MODEL_ORDER = ["NDP", "NDP + HL", "R-NDP", "R-NDP + HL"]
MODEL_COLOURS = {
    "NDP": "#4C78A8",
    "NDP + HL": "#72B7B2",
    "R-NDP": "#F58518",
    "R-NDP + HL": "#E45756",
}

TASK_ORDER = [
    "CartPole-v1",
    "popgym-PositionOnlyCartPoleEasy-v0",
    "Acrobot-v1",
    "MountainCar-v0",
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


def draw_grouped_boxplots(
    axis: plt.Axes,
    results: pd.DataFrame,
    tasks: list[str],
    metric: str,
    show_outliers: bool,
) -> None:
    task_centres = np.arange(len(tasks), dtype=float)
    model_offsets = np.linspace(-0.3, 0.3, len(MODEL_ORDER))
    rng = np.random.default_rng(2026)

    for task_centre, task_name in zip(task_centres, tasks):
        task_results = results[results["task"] == task_name]
        for model_offset, model_label in zip(model_offsets, MODEL_ORDER):
            values = task_results.loc[
                task_results["model_label"] == model_label,
                metric,
            ].dropna().to_numpy(dtype=float)
            if not values.size:
                continue

            position = task_centre + model_offset
            boxes = axis.boxplot(
                [values],
                positions=[position],
                widths=0.16,
                patch_artist=True,
                showfliers=show_outliers,
                medianprops={"color": "black", "linewidth": 1.3},
                whiskerprops={"color": "#555555", "linewidth": 0.9},
                capprops={"color": "#555555", "linewidth": 0.9},
                boxprops={"edgecolor": "#333333", "linewidth": 0.9},
                flierprops={
                    "marker": "o",
                    "markersize": 2.5,
                    "markerfacecolor": MODEL_COLOURS[model_label],
                    "markeredgecolor": "none",
                    "alpha": 0.5,
                },
            )
            boxes["boxes"][0].set_facecolor(MODEL_COLOURS[model_label])
            boxes["boxes"][0].set_alpha(0.82)

            jitter = rng.uniform(-0.025, 0.025, size=len(values))
            axis.scatter(
                position + jitter,
                values,
                s=8,
                color=MODEL_COLOURS[model_label],
                edgecolors="white",
                linewidths=0.2,
                alpha=0.45,
                zorder=3,
            )

    labels = [
        TASK_LABELS.get(task, task).replace("PositionOnlyCartPole", "PositionOnly\nCartPole")
        for task in tasks
    ]
    axis.set_xticks(task_centres, labels)
    axis.set_xlim(-0.6, len(tasks) - 0.4)
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

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )
    figure, axis = plt.subplots(
        figsize=(max(9.5, 2.0 * len(tasks)), 5.2),
        constrained_layout=True,
    )
    draw_grouped_boxplots(axis, scoped_results, tasks, metric, show_outliers)
    axis.set_ylabel(METRICS[metric]["label"] + " (lower = more converged)")
    axis.set_ylim(bottom=0)
    figure.legend(
        handles=[
            Patch(facecolor=MODEL_COLOURS[model], edgecolor="#333333", label=model)
            for model in MODEL_ORDER
        ],
        loc="outside upper center",
        ncol=len(MODEL_ORDER),
        frameon=False,
    )

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
