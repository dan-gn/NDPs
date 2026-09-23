"""Scatter plot of functional nodes and edges in the best graph of each run."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from plot_fitness_boxplots import MODEL_ORDER, TASK_LABELS, TASK_ORDER, load_results
from plot_graph_size_boxplots import MODEL_COLOURS, add_used_graph_sizes


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot used nodes against used edges for each task and model."
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Root directory containing experiments_log.csv files and saved results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/nodes_vs_edges"),
        help="Directory for the PNG and PDF figures.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Task IDs to include. Defaults to study tasks with available results.",
    )
    return parser.parse_args()


def create_figure(
    results: pd.DataFrame,
    tasks: list[str],
    output_dir: Path,
) -> tuple[Path, Path]:
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
        figsize=(6.6 * n_columns, 4.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for axis, task_name in zip(axes.flat, tasks):
        task_results = results[results["task"] == task_name]
        for model_label in MODEL_ORDER:
            model_results = task_results[
                task_results["model_label"] == model_label
            ]
            if model_results.empty:
                continue
            axis.scatter(
                model_results["used_nodes"],
                model_results["used_edges"],
                color=MODEL_COLOURS[model_label],
                s=34,
                alpha=0.65,
                edgecolors="white",
                linewidths=0.35,
                label=model_label,
            )

        if task_results.empty:
            axis.text(
                0.5,
                0.5,
                "No results available",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#666666",
                fontstyle="italic",
            )

        axis.set_title(TASK_LABELS.get(task_name, task_name))
        axis.set_xlabel("Number of used nodes")
        axis.set_ylabel("Number of used edges")
        axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)

    for axis in axes.flat[len(tasks):]:
        axis.set_visible(False)

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=7,
                markerfacecolor=MODEL_COLOURS[model],
                markeredgecolor="white",
                label=model,
            )
            for model in MODEL_ORDER
        ],
        loc="outside upper center",
        ncol=len(MODEL_ORDER),
        frameon=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "used_nodes_vs_edges.png"
    pdf_path = output_dir / "used_nodes_vs_edges.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    args = parse_arguments()
    results = add_used_graph_sizes(load_results(args.results))

    if args.tasks:
        tasks = args.tasks
        results = results[results["task"].isin(tasks)].copy()
    else:
        available_tasks = set(results["task"])
        tasks = [task for task in TASK_ORDER if task in available_tasks]
        tasks.extend(sorted(available_tasks.difference(tasks)))

    if not tasks:
        raise ValueError("No tasks were selected for plotting.")

    png_path, pdf_path = create_figure(results, tasks, args.output_dir)
    print(f"PNG figure: {png_path}")
    print(f"PDF figure: {pdf_path}")
    print(
        results.groupby(["task", "model_label"], observed=True)["seed"]
        .nunique()
        .rename("runs")
        .to_string()
    )


if __name__ == "__main__":
    main()
