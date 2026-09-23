"""Plot node and edge counts of the best graph from each experiment."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Saved experiment objects reference project packages such as Tasks, NDP,
# Graph, and Optimisation. When this file is launched from Utilities, Python
# otherwise places only the Utilities directory on the import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plot_fitness_boxplots import (
    MODEL_ORDER,
    TASK_LABELS,
    TASK_ORDER,
    load_results,
)


MODEL_COLOURS = {
    "NDP": "#4C78A8",
    "NDP + HL": "#72B7B2",
    "R-NDP": "#F58518",
    "R-NDP + HL": "#E45756",
}

METRICS = {
    "used_nodes": {
        "label": "Number of used nodes",
    },
    "used_edges": {
        "label": "Number of used edges",
    },
}


def local_result_file(row: pd.Series) -> Path:
    """Resolve a logged Colab/WSL output filename beside its source CSV."""
    filename = Path(str(row["filename"]).replace("\\", "/")).name
    return Path(row["source_log"]).parent / filename


def read_used_counts_from_result(row: pd.Series) -> tuple[float, float]:
    """Read functional graph sizes stored on the best EA individual."""
    result_file = local_result_file(row)
    if not result_file.exists():
        raise FileNotFoundError(
            "The used-node/edge fields are absent from the CSV and the saved "
            f"result could not be found at {result_file}."
        )

    with result_file.open("rb") as handle:
        output = pickle.load(handle)

    optimiser = output["optimiser"]
    individual = optimiser.best_individual
    used_nodes = individual.best_graph_used_nodes
    used_edges = individual.best_graph_used_edges

    if used_nodes is None or used_edges is None:
        graph = individual.best_graph
        task = output["task"]
        n_inputs = task.parameters["graph_n_inputs"]
        n_outputs = task.parameters["graph_n_outputs"]
        used_nodes = graph.get_number_of_used_nodes(n_inputs, n_outputs)
        used_edges = graph.get_number_of_used_edges(n_inputs, n_outputs)

    return float(used_nodes), float(used_edges)


def add_used_graph_sizes(results: pd.DataFrame) -> pd.DataFrame:
    """Use logged functional sizes, falling back to each saved result file."""
    results = results.copy()
    logged_nodes = results.get(
        "best_graph_used_nodes",
        pd.Series(np.nan, index=results.index),
    )
    logged_edges = results.get(
        "best_graph_used_edges",
        pd.Series(np.nan, index=results.index),
    )
    results["used_nodes"] = pd.to_numeric(logged_nodes, errors="coerce")
    results["used_edges"] = pd.to_numeric(logged_edges, errors="coerce")

    missing = results["used_nodes"].isna() | results["used_edges"].isna()
    for index in results.index[missing]:
        used_nodes, used_edges = read_used_counts_from_result(results.loc[index])
        results.loc[index, "used_nodes"] = used_nodes
        results.loc[index, "used_edges"] = used_edges

    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot node and edge counts of the best developed graphs."
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Root result directory containing experiments_log.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/graph_size_boxplots"),
        help="Directory in which the PNG and PDF figures are saved.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help=(
            "Optional task names to include. By default, the five study tasks "
            "are shown, including empty groups for unavailable tasks."
        ),
    )
    parser.add_argument(
        "--show-outliers",
        action="store_true",
        help="Show points beyond the boxplot whiskers.",
    )
    return parser.parse_args()


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


def create_figure(
    results: pd.DataFrame,
    tasks: list[str],
    output_dir: Path,
    show_outliers: bool,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )
    figure, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(max(9.5, 2.0 * len(tasks)), 8.5),
        constrained_layout=True,
    )
    for axis, metric in zip(axes, METRICS):
        draw_grouped_boxplots(axis, results, tasks, metric, show_outliers)
        axis.set_ylabel(METRICS[metric]["label"])
    axes[0].tick_params(labelbottom=False)

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
    png_path = output_dir / "best_graph_size_boxplots.png"
    pdf_path = output_dir / "best_graph_size_boxplots.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    args = parse_arguments()
    results = load_results(args.results)
    results = add_used_graph_sizes(results)

    available_tasks = set(results["task"])
    if args.tasks:
        tasks_to_plot = args.tasks
        results = results[results["task"].isin(args.tasks)].copy()
    else:
        tasks_to_plot = TASK_ORDER.copy()
        tasks_to_plot.extend(sorted(available_tasks.difference(tasks_to_plot)))

    if not tasks_to_plot:
        raise ValueError("No tasks were selected for plotting.")

    png_path, pdf_path = create_figure(
        results,
        tasks_to_plot,
        args.output_dir,
        args.show_outliers,
    )
    print(f"Combined graph-size PNG: {png_path}")
    print(f"Combined graph-size PDF: {pdf_path}")

    counts = (
        results.groupby(["task", "model_label"], observed=True)["seed"]
        .nunique()
        .rename("seeds")
        .reset_index()
    )
    print("\nRuns included:")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
