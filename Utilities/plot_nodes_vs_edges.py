"""Scatter plot of functional nodes and edges in the best graph of each run."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from plot_fitness_boxplots import MODEL_ORDER, TASK_LABELS, TASK_ORDER
from plot_graph_size_boxplots import add_used_graph_sizes

MODEL_COLOURS = {
    "NDP": "#4E8098",
    "NDP + HL": "#DDB771",
    "R-NDP": "#D26760",
    "R-NDP + HL": "#5A8646",
}
MODEL_MARKERS = {
    "NDP": "o",
    "NDP + HL": "s",
    "R-NDP": "^",
    "R-NDP + HL": "D",
}


def load_paper_results(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("experiments_log.csv"))
    if not paths:
        raise FileNotFoundError(f"No experiments_log.csv files below {root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"task", "model", "hebbian", "seed"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        frame = frame.loc[frame["model"].isin({"standard_ndp", "rewiring_ndp"})].copy()
        frame["source_log"] = str(path)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    results["hebbian"] = results["hebbian"].astype(str).str.lower().map({"true": True, "false": False})
    if results["hebbian"].isna().any():
        raise ValueError("Invalid Hebbian flag in experiment logs")
    results["seed"] = pd.to_numeric(results["seed"], errors="raise").astype(int)
    key = ["task", "model", "hebbian", "seed"]
    if results.duplicated(key).any():
        raise ValueError("Duplicate task/model/Hebbian/seed runs in experiment logs")
    results["model_label"] = [
        ("NDP" if model == "standard_ndp" else "R-NDP") + (" + HL" if hebbian else "")
        for model, hebbian in zip(results["model"], results["hebbian"])
    ]
    return results


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
            point_counts = (
                model_results.groupby(["used_nodes", "used_edges"], as_index=False)
                .size()
            )
            axis.scatter(
                point_counts["used_nodes"],
                point_counts["used_edges"],
                color=MODEL_COLOURS[model_label],
                marker=MODEL_MARKERS[model_label],
                s=30 + 35 * (point_counts["size"] - 1),
                alpha=0.75,
                edgecolors="white",
                linewidths=0.5,
                label=model_label,
            )
            for point in point_counts.itertuples(index=False):
                if point.size > 1:
                    axis.annotate(
                        str(point.size),
                        (point.used_nodes, point.used_edges),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#222222",
                        zorder=4,
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

    model_handles = [
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS[model],
                linestyle="none",
                markersize=math.sqrt(30),
                markerfacecolor=MODEL_COLOURS[model],
                markeredgecolor="white",
                label=model,
            )
            for model in MODEL_ORDER
    ]
    count_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=math.sqrt(30 + 35 * (count - 1)),
            markerfacecolor="#999999",
            markeredgecolor="white",
            label=f"{count} {'run' if count == 1 else 'runs'}",
        )
        for count in (1, 2, 5)
    ]
    figure.legend(
        handles=model_handles + count_handles,
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
    metrics_path = args.results / "figures" / "matched_run_metrics.csv"
    if metrics_path.is_file():
        results = pd.read_csv(metrics_path)
        required = {"task", "model", "hebbian", "seed", "model_label", "used_nodes", "used_edges"}
        missing = required - set(results)
        if missing:
            raise ValueError(f"{metrics_path} lacks columns: {sorted(missing)}")
        print(f"Using previously computed functional graph sizes from {metrics_path}")
    else:
        results = add_used_graph_sizes(load_paper_results(args.results))

    if args.tasks:
        tasks = args.tasks
        results = results[results["task"].isin(tasks)].copy()
    else:
        available_tasks = set(results["task"])
        tasks = [task for task in TASK_ORDER if task in available_tasks]
        tasks.extend(sorted(available_tasks.difference(tasks)))

    if not tasks:
        raise ValueError("No tasks were selected for plotting.")

    point_counts = (
        results.groupby(
            ["task", "model_label", "used_nodes", "used_edges"],
            observed=True,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "runs"})
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = args.output_dir / "used_nodes_vs_edges_counts.csv"
    point_counts.to_csv(counts_path, index=False)

    png_path, pdf_path = create_figure(results, tasks, args.output_dir)
    print(f"PNG figure: {png_path}")
    print(f"PDF figure: {pdf_path}")
    print(f"Counts CSV: {counts_path}")
    print(
        results.groupby(["task", "model_label"], observed=True)["seed"]
        .nunique()
        .rename("runs")
        .to_string()
    )


if __name__ == "__main__":
    main()
