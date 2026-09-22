"""Create publication-ready training/testing fitness boxplots.

The script recursively discovers ``experiments_log.csv`` files below a result
directory. It produces a multi-panel figure with one panel per task and
side-by-side Training and Testing boxes for every model condition.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "task",
    "seed",
    "model",
    "hebbian",
    "best_score_mean",
    "best_score_test",
}

MODEL_ORDER = [
    "NDP",
    "NDP + HL",
    "R-NDP",
    "R-NDP + HL",
]

MODEL_LABELS = {
    ("standard_ndp", False): "NDP",
    ("standard_ndp", True): "NDP + HL",
    ("rewiring_ndp", False): "R-NDP",
    ("rewiring_ndp", True): "R-NDP + HL",
    # Legacy identifier retained for previously generated result logs.
    ("hebbian_ndp", False): "R-NDP",
    ("hebbian_ndp", True): "R-NDP + HL",
}

SPLIT_COLOURS = {
    "Training": "#4C78A8",
    "Testing": "#F58518",
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

TASK_TARGETS = {
    "CartPole-v1": -500.0,
    "Acrobot-v1": 75.0,
    "MountainCar-v0": 110.0,
    "popgym-PositionOnlyCartPoleEasy-v0": -1.0,
    "LunarLander-v3": -200.0,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training and held-out testing fitness distributions for each "
            "NDP model condition."
        )
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Root result directory containing experiments_log.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/fitness_boxplots"),
        help="Directory in which the PNG, PDF, and combined CSV are saved.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional task names to include. By default, all tasks are included.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title displayed above the complete figure.",
    )
    parser.add_argument(
        "--show-outliers",
        action="store_true",
        help="Show points beyond the boxplot whiskers.",
    )
    return parser.parse_args()


def parse_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalised = str(value).strip().lower()
    if normalised in {"true", "1", "yes"}:
        return True
    if normalised in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a Boolean value.")


def load_results(results_root: Path) -> pd.DataFrame:
    log_files = sorted(results_root.rglob("experiments_log.csv"))
    if not log_files:
        raise FileNotFoundError(
            f"No experiments_log.csv files were found below {results_root}."
        )

    frames = []
    for log_file in log_files:
        frame = pd.read_csv(log_file)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{log_file} is missing required columns: {sorted(missing)}"
            )
        frame["source_log"] = str(log_file)
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    results["hebbian"] = results["hebbian"].map(parse_boolean)
    results["model_label"] = [
        MODEL_LABELS.get((model, hebbian))
        for model, hebbian in zip(results["model"], results["hebbian"])
    ]

    unsupported = results[results["model_label"].isna()][["model", "hebbian"]]
    if not unsupported.empty:
        conditions = unsupported.drop_duplicates().to_dict("records")
        print(f"Skipping unsupported model conditions: {conditions}")
        results = results[results["model_label"].notna()].copy()

    for column in ("best_score_mean", "best_score_test"):
        results[column] = pd.to_numeric(results[column], errors="coerce")

    results = results.dropna(subset=["best_score_mean", "best_score_test"])
    if results.empty:
        raise ValueError("No complete training/testing fitness pairs were found.")

    duplicate_key = ["task", "seed", "model", "hebbian"]
    duplicated = results.duplicated(duplicate_key, keep=False)
    if duplicated.any():
        examples = results.loc[duplicated, duplicate_key].drop_duplicates()
        raise ValueError(
            "Multiple rows were found for the same task, seed, and model "
            "condition. Use a result directory containing one final run per "
            f"condition. Examples:\n{examples.head(10).to_string(index=False)}"
        )

    return results


def reshape_results(results: pd.DataFrame) -> pd.DataFrame:
    long_results = results.melt(
        id_vars=["task", "seed", "model", "hebbian", "model_label"],
        value_vars=["best_score_mean", "best_score_test"],
        var_name="evaluation_set",
        value_name="fitness",
    )
    long_results["evaluation_set"] = long_results["evaluation_set"].map(
        {
            "best_score_mean": "Training",
            "best_score_test": "Testing",
        }
    )
    return long_results


def draw_task_panel(
    axis: plt.Axes,
    task_results: pd.DataFrame,
    task_name: str,
    show_outliers: bool,
) -> None:
    present_models = [
        model for model in MODEL_ORDER
        if model in set(task_results["model_label"])
    ]
    if not present_models:
        present_models = MODEL_ORDER.copy()
    centres = np.arange(len(present_models), dtype=float)
    width = 0.32
    offsets = {"Training": -width / 1.8, "Testing": width / 1.8}

    for evaluation_set in ("Training", "Testing"):
        distributions = []
        positions = []
        for centre, model_label in zip(centres, present_models):
            values = task_results.loc[
                (task_results["model_label"] == model_label)
                & (task_results["evaluation_set"] == evaluation_set),
                "fitness",
            ].to_numpy()
            if values.size:
                distributions.append(values)
                positions.append(centre + offsets[evaluation_set])

        if not distributions:
            continue

        boxes = axis.boxplot(
            distributions,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=show_outliers,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
            boxprops={"edgecolor": "#333333", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": SPLIT_COLOURS[evaluation_set],
                "markeredgecolor": "none",
                "alpha": 0.5,
            },
        )
        for box in boxes["boxes"]:
            box.set_facecolor(SPLIT_COLOURS[evaluation_set])
            box.set_alpha(0.82)

        # Show individual seeds without obscuring the box summaries.
        rng = np.random.default_rng(2026 if evaluation_set == "Training" else 2027)
        for position, values in zip(positions, distributions):
            jitter = rng.uniform(-0.045, 0.045, size=len(values))
            axis.scatter(
                position + jitter,
                values,
                s=10,
                color=SPLIT_COLOURS[evaluation_set],
                edgecolors="white",
                linewidths=0.25,
                alpha=0.45,
                zorder=3,
            )

    target = TASK_TARGETS.get(task_name)
    if target is not None:
        axis.axhline(
            target,
            color="#B22222",
            linestyle="--",
            linewidth=1.4,
            zorder=1,
        )
        axis.annotate(
            f"Target = {target:g}",
            xy=(1.0, target),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color="#8B1A1A",
            fontsize=9,
        )

    if task_results.empty:
        axis.text(
            0.5,
            0.55,
            "No results available",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#666666",
            fontsize=11,
            fontstyle="italic",
        )

    axis.set_title(TASK_LABELS.get(task_name, task_name))
    axis.set_xticks(centres, present_models, rotation=18, ha="right")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def plot_results(
    long_results: pd.DataFrame,
    output_dir: Path,
    title: str | None,
    show_outliers: bool,
    tasks: list[str],
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
        figsize=(7.2 * n_columns, 4.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for axis, task_name in zip(axes.flat, tasks):
        task_results = long_results[long_results["task"] == task_name]
        draw_task_panel(axis, task_results, task_name, show_outliers)
        axis.set_ylabel("Fitness (lower is better)")

    for axis in axes.flat[len(tasks):]:
        axis.set_visible(False)

    legend = [
        Patch(facecolor=colour, edgecolor="#333333", label=label)
        for label, colour in SPLIT_COLOURS.items()
    ]
    legend.append(
        Line2D(
            [0],
            [0],
            color="#B22222",
            linestyle="--",
            linewidth=1.4,
            label="Target",
        )
    )
    figure.legend(
        handles=legend,
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    if title:
        figure.suptitle(title, fontsize=13, y=1.025)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "training_testing_fitness_boxplots.png"
    pdf_path = output_dir / "training_testing_fitness_boxplots.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    args = parse_arguments()
    results = load_results(args.results)

    available_tasks = set(results["task"])
    if args.tasks:
        tasks_to_plot = args.tasks
        results = results[results["task"].isin(args.tasks)].copy()
        if results.empty:
            raise ValueError(
                "None of the requested tasks were found. Available tasks: "
                f"{sorted(load_results(args.results)['task'].unique())}"
            )
    else:
        # Always reserve panels for the five study tasks, including tasks for
        # which no completed results are available yet.
        tasks_to_plot = TASK_ORDER.copy()
        tasks_to_plot.extend(sorted(available_tasks.difference(tasks_to_plot)))

    long_results = reshape_results(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = args.output_dir / "combined_fitness_results.csv"
    long_results.to_csv(combined_csv, index=False)

    png_path, pdf_path = plot_results(
        long_results,
        args.output_dir,
        args.title,
        args.show_outliers,
        tasks_to_plot,
    )

    counts = (
        results.groupby(["task", "model_label"], observed=True)["seed"]
        .nunique()
        .rename("seeds")
        .reset_index()
    )
    print(counts.to_string(index=False))
    print(f"\nCombined data: {combined_csv}")
    print(f"PNG figure:    {png_path}")
    print(f"PDF figure:    {pdf_path}")


if __name__ == "__main__":
    main()
