"""Plot signed fitness gaps and report target attainment from combined results."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from plot_fitness_boxplots import MODEL_ORDER, SPLIT_COLOURS, TASK_LABELS, TASK_ORDER, TASK_TARGETS


MODEL_PALETTE = {
    "DE": "#67458A",
    "DE + HL": "#943D70",
    "NDP": "#4C78A8",
    "NDP + HL": "#72B7B2",
    "R-NDP": "#F58518",
    "R-NDP + HL": "#E45756",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Combined fitness results CSV")
    parser.add_argument("--raw", action="store_true", help="Plot original fitness scores")
    args = parser.parse_args()
    results = pd.read_csv(args.results)
    results["gap"] = results["fitness"] - results["task"].map(TASK_TARGETS)
    results["attained"] = results["gap"] <= 0
    scale = results.groupby("task")["gap"].transform(lambda gaps: gaps.abs().max())
    results["normalised_gap"] = results["gap"] / scale.mask(scale == 0, 1)

    score_column = "fitness" if args.raw else "normalised_gap"
    if args.raw:
        figure = plt.figure(figsize=(14, 6.3), layout="constrained")
        figure.get_layout_engine().set(w_pad=0.005, wspace=0)
        grid = figure.add_gridspec(2, 6)
        panels = [grid[0, 1:3], grid[0, 3:5],
                  grid[1, :2], grid[1, 2:4], grid[1, 4:]]
    else:
        figure = plt.figure(figsize=(10, 8.5), layout="constrained")
        grid = figure.add_gridspec(3, 4)
        panels = [grid[0, :2], grid[0, 2:], grid[1, :2], grid[1, 2:], grid[2, 1:3]]
    for task, panel in zip(TASK_ORDER, panels):
        axis = figure.add_subplot(panel)
        task_results = results[results["task"] == task]
        for row, model in enumerate(MODEL_ORDER):
            model_results = task_results[task_results["model_label"] == model]
            for split, offset, marker in (("Training", -0.15, "o"), ("Testing", 0.15, "D")):
                split_results = model_results[model_results["evaluation_set"] == split]
                values = split_results[score_column]
                if values.empty:
                    continue
                lower, median, upper = values.quantile([0.25, 0.5, 0.75])
                opacity = 0.5 if split == "Training" else 1.0
                colour = SPLIT_COLOURS[split]
                if args.raw:
                    colour = MODEL_PALETTE[model]
                    attained = split_results["attained"].sum()
                    axis.annotate(
                        str(attained), (upper, row + offset),
                        xytext=(4, 0), textcoords="offset points",
                        color=colour, ha="left", va="center", fontsize=8,
                    )
                axis.hlines(row + offset, lower, upper, color=colour, linewidth=2, alpha=opacity)
                axis.plot(median, row + offset, marker, color=colour, markersize=5, alpha=opacity)
        axis.axvline(TASK_TARGETS[task] if args.raw else 0, color="#4A4A4A", linestyle="--", linewidth=1, alpha=0.6)
        axis.set(title=TASK_LABELS[task], ylim=(len(MODEL_ORDER) - 0.5, -0.5))
        if args.raw:
            axis.margins(x=0.10)
            axis.set_xlabel("Fitness")
        else:
            axis.set_xlim(-max(0.05, -results["normalised_gap"].min()) - 0.02, 1.03)
        axis.set_yticks(range(len(MODEL_ORDER)),
                        [model.replace("R-NDP", "RNDP") for model in MODEL_ORDER])
        if args.raw and task not in (TASK_ORDER[0], TASK_ORDER[2]):
            axis.tick_params(axis="y", labelleft=False)
        axis.grid(axis="x", color="#E5E5E5", alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([], [], color="#444444" if args.raw else SPLIT_COLOURS["Training"], marker="o", alpha=0.5, label="Training"),
        Line2D([], [], color="#444444" if args.raw else SPLIT_COLOURS["Testing"], marker="D", label="Testing"),
        Line2D([], [], color="#4A4A4A", linestyle="--", linewidth=1, alpha=0.6, label="Target"),
    ]
    if args.raw:
        handles.append(Line2D([], [], linestyle="none", marker=r"$n$",
                              color="#4A4A4A", label="Runs that met the target"))
    figure.legend(handles=handles, loc="outside upper center", ncols=len(handles), frameon=False)
    if not args.raw:
        xlabel = figure.supxlabel(
            "Normalised fitness gap (negative values beat the target)"
        )
    figure.canvas.draw()
    if args.raw:
        figure.set_layout_engine("none")
        for row_axes in (figure.axes[:2], figure.axes[2:]):
            positions = [axis.get_position().frozen() for axis in row_axes]
            gaps = [right.x0 - left.x1 for left, right in zip(positions, positions[1:])]
            shift = sum(gaps) / 4
            for index, (axis, position) in enumerate(zip(row_axes, positions)):
                axis.set_position(position.translated(shift, 0))
                if index < len(gaps):
                    shift -= gaps[index] / 2
    if not args.raw:
        left = min(axis.get_position().x0 for axis in figure.axes)
        right = max(axis.get_position().x1 for axis in figure.axes)
        xlabel.set_x((left + right) / 2)

    filename = "raw_fitness_dotplot_with_attainment" if args.raw else "normalised_fitness_dotplot"
    save_options = {"dpi": 300}
    if args.raw:
        figure.canvas.draw()
        bounds = figure.get_tightbbox(figure.canvas.get_renderer())
        save_options["bbox_inches"] = Bbox.from_extents(
            bounds.x0, 0, bounds.x1, figure.get_figheight()
        )
    for extension in ("png", "pdf"):
        figure.savefig(args.results.with_name(f"{filename}.{extension}"), **save_options)
    plt.close(figure)

    counts = results.groupby(["task", "model_label", "evaluation_set"])["attained"].agg(
        attained="sum", total="size"
    )
    counts.to_csv(args.results.with_name("target_attainment_counts.csv"))
    print(counts.assign(count=counts.attained.astype(str) + "/" + counts.total.astype(str))
          ["count"].unstack().to_string())


if __name__ == "__main__":
    main()
