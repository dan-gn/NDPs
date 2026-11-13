"""Render graph size and node-state diversity dot plots from matched runs."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from plot_graph_size_boxplots import MODEL_COLOURS, TASK_LABELS, TASK_ORDER


def plot_metrics(results, metrics, output):
    figure, axes = plt.subplots(
        1, len(metrics), figsize=(4.8 * len(metrics) + 1, 6.2),
        sharey=True, squeeze=False, layout="constrained",
    )
    rows = np.arange(len(TASK_ORDER))
    offsets = np.linspace(-0.30, 0.30, len(MODEL_COLOURS))
    grouped = results.groupby(["task", "model_label"])
    for axis, (metric, label) in zip(axes.flat, metrics):
        for row, task in enumerate(TASK_ORDER):
            if metric.startswith("diversity_") and row % 2 == 0:
                axis.axhspan(row - 0.5, row + 0.5, color="#F8F8F8",
                             linewidth=0, zorder=-1)
            for offset, (model, colour) in zip(offsets, MODEL_COLOURS.items()):
                if (task, model) not in grouped.groups:
                    continue
                values = grouped.get_group((task, model))[metric].dropna()
                if values.empty:
                    continue
                lower, median, upper = values.quantile([0.25, 0.5, 0.75])
                y = row + offset
                if metric.startswith("diversity_"):
                    if model != "NDP" and (task, "NDP") in grouped.groups:
                        reference = grouped.get_group((task, "NDP"))[metric].dropna().median()
                        percentage = (median / reference - 1) * 100 if reference > 0 else np.nan
                        text = f"{percentage:+.1f}%" if np.isfinite(percentage) else "undefined"
                        axis.annotate(
                            text, (upper, y), xytext=(4, 0),
                            textcoords="offset points", color=colour,
                            ha="left", va="center", fontsize=8,
                        )
                else:
                    axis.hlines(y, values.min(), values.max(), color=colour,
                                linewidth=2, alpha=0.45, zorder=1)
                axis.hlines(y, lower, upper, color=colour, linewidth=2, zorder=2)
                axis.plot(median, y, "o", color=colour, markersize=5, zorder=3)
        if metric.startswith("diversity_"):
            axis.margins(x=0.22)
        axis.set_xlabel(label + ("\nPercentages show median change relative to NDP"
                                 if metric.startswith("diversity_") else ""))
        axis.grid(axis="x", color="#E5E5E5", alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if metric in ("used_nodes", "used_edges"):
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].set_yticks(rows, [TASK_LABELS[task] for task in TASK_ORDER])
    axes[0, 0].set_ylim(len(TASK_ORDER) - 0.5, -0.5)
    figure.legend(
        handles=[Line2D([], [], color=colour, marker="o", label=model.replace("R-NDP", "RNDP"))
                 for model, colour in MODEL_COLOURS.items()],
        loc="outside upper center", ncols=len(MODEL_COLOURS), frameon=False,
    )
    for extension in (".png", ".pdf"):
        figure.savefig(output.with_suffix(extension), dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Matched run metrics CSV")
    parser.add_argument("--diversity-only", action="store_true")
    args = parser.parse_args()
    results = pd.read_csv(args.results)
    output = args.results.parent
    if not args.diversity_only:
        plot_metrics(results, [("used_nodes", "Number of used nodes"),
                           ("used_edges", "Number of used edges")],
                 output / "best_graph_size_dotplots")
    for scope, label in (("all", "All nodes"), ("functional", "Functional nodes")):
        plot_metrics(results, [(f"diversity_{scope}", f"Node-state diversity ({label.lower()})")],
                     output / f"node_state_diversity_{scope}_dotplot")


if __name__ == "__main__":
    main()
