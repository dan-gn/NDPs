"""Plot median functional graph sizes with quartile annotations."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_graph_size_boxplots import MODEL_COLOURS, TASK_LABELS, TASK_ORDER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Matched run metrics CSV")
    args = parser.parse_args()
    results = pd.read_csv(args.results)
    models = list(MODEL_COLOURS)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    figure.get_layout_engine().set(wspace=0.02)
    for axis, metric, title in zip(
        axes, ("used_nodes", "used_edges"), ("Used nodes", "Used edges")
    ):
        grouped = results.groupby(["task", "model_label"])[metric]
        quartiles = grouped.quantile([0.25, 0.5, 0.75]).unstack()
        order = pd.MultiIndex.from_product([TASK_ORDER, models])
        quartiles = quartiles.reindex(order)
        medians = quartiles[0.5].to_numpy().reshape(len(TASK_ORDER), len(models))
        heatmap = axis.imshow(np.ma.masked_invalid(medians), cmap="YlGnBu", aspect="auto", vmin=0)
        axis.set_title(title, fontsize=12)
        axis.set_xticks(range(len(models)),
                         [model.replace("R-NDP", "RNDP") for model in models],
                         rotation=18, ha="right")
        axis.set_yticks(range(len(TASK_ORDER)), [TASK_LABELS[task] for task in TASK_ORDER])
        if axis is axes[1]:
            axis.tick_params(axis="y", labelleft=False)
        axis.set_xticks(np.arange(len(models) + 1) - 0.5, minor=True)
        axis.set_yticks(np.arange(len(TASK_ORDER) + 1) - 0.5, minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.spines[:].set_visible(False)
        for row, task in enumerate(TASK_ORDER):
            for column, model in enumerate(models):
                lower, median, upper = quartiles.loc[(task, model), [0.25, 0.5, 0.75]]
                if pd.isna(median):
                    axis.text(column, row, "No data", ha="center", va="center",
                              color="#555555", fontsize=9)
                    continue
                red, green, blue, _ = heatmap.cmap(heatmap.norm(median))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                colour = "white" if luminance < 0.5 else "#222222"
                axis.text(column, row - 0.12, f"{median:g}",
                          ha="center", va="center", color=colour,
                          fontsize=10, fontweight="bold")
                axis.text(column, row + 0.15, f"[{lower:g}, {upper:g}]",
                          ha="center", va="center", color=colour, fontsize=8)
        figure.colorbar(heatmap, ax=axis, label="Median count", shrink=0.9)
    for extension in ("png", "pdf"):
        path = args.results.with_name(f"best_graph_size_heatmaps.{extension}")
        figure.savefig(path, dpi=300)
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
