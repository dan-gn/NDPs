"""Compare 16- and 32-node R-NDP runs using matched optimiser seeds.

Run from the repository root, after copying this file to Utilities/:
    python Utilities/plot_rndp_size_comparison.py RESULTS_16 RESULTS_32 \
        --output-dir Results/rndp_16_vs_32

Only load pickle files produced by your own experiments: pickle is not a safe
format for files from untrusted sources.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyse_node_state_diversity import (
    load_best_graph,
    mean_featurewise_std,
    scope_node_ids,
)


TASKS = [
    ("CartPole-v1", "CartPole"),
    ("popgym-PositionOnlyCartPoleEasy-v0", "PositionOnlyCartPole"),
    ("Acrobot-v1", "Acrobot"),
    ("MountainCar-v0", "MountainCar"),
]
KEY = ["task", "hebbian", "seed"]
BLUE = "#4C78A8"
ORANGE = "#F58518"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_16", type=Path, help="16-node result directory")
    parser.add_argument("results_32", type=Path, help="32-node result directory")
    parser.add_argument("--output-dir", type=Path, default=Path("Results/rndp_16_vs_32"))
    parser.add_argument("--tasks", nargs="+", help="Optional exact task IDs to include")
    parser.add_argument("--hide-outliers", action="store_true")
    return parser.parse_args()


def read_logs(root: Path, node_limit: int) -> pd.DataFrame:
    log_files = sorted(root.rglob("experiments_log.csv"))
    if not log_files:
        raise FileNotFoundError(f"No experiments_log.csv files below {root}")
    frames = []
    for path in log_files:
        frame = pd.read_csv(path)
        required = {*KEY, "model", "filename", "best_score_mean", "best_score_test"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        frame = frame.loc[frame["model"].isin(["hebbian_ndp", "rewiring_ndp"])].copy()
        if frame.empty:
            continue
        frame["source_log"] = str(path)
        frame["node_limit"] = node_limit
        frames.append(frame)
    if not frames:
        raise ValueError(f"No R-NDP results below {root}")
    results = pd.concat(frames, ignore_index=True)
    results["hebbian"] = results["hebbian"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if results["hebbian"].isna().any():
        raise ValueError(f"Invalid Hebbian flag in {root}")
    results["seed"] = pd.to_numeric(results["seed"], errors="raise").astype(int)
    if results.duplicated(KEY).any():
        duplicates = results.loc[results.duplicated(KEY, keep=False), KEY]
        raise ValueError(f"Duplicate task/Hebbian/seed rows in {root}:\n{duplicates.head()}")
    for column in ("best_score_mean", "best_score_test"):
        results[column] = pd.to_numeric(results[column], errors="raise")
    return results


def match_seeds(results_16: pd.DataFrame, results_32: pd.DataFrame) -> pd.DataFrame:
    keys_16 = results_16[KEY].assign(in_16=True)
    keys_32 = results_32[KEY].assign(in_32=True)
    matched = keys_16.merge(keys_32, on=KEY, how="inner")[KEY]
    if matched.empty:
        raise ValueError("The two configurations have no matched task/Hebbian/seed runs")
    combined = pd.concat(
        [results_16.merge(matched, on=KEY), results_32.merge(matched, on=KEY)],
        ignore_index=True,
    )
    return combined.sort_values(["task", "hebbian", "seed", "node_limit"])


def add_graph_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Reuse the best-graph and node-scope logic of the diversity analysis."""
    records = []
    for _, row in results.iterrows():
        output, graph, _ = load_best_graph(row)
        task = output["task"]
        n_inputs = int(task.parameters["graph_n_inputs"])
        n_outputs = int(task.parameters["graph_n_outputs"])
        scopes = scope_node_ids(graph, n_inputs, n_outputs)
        states = np.asarray(graph.nodes_states, dtype=float)
        if states.ndim != 2 or states.shape[0] != graph.number_of_nodes():
            raise ValueError(f"Invalid node states for {row[KEY].to_dict()}")
        record = row.to_dict()
        record["used_nodes"] = len(scopes["functional"])
        record["used_edges"] = graph.get_number_of_used_edges(n_inputs, n_outputs)
        for scope in ("all", "functional"):
            selected = states[scopes[scope]]
            record[f"state_diversity_{scope}"] = mean_featurewise_std(selected)
        records.append(record)
    return pd.DataFrame(records)


def draw_panel(ax: plt.Axes, data: pd.DataFrame, metric: str, show_outliers: bool) -> None:
    positions = [1, 2, 4, 5]
    groups = []
    for hebbian, node_limit in ((False, 16), (False, 32), (True, 16), (True, 32)):
        values = data.loc[
            (data["hebbian"] == hebbian) & (data["node_limit"] == node_limit), metric
        ].dropna().to_numpy(dtype=float)
        groups.append(values)
    plotted = [(p, values) for p, values in zip(positions, groups) if values.size]
    if plotted:
        boxes = ax.boxplot(
            [values for _, values in plotted],
            positions=[p for p, _ in plotted],
            widths=0.62,
            patch_artist=True,
            showfliers=show_outliers,
            medianprops={"color": "#202020", "linewidth": 1.3},
        )
        for box, (position, _) in zip(boxes["boxes"], plotted):
            box.set_facecolor(BLUE if position in (1, 4) else ORANGE)
            box.set_edgecolor("#444444")
            box.set_alpha(0.85)
    ax.set_xlim(0.4, 5.6)
    ax.set_xticks(positions, [f"{limit}\n(n={len(values)})" for limit, values in
                              zip((16, 32, 16, 32), groups)])
    ax.axvline(3, color="#cccccc", linewidth=0.8)
    ax.text(1.5, 1.02, "No Hebbian", ha="center", transform=ax.get_xaxis_transform(), fontsize=9)
    ax.text(4.5, 1.02, "Hebbian", ha="center", transform=ax.get_xaxis_transform(), fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_grid(results: pd.DataFrame, task_names: list[tuple[str, str]],
              metric: str, ylabel: str, title: str, out: Path,
              show_outliers: bool) -> None:
    ncols = min(2, len(task_names))
    nrows = int(np.ceil(len(task_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.4 * nrows),
                             squeeze=False, constrained_layout=True)
    for ax, (task, label) in zip(axes.flat, task_names):
        draw_panel(ax, results[results["task"] == task], metric, show_outliers)
        ax.set_title(label, pad=27, fontsize=12)
        ax.set_ylabel(ylabel)
    for ax in list(axes.flat)[len(task_names):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = match_seeds(read_logs(args.results_16, 16), read_logs(args.results_32, 32))
    chosen = TASKS if not args.tasks else [(k, v) for k, v in TASKS if k in args.tasks]
    if not chosen:
        raise ValueError("No selected tasks have a configured label")
    results = results[results["task"].isin([task for task, _ in chosen])].copy()
    if results.empty:
        raise ValueError("No matched runs for the selected tasks")
    counts = results.groupby(["task", "hebbian", "node_limit"]).size()
    print("Matched run counts:\n", counts.to_string())
    results = add_graph_metrics(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        ("best_score_mean", "Training fitness (lower is better)", "Training fitness", "training_fitness"),
        ("best_score_test", "Testing fitness (lower is better)", "Held-out testing fitness", "testing_fitness"),
        ("used_nodes", "Functional nodes", "Used nodes", "used_nodes"),
        ("used_edges", "Functional edges", "Used edges", "used_edges"),
        ("state_diversity_all", "Mean feature-wise standard deviation", "All-node state diversity", "state_diversity_all"),
        ("state_diversity_functional", "Mean feature-wise standard deviation", "Functional-node state diversity", "state_diversity_functional"),
    ]
    for metric, ylabel, title, filename in figures:
        plot_grid(results, chosen, metric, ylabel, title,
                  args.output_dir / filename, not args.hide_outliers)
    results.to_csv(args.output_dir / "matched_run_metrics.csv", index=False)
    print(f"Saved six PNG/PDF boxplots and matched_run_metrics.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
