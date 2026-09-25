"""Create paper figures comparing standard NDP with 32-node R-NDP.

Copy this file to Utilities/ and run from the project root:

    python Utilities/plot_ndp_vs_rndp32.py NDP_RESULTS RNDP32_RESULTS \
        --output-dir Results/ndp_vs_rndp32_figures

The script loads saved .pkl results to recover functional graph sizes and
node-state diversity. Only use it with your own trusted experiment files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import plot_fitness_boxplots as fitness_plots
import plot_graph_size_boxplots as graph_size_plots
import plot_node_state_diversity_boxplots as diversity_plots
from analyse_node_state_diversity import analyse_run
from plot_graph_size_boxplots import add_used_graph_sizes


KEY = ["task", "hebbian", "seed"]
FITNESS_COLOURS = {"Training": "#4E8098", "Testing": "#D26760"}
MODEL_ORDER = ("NDP", "R-NDP 32", "NDP + HL", "R-NDP 32 + HL")
MODEL_COLOURS = {
    "NDP": "#4E8098",
    "NDP + HL": "#DDB771",
    "R-NDP": "#D26760",
    "R-NDP + HL": "#5A8646",
}
METRICS = (
    ("used_nodes", "Number of used nodes"),
    ("used_edges", "Number of used edges"),
)
SCOPES = (
    ("all", "All-node state diversity"),
    ("functional", "Functional-node state diversity"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ndp_results", type=Path, help="Results containing standard NDP runs")
    parser.add_argument("rndp32_results", type=Path, nargs="?",
                        help="Results containing 32-node R-NDP runs; defaults to the first root")
    parser.add_argument(
        "--extra-ndp-results", type=Path, action="append", default=[],
        help="Optional additional standard NDP result root (e.g. LunarLander)",
    )
    parser.add_argument(
        "--extra-rndp-results", type=Path, action="append", default=[],
        help="Optional additional 32-node R-NDP result root (e.g. LunarLander)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Results/ndp_vs_rndp32_figures")
    )
    parser.add_argument(
        "--test-summary", type=Path,
        help="100-seed re-evaluation summary CSV; defaults to the first root's "
             "posthoc_test_evaluation/best_individual_reevaluation_summary.csv",
    )
    parser.add_argument("--tasks", nargs="+", help="Optional exact task IDs")
    parser.add_argument("--hide-outliers", action="store_true")
    return parser.parse_args()


def read_condition(root: Path, model_names: set[str], label: str) -> pd.DataFrame:
    paths = sorted(root.rglob("experiments_log.csv"))
    if not paths:
        raise FileNotFoundError(f"No experiments_log.csv files below {root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {*KEY, "model", "filename", "best_score_mean", "best_score_test"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        frame = frame.loc[frame["model"].isin(model_names)].copy()
        if frame.empty:
            continue
        frame["source_log"] = str(path)
        frame["configuration"] = label
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    results = pd.concat(frames, ignore_index=True)
    results["hebbian"] = results["hebbian"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if results["hebbian"].isna().any():
        raise ValueError(f"Invalid Hebbian flag below {root}")
    results["seed"] = pd.to_numeric(results["seed"], errors="raise").astype(int)
    for metric in ("best_score_mean", "best_score_test"):
        results[metric] = pd.to_numeric(results[metric], errors="raise")
    if results.duplicated(KEY).any():
        duplicate_rows = results.loc[results.duplicated(KEY, keep=False)]
        check_columns = [
            column for column in (
                "best_score_mean", "best_score_test", "best_graph_n_nodes",
                "best_graph_used_nodes", "best_graph_used_edges",
            ) if column in results.columns
        ]
        disagreement = duplicate_rows.groupby(KEY, dropna=False)[check_columns].nunique(dropna=False)
        if disagreement.gt(1).any(axis=None):
            conflicts = disagreement.loc[disagreement.gt(1).any(axis=1)]
            raise ValueError(
                f"Duplicate {label} runs have conflicting results below {root}:\n"
                f"{conflicts}"
            )
        results["saved_result_exists"] = [
            (Path(source_log).parent / Path(str(filename).replace("\\", "/")).name).is_file()
            for source_log, filename in zip(results["source_log"], results["filename"])
        ]
        results = results.sort_values(
            ["saved_result_exists", "filename"], ascending=[False, True],
            kind="stable",
        )
        removed = int(results.duplicated(KEY).sum())
        results = results.drop_duplicates(KEY, keep="first").drop(columns="saved_result_exists")
        print(f"Collapsed {removed} repeated {label} log entries below {root}")
    return results


def load_matched_results(ndp_roots: list[Path], rndp_roots: list[Path]) -> pd.DataFrame:
    ndp_frames = [read_condition(root, {"standard_ndp"}, "NDP") for root in ndp_roots]
    ndp_frames = [frame for frame in ndp_frames if not frame.empty]
    if not ndp_frames:
        raise ValueError("No standard NDP results were found")
    ndp = pd.concat(ndp_frames, ignore_index=True)
    duplicates = ndp.duplicated(KEY)
    if duplicates.any():
        print(f"Using the first standard NDP result for {duplicates.sum()} duplicate runs")
        ndp = ndp.drop_duplicates(KEY, keep="first")

    rndp_frames = [
        read_condition(root, {"hebbian_ndp", "rewiring_ndp"}, "R-NDP 32")
        for root in rndp_roots
    ]
    rndp_frames = [frame for frame in rndp_frames if not frame.empty]
    if not rndp_frames:
        raise ValueError(f"No R-NDP results found below {rndp_roots}")
    rndp = pd.concat(rndp_frames, ignore_index=True)
    duplicates = rndp.duplicated(KEY)
    if duplicates.any():
        print(f"Using the first R-NDP result for {duplicates.sum()} duplicate runs")
        rndp = rndp.drop_duplicates(KEY, keep="first")
    if "best_graph_n_nodes" in rndp:
        counts = pd.to_numeric(rndp["best_graph_n_nodes"], errors="coerce").dropna()
        if not counts.empty and not counts.eq(32).all():
            raise ValueError("The R-NDP results include graphs that do not have 32 nodes")
    matched_keys = ndp[KEY].merge(rndp[KEY], on=KEY, how="inner")
    if matched_keys.empty:
        raise ValueError("No task/Hebbian/seed runs are shared by NDP and R-NDP 32")
    results = pd.concat(
        [ndp.merge(matched_keys, on=KEY), rndp.merge(matched_keys, on=KEY)],
        ignore_index=True,
    )
    results["model_label"] = [
        ("NDP" if config == "NDP" else "R-NDP") + (" + HL" if hebbian else "")
        for config, hebbian in zip(results["configuration"], results["hebbian"])
    ]
    return results.sort_values(["task", "hebbian", "seed", "configuration"])


def use_reevaluated_testing(results: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    if not summary_path.is_file():
        raise FileNotFoundError(f"100-seed testing summary not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = {
        "task", "model", "hebbian", "optimizer_seed", "test_seed_start",
        "test_seed_end", "n_test_rollouts", "test_fitness_mean",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"{summary_path} lacks columns: {sorted(missing)}")
    summary = summary.copy()
    summary["hebbian"] = summary["hebbian"].map(fitness_plots.parse_boolean)
    summary["seed"] = pd.to_numeric(summary["optimizer_seed"], errors="raise").astype(int)
    for column, expected in (
        ("test_seed_start", 10_000), ("test_seed_end", 10_099),
        ("n_test_rollouts", 100),
    ):
        values = pd.to_numeric(summary[column], errors="raise")
        if not values.eq(expected).all():
            raise ValueError(f"{summary_path}: {column} must be {expected} for every run")
    summary["test_fitness_mean"] = pd.to_numeric(
        summary["test_fitness_mean"], errors="raise"
    )
    keys = ["task", "model", "hebbian", "seed"]
    if summary.duplicated(keys).any():
        raise ValueError(f"Duplicate task/model/Hebbian/seed rows in {summary_path}")
    if results.duplicated(keys).any():
        raise ValueError("Duplicate task/model/Hebbian/seed rows in experiment logs")
    merged = results.merge(
        summary[keys + ["test_fitness_mean"]], on=keys,
        how="left", validate="one_to_one", indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        absent = merged.loc[merged["_merge"] != "both", keys]
        raise ValueError(
            "Some plotted runs lack a 100-seed testing result:\n"
            + absent.head(10).to_string(index=False)
        )
    merged = merged.drop(columns="_merge")
    merged["original_test_fitness"] = merged["best_score_test"]
    merged["best_score_test"] = merged["test_fitness_mean"]
    print(f"Using 100-seed testing fitness for {len(merged)} runs from {summary_path}")
    return merged


def save_fitness_figure(results: pd.DataFrame, tasks: list[str], output_dir: Path,
                        show_outliers: bool) -> None:
    long_results = fitness_plots.reshape_results(results)
    columns = min(2, len(tasks))
    rows = int(np.ceil(len(tasks) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(7.8 * columns, 4.7 * rows),
                                constrained_layout=True, squeeze=False)
    previous_order = fitness_plots.MODEL_ORDER
    fitness_plots.MODEL_ORDER = list(MODEL_ORDER)
    try:
        for axis, task in zip(axes.flat, tasks):
            panel = long_results.loc[long_results["task"] == task]
            fitness_plots.draw_task_panel(axis, panel, task, show_outliers)
            present = [label for label in MODEL_ORDER if label in set(panel["model_label"])]
            if not present:
                present = list(MODEL_ORDER)
            counts = results.loc[results["task"] == task].groupby("model_label")["seed"].nunique()
            axis.set_xticklabels(
                [f"{label}\n(n={counts.get(label, 0)})" for label in present],
                rotation=20, ha="right",
            )
            axis.set_ylabel("Fitness (lower is better)")
    finally:
        fitness_plots.MODEL_ORDER = previous_order
    for axis in list(axes.flat)[len(tasks):]:
        axis.axis("off")
    figure.legend(
        handles=[
            Patch(facecolor=fitness_plots.SPLIT_COLOURS[name], edgecolor="#333333", label=name)
            for name in ("Training", "Testing")
        ] + [Line2D([0], [0], color="#B22222", linestyle="--", label="Target")],
        loc="outside lower center", ncol=3, frameon=False,
    )
    figure.suptitle("Training and testing fitness: NDP vs R-NDP 32", fontsize=15)
    save_figure(figure, output_dir / "training_testing_fitness_boxplots")


def draw_grouped_boxplots(axis: plt.Axes, results: pd.DataFrame, tasks: list[str],
                          metric: str, show_outliers: bool) -> None:
    offsets = dict(zip(MODEL_ORDER, (-0.30, -0.10, 0.10, 0.30)))
    for centre, task in enumerate(tasks):
        task_results = results.loc[results["task"] == task]
        for label in MODEL_ORDER:
            values = task_results.loc[task_results["model_label"] == label, metric].dropna()
            if values.empty:
                continue
            boxes = axis.boxplot(
                [values.to_numpy(dtype=float)], positions=[centre + offsets[label]],
                widths=0.16, patch_artist=True, showfliers=show_outliers,
                medianprops={"color": "black", "linewidth": 1.3},
            )
            boxes["boxes"][0].set_facecolor(MODEL_COLOURS[label])
            boxes["boxes"][0].set_edgecolor("#333333")
    axis.set_xticks(range(len(tasks)), [fitness_plots.TASK_LABELS.get(t, t) for t in tasks])
    axis.set_xlim(-0.65, len(tasks) - 0.35)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for centre, task in enumerate(tasks):
        if results.loc[results["task"] == task].empty:
            axis.text(centre, 0.5, "Pending", ha="center", va="center",
                      transform=axis.get_xaxis_transform(), color="#777777")


def save_two_row_figure(results: pd.DataFrame, tasks: list[str],
                        rows: tuple[tuple[str, str], ...], title: str,
                        destination: Path, show_outliers: bool) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(max(10, 2.4 * len(tasks)), 8.4),
                                sharex=True, constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes, rows):
        draw_grouped_boxplots(axis, results, tasks, metric, show_outliers)
        axis.set_ylabel(ylabel)
    axes[0].tick_params(labelbottom=False)
    figure.legend(
        handles=[Patch(facecolor=MODEL_COLOURS[label], edgecolor="#333333", label=label)
                 for label in MODEL_ORDER],
        loc="outside lower center", ncol=4, frameon=False,
    )
    figure.suptitle(title, fontsize=15)
    save_figure(figure, destination)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def add_diversity(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        scopes = {item["scope"]: item for item in analyse_run(row, threshold=1e-3)}
        record = row.to_dict()
        record["diversity_all"] = scopes["all"]["mean_featurewise_std"]
        record["diversity_functional"] = scopes["functional"]["mean_featurewise_std"]
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_arguments()
    ndp_roots = [args.ndp_results, *args.extra_ndp_results]
    rndp_roots = [args.rndp32_results or args.ndp_results, *args.extra_rndp_results]
    results = load_matched_results(ndp_roots, rndp_roots)
    tasks = args.tasks or fitness_plots.TASK_ORDER
    unsupported = set(tasks) - set(fitness_plots.TASK_LABELS)
    if unsupported:
        raise ValueError(f"Unknown task IDs: {sorted(unsupported)}")
    results = results.loc[results["task"].isin(tasks)].copy()
    if results.empty:
        raise ValueError("No matched runs for the selected tasks")
    summary_path = args.test_summary or (
        args.ndp_results / "posthoc_test_evaluation"
        / "best_individual_reevaluation_summary.csv"
    )
    results = use_reevaluated_testing(results, summary_path)
    print("Matched run counts:\n", results.groupby(["task", "model_label"]).size().to_string())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_fitness_colours = fitness_plots.SPLIT_COLOURS
    fitness_plots.SPLIT_COLOURS = FITNESS_COLOURS
    try:
        fitness_plots.plot_results(
            fitness_plots.reshape_results(results), args.output_dir, None,
            not args.hide_outliers, tasks,
        )
    finally:
        fitness_plots.SPLIT_COLOURS = previous_fitness_colours
    results = add_used_graph_sizes(results)
    previous_graph_colours = graph_size_plots.MODEL_COLOURS
    graph_size_plots.MODEL_COLOURS = MODEL_COLOURS
    try:
        graph_size_plots.create_figure(results, tasks, args.output_dir, not args.hide_outliers)
    finally:
        graph_size_plots.MODEL_COLOURS = previous_graph_colours
    results = add_diversity(results)
    diversity_rows = []
    for scope in ("all", "functional"):
        scoped = results.copy()
        scoped["scope"] = scope
        scoped["mean_featurewise_std"] = scoped[f"diversity_{scope}"]
        diversity_rows.append(scoped)
    diversity_results = pd.concat(diversity_rows, ignore_index=True)
    previous_diversity_colours = diversity_plots.MODEL_COLOURS
    diversity_plots.MODEL_COLOURS = MODEL_COLOURS
    try:
        for scope in ("all", "functional"):
            diversity_plots.create_scope_figure(
                diversity_results, tasks, scope, "mean_featurewise_std",
                args.output_dir, not args.hide_outliers,
            )
    finally:
        diversity_plots.MODEL_COLOURS = previous_diversity_colours
    results.to_csv(args.output_dir / "matched_run_metrics.csv", index=False)
    print(f"Saved original-style fitness, graph-size, and diversity figures to {args.output_dir}")


if __name__ == "__main__":
    main()
