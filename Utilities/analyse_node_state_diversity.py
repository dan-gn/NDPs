"""Analyse node-state convergence in the best graph from each EA run."""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plot_fitness_boxplots import MODEL_ORDER, TASK_LABELS, TASK_ORDER, load_results


SCOPES = {
    "all": "All nodes",
    "functional": "Functional nodes",
    "functional_hidden": "Functional hidden nodes",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure node-state convergence in saved best individuals."
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Root result directory containing experiment logs and .pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/node_state_diversity"),
        help="Directory for the run-level data and summary tables.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="A graph is near-converged below this diversity value.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional task names to include. By default, all available tasks are used.",
    )
    return parser.parse_args()


def local_result_file(row: pd.Series) -> Path:
    filename = Path(str(row["filename"]).replace("\\", "/")).name
    return Path(row["source_log"]).parent / filename


def load_best_graph(row: pd.Series):
    result_file = local_result_file(row)
    if not result_file.exists():
        raise FileNotFoundError(
            f"Saved result not found for seed {row['seed']}: {result_file}"
        )
    with result_file.open("rb") as handle:
        output = pickle.load(handle)

    optimiser = output["optimiser"]
    graph = optimiser.best_individual.best_graph
    if graph is None:
        raise ValueError(f"No best graph was stored in {result_file}.")
    return output, graph, result_file


def mean_featurewise_std(states: np.ndarray) -> float:
    """Mean sample standard deviation across node-state dimensions."""
    if states.shape[0] < 2:
        return math.nan
    return float(np.std(states, axis=0, ddof=1).mean())


def mean_pairwise_distance(states: np.ndarray) -> float:
    """Mean Euclidean distance, normalised by sqrt(state dimension)."""
    n_nodes, state_dim = states.shape
    if n_nodes < 2:
        return math.nan
    differences = states[:, None, :] - states[None, :, :]
    distances = np.linalg.norm(differences, axis=2) / math.sqrt(state_dim)
    upper_triangle = distances[np.triu_indices(n_nodes, k=1)]
    return float(upper_triangle.mean())


def scope_node_ids(graph, n_inputs: int, n_outputs: int) -> dict[str, list[int]]:
    n_nodes = graph.number_of_nodes()
    all_ids = list(range(n_nodes))
    functional_ids = list(graph.get_used_node_ids(n_inputs, n_outputs))

    input_ids = set(range(min(n_inputs, n_nodes)))
    effective_outputs = min(n_outputs, max(0, n_nodes - n_inputs))
    output_ids = set(range(n_nodes - effective_outputs, n_nodes))
    functional_hidden_ids = [
        node_id
        for node_id in functional_ids
        if node_id not in input_ids and node_id not in output_ids
    ]
    return {
        "all": all_ids,
        "functional": functional_ids,
        "functional_hidden": functional_hidden_ids,
    }


def analyse_run(row: pd.Series, threshold: float) -> list[dict[str, object]]:
    output, graph, result_file = load_best_graph(row)
    states = np.asarray(graph.nodes_states, dtype=float)
    if states.ndim != 2 or states.shape[0] != graph.number_of_nodes():
        raise ValueError(
            f"Invalid node-state matrix {states.shape} in {result_file}."
        )

    task = output["task"]
    n_inputs = int(task.parameters["graph_n_inputs"])
    n_outputs = int(task.parameters["graph_n_outputs"])
    ids_by_scope = scope_node_ids(graph, n_inputs, n_outputs)

    records = []
    for scope, node_ids in ids_by_scope.items():
        selected_states = states[node_ids] if node_ids else states[:0]
        diversity = mean_featurewise_std(selected_states)
        pairwise = mean_pairwise_distance(selected_states)
        records.append(
            {
                "task": row["task"],
                "seed": int(row["seed"]),
                "model": row["model"],
                "hebbian": bool(row["hebbian"]),
                "model_label": row["model_label"],
                "scope": scope,
                "scope_label": SCOPES[scope],
                "n_nodes_in_scope": len(node_ids),
                "state_dimension": states.shape[1],
                "mean_featurewise_std": diversity,
                "mean_pairwise_distance": pairwise,
                "near_converged": bool(diversity < threshold)
                if np.isfinite(diversity)
                else np.nan,
                "threshold": threshold,
                "result_file": str(result_file),
            }
        )
    return records


def summarise(run_results: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["task", "model_label", "scope", "scope_label"]
    summary = (
        run_results.groupby(group_columns, observed=True, sort=False)
        .agg(
            runs=("mean_featurewise_std", "count"),
            diversity_mean=("mean_featurewise_std", "mean"),
            diversity_std=("mean_featurewise_std", "std"),
            diversity_q1=("mean_featurewise_std", lambda x: x.quantile(0.25)),
            diversity_median=("mean_featurewise_std", "median"),
            diversity_q3=("mean_featurewise_std", lambda x: x.quantile(0.75)),
            pairwise_mean=("mean_pairwise_distance", "mean"),
            pairwise_median=("mean_pairwise_distance", "median"),
            near_convergence_rate=("near_converged", "mean"),
        )
        .reset_index()
    )
    return summary


def format_number(value: float) -> str:
    return "--" if not np.isfinite(value) else f"{value:.4f}"


def make_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    available_tasks = set(summary["task"])
    ordered_tasks = [task for task in TASK_ORDER if task in available_tasks]
    ordered_tasks.extend(sorted(available_tasks.difference(ordered_tasks)))

    for task in ordered_tasks:
        for model in MODEL_ORDER:
            model_rows = summary[
                (summary["task"] == task)
                & (summary["model_label"] == model)
            ]
            if model_rows.empty:
                continue

            row: dict[str, object] = {
                "Task": TASK_LABELS.get(task, task),
                "Model": model,
            }
            for scope, prefix in (("all", "All nodes"), ("functional", "Functional nodes")):
                selected = model_rows[model_rows["scope"] == scope]
                if selected.empty:
                    row[f"{prefix} diversity"] = "--"
                    row[f"{prefix} converged"] = "--"
                    continue
                values = selected.iloc[0]
                row[f"{prefix} diversity"] = (
                    f"{format_number(values['diversity_median'])} "
                    f"[{format_number(values['diversity_q1'])}, "
                    f"{format_number(values['diversity_q3'])}]"
                )
                rate = values["near_convergence_rate"]
                row[f"{prefix} converged"] = (
                    "--" if not np.isfinite(rate) else f"{100 * rate:.1f}%"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def escape_latex(value: object) -> str:
    return str(value).replace("%", r"\%").replace("_", r"\_")


def write_latex_table(paper_table: pd.DataFrame, output: Path, threshold: float) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Node-state diversity of the best evolved graphs. Values "
            r"are medians [interquartile ranges] across runs. Converged gives "
            rf"the percentage of runs with diversity below ${threshold:g}$.}}"
        ),
        r"\label{tab:node-state-diversity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"\textbf{Task} & \textbf{Model} & "
            r"\textbf{All-node diversity} & \textbf{Converged} & "
            r"\textbf{Functional-node diversity} & \textbf{Converged} \\"
        ),
        r"\midrule",
    ]
    previous_task = None
    for _, row in paper_table.iterrows():
        task = escape_latex(row["Task"])
        displayed_task = task if task != previous_task else ""
        previous_task = task
        lines.append(
            " & ".join(
                [
                    displayed_task,
                    escape_latex(row["Model"]),
                    escape_latex(row["All nodes diversity"]),
                    escape_latex(row["All nodes converged"]),
                    escape_latex(row["Functional nodes diversity"]),
                    escape_latex(row["Functional nodes converged"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    if args.threshold <= 0:
        raise ValueError("--threshold must be greater than zero.")

    results = load_results(args.results)
    if args.tasks:
        results = results[results["task"].isin(args.tasks)].copy()
        if results.empty:
            raise ValueError("None of the requested tasks were found.")

    records = []
    total = len(results)
    for position, (_, row) in enumerate(results.iterrows(), start=1):
        print(
            f"[{position}/{total}] {row['task']} | {row['model_label']} | "
            f"seed={int(row['seed'])}"
        )
        records.extend(analyse_run(row, args.threshold))

    run_results = pd.DataFrame(records)
    summary = summarise(run_results)
    paper_table = make_paper_table(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "node_state_diversity_runs.csv"
    summary_path = args.output_dir / "node_state_diversity_summary.csv"
    paper_csv_path = args.output_dir / "node_state_diversity_table.csv"
    latex_path = args.output_dir / "node_state_diversity_table.tex"

    run_results.to_csv(run_path, index=False)
    summary.to_csv(summary_path, index=False)
    paper_table.to_csv(paper_csv_path, index=False)
    write_latex_table(paper_table, latex_path, args.threshold)

    print("\nPaper table:")
    print(paper_table.to_string(index=False))
    print(f"\nRun-level results: {run_path}")
    print(f"Full summary:      {summary_path}")
    print(f"Paper table CSV:   {paper_csv_path}")
    print(f"Paper table LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
