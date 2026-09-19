from __future__ import annotations

import argparse
import csv
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def scalar_fitness(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def evaluate(objective_function, genotype):
    result = objective_function(genotype)
    if not isinstance(result, tuple) or len(result) < 4:
        raise ValueError(
            "Expected the objective function to return "
            "(fitness, ..., best_graph, best_graph_fitness)."
        )
    return scalar_fitness(result[0]), result[2]


def functional_edges(graph, n_inputs: int, n_outputs: int) -> set[tuple[int, int]]:
    if graph is None:
        return set()
    return set(graph.get_used_edge_ids(n_inputs, n_outputs))


def functional_nodes(graph, n_inputs: int, n_outputs: int) -> set[int]:
    if graph is None:
        return set()
    return set(graph.get_used_node_ids(n_inputs, n_outputs))


def jaccard(left: set, right: set) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def percentile_summary(values: list[float]) -> str:
    q0, q25, q50, q75, q100 = np.percentile(values, [0, 25, 50, 75, 100])
    return (
        f"min={q0:.4f}, Q25={q25:.4f}, median={q50:.4f}, "
        f"Q75={q75:.4f}, max={q100:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the sensitivity of a saved EA best genotype to mutation."
    )
    parser.add_argument("result", type=Path, help="Path to an experiment .pkl file")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("mutation_sensitivity.csv"),
        help="Where to save per-offspring measurements",
    )
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")

    with args.result.open("rb") as result_file:
        output = pickle.load(result_file)

    optimiser = output["optimiser"]
    parent = optimiser.best_individual
    parent_genotype = np.asarray(parent.genotype, dtype=float).copy()

    if parent_genotype.ndim != 1:
        raise ValueError("Expected a one-dimensional genotype.")
    if optimiser.model is None or "ndp" not in optimiser.model:
        raise ValueError("This diagnostic currently expects an NDP result.")

    n_inputs = optimiser.graph_n_inputs
    n_outputs = optimiser.graph_n_outputs

    # Re-evaluate the parent so parent and offspring use the same evaluation path.
    parent_fitness, parent_graph = evaluate(
        optimiser.objective_function,
        parent_genotype.copy(),
    )
    parent_edges = functional_edges(parent_graph, n_inputs, n_outputs)
    parent_nodes = functional_nodes(parent_graph, n_inputs, n_outputs)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = []
    for sample in range(args.samples):
        child_genotype = optimiser.mutate(parent_genotype.copy())
        changed_mask = child_genotype != parent_genotype
        changed_parameters = int(np.count_nonzero(changed_mask))
        mutation_delta = child_genotype - parent_genotype

        child_fitness, child_graph = evaluate(
            optimiser.objective_function,
            child_genotype,
        )
        child_edges = functional_edges(child_graph, n_inputs, n_outputs)
        child_nodes = functional_nodes(child_graph, n_inputs, n_outputs)

        rows.append(
            {
                "sample": sample,
                "changed_parameters": changed_parameters,
                "mutation_l2": float(np.linalg.norm(mutation_delta)),
                "mutation_max_abs": float(np.max(np.abs(mutation_delta))),
                "fitness": child_fitness,
                "fitness_delta": child_fitness - parent_fitness,
                "used_nodes": len(child_nodes),
                "used_edges": len(child_edges),
                "node_jaccard": jaccard(parent_nodes, child_nodes),
                "edge_jaccard": jaccard(parent_edges, child_edges),
                "edges_added": len(child_edges - parent_edges),
                "edges_removed": len(parent_edges - child_edges),
            }
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fitnesses = [row["fitness"] for row in rows]
    deltas = [row["fitness_delta"] for row in rows]
    changed = [row for row in rows if row["changed_parameters"] > 0]
    same_topology = [row for row in rows if row["edge_jaccard"] == 1.0]
    worse = [row for row in rows if row["fitness_delta"] > 0]
    better = [row for row in rows if row["fitness_delta"] < 0]
    equal = [row for row in rows if np.isclose(row["fitness_delta"], 0.0)]
    common_fitnesses = Counter(round(value, 4) for value in fitnesses).most_common(5)

    print(f"Result: {args.result}")
    print(f"Mutation probability: {optimiser.mutation_probability}")
    print(f"Mutation eta: {optimiser.mutation_eta}")
    print(f"Genotype parameters: {len(parent_genotype)}")
    print(f"Parent stored fitness: {scalar_fitness(parent.fitness):.4f}")
    print(f"Parent re-evaluated fitness: {parent_fitness:.4f}")
    print(f"Parent used nodes: {len(parent_nodes)}")
    print(f"Parent used edges: {len(parent_edges)}")
    print()
    print(f"Mutated samples: {args.samples}")
    print(f"Samples changing at least one parameter: {len(changed)} ({100 * len(changed) / args.samples:.1f}%)")
    print(f"Samples with identical functional topology: {len(same_topology)} ({100 * len(same_topology) / args.samples:.1f}%)")
    print(f"Better than parent: {len(better)} ({100 * len(better) / args.samples:.1f}%)")
    print(f"Equal to parent: {len(equal)} ({100 * len(equal) / args.samples:.1f}%)")
    print(f"Worse than parent: {len(worse)} ({100 * len(worse) / args.samples:.1f}%)")
    print(f"Offspring fitness: {percentile_summary(fitnesses)}")
    print(f"Fitness change: {percentile_summary(deltas)}")
    print(f"Most common offspring fitness values: {common_fitnesses}")

    if changed:
        changed_deltas = [row["fitness_delta"] for row in changed]
        changed_edge_jaccard = [row["edge_jaccard"] for row in changed]
        print(f"Changed-only fitness change: {percentile_summary(changed_deltas)}")
        print(f"Changed-only edge Jaccard: {percentile_summary(changed_edge_jaccard)}")

    print(f"Per-offspring data saved to: {args.csv}")


if __name__ == "__main__":
    # Permit unpickling project classes when invoked from Tests/.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    main()
