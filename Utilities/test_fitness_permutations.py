"""Compare mean testing fitness using unpaired, two-sided permutation tests.

Run from the project root with:
    python Utilities/test_fitness_permutations.py

Holm correction is applied separately across all tasks within each family:
pairwise non-Hebbian encodings, and Hebbian versus non-Hebbian counterparts.
Lower fitness is better. Inputs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import permutation_test


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "Results/fixed_ann_ndp_rndp_comparison/combined_fitness_results.csv"
MODELS = ("DE", "NDP", "RNDP")
TASKS = (
    "CartPole-v1",
    "popgym-PositionOnlyCartPoleEasy-v0",
    "Acrobot-v1",
    "MountainCar-v0",
    "LunarLander-v3",
)


def mean_difference(first, second, axis=0):
    return np.mean(first, axis=axis) - np.mean(second, axis=axis)


def load_testing_groups(path: Path, expected_runs: int):
    groups = {}
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"task", "seed", "model_label", "evaluation_set", "fitness"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input must contain columns {sorted(required)}")
        for row in reader:
            if row["evaluation_set"].strip() != "Testing":
                continue
            model = row["model_label"].strip().replace("R-NDP", "RNDP")
            model = model.replace(" ", "")
            key = (row["task"], model)
            run = (*key, int(row["seed"]))
            if run in seen:
                raise ValueError(f"Duplicate testing run: {run}")
            seen.add(run)
            fitness = float(row["fitness"])
            if not np.isfinite(fitness):
                raise ValueError(f"Non-finite fitness for {run}")
            groups.setdefault(key, []).append((run[-1], fitness))
    expected = {(task, model + suffix) for task in TASKS
                for model in MODELS for suffix in ("", "+HL")}
    if set(groups) != expected:
        raise ValueError(f"Unexpected conditions. Missing: {expected - set(groups)}; "
                         f"extra: {set(groups) - expected}")
    for key, values in groups.items():
        if len(values) != expected_runs:
            raise ValueError(f"{key} has {len(values)} runs; expected {expected_runs}")
    return {key: np.array([value for _, value in sorted(values)])
            for key, values in groups.items()}


def holm_adjust(p_values):
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate(p_values[order] * np.arange(len(p_values), 0, -1))
    )
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--permutations", type=int, default=99_999)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--expected-runs", type=int, default=30)
    args = parser.parse_args()
    if args.permutations < 1 or args.expected_runs < 2 or not 0 < args.alpha < 1:
        parser.error("Require positive permutations, at least two runs, and 0 < alpha < 1")
    groups = load_testing_groups(args.input, args.expected_runs)
    comparisons = {
        "non_hebbian": list(itertools.combinations(MODELS, 2)),
        "hebbian_effect": [(model + "+HL", model) for model in MODELS],
    }
    random_states = np.random.SeedSequence(args.seed).spawn(30)
    rows = []
    for family, pairs in comparisons.items():
        family_rows = []
        for task in TASKS:
            for first, second in pairs:
                first_values = groups[task, first]
                second_values = groups[task, second]
                result = permutation_test(
                    (first_values, second_values), mean_difference,
                    permutation_type="independent", vectorized=True,
                    alternative="two-sided", n_resamples=args.permutations,
                    batch=2048,
                    random_state=np.random.default_rng(random_states[len(rows) + len(family_rows)]),
                )
                family_rows.append({
                    "family": family, "task": task,
                    "first": first, "second": second,
                    "n_first": len(first_values), "n_second": len(second_values),
                    "mean_first": float(first_values.mean()),
                    "mean_second": float(second_values.mean()),
                    "mean_difference_first_minus_second": float(result.statistic),
                    "p_raw": float(result.pvalue),
                })
        adjusted = holm_adjust([row["p_raw"] for row in family_rows])
        for row, p_adjusted in zip(family_rows, adjusted):
            row["p_holm"] = float(p_adjusted)
            row["significant"] = bool(p_adjusted <= args.alpha)
            difference = row["mean_difference_first_minus_second"]
            row["lower_mean_model"] = (row["first"] if difference < 0 else
                                       row["second"] if difference > 0 else "equal")
            if row["significant"]:
                print(f"{family} | {row['task']} | {row['lower_mean_model']} lower mean "
                      f"({row['first']} vs {row['second']}), Holm p={p_adjusted:.6g}")
        rows.extend(family_rows)
    output_dir = args.output_dir or args.input.parent / "permutation_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "testing_fitness_permutation_tests.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "input": str(args.input.resolve()), "permutations_per_test": args.permutations,
        "random_seed": args.seed, "alpha": args.alpha,
        "statistic": "mean(first) - mean(second)",
        "alternative": "two-sided", "permutation_type": "independent",
        "multiple_testing": "Holm, separately within each family of 15 tests",
        "p_value_convention": "SciPy: twice the smaller one-sided p, plus-one correction",
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "tests": len(rows),
    }
    (output_dir / "settings.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rows)} comparisons and settings to {output_dir}")


if __name__ == "__main__":
    main()
