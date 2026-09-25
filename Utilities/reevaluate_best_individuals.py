"""Post-hoc reevaluation of saved best individuals on a larger test set."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plot_fitness_boxplots import MODEL_LABELS, parse_boolean


DEFAULT_TEST_SEED = 10_000
SUMMARY_FILENAME = "best_individual_reevaluation_summary.csv"
ROLLOUT_FILENAME = "best_individual_reevaluation_rollouts.csv"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reevaluate each saved EA best individual on a larger, common set "
            "of held-out environment seeds."
        )
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Root directory containing experiments_log.csv and result .pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to RESULTS/posthoc_test_evaluation.",
    )
    parser.add_argument(
        "--n-test-rollouts",
        type=int,
        default=100,
        help="Number of held-out episodes per best individual (default: 100).",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=DEFAULT_TEST_SEED,
        help="First held-out environment seed (default: 10000).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional exact environment names to include.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["standard_ndp", "rewiring_ndp", "hebbian_ndp"],
        help="Optional stored model identifiers to include.",
    )
    parser.add_argument(
        "--policy-hebbian",
        nargs="+",
        choices=["true", "false"],
        help="Optional policy-level Hebbian conditions to include.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of runs, useful for a smoke test.",
    )
    return parser.parse_args()


def discover_logs(results_root: Path) -> pd.DataFrame:
    log_files = sorted(results_root.rglob("experiments_log.csv"))
    if not log_files:
        raise FileNotFoundError(
            f"No experiments_log.csv files were found below {results_root}."
        )

    frames = []
    for log_file in log_files:
        frame = pd.read_csv(log_file)
        required = {"filename", "algorithm", "task", "seed", "model", "hebbian"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{log_file} is missing columns: {sorted(missing)}")
        frame["source_log"] = str(log_file)
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    results["hebbian"] = results["hebbian"].map(parse_boolean)
    results["seed"] = pd.to_numeric(results["seed"], errors="raise").astype(int)
    return results


def local_result_file(row: pd.Series) -> Path:
    filename = Path(str(row["filename"]).replace("\\", "/")).name
    return Path(row["source_log"]).parent / filename


def run_key(row: pd.Series, test_seed: int, n_rollouts: int) -> tuple:
    return (
        str(row["task"]),
        str(row["model"]),
        bool(row["hebbian"]),
        int(row["seed"]),
        int(test_seed),
        int(n_rollouts),
    )


def load_completed_keys(summary_path: Path) -> set[tuple]:
    if not summary_path.exists():
        return set()
    summary = pd.read_csv(summary_path)
    required = {
        "task",
        "model",
        "hebbian",
        "optimizer_seed",
        "test_seed_start",
        "n_test_rollouts",
    }
    if not required.issubset(summary.columns):
        return set()
    return {
        (
            str(row.task),
            str(row.model),
            parse_boolean(row.hebbian),
            int(row.optimizer_seed),
            int(row.test_seed_start),
            int(row.n_test_rollouts),
        )
        for row in summary.itertuples(index=False)
    }


def append_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def reevaluate(row: pd.Series, n_rollouts: int, test_seed: int):
    result_file = local_result_file(row)
    if not result_file.exists():
        raise FileNotFoundError(f"Saved result not found: {result_file}")

    with result_file.open("rb") as handle:
        output = pickle.load(handle)

    if str(row["algorithm"]).upper() != "EA":
        raise ValueError(
            f"Only EA results are supported, received {row['algorithm']} in {result_file}."
        )

    task = output["task"]
    optimiser = output["optimiser"]
    best_individual = optimiser.best_individual
    graph = best_individual.best_graph
    if graph is None:
        raise ValueError(f"The best individual in {result_file} has no stored graph.")

    # Evaluate the exact graph selected during optimisation. This avoids any
    # additional developmental randomness during the post-hoc comparison.
    test_fitness, rollout_fitness = task.evaluate_graph(
        graph,
        n_rollouts=n_rollouts,
        env_seed=test_seed,
        hebbian=bool(row["hebbian"]),
    )
    rollout_fitness = np.asarray(rollout_fitness, dtype=float)
    cumulative_returns = -rollout_fitness

    standard_error = (
        float(np.std(rollout_fitness, ddof=1) / np.sqrt(n_rollouts))
        if n_rollouts > 1
        else np.nan
    )
    ci_half_width = 1.96 * standard_error if np.isfinite(standard_error) else np.nan
    model_label = MODEL_LABELS.get(
        (str(row["model"]), bool(row["hebbian"])),
        ("R-NDP + HL" if bool(row["hebbian"]) else "R-NDP")
        if str(row["model"]) == "rewiring_ndp" else str(row["model"]),
    )

    original_test_fitness = pd.to_numeric(
        pd.Series([row.get("best_score_test", np.nan)]), errors="coerce"
    ).iloc[0]
    target = getattr(task, "target", np.nan)
    summary = {
        "task": str(row["task"]),
        "model": str(row["model"]),
        "hebbian": bool(row["hebbian"]),
        "model_label": model_label,
        "optimizer_seed": int(row["seed"]),
        "test_seed_start": int(test_seed),
        "test_seed_end": int(test_seed + n_rollouts - 1),
        "n_test_rollouts": int(n_rollouts),
        "original_test_fitness": original_test_fitness,
        "test_fitness_mean": float(test_fitness),
        "test_fitness_std": float(np.std(rollout_fitness, ddof=1))
        if n_rollouts > 1
        else np.nan,
        "test_fitness_standard_error": standard_error,
        "test_fitness_ci95_low": float(test_fitness - ci_half_width),
        "test_fitness_ci95_high": float(test_fitness + ci_half_width),
        "test_fitness_median": float(np.median(rollout_fitness)),
        "test_fitness_q1": float(np.quantile(rollout_fitness, 0.25)),
        "test_fitness_q3": float(np.quantile(rollout_fitness, 0.75)),
        "test_return_mean": float(np.mean(cumulative_returns)),
        "test_return_std": float(np.std(cumulative_returns, ddof=1))
        if n_rollouts > 1
        else np.nan,
        "target_fitness": target,
        "target_achieved": bool(test_fitness <= target)
        if np.isfinite(target)
        else np.nan,
        "result_file": str(result_file),
    }

    rollout_rows = pd.DataFrame(
        {
            "task": str(row["task"]),
            "model": str(row["model"]),
            "hebbian": bool(row["hebbian"]),
            "model_label": model_label,
            "optimizer_seed": int(row["seed"]),
            "environment_seed": np.arange(test_seed, test_seed + n_rollouts),
            "fitness": rollout_fitness,
            "cumulative_return": cumulative_returns,
        }
    )
    return summary, rollout_rows


def sort_outputs(summary_path: Path, rollout_path: Path) -> None:
    summary = pd.read_csv(summary_path)
    summary = summary.sort_values(
        ["task", "model_label", "optimizer_seed", "test_seed_start"]
    )
    summary.to_csv(summary_path, index=False)

    rollouts = pd.read_csv(rollout_path)
    rollouts = rollouts.sort_values(
        ["task", "model_label", "optimizer_seed", "environment_seed"]
    )
    rollouts.to_csv(rollout_path, index=False)


def main() -> None:
    args = parse_arguments()
    if args.n_test_rollouts < 1:
        raise ValueError("--n-test-rollouts must be at least 1.")

    results = discover_logs(args.results)
    results = results[results["algorithm"].str.upper() == "EA"].copy()
    if args.tasks:
        results = results[results["task"].isin(args.tasks)]
    if args.models:
        results = results[results["model"].isin(args.models)]
    if args.policy_hebbian:
        requested = {value == "true" for value in args.policy_hebbian}
        results = results[results["hebbian"].isin(requested)]

    duplicate_columns = ["task", "model", "hebbian", "seed"]
    duplicates = results.duplicated(duplicate_columns, keep=False)
    if duplicates.any():
        examples = results.loc[duplicates, duplicate_columns].drop_duplicates()
        raise ValueError(
            "Multiple saved runs share the same task/model/seed identity. "
            "Use a directory containing one study configuration. Examples:\n"
            + examples.head(10).to_string(index=False)
        )

    results = results.sort_values(duplicate_columns).reset_index(drop=True)
    if args.limit is not None:
        results = results.head(args.limit)
    if results.empty:
        raise ValueError("No matching EA results were found.")

    output_dir = args.output_dir or args.results / "posthoc_test_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    rollout_path = output_dir / ROLLOUT_FILENAME
    completed = load_completed_keys(summary_path)

    pending = [
        (_, row)
        for _, row in results.iterrows()
        if run_key(row, args.test_seed, args.n_test_rollouts) not in completed
    ]
    print(
        f"Selected {len(results)} runs; {len(pending)} require evaluation on "
        f"seeds {args.test_seed}-{args.test_seed + args.n_test_rollouts - 1}."
    )

    for position, (_, row) in enumerate(pending, start=1):
        print(
            f"[{position}/{len(pending)}] {row['task']} | {row['model']} | "
            f"policy_hebbian={row['hebbian']} | optimizer_seed={row['seed']}"
        )
        summary, rollout_rows = reevaluate(
            row,
            args.n_test_rollouts,
            args.test_seed,
        )
        # Write rollout details first. The summary row acts as the completion
        # marker used when resuming an interrupted evaluation.
        append_frame(rollout_rows, rollout_path)
        append_frame(pd.DataFrame([summary]), summary_path)

    if summary_path.exists() and rollout_path.exists():
        sort_outputs(summary_path, rollout_path)

    print(f"\nSummary:  {summary_path}")
    print(f"Rollouts: {rollout_path}")


if __name__ == "__main__":
    main()
