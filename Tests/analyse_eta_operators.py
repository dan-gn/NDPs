"""Empirically compare polynomial-mutation and SBX eta values.

This is an operator-level analysis: it does not run an environment or optimise a
policy.  Every eta is evaluated with the same underlying random draws so that
the comparisons are paired and reproducible.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ETA_VALUES = (1, 5, 10, 15, 20, 25, 30)
PERCENTILES = (25, 50, 75, 90, 95, 99, 99.9)


def polynomial_mutation_delta(random_values: np.ndarray, eta: float) -> np.ndarray:
    """Exact delta used by EvolutionaryAlgorithm.polynomial_muatation."""
    delta = np.empty_like(random_values)
    lower = random_values < 0.5
    delta[lower] = (2.0 * random_values[lower]) ** (1.0 / (eta + 1.0)) - 1.0
    delta[~lower] = 1.0 - (
        2.0 * (1.0 - random_values[~lower])
    ) ** (1.0 / (eta + 1.0))
    return delta


def sbx_beta(random_values: np.ndarray, eta: float) -> np.ndarray:
    """Exact beta used by EvolutionaryAlgorithm.sbx."""
    beta = np.empty_like(random_values)
    lower = random_values <= 0.5
    beta[lower] = (2.0 * random_values[lower]) ** (1.0 / (eta + 1.0))
    beta[~lower] = (
        1.0 / (2.0 * (1.0 - random_values[~lower]))
    ) ** (1.0 / (eta + 1.0))
    return beta


def summarise(operator: str, eta: float, values: np.ndarray) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "operator": operator,
        "eta": eta,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }
    for percentile, value in zip(PERCENTILES, np.percentile(values, PERCENTILES)):
        row[f"p{percentile:g}"] = float(value)
    return row


def save_summary(rows: list[dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    mutation_samples: dict[int, np.ndarray],
    sbx_samples: dict[int, np.ndarray],
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot.")
        return

    quantiles = np.linspace(0.0, 0.999, 500)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for eta in ETA_VALUES:
        axes[0].plot(
            quantiles,
            np.quantile(mutation_samples[eta], quantiles),
            label=f"eta={eta}",
        )
        axes[1].plot(
            quantiles,
            np.quantile(sbx_samples[eta], quantiles),
            label=f"eta={eta}",
        )

    axes[0].set_title("Polynomial mutation")
    axes[0].set_xlabel("Quantile")
    axes[0].set_ylabel("Absolute parameter change")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Simulated binary crossover")
    axes[1].set_xlabel("Quantile")
    axes[1].set_ylabel("Nearest-parent distance / parent separation")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/eta_operator_analysis"),
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    rng = np.random.default_rng(args.seed)
    # Reuse the draws for every eta to make comparisons directly paired.
    mutation_random = rng.random(args.samples)
    sbx_random = rng.random(args.samples)

    rows: list[dict[str, float | str]] = []
    mutation_samples: dict[int, np.ndarray] = {}
    sbx_samples: dict[int, np.ndarray] = {}

    print(f"Samples per eta: {args.samples:,}")
    print(f"Seed: {args.seed}")
    print()
    print(" eta | mutation median | mutation p95 | SBX median | SBX p95 | SBX p99")
    print("-----+-----------------+--------------+------------+---------+--------")

    for eta in ETA_VALUES:
        mutation_change = np.abs(polynomial_mutation_delta(mutation_random, eta))

        beta = sbx_beta(sbx_random, eta)
        # For either SBX child, this is its distance to the nearest parent,
        # divided by the distance between the two parents.
        sbx_nearest_parent_distance = np.abs(beta - 1.0) / 2.0

        mutation_samples[eta] = mutation_change
        sbx_samples[eta] = sbx_nearest_parent_distance
        rows.append(summarise("mutation_absolute_change", eta, mutation_change))
        rows.append(
            summarise(
                "sbx_normalized_nearest_parent_distance",
                eta,
                sbx_nearest_parent_distance,
            )
        )

        mutation_q = np.percentile(mutation_change, [50, 95])
        sbx_q = np.percentile(sbx_nearest_parent_distance, [50, 95, 99])
        print(
            f"{eta:>4} | {mutation_q[0]:>15.6f} | {mutation_q[1]:>12.6f} | "
            f"{sbx_q[0]:>10.6f} | {sbx_q[1]:>7.6f} | {sbx_q[2]:>7.6f}"
        )

    summary_path = args.output_dir / "eta_operator_summary.csv"
    plot_path = args.output_dir / "eta_operator_distributions.png"
    save_summary(rows, summary_path)
    save_plot(mutation_samples, sbx_samples, plot_path)

    print()
    print(f"Summary saved to: {summary_path}")
    if plot_path.exists():
        print(f"Plot saved to:    {plot_path}")


if __name__ == "__main__":
    main()
