import sys
import warnings
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Optimisation.ea import EvolutionaryAlgorithm, Individual


def make_optimiser(fitness_values):
    optimiser = EvolutionaryAlgorithm(
        n_variables=1,
        objective_function=lambda genotype: None,
        population_size=len(fitness_values),
        max_iterations=1,
        max_stagnment=1,
        run_in_parallel=False,
    )
    optimiser.population = [
        Individual(1, genotype=np.zeros(1), fitness=fitness)
        for fitness in fitness_values
    ]
    return optimiser


def test_extreme_fitness_values_do_not_overflow():
    optimiser = make_optimiser([-1e308, 0.0, 1e308])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        weights = optimiser.compute_parent_selection_prob()

    assert np.all(np.isfinite(weights))
    assert weights[0] > weights[1] > weights[2]
    assert np.max(weights) == 1.0


def test_mixed_sign_values_preserve_minimisation_order():
    optimiser = make_optimiser([-100.0, -1.0, 1.0, 100.0])
    weights = optimiser.compute_parent_selection_prob()

    assert np.all(np.diff(weights) < 0)


def test_non_finite_fitness_values_raise_clear_error():
    invalid_populations = [
        [0.0, -np.inf, 10.0],
        [0.0, np.inf, 10.0],
        [0.0, np.nan, 10.0],
    ]

    for fitness_values in invalid_populations:
        optimiser = make_optimiser(fitness_values)

        try:
            optimiser.compute_parent_selection_prob()
        except ValueError as error:
            assert "non-finite fitness values" in str(error)
        else:
            raise AssertionError(
                f"Expected ValueError for fitness values {fitness_values}."
            )


def test_zero_weight_tournament_falls_back_to_valid_index():
    optimiser = make_optimiser([0.0, 1.0])
    np.random.seed(7)

    draws = [optimiser.roulette_wheel(np.zeros(2)) for _ in range(20)]

    assert all(draw in (0, 1) for draw in draws)


def main():
    tests = [
        test_extreme_fitness_values_do_not_overflow,
        test_mixed_sign_values_preserve_minimisation_order,
        test_non_finite_fitness_values_raise_clear_error,
        test_zero_weight_tournament_falls_back_to_valid_index,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All parent-selection probability tests passed.")


if __name__ == "__main__":
    main()
