import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Optimisation.ea import EvolutionaryAlgorithm


def make_optimiser():
    return EvolutionaryAlgorithm(
        n_variables=4,
        objective_function=lambda genotype: None,
        population_size=4,
        max_iterations=200,
        max_stagnment=20,
        mutation_eta_min=5,
        mutation_eta_max=15,
        sbx_eta_min=5,
        sbx_eta_max=15,
        eta_schedule_iterations=100,
        run_in_parallel=False,
    )


def test_eta_schedule_starts_at_minimum():
    optimiser = make_optimiser()
    optimiser.init_optimisation_variables()

    assert optimiser.mutation_eta == 5
    assert optimiser.sbx_eta == 5


def test_eta_schedule_interpolates_linearly():
    optimiser = make_optimiser()
    optimiser.iterations_since_restart = 50
    optimiser.update_eta_schedule()

    assert np.isclose(optimiser.mutation_eta, 10)
    assert np.isclose(optimiser.sbx_eta, 10)


def test_eta_schedule_is_capped_at_maximum():
    optimiser = make_optimiser()
    optimiser.iterations_since_restart = 150
    optimiser.update_eta_schedule()

    assert optimiser.mutation_eta == 15
    assert optimiser.sbx_eta == 15


def test_eta_schedule_resets_to_minimum():
    optimiser = make_optimiser()
    optimiser.iterations_since_restart = 100
    optimiser.update_eta_schedule()
    optimiser.iterations_since_restart = 0
    optimiser.update_eta_schedule()

    assert optimiser.mutation_eta == 5
    assert optimiser.sbx_eta == 5


def main():
    tests = [
        test_eta_schedule_starts_at_minimum,
        test_eta_schedule_interpolates_linearly,
        test_eta_schedule_is_capped_at_maximum,
        test_eta_schedule_resets_to_minimum,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All adaptive eta schedule tests passed.")


if __name__ == "__main__":
    main()
