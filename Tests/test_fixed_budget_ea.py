import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Optimisation.ea import EvolutionaryAlgorithm


def objective(genotype, env_seed=0):
    fitness = 0.0
    return fitness, None, None, fitness


def make_optimizer(stop_on_target):
    return EvolutionaryAlgorithm(
        n_variables=4,
        objective_function=objective,
        population_size=4,
        max_iterations=3,
        max_stagnment=100,
        run_in_parallel=False,
        stop_on_target=stop_on_target,
    )


def test_default_early_stopping_is_preserved():
    optimizer = make_optimizer(stop_on_target=True)
    optimizer.run(stop_criteria=1.0, seed=7)

    assert optimizer.goal_achieved is True
    assert optimizer.goal_achieved_it == 0
    assert optimizer.i == 0
    assert optimizer.optimisation_evaluations == 8


def test_fixed_budget_continues_after_target():
    optimizer = make_optimizer(stop_on_target=False)
    optimizer.run(stop_criteria=1.0, seed=7)

    assert optimizer.goal_achieved is True
    assert optimizer.goal_achieved_it == 0
    assert optimizer.i == 2
    assert optimizer.optimisation_evaluations == 16
    assert optimizer.optimisation_evaluations == (
        optimizer.population_size * (optimizer.max_iterations + 1)
    )


def main():
    tests = (
        test_default_early_stopping_is_preserved,
        test_fixed_budget_continues_after_target,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("All fixed-budget EA tests passed.")


if __name__ == "__main__":
    main()
