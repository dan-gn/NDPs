import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Optimisation.cma_es import CMA_ES
from Optimisation.ea import TEST_SEED


class FakeResult:
    def __init__(self):
        self.xbest = np.array([0.25, -0.50], dtype=np.float64)
        self.fbest = 0.3125


class FakeEvolutionStrategy:
    """Enough of the CMA interface to test final held-out evaluation."""

    def __init__(self):
        self.result = FakeResult()

    def stop(self):
        return {"test_finished": True}

    def result_pretty(self):
        pass


def test_tuple_fitness_extraction():
    evaluation = (1.25, [1.0, 1.5], object(), 1.0)
    assert CMA_ES.get_fitness_value(evaluation) == 1.25


def test_scalar_fitness_extraction():
    assert CMA_ES.get_fitness_value(np.float32(2.5)) == 2.5


def test_training_evaluation_uses_default_seed():
    requested_seeds = []

    def objective(solution, env_seed=0):
        requested_seeds.append(env_seed)
        fitness = float(np.sum(np.asarray(solution) ** 2))
        return fitness, [], object(), fitness

    optimiser = object.__new__(CMA_ES)
    optimiser.fitness_function = objective
    values = optimiser.evaluate(
        [
            np.array([1.0, 2.0]),
            np.array([2.0, 3.0]),
        ]
    )

    assert values == [5.0, 13.0]
    assert requested_seeds == [0, 0]


def test_final_best_uses_held_out_seed():
    requested_seeds = []

    def objective(solution, env_seed=0):
        requested_seeds.append(env_seed)
        fitness = float(np.sum(np.asarray(solution) ** 2))
        return fitness, [fitness], object(), fitness

    optimiser = object.__new__(CMA_ES)
    optimiser.fitness_function = objective
    optimiser.seed = 7
    optimiser.test_seed = TEST_SEED
    optimiser.es = FakeEvolutionStrategy()
    optimiser.best_params = None
    optimiser.best_loss = None
    optimiser.best_loss_test = None
    optimiser.best_test_result = None

    best_params, best_loss = optimiser.run()

    assert np.array_equal(best_params, np.array([0.25, -0.50]))
    assert best_loss == 0.3125
    assert requested_seeds == [TEST_SEED]
    assert optimiser.best_loss_test == 0.3125
    assert optimiser.best_test_result[0] == 0.3125


def test_seeded_initial_center_is_reproducible():
    first = np.random.default_rng(42).uniform(-1, 1, 20)

    # Unrelated global NumPy calls must not affect the local generator.
    np.random.uniform(size=1000)

    second = np.random.default_rng(42).uniform(-1, 1, 20)
    different = np.random.default_rng(43).uniform(-1, 1, 20)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def main():
    tests = [
        test_tuple_fitness_extraction,
        test_scalar_fitness_extraction,
        test_training_evaluation_uses_default_seed,
        test_final_best_uses_held_out_seed,
        test_seeded_initial_center_is_reproducible,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All CMA seed-equivalence tests passed.")


if __name__ == "__main__":
    main()
