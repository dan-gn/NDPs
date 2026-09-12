import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Optimisation.ea import EvolutionaryAlgorithm


class RecordingExecutor:
    instances = []

    def __init__(self, max_workers, mp_context=None):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.map_calls = 0
        self.shutdown_calls = 0
        self.__class__.instances.append(self)

    def map(self, function, iterable):
        self.map_calls += 1
        return map(function, iterable)

    def shutdown(self):
        self.shutdown_calls += 1


def objective(genotype, env_seed=None):
    fitness = float(np.sum(np.square(genotype)))
    return fitness, None, None, fitness


def make_optimizer(run_in_parallel):
    return EvolutionaryAlgorithm(
        n_variables=4,
        objective_function=objective,
        population_size=4,
        max_iterations=2,
        max_stagnment=100,
        run_in_parallel=run_in_parallel,
        cores=3,
    )


def test_parallel_run_reuses_one_pool():
    RecordingExecutor.instances.clear()

    with patch("Optimisation.ea.ProcessPoolExecutor", RecordingExecutor):
        optimizer = make_optimizer(run_in_parallel=True)
        optimizer.run(stop_criteria=-np.inf, seed=7)

    assert len(RecordingExecutor.instances) == 1
    executor = RecordingExecutor.instances[0]
    assert executor.max_workers == 3
    assert executor.mp_context.get_start_method() == "spawn"
    assert executor.map_calls == 3  # initial population + two generations
    assert executor.shutdown_calls == 1


def test_sequential_run_does_not_create_pool():
    RecordingExecutor.instances.clear()

    with patch("Optimisation.ea.ProcessPoolExecutor", RecordingExecutor):
        optimizer = make_optimizer(run_in_parallel=False)
        optimizer.run(stop_criteria=-np.inf, seed=7)

    assert RecordingExecutor.instances == []


def test_pool_closes_when_evaluation_fails():
    RecordingExecutor.instances.clear()

    def failing_objective(genotype, env_seed=None):
        raise RuntimeError("deliberate evaluation failure")

    optimizer = make_optimizer(run_in_parallel=True)
    optimizer.objective_function = failing_objective

    try:
        with patch("Optimisation.ea.ProcessPoolExecutor", RecordingExecutor):
            optimizer.run(stop_criteria=-np.inf, seed=7)
    except RuntimeError as error:
        assert str(error) == "deliberate evaluation failure"
    else:
        raise AssertionError("The deliberate evaluation failure was not raised")

    assert len(RecordingExecutor.instances) == 1
    assert RecordingExecutor.instances[0].shutdown_calls == 1


def main():
    tests = [
        test_parallel_run_reuses_one_pool,
        test_sequential_run_does_not_create_pool,
        test_pool_closes_when_evaluation_fails,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All persistent EA pool tests passed.")


if __name__ == "__main__":
    main()
