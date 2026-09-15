import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Optimisation.cma_es import CMA_ES
from Optimisation.ea import EvolutionaryAlgorithm, TEST_SEED


class FakeGraph:
    def number_of_nodes(self):
        return 3

    def number_of_edges(self):
        return 3


def test_ea_counts_search_and_heldout_calls_separately():
    observed = {"optimisation": 0, "heldout": 0}

    def objective(genotype, env_seed=0):
        if env_seed == TEST_SEED:
            observed["heldout"] += 1
        else:
            observed["optimisation"] += 1

        fitness = float(np.sum(np.asarray(genotype) ** 2))
        return fitness, [fitness], FakeGraph(), fitness

    population_size = 4
    generations = 2
    optimiser = EvolutionaryAlgorithm(
        n_variables=3,
        objective_function=objective,
        population_size=population_size,
        max_iterations=generations,
        max_stagnment=100,
        run_in_parallel=False,
    )
    optimiser.run(stop_criteria=-np.inf, seed=7)

    expected_optimisation_calls = population_size * (generations + 1)
    assert observed["optimisation"] == expected_optimisation_calls
    assert optimiser.optimisation_evaluations == expected_optimisation_calls
    assert optimiser.training_reevaluations == 0
    assert observed["heldout"] == 1
    assert optimiser.heldout_evaluations == 1


class FakeCMALogger:
    def add(self):
        pass


class FakeCMAState:
    def __init__(self, population_size=4, iterations=3):
        self.popsize = population_size
        self.iterations = iterations
        self.countiter = 0
        self.logger = FakeCMALogger()
        self.result = SimpleNamespace(
            xbest=np.zeros(3, dtype=np.float64),
            fbest=0.0,
        )

    def stop(self):
        return {"maxiter": self.iterations} if self.countiter >= self.iterations else {}

    def ask(self):
        return [np.full(3, value, dtype=np.float64) for value in range(self.popsize)]

    def tell(self, solutions, fitness_values):
        assert len(solutions) == self.popsize
        assert len(fitness_values) == self.popsize
        self.countiter += 1

    def disp(self):
        pass

    def result_pretty(self):
        pass


def test_cma_counts_search_and_final_calls_separately():
    observed = {"optimisation": 0, "training": 0, "heldout": 0}

    def objective(solution, env_seed=None):
        if env_seed is None:
            observed["optimisation"] += 1
        elif env_seed == 0:
            observed["training"] += 1
        elif env_seed == TEST_SEED:
            observed["heldout"] += 1
        else:
            raise AssertionError(f"Unexpected environment seed: {env_seed}")

        fitness = float(np.sum(np.asarray(solution) ** 2))
        return fitness, [fitness], FakeGraph(), fitness

    population_size = 4
    iterations = 3

    optimiser = object.__new__(CMA_ES)
    optimiser.fitness_function = objective
    optimiser.seed = 7
    optimiser.test_seed = TEST_SEED
    optimiser.es = FakeCMAState(population_size, iterations)

    optimiser.run()

    expected_optimisation_calls = population_size * iterations
    assert observed == {
        "optimisation": expected_optimisation_calls,
        "training": 1,
        "heldout": 1,
    }
    assert optimiser.optimisation_evaluations == expected_optimisation_calls
    assert optimiser.training_reevaluations == 1
    assert optimiser.heldout_evaluations == 1


def main():
    tests = [
        test_ea_counts_search_and_heldout_calls_separately,
        test_cma_counts_search_and_final_calls_separately,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All evaluation-counter tests passed.")


if __name__ == "__main__":
    main()
