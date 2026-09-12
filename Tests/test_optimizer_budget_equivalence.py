import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import Experiments.experiments_2 as experiments
from Optimisation.cma_es import CMA_ES
from Optimisation.ea import TEST_SEED


class FakeTask:
    def __init__(self):
        self.name = "FakeTask-v0"
        self.target = -123.0
        self.parameters = {
            "model": "standard_ndp",
            "initial_node_state_mode": "coevolve",
            "state_dim": 5,
            "n_nodes": 16,
            "hebbian": False,
            "population_size": 8,
            "generations": 12,
            "stagnant_generation": 4,
        }

    def evaluate_ndp(self, vector, env_seed=0):
        return float(np.sum(np.asarray(vector) ** 2))

    def summary(self):
        pass


class FakeNDP:
    def __init__(self, parameters):
        self.parameters = parameters

    def get_total_number_of_mlp_parameters(self):
        return 10

    def summary(self):
        pass


class FakeCMA:
    instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.run_arguments = None
        FakeCMA.instance = self

    def run(self, *args):
        self.run_arguments = args
        return np.zeros(15), 1.5


class FakeEA:
    instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.run_arguments = None
        FakeEA.instance = self

    def run(self, *args):
        self.run_arguments = args
        return np.zeros(15), 2.5


def test_cma_options_are_applied():
    optimiser = CMA_ES(
        fitness_function=lambda x: float(np.sum(np.asarray(x) ** 2)),
        x0=np.zeros(4),
        sigma0=0.1,
        seed=7,
        population_size=8,
        max_iterations=12,
    )

    assert optimiser.es.popsize == 8
    assert optimiser.es.opts["maxiter"] == 12
    assert optimiser.population_size == 8
    assert optimiser.max_iterations == 12


def test_experiment_passes_budget_and_calls_cma_without_arguments():
    original_ndp = experiments.NeuralDevelopmentalProgram
    original_cma = experiments.CMA_ES

    try:
        experiments.NeuralDevelopmentalProgram = FakeNDP
        experiments.CMA_ES = FakeCMA

        task = FakeTask()
        output = experiments.experiment(task, optimisation_algorithm="CMA", seed=7)

        optimiser = FakeCMA.instance
        assert optimiser.kwargs["population_size"] == 8
        assert optimiser.kwargs["max_iterations"] == 13
        assert optimiser.kwargs["test_seed"] == TEST_SEED
        assert optimiser.run_arguments == ()
        assert output["optimiser"] is optimiser
    finally:
        experiments.NeuralDevelopmentalProgram = original_ndp
        experiments.CMA_ES = original_cma


def test_experiment_passes_budget_and_calls_ea_with_target_and_seed():
    original_ndp = experiments.NeuralDevelopmentalProgram
    original_ea = experiments.EvolutionaryAlgorithm

    try:
        experiments.NeuralDevelopmentalProgram = FakeNDP
        experiments.EvolutionaryAlgorithm = FakeEA

        task = FakeTask()
        output = experiments.experiment(task, optimisation_algorithm="EA", seed=7)

        optimiser = FakeEA.instance
        assert optimiser.kwargs["population_size"] == 8
        assert optimiser.kwargs["max_iterations"] == 12
        assert optimiser.run_arguments == (task.target, 7)
        assert output["optimiser"] is optimiser
    finally:
        experiments.NeuralDevelopmentalProgram = original_ndp
        experiments.EvolutionaryAlgorithm = original_ea


def main():
    tests = [
        test_cma_options_are_applied,
        test_experiment_passes_budget_and_calls_cma_without_arguments,
        test_experiment_passes_budget_and_calls_ea_with_target_and_seed,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All optimizer budget-equivalence tests passed.")


if __name__ == "__main__":
    main()
