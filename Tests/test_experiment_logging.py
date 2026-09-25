import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Utilities.utilities import create_experiment_log
from Optimisation.ea import EvolutionaryAlgorithm, Individual, TEST_SEED


class FakeGraph:
    def number_of_nodes(self):
        return 7

    def number_of_edges(self):
        return 11

    def get_number_of_used_nodes(self, n_inputs, n_outputs):
        assert (n_inputs, n_outputs) == (2, 1)
        return 5

    def get_number_of_used_edges(self, n_inputs, n_outputs):
        assert (n_inputs, n_outputs) == (2, 1)
        return 9

    def are_all_outputs_reachable(self, n_inputs, n_outputs):
        assert (n_inputs, n_outputs) == (2, 1)
        return True

    def get_unreachable_outputs(self, n_inputs, n_outputs):
        assert (n_inputs, n_outputs) == (2, 1)
        return []


class FakeCMAState:
    popsize = 8
    countiter = 12
    N = 4
    opts = {"maxiter": 30}

    def stop(self):
        return {"maxiter": 30}


def fake_task():
    return SimpleNamespace(
        name="TestEnvironment-v0",
        parameters={
            "initial_node_state_mode": "coevolve",
            "model": "rewiring_ndp",
            "hebbian": True,
            "graph_n_inputs": 2,
            "graph_n_outputs": 1,
            "population_size": 8,
            "generations": 30,
        },
    )


def make_cma():
    graph = FakeGraph()
    return SimpleNamespace(
        es=FakeCMAState(),
        best_params=np.zeros(4),
        best_loss=-50.0,
        best_loss_test=-45.0,
        best_training_result=(-50.0, [-25.0, -25.0], graph, -25.0),
        best_test_result=(-45.0, [-20.0, -25.0], graph, -20.0),
        optimisation_evaluations=248,
        training_reevaluations=1,
        heldout_evaluations=1,
    )


def make_ea():
    graph = FakeGraph()
    return SimpleNamespace(
        population_size=8,
        max_iterations=30,
        i=12,
        n_variables=4,
        max_stagnment=10,
        stop_on_target=False,
        goal_achieved=False,
        best_individual=SimpleNamespace(
            fitness=-50.0,
            fitness_test=-45.0,
            best_graph=graph,
            best_graph_fitness=-25.0,
            best_graph_fitness_test=-20.0,
            best_graph_used_nodes=5,
            best_graph_used_edges=9,
        ),
        best_individual_by_graph=SimpleNamespace(
            best_graph=graph,
            best_graph_fitness=-25.0,
            fitness_test=-20.0,
            best_graph_fitness_test=-18.0,
        ),
        optimisation_evaluations=248,
        training_reevaluations=0,
        heldout_evaluations=1,
    )


def create_log(algorithm, optimiser):
    return create_experiment_log(
        output_filename="output.pkl",
        optimisation_algorithm=algorithm,
        task=fake_task(),
        seed=3,
        optimiser=optimiser,
        elapsed_time=4.5,
    )


def test_ea_and_cma_have_matching_columns():
    ea_log = create_log("EA", make_ea())
    cma_log = create_log("CMA", make_cma())
    assert set(ea_log) == set(cma_log)


def test_cma_values():
    log = create_log("CMA", make_cma())
    assert log["best_score_mean"] == -50.0
    assert log["best_score_test"] == -45.0
    assert log["best_graph"] == -25.0
    assert log["best_graph_test"] == -20.0
    assert log["best_graph_n_nodes"] == 7
    assert log["best_graph_n_edges"] == 11
    assert log["best_graph_used_nodes"] == 5
    assert log["best_graph_used_edges"] == 9
    assert log["best_graph_are_all_outputs_reachable"] is True
    assert log["best_graph_n_unreachable_outputs"] == 0
    assert log["best_graph_unreachable_output_ids"] == "[]"
    assert log["population_size"] == 8
    assert log["optimiser_iterations"] == 12
    assert log["n_variables"] == 4
    assert log["stop_on_target"] is False
    assert log["stop_reason"] == "maxiter"
    assert log["configured_generations"] == 30
    assert log["configured_optimisation_evaluations"] == 248
    assert log["optimisation_evaluations"] == 248
    assert log["training_reevaluations"] == 1
    assert log["heldout_evaluations"] == 1
    assert log["total_objective_calls"] == 250


def test_ea_values():
    log = create_log("EA", make_ea())
    assert log["best_score_mean"] == -50.0
    assert log["best_score_test"] == -45.0
    assert log["best_graph"] == -25.0
    assert log["best_graph_test"] == -20.0
    assert log["best_graph_n_nodes"] == 7
    assert log["best_graph_n_edges"] == 11
    assert log["best_graph_used_nodes"] == 5
    assert log["best_graph_used_edges"] == 9
    assert log["best_graph_are_all_outputs_reachable"] is True
    assert log["best_graph_n_unreachable_outputs"] == 0
    assert log["best_graph_unreachable_output_ids"] == "[]"
    assert log["stop_on_target"] is False
    assert log["stop_reason"] == "completed_budget"
    assert log["configured_generations"] == 30
    assert log["configured_optimisation_evaluations"] == 248
    assert log["optimisation_evaluations"] == 248
    assert log["training_reevaluations"] == 0
    assert log["heldout_evaluations"] == 1
    assert log["total_objective_calls"] == 249
    assert log["optimiser_iterations"] == 13


def test_scalar_cma_objective_is_safe():
    optimiser = make_cma()
    optimiser.best_training_result = -50.0
    optimiser.best_test_result = -45.0

    log = create_log("CMA", optimiser)
    assert log["best_graph"] is None
    assert log["best_graph_test"] is None
    assert log["best_graph_n_nodes"] is None
    assert log["best_graph_n_edges"] is None
    assert log["best_graph_used_nodes"] is None
    assert log["best_graph_used_edges"] is None
    assert log["best_graph_are_all_outputs_reachable"] is None
    assert log["best_graph_n_unreachable_outputs"] is None
    assert log["best_graph_unreachable_output_ids"] is None


def test_best_observed_graph_update_does_not_use_heldout_seed():
    held_out_genotypes = []

    def objective(genotype, env_seed=0):
        if env_seed == TEST_SEED:
            held_out_genotypes.append(np.asarray(genotype).copy())
        value = float(np.asarray(genotype)[0])
        return value, [value], FakeGraph(), value

    optimiser = EvolutionaryAlgorithm(
        n_variables=1,
        objective_function=objective,
        population_size=2,
        max_iterations=1,
        max_stagnment=1,
        run_in_parallel=False,
    )
    optimiser.population = [
        Individual(1, genotype=np.array([50.0]), fitness=-100.0),
        Individual(1, genotype=np.array([60.0]), fitness=-90.0),
    ]
    optimiser.best_individual.genotype = np.array([99.0])
    optimiser.best_individual.fitness = -1000.0
    optimiser.stagnment_iterations = 0

    offspring = [
        Individual(
            1,
            genotype=np.array([7.0]),
            fitness=7.0,
            best_graph=FakeGraph(),
            best_graph_fitness=7.0,
        ),
        Individual(
            1,
            genotype=np.array([8.0]),
            fitness=8.0,
            best_graph=FakeGraph(),
            best_graph_fitness=8.0,
        ),
    ]

    optimiser.elitism(offspring)

    assert held_out_genotypes == []
    assert np.array_equal(
        optimiser.best_individual_by_graph.genotype,
        np.array([7.0]),
    )


def main():
    tests = [
        test_ea_and_cma_have_matching_columns,
        test_cma_values,
        test_ea_values,
        test_scalar_cma_objective_is_safe,
        test_best_observed_graph_update_does_not_use_heldout_seed,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All experiment logging tests passed.")


if __name__ == "__main__":
    main()
