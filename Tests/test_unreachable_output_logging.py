import sys
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if not (REPOSITORY_ROOT / "Utilities").exists():
    REPOSITORY_ROOT = REPOSITORY_ROOT / "NDPs"
sys.path.insert(0, str(REPOSITORY_ROOT))

from Utilities.utilities import create_experiment_log


class FakeGraph:
    def __init__(self, unreachable_outputs):
        self.unreachable_outputs = list(unreachable_outputs)

    def number_of_nodes(self):
        return 7

    def number_of_edges(self):
        return 11

    def get_number_of_used_nodes(self, n_inputs, n_outputs):
        assert (n_inputs, n_outputs) == (2, 3)
        return 5

    def get_number_of_used_edges(self, n_inputs, n_outputs):
        assert (n_inputs, n_outputs) == (2, 3)
        return 9

    def get_unreachable_outputs(self, n_inputs, n_outputs):
        assert (n_inputs, n_outputs) == (2, 3)
        return self.unreachable_outputs.copy()


def make_task():
    return SimpleNamespace(
        name="TestEnvironment-v0",
        parameters={
            "initial_node_state_mode": "coevolve",
            "model": "rewiring_ndp",
            "hebbian": False,
            "graph_n_inputs": 2,
            "graph_n_outputs": 3,
            "population_size": 8,
            "generations": 30,
        },
    )


def make_ea(graph):
    return SimpleNamespace(
        population_size=8,
        i=12,
        n_variables=4,
        max_stagnment=10,
        stop_on_target=False,
        goal_achieved=False,
        optimisation_evaluations=104,
        training_reevaluations=0,
        heldout_evaluations=2,
        best_individual=SimpleNamespace(
            fitness=-50.0,
            fitness_test=-45.0,
            best_graph=graph,
            best_graph_fitness=-25.0,
            best_graph_fitness_test=-18.0,
            best_graph_used_nodes=5,
            best_graph_used_edges=9,
        ),
        best_individual_by_graph=SimpleNamespace(
            best_graph=graph,
            best_graph_fitness=-25.0,
            best_graph_fitness_test=-18.0,
        ),
    )


def create_log(unreachable_outputs):
    return create_experiment_log(
        output_filename="output.pkl",
        optimisation_algorithm="EA",
        task=make_task(),
        seed=3,
        optimiser=make_ea(FakeGraph(unreachable_outputs)),
        elapsed_time=4.5,
    )


def test_unreachable_output_ids_are_logged():
    log = create_log([4, 6])
    assert log["best_graph_are_all_outputs_reachable"] is False
    assert log["best_graph_n_unreachable_outputs"] == 2
    assert log["best_graph_unreachable_output_ids"] == "[4, 6]"


def test_reachable_graph_logs_an_empty_list():
    log = create_log([])
    assert log["best_graph_are_all_outputs_reachable"] is True
    assert log["best_graph_n_unreachable_outputs"] == 0
    assert log["best_graph_unreachable_output_ids"] == "[]"


def main():
    tests = [
        test_unreachable_output_ids_are_logged,
        test_reachable_graph_logs_an_empty_list,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("All unreachable-output logging tests passed.")


if __name__ == "__main__":
    main()
