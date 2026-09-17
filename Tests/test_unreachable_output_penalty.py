from pathlib import Path
import sys
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Tasks.task import Task


class FakeGraph:
    def __init__(self, unreachable_outputs):
        self.unreachable_outputs = list(unreachable_outputs)
        self.calls = []

    def get_unreachable_outputs(self, n_inputs, n_outputs):
        self.calls.append((n_inputs, n_outputs))
        return self.unreachable_outputs


def make_parameters(**overrides):
    parameters = {
        "graph_n_inputs": 8,
        "graph_n_outputs": 4,
        "network_extra_thinking": 5,
        "n_cycles": 5,
        "n_repeats": 1,
        "n_rollouts": 3,
        "initial_node_state_mode": "coevolve",
        "state_dim": 5,
        "invalid_graph_fitness": 1_000.0,
    }
    parameters.update(overrides)
    return parameters


def test_unreachable_graph_skips_standard_policy_and_rollouts():
    task = Task(make_parameters())
    graph = FakeGraph([61])

    with (
        patch(
            "Tasks.task.PolicyNetwork",
            side_effect=AssertionError("Policy construction must be skipped"),
        ),
        patch.object(
            task,
            "evaluate_policy",
            side_effect=AssertionError("Environment rollouts must be skipped"),
        ),
    ):
        fitness, rollouts = task.evaluate_graph(graph, hebbian=False)

    assert fitness == 1_000.0
    assert rollouts == [1_000.0, 1_000.0, 1_000.0]
    assert graph.calls == [(8, 4)]


def test_unreachable_graph_skips_hebbian_policy_and_requested_rollouts():
    task = Task(make_parameters(invalid_graph_fitness=2_000.0))
    graph = FakeGraph([60, 63])

    with (
        patch(
            "Tasks.task.NcHebbianLearningPolicyNetwork",
            side_effect=AssertionError("Policy construction must be skipped"),
        ),
        patch.object(
            task,
            "evaluate_policy",
            side_effect=AssertionError("Environment rollouts must be skipped"),
        ),
    ):
        fitness, rollouts = task.evaluate_graph(
            graph,
            n_rollouts=2,
            hebbian=True,
        )

    assert fitness == 2_000.0
    assert rollouts == [2_000.0, 2_000.0]
    assert graph.calls == [(8, 4)]


def test_reachable_graph_follows_normal_evaluation_path():
    task = Task(make_parameters())
    graph = FakeGraph([])
    fake_policy = object()

    with (
        patch("Tasks.task.PolicyNetwork", return_value=fake_policy) as policy_class,
        patch.object(
            task,
            "evaluate_policy",
            return_value=(-123.0, [-120.0, -126.0]),
        ) as evaluate_policy,
    ):
        result = task.evaluate_graph(
            graph,
            n_rollouts=2,
            env_seed=7,
            hebbian=False,
        )

    assert result == (-123.0, [-120.0, -126.0])
    policy_class.assert_called_once_with(graph, 8, 4, 5)
    evaluate_policy.assert_called_once_with(
        fake_policy,
        False,
        2,
        7,
        False,
        False,
    )


def test_missing_configuration_uses_defensive_fallback():
    parameters = make_parameters()
    del parameters["invalid_graph_fitness"]
    task = Task(parameters)

    assert task.invalid_graph_fitness == 1_000_000.0


def main():
    tests = [
        test_unreachable_graph_skips_standard_policy_and_rollouts,
        test_unreachable_graph_skips_hebbian_policy_and_requested_rollouts,
        test_reachable_graph_follows_normal_evaluation_path,
        test_missing_configuration_uses_defensive_fallback,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All unreachable-output penalty tests passed.")


if __name__ == "__main__":
    main()
