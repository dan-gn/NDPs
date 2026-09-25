import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Graph.graph_nx import Graphnx
from Tasks.pendulum import Pendulum


def make_pendulum_graph():
    states = np.array(
        [
            [0.20, 0.10, 0.05, 0.05, 0.01],
            [0.25, 0.10, 0.05, 0.05, 0.01],
            [0.30, 0.10, 0.05, 0.05, 0.01],
            [0.35, 0.10, 0.05, 0.05, 0.01],
        ],
        dtype=np.float32,
    )

    graph = Graphnx(
        state_dim=5,
        weighted_graph_flag=True,
        propagation_mode="directed",
    )
    graph.add_nodes_from(states)
    graph.add_edges_from(
        [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (0, 3),
            (1, 3),
            (2, 3),
        ]
    )

    weights = np.zeros((4, 4), dtype=np.float32)
    weights[0, 0] = 0.10
    weights[1, 1] = 0.10
    weights[2, 2] = 0.10
    weights[3, 3] = 0.10
    weights[0, 3] = 0.20
    weights[1, 3] = -0.15
    weights[2, 3] = 0.25
    graph.update_weight_matrix(weights)
    return graph


def test_pendulum_spaces_match_task():
    task = Pendulum()
    env = gym.make(task.name)

    try:
        assert task.graph_n_inputs == env.observation_space.shape[0] == 3
        assert task.graph_n_outputs == env.action_space.shape[0] == 1
        assert np.array_equal(task.action_low, env.action_space.low)
        assert np.array_equal(task.action_high, env.action_space.high)
    finally:
        env.close()


def test_pendulum_action_scaling_and_dtype():
    task = Pendulum()

    centre = task.compute_action(torch.tensor([[0.0]], dtype=torch.float32))
    upper = task.compute_action(torch.tensor([[100.0]], dtype=torch.float32))
    lower = task.compute_action(torch.tensor([[-100.0]], dtype=torch.float32))

    assert centre.shape == (1,)
    assert centre.dtype == np.float32
    assert np.allclose(centre, [0.0])
    assert np.allclose(upper, task.action_high)
    assert np.allclose(lower, task.action_low)


def test_pendulum_environment_accepts_scaled_action():
    task = Pendulum()
    env = gym.make(task.name)

    try:
        env.reset(seed=0)
        action = task.compute_action(torch.tensor([[100.0]], dtype=torch.float32))
        observation, reward, terminated, truncated, _ = env.step(action)

        assert observation.shape == (3,)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_standard_rollout_is_seed_reproducible():
    task = Pendulum()
    graph = make_pendulum_graph()

    first_loss, first_rewards = task.evaluate_graph(
        graph, n_rollouts=1, env_seed=123, hebbian=False
    )
    second_loss, second_rewards = task.evaluate_graph(
        graph, n_rollouts=1, env_seed=123, hebbian=False
    )

    assert np.isfinite(first_loss)
    assert np.allclose(first_loss, second_loss)
    assert np.allclose(first_rewards, second_rewards)


def test_hebbian_rollout_runs_and_is_seed_reproducible():
    task = Pendulum()
    graph = make_pendulum_graph()

    first_loss, first_rewards = task.evaluate_graph(
        graph, n_rollouts=1, env_seed=123, hebbian=True
    )
    second_loss, second_rewards = task.evaluate_graph(
        graph, n_rollouts=1, env_seed=123, hebbian=True
    )

    assert np.isfinite(first_loss)
    assert np.allclose(first_loss, second_loss)
    assert np.allclose(first_rewards, second_rewards)


def main():
    tests = [
        test_pendulum_spaces_match_task,
        test_pendulum_action_scaling_and_dtype,
        test_pendulum_environment_accepts_scaled_action,
        test_standard_rollout_is_seed_reproducible,
        test_hebbian_rollout_runs_and_is_seed_reproducible,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All Pendulum task tests passed.")


if __name__ == "__main__":
    main()
