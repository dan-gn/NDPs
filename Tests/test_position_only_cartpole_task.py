import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import popgym
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Graph.graph_nx import Graphnx
from Tasks.position_only_cartpole import PositionOnlyCartPole


def make_policy_graph():
    states = np.array(
        [
            [0.20, 0.10, 0.05, 0.05, 0.01],
            [0.25, 0.10, 0.05, 0.05, 0.01],
            [0.30, 0.10, 0.05, 0.05, 0.01],
        ],
        dtype=np.float32,
    )

    graph = Graphnx(
        state_dim=5,
        weighted_graph_flag=True,
        propagation_mode="directed",
    )
    graph.add_nodes_from(states)
    graph.add_edges_from([(0, 0), (1, 1), (2, 2), (0, 2), (1, 2)])

    weights = np.zeros((3, 3), dtype=np.float32)
    weights[0, 0] = 0.10
    weights[1, 1] = 0.10
    weights[2, 2] = 0.10
    weights[0, 2] = 0.20
    weights[1, 2] = -0.15
    graph.update_weight_matrix(weights)
    return graph


def test_environment_registration_and_spaces():
    task = PositionOnlyCartPole()
    assert task.name in gym.registry

    env = gym.make(task.name)
    try:
        assert task.graph_n_inputs == env.observation_space.shape[0] == 2
        assert task.graph_n_outputs == 1
        assert env.action_space.n == 2
        assert task.target == -task.n_rollouts
    finally:
        env.close()


def test_observation_removes_velocity_components():
    partial_env = popgym.envs.PositionOnlyCartPoleEasy()
    full_env = gym.make("CartPole-v1")

    try:
        partial_obs, _ = partial_env.reset(seed=123)
        full_obs, _ = full_env.reset(seed=123)

        assert partial_obs.shape == (2,)
        assert full_obs.shape == (4,)
        assert np.allclose(partial_obs, full_obs[[0, 2]])
    finally:
        partial_env.close()
        full_env.close()


def test_binary_action_conversion():
    task = PositionOnlyCartPole()

    assert task.compute_action(torch.tensor([[-100.0]])) == 0
    assert task.compute_action(torch.tensor([[100.0]])) == 1


def test_reward_scale():
    env = popgym.envs.PositionOnlyCartPoleEasy()
    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, _ = env.step(0)

        assert np.isclose(reward, 1.0 / 200.0)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_standard_rollout_is_seed_reproducible():
    task = PositionOnlyCartPole()
    graph = make_policy_graph()

    first_loss, first_rewards = task.evaluate_graph(
        graph, n_rollouts=1, env_seed=123, hebbian=False
    )
    second_loss, second_rewards = task.evaluate_graph(
        graph, n_rollouts=1, env_seed=123, hebbian=False
    )

    assert np.isfinite(first_loss)
    assert np.allclose(first_loss, second_loss)
    assert np.allclose(first_rewards, second_rewards)


def test_hebbian_rollout_is_seed_reproducible():
    task = PositionOnlyCartPole()
    graph = make_policy_graph()

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
        test_environment_registration_and_spaces,
        test_observation_removes_velocity_components,
        test_binary_action_conversion,
        test_reward_scale,
        test_standard_rollout_is_seed_reproducible,
        test_hebbian_rollout_is_seed_reproducible,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All PositionOnlyCartPole task tests passed.")


if __name__ == "__main__":
    main()
