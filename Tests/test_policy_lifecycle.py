import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Graph.graph_nx import Graphnx
from NDP.policy_network import NcHebbianLearningPolicyNetwork, PolicyNetwork


def make_graph():
    # The first five state entries are also the NcHL coefficients.
    states = np.array(
        [
            [0.50, 0.20, 0.30, 0.40, 0.10],
            [0.60, -0.10, 0.25, 0.30, -0.20],
            [0.40, 0.35, -0.15, -0.25, 0.30],
        ],
        dtype=np.float32,
    )

    graph = Graphnx(
        state_dim=5,
        weighted_graph_flag=True,
        propagation_mode="directed",
    )
    graph.add_nodes_from(states)

    # Node 0 is the input, node 1 is recurrent, and node 2 is the output.
    graph.add_edges_from([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)])

    weights = np.zeros((3, 3), dtype=np.float32)
    weights[0, 0] = 0.20
    weights[0, 1] = 0.50
    weights[1, 1] = 0.40
    weights[1, 2] = 0.70
    weights[2, 2] = 0.30
    graph.update_weight_matrix(weights)
    return graph


def make_standard_policy():
    return PolicyNetwork(
        graph=make_graph(),
        n_inputs=1,
        n_outputs=1,
        network_extra_thinking=0,
    )


def make_hebbian_policy():
    return NcHebbianLearningPolicyNetwork(
        graph=make_graph(),
        n_inputs=1,
        n_outputs=1,
        network_extra_thinking=0,
    )


def test_activations_persist_between_timesteps():
    first_observation = torch.tensor([[0.80]], dtype=torch.float32)
    second_observation = torch.tensor([[-0.20]], dtype=torch.float32)

    persistent_policy = make_standard_policy()
    persistent_policy(first_observation, steps=1)
    state_after_first_step = persistent_policy.current_activations.clone()
    persistent_output = persistent_policy(second_observation, steps=1)

    fresh_policy = make_standard_policy()
    fresh_output = fresh_policy(second_observation, steps=1)

    assert state_after_first_step[0, 1] != 0
    assert not torch.allclose(persistent_output, fresh_output)


def test_activation_reset_starts_a_fresh_rollout():
    first_observation = torch.tensor([[0.80]], dtype=torch.float32)
    second_observation = torch.tensor([[-0.20]], dtype=torch.float32)

    reset_policy = make_standard_policy()
    reset_policy(first_observation, steps=1)
    reset_policy.reset_activations()
    assert reset_policy.current_activations is None
    reset_output = reset_policy(second_observation, steps=1)

    fresh_output = make_standard_policy()(second_observation, steps=1)
    assert torch.allclose(reset_output, fresh_output)


def test_standard_weights_never_change():
    policy = make_standard_policy()
    initial_weights = policy.weights.clone()

    policy(torch.tensor([[0.80]], dtype=torch.float32), steps=3)
    policy(torch.tensor([[-0.20]], dtype=torch.float32), steps=3)

    assert torch.equal(policy.weights, initial_weights)


def test_hebbian_weights_change_and_respect_adjacency():
    policy = make_hebbian_policy()
    initial_weights = policy.weights.clone()

    policy(torch.tensor([[0.80]], dtype=torch.float32), steps=3)

    existing_edges = policy.adjacency_matrix.bool()
    missing_edges = ~existing_edges

    assert torch.any(policy.current_weights[existing_edges] != initial_weights[existing_edges])
    assert torch.all(policy.current_weights[missing_edges] == 0)
    assert torch.equal(policy.weights, initial_weights)


def test_hebbian_resets_are_independent():
    policy = make_hebbian_policy()
    observation = torch.tensor([[0.80]], dtype=torch.float32)

    policy(observation, steps=3)
    learned_weights = policy.current_weights.clone()
    assert not torch.equal(learned_weights, policy.weights)

    # Resetting activations must not silently reset learned weights.
    policy.reset_activations()
    assert policy.current_activations is None
    assert torch.equal(policy.current_weights, learned_weights)

    # Rollout setup explicitly resets the weights as a separate operation.
    policy.reset_weights()
    assert torch.equal(policy.current_weights, policy.weights)


def test_rollout_reset_is_reproducible():
    policy = make_hebbian_policy()
    observation = torch.tensor([[0.80]], dtype=torch.float32)

    policy.reset_activations()
    policy.reset_weights()
    first_output = policy(observation, steps=3)
    first_state = policy.current_activations.clone()
    first_learned_weights = policy.current_weights.clone()

    policy.reset_activations()
    policy.reset_weights()
    second_output = policy(observation, steps=3)

    assert torch.equal(second_output, first_output)
    assert torch.equal(policy.current_activations, first_state)
    assert torch.equal(policy.current_weights, first_learned_weights)


def main():
    tests = [
        test_activations_persist_between_timesteps,
        test_activation_reset_starts_a_fresh_rollout,
        test_standard_weights_never_change,
        test_hebbian_weights_change_and_respect_adjacency,
        test_hebbian_resets_are_independent,
        test_rollout_reset_is_reproducible,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All policy lifecycle tests passed.")


if __name__ == "__main__":
    main()
