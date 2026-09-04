import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from NDP.ndp_nchl import HebbianNeuralDevelopmentalProgram
from NDP.ndp_nx import NeuralDevelopmentalProgram


def deterministic_model_vector(ndp):
    count = int(ndp.get_total_number_of_mlp_parameters())
    return np.linspace(-0.35, 0.35, count, dtype=np.float32)


def variant_config(strategy, seed_gene=-1.0):
    n_nodes = 8
    state_dim = 5
    node_states = np.linspace(
        -0.9,
        0.9,
        n_nodes * state_dim,
        dtype=np.float32,
    )
    shared_state = np.concatenate(
        [np.array([seed_gene], dtype=np.float32), node_states]
    )[None, :]

    return {
        "state_dim": state_dim,
        "n_nodes": n_nodes,
        "initial_graph_density": 0.35,
        "weighted_graph_flag": True,
        "network_extra_thinking": 0,
        "initial_node_state_mode": "coevolve",
        "shared_initial_node_state": shared_state,
        "noise_while_growing": False,
        "noise_while_growing_interval": 0.0,
        "edge_growing_rate": 2,
        "pruning_threshold": 0.2,
        "creating_threshold": 0.0,
        "gca_hidden_size": 4,
        "create_edge_hidden_size": 4,
        "remove_edge_hidden_size": 4,
        "wp_hidden_size": 4,
        "graph_n_inputs": 2,
        "graph_n_outputs": 1,
        "add_edge_strategy": strategy,
        "hebbian": False,
        "model": "hebbian_ndp",
    }


def standard_config():
    return {
        "state_dim": 5,
        "weighted_graph_flag": True,
        "initial_graph": "one_node",
        "network_extra_thinking": 1,
        "initial_node_state_mode": "coevolve",
        "shared_initial_node_state": np.array(
            [[0.2, -0.4, 0.6, -0.8, 1.0]], dtype=np.float32
        ),
        "noise_while_growing": False,
        "noise_while_growing_interval": 0.0,
        "add_hidden_node_to_minimal_network": False,
        "pruning_flag": False,
        "pruning_threshold": 0.2,
        "gca_hidden_size": 4,
        "rm_hidden_size": 4,
        "wp_hidden_size": 4,
        "graph_n_inputs": 2,
        "graph_n_outputs": 1,
        "hebbian": False,
        "model": "standard_ndp",
    }


def build_ndp(ndp_class, config, model_vector=None):
    ndp = ndp_class(config)
    if model_vector is None:
        model_vector = deterministic_model_vector(ndp)
    ndp.update_mlp_weights(model_vector)
    return ndp, model_vector


def assert_same_graph(first, second):
    assert first.propagation_mode == second.propagation_mode
    assert tuple(sorted(first.edges())) == tuple(sorted(second.edges()))
    assert np.array_equal(first.nodes_states, second.nodes_states)
    assert np.array_equal(first.get_weight_matrix(), second.get_weight_matrix())
    assert np.array_equal(first.get_adjacency_matrix(), second.get_adjacency_matrix())


def test_variant_reproducibility(strategy):
    config = variant_config(strategy)
    template, model_vector = build_ndp(HebbianNeuralDevelopmentalProgram, config)
    del template

    for cycles in (0, 1, 2):
        first, _ = build_ndp(
            HebbianNeuralDevelopmentalProgram,
            config,
            model_vector,
        )
        first_graph = first.develope(cycles)

        # Unrelated use of NumPy's global RNG must not affect coevolved development.
        np.random.uniform(size=1000)

        second, _ = build_ndp(
            HebbianNeuralDevelopmentalProgram,
            config,
            model_vector,
        )
        second_graph = second.develope(cycles)
        assert_same_graph(first_graph, second_graph)


def test_variant_seed_changes_initial_topology():
    first_config = variant_config("all_disconnected", seed_gene=-1.0)
    second_config = variant_config("all_disconnected", seed_gene=1.0)

    first, model_vector = build_ndp(
        HebbianNeuralDevelopmentalProgram,
        first_config,
    )
    second, _ = build_ndp(
        HebbianNeuralDevelopmentalProgram,
        second_config,
        model_vector,
    )

    first_graph = first.generate_initial_seed_graph()
    second_graph = second.generate_initial_seed_graph()

    assert tuple(sorted(first_graph.edges())) != tuple(sorted(second_graph.edges()))


def test_standard_reproducibility():
    config = standard_config()
    template, model_vector = build_ndp(NeuralDevelopmentalProgram, config)
    del template

    for cycles in (0, 1, 2):
        first, _ = build_ndp(NeuralDevelopmentalProgram, config, model_vector)
        first_graph = first.develope(cycles)

        np.random.uniform(size=1000)

        second, _ = build_ndp(NeuralDevelopmentalProgram, config, model_vector)
        second_graph = second.develope(cycles)
        assert_same_graph(first_graph, second_graph)


def main():
    tests = []

    for strategy in ("all_disconnected", "two_hop", "similarity"):
        tests.append(
            (
                f"variant_reproducibility_{strategy}",
                lambda selected=strategy: test_variant_reproducibility(selected),
            )
        )

    tests.extend(
        [
            (
                "variant_seed_changes_initial_topology",
                test_variant_seed_changes_initial_topology,
            ),
            ("standard_reproducibility", test_standard_reproducibility),
        ]
    )

    for name, test in tests:
        test()
        print(f"PASS: {name}")

    print("All development reproducibility tests passed.")


if __name__ == "__main__":
    main()
