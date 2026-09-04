import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Graph.graph_nx import Graphnx
from NDP.ndp_nchl import HebbianNeuralDevelopmentalProgram


def make_ndp(n_inputs=2, max_edges=2, seed=123):
    """Build only the state needed by the structural helper methods."""
    ndp = object.__new__(HebbianNeuralDevelopmentalProgram)
    ndp.graph_n_inputs = n_inputs
    ndp.max_edges_to_add_per_node = max_edges
    ndp.rng = np.random.default_rng(seed)
    return ndp


def make_graph(states):
    graph = Graphnx(
        state_dim=states.shape[1],
        weighted_graph_flag=True,
        propagation_mode="directed",
    )
    graph.add_nodes_from(states.astype(np.float32))
    for node in graph.nodes():
        graph.add_edge(node, node)
    return graph


def edge_set(edges):
    return {tuple(map(int, edge)) for edge in np.asarray(edges).reshape(-1, 2)}


def assert_valid_new_edges(edges, graph, n_inputs):
    for source, target in edge_set(edges):
        assert source != target, f"Unexpected self-loop candidate: {(source, target)}"
        assert target >= n_inputs, f"Unexpected edge into input node: {(source, target)}"
        assert not graph.has_edge(source, target), f"Existing edge returned: {(source, target)}"


def test_all_disconnected():
    ndp = make_ndp(n_inputs=2)
    graph = make_graph(np.eye(5, dtype=np.float32))
    graph.add_edge(0, 2)
    graph.add_edge(2, 3)

    candidates = ndp.get_all_disconnected_nodes(graph)
    assert_valid_new_edges(candidates, graph, n_inputs=2)

    expected = {
        (source, target)
        for source in range(5)
        for target in range(2, 5)
        if source != target and not graph.has_edge(source, target)
    }
    assert edge_set(candidates) == expected


def test_two_hop():
    ndp = make_ndp(n_inputs=2)
    graph = make_graph(np.eye(5, dtype=np.float32))
    graph.add_edges_from([(0, 2), (1, 2), (2, 3), (3, 4)])

    candidates = ndp.get_two_hop_disconnected_nodes(graph)
    assert_valid_new_edges(candidates, graph, n_inputs=2)
    assert edge_set(candidates) == {(0, 3), (1, 3), (2, 4)}


def test_similarity():
    ndp = make_ndp(n_inputs=1, max_edges=1)
    states = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    graph = make_graph(states)
    graph.add_edge(0, 1)

    candidates = ndp.get_similar_disconnected_nodes(graph)
    assert_valid_new_edges(candidates, graph, n_inputs=1)
    assert edge_set(candidates) == {(0, 2), (1, 2), (2, 1), (3, 2)}


def test_seeded_sampling_is_reproducible():
    candidates = [
        (source, target)
        for source in range(4)
        for target in range(2, 6)
        if source != target
    ]

    first = make_ndp(seed=987).sample_n_disconnected_edges_per_node(candidates)
    second = make_ndp(seed=987).sample_n_disconnected_edges_per_node(candidates)
    assert np.array_equal(first, second)


def test_pruning_preserves_self_loops():
    ndp = make_ndp(n_inputs=1)
    ndp.remove_edge_model = object()
    ndp.pruning_threshold = 0.0
    ndp.get_model_decision = lambda source, target, model, threshold: np.ones(
        len(source), dtype=bool
    )

    graph = make_graph(np.eye(4, dtype=np.float32))
    non_self_edges = {(0, 1), (1, 2), (2, 3)}
    graph.add_edges_from(non_self_edges)

    removable = ndp.choose_edges_to_remove(graph)
    assert edge_set(removable) == non_self_edges
    assert all(source != target for source, target in edge_set(removable))

    graph.remove_edges_from(removable)
    assert all(graph.has_edge(node, node) for node in graph.nodes())


def test_zero_weight_edge_remains_structural():
    graph = make_graph(np.eye(3, dtype=np.float32))
    graph.add_edge(0, 2)

    weights = graph.get_weight_matrix()
    weights[0, 2] = 0.0
    graph.update_weight_matrix(weights)

    assert graph.has_edge(0, 2)
    assert graph.get_weight_matrix()[0, 2] == 0.0
    assert graph.get_adjacency_matrix()[0, 2]


def main():
    tests = [
        test_all_disconnected,
        test_two_hop,
        test_similarity,
        test_seeded_sampling_is_reproducible,
        test_pruning_preserves_self_loops,
        test_zero_weight_edge_remains_structural,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All deterministic structural tests passed.")


if __name__ == "__main__":
    main()
