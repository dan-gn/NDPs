import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Graph.graph_nx import Graphnx


def make_graph(n_nodes, edges=()):
    graph = Graphnx(state_dim=5, weighted_graph_flag=True)
    graph.add_nodes_from(np.zeros((n_nodes, 5), dtype=np.float32))
    graph.add_edges_from(edges)
    return graph


def test_undersized_graph_reports_outputs_as_unreachable():
    graph = make_graph(1, [(0, 0)])

    assert graph.get_unreachable_outputs(n_inputs=6, n_outputs=3) == [6, 7, 8]
    assert graph.are_all_outputs_reachable(n_inputs=6, n_outputs=3) is False


def test_sufficient_but_disconnected_graph_reports_outputs():
    graph = make_graph(4, [(0, 0), (1, 1), (2, 2), (3, 3)])

    assert graph.get_unreachable_outputs(n_inputs=2, n_outputs=2) == [2, 3]
    assert graph.are_all_outputs_reachable(n_inputs=2, n_outputs=2) is False


def test_connected_outputs_are_reachable():
    graph = make_graph(
        4,
        [(0, 0), (1, 1), (2, 2), (3, 3), (0, 2), (1, 3)],
    )

    assert graph.get_unreachable_outputs(n_inputs=2, n_outputs=2) == []
    assert graph.are_all_outputs_reachable(n_inputs=2, n_outputs=2) is True


def test_only_disconnected_output_is_reported():
    graph = make_graph(
        4,
        [(0, 0), (1, 1), (2, 2), (3, 3), (0, 2)],
    )

    assert graph.get_unreachable_outputs(n_inputs=2, n_outputs=2) == [3]
    assert graph.are_all_outputs_reachable(n_inputs=2, n_outputs=2) is False


def main():
    tests = [
        test_undersized_graph_reports_outputs_as_unreachable,
        test_sufficient_but_disconnected_graph_reports_outputs,
        test_connected_outputs_are_reachable,
        test_only_disconnected_output_is_reported,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All graph reachability tests passed.")


if __name__ == "__main__":
    main()
