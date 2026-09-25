import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from NDP.mlps import WeightPredictionModel
from NDP.ndp_nx import NeuralDevelopmentalProgram


class FakeGraph:
    def __init__(self, node_states, edges):
        self.nodes_states = np.asarray(node_states, dtype=np.float32)
        self._edges = list(edges)
        self._weights = np.zeros(
            (len(self.nodes_states), len(self.nodes_states)),
            dtype=np.float32,
        )
        self.update_calls = 0

    def edges(self):
        return list(self._edges)

    def get_weight_matrix(self):
        return self._weights.copy()

    def update_weight_matrix(self, weights):
        self._weights = np.asarray(weights, dtype=np.float32).copy()
        self.update_calls += 1


def make_ndp(model):
    ndp = object.__new__(NeuralDevelopmentalProgram)
    ndp.weight_prediction_model = model
    return ndp


def sequential_predictions(model, node_states, edges):
    predictions = []
    for source, target in edges:
        source_state = torch.tensor(node_states[source], dtype=torch.float32)
        target_state = torch.tensor(node_states[target], dtype=torch.float32)
        predictions.append(model(source_state, target_state).item())
    return np.asarray(predictions, dtype=np.float32)


def test_model_batch_matches_edge_by_edge_predictions():
    torch.manual_seed(7)
    model = WeightPredictionModel(state_dim=5, hidden_dim=5)
    source_states = torch.randn(20, 5)
    target_states = torch.randn(20, 5)

    batched = model(source_states, target_states).squeeze(-1)
    sequential = torch.stack(
        [
            model(source_states[i], target_states[i]).squeeze()
            for i in range(len(source_states))
        ]
    )

    assert torch.allclose(batched, sequential, atol=1e-7)


def test_predict_weights_matches_sequential_reference():
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    model = WeightPredictionModel(state_dim=5, hidden_dim=5)
    node_states = rng.normal(size=(6, 5)).astype(np.float32)
    edges = [(0, 0), (0, 3), (1, 4), (2, 5), (4, 2), (5, 5)]
    graph = FakeGraph(node_states, edges)
    expected = sequential_predictions(model, node_states, edges)

    result = make_ndp(model).predict_weights(graph)

    assert result is graph
    assert graph.update_calls == 1
    for (source, target), expected_weight in zip(edges, expected):
        assert np.isclose(graph._weights[source, target], expected_weight)

    adjacency = np.zeros_like(graph._weights, dtype=bool)
    for source, target in edges:
        adjacency[source, target] = True
    assert np.all(graph._weights[~adjacency] == 0.0)


def test_predict_weights_handles_graph_without_edges():
    torch.manual_seed(13)
    model = WeightPredictionModel(state_dim=5, hidden_dim=5)
    graph = FakeGraph(np.zeros((3, 5), dtype=np.float32), [])

    result = make_ndp(model).predict_weights(graph)

    assert result is graph
    assert graph.update_calls == 0
    assert np.all(graph._weights == 0.0)


def main():
    tests = [
        test_model_batch_matches_edge_by_edge_predictions,
        test_predict_weights_matches_sequential_reference,
        test_predict_weights_handles_graph_without_edges,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("All batched weight-prediction tests passed.")


if __name__ == "__main__":
    main()
