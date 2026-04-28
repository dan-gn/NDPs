

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Graph.ndp_graph import Node, Graph

# Compute action from environment
def compute_action(env_name, action):
    if env_name == 'CartPole-v1':
        action =  torch.sigmoid(action)
        return int(torch.round(action))
    elif env_name in ['MountainCar-v0', 'Acrobot-v1']:
        # return int(nn.functional.hardtanh(action, 0, 2))
        probs = F.softmax(action, dim=0)  
        return int(probs.argmax())
    elif env_name == 'LunarLander-v3':
        # return int(nn.functional.hardtanh(action, 0, 3))
        probs = F.softmax(action, dim=0)  
        return int(probs.argmax())


class GraphANN(nn.Module):

    def __init__(self, graph:Graph):
        super().__init__()
        
        self.graph = graph

        self.node_idx = {node.node_id:idx for idx, node in enumerate(graph.nodes)}
        self.n_nodes = len(self.node_idx)

        self.input_nodes = [node.node_id for node in graph.nodes if node.node_type == 'input']
        self.output_nodes = [node.node_id for node in graph.nodes if node.node_type == 'output']

        weights = torch.zeros((self.n_nodes, self.n_nodes), dtype=torch.float32)

        for (input_node, output_node), edge_weight in graph.edges.items():
            input_idx = self.node_idx[input_node]
            output_idx = self.node_idx[output_node]

            weights[input_idx, output_idx] = edge_weight

        self.register_buffer('weights', weights)

    def forward(self, x, steps=None):

        if steps is None:
            steps = max(1, self.graph.get_diameter())

        batch_size = x.shape[0]

        activations = torch.zeros(
            batch_size,
            self.n_nodes,
            dtype=torch.float32,
            device=x.device
        )

        input_indices = [self.node_idx[i] for i in self.input_nodes]
        output_indices = [self.node_idx[i] for i in self.output_nodes]

        activations[:, input_indices] = x

        for _ in range(steps-1):
            new_activations = torch.tanh(activations @ self.weights.T)
            new_activations[:, input_indices] = x
            activations = new_activations

        new_activations = activations @ self.weights.T
        return new_activations[:, output_indices]
                                         
                        











