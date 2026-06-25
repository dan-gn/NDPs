'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import torch
import torch.nn.functional as F

from Tasks.task import Task
from NDP.ndp_nx import NeuralDevelopmentalProgram
from NDP.policy_network import PolicyNetwork


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
XOR
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

X_XOR = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

Y_XOR = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

XOR_PARAMETERS = {
    'state_dim' : 1,
    'weighted_graph_flag' : True,
    'initial_graph' : 'one_node',
    'node_state_random_init' : True,
    'add_hidden_node_to_minimal_network' : True, 
    'pruning_flag' : False,
    'pruning_threshold': 0.3,
    'gca_hidden_size' : 1,
    'rm_hidden_size' : 1,
    'wp_hidden_size' : 1,
    'graph_n_inputs' : X_XOR.size()[1],
    'graph_n_outputs' : Y_XOR.size()[1],
    'n_cycles' : 4,
    'n_repeats' : 10,
}


class XOR(Task):

    def __init__(self, parameters=XOR_PARAMETERS):
        super().__init__(parameters)
        self.name = 'XOR'
        self.target = 0.1

    def evaluate_graph(self, graph):
        with torch.no_grad():
            ann = PolicyNetwork(graph, self.graph_n_inputs, self.graph_n_outputs)
            predictions = ann(X_XOR)
            # predictions =  torch.sigmoid(predictions)
            loss = F.mse_loss(predictions, Y_XOR)
            # accuracy = 

        return loss.item(), torch.round(torch.sigmoid(predictions))

    def evaluate_ndp(self, params, return_rewards=True):
        ndp = NeuralDevelopmentalProgram(XOR_PARAMETERS)
        # weights = np.tanh(params)
        weights = np.clip(params, -1.0, 1.0)
        ndp.update_mlp_weights(weights)

        graphs = []
        loss_list = []
        predictions_list = []
        for _ in range(self.n_repeats):
            graph = ndp.develope(self.n_cycles)
            loss, predictions = self.evaluate_graph(graph)
            graphs.append(graph)
            loss_list.append(loss)
            predictions_list.append(predictions)

        best_loss_idx = np.argmin(loss)
        best_loss = loss_list[best_loss_idx]
        best_graph = graphs[best_loss_idx]

        if return_rewards:
            return np.mean(loss_list), predictions_list, best_graph, best_loss
        else:
            return np.mean(loss_list)

