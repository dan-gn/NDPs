'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import torch
import torch.nn.functional as F

from Tasks.task import Task
from NDP.ndp import NeuralDevelopmentalProgram
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
    'graph_n_inputs' : X_XOR.size()[1],
    'graph_n_outputs' : Y_XOR.size()[1],
    'state_dim' : 1,
    'weighted_graph_flag' : True,
    'pruning_flag' : False,
    'pruning_threshold': 0.3,
    'initial_graph' : 'minimal_network',
    'gca_hidden_size' : 1,
    'rm_hidden_size' : 1,
    'wp_hidden_size' : 1,
    'n_cycles' : 4
}


class XOR(Task):

    def __init__(self):
        super().__init__(parameters = XOR_PARAMETERS)
        self.n_cycles = 4

    def evaluate_graph(self, graph):
        with torch.no_grad():
            ann = PolicyNetwork(graph, XOR_PARAMETERS['graph_n_inputs'], XOR_PARAMETERS['graph_n_outputs'])
            predictions = ann(X_XOR)
            predictions =  torch.sigmoid(predictions)
            loss = F.mse_loss(predictions, Y_XOR)

        return loss.item(), torch.round(predictions)

    def evaluate_ndp(self, params, return_predictions=False):
        ndp = NeuralDevelopmentalProgram(XOR_PARAMETERS)
        ndp.update_model_parameters(params)

        loss_list = []
        predictions_list = []
        for _ in range(5):
            graph = ndp.develope(self.n_cycles)
            loss, predictions = self.evaluate_graph(graph)
            loss_list.append(loss)
            predictions_list.append(predictions_list)

        if return_predictions:
            return np.mean(loss_list), predictions
        else:
            return np.mean(loss_list)

