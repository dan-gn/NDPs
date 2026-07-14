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
from NDP.policy_network import PolicyNetwork, NcHebbianLearningPolicyNetwork


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
    'state_dim': 5,
    'weighted_graph_flag': True,
    'initial_graph': 'one_node',
    'network_extra_thinking': 0,
    'initial_node_state_mode': 'coevolve',
    'shared_initial_node_state': None,
    'noise_while_growing': False,
    'noise_while_growing_interval': None,
    'add_hidden_node_to_minimal_network': False, 
    'pruning_flag': False,
    'pruning_threshold': None,
    'gca_hidden_size': 1,
    'rm_hidden_size': 1,
    'wp_hidden_size': 1,
    'graph_n_inputs': X_XOR.size()[1],
    'graph_n_outputs': Y_XOR.size()[1],
    'n_cycles': 4,
    'n_repeats': 2,
}


class XOR(Task):

    def __init__(self, parameters=XOR_PARAMETERS):
        super().__init__(parameters)
        self.name = 'XOR'
        self.target = 0.15

    def evaluate_graph(self, graph):
        with torch.no_grad():
            ann = PolicyNetwork(graph, self.graph_n_inputs, self.graph_n_outputs, self.network_extra_thinking)
            # ann = NcHebbianLearningPolicyNetwork(graph, self.graph_n_inputs, self.graph_n_outputs, self.network_extra_thinking)
            predictions = ann(X_XOR)
            # predictions =  torch.sigmoid(predictions)
            loss = F.mse_loss(predictions, Y_XOR)
            # accuracy = 

        return loss.item(), torch.round(torch.sigmoid(predictions))

    def evaluate_ndp(self, params:np.array, return_rewards:bool=True):
        ndp_config = dict(self.parameters)

        # params_bounded = np.tanh(params)
        params_bounded = np.clip(params, -1.0, 1.0, dtype=np.float32)

        # print(params_bounded, params_bounded.shape, type(params_bounded))

        if ndp_config['initial_node_state_mode'] == 'coevolve':
            state_dim = ndp_config['state_dim']
            ndp_config['shared_initial_node_state'] = params_bounded[np.newaxis,:state_dim]
            weights = params_bounded[state_dim:]

            # print(ndp_config['shared_initial_node_state'], ndp_config['shared_initial_node_state'].shape, type(ndp_config['shared_initial_node_state']))
            # print(weights, weights.shape)
        else:
            weights = params_bounded
            
        ndp = NeuralDevelopmentalProgram(ndp_config)
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

