'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import torch.nn.functional as F
import gymnasium as gym

from Tasks.task import Task

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Acrobot
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
env = gym.make("Acrobot-v1")

ACROBOT_PARAMETERS = {
    # Standard NDP Parameters
    'state_dim': 5,
    'weighted_graph_flag': True,
    'initial_graph': 'one_node',
    'network_extra_thinking': 5,
    'initial_node_state_mode': 'coevolve',
    'shared_initial_node_state': None,
    'noise_while_growing': False,
    'noise_while_growing_interval': None,
    'add_hidden_node_to_minimal_network': False,
    'pruning_flag': False,
    'pruning_threshold': 0.5,
    'gca_hidden_size': 5,
    'rm_hidden_size': 5,
    'wp_hidden_size': 5,
    'graph_n_inputs': env.observation_space.shape[0],  # 3
    'graph_n_outputs': env.action_space.n,
    'n_cycles': 5,
    'n_repeats': 1,
    'n_rollouts' : 10,
    # Activate Hebbian Version
    'hebbian': False,  
    # Choose between starndard or variant
    'model': 'standard_ndp',
    # Variant NDP Parameters
    'n_nodes': 16,
    'initial_graph_density': 0.2,
    'create_edge_hidden_size': 5,
    'remove_edge_hidden_size': 5,
    'edge_growing_rate': 2, # Max number of edges to add per node in each cycle
    'creating_threshold': 0,
}


class Acrobot(Task):

    def __init__(self, parameters=ACROBOT_PARAMETERS):
        super().__init__(parameters)
        self.name = 'Acrobot-v1'
        self.target = parameters['n_rollouts'] * 75
