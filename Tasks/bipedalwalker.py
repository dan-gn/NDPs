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
Bipedal Walker
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
env = gym.make("BipedalWalker-v3")

BIPEDALWALKER_PARAMETERS = {
    # Standard NDP Parameters
    'state_dim': 16,
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
    'gca_hidden_size': 10,
    'rm_hidden_size': 10,
    'wp_hidden_size': 10,
    'graph_n_inputs': env.observation_space.shape[0],  # 24
    'graph_n_outputs': env.action_space.shape[0],   # 4
    'n_cycles': 6,
    'n_repeats': 1,
    'n_rollouts': 100,
    # Activate Hebbian Version
    'hebbian': False,  
    # Choose between starndard or variant
    'model': 'standard_ndp',
    # Variant NDP Parameters
    'n_nodes': 64,
    'initial_graph_density': 0.2,
    'create_edge_hidden_size': 5,
    'remove_edge_hidden_size': 5,
    'edge_growing_rate': 2, # Max number of edges to add per node in each cycle
    'creating_threshold': 0,
    'add_edge_strategy': 'all_disconnected',
    # Optimizer parameters
    'population_size': 64,
    'generations': 2000,
    # 'population_size': 10,
    # 'generations': 1,
    'stagnant_generation': 250,
}


class BipedalWalker(Task):

    def __init__(self, parameters=BIPEDALWALKER_PARAMETERS):
        super().__init__(parameters)
        self.name = 'BipedalWalker-v3'
        self.action_space_type = 'continuous'
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.target = parameters['n_rollouts'] * (-300)
