'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import torch
import gymnasium as gym

from Tasks.task import Task

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CartPole
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
env = gym.make("CartPole-v1")

CARTPOLE_PARAMETERS = {
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
    'graph_n_inputs': env.observation_space.shape[0],  # 4
    'graph_n_outputs': 1,  # DON'T USE env.action_space.n for CartPole, use 1 instead cause it makes more sense (and works better).
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
    # Optimizer parameters
    'population_size': 64,
    'generations': 500,
}



class CartPole(Task):

    def __init__(self, parameters=CARTPOLE_PARAMETERS):
        super().__init__(parameters)
        self.name = 'CartPole-v1'
        self.target = parameters['n_rollouts'] * (-500)