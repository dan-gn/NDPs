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
    'state_dim' : 5,
    'weighted_graph_flag' : True,
    'initial_graph': 'one_node',
    'node_state_random_init' : True,
    'shared_initial_node_state_flag' : True,
    'noise_while_growing' : False,
    'noise_while_growing_interval' : 0.15,
    'noise_while_growing' : False,
    'add_hidden_node_to_minimal_network': True,
    'pruning_flag' : False,
    'pruning_threshold': 0.3,
    'gca_hidden_size' : 5,
    'rm_hidden_size' : 5,
    'wp_hidden_size' : 5,
    'graph_n_inputs': env.observation_space.shape[0],  # 4
    'graph_n_outputs': 1,
    'n_cycles': 5,
    'n_repeats': 5,
    'n_rollouts' : 10
}


class CartPole(Task):

    def __init__(self, parameters=CARTPOLE_PARAMETERS):
        super().__init__(parameters)
        self.name = 'CartPole-v1'
        self.target = parameters['n_rollouts'] * (-500)