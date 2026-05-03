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
    'add_hidden_node_to_minimal_network': True,
    'pruning_flag' : False,
    'pruning_threshold': 0.3,
    'gca_hidden_size' : 5,
    'rm_hidden_size' : 5,
    'wp_hidden_size' : 5,
    'graph_n_inputs': env.observation_space.shape[0],  # 4
    'graph_n_outputs': 1,
    'n_cycles': 5
}

class CartPole(Task):

    def __init__(self):
        super().__init__(parameters = CARTPOLE_PARAMETERS)
        self.env_name = 'CartPole-v1'

    def compute_action(self, output):
        action =  torch.sigmoid(output)
        return int(torch.round(action))