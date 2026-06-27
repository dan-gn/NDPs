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
    'state_dim' : 5,
    'weighted_graph_flag' : True,
    'initial_graph': 'one_node',
    'node_state_random_init' : True,
    'add_hidden_node_to_minimal_network': True,
    'pruning_flag' : False,
    'pruning_threshold': 0.3,
    'gca_hidden_size' : 5,
    'rm_hidden_size' : 5,
    'wp_hidden_size' : 5,
    'graph_n_inputs': env.observation_space.shape[0],  # 3
    'graph_n_outputs': env.action_space.n,
    'n_cycles': 5,
    'n_repeats': 5,
    'n_rollouts' : 5
}


class Acrobot(Task):

    def __init__(self, parameters=ACROBOT_PARAMETERS):
        super().__init__(parameters)
        self.name = 'Acrobot-v1'
        self.target = parameters['n_rollouts'] * 75
