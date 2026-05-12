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
MountainCar
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
env = gym.make("MountainCar-v0")

MOUNTAINCAR_PARAMETERS = {
    'state_dim' : 5,
    'weighted_graph_flag' : True,
    'initial_graph': 'minimal_network',
    'node_state_random_init' : True,
    'add_hidden_node_to_minimal_network': True,
    'pruning_flag' : False,
    'pruning_threshold': 0.3,
    'gca_hidden_size' : 5,
    'rm_hidden_size' : 5,
    'wp_hidden_size' : 5,
    'graph_n_inputs': env.observation_space.shape[0],  # 4
    'graph_n_outputs': env.action_space.n,
    'n_cycles': 5,
    'n_repeats': 3,
    'n_rollouts' : 3
}

MOUNTAINCAR_PARAMETERS['target'] = MOUNTAINCAR_PARAMETERS['n_rollouts'] * 110

class MountainCar(Task):

    def __init__(self, parameters=MOUNTAINCAR_PARAMETERS):
        super().__init__(parameters)
        self.name = 'MountainCar-v0'
