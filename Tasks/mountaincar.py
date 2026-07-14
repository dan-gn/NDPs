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
    'state_dim': 5,
    'weighted_graph_flag': True,
    'initial_graph': 'one_node', 
    'network_extra_thinking': 5,
    'initial_node_state_mode': 'coevolve',
    'shared_initial_node_state': None,
    'noise_while_growing': True,
    'noise_while_growing_interval': 0.15,
    'add_hidden_node_to_minimal_network': False,
    'pruning_flag': False,
    'pruning_threshold': None,
    'gca_hidden_size': 5,
    'rm_hidden_size': 5,
    'wp_hidden_size': 5,
    'graph_n_inputs': env.observation_space.shape[0],  # 3
    'graph_n_outputs': env.action_space.n,
    'n_cycles': 5,
    'n_repeats': 1,
    'n_rollouts': 10,
    'hebbian': True
}


class MountainCar(Task):

    def __init__(self, parameters=MOUNTAINCAR_PARAMETERS):
        super().__init__(parameters)
        self.name = 'MountainCar-v0'
        self.target = parameters['n_rollouts'] * 110
        self.truncated_penalty = 200