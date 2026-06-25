'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import  numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Tasks.task import Task
from NDP.policy_network import PolicyNetwork
from Graph.ndp_graph import Graphnx


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MountainCar
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
env = gym.make("MountainCar-v0")

MOUNTAINCAR_PARAMETERS = {
    'state_dim' : 16,
    'weighted_graph_flag' : True,
    'initial_graph': 'one_node',
    'node_state_random_init' : True,
    'add_hidden_node_to_minimal_network': True,
    'pruning_flag' : False,
    'pruning_threshold': 0.3,
    'gca_hidden_size' : 10,
    'rm_hidden_size' : 10,
    'wp_hidden_size' : 10,
    'graph_n_inputs': env.observation_space.shape[0],  # 3
    'graph_n_outputs': env.action_space.n,
    'n_cycles': 5,
    'n_repeats': 3,
    'n_rollouts' : 3
}


class MountainCar(Task):

    def __init__(self, parameters=MOUNTAINCAR_PARAMETERS):
        super().__init__(parameters)
        self.name = 'MountainCar-v0'
        self.target = parameters['n_rollouts'] * 110
        self.truncated_penalty = 200
        print(self.graph_n_outputs)