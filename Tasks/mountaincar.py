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
    'initial_graph': 'minimal_graph',
    'node_state_random_init' : True,
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
    'n_rollouts' : 10,
    'target' : 10 * 100    
}

class MountainCar(Task):

    def __init__(self, parameters=MOUNTAINCAR_PARAMETERS):
        super().__init__(parameters)
        self.name = 'MountainCar-v0'

    def compute_action(self, output):
        probs = F.softmax(output, dim=0)  
        return int(probs.argmax())