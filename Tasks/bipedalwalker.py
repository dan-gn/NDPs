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
    'invalid_graph_fitness': 2_000.0,
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

    # Task parameters
    'graph_n_inputs': env.observation_space.shape[0],  # 24
    'graph_n_outputs': env.action_space.shape[0],   # 4
    'n_cycles': 6,
    'n_repeats': 1,
    'n_rollouts': 20,

    # Model selection
    'hebbian': False,  
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
    'population_size': 128,
    'generations': 2000,
    'stagnant_generation': 200,
    'crossover_probability': 0.8,
    'mutation_probability': None, # If none then 1/n_variables
    'mutation_eta_min': 5,
    'mutation_eta_max': 20,
    'sbx_eta_min': 10,
    'sbx_eta_max': 30,
    'eta_schedule_iterations': 100
}


class BipedalWalker(Task):

    def __init__(self, parameters=BIPEDALWALKER_PARAMETERS):
        super().__init__(parameters)
        self.name = 'BipedalWalker-v3'
        self.action_space_type = 'continuous'
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        # self.target = parameters['n_rollouts'] * (-300)
        self.target = -300
