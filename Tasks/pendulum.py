'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import gymnasium as gym
from Tasks.task import Task


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Pendulum
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
env = gym.make("Pendulum-v1")


PENDULUM_PARAMETERS = {
    # Standard NDP parameters
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

    # Task dimensions
    'graph_n_inputs': env.observation_space.shape[0],  # 3
    'graph_n_outputs': env.action_space.shape[0],      # 1
    'n_cycles': 5,
    'n_repeats': 1,
    'n_rollouts': 10,

    # Policy configuration
    'hebbian': False,
    'model': 'standard_ndp',

    # Variant NDP parameters
    'n_nodes': 16,
    'initial_graph_density': 0.2,
    'create_edge_hidden_size': 5,
    'remove_edge_hidden_size': 5,
    'edge_growing_rate': 2,
    'creating_threshold': 0,
    'add_edge_strategy': 'all_disconnected',

    # Optimizer parameters
    'population_size': 64,
    'generations': 500,
    # 'population_size': 10,
    # 'generations': 1,
    'stagnant_generation': 250,
}


class Pendulum(Task):

    def __init__(self, parameters=PENDULUM_PARAMETERS):
        super().__init__(parameters)
        self.name = 'Pendulum-v1'
        self.action_space_type = 'continuous'
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.target = parameters['n_rollouts'] * 200