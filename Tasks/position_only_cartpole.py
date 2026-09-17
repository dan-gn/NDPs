'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import gymnasium as gym
import popgym

from Tasks.task import Task

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Position Only Cart Pole
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

ENVIRONMENT_NAME = 'popgym-PositionOnlyCartPoleEasy-v0'

# Importing popgym registers this environment with Gymnasium.
env = gym.make(ENVIRONMENT_NAME)


POSITION_ONLY_CARTPOLE_PARAMETERS = {
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

    # Task parameters
    'graph_n_inputs': env.observation_space.shape[0],  # 2
    'graph_n_outputs': 1,  # Binary action
    'n_cycles': 5,
    'n_repeats': 1,
    'n_rollouts': 10,

    # Model selection
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
    'stagnant_generation': 200,
    'crossover_probability': 0.8,
    'mutation_probability': None, # If none then 1/n_variables
    'mutation_eta_min': 5,
    'mutation_eta_max': 15,
    'sbx_eta_min': 5,
    'sbx_eta_max': 15,
    'eta_schedule_iterations': 100
}


class PositionOnlyCartPole(Task):

    def __init__(self, parameters=POSITION_ONLY_CARTPOLE_PARAMETERS):
        super().__init__(parameters)
        self.name = ENVIRONMENT_NAME

        # A complete 200-step rollout returns approximately 1.0.
        # evaluate_graph minimizes the negative cumulative reward.
        self.target = -1.0
