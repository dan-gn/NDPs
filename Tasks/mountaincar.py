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


class MountainCar(Task):

    def __init__(self, parameters=MOUNTAINCAR_PARAMETERS):
        super().__init__(parameters)
        self.name = 'MountainCar-v0'
        self.target = parameters['n_rollouts'] * 110

    def evaluate_graph(self, graph, render=False, verbose=False):
        """
        Evaluates a developed NDP graph on task.

        Returns:
            mean_reward: average cumulative reward over rollouts
            rewards: list with cumulative reward of each rollout
        """

        env = gym.make(self.name, render_mode="human" if render else None)

        rewards = []
        with torch.no_grad():
            if verbose:
                print('Creating ANN from graph')
            ann = PolicyNetwork(graph, n_inputs=self.graph_n_inputs, n_outputs=self.graph_n_outputs)
            if verbose:
                print('Done!')

            for i in range(self.n_rollouts):
                actions_hist = []
                obs, _ = env.reset()

                done = False
                truncated = False
                cumulative_reward = 0.0

                while not done and not truncated:
                    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                    # print('\nSTEP\n')
                    output = ann(x)

                    action = self.compute_action(output)
                    actions_hist.append(action)

                    obs, reward, done, truncated, _ = env.step(action)

                    cumulative_reward += reward

                if truncated:
                    cumulative_reward -= 200


                # print(np.mean(actions_hist))
                if verbose:
                    print(f'Rollout {i}: Reward = {cumulative_reward}, Mean Action = {np.mean(actions_hist)}')
                rewards.append(-cumulative_reward)

        env.close()

        return np.sum(rewards), rewards