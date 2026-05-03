'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import torch
import gymnasium as gym

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from NDP.ndp import NeuralDevelopmentalProgram
from NDP.policy_network import PolicyNetwork

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
General task
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class Task:

    def __init__(self, parameters):
        self.parameters = dict(parameters)
        self.env_name = None

    def evaluate_graph(self, graph, n_rollouts=10, render=False, verbose=False):
        """
        Evaluates a developed NDP graph on task.

        Returns:
            mean_reward: average cumulative reward over rollouts
            rewards: list with cumulative reward of each rollout
        """

        env = gym.make(self.env_name, render_mode="human" if render else None)

        rewards = []
        with torch.no_grad():
            if verbose:
                print('Creating ANN from graph')
            ann = PolicyNetwork(graph, n_inputs=self.parameters['graph_n_inputs'], n_outputs=self.parameters['graph_n_outputs'])
            if verbose:
                print('Done!')

            for i in range(n_rollouts):
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

                # print(np.mean(actions_hist))
                if verbose:
                    print(f'Rollout {i}: Reward = {cumulative_reward}, Mean Action = {np.mean(actions_hist)}')
                rewards.append(cumulative_reward)

        env.close()

        return np.sum(rewards), rewards

    def evaluate_ndp(self, params, return_rewards=False, render=False):
        ndp = NeuralDevelopmentalProgram(self.parameters)
        ndp.update_mlp_weights(params)

        cummulative_rewards = []
        rollout_rewards = []
        for i in range(self.parameters['n_repeats']):
            graph = ndp.develope(self.parameters['n_cycles'])
            cummulative_r, rollout_r= self.evaluate_graph(graph, n_rollouts=self.parameters['n_rollouts'], render=render)
            cummulative_rewards.append(-cummulative_r)
            rollout_rewards += rollout_r

        if return_rewards:
            return np.mean(cummulative_rewards), rollout_rewards
        else:
            return np.mean(cummulative_rewards)

    def compute_action(self, output):
        raise NotImplementedError('Subclasses of Task should implement the method compute_action().')

        