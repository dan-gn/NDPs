'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import torch
import torch.nn.functional as F
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
        self.name = None
        self.graph_n_inputs = self.parameters['graph_n_inputs']
        self.graph_n_outputs = self.parameters['graph_n_outputs']
        self.n_cycles = self.parameters['n_cycles']
        self.n_repeats = self.parameters['n_repeats']
        if 'n_rollouts' in self.parameters:
            self.n_rollouts = self.parameters['n_rollouts']


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

                # print(np.mean(actions_hist))
                if verbose:
                    print(f'Rollout {i}: Reward = {cumulative_reward}, Mean Action = {np.mean(actions_hist)}')
                rewards.append(cumulative_reward)

        env.close()

        return np.sum(rewards), rewards

    def evaluate_ndp(self, params, return_rewards=False, render=False):
        ndp = NeuralDevelopmentalProgram(self.parameters)
        weights = np.tanh(params)
        ndp.update_mlp_weights(weights)

        cummulative_rewards = []
        rollout_rewards = []
        for i in range(self.n_repeats):
            graph = ndp.develope(self.n_cycles)
            cummulative_r, rollout_r= self.evaluate_graph(graph, render=render)
            cummulative_rewards.append(-cummulative_r)
            rollout_rewards += rollout_r

        if return_rewards:
            return np.mean(cummulative_rewards), rollout_rewards
        else:
            return np.mean(cummulative_rewards)

    def compute_action(self, output=None):
        if self.graph_n_outputs == 1:   # Binary output
            action =  torch.sigmoid(output)
            return int(torch.round(action))
        else:   # Integer output
            probs =  F.softmax(output, dim=0)
            return int(probs.argmax())
    
    def summary(self):
        print('-------------------------------------')
        print('Task')
        print('-------------------------------------')
        print(f'Name = {self.name}')
        print(f'Target value = {self.parameters['target']}')
        print('-------------------------------------')

        