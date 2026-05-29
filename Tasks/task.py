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

from NDP.ndp_nx import NeuralDevelopmentalProgram
from NDP.policy_network import PolicyNetwork
from Graph.ndp_graph import Graphnx

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
General task
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class Task:

    def __init__(self, parameters:dict):
        self.parameters = dict(parameters)
        self.name = None
        self.graph_n_inputs = parameters['graph_n_inputs']
        self.graph_n_outputs = parameters['graph_n_outputs']
        self.n_cycles = parameters['n_cycles']
        self.n_repeats = parameters['n_repeats']
        self.n_rollouts = parameters['n_rollouts'] if 'n_rollouts' in parameters else None
        self.target = parameters['target'] if 'target' in parameters else None


    def evaluate_graph(self, graph:Graphnx, n_rollouts:int=None, env_seed:int=None, render:str=False, verbose:bool=False):
        """
        Evaluates a developed NDP graph on task.

        Returns:
            mean_reward: average cumulative reward over rollouts
            rewards: list with cumulative reward of each rollout
        """

        env = gym.make(self.name, render_mode="human" if render else None)

        if n_rollouts is None:
            n_rollouts = self.n_rollouts

        rewards = []
        with torch.no_grad():
            if verbose:
                print('Creating ANN from graph')
            ann = PolicyNetwork(graph, n_inputs=self.graph_n_inputs, n_outputs=self.graph_n_outputs)
            if verbose:
                print('Done!')

            for i in range(n_rollouts):
                actions_hist = []
                seed = env_seed + i if env_seed is not None else None
                obs, _ = env.reset(seed=seed)

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
                rewards.append(-cumulative_reward)

        env.close()

        return np.sum(rewards), rewards

    def evaluate_ndp(self, params:np.array, n_rollouts:int=None, return_rollouts:bool=True, render:str=False):
        ndp = NeuralDevelopmentalProgram(self.parameters)
        # weights = np.tanh(params)
        weights = np.clip(params, -1.0, 1.0)
        ndp.update_mlp_weights(weights)

        if n_rollouts is None:
            n_rollouts = self.n_rollouts

        graphs = []
        rewards = []
        rollouts = []
        for _ in range(self.n_repeats):
            graph = ndp.develope(self.n_cycles)
            reward, rollout = self.evaluate_graph(graph, n_rollouts, render=render)
            graphs.append(graph)
            rewards.append(reward)
            rollouts.extend(rollout)

        best_reward_idx = np.argmin(rewards)
        best_reward = rewards[best_reward_idx] 
        best_graph = graphs[best_reward_idx]

        if return_rollouts:
            return np.mean(rewards), rollouts, best_graph, best_reward
        else:
            return np.mean(rewards)

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
        print(f'Target value = {self.target}')
        print('-------------------------------------')
        print('\n')

        