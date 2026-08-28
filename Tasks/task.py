'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import jax

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# from NDP.ndp_nx import NeuralDevelopmentalProgram
from NDP.ndp_nx_jax import NeuralDevelopmentalProgramJax
from NDP.ndp_nchl import HebbianNeuralDevelopmentalProgram
from NDP.policy_network import PolicyNetwork, NcHebbianLearningPolicyNetwork
from Graph.graph_nx import Graphnx

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
        self.network_extra_thinking = parameters['network_extra_thinking']
        self.n_cycles = parameters['n_cycles']
        self.n_repeats = parameters['n_repeats']
        self.n_rollouts = parameters['n_rollouts'] if 'n_rollouts' in parameters else None
        self.target = parameters['target'] if 'target' in parameters else None
        self.truncated_penalty = 0
        self.action_space_type = 'discrete'
        if parameters['initial_node_state_mode'] == 'random_shared':
            self.parameters['shared_initial_node_state'] = np.zeros((1, parameters['state_dim']))
            ndp = NeuralDevelopmentalProgramJax(self.parameters)
            self.parameters['shared_initial_node_state'] = ndp._genereate_node_state()


    def evaluate_graph(self, graph:Graphnx, n_rollouts:int=None, env_seed:int=0, render:str=False, hebbian:bool=False, verbose:bool=False):
        """
        Evaluates a developed NDP graph on task.

        Returns:
            mean_reward: average cumulative reward over rollouts
            rewards: list with cumulative reward of each rollout
        """
        if self.name == 'LunarLander-v3':
            env = gym.make(self.name, continuous=False, gravity=-10.0, enable_wind=False, render_mode="human" if render else None)
        else:
            env = gym.make(self.name, render_mode="human" if render else None)


        if n_rollouts is None:
            n_rollouts = self.n_rollouts

        rewards = []
        with torch.no_grad():
            if verbose:
                print('Creating ANN from graph')

            if hebbian:
                ann = NcHebbianLearningPolicyNetwork(graph, self.graph_n_inputs, self.graph_n_outputs, self.network_extra_thinking)
            else:
                ann = PolicyNetwork(graph, self.graph_n_inputs, self.graph_n_outputs, self.network_extra_thinking)

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
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                    output = ann(obs)

                    action = self.compute_action(output)
                    actions_hist.append(action)

                    obs, reward, done, truncated, _ = env.step(action)

                    cumulative_reward += reward

                if truncated:
                    cumulative_reward -= self.truncated_penalty

                # print(np.min(actions_hist), np.mean(actions_hist), np.max(actions_hist), len(actions_hist))
                # a = np.mean(actions_hist)
                # if not a.is_integer():
                #     print(np.min(actions_hist), np.mean(actions_hist), np.max(actions_hist), len(actions_hist), cumulative_reward)
                #     # print(a, cumulative_reward)
                if verbose:
                    print(f'Rollout {i}: Reward = {cumulative_reward}, Mean Action = {np.mean(actions_hist)}')
                rewards.append(-cumulative_reward)

        env.close()

        return np.sum(rewards), rewards

    def evaluate_ndp(self, params:np.array, n_rollouts:int=None, return_rollouts:bool=True, render:str=False):
        ndp_config = dict(self.parameters)

        # params_bounded = np.tanh(params)
        params_bounded = np.clip(params, -1.0, 1.0, dtype=np.float32)

        if ndp_config['initial_node_state_mode'] == 'coevolve':
            if ndp_config['model'] == 'hebbian_ndp':
                split_index = 1 + (ndp_config['state_dim'] * ndp_config['n_nodes'])
            elif ndp_config['model'] == 'standard_ndp':
                split_index = ndp_config['state_dim']
            ndp_config['shared_initial_node_state'] = params_bounded[np.newaxis, :split_index]
            weights = params_bounded[split_index:]
        else:
            weights = params_bounded

        if ndp_config['model'] == 'standard_ndp':
            ndp = NeuralDevelopmentalProgramJax(ndp_config)
        elif ndp_config['model'] == 'hebbian_ndp':
            ndp = HebbianNeuralDevelopmentalProgram(ndp_config)
        else:
            raise ValueError('Model on task should be standard_ndp or hebbian_ndp.')

        params = ndp.update_mlp_weights(weights)

        if n_rollouts is None:
            n_rollouts = self.n_rollouts

        key = jax.random.PRNGKey(42)

        graphs = []
        rewards = []
        rollouts = []
        for _ in range(self.n_repeats):
            graph = ndp.develope(self.n_cycles, params=params, key=key)
            reward, rollout = self.evaluate_graph(graph, n_rollouts, render=render, hebbian=ndp_config['hebbian'])
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
        # Discrete action space
        if self.action_space_type == 'discrete':
            # Binary output
            if self.graph_n_outputs == 1:   
                action =  torch.sigmoid(output)
                return int(torch.round(action))
            # Integer output
            else:   
                probs =  F.softmax(output, dim=1)
                return int(probs.argmax())
        # Continuous action space
        elif self.action_space_type == 'continuous':
            action = torch.tanh(output)
            return action.numpy().reshape(-1).astype(np.float32)
        else:
            raise ValueError('Action Space Type should be either discrete or continuous.')
    
    def summary(self):
        print('-------------------------------------')
        print('Task')
        print('-------------------------------------')
        print(f'Name = {self.name}')
        print(f'Target value = {self.target}')
        print('-------------------------------------\n')


        