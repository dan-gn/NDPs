'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym
import time

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from NDP.ndp_nx_jax import NeuralDevelopmentalProgramJax
from NDP.ndp_nchl_jax import HebbianNeuralDevelopmentalProgramJax
from NDP.policy_network_jax import PolicyNetworkJax, NcHebbianLearningPolicyNetworkJax
from Graph.graph_jax import GraphJax

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
General task
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class TaskJax:

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
            key = jax.random.PRNGKey(0)
            self.parameters['shared_initial_node_state'] = ndp._genereate_node_state(key)


    def evaluate_graph(self, graph:GraphJax, n_rollouts:int=None, env_seed:int=0, render:bool=False, hebbian:bool=False, verbose:bool=False):
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

        steps = max(1, graph.get_largest_subgraph_diameter() + self.network_extra_thinking)

        if verbose:
            print('Creating ANN from graph')

        if hebbian:
            ann = NcHebbianLearningPolicyNetworkJax(self.graph_n_inputs, self.graph_n_outputs, self.network_extra_thinking)
        else:
            ann = PolicyNetworkJax(self.graph_n_inputs, self.graph_n_outputs, self.network_extra_thinking)
            forward_fn = jax.jit(ann.forward)

        if verbose:
            print('Done!')

        rewards = []

        for i in range(n_rollouts):
            seed = env_seed + i if env_seed is not None else None
            obs, _ = env.reset(seed=seed)

            if hebbian:
                policy_state = ann.initial_state(graph)

            terminated = False
            truncated = False
            cumulative_reward = 0.0
            actions_hist = []

            policy_time = 0
            action_time = 0
            env_time = 0
            while not terminated and not truncated:

                start = time.time()

                if hebbian:
                    output, policy_state = ann.forward(graph, obs, policy_state, steps)
                else:
                    output = forward_fn(graph, obs, steps)

                jax.block_until_ready(output)
                policy_time += time.time() - start


                start = time.time()
                action = self.compute_action(output)
                action_time += time.time() - start

                actions_hist.append(action)

                start = time.time()
                obs, reward, terminated, truncated, _ = env.step(action)
                env_time += time.time() - start

                cumulative_reward += reward

            print('Policy time', policy_time)
            print('Action time', action_time)
            print('Env time', env_time)

            if truncated:
                cumulative_reward -= self.truncated_penalty

            if verbose:
                print(f'Rollout {i}: Reward = {cumulative_reward}, Mean Action = {np.mean(actions_hist)}')
            rewards.append(-cumulative_reward)

        env.close()

        return np.sum(rewards), rewards


    def evaluate_ndp(self, ndp_vector:jax.Array, n_rollouts:int=None, return_rollouts:bool=True, render:bool=False, key:jax.Array=None):
        ndp_config = dict(self.parameters)
        ndp_vector = jnp.asarray(ndp_vector, dtype=jnp.float32)
        ndp_vector = jnp.clip(ndp_vector, -1.0, 1.0)

        if ndp_config['initial_node_state_mode'] == 'coevolve':
            if ndp_config['model'] == 'hebbian_ndp':
                split_index = 1 + (ndp_config['state_dim'] * ndp_config['n_nodes'])
            elif ndp_config['model'] == 'standard_ndp':
                split_index = ndp_config['state_dim']
            ndp_config['shared_initial_node_state'] = ndp_vector[:split_index][None, :]
            weights = ndp_vector[split_index:]
        else:
            weights = ndp_vector

        if ndp_config['model'] == 'standard_ndp':
            ndp = NeuralDevelopmentalProgramJax(ndp_config)
        elif ndp_config['model'] == 'hebbian_ndp':
            ndp = HebbianNeuralDevelopmentalProgramJax(ndp_config)
        else:
            raise ValueError('Model on task should be standard_ndp or hebbian_ndp.')

        params = ndp.unpack_mlp_parameters(weights)

        if n_rollouts is None:
            n_rollouts = self.n_rollouts

        if key is None:
            key = jax.random.PRNGKey(0)

        repeat_keys = jax.random.split(key, self.n_repeats)

        graphs = []
        rewards = []
        rollouts = []
        develop_fn = jax.jit(ndp.develope, static_argnames=("n_cycles", "debug"))
        for repeat_key in repeat_keys:
            start = time.time()
            # graph = ndp.develope(self.n_cycles, params=params, key=repeat_key)

            graph = develop_fn(n_cycles=self.n_cycles, params=params, key=repeat_key, debug=False)

            jax.block_until_ready(graph.nodes_states)
            print('Development', time.time() - start)

            start = time.time()
            reward, rollout = self.evaluate_graph(graph, n_rollouts, render=render, hebbian=ndp_config['hebbian'])
            print('Rollouts', time.time() - start)
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

    def compute_action(self, output:jax.Array):

        # Discrete action space
        if self.action_space_type == 'discrete':
            # Binary output
            if self.graph_n_outputs == 1:   
                action =  jax.nn.sigmoid(output)
                action = jnp.round(action)
            # Integer output
            else:   
                action = jnp.argmax(output, axis=1)
            return int(action.item())

        # Continuous action space
        elif self.action_space_type == 'continuous':
            action = jnp.tanh(output)
            return np.asarray(action).reshape(-1).astype(np.float32)

        else:
            raise ValueError('Action Space Type should be either discrete or continuous.')
    
    def summary(self):
        print('-------------------------------------')
        print('Task')
        print('-------------------------------------')
        print(f'Name = {self.name}')
        print(f'Target value = {self.target}')
        print('-------------------------------------\n')


        