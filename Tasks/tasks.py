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
from NDP.graph_ann import GraphANN

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
General task
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class Task:

    def __init__(self, parameters, fitness_function):
        self.parameters = parameters
        self.fitness_function = fitness_function

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
XOR
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

X_XOR = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

Y_XOR = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

XOR_PARAMETERS = {
    'graph_n_inputs' : X_XOR.size()[1],
    'graph_n_outputs' : Y_XOR.size()[1],
    'state_dim' : 1,
    'weighted_graph_flag' : True,
    'pruning_flag' : False,
    'pruning_threshold': 0.3,
    'initial_graph' : 'minimal_network',
    'gca_hidden_size' : 1,
    'rm_hidden_size' : 1,
    'wp_hidden_size' : 1,
}

def evaluate_graph_on_xor(graph):
    with torch.no_grad():
        ann = GraphANN(graph, XOR_PARAMETERS['graph_n_inputs'], XOR_PARAMETERS['graph_n_outputs'])
        predictions = ann(X_XOR)
        predictions =  torch.sigmoid(predictions)
        # print(predictions)
        # predictions_01 = (predictions + 1.0) / 2.0
        loss = F.mse_loss(predictions, Y_XOR)

    return loss.item(), torch.round(predictions)

def evaluate_ndp_on_xor(params, return_predictions=False):
    n_cycles = 4
    ndp = NeuralDevelopmentalProgram(XOR_PARAMETERS)
    ndp.update_model_parameters(params)

    loss_list = []
    predictions_list = []
    for _ in range(5):
        graph = ndp.develope(n_cycles)
        loss, predictions = evaluate_graph_on_xor(graph)
        loss_list.append(loss)
        predictions_list.append(predictions_list)

    if return_predictions:
        return np.mean(loss_list), predictions
    else:
        return np.mean(loss_list)


xor = Task(XOR_PARAMETERS, evaluate_ndp_on_xor) 

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CartPole
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
env = gym.make("CartPole-v1")

CARTPOLE_PARAMETERS = {
    'state_dim' : 5,
    'weighted_graph_flag' : True,
    'initial_graph': 'one_node',
    'add_hidden_node_to_minimal_network': True,
    'pruning_flag' : False,
    'pruning_threshold': 0.3,
    'gca_hidden_size' : 5,
    'rm_hidden_size' : 5,
    'wp_hidden_size' : 5,
    'graph_n_inputs': env.observation_space.shape[0],  # 4
    'graph_n_outputs': 1
}

def evaluate_graph_on_cartpole(graph, n_rollouts=10, render=False, verbose=False):
    """
    Evaluates a developed NDP graph on CartPole-v1.

    Returns:
        mean_reward: average cumulative reward over rollouts
        rewards: list with cumulative reward of each rollout
    """

    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    rewards = []
    with torch.no_grad():
        if verbose:
            print('Creating ANN from graph')
        ann = GraphANN(graph, n_inputs=CARTPOLE_PARAMETERS['graph_n_inputs'], n_outputs=CARTPOLE_PARAMETERS['graph_n_outputs'])
        if verbose:
            print('Done!')

        for i in range(n_rollouts):
            actions_hist = []
            obs, info = env.reset()

            done = False
            truncated = False
            cumulative_reward = 0.0

            while not done and not truncated:
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                # print('\nSTEP\n')
                output = ann(x)

                # CartPole has 2 actions: left or right
                # action = torch.argmax(output, dim=1).item()
                action =  torch.sigmoid(output)
                action = int(torch.round(action))
                actions_hist.append(output)
                # print(action)
                obs, reward, done, truncated, info = env.step(action)

                cumulative_reward += reward

            # print(np.mean(actions_hist))
            if verbose:
                print(f'Rollout {i}: Reward = {cumulative_reward}, Mean Action = {np.mean(actions_hist)}')
            rewards.append(cumulative_reward)

    env.close()

    mean_reward = np.mean(rewards)

    return np.sum(rewards), rewards

def evaluate_ndp_on_cartpole(params, return_rewards=False):
    ndp = NeuralDevelopmentalProgram(CARTPOLE_PARAMETERS)
    n_cycles = 5
    ndp.update_model_parameters(params)

    cummulative_rewards = []
    rollout_rewards = []
    for i in range(5):
        graph = ndp.develope(n_cycles)
        cummulative_r, rollout_r= evaluate_graph_on_cartpole(graph, n_rollouts=5)
        cummulative_rewards.append(-cummulative_r)
        rollout_rewards += rollout_r


    if return_rewards:
        return np.mean(cummulative_rewards), rollout_rewards
    else:
        return np.mean(cummulative_rewards)
    
cartpole = Task(CARTPOLE_PARAMETERS, evaluate_ndp_on_cartpole)