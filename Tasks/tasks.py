import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from NDP.ndp import NeuralDevelopmentalProgram
from NDP.graph_ann import GraphANN


import numpy as np
import torch
import torch.nn.functional as F
import cma

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
    'n_inputs' : X_XOR.size()[1],
    'n_outputs' : Y_XOR.size()[1]
}

def evaluate_ndp_on_xor(mlp_parameters, n_cycles = 1):

    ndp = NeuralDevelopmentalProgram(task=XOR_PARAMETERS)

    with torch.no_grad():
        graph = ndp.develope(n_cycles)
        ann = GraphANN(graph)
        predictions = ann(X_XOR)
        predictions_01 = (predictions + 1.0) / 2.0
        loss = F.mse_loss(predictions_01, Y_XOR)

    return loss.item(), predictions

def evaluate_graph_on_xor(graph):

    with torch.no_grad():
        ann = GraphANN(graph)
        predictions = ann(X_XOR)
        predictions =  torch.sigmoid(predictions)
        # predictions_01 = (predictions + 1.0) / 2.0
        loss = F.mse_loss(predictions, Y_XOR)

    return loss.item(), predictions


import gymnasium as gym

env = gym.make("CartPole-v1")

CARTPOLE_PARAMETERS = {
    'n_inputs': env.observation_space.shape[0],  # 4
    'n_outputs': 1
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
        ann = GraphANN(graph)
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

                output = ann(x)


                # CartPole has 2 actions: left or right
                # action = torch.argmax(output, dim=1).item()
                action =  torch.sigmoid(output)
                action = int(torch.round(action))
                actions_hist.append(action)
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