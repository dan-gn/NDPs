'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


import numpy as np
import torch
import torch.nn.functional as F

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from NDP.ndp import NeuralDevelopmentalProgram
from NDP.graph_ann import GraphANN
from Tasks.tasks import XOR_PARAMETERS, evaluate_graph_on_xor
from Tasks.tasks import CARTPOLE_PARAMETERS, evaluate_graph_on_cartpole
from CMA_ES.cma_es import CMA_ES
import cma
from CMA_ES.ea import EvolutionaryAlgorithm


def evaluate_ndp_on_xor(params):
    ndp = NeuralDevelopmentalProgram(task=XOR_PARAMETERS)
    n_cycles = 10
    ndp.update_model_parameters(params)

    graph = ndp.develope(n_cycles)
    loss, _ = evaluate_graph_on_xor(graph)

    return loss

def evaluate_ndp_on_cartpole(params):
    ndp = NeuralDevelopmentalProgram(task=CARTPOLE_PARAMETERS)
    n_cycles = 5
    ndp.update_model_parameters(params)

    graph = ndp.develope(n_cycles)
    # graph.summary()

    mean_reward, rewards = evaluate_graph_on_cartpole(
        graph,
        n_rollouts=5
    )

    return -mean_reward

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main function (mainly for testing)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    
    # n_cycles = 10
    # ndp = NeuralDevelopmentalProgram(task=CARTPOLE_PARAMETERS)

    # n_params = ndp.get_total_number_of_parameters()
    # mlp_params = np.random.uniform(-5, 5, n_params)
    # ndp.update_model_parameters(mlp_params)
    # graph = ndp.develope(n_cycles, debug=False)
    # # graph.summary()
    # loss, predictions = evaluate_graph_on_cartpole(graph, verbose=False)

    # print("Loss:", loss)
    # print("Predictions:")
    # print(predictions)


    # Initial parameters
    print('This is an initial test of the NDP!')
    ndp = NeuralDevelopmentalProgram(task=CARTPOLE_PARAMETERS)
    n_cycles = 10
    n_params = ndp.get_total_number_of_parameters()
    print(f'Number of NDP parameters {n_params}')
    x0 = np.random.uniform(-1, 1, n_params)
    sigma0 = 0.5

    # Set random seed
    seed = 0

    # Initialise optimiser
    # optimiser = CMA_ES(evaluate_ndp_on_cartpole, x0, sigma0, seed)
    optimiser = EvolutionaryAlgorithm(n_params, 100, 100, 200, 'name', 'env', 10, 10, evaluate_ndp_on_cartpole, run_in_parallel=True, cores = 47)

    # Run optimisation
    print('Starting optimistaion!')
    best_params, best_loss = optimiser.run(-5000, 0, 0)
    print('Optimisation finished!')

    print('Evaluating best model!')
    ndp.update_model_parameters(best_params)
    graph = ndp.develope(n_cycles)
    loss, predictions = evaluate_graph_on_cartpole(graph)
    print('Evaluation finished!')

    print("\nBest CMA loss:", best_loss)
    print("Final loss:", loss)
    print("Predictions:")
    print(predictions)