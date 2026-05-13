'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import random
import torch

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Import NDP Class
from NDP.ndp import NeuralDevelopmentalProgram

# Import optimisation algorithms
from Optimisation.cma_es import CMA_ES
from Optimisation.ea import EvolutionaryAlgorithm

# Import tasks
from Tasks.acrobot import Acrobot
from Tasks.cartpole import CartPole
from Tasks.mountaincar import MountainCar
from Tasks.lunarlander import LunarLander
from Tasks.xor import XOR


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Simple Test:
* Runs an NDP.
* MLP Parameters are optimised.
* Test the generated graph on an environment (you need to define it, sorry)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def simple_test():

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    task = XOR() 
    ndp = NeuralDevelopmentalProgram(task.parameters)

    n_params = ndp.get_total_number_of_mlp_parameters()
    mlp_weights = np.random.uniform(-1, 1, n_params)
    ndp.update_mlp_weights(mlp_weights)

    graph = ndp.develope(task.n_cycles, debug=False)
    # graph.summary()

    mean_reward, rollouts = task.evaluate_graph(graph)
    print('------------------------------------')
    print("Final reward:", mean_reward)
    print('------------------------------------')
    print("Rollouts:")
    print(rollouts)
    graph.summary(full=False)

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Simple Test with optimisation:
* Runs an evolutionary algorithm to optimise the MLP parameters for an NDP.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def test_with_optimisation():
    # Task
    task = XOR()

    # Params
    ndp_params = task.parameters
    evaluate_ndp = task.evaluate_ndp

    # Initial parameters
    ndp = NeuralDevelopmentalProgram(ndp_params)

    print('This is an initial test of the NDP!')
    task.summary()
    ndp.summary()

    n_params = ndp.get_total_number_of_mlp_parameters()
    # print(f'Number of NDP parameters {n_params}')

    optimisation_algorithm = 'EA'


    # Run optimisation
    print('Starting optimisation!')

    if optimisation_algorithm == 'CMA':
        # CMA
        seed = 0
        x0 = np.random.uniform(-1, 1, n_params)
        sigma0 = 0.5
        optimiser = CMA_ES(evaluate_ndp, x0, sigma0, seed)  
        best_params, best_loss = optimiser.run()

    else:
        # EA
        population_size = 50
        n_iterations = 100
        # optimiser = EvolutionaryAlgorithm(n_params, 100, 100, 200, 'name', 'env', 10, 10, evaluate_ndp_on_cartpole, run_in_parallel=True, cores = 47)
        optimiser = EvolutionaryAlgorithm(n_params, n_iterations, population_size, 250, 'name', 'env', 10, 10, evaluate_ndp, run_in_parallel=True, cores = 4)
        best_params, best_loss = optimiser.run(task.target, 0, 0)

    print('Optimisation finished!')

    print('Evaluating best model!')
    mean_reward, rollouts, best_graph, _ = evaluate_ndp(best_params, return_rewards=True)
    print('Evaluation finished!')

    print("\nBest reward:", best_loss)
    print("Final reward:", mean_reward)
    print("Rollouts:")
    print(rollouts)
    print('Best graph')
    best_graph.summary(full=False)


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main function 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':

    test_with_optimisation()
