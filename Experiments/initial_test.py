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

from NDP.ndp import NeuralDevelopmentalProgram
from Tasks.tasks import xor, evaluate_graph_on_xor
from Tasks.tasks import cartpole, evaluate_graph_on_cartpole
from Optimisation.cma_es import CMA_ES
from Optimisation.ea import EvolutionaryAlgorithm


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
    
    ndp = NeuralDevelopmentalProgram(xor.parameters)

    n_params = ndp.get_total_number_of_mlp_parameters()
    mlp_params = np.random.uniform(-1, 1, n_params)
    ndp.update_model_parameters(mlp_params)

    n_cycles = 1
    graph = ndp.develope(n_cycles, debug=False)
    # graph.summary()

    loss, predictions = evaluate_graph_on_cartpole(graph)
    print("Loss:", loss)
    print("Predictions:")
    print(predictions)


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Simple Test with optimisation:
* Runs an evolutionary algorithm to optimise the MLP parameters for an NDP.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def test_with_optimisation():
    # Task
    task = cartpole

    # Params
    fitness_function = task.fitness_function
    ndp_params = task.parameters

    # Initial parameters
    print('This is an initial test of the NDP!')
    ndp = NeuralDevelopmentalProgram(ndp_params)
    n_params = ndp.get_total_number_of_mlp_parameters()
    print(f'Number of NDP parameters {n_params}')



    optimisation_algorithm = 'EA'

    # Run optimisation
    print('Starting optimistaion!')

    if optimisation_algorithm == 'CMA':
        # CMA
        seed = 0
        x0 = np.random.uniform(-1, 1, n_params)
        sigma0 = 0.5
        optimiser = CMA_ES(fitness_function, x0, sigma0, seed)  
        best_params, best_loss = optimiser.run()

    else:
        # EA
        population_size = 50
        n_iterations = 10
        # optimiser = EvolutionaryAlgorithm(n_params, 100, 100, 200, 'name', 'env', 10, 10, evaluate_ndp_on_cartpole, run_in_parallel=True, cores = 47)
        optimiser = EvolutionaryAlgorithm(n_params, n_iterations, population_size, 200, 'name', 'env', 10, 10, fitness_function, run_in_parallel=True, cores = 4)
        best_params, best_loss = optimiser.run(-50000, 0, 0)

    print('Optimisation finished!')

    print('Evaluating best model!')
    loss, predictions = fitness_function(best_params, True)
    print('Evaluation finished!')

    print("\nBest CMA loss:", best_loss)
    print("Final loss:", loss)
    print("Predictions:")
    print(predictions)


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main function 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':

    test_with_optimisation()
