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
from NDP.ndp_nx import NeuralDevelopmentalProgram

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
* Choose an environment.
* Randomly initialise the MLP weights.
* Runs the NDP to generate a graph.
* Test the generated graph on the chosen environment. 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def simple_test_v1():

    # seed = 0
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)

    task = MountainCar() 
    ndp = NeuralDevelopmentalProgram(task.parameters)

    n_params = ndp.get_total_number_of_mlp_parameters()
    mlp_weights = np.random.uniform(-1, 1, n_params)
    # mlp_weights = np.ones(n_params)*0.1
    ndp.update_mlp_weights(mlp_weights)

    graph = ndp.develope(task.n_cycles, debug=False)
    graph.summary(full=False)

    mean_reward, rollouts = task.evaluate_graph(graph)
    print('------------------------------------')
    print("Final reward:", mean_reward)
    print('------------------------------------')
    print("Rollouts:")
    print(rollouts)


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Simple Test 2:
* Choose an environment.
* Randomly initialise the MLP weights.
* Runs the NDP to generate MULTIPLE graphs.
* Test the generated graphs on the chosen environment. 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def simple_test_v2():

    # seed = 0
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)

    task = Acrobot() 
    ndp = NeuralDevelopmentalProgram(task.parameters)

    n_params = ndp.get_total_number_of_mlp_parameters()
    mlp_weights = np.random.uniform(-1, 1, n_params)
    # mlp_weights = np.ones(n_params)
    
    mean_reward, rollouts, best_graph, best_reward = task.evaluate_ndp(mlp_weights)
    best_graph.summary(full=False)
    print('------------------------------------')
    print("Final reward:", mean_reward)
    print("Best reward:", best_reward)
    print('------------------------------------')
    print("Rollouts:")
    print(rollouts)
    # graph.summary(full=False)


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Simple Test with optimisation:
* Runs an evolutionary algorithm to optimise the MLP parameters for an NDP.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def test_with_optimisation():
    # Task
    task = CartPole()

    # Params
    ndp_params = task.parameters
    evaluate_ndp = task.evaluate_ndp
    evaluate_graph = task.evaluate_graph

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
        seed = None
        colab = False
        cores = os.cpu_count() - 1 if colab else 4
        execution_environment = 'Google Colab' if colab else 'Local Computer'
        print(f'Running on {execution_environment}')
        print(f'Number of cores {cores}')
        optimiser = EvolutionaryAlgorithm(
            ndp_params = ndp_params,
            n_variables = n_params,
            max_iterations = 50, 
            population_size = 50,
            max_stagnment = 250,
            model_name = None,
            environment_name = None,
            tries = None,
            lambda_value = None,
            objective_function = evaluate_ndp,
            run_in_parallel = True,
            cores = cores
        )
        best_params, best_loss = optimiser.run(task.target, seed, 0)

    print('Optimisation finished!')
    print("\nBest mean reward:", best_loss)

    best_graph = optimiser.best_individual_by_graph.best_graph

    print('Evaluating best model!')
    if isinstance(task, XOR):
        best_graph_reward, best_graph_predictions = evaluate_graph(best_graph)
        mean_reward, rollouts, best_graph, _ = evaluate_ndp(best_params)
        rollouts = np.array(rollouts).mean(axis=0)
    else:
        best_graph_reward, best_graph_predictions = evaluate_graph(best_graph, n_rollouts=100)
        mean_reward, rollouts, best_graph, _ = evaluate_ndp(best_params, n_rollouts=100)
        rollouts = np.array(rollouts).mean(axis=0)
    print('Evaluation finished!')
    

    print('\nBest graph results on testing')
    print(f'Reward: {best_graph_reward}')
    print("Rollouts:")
    print(best_graph_predictions)
    print('Graph')
    best_graph.summary(full=False)


    print('\nBest model results on testing')
    print("Mean reward:", mean_reward)
    print("Mean rollout:")
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

