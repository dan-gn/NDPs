'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import random
import torch

import pandas as pd
import pickle
import time
import datetime

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Import NDP Class
from NDP.ndp_nx import NeuralDevelopmentalProgram
from NDP.ndp_nchl import HebbianNeuralDevelopmentalProgram
from NDP.ndp_nx_jax import NeuralDevelopmentalProgramJax
from NDP.ndp_nchl_jax import HebbianNeuralDevelopmentalProgramJax

# Import optimisation algorithms
from Optimisation.cma_es import CMA_ES
from Optimisation.ea import EvolutionaryAlgorithm

# Import tasks
from Tasks.task import Task
from Tasks.acrobot import Acrobot
from Tasks.cartpole import CartPole
from Tasks.mountaincar import MountainCar
from Tasks.lunarlander import LunarLander
from Tasks.bipedalwalker import BipedalWalker
from Tasks.xor import XOR

# Import Utilities
from Utilities.utilities import append_line_to_csv
from Utilities.utilities import is_running_in_colab

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Simple Test with optimisation:
* Runs an evolutionary algorithm to optimise the MLP parameters for an NDP.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def experiment(task:Task, optimisation_algorithm:str='EA', seed:int=None):

    # Params
    ndp_params = task.parameters
    evaluate_ndp = task.evaluate_ndp
    evaluate_graph = task.evaluate_graph

    # Initial parameters
    if ndp_params['model'] == 'standard_ndp':
        ndp = NeuralDevelopmentalProgramJax(ndp_params)
    elif ndp_params['model'] == 'hebbian_ndp':
        ndp = HebbianNeuralDevelopmentalProgramJax(ndp_params)
    else:
        raise ValueError('Model on task should be standard_ndp or hebbian_ndp.')

    print('This is an initial test of the NDP!')
    task.summary()
    ndp.summary()

    n_params = ndp.get_total_number_of_mlp_parameters()
    if ndp_params['initial_node_state_mode'] == 'coevolve':
        if ndp_params['model'] == 'hebbian_ndp':
            n_params += 1 + (ndp_params['state_dim'] * ndp_params['n_nodes'])
        else:
            n_params += ndp_params['state_dim']
    print(f'Number of NDP optimisation parameters {n_params}')

    # Run optimisation
    print(f'Seed = {seed}')
    print('Starting optimisation!')

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if optimisation_algorithm == 'CMA':
        # CMA
        x0 = np.random.uniform(-1, 1, n_params)
        sigma0 = 0.1
        optimiser = CMA_ES(evaluate_ndp, x0, sigma0, seed)  
        best_params, best_loss = optimiser.run()

    else:
        # EA
        colab = is_running_in_colab()
        cores = os.cpu_count() - 1 if colab else min(os.cpu_count() - 1, 4)
        run_in_parallel = False if cores > 1 else False
        execution_environment = 'Google Colab' if colab else 'Local Computer'
        print(f'Running on {execution_environment}')
        print(f'Running in parallel: {run_in_parallel}')
        if run_in_parallel:
            print(f'Number of cores {cores}')
        optimiser = EvolutionaryAlgorithm(
            n_variables = n_params,
            population_size = task.parameters['population_size'],
            max_iterations = task.parameters['generations'], 
            max_stagnment = task.parameters['stagnant_generation'],
            objective_function = evaluate_ndp,
            run_in_parallel = run_in_parallel,
            cores = cores
        )
        best_params, best_loss = optimiser.run(task.target, seed)

    print('Optimisation finished!')
    print("\nBest mean reward:", best_loss)

    output = {
        'task': task, 
        'optimisation_algorithm': optimisation_algorithm,
        'seed': seed,
        'optimiser': optimiser
    }

    return output


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main function: Definition
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def main():

    tasks = [
        # XOR(),
        CartPole(),
        # Acrobot(),
        # MountainCar(), 
        # LunarLander(),
        # BipedalWalker()
    ]

    models = [
        'standard_ndp',
        # 'hebbian_ndp'
    ]

    hebbian_flags = [
        False,
        # True
    ]


    initial_seed = 0
    final_seed = 1
    optimisation_algorithm = 'EA'

    for task in tasks:
        output_folder = f'Results/august2026_v2/experiments_3/{task.name}'
        if is_running_in_colab():
            output_folder = '../drive/MyDrive/' + output_folder
        os.makedirs(output_folder, exist_ok=True)
        for model in models:
            task.parameters['model'] = model
            for hebbian_flag in hebbian_flags:
                task.parameters['hebbian'] = hebbian_flag
                for seed in range(initial_seed, final_seed):
                    start_time = time.time()
                    output = experiment(task, optimisation_algorithm, seed)
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    output_filename = f'{output_folder}/output-{task.name}-{optimisation_algorithm}-seed_{seed}-time_{timestamp}.pkl'
                    with open(output_filename, 'wb') as file:
                        pickle.dump(output, file)

                    log_file = f'{output_folder}/experiments_log.csv'
                    optimiser = output['optimiser']
                    new_line = {
                        'filename' : output_filename,
                        'algorithm' : optimisation_algorithm,
                        'task' : task.name,
                        'initial_node_state_mode' : task.parameters['initial_node_state_mode'],
                        'seed' : seed,
                        'population_size': optimiser.population_size,
                        'max_iterations': optimiser.max_iterations,
                        'n_iterations' : optimiser.i,
                        'best_score_mean': optimiser.best_individual.fitness, 
                        'best_graph': optimiser.best_individual_by_graph.best_graph_fitness,
                        'best_graph_n_nodes': optimiser.best_individual_by_graph.best_graph.number_of_nodes(),
                        'best_graph_n_edges': optimiser.best_individual_by_graph.best_graph.number_of_edges(),
                        'n_variables' : optimiser.n_variables,
                        'max_stagnment' : optimiser.max_stagnment,
                        'goal_achieved' : optimiser.goal_achieved,
                        'model': task.parameters['model'],
                        'hebbian': task.parameters['hebbian'],
                        'time' : time.time() - start_time,
                        }
                    append_line_to_csv(log_file, new_line)






'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main function: Execution
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

if __name__ == '__main__':

    main()




