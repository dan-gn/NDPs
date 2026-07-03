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

# Import optimisation algorithms
from Optimisation.cma_es import CMA_ES
from Optimisation.ea import EvolutionaryAlgorithm

# Import tasks
from Tasks.task import Task
from Tasks.acrobot import Acrobot
from Tasks.cartpole import CartPole
from Tasks.mountaincar import MountainCar
from Tasks.lunarlander import LunarLander
from Tasks.xor import XOR


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Utilities
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def append_line_to_csv(file_path, new_line_dict):
    if os.path.exists(file_path):
        # Read the existing CSV file
        df = pd.read_csv(file_path)
        # Append the new line (as a dictionary)
        df = pd.concat([df, pd.DataFrame([new_line_dict])])
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame([new_line_dict])
    # Save it to CSV
    df.to_csv(file_path, index=False)


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
    ndp = NeuralDevelopmentalProgram(ndp_params)

    print('This is an initial test of the NDP!')
    task.summary()
    ndp.summary()

    n_params = ndp.get_total_number_of_mlp_parameters()
    # print(f'Number of NDP parameters {n_params}')

    # Run optimisation
    print(f'Seed = {seed}')
    print('Starting optimisation!')

    if optimisation_algorithm == 'CMA':
        # CMA
        x0 = np.random.uniform(-1, 1, n_params)
        sigma0 = 0.1
        optimiser = CMA_ES(evaluate_ndp, x0, sigma0, seed)  
        best_params, best_loss = optimiser.run()

    else:
        # EA
        population_size = 50
        n_iterations = 1000
        colab = False
        cores = os.cpu_count() - 1 if colab else 4
        execution_environment = 'Google Colab' if colab else 'Local Computer'
        print(f'Running on {execution_environment}')
        print(f'Number of cores {cores}')
        optimiser = EvolutionaryAlgorithm(n_params, n_iterations, population_size, 250, 'name', 'env', 10, 10, evaluate_ndp, run_in_parallel=True, cores = cores)
        best_params, best_loss = optimiser.run(task.target, seed, 0)

    print('Optimisation finished!')
    print("\nBest mean reward:", best_loss)

    output = {
        'task': task, 
        'optimisation_algorithm': optimisation_algorithm,
        'seed': seed,
        'optimiser': optimiser
    }

    return output

def main():

    tasks = [
        # XOR(),
        # Acrobot(),
        # CartPole(),
        # MountainCar(), 
        LunarLander(),
    ]

    initial_seed = 0
    final_seed = 30
    optimisation_algorithm = 'EA'

    for task in tasks:
        output_folder = f'NDPs/Results/experiments_1/{task.name}'
        os.makedirs(output_folder, exist_ok=True)
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
                'random_state' : task.parameters['node_state_random_init'],
                'seed' : seed,
                'population_size': optimiser.population_size,
                'max_iterations': optimiser.max_iterations,
                'n_iterations' : optimiser.i,
                'best_score_mean': optimiser.best_individual.fitness, 
                'best_graph': optimiser.best_individual_by_graph.best_graph_fitness,
                'n_variables' : optimiser.n_variables,
                'max_stagnment' : optimiser.max_stagnment,
                'time' : time.time() - start_time,
                }
            append_line_to_csv(log_file, new_line)






'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main function 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':

    main()




