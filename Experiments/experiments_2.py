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
from pathlib import Path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Import NDP Class
from NDP.ndp_nx import NeuralDevelopmentalProgram
from NDP.ndp_nchl import HebbianNeuralDevelopmentalProgram
from Baseline.fixed_mlp import FixedMLP

# Import optimisation algorithms
from Optimisation.cma_es import CMA_ES
from Optimisation.ea import EvolutionaryAlgorithm, TEST_SEED

# Import tasks
from Tasks.task import Task
from Tasks.acrobot import Acrobot
from Tasks.cartpole import CartPole
from Tasks.mountaincar import MountainCar
from Tasks.lunarlander import LunarLander
from Tasks.bipedalwalker import BipedalWalker
from Tasks.pendulum import Pendulum
from Tasks.position_only_cartpole import PositionOnlyCartPole
from Tasks.xor import XOR

# Import Utilities
from Utilities.utilities import create_experiment_log, append_line_to_csv
from Utilities.utilities import is_running_in_colab

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Simple Test with optimisation:
* Runs an evolutionary algorithm to optimise the MLP parameters for an NDP.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def is_experiment_completed(experiment_path:str):
    path = Path(experiment_path)
    if path.exists():
        return True
    return False

def create_experiment_completed(experiment_path:str):
    path = Path(experiment_path)
    path.write_text(experiment_path, encoding="utf-8")


def experiment(task:Task, optimisation_algorithm:str='EA', seed:int=None, stop_on_target:bool=True):

    # Params
    ndp_params = task.parameters
    evaluate_ndp = task.evaluate_ndp

    # Initial parameters
    if ndp_params['model'] == 'standard_ndp':
        model = NeuralDevelopmentalProgram(ndp_params)
    elif ndp_params['model'] == 'hebbian_ndp':
        model = HebbianNeuralDevelopmentalProgram(ndp_params)
    elif ndp_params['model'] == 'fixed_mlp':
        model = FixedMLP(ndp_params)
        evaluate_ndp = task.evaluate_fixed_mlp
    else:
        raise ValueError('Model on task should be standard_ndp, hebbian_ndp or fixed_mlp.')

    print('Experiment begins!')
    task.summary()
    model.summary()

    n_params = model.get_total_number_of_mlp_parameters()
    if (ndp_params['initial_node_state_mode'] == 'coevolve') and (ndp_params['model'] != 'fixed_mlp'):
        if ndp_params['model'] == 'hebbian_ndp':
            n_params += 1 + (ndp_params['state_dim'] * ndp_params['n_nodes'])
        else:
            n_params += ndp_params['state_dim']
    print(f'Number of NDP optimisation parameters {n_params}')
    print(f"Hebbian model = {ndp_params['hebbian']}")

    # Run optimisation
    print(f'Seed = {seed}')
    print('Starting optimisation!')

    if optimisation_algorithm == 'CMA':
        # CMA
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-1, 1, n_params)

        sigma0 = 0.1

        optimiser = CMA_ES(
            fitness_function = evaluate_ndp,
            x0 = x0,
            sigma0 = sigma0,
            seed = seed,
            test_seed = TEST_SEED,
            population_size = task.parameters['population_size'],
            max_iterations = task.parameters['generations'] + 1
        )

        best_params, best_loss = optimiser.run()

    else:
        # EA
        colab = is_running_in_colab()
        available_cores = max(1, os.cpu_count() - 1)
        default_max_cores = available_cores if colab else 6
        max_cores = int(os.environ.get('NDP_MAX_CORES', default_max_cores))
        cores = max(1, (min(available_cores, max_cores)))
        run_in_parallel = True if cores > 1 else False
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
            cores = cores,
            stop_on_target = stop_on_target,
            model = ndp_params['model'],
            graph_n_inputs = ndp_params['graph_n_inputs'],
            graph_n_outputs = ndp_params['graph_n_outputs']
        )

        best_params, best_loss = optimiser.run(task.target, seed)

    print('Optimisation finished!')
    print("\nBest reward:", best_loss)

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
        # BipedalWalker(),
        # Pendulum(),
        # PositionOnlyCartPole()
    ]

    models = [
        'standard_ndp',
        'hebbian_ndp'
    ]

    hebbian_flags = [
        False,
        True
    ]


    initial_seed = 0
    final_seed = 1
    optimisation_algorithm = 'EA'

    for task in tasks:
        output_folder = f'Results/september2026_ICLR_9/experiments_2/{task.name}'
        if is_running_in_colab():
            output_folder = '../drive/MyDrive/' + output_folder
        os.makedirs(output_folder, exist_ok=True)
        for model in models:
            task.parameters['model'] = model
            for hebbian_flag in hebbian_flags:
                task.parameters['hebbian'] = hebbian_flag
                for seed in range(initial_seed, final_seed):
                    output_filename = f'{output_folder}/output--{model}--hebbian{hebbian_flag}-{task.name}-{optimisation_algorithm}-seed_{seed}'
                    completed_filename = f'{output_filename}_completed'
                    if is_experiment_completed(completed_filename):
                        print(f'Test skipped: {output_filename}')
                        continue
                    start_time = time.time()
                    output = experiment(task, optimisation_algorithm, seed)
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    output_filename = f'{output_filename}-time_{timestamp}.pkl'
                    with open(output_filename, 'wb') as file:
                        pickle.dump(output, file)
                    create_experiment_completed(completed_filename)

                    log_file = f'{output_folder}/experiments_log.csv'
                    optimiser = output['optimiser']
                    new_line = create_experiment_log(output_filename, optimisation_algorithm, task, seed, optimiser, elapsed_time=time.time() - start_time)
                    append_line_to_csv(log_file, new_line)






'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main function: Execution
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

if __name__ == '__main__':

    main()




