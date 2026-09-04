'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import pickle
import time
import datetime

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Import NDP Class
from NDP.ndp_nx_jax import NeuralDevelopmentalProgramJax
from NDP.ndp_nchl_jax import HebbianNeuralDevelopmentalProgramJax

# Import optimisation algorithms
from Optimisation.ea_jax import EvolutionaryAlgorithmJax

# Import tasks
from Tasks.task_jax import TaskJax
from Tasks.acrobot import Acrobot
from Tasks.cartpole import CartPoleJax
from Tasks.mountaincar import MountainCarJax
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

def experiment(task:TaskJax, optimisation_algorithm:str='EA', seed:int=None):

    # Params
    ndp_params = task.parameters
    evaluate_ndp = task.evaluate_ndp

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

    # Execution environment
    colab = is_running_in_colab()
    execution_environment = 'Google Colab' if colab else 'Local Computer'
    print(f'Running on {execution_environment}')

    # EA
    optimiser = EvolutionaryAlgorithmJax(
        n_variables = n_params,
        population_size = task.parameters['population_size'],
        max_iterations = task.parameters['generations'], 
        max_stagnment = task.parameters['stagnant_generation'],
        objective_function = evaluate_ndp,
    )
    best_params, best_fitness = optimiser.run(task.target, seed)

    print('Optimisation finished!')
    print("\nBest mean reward:", best_fitness)

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
        # CartPoleJax(),
        # Acrobot(),
        MountainCarJax(), 
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
        output_folder = f'Results/september2026/jax/experiments_3/{task.name}'
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
                        'best_score_mean': float(optimiser.best_individual.fitness), 
                        'best_graph': float(optimiser.best_individual_by_graph.best_graph_fitness),
                        'best_graph_n_nodes': int(optimiser.best_individual_by_graph.best_graph.number_of_nodes()),
                        'best_graph_n_edges': int(optimiser.best_individual_by_graph.best_graph.number_of_edges()),
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




