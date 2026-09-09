'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import matplotlib.pyplot as plt
import cma

import numpy as np
import random
import torch

from typing import Callable

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Utility functions
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def fitness_function(x):
    return cma.ff.rosen(x)

    

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Covariance Matrix Adaptation - Evolutionary Estrategy 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class CMA_ES:

    def __init__(
            self, 
            fitness_function:Callable[..., tuple], 
            x0:np.array, 
            sigma0:float, 
            seed:int = None,
            test_seed:int = None,
            population_size:int = None, 
            max_iterations:int = None
        ):

        self.fitness_function = fitness_function

        cma_options = {'verb_time': 0}
        if seed is not None:
            cma_options['seed'] = seed

        if population_size is not None:
            cma_options['popsize'] = population_size

        if max_iterations is not None:
            cma_options['maxiter'] = max_iterations


        self.seed = seed
        self.test_seed = test_seed
        self.population_size = population_size
        self.max_iterations = max_iterations

        self.es = cma.CMAEvolutionStrategy(x0, sigma0, cma_options)

        self.init_optimisation_variables()


    def init_optimisation_variables(self):
        self.best_params = None
        self.best_loss = None
        self.best_loss_test = None
        self.best_training_result = None
        self.best_test_result = None
        self.optimisation_evaluations = 0
        self.training_reevaluations = 0
        self.heldout_evaluations = 0


    def set_seed(self, seed:int):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @staticmethod
    def get_fitness_value(evaluation):
        if isinstance(evaluation, tuple):
            return float(evaluation[0])
        return float(evaluation)

    def evaluate(self, solutions):
        fitness_values = []

        for solution in solutions:
            evaluation = self.fitness_function(solution)
            fitness_values.append(self.get_fitness_value(evaluation))

        self.optimisation_evaluations += len(solutions)
        return fitness_values


    def run(self):
        self.init_optimisation_variables()
        self.set_seed(self.seed)

        while not self.es.stop():
            solutions = self.es.ask()
            fitness_values = self.evaluate(solutions)
            self.es.tell(solutions, fitness_values)
            self.es.logger.add()
            self.es.disp()

        self.es.result_pretty()
        # cma.plot()
        # plt.show(block=True)

        self.best_params = np.asarray(self.es.result.xbest)
        self.best_loss = float(self.es.result.fbest)

        self.best_training_result = self.fitness_function(self.best_params, env_seed=0)
        self.training_reevaluations += 1

        if self.test_seed is not None:
            self.best_test_result = self.fitness_function(self.best_params, env_seed=self.test_seed)
            self.best_loss_test = self.get_fitness_value(self.best_test_result)
            self.heldout_evaluations += 1

        return self.best_params, self.best_loss


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main function (mainly for testing) 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

if __name__ == "__main__":

    # Set random seed
    seed = 0

    # Initial model
    x0 = 12 * [0]
    sigma0 = 0.5

    # Initialise optimiser
    optimiser = CMA_ES(fitness_function, x0, sigma0, seed)

    # Run optimisation
    optimiser.run()

