'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import matplotlib.pyplot as plt
import cma

import numpy as np
import random

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Utility functions
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def fitness_function(x):
    return cma.ff.rosen(x)

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Covariance Matrix Adaptation - Evolutionary Estrategy 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class CMA_ES:

    def __init__(self, fitness_function, x0, sigma0, seed=None):
        self.fitness_function = fitness_function
        cma_options = {'verb_time': 0}
        if seed is not None:
            cma_options['seed'] = seed
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, cma_options)
        self.seed = seed

    def evaluate(self, solutions):
        # return [cma.ff.rosen(x) for x in solutions]
        return [self.fitness_function(x) for x in solutions]

    def run(self):
        set_seed(self.seed)
        while not self.es.stop():
            solutions = self.es.ask()
            self.es.tell(solutions, self.evaluate(solutions))
            self.es.logger.add()
            self.es.disp()
        self.es.result_pretty()
        # cma.plot()
        # plt.show(block=True)


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

