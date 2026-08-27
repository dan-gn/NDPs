"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import random
import torch
import time
import torch.nn.functional as F
import math
from concurrent.futures import ProcessPoolExecutor

from typing import Callable

from Graph.graphnx import Graphnx

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Evolutionary Algorithm

There are two classes:
1. Individual - Class for each individual of the population for the evolutionary algorithm.
2. Evolutionary Algorithm - Class for the evolutionary algorithm to perform the optimisation. 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

class Individual:

    def __init__(self, n_variables:int, genotype:np.array = None, fitness:float = None, best_graph:Graphnx = None, best_graph_fitness:float = None):
        self.n_variables = n_variables
        self.genotype = genotype
        self.fitness = fitness
        self.fitness_test = None
        self.initial_value_range = 1 # +- initial_value_range
        self.best_graph = best_graph
        self.best_graph_fitness = best_graph_fitness

    def random_initialise(self):
        self.genotype = np.random.uniform(-self.initial_value_range, self.initial_value_range, self.n_variables)

class EvolutionaryAlgorithm:

    def __init__(
            self,
            n_variables: int,
            objective_function: Callable[..., tuple],
            population_size: int,
            max_iterations: int,
            max_stagnment: int,
            mutation_probability: int = 0.01, 
            mutation_eta: int = 5, 
            sbx_eta: int = 5, 
            elitism_proportion: float = 0.1, 
            run_in_parallel: bool = False,
            cores: int = 4
        ):
        self.n_variables = n_variables
        self.objective_function = objective_function
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.max_stagnment = max_stagnment
        # self.mutation_probability = 1 / n_variables
        self.mutation_probability = mutation_probability
        self.mutation_eta = mutation_eta
        self.sbx_eta = sbx_eta
        self.elitism_proportion = elitism_proportion
        self.elitism_index = int(self.elitism_proportion * self.population_size)
        self.run_in_parallel = run_in_parallel
        self.cores = cores
        self.init_best_individual()

    # Sets seed 
    def set_seed(self, seed:int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Initialises best individual with fitness value inf
    def init_best_individual(self):
        self.best_individual = Individual(self.n_variables)
        self.best_individual.fitness = float("inf")
        self.best_individual_by_graph = Individual(self.n_variables)
        self.best_individual_by_graph.best_graph_fitness = float("inf")

    # Initialises population
    def initialise_population(self) -> list:
        population = [Individual(self.n_variables) for _ in range(self.population_size)]
        for i, member in enumerate(population):
            population[i].random_initialise()
            population[i].fitness, _, population[i].best_graph, population[i].best_graph_fitness = self.objective_function(population[i].genotype)
            if population[i].fitness < self.best_individual.fitness:
                self.best_individual.genotype = population[i].genotype
                self.best_individual.fitness = population[i].fitness
                self.best_individual.best_graph = population[i].best_graph
                self.best_individual.best_graph_fitness = population[i].best_graph_fitness
                self.best_individual.fitness_test, _, _, _ = self.objective_function(self.best_individual.genotype) 
            if population[i].best_graph_fitness < self.best_individual_by_graph.best_graph_fitness:
                self.best_individual_by_graph.genotype = population[i].genotype
                self.best_individual_by_graph.fitness = population[i].fitness
                self.best_individual_by_graph.best_graph = population[i].best_graph
                self.best_individual_by_graph.best_graph_fitness = population[i].best_graph_fitness
                self.best_individual_by_graph.fitness_test, _, _, _ = self.objective_function(self.best_individual.genotype)
        return population

    # Initialises individual (for parallel running) 
    def run_initialise_individual(self, core_seed:int) -> Individual:
        self.set_seed(core_seed)
        individual = Individual(self.n_variables)
        individual.random_initialise()
        individual.fitness, _, individual.best_graph, individual.best_graph_fitness = self.objective_function(individual.genotype)
        return individual

    # Initialises population (for parallel running) 
    def parallel_initialise_population(self) -> list:
        with ProcessPoolExecutor(max_workers=self.cores) as executor:
            population = list(executor.map(self.run_initialise_individual, range(self.n_core_seed, self.n_core_seed + self.population_size)))
        self.n_core_seed += self.population_size
        population = sorted(population, key=lambda x: x.best_graph_fitness)
        if population[0].best_graph_fitness < self.best_individual_by_graph.best_graph_fitness:
            self.best_individual_by_graph.genotype = population[0].genotype
            self.best_individual_by_graph.fitness = population[0].fitness
            self.best_individual_by_graph.best_graph = population[0].best_graph
            self.best_individual_by_graph.best_graph_fitness = population[0].best_graph_fitness
        population = sorted(population, key=lambda x: x.fitness)
        if population[0].fitness < self.best_individual.fitness:
            self.best_individual.genotype = population[0].genotype
            self.best_individual.fitness = population[0].fitness
            self.best_individual.best_graph = population[0].best_graph
            self.best_individual.best_graph_fitness = population[0].best_graph_fitness
            self.best_individual.fitness_test, _, _, _= self.objective_function(self.best_individual.genotype)
        return population

    # Random Roulette Wheel for parent selection
    def roulette_wheel(self, p:np.array) -> int:
        r = np.random.uniform(0, 1) * sum(p)	
        q = np.cumsum(p)
        return next(idx for idx, value in enumerate(q) if value >= r)

    # Parent selection by tournament
    def tournament_selection(self, n_competitors:int=2) -> list:
        all_indexes = list(range(self.population_size))
        parents = []
        for i in range(2):
            draw = np.random.permutation(all_indexes)
            competitors = draw[:n_competitors]
            winner = self.roulette_wheel(self.probs[competitors])
            parents.append(competitors[winner])
            all_indexes.remove(parents[i])
        return [self.population[parents[0]], self.population[parents[1]]]

    # Gets parent selection probabilities
    def compute_parent_selection_prob(self, beta:float=1.0) -> float:
        # Get an array of all cost of current population, add acceptance criteria value
        # and divide by the mean of the array to avoid overflow while computing exponential
        fitness = np.array([member.fitness for member in self.population]) 
        mean_fitness = abs(np.mean(fitness))
        if mean_fitness != 0 and mean_fitness != math.inf:
            fitness /= mean_fitness
        return np.exp(-beta * fitness)

    # Parent selection 
    def parent_selection(self) -> list:
        self.probs = self.compute_parent_selection_prob()
        parents = [self.tournament_selection() for _ in range(int(self.population_size/2))]
        return parents

    # Simulated Binary Crossover (SBX) 
    def sbx(self, parents:list) -> tuple:
        # Ensure parents are numpy arrays
        parent1 = np.array(parents[0].genotype)
        parent2 = np.array(parents[1].genotype)
        # Random numbers for each dimension
        rand = np.random.rand(len(parent1))
        # Compute beta values for each dimension
        beta = np.empty_like(rand)
        mask = rand <= 0.5
        beta[mask] = (2 * rand[mask]) ** (1 / (self.sbx_eta + 1))
        beta[~mask] = (1 / (2 * (1 - rand[~mask]))) ** (1 / (self.sbx_eta + 1))
        # Create offspring
        offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
        return offspring1, offspring2

    # Polynomial mutation
    def polynomial_muatation(self, x:np.array) -> np.array:
        r = np.random.uniform(0, 1)
        if r < 0.5:
            delta = (2*r) ** (1 / (self.mutation_eta+1)) - 1
        else:
            delta = 1 - (2 * (1-r)) ** (1 / (self.mutation_eta+1))
        return x + delta

    # Elitism 
    def elitism(self, offspring:list):
        offspring = sorted(offspring, key=lambda x: x.best_graph_fitness)
        if offspring[0].best_graph_fitness < self.best_individual_by_graph.best_graph_fitness:
            self.best_individual_by_graph.genotype = offspring[0].genotype.copy()
            self.best_individual_by_graph.fitness = offspring[0].fitness
            self.best_individual_by_graph.best_graph = offspring[0].best_graph
            self.best_individual_by_graph.best_graph_fitness = offspring[0].best_graph_fitness

        self.population = sorted(self.population, key=lambda x: x.fitness)
        offspring = sorted(offspring, key=lambda x: x.fitness)
        self.population[self.elitism_index:] = offspring[:-self.elitism_index]

        if offspring[0].fitness < self.best_individual.fitness:
            self.best_individual.genotype = offspring[0].genotype.copy()
            self.best_individual.fitness = offspring[0].fitness
            self.best_individual.best_graph = offspring[0].best_graph
            self.best_individual.best_graph_fitness = offspring[0].best_graph_fitness
            self.best_individual.fitness_test, _, _, _= self.objective_function(self.best_individual.genotype) 
            self.stagnment_iterations = -1
        self.stagnment_iterations += 1

    # Mutation
    def mutate(self, genotype:np.array) -> np.array:
        genotype = np.array(genotype)  # ensure it's a NumPy array
        random_values = np.random.uniform(0, 1, size=genotype.shape)
        mutation_mask = random_values <= self.mutation_probability
        for idx in np.where(mutation_mask)[0]:
            genotype[idx] = self.polynomial_muatation(genotype[idx])
        return genotype

    # Calls Crossover and Mutation
    def crossover_and_mutation(self, parents:list) -> list:
        # offspring = [Individual(self.n_variables) for _ in range(self.population_size)]
        # for i, p in enumerate(parents):
        #     genotype1, genotype2 = self.sbx(p)
        #     offspring[i*2].genotype = self.mutate(genotype1)
        #     offspring[i*2].fitness = self.evaluate(offspring[i*2].genotype, seed = self.seed)
        #     offspring[i*2 + 1].genotype = self.mutate(genotype2)
        #     offspring[i*2 + 1].fitness = self.evaluate(offspring[i*2 + 1].genotype, seed = self.seed)
        # return offspring
        offspring = []
        for p in parents:
            genotype1, genotype2 = self.sbx(p)
            mutated_g1 = self.mutate(genotype1)
            fitness_g1, _, best_graph_g1, best_graph_fitness_g1 = self.objective_function(mutated_g1)
            offspring.append(Individual(self.n_variables, genotype=mutated_g1, fitness=fitness_g1, best_graph=best_graph_g1, best_graph_fitness=best_graph_fitness_g1))
            mutated_g2 = self.mutate(genotype2)
            fitness_g2, _, best_graph_g2, best_graph_fitness_g2 = self.objective_function(mutated_g2)
            offspring.append(Individual(self.n_variables, genotype=mutated_g2, fitness=fitness_g2, best_graph=best_graph_g2, best_graph_fitness=best_graph_fitness_g2))
        return offspring


    # Runs Single Crossover and Mutation (for parallel execution)
    # This function creates a couple of offsprings by performing parent seletction, crossover, mutation and evaluation. 
    # When running in parallel, each core starts its own random generator, so I included the input "core_seed", so each time the iteration ensure a different random process.
    def run_single_crossover_and_mutation(self, core_seed:int) -> list:
        self.set_seed(core_seed)

        # Parent Selection
        # parents = self.parents[np.mod(core_seed, self.n_core_seed)]
        parents = self.tournament_selection() 

        # Crossover
        genotype1, genotype2 = self.sbx(parents)

        # Mutation
        mutated_g1 = self.mutate(genotype1)
        mutated_g2 = self.mutate(genotype2)
        
        # Evaluation
        fitness_g1, _, best_graph_g1, best_graph_fitness_g1 = self.objective_function(mutated_g1)
        fitness_g2, _, best_graph_g2, best_graph_fitness_g2 = self.objective_function(mutated_g2)

        # Return offspring individuals
        offspring1 = Individual(self.n_variables, genotype=mutated_g1, fitness=fitness_g1, best_graph=best_graph_g1, best_graph_fitness=best_graph_fitness_g1)
        offspring2 = Individual(self.n_variables, genotype=mutated_g2, fitness=fitness_g2, best_graph=best_graph_g2, best_graph_fitness=best_graph_fitness_g2)
        return [offspring1, offspring2]


    # Calls Crossover and Mutation (for parallel execution)
    def parallel_crossover_and_mutation(self) -> list:
        self.probs = self.compute_parent_selection_prob()
        n_couples = int(self.population_size/2)
        with ProcessPoolExecutor(max_workers=self.cores) as executor:
            offspring = list(executor.map(self.run_single_crossover_and_mutation, range(self.n_core_seed, self.n_core_seed + n_couples)))
        self.n_core_seed += n_couples
        offspring = np.array(offspring).flatten().tolist()
        return offspring

    # Updates population
    def update_population(self):
        if not self.run_in_parallel:
            # start_time = time.time()
            parents = self.parent_selection()
            offspring = self.crossover_and_mutation(parents)
            # print(f'Normal time = {time.time() - start_time}')
        else:
            # start_time = time.time()
            offspring = self.parallel_crossover_and_mutation()
            # print(f'Parallel time = {time.time() - start_time}')
        self.elitism(offspring)

    # Runs the Evolutionary Algorithm 
    def run(self, stop_criteria:int, seed:int) -> tuple:
        self.goal_achieved = False
        self.goal_achieved_it = None
        self.goal_achieved_individual = None
        self.goal_achieved_fitness = None
        self.record = np.zeros(self.max_iterations + 1)
        self.seed = seed
        self.stagnment_iterations = 0
        self.n_core_seed = np.random.randint(1, 2**14)   # These is the seed for the cores in parallel computing
        print('Initialising population...')
        if self.run_in_parallel:
            self.population = self.parallel_initialise_population()
        else:
            self.population = self.initialise_population()
        print('Done!')
        for self.i in range(self.max_iterations):
            start_time = time.time()
            self.record[self.i] = self.best_individual.fitness
            if self.stagnment_iterations >= self.max_stagnment:
                print('Restart population!')
                self.stagnment_iterations = -1
                self.population = self.parallel_initialise_population()
                self.population = sorted(self.population, key=lambda x: x.fitness)
                self.population[-1].genotype = self.best_individual.genotype
                self.population[-1].fitness = self.best_individual.fitness
            else:
                self.update_population()
            # if self.i % int(self.max_iterations/200) == 0:
            if self.i % 10 == 0:
                print(f'Iteration = {self.i}, Mean fitness = {np.mean([xi.fitness for xi in self.population]):.4f}, Best fitness = {self.best_individual.fitness:.4f}, Best fitness testing = {self.best_individual.fitness_test:.4f}, Best graph fitness = {self.best_individual_by_graph.best_graph_fitness:0.4f}, Iteration time = {time.time() - start_time:.2f}')
            if self.best_individual.fitness <= stop_criteria and not self.goal_achieved:
                print('Stop criteria achieved!')
                self.goal_achieved = True
                self.goal_achieved_it = self.i
                self.goal_achieved_individual = np.copy(self.best_individual.genotype)
                self.goal_achieved_fitness = self.best_individual.fitness
                break
            # print(f'Iteration total time = {time.time() - start_time}')
        self.record[self.i + 1] = self.best_individual.fitness
        return self.best_individual.genotype, self.best_individual.fitness