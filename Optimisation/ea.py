"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import random
import torch
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from typing import Callable

from Graph.graph_nx import Graphnx

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Evolutionary Algorithm

There are two classes:
1. Individual - Class for each individual of the population for the evolutionary algorithm.
2. Evolutionary Algorithm - Class for the evolutionary algorithm to perform the optimisation. 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

TEST_SEED = 10000

class Individual:

    def __init__(
            self, 
            n_variables:int, 
            genotype:np.array = None, 
            fitness:float = None, 
            best_graph:Graphnx = None, 
            best_graph_fitness:float = None, 
            best_graph_fitness_test:float = None):

        self.n_variables = n_variables
        self.genotype = genotype
        self.fitness = fitness
        self.fitness_test = None
        self.initial_value_range = 1 # +- initial_value_range
        self.best_graph = best_graph
        self.best_graph_fitness = best_graph_fitness
        self.best_graph_fitness_test = best_graph_fitness_test
        self.best_graph_used_nodes = None
        self.best_graph_used_edges = None

    def random_initialise(self):
        self.genotype = np.random.uniform(-self.initial_value_range, self.initial_value_range, self.n_variables)

class EvolutionaryAlgorithm:

    # ---------------------------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------------------------

    def __init__(
            self,
            n_variables: int,
            objective_function: Callable[..., tuple],
            population_size: int,
            max_iterations: int,
            max_stagnment: int,
            crossover_probability: float = 0.8,
            mutation_probability: float = None, 
            mutation_eta_min: float = 5, 
            mutation_eta_max: float = 15, 
            sbx_eta_min: float = 5, 
            sbx_eta_max: float = 15,
            eta_schedule_iterations:int = 100, 
            elitism_proportion: float = 0.1, 
            run_in_parallel: bool = False,
            cores: int = 4,
            graph_n_inputs: int = None,
            model: str = None,
            graph_n_outputs: int = None,
            stop_on_target: bool = True
        ):
        self.n_variables = n_variables
        self.objective_function = objective_function
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.max_stagnment = max_stagnment
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability if mutation_probability is not None else 1 / n_variables
        self.mutation_eta_min = mutation_eta_min
        self.mutation_eta_max = mutation_eta_max
        self.sbx_eta_min = sbx_eta_min
        self.sbx_eta_max = sbx_eta_max
        self.eta_schedule_iterations = eta_schedule_iterations 
        self.elitism_proportion = elitism_proportion
        self.elitism_index = max(1, int(self.elitism_proportion * self.population_size))
        self.run_in_parallel = run_in_parallel
        self.cores = cores
        self.stop_on_target = stop_on_target
        self.optimisation_evaluations = 0
        self.training_reevaluations = 0
        self.heldout_evaluations = 0
        self.model = model
        self.graph_n_inputs = graph_n_inputs
        self.graph_n_outputs = graph_n_outputs
        self.init_best_individual()

    # Sets seed 
    def set_seed(self, seed:int):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # Initialises best individual with fitness value inf
    def init_best_individual(self):
        self.best_individual = Individual(self.n_variables)
        self.best_individual.fitness = float("inf")
        self.best_individual_by_graph = Individual(self.n_variables)
        self.best_individual_by_graph.best_graph_fitness = float("inf")

    # ---------------------------------------------------------------------------------------
    # Fitness evaluation
    # ---------------------------------------------------------------------------------------

    # Function to test on the testing set of seeds
    def evaluate_heldout(self, genotype):
        self.heldout_evaluations += 1
        return self.objective_function(genotype, env_seed = TEST_SEED)

    def evaluate_final_heldout(self):
        self.best_individual.fitness_test, _, _, self.best_individual.best_graph_fitness_test = self.evaluate_heldout(self.best_individual.genotype)
        # Commented cause the NDP development is deterministic by using coevolve strategy
        # if self.best_individual.best_graph_fitness == self.best_individual_by_graph.best_graph_fitness:
        #     self.best_individual_by_graph.fitness_test = self.best_individual.fitness_test
        #     self.best_individual_by_graph.best_graph_fitness_test = self.best_individual.best_graph_fitness_test
        # else:
        #     self.best_individual_by_graph.fitness_test, _, _, self.best_individual_by_graph.best_graph_fitness_test = self.evaluate_heldout(self.best_individual_by_graph.genotype)

    # ---------------------------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------------------------

    # Random Roulette Wheel for parent selection
    def roulette_wheel(self, p:np.array) -> int:
        weights = np.asarray(p, dtype=float)
        total_weight = np.sum(weights)

        if not np.isfinite(total_weight) or total_weight <= 0:
            return np.random.randint(len(weights))

        r = np.random.uniform(0, total_weight)
        cumulative_weights = np.cumsum(weights)
        return min(int(np.searchsorted(cumulative_weights, r, side="right")), len(weights) - 1)

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
    def compute_parent_selection_prob(self, beta:float=1.0) -> np.ndarray:
        fitness = np.asarray([member.fitness for member in self.population], dtype=float)

        if not np.all(np.isfinite(fitness)):
            invalid_fitness = fitness[~np.isfinite(fitness)]

            raise ValueError(
                "Parent selection received non-finite fitness values: "
                f"{invalid_fitness}"
            )

        absolute_fitness = np.abs(fitness)
        maximum_absolute_fitness = float(np.max(absolute_fitness))
        if maximum_absolute_fitness == 0.0:
            scale = 1.0
        else:
            scale = maximum_absolute_fitness * float(np.mean(absolute_fitness / maximum_absolute_fitness))

        log_weights = -beta * fitness / scale
        log_weights -= np.max(log_weights)

        weights = np.exp(log_weights)
        return weights

    def update_eta_schedule(self):
        self.adaptability_coefficient = np.clip(self.adaptability_coefficient, 0, self.eta_schedule_iterations)
        progress = self.adaptability_coefficient / self.eta_schedule_iterations
        self.mutation_eta = self.mutation_eta_min + progress * (self.mutation_eta_max - self.mutation_eta_min)
        self.sbx_eta = self.sbx_eta_min + progress * (self.sbx_eta_max - self.sbx_eta_min)

    # ---------------------------------------------------------------------------------------
    # Evolutionary Functions Implemented on Parallel
    # ---------------------------------------------------------------------------------------

    # Initialises individual (for parallel running) 
    def run_initialise_individual(self, core_seed:int) -> Individual:
        self.set_seed(core_seed)
        individual = Individual(self.n_variables)
        individual.random_initialise()
        individual.fitness, _, individual.best_graph, individual.best_graph_fitness = self.objective_function(individual.genotype)
        return individual

    # Initialises population (for parallel running) 
    def parallel_initialise_population(self, executor) -> list:
        population = list(executor.map(self.run_initialise_individual, range(self.n_core_seed, self.n_core_seed + self.population_size)))
        self.n_core_seed += self.population_size
        population = sorted(population, key=lambda x: x.best_graph_fitness)
        if population[0].best_graph_fitness < self.best_individual_by_graph.best_graph_fitness:
            self.best_individual_by_graph.genotype = population[0].genotype.copy()
            self.best_individual_by_graph.fitness = population[0].fitness
            self.best_individual_by_graph.best_graph = population[0].best_graph
            self.best_individual_by_graph.best_graph_fitness = population[0].best_graph_fitness
            if self.model is not None and 'ndp' in self.model:
                self.best_individual_by_graph.best_graph_used_nodes = population[0].best_graph.get_number_of_used_nodes(self.graph_n_inputs, self.graph_n_outputs)
                self.best_individual_by_graph.best_graph_used_edges = population[0].best_graph.get_number_of_used_edges(self.graph_n_inputs, self.graph_n_outputs)
        population = sorted(population, key=lambda x: x.fitness)
        if population[0].fitness < self.best_individual.fitness:
            self.best_individual.genotype = population[0].genotype.copy()
            self.best_individual.fitness = population[0].fitness
            self.best_individual.best_graph = population[0].best_graph
            self.best_individual.best_graph_fitness = population[0].best_graph_fitness
            if self.model is not None and 'ndp' in self.model:
                self.best_individual.best_graph_used_nodes = population[0].best_graph.get_number_of_used_nodes(self.graph_n_inputs, self.graph_n_outputs)
                self.best_individual.best_graph_used_edges = population[0].best_graph.get_number_of_used_edges(self.graph_n_inputs, self.graph_n_outputs)
        return population

    # Runs Single Crossover and Mutation (for parallel execution)
    # This function creates a couple of offsprings by performing parent seletction, crossover, mutation and evaluation. 
    # When running in parallel, each core starts its own random generator, so I included the input "core_seed", so each time the iteration ensure a different random process.
    def run_single_crossover_and_mutation(self, core_seed:int) -> list:
        self.set_seed(core_seed)

        # Parent Selection
        parents = self.tournament_selection() 

        # Crossover
        genotype1, genotype2 = self.crossover(parents)

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
    def parallel_crossover_and_mutation(self, executor) -> list:
        self.probs = self.compute_parent_selection_prob()
        n_couples = int(self.population_size/2)
        offspring = list(executor.map(self.run_single_crossover_and_mutation, range(self.n_core_seed, self.n_core_seed + n_couples)))
        self.n_core_seed += n_couples
        offspring = np.array(offspring).flatten().tolist()
        return offspring

    # ---------------------------------------------------------------------------------------
    # Evolutionary Process 
    # ---------------------------------------------------------------------------------------

    # Initialises population
    def initialise_population(self) -> list:
        population = [Individual(self.n_variables) for _ in range(self.population_size)]
        for i, member in enumerate(population):
            population[i].random_initialise()
            population[i].fitness, _, population[i].best_graph, population[i].best_graph_fitness = self.objective_function(population[i].genotype)
            if population[i].fitness < self.best_individual.fitness:
                self.best_individual.genotype = population[i].genotype.copy()
                self.best_individual.fitness = population[i].fitness
                self.best_individual.best_graph = population[i].best_graph
                self.best_individual.best_graph_fitness = population[i].best_graph_fitness
                if self.model is not None and 'ndp' in self.model:
                    self.best_individual.best_graph_used_nodes = population[i].best_graph.get_number_of_used_nodes(self.graph_n_inputs, self.graph_n_outputs)
                    self.best_individual.best_graph_used_edges = population[i].best_graph.get_number_of_used_edges(self.graph_n_inputs, self.graph_n_outputs)
            if population[i].best_graph_fitness < self.best_individual_by_graph.best_graph_fitness:
                self.best_individual_by_graph.genotype = population[i].genotype.copy()
                self.best_individual_by_graph.fitness = population[i].fitness
                self.best_individual_by_graph.best_graph = population[i].best_graph
                self.best_individual_by_graph.best_graph_fitness = population[i].best_graph_fitness
                if self.model is not None and 'ndp' in self.model:
                    self.best_individual_by_graph.best_graph_used_nodes = population[i].best_graph.get_number_of_used_nodes(self.graph_n_inputs, self.graph_n_outputs)
                    self.best_individual_by_graph.best_graph_used_edges = population[i].best_graph.get_number_of_used_edges(self.graph_n_inputs, self.graph_n_outputs)
        return population

    # Parent selection 
    def parent_selection(self) -> list:
        self.probs = self.compute_parent_selection_prob()
        parents = [self.tournament_selection() for _ in range(int(self.population_size/2))]
        return parents

    # Simulated Binary Crossover (SBX) 
    def sbx(self, parents:list) -> tuple:
        # Ensure parents are numpy arrays
        parent1 = np.asarray(parents[0].genotype)
        parent2 = np.asarray(parents[1].genotype)
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
            if self.model is not None and 'ndp' in self.model:
                self.best_individual_by_graph.best_graph_used_nodes = offspring[0].best_graph.get_number_of_used_nodes(self.graph_n_inputs, self.graph_n_outputs)
                self.best_individual_by_graph.best_graph_used_edges = offspring[0].best_graph.get_number_of_used_edges(self.graph_n_inputs, self.graph_n_outputs)

        self.population = sorted(self.population, key=lambda x: x.fitness)
        offspring = sorted(offspring, key=lambda x: x.fitness)
        self.population[self.elitism_index:] = offspring[:-self.elitism_index]

        if offspring[0].fitness < self.best_individual.fitness:
            self.best_individual.genotype = offspring[0].genotype.copy()
            self.best_individual.fitness = offspring[0].fitness
            self.best_individual.best_graph = offspring[0].best_graph
            self.best_individual.best_graph_fitness = offspring[0].best_graph_fitness
            if self.model is not None and 'ndp' in self.model:
                self.best_individual.best_graph_used_nodes = offspring[0].best_graph.get_number_of_used_nodes(self.graph_n_inputs, self.graph_n_outputs)
                self.best_individual.best_graph_used_edges = offspring[0].best_graph.get_number_of_used_edges(self.graph_n_inputs, self.graph_n_outputs)
            self.stagnment_iterations = -1
        self.stagnment_iterations += 1

    # Crossover
    def crossover(self, parents:list) -> tuple:
        random_value = np.random.uniform()
        if random_value <= self.crossover_probability:
            return self.sbx(parents)
        else:
            parent1 = np.asarray(parents[0].genotype).copy()
            parent2 = np.asarray(parents[1].genotype).copy()
            return parent1, parent2 

    # Mutation
    def mutate(self, genotype:np.array) -> np.array:
        genotype = np.asarray(genotype)  
        random_values = np.random.uniform(0, 1, size=genotype.shape)
        mutation_mask = random_values <= self.mutation_probability
        for idx in np.where(mutation_mask)[0]:
            genotype[idx] = self.polynomial_muatation(genotype[idx])
        return genotype

    # Calls Crossover and Mutation
    def crossover_and_mutation(self, parents:list) -> list:
        offspring = []
        for p in parents:
            genotype1, genotype2 = self.crossover(p)
            mutated_g1 = self.mutate(genotype1)
            fitness_g1, _, best_graph_g1, best_graph_fitness_g1 = self.objective_function(mutated_g1)
            offspring.append(Individual(self.n_variables, genotype=mutated_g1, fitness=fitness_g1, best_graph=best_graph_g1, best_graph_fitness=best_graph_fitness_g1))
            mutated_g2 = self.mutate(genotype2)
            fitness_g2, _, best_graph_g2, best_graph_fitness_g2 = self.objective_function(mutated_g2)
            offspring.append(Individual(self.n_variables, genotype=mutated_g2, fitness=fitness_g2, best_graph=best_graph_g2, best_graph_fitness=best_graph_fitness_g2))
        return offspring

    # Updates population
    def update_population(self, executor=None):
        if not self.run_in_parallel:
            parents = self.parent_selection()
            offspring = self.crossover_and_mutation(parents)
        else:
            offspring = self.parallel_crossover_and_mutation(executor)
        self.optimisation_evaluations += self.population_size
        self.elitism(offspring)

    # ---------------------------------------------------------------------------------------
    # Run Optimisation
    # ---------------------------------------------------------------------------------------

    def init_optimisation_variables(self):
        self.goal_achieved = False
        self.goal_achieved_it = None
        self.goal_achieved_individual = None
        self.goal_achieved_fitness = None
        self.optimisation_evaluations = 0
        self.training_reevaluations = 0
        self.heldout_evaluations = 0
        self.record = np.zeros(self.max_iterations + 1)
        self.stagnment_iterations = 0
        self.adaptability_coefficient = 0
        self.update_eta_schedule()

    # Runs the Evolutionary Algorithm 
    def run(self, stop_criteria:int, seed:int) -> tuple:
        self.init_optimisation_variables()
        self.set_seed(seed)
        self.n_core_seed = np.random.randint(1, 2**14)   # These is the seed for the cores in parallel computing
        executor = None

        try:
            self.summary()
            print('Initialising population...')
            if self.run_in_parallel:
                executor = ProcessPoolExecutor(max_workers=self.cores, mp_context=mp.get_context("spawn"))
                self.population = self.parallel_initialise_population(executor)
            else:
                self.population = self.initialise_population()
            self.optimisation_evaluations += self.population_size
            print('Done!')
            for self.i in range(self.max_iterations):
                start_time = time.time()
                self.update_eta_schedule()
                self.record[self.i] = self.best_individual.fitness
                if self.stagnment_iterations >= self.max_stagnment:
                    print('Restart population!')
                    self.stagnment_iterations = 0
                    self.adaptability_coefficient = 0
                    self.update_eta_schedule()  # Just updating the values to get them on the print
                    if self.run_in_parallel:
                        self.population = self.parallel_initialise_population(executor)
                    else:
                        self.population = self.initialise_population()

                    self.optimisation_evaluations += self.population_size
                    self.population = sorted(self.population, key=lambda x: x.fitness)
                    self.population[-1] = Individual(
                        self.n_variables, 
                        genotype = self.best_individual.genotype.copy(), 
                        fitness = self.best_individual.fitness,
                        best_graph =  self.best_individual.best_graph,
                        best_graph_fitness=self.best_individual.best_graph_fitness
                        )
                else:
                    self.update_population(executor)
                    if (self.stagnment_iterations >= (self.max_stagnment - self.eta_schedule_iterations)):
                        self.adaptability_coefficient -= 1
                    else:
                        self.adaptability_coefficient += 1




                # if self.i % int(self.max_iterations/200) == 0:
                if self.i % 25 == 0:
                    fitness_values = np.asarray(
                        [individual.fitness for individual in self.population],
                        dtype=float,
                    )

                    q25, median, q75 = np.percentile(fitness_values, [25, 50, 75])

                    fitness_summary = (
                        f"Mean = {np.mean(fitness_values):.4f}, "
                        f"Best = {np.min(fitness_values):.4f}, "
                        f"Q25 = {q25:.4f}, "
                        f"Median = {median:.4f}, "
                        f"Q75 = {q75:.4f}, "
                        f"Worst = {np.max(fitness_values):.4f},"
                        # f"Best-ever fitness = {self.best_individual.fitness:.4f}"
                    )
                    # print(f'Iteration = {self.i}, Mean fitness = {np.mean([xi.fitness for xi in self.population]):.4f}, Best fitness = {self.best_individual.fitness:.4f}, Best graph fitness = {self.best_individual_by_graph.best_graph_fitness:0.4f}, Iteration time = {time.time() - start_time:.2f}')
                    if self.model is not None and 'ndp' in self.model:
                        print(
                            f"Iteration = {self.i}, "
                            f"{fitness_summary} "
                            f"Used nodes = "
                            f"{self.best_individual.best_graph_used_nodes}, "
                            f"Used edges = "
                            f"{self.best_individual.best_graph_used_edges}, "
                            f"Mutation eta = {self.mutation_eta:.2f}, "
                            f"SBX eta = {self.sbx_eta:.2f}, "
                            f"Iteration time = {time.time() - start_time:.2f}"
                        )
                    else:
                        print(
                            f"Iteration = {self.i}, "
                            f"{fitness_summary}, "
                            f"Iteration time = {time.time() - start_time:.2f}"
                        )

                if self.best_individual.fitness <= stop_criteria and not self.goal_achieved:
                    print('Target achieved!')
                    self.goal_achieved = True
                    self.goal_achieved_it = self.i
                    self.goal_achieved_individual = np.copy(self.best_individual.genotype)
                    self.goal_achieved_fitness = self.best_individual.fitness
                    if self.stop_on_target:
                        break
                # print(f'Iteration total time = {time.time() - start_time}')

        finally:
            if executor is not None:
                executor.shutdown()

        self.record[self.i + 1] = self.best_individual.fitness
        self.evaluate_final_heldout()
        return self.best_individual.genotype, self.best_individual.fitness

    def summary(self):
        print('\n------------------------------------')
        print('Evolutionary Algorithm')
        print('------------------------------------')
        print(f"Crossover probability: {self.crossover_probability}")
        print(f"Mutation probability: {self.mutation_probability}")
        print(f"Mutation eta: {self.mutation_eta}")
        print(f"SBX eta: {self.sbx_eta}")
        print('------------------------------------')
        print('\n')