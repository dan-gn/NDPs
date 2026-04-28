import numpy as np
import random
import torch
import time
import torch.nn.functional as F

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
OBJECTIVE FUNCTION
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Compute action from environment
def compute_action(env_name, action):
    if env_name == 'CartPole-v1':
        action =  torch.sigmoid(action)
        return int(torch.round(action))
    elif env_name in ['MountainCar-v0', 'Acrobot-v1']:
        # return int(nn.functional.hardtanh(action, 0, 2))
        probs = F.softmax(action, dim=0)  
        return int(probs.argmax())
    elif env_name == 'LunarLander-v3':
        # return int(nn.functional.hardtanh(action, 0, 3))
        probs = F.softmax(action, dim=0)  
        return int(probs.argmax())   


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
EVOLUTIONARY ALGORITHM
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
import math
from concurrent.futures import ProcessPoolExecutor

class Individual:

    def __init__(self, n_variables, genotype = None, fitness = None):
        self.n_variables = n_variables
        self.genotype = genotype
        self.fitness = fitness
        self.fitness_test = None
        self.initial_value_range = 5 # +- initial_value_range

    def random_initialise(self):
        self.genotype = np.random.uniform(-self.initial_value_range, self.initial_value_range, self.n_variables)

class EvolutionaryAlgorithm:

    def __init__(self, n_variables, max_iterations, population_size, max_stagnment, model_name, environment_name, tries, lambda_value, objective_function, run_in_parallel = True, cores = 4):
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.n_variables = n_variables
        # self.mutation_probability = 1 / n_variables
        self.mutation_probability = 0.01
        self.mutation_eta = 5
        self.sbx_eta = 5
        self.elitism_proportion = 0.1
        self.elitism_index = int(self.elitism_proportion * self.population_size)
        self.init_best_individual()
        self.max_stagnment = max_stagnment
        self.model_name = model_name
        self.environment_name = environment_name
        self.tries = tries
        self.lambda_value = lambda_value
        self.run_in_parallel = run_in_parallel
        self.objective_function = objective_function
        self.cores = cores

    def init_best_individual(self):
        self.best_individual = Individual(self.n_variables)
        self.best_individual.fitness = float("inf")

    def initialise_population(self):
        population = [Individual(self.n_variables) for _ in range(self.population_size)]
        for i, member in enumerate(population):
            population[i].random_initialise()
            population[i].fitness = self.objective_function(population[i].genotype)
            if population[i].fitness < self.best_individual.fitness:
                self.best_individual.genotype = population[i].genotype
                self.best_individual.fitness = population[i].fitness
                self.best_individual.fitness_test = self.objective_function(self.best_individual.genotype) 
        return population
    
    def run_initialise_individual(self, core_seed):
        print(f'Core seed:{core_seed - self.n_core_seed}')
        self.set_seed(core_seed)
        individual = Individual(self.n_variables)
        individual.random_initialise()
        individual.fitness = self.objective_function(individual.genotype)
        return individual
    
    def parallel_initialise_population(self):
        with ProcessPoolExecutor(max_workers=self.cores) as executor:
            population = list(executor.map(self.run_initialise_individual, range(self.n_core_seed, self.n_core_seed + self.population_size)))
        self.n_core_seed += self.population_size
        population = sorted(population, key=lambda x: x.fitness)
        if population[0].fitness < self.best_individual.fitness:
            self.best_individual.genotype = population[0].genotype
            self.best_individual.fitness = population[0].fitness
            self.best_individual.fitness_test = self.objective_function(self.best_individual.genotype) 
        return population

    def roulette_wheel(self, p):
        r = np.random.uniform(0, 1) * sum(p)	
        q = np.cumsum(p)
        return next(idx for idx, value in enumerate(q) if value >= r)

    def tournament_selection(self, n_competitors=2):
        all_indexes = list(range(self.population_size))
        parents = []
        for i in range(2):
            draw = np.random.permutation(all_indexes)
            competitors = draw[:n_competitors]
            winner = self.roulette_wheel(self.probs[competitors])
            parents.append(competitors[winner])
            all_indexes.remove(parents[i])
        return [self.population[parents[0]], self.population[parents[1]]]

    # Get parent selection probabilities
    def compute_parent_selection_prob(self, beta=1):
        # Get an array of all cost of current population, add acceptance criteria value
        # and divide by the mean of the array to avoid overflow while computing exponential
        fitness = np.array([member.fitness for member in self.population]) 
        mean_fitness = abs(np.mean(fitness))
        if mean_fitness != 0 and mean_fitness != math.inf:
            fitness /= mean_fitness
        return np.exp(-beta * fitness)
    
    def parent_selection(self):
        self.probs = self.compute_parent_selection_prob()
        parents = [self.tournament_selection() for _ in range(int(self.population_size/2))]
        # return np.array(parents).reshape(-1, 2).tolist()
        return parents
    
    def sbx(self, parents):
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

    def polynomial_muatation(self, x):
        r = np.random.uniform(0, 1)
        if r < 0.5:
            delta = (2*r) ** (1 / (self.mutation_eta+1)) - 1
        else:
            delta = 1 - (2 * (1-r)) ** (1 / (self.mutation_eta+1))
        return x + delta

    def elitism(self, offspring):
        self.population = sorted(self.population, key=lambda x: x.fitness)
        offspring = sorted(offspring, key=lambda x: x.fitness)
        self.population[self.elitism_index:] = offspring[:-self.elitism_index]

        if offspring[0].fitness < self.best_individual.fitness:
            self.best_individual.genotype = offspring[0].genotype
            self.best_individual.fitness = offspring[0].fitness
            self.best_individual.fitness_test = self.objective_function(self.best_individual.genotype) 
            self.stagnment_iterations = -1
        self.stagnment_iterations += 1

    def mutate(self, genotype):
        genotype = np.array(genotype)  # ensure it's a NumPy array
        random_values = np.random.uniform(0, 1, size=genotype.shape)
        mutation_mask = random_values <= self.mutation_probability
        for idx in np.where(mutation_mask)[0]:
            genotype[idx] = self.polynomial_muatation(genotype[idx])
        return genotype

    def crossover_and_mutation(self, parents):
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
            fitness_g1 = self.objective_function(mutated_g1)
            offspring.append(Individual(self.n_variables, genotype=mutated_g1, fitness=fitness_g1))
            mutated_g2 = self.mutate(genotype2)
            fitness_g2 = self.objective_function(mutated_g2)
            offspring.append(Individual(self.n_variables, genotype=mutated_g2, fitness=fitness_g2))
        return offspring
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


    # This function is for running the evolutionary algorithm in parallel. 
    # This function creates a couple of offsprings by performing parent seletction, crossover, mutation and evaluation. 
    # When running in parallel, each core starts its own random generator, so I included the input "core_seed", so each time the iteration ensure a different random process.
    def run_single_crossover_and_mutation(self, core_seed):
        self.set_seed(core_seed)

        # Parent Selection
        # parents = self.parents[np.mod(core_seed, self.n_core_seed)]
        parents = self.tournament_selection() 

        # Crossover
        genotype1, genotype2 = self.sbx(parents)

        # Mutation
        mutated_g1 = self.mutate(genotype1)
        
        # Evaluation
        fitness_g1 = self.objective_function(mutated_g1)
        mutated_g2 = self.mutate(genotype2)
        fitness_g2 = self.objective_function(mutated_g2)
        offspring1 = Individual(self.n_variables, genotype=mutated_g1, fitness=fitness_g1)
        offspring2 = Individual(self.n_variables, genotype=mutated_g2, fitness=fitness_g2)
        return [offspring1, offspring2]


    def parallel_crossover_and_mutation(self):
        self.probs = self.compute_parent_selection_prob()
        n_couples = int(self.population_size/2)
        with ProcessPoolExecutor(max_workers=self.cores) as executor:
            offspring = list(executor.map(self.run_single_crossover_and_mutation, range(self.n_core_seed, self.n_core_seed + n_couples)))
        self.n_core_seed += n_couples
        offspring = np.array(offspring).flatten().tolist()

        return offspring

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

    def run(self, stop_criteria, seed, env_initial_seed):
        self.goal_achieved = False
        self.goal_achieved_it = None
        self.goal_achieved_individual = None
        self.goal_achieved_fitness = None
        self.record = np.zeros(self.max_iterations + 1)
        self.seed = seed
        self.env_initial_seed = env_initial_seed
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
            if self.i % 1 == 0:
                print(f'Iteration = {self.i}, Mean fitness = {np.mean([xi.fitness for xi in self.population])}, Best fitness = {self.best_individual.fitness}, Best fitness testing = {self.best_individual.fitness_test}, Iteration time = {time.time() - start_time:.2f}')
            if self.best_individual.fitness <= stop_criteria and not self.goal_achieved:
                print('Stop criteria achieved!')
                self.goal_achieved = True
                self.goal_achieved_it = self.i
                self.goal_achieved_individual = np.copy(self.best_individual.genotype)
                self.goal_achieved_fitness = self.best_individual.fitness
                # break
            # print(f'Iteration total time = {time.time() - start_time}')
        self.record[self.i + 1] = self.best_individual.fitness
        return self.best_individual.genotype, self.best_individual.fitness