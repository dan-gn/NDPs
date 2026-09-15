import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Optimisation.ea import EvolutionaryAlgorithm


EVALUATION_DELAY_SECONDS = 0.005


def objective(genotype, env_seed=None):
    # A short deterministic evaluation makes process-pool startup visible without
    # requiring a full environment experiment.
    time.sleep(EVALUATION_DELAY_SECONDS)
    fitness = float(np.sum(np.square(genotype)))
    return fitness, None, None, fitness


def make_optimizer(population_size, generations, cores):
    return EvolutionaryAlgorithm(
        n_variables=16,
        objective_function=objective,
        population_size=population_size,
        max_iterations=generations,
        max_stagnment=generations + 1,
        run_in_parallel=True,
        cores=cores,
    )


def prepare(optimizer, seed):
    optimizer.init_optimisation_variables()
    optimizer.set_seed(seed)
    optimizer.n_core_seed = np.random.randint(1, 2**14)


def run_with_persistent_pool(optimizer, seed):
    prepare(optimizer, seed)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=optimizer.cores) as executor:
        optimizer.population = optimizer.parallel_initialise_population(executor)
        for optimizer.i in range(optimizer.max_iterations):
            optimizer.update_population(executor)
    return time.perf_counter() - start


def run_with_recreated_pools(optimizer, seed):
    prepare(optimizer, seed)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=optimizer.cores) as executor:
        optimizer.population = optimizer.parallel_initialise_population(executor)
    for optimizer.i in range(optimizer.max_iterations):
        with ProcessPoolExecutor(max_workers=optimizer.cores) as executor:
            optimizer.update_population(executor)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--cores", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.population % 2 != 0:
        raise ValueError("Population size must be even.")

    legacy = make_optimizer(args.population, args.generations, args.cores)
    persistent = make_optimizer(args.population, args.generations, args.cores)

    legacy_seconds = run_with_recreated_pools(legacy, args.seed)
    persistent_seconds = run_with_persistent_pool(persistent, args.seed)

    np.testing.assert_allclose(
        persistent.best_individual.genotype,
        legacy.best_individual.genotype,
    )
    assert persistent.best_individual.fitness == legacy.best_individual.fitness

    saved_seconds = legacy_seconds - persistent_seconds
    speedup = legacy_seconds / persistent_seconds
    reduction = 100.0 * saved_seconds / legacy_seconds

    print(f"Legacy recreated pools: {legacy_seconds:.3f} s")
    print(f"Persistent pool:        {persistent_seconds:.3f} s")
    print(f"Time saved:             {saved_seconds:.3f} s ({reduction:.1f}%)")
    print(f"Speedup:                {speedup:.2f}x")
    print("PASS: both pool strategies produced the same best individual")


if __name__ == "__main__":
    main()
