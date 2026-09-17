import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import Tasks.task as task_module
from Optimisation.ea import EvolutionaryAlgorithm, TEST_SEED
from Tasks.task import Task


class RecordingEnvironment:
    def __init__(self):
        self.reset_seeds = []
        self.observation_space = task_module.gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 1.0, True, False, {}

    def close(self):
        pass


class StubPolicy:
    def __init__(self, graph, n_inputs, n_outputs, network_extra_thinking):
        self.n_outputs = n_outputs

    def reset_activations(self):
        pass

    def reset_weights(self):
        pass

    def __call__(self, observation):
        return torch.zeros((1, self.n_outputs), dtype=torch.float32)


class StubGraph:
    def get_unreachable_outputs(self, n_inputs, n_outputs):
        return []


def make_task(n_rollouts=10):
    task = object.__new__(Task)
    task.name = "RecordingEnvironment-v0"
    task.parameters = {"normalize_observations": False}
    task.graph_n_inputs = 1
    task.graph_n_outputs = 1
    task.network_extra_thinking = 0
    task.n_rollouts = n_rollouts
    task.truncated_penalty = 0
    task.action_space_type = "discrete"
    return task


def recorded_rollout_seeds(monkey_env, hebbian, env_seed):
    original_make = task_module.gym.make
    original_standard = task_module.PolicyNetwork
    original_hebbian = task_module.NcHebbianLearningPolicyNetwork

    try:
        task_module.gym.make = lambda *args, **kwargs: monkey_env
        task_module.PolicyNetwork = StubPolicy
        task_module.NcHebbianLearningPolicyNetwork = StubPolicy

        make_task().evaluate_graph(
            graph=StubGraph(),
            env_seed=env_seed,
            hebbian=hebbian,
        )
    finally:
        task_module.gym.make = original_make
        task_module.PolicyNetwork = original_standard
        task_module.NcHebbianLearningPolicyNetwork = original_hebbian

    return monkey_env.reset_seeds


def test_training_rollout_seeds():
    environment = RecordingEnvironment()
    seeds = recorded_rollout_seeds(environment, hebbian=False, env_seed=0)
    assert seeds == list(range(10))


def test_testing_rollout_seeds():
    environment = RecordingEnvironment()
    seeds = recorded_rollout_seeds(
        environment,
        hebbian=False,
        env_seed=TEST_SEED,
    )
    assert seeds == list(range(TEST_SEED, TEST_SEED + 10))


def test_standard_and_hebbian_use_identical_seeds():
    standard_environment = RecordingEnvironment()
    hebbian_environment = RecordingEnvironment()

    standard_seeds = recorded_rollout_seeds(
        standard_environment,
        hebbian=False,
        env_seed=0,
    )
    hebbian_seeds = recorded_rollout_seeds(
        hebbian_environment,
        hebbian=True,
        env_seed=0,
    )

    assert standard_seeds == hebbian_seeds == list(range(10))


def test_training_and_testing_sets_do_not_overlap():
    training_seeds = set(range(10))
    testing_seeds = set(range(TEST_SEED, TEST_SEED + 10))
    assert training_seeds.isdisjoint(testing_seeds)


def test_ea_requests_the_held_out_seed():
    requested_seeds = []

    def recording_objective(genotype, env_seed=0):
        requested_seeds.append(env_seed)
        fitness = float(np.sum(genotype**2))
        graph = object()
        return fitness, [fitness], graph, fitness

    optimiser = EvolutionaryAlgorithm(
        n_variables=2,
        objective_function=recording_objective,
        population_size=1,
        max_iterations=1,
        max_stagnment=1,
        run_in_parallel=False,
    )
    optimiser.init_optimisation_variables()
    optimiser.initialise_population()

    assert requested_seeds == [0]
    optimiser.evaluate_final_heldout()
    assert requested_seeds == [0, TEST_SEED]


def main():
    tests = [
        test_training_rollout_seeds,
        test_testing_rollout_seeds,
        test_standard_and_hebbian_use_identical_seeds,
        test_training_and_testing_sets_do_not_overlap,
        test_ea_requests_the_held_out_seed,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All seed-separation tests passed.")


if __name__ == "__main__":
    main()
