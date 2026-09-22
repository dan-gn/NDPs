import csv
import statistics
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from NDP.rewiring_ndp import RewiringNeuralDevelopmentalProgram
from NDP.ndp_nx import NeuralDevelopmentalProgram
from NDP.policy_network import NcHebbianLearningPolicyNetwork, PolicyNetwork
from Tasks.bipedalwalker import BipedalWalker
from Tasks.cartpole import CartPole
from Tasks.pendulum import Pendulum
from Tasks.position_only_cartpole import PositionOnlyCartPole


OUTPUT_FILE = Path("Results/september2026_profiling/runtime_profile.csv")
MAX_ROLLOUT_STEPS = 200
SEED = 0

TASK_CLASSES = [
    CartPole,
    Pendulum,
    PositionOnlyCartPole,
    BipedalWalker,
]
MODELS = ["standard_ndp", "rewiring_ndp"]
POLICY_HEBBIAN_FLAGS = [False, True]


def percentile(values, percentile_value):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), percentile_value))


def build_ndp_and_develop(task, model, seed):
    config = dict(task.parameters)
    config["model"] = model

    if model == "standard_ndp":
        ndp = NeuralDevelopmentalProgram(config)
    elif model in ("rewiring_ndp", "hebbian_ndp"):
        ndp = RewiringNeuralDevelopmentalProgram(config)
    else:
        raise ValueError(f"Unknown model: {model}")

    n_parameters = ndp.get_total_number_of_mlp_parameters()
    if config["initial_node_state_mode"] == "coevolve":
        if model in ("rewiring_ndp", "hebbian_ndp"):
            n_parameters += 1 + config["state_dim"] * config["n_nodes"]
        else:
            n_parameters += config["state_dim"]

    rng = np.random.default_rng(seed)
    vector = rng.uniform(-1.0, 1.0, n_parameters).astype(np.float32)
    vector = np.clip(vector, -1.0, 1.0)

    if config["initial_node_state_mode"] == "coevolve":
        if model in ("rewiring_ndp", "hebbian_ndp"):
            split_index = 1 + config["state_dim"] * config["n_nodes"]
        else:
            split_index = config["state_dim"]
        config["shared_initial_node_state"] = vector[np.newaxis, :split_index]
        weights = vector[split_index:]
    else:
        weights = vector

    if model == "standard_ndp":
        ndp = NeuralDevelopmentalProgram(config)
    else:
        ndp = RewiringNeuralDevelopmentalProgram(config)
    ndp.update_mlp_weights(weights)

    np.random.seed(seed)
    torch.manual_seed(seed)
    start = time.perf_counter()
    graph = ndp.develope(config["n_cycles"])
    development_seconds = time.perf_counter() - start

    return graph, development_seconds, n_parameters


def make_environment(task):
    return gym.make(task.name)


def profile_policy(task, graph, hebbian, seed):
    if hebbian:
        policy = NcHebbianLearningPolicyNetwork(
            graph,
            task.graph_n_inputs,
            task.graph_n_outputs,
            task.network_extra_thinking,
        )
    else:
        policy = PolicyNetwork(
            graph,
            task.graph_n_inputs,
            task.graph_n_outputs,
            task.network_extra_thinking,
        )

    environment = make_environment(task)
    observation, _ = environment.reset(seed=seed)
    policy.reset_activations()
    if hebbian:
        policy.reset_weights()

    policy_times = []
    action_times = []
    environment_times = []
    cumulative_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    with torch.no_grad():
        while not terminated and not truncated and steps < MAX_ROLLOUT_STEPS:
            observation_tensor = torch.tensor(
                observation,
                dtype=torch.float32,
            ).unsqueeze(0)

            start = time.perf_counter()
            output = policy(observation_tensor)
            policy_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            action = task.compute_action(output)
            action_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            observation, reward, terminated, truncated, _ = environment.step(action)
            environment_times.append(time.perf_counter() - start)

            cumulative_reward += float(reward)
            steps += 1

    environment.close()

    return {
        "policy_seconds": sum(policy_times),
        "policy_ms_per_step": 1000.0 * statistics.mean(policy_times),
        "policy_p95_ms": 1000.0 * percentile(policy_times, 95),
        "action_ms_per_step": 1000.0 * statistics.mean(action_times),
        "environment_seconds": sum(environment_times),
        "environment_ms_per_step": 1000.0 * statistics.mean(environment_times),
        "rollout_seconds": sum(policy_times) + sum(action_times) + sum(environment_times),
        "rollout_steps": steps,
        "cumulative_reward": cumulative_reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def main():
    rows = []

    for task_class in TASK_CLASSES:
        task = task_class()

        for model in MODELS:
            graph, development_seconds, n_parameters = build_ndp_and_develop(
                task,
                model,
                SEED,
            )

            for policy_hebbian in POLICY_HEBBIAN_FLAGS:
                timing = profile_policy(task, graph, policy_hebbian, SEED)
                row = {
                    "task": task.name,
                    "model": model,
                    "policy_hebbian": policy_hebbian,
                    "seed": SEED,
                    "n_parameters": n_parameters,
                    "n_cycles": task.n_cycles,
                    "graph_nodes": graph.number_of_nodes(),
                    "graph_edges": graph.number_of_edges(),
                    "propagation_distance": graph.get_propagation_distance(),
                    "development_seconds": development_seconds,
                    **timing,
                }
                rows.append(row)
                print(
                    f"PASS: {task.name} | {model} | "
                    f"policy_hebbian={policy_hebbian} | "
                    f"development={development_seconds:.4f}s | "
                    f"policy={timing['policy_ms_per_step']:.4f}ms/step | "
                    f"environment={timing['environment_ms_per_step']:.4f}ms/step"
                )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} profiling rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
