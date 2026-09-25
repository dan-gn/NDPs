import csv, os, platform, sys, time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from NDP.rewiring_ndp import RewiringNeuralDevelopmentalProgram
from NDP.ndp_nx import NeuralDevelopmentalProgram
from NDP.policy_network import NcHebbianLearningPolicyNetwork, PolicyNetwork
from Tasks.bipedalwalker import BipedalWalker
from Tasks.cartpole import CartPole
from Tasks.pendulum import Pendulum
from Tasks.position_only_cartpole import PositionOnlyCartPole

OUT = Path("Results/september2026_profiling/runtime_profile_v3.csv")
SEEDS, ROLLOUTS, STEP_CAP = (0, 1, 2), 3, 200
HORIZON = {"CartPole-v1": 500, "Pendulum-v1": 200,
 "popgym-PositionOnlyCartPoleEasy-v0": 200, "BipedalWalker-v3": 1600}
TASKS = (CartPole, Pendulum, PositionOnlyCartPole, BipedalWalker)


def develop(task, model, seed):
    cfg = dict(task.parameters); cfg["model"] = model
    cls = NeuralDevelopmentalProgram if model == "standard_ndp" else RewiringNeuralDevelopmentalProgram
    base = cls(cfg); n = base.get_total_number_of_mlp_parameters()
    state_n = 0
    if cfg["initial_node_state_mode"] == "coevolve":
        state_n = cfg["state_dim"] if model == "standard_ndp" else 1 + cfg["state_dim"] * cfg["n_nodes"]
        n += state_n
    vector = np.random.default_rng(seed).uniform(-1, 1, n).astype(np.float32)
    if state_n: cfg["shared_initial_node_state"] = vector[np.newaxis, :state_n]
    ndp = cls(cfg); ndp.update_mlp_weights(vector[state_n:])
    np.random.seed(seed); torch.manual_seed(seed)
    start = time.perf_counter(); graph = ndp.develope(cfg["n_cycles"])
    return graph, time.perf_counter() - start, n


def reset(policy, hebbian):
    policy.reset_activations()
    if hebbian: policy.reset_weights()


def profile(task, graph, hebbian, seed):
    cls = NcHebbianLearningPolicyNetwork if hebbian else PolicyNetwork
    policy = cls(graph, task.graph_n_inputs, task.graph_n_outputs, task.network_extra_thinking)
    # Warm-up is deliberately excluded from measurements.
    env = gym.make(task.name); obs, _ = env.reset(seed=seed)
    with torch.no_grad():
        out = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        env.step(task.compute_action(out))
    env.close(); reset(policy, hebbian)
    pt, at, et, lt, lengths = [], [], [], [], []
    for rollout in range(ROLLOUTS):
        env = gym.make(task.name); obs, _ = env.reset(seed=seed + rollout)
        reset(policy, hebbian); terminated = truncated = False; steps = 0
        with torch.no_grad():
            while not terminated and not truncated and steps < STEP_CAP:
                loop = time.perf_counter(); x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                start = time.perf_counter(); out = policy(x); pt.append(time.perf_counter()-start)
                start = time.perf_counter(); action = task.compute_action(out); at.append(time.perf_counter()-start)
                start = time.perf_counter(); obs, _, terminated, truncated, _ = env.step(action); et.append(time.perf_counter()-start)
                lt.append(time.perf_counter()-loop); steps += 1
        env.close(); lengths.append(steps)
    return {"profile_rollouts": ROLLOUTS, "mean_rollout_steps": float(np.mean(lengths)),
      "min_rollout_steps": min(lengths), "max_rollout_steps": max(lengths),
      "policy_ms_per_step": 1000*float(np.mean(pt)), "policy_p95_ms": 1000*float(np.percentile(pt,95)),
      "action_ms_per_step": 1000*float(np.mean(at)), "environment_ms_per_step": 1000*float(np.mean(et)),
      "full_loop_ms_per_step": 1000*float(np.mean(lt))}


def main():
    rows = []
    for task_cls in TASKS:
      task = task_cls(); horizon = HORIZON[task.name]
      budget = task.parameters["population_size"] * (task.parameters["generations"] + 1)
      for model in ("standard_ndp", "rewiring_ndp"):
       for seed in SEEDS:
        graph, dev, n_params = develop(task, model, seed)
        complete = graph.number_of_nodes() >= task.graph_n_inputs + task.graph_n_outputs
        for hebbian in (False, True):
          timing = profile(task, graph, hebbian, seed)
          objective = dev + task.n_repeats*task.n_rollouts*horizon*timing["full_loop_ms_per_step"]/1000
          row = {"task":task.name,"model":model,"policy_hebbian":hebbian,"seed":seed,
           "cpu":platform.processor(),"logical_cpus":os.cpu_count(),"torch_threads":torch.get_num_threads(),
           "n_parameters":n_params,"graph_nodes":graph.number_of_nodes(),"graph_edges":graph.number_of_edges(),
           "interface_complete":complete,"propagation_distance":graph.get_propagation_distance(),
           "development_seconds":dev,**timing,"episode_horizon":horizon,
           "projected_objective_seconds":objective,"configured_optimisation_evaluations":budget,
           "projected_cpu_hours_per_seed":budget*objective/3600}
          rows.append(row)
          print(f"PASS: {task.name} | {model} | hebbian={hebbian} | seed={seed} | nodes={row['graph_nodes']} | complete={complete} | dev={dev:.4f}s | loop={timing['full_loop_ms_per_step']:.4f}ms")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
      writer=csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUT}")

if __name__ == "__main__": main()
