import argparse
import datetime
import pickle
import sys
import time
import traceback
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiments.experiments_2 import experiment
from Tasks.acrobot import Acrobot
from Tasks.bipedalwalker import BipedalWalker
from Tasks.cartpole import CartPole
from Tasks.lunarlander import LunarLander
from Tasks.mountaincar import MountainCar
from Tasks.pendulum import Pendulum
from Tasks.position_only_cartpole import PositionOnlyCartPole
from Utilities.utilities import append_line_to_csv, create_experiment_log


TASKS = {
    "cartpole": CartPole,
    "acrobot": Acrobot,
    "mountaincar": MountainCar,
    "lunarlander": LunarLander,
    "bipedalwalker": BipedalWalker,
    "pendulum": Pendulum,
    "positiononlycartpole": PositionOnlyCartPole,
}
MODELS = ("standard_ndp", "rewiring_ndp")
HEBBIAN_FLAGS = (False, True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run resumable NDP pilot experiments.")
    parser.add_argument("--tasks", nargs="+", default=["all"], choices=["all", *TASKS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("Results/september2026_pilot"))
    return parser.parse_args()


def configure_task(task, population_size, generations, rollouts, model, hebbian):
    original_rollouts = task.n_rollouts
    if task.target is not None and original_rollouts:
        task.target = task.target * rollouts / original_rollouts

    task.n_rollouts = rollouts
    task.parameters["n_rollouts"] = rollouts
    task.parameters["population_size"] = population_size
    task.parameters["generations"] = generations
    task.parameters["stagnant_generation"] = generations
    task.parameters["model"] = model
    task.parameters["hebbian"] = hebbian
    task.parameters["add_edge_strategy"] = "all_disconnected"


def configuration_folder(output_root, task, model, hebbian):
    return output_root / task.name / f"{model}-policy_hebbian_{hebbian}"


def marker_path(folder, seed):
    return folder / f"seed_{seed}.complete"


def failure_path(folder, seed):
    return folder / f"seed_{seed}.failure.txt"


def clear_stale_failure(folder, seed):
    failure_path(folder, seed).unlink(missing_ok=True)


def run_one(task_class, model, hebbian, seed, args):
    task = task_class()
    configure_task(task, args.population_size, args.generations, args.rollouts, model, hebbian)
    folder = configuration_folder(args.output, task, model, hebbian)
    folder.mkdir(parents=True, exist_ok=True)
    marker = marker_path(folder, seed)

    label = f"{task.name} | {model} | policy_hebbian={hebbian} | seed={seed}"
    if marker.exists():
        clear_stale_failure(folder, seed)
        print(f"SKIP: {label}")
        return "skipped"

    start = time.time()
    try:
        output = experiment(task, optimisation_algorithm="EA", seed=seed)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = folder / f"output-{task.name}-EA-seed_{seed}-time_{timestamp}.pkl"
        with filename.open("wb") as file:
            pickle.dump(output, file)

        optimiser = output["optimiser"]
        log = create_experiment_log(
            str(filename), "EA", task, seed, optimiser, time.time() - start
        )
        append_line_to_csv(folder / "experiments_log.csv", log)
        marker.write_text(str(filename), encoding="utf-8")
        clear_stale_failure(folder, seed)
        print(f"PASS: {label} | {time.time() - start:.1f}s")
        return "completed"
    except Exception:
        failure = failure_path(folder, seed)
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAIL: {label} | details: {failure}")
        return "failed"


def main():
    args = parse_arguments()
    selected = list(TASKS) if "all" in args.tasks else args.tasks
    counts = {"completed": 0, "skipped": 0, "failed": 0}

    for task_name in selected:
        for model in MODELS:
            for hebbian in HEBBIAN_FLAGS:
                for seed in args.seeds:
                    status = run_one(TASKS[task_name], model, hebbian, seed, args)
                    counts[status] += 1

    print(f"Pilot summary: {counts}")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
