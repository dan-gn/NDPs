import datetime
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from NDP.rewiring_ndp import RewiringNeuralDevelopmentalProgram
from NDP.ndp_nx import NeuralDevelopmentalProgram
from Optimisation.ea import EvolutionaryAlgorithm
from Tasks.acrobot import Acrobot
from Tasks.bipedalwalker import BipedalWalker
from Tasks.cartpole import CartPole
from Tasks.lunarlander import LunarLander
from Tasks.mountaincar import MountainCar
from Tasks.pendulum import Pendulum
from Tasks.position_only_cartpole import PositionOnlyCartPole
from Utilities.utilities import append_line_to_csv, create_experiment_log


OUTPUT_ROOT = Path("Results/september2026_smoke_test")
SEED = 0

TASK_CLASSES = [
    CartPole,
    Acrobot,
    MountainCar,
    LunarLander,
    BipedalWalker,
    Pendulum,
    PositionOnlyCartPole,
]

MODEL_CONFIGURATIONS = [
    ("standard_ndp", False),
    ("standard_ndp", True),
    ("rewiring_ndp", False),
    ("rewiring_ndp", True),
]


def configure_for_smoke_test(task):
    # These changes apply only to this temporary task instance.
    task.parameters["n_rollouts"] = 1
    task.parameters["n_repeats"] = 1
    task.parameters["population_size"] = 2
    task.parameters["generations"] = 1
    task.parameters["stagnant_generation"] = 1

    task.n_rollouts = 1
    task.n_repeats = 1

    # Always complete the one requested generation.
    task.target = -np.inf


def get_number_of_variables(task):
    if task.parameters["model"] == "standard_ndp":
        ndp = NeuralDevelopmentalProgram(task.parameters)
    elif task.parameters["model"] in ("rewiring_ndp", "hebbian_ndp"):
        ndp = RewiringNeuralDevelopmentalProgram(task.parameters)
    else:
        raise ValueError("Unknown NDP model.")

    n_variables = ndp.get_total_number_of_mlp_parameters()

    if task.parameters["initial_node_state_mode"] == "coevolve":
        if task.parameters["model"] in ("rewiring_ndp", "hebbian_ndp"):
            n_variables += 1 + (
                task.parameters["state_dim"] * task.parameters["n_nodes"]
            )
        else:
            n_variables += task.parameters["state_dim"]

    return n_variables


def run_configuration(task, model, hebbian):
    task.parameters["model"] = model
    task.parameters["hebbian"] = hebbian

    n_variables = get_number_of_variables(task)
    optimiser = EvolutionaryAlgorithm(
        n_variables=n_variables,
        population_size=task.parameters["population_size"],
        max_iterations=task.parameters["generations"],
        max_stagnment=task.parameters["stagnant_generation"],
        objective_function=task.evaluate_ndp,
        run_in_parallel=False,
        cores=1,
    )

    start_time = time.time()
    optimiser.run(task.target, SEED)
    elapsed_time = time.time() - start_time

    configuration_name = f"{model}-policy_hebbian_{hebbian}"
    output_folder = OUTPUT_ROOT / task.name / configuration_name
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = output_folder / (
        f"output-{task.name}-EA-seed_{SEED}-time_{timestamp}.pkl"
    )

    output = {
        "task": task,
        "optimisation_algorithm": "EA",
        "seed": SEED,
        "optimiser": optimiser,
    }
    with output_filename.open("wb") as file:
        pickle.dump(output, file)

    log = create_experiment_log(
        str(output_filename),
        "EA",
        task,
        SEED,
        optimiser,
        elapsed_time=elapsed_time,
    )
    append_line_to_csv(output_folder / "experiments_log.csv", log)

    assert output_filename.exists()
    assert np.isfinite(optimiser.best_individual.fitness)
    assert np.isfinite(optimiser.best_individual.fitness_test)
    assert optimiser.best_individual.best_graph is not None
    assert optimiser.best_individual_by_graph.best_graph is not None

    print(
        f"PASS: {task.name} | {model} | "
        f"policy_hebbian={hebbian} | {elapsed_time:.2f}s"
    )


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    completed = 0
    failures = []

    for task_class in TASK_CLASSES:
        for model, hebbian in MODEL_CONFIGURATIONS:
            task = task_class()
            configure_for_smoke_test(task)

            try:
                run_configuration(task, model, hebbian)
                completed += 1
            except Exception as error:
                failures.append((task.name, model, hebbian, error))
                print(
                    f"FAIL: {task.name} | {model} | "
                    f"policy_hebbian={hebbian}: {error!r}"
                )

    print(f"\nCompleted {completed}/{len(TASK_CLASSES) * len(MODEL_CONFIGURATIONS)} configurations.")

    if failures:
        print("\nFailures:")
        for task_name, model, hebbian, error in failures:
            print(f"- {task_name} | {model} | policy_hebbian={hebbian}: {error!r}")
        raise AssertionError(f"{len(failures)} smoke-test configurations failed.")

    print("All end-to-end smoke experiments passed.")


if __name__ == "__main__":
    main()
