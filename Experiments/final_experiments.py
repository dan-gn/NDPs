import argparse
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
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
from Utilities.utilities import (
    append_line_to_csv,
    create_experiment_log,
    is_running_in_colab,
)


TASKS = {
    "cartpole": CartPole,
    "acrobot": Acrobot,
    "mountaincar": MountainCar,
    "pendulum": Pendulum,
    "positiononlycartpole": PositionOnlyCartPole,
    "lunarlander": LunarLander,
    "bipedalwalker": BipedalWalker,
}

MODELS = ("standard_ndp", "hebbian_ndp")
POLICY_HEBBIAN_OPTIONS = (False, True)
INITIAL_OPTIMIZER_SEEDS = tuple(range(10))
FINAL_OPTIMIZER_SEEDS = tuple(range(30))
COLAB_MOUNT_ROOT = Path("/content/drive/MyDrive")
COLAB_RESULTS_ROOT = COLAB_MOUNT_ROOT / "ICLR"


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def resolve_output_root(output_root):
    output_root = Path(output_root)
    if not is_running_in_colab():
        return output_root

    if output_root.is_absolute():
        try:
            output_root.relative_to(COLAB_RESULTS_ROOT)
        except ValueError as error:
            raise ValueError(
                "On Colab, an absolute --output path must be inside "
                f"{COLAB_RESULTS_ROOT}."
            ) from error
        return output_root

    return COLAB_RESULTS_ROOT / output_root


def resolved_manifest():
    task_settings = {}
    for task_name in TASKS:
        task = TASKS[task_name]()
        task_settings[task.name] = dict(task.parameters)

    return {
        "algorithm": "EA",
        "stop_on_target": False,
        "optimizer_seeds": list(FINAL_OPTIMIZER_SEEDS),
        "models": list(MODELS),
        "policy_hebbian_options": list(POLICY_HEBBIAN_OPTIONS),
        "tasks": task_settings,
    }


def write_or_validate_manifest(output_root, manifest):
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "experiment_manifest.json"
    serialised = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        default=json_default,
    )
    resolved = json.loads(serialised)

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != resolved:
            raise ValueError(
                f"Configuration differs from existing manifest: {manifest_path}. "
                "Use a different --output folder for a different experiment."
            )
        print(f"Validated existing manifest: {manifest_path}")
    else:
        manifest_path.write_text(serialised + "\n", encoding="utf-8")
        print(f"Wrote experiment manifest: {manifest_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run resumable final NDP experiments.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", *TASKS],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(INITIAL_OPTIMIZER_SEEDS)
    )
    parser.add_argument("--cores", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Results/september2026_final"),
    )
    return parser.parse_args()


def configure_task(task, model, policy_hebbian):
    task.parameters["model"] = model
    task.parameters["hebbian"] = policy_hebbian
    task.parameters["add_edge_strategy"] = "all_disconnected"


def condition_folder(output_root, task, model, policy_hebbian):
    return output_root / task.name / f"{model}-policy_hebbian_{policy_hebbian}"


def completion_path(folder, seed):
    return folder / f"seed_{seed}.complete"


def failure_path(folder, seed):
    return folder / f"seed_{seed}.failure.txt"


def run_one(task_class, model, policy_hebbian, seed, output_root):
    task = task_class()
    configure_task(task, model, policy_hebbian)

    folder = condition_folder(output_root, task, model, policy_hebbian)
    folder.mkdir(parents=True, exist_ok=True)
    completion = completion_path(folder, seed)
    failure = failure_path(folder, seed)

    label = (
        f"{task.name} | {model} | policy_hebbian={policy_hebbian} | seed={seed}"
    )

    if completion.exists():
        failure.unlink(missing_ok=True)
        print(f"SKIP: {label}")
        return "skipped"

    start = time.time()
    try:
        output = experiment(task, optimisation_algorithm="EA", seed=seed, stop_on_target=False)
        optimiser = output["optimiser"]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = folder / (
            f"output-{task.name}-EA-seed_{seed}-time_{timestamp}.pkl"
        )

        with output_file.open("wb") as file:
            pickle.dump(output, file)

        log = create_experiment_log(
            str(output_file),
            "EA",
            task,
            seed,
            optimiser,
            elapsed_time=time.time() - start,
        )
        append_line_to_csv(folder / "experiments_log.csv", log)

        completion.write_text(str(output_file) + "\n", encoding="utf-8")
        failure.unlink(missing_ok=True)
        print(f"PASS: {label} | output={output_file}")
        return "completed"

    except Exception:
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAIL: {label} | details={failure}")
        return "failed"


def main():
    args = parse_arguments()
    if args.cores < 1:
        raise ValueError("--cores must be at least 1.")
    if not args.seeds:
        raise ValueError("At least one optimizer seed is required.")
    invalid_seeds = sorted(set(args.seeds) - set(FINAL_OPTIMIZER_SEEDS))
    if invalid_seeds:
        raise ValueError(
            f"Seeds {invalid_seeds} are outside the final study seeds "
            f"{list(FINAL_OPTIMIZER_SEEDS)}."
        )

    colab = is_running_in_colab()
    args.output = resolve_output_root(args.output)
    if colab and not args.dry_run and not COLAB_MOUNT_ROOT.exists():
        raise RuntimeError(
            f"Google Drive is not mounted at {COLAB_MOUNT_ROOT}. "
            "Mount Drive before starting the experiments."
        )

    os.environ["NDP_MAX_CORES"] = str(args.cores)
    selected_tasks = list(TASKS) if "all" in args.tasks else args.tasks

    if args.dry_run:
        for task_name in selected_tasks:
            task = TASKS[task_name]()
            print(
                f"{task.name}: population={task.parameters['population_size']}, "
                f"generations={task.parameters['generations']}, "
                f"rollouts={task.parameters['n_rollouts']}, "
                f"repeats={task.parameters['n_repeats']}"
            )
        planned_runs = len(selected_tasks) * len(MODELS) * len(
            POLICY_HEBBIAN_OPTIONS
        ) * len(args.seeds)
        final_study_runs = (
            len(TASKS)
            * len(MODELS)
            * len(POLICY_HEBBIAN_OPTIONS)
            * len(FINAL_OPTIMIZER_SEEDS)
        )
        print(
            f"DRY RUN: current batch={planned_runs} EA runs, seeds={args.seeds}, "
            f"cores={args.cores}, output={args.output}"
        )
        print(f"FINAL STUDY: {final_study_runs} EA runs, seeds=0-29")
        return

    manifest = resolved_manifest()
    write_or_validate_manifest(args.output, manifest)

    counts = {"completed": 0, "skipped": 0, "failed": 0}

    for task_name in selected_tasks:
        for model in MODELS:
            for policy_hebbian in POLICY_HEBBIAN_OPTIONS:
                for seed in args.seeds:
                    status = run_one(
                        TASKS[task_name],
                        model,
                        policy_hebbian,
                        seed,
                        args.output,
                    )
                    counts[status] += 1

    print(
        "Final experiment summary: "
        f"completed={counts['completed']}, "
        f"skipped={counts['skipped']}, "
        f"failed={counts['failed']}"
    )
    if counts["failed"]:
        raise AssertionError(f"{counts['failed']} final experiment runs failed.")


if __name__ == "__main__":
    main()
