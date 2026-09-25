"""Assemble the final NDP / 32-node R-NDP results without changing sources.

Run from the NDPs project root. The output is a self-contained folder with one
experiments_log.csv and the selected .pkl files per task/model condition.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


TASKS = (
    "CartPole-v1",
    "popgym-PositionOnlyCartPoleEasy-v0",
    "Acrobot-v1",
    "MountainCar-v0",
    "LunarLander-v3",
)
R_NDP_NAMES = {"hebbian_ndp", "rewiring_ndp"}
KEY_FIELDS = ("task", "model", "hebbian", "seed")
COMPARE_FIELDS = (
    "best_score_mean", "best_score_test", "best_graph_n_nodes",
    "best_graph_used_nodes", "best_graph_used_edges",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ndp_and_16_root", type=Path,
                        help="Original results root; provides standard NDP for every task")
    parser.add_argument("rndp32_root", type=Path,
                        help="32-node R-NDP results root; provides non-LunarLander tasks")
    parser.add_argument("--lunar-rndp-root", type=Path, required=True,
                        help="LunarLander 32-node R-NDP result directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="New, non-existing directory for the paper dataset")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalise_hebbian(value: str) -> str:
    result = str(value).strip().lower()
    if result not in {"true", "false"}:
        raise ValueError(f"Unrecognised Hebbian flag: {value!r}")
    return result.capitalize()


def read_rows(root: Path, models: set[str], tasks: set[str]) -> list[dict[str, str]]:
    paths = sorted(root.rglob("experiments_log.csv"))
    if not paths:
        raise FileNotFoundError(f"No experiments_log.csv found below {root}")
    rows: list[dict[str, str]] = []
    for log_path in paths:
        with log_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("model") not in models or row.get("task") not in tasks:
                    continue
                missing = [field for field in (*KEY_FIELDS, "filename") if not row.get(field)]
                if missing:
                    raise ValueError(f"{log_path} has a row missing {missing}")
                row = dict(row)
                row["_source_model"] = row["model"]
                if row["model"] in R_NDP_NAMES:
                    row["model"] = "rewiring_ndp"
                row["hebbian"] = normalise_hebbian(row["hebbian"])
                row["seed"] = str(int(row["seed"]))
                row["_source_log"] = str(log_path)
                row["_source_file"] = str(
                    log_path.parent / Path(row["filename"].replace("\\", "/")).name
                )
                rows.append(row)
    if not rows:
        raise ValueError(f"No selected results found below {root}")
    return rows


def deduplicate(rows: list[dict[str, str]], label: str) -> tuple[list[dict[str, str]], int]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in KEY_FIELDS)
        grouped.setdefault(key, []).append(row)
    selected = []
    for key, group in grouped.items():
        for field in COMPARE_FIELDS:
            values = {row.get(field, "") for row in group}
            if len(values) > 1:
                raise ValueError(f"Conflicting duplicate {label} result for {key}: {field}={values}")
        # Use the earliest saved output that is actually present locally.
        present = sorted(
            (row for row in group if Path(row["_source_file"]).is_file()),
            key=lambda row: row["filename"],
        )
        if not present:
            raise FileNotFoundError(
                f"No saved .pkl exists for {label} {key}; checked "
                f"{[row['_source_file'] for row in group]}"
            )
        selected.append(present[0])
    return selected, len(rows) - len(selected)


def condition_directory(row: dict[str, str]) -> Path:
    return Path(row["task"]) / f"{row['model']}-policy_hebbian_{row['hebbian']}"


def main() -> None:
    args = arguments()
    sources = (
        ("standard NDP", args.ndp_and_16_root, {"standard_ndp"}, set(TASKS)),
        ("32-node R-NDP", args.rndp32_root, R_NDP_NAMES, set(TASKS) - {"LunarLander-v3"}),
        ("LunarLander 32-node R-NDP", args.lunar_rndp_root,
         R_NDP_NAMES, {"LunarLander-v3"}),
    )
    chosen: list[dict[str, str]] = []
    duplicate_counts = {}
    for label, root, models, tasks in sources:
        rows, repeated = deduplicate(read_rows(root, models, tasks), label)
        chosen.extend(rows)
        duplicate_counts[label] = repeated

    keys = [tuple(row[field] for field in KEY_FIELDS) for row in chosen]
    if len(keys) != len(set(keys)):
        raise ValueError("The selected source folders overlap in task/model/Hebbian/seed")
    for row in chosen:
        if row["model"] in R_NDP_NAMES:
            count = row.get("best_graph_n_nodes")
            if not count or float(count) != 32:
                raise ValueError(
                    f"R-NDP is not confirmed as 32 nodes: {row['task']} seed={row['seed']} "
                    f"hebbian={row['hebbian']} best_graph_n_nodes={count!r}"
                )

    seed_sets: dict[tuple[str, str, str], set[int]] = {}
    for row in chosen:
        condition = (row["task"], row["model"], row["hebbian"])
        seed_sets.setdefault(condition, set()).add(int(row["seed"]))
    for task in TASKS:
        conditions = [
            (task, model, hebbian)
            for model in ("standard_ndp", "rewiring_ndp")
            for hebbian in ("False", "True")
        ]
        reference = seed_sets.get(conditions[0], set())
        for condition in conditions[1:]:
            if seed_sets.get(condition, set()) != reference:
                raise ValueError(
                    f"Seed sets differ for {task}: "
                    f"{[(item[1:], sorted(seed_sets.get(item, set()))) for item in conditions]}"
                )
    counts = Counter((row["task"], row["model"], row["hebbian"]) for row in chosen)
    for (task, model, hebbian), count in sorted(counts.items()):
        print(f"{task} | {model} | Hebbian={hebbian}: {count} seeds")
    print(f"Selected {len(chosen)} runs; collapsed {sum(duplicate_counts.values())} repeated log entries")
    if args.dry_run:
        print(f"DRY RUN: no files written; destination would be {args.output}")
        return
    if args.output.exists():
        raise FileExistsError(f"Destination already exists: {args.output}")

    args.output.mkdir(parents=True)
    output_rows: dict[Path, list[dict[str, str]]] = {}
    provenance = []
    for row in sorted(chosen, key=lambda r: tuple(r[field] for field in KEY_FIELDS)):
        relative_dir = condition_directory(row)
        destination_dir = args.output / relative_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        source_file = Path(row["_source_file"])
        destination_file = destination_dir / source_file.name
        if destination_file.exists():
            raise FileExistsError(f"Result filename collision: {destination_file}")
        shutil.copy2(source_file, destination_file)
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        clean["filename"] = str(destination_file)
        output_rows.setdefault(relative_dir, []).append(clean)
        provenance.append({
            "task": row["task"], "model": row["model"],
            "source_model": row["_source_model"],
            "hebbian": row["hebbian"], "seed": int(row["seed"]),
            "source_log": row["_source_log"], "source_file": str(source_file),
            "destination_file": str(destination_file),
        })

    for relative_dir, rows in output_rows.items():
        fieldnames = list(dict.fromkeys(field for row in rows for field in row))
        with (args.output / relative_dir / "experiments_log.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "description": "Standard NDP and 32-node R-NDP paper results; source data unchanged",
        "selection": "one saved run per task/model/Hebbian/seed; equivalent repeats use earliest available file",
        "source_roots": {label: str(root) for label, root, _, _ in sources},
        "collapsed_duplicate_rows": duplicate_counts,
        "run_count": len(chosen),
        "runs": provenance,
    }
    with (args.output / "paper_results_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote {len(chosen)} runs to {args.output}")


if __name__ == "__main__":
    main()
