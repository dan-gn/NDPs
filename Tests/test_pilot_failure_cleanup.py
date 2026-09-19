import sys
from pathlib import Path
from tempfile import TemporaryDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if not (REPOSITORY_ROOT / "Experiments").exists():
    REPOSITORY_ROOT = REPOSITORY_ROOT / "NDPs"
sys.path.insert(0, str(REPOSITORY_ROOT))

from Experiments.pilot_experiments import clear_stale_failure, failure_path, marker_path


def test_failure_and_completion_paths_are_seed_specific():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        assert failure_path(folder, 2) == folder / "seed_2.failure.txt"
        assert marker_path(folder, 2) == folder / "seed_2.complete"


def test_success_cleanup_removes_only_matching_failure():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        matching_failure = failure_path(folder, 2)
        other_failure = failure_path(folder, 3)
        matching_failure.write_text("old failure", encoding="utf-8")
        other_failure.write_text("current failure", encoding="utf-8")

        clear_stale_failure(folder, 2)

        assert not matching_failure.exists()
        assert other_failure.exists()


def main():
    tests = [
        test_failure_and_completion_paths_are_seed_specific,
        test_success_cleanup_removes_only_matching_failure,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("All pilot failure-cleanup tests passed.")


if __name__ == "__main__":
    main()
