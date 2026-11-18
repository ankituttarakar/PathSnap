"""Lightweight environment verification for PathSnap.

Checks — without modifying anything — that:
    1. The Python version is adequate.
    2. All required third-party packages import, and their versions are shown.
    3. The ``pathsnap`` package imports and config paths resolve.
    4. The raw dataset is present with the expected trip counts (80 / 10).

Run from the repository root:

    python scripts/verify_env.py

Exit code 0 = all good; 1 = at least one check failed. No packages are
installed and no data is written.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("networkx", "networkx"),
    ("tqdm", "tqdm"),
    ("folium", "folium"),
    ("pyproj", "pyproj"),
    ("shapely", "shapely"),
    ("geopandas", "geopandas"),
    ("haversine", "haversine"),
    ("joblib", "joblib"),
    ("optuna", "optuna"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
]

MIN_PYTHON = (3, 10)
EXPECTED_TRIP_COUNTS = {"train": 80, "student_test": 10}


def _ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_python() -> bool:
    print("Python version")
    ok = sys.version_info >= MIN_PYTHON
    (_ok if ok else _fail)(
        f"{sys.version.split()[0]} (need >= {'.'.join(map(str, MIN_PYTHON))})"
    )
    return ok


def check_packages() -> bool:
    print("Required packages")
    all_ok = True
    for import_name, dist_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            try:
                version = metadata.version(dist_name)
            except metadata.PackageNotFoundError:
                version = "?"
            _ok(f"{dist_name:14s} {version}")
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            all_ok = False
            _fail(f"{dist_name:14s} import failed: {exc}")
    return all_ok


def check_package_and_paths() -> bool:
    print("pathsnap package + config")
    try:
        from pathsnap.config import CONFIG

        _ok(f"import pathsnap (root: {CONFIG.paths.root})")
        _ok(f"seed = {CONFIG.seed}, cv_folds = {CONFIG.cv_folds}")
        return True
    except Exception as exc:  # noqa: BLE001
        _fail(f"could not import pathsnap.config: {exc}")
        return False


def check_dataset() -> bool:
    print("Raw dataset presence")
    try:
        from pathsnap.config import CONFIG
    except Exception as exc:  # noqa: BLE001
        _fail(f"cannot check dataset (config import failed): {exc}")
        return False

    all_ok = True
    for partition, expected in EXPECTED_TRIP_COUNTS.items():
        pdir = CONFIG.paths.partition_dir(partition)
        if not pdir.exists():
            all_ok = False
            _fail(f"{partition}: directory missing ({pdir})")
            continue
        n = sum(1 for d in pdir.iterdir() if d.is_dir())
        ok = n == expected
        all_ok = all_ok and ok
        (_ok if ok else _fail)(f"{partition}: {n} trips (expected {expected})")
    return all_ok


def main() -> int:
    print("=" * 60)
    print("PathSnap environment verification")
    print("=" * 60)
    results = [
        check_python(),
        check_packages(),
        check_package_and_paths(),
        check_dataset(),
    ]
    print("-" * 60)
    if all(results):
        print("RESULT: all checks passed.")
        return 0
    print("RESULT: one or more checks FAILED (see above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
