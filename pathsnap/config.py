"""Central configuration: paths, random seeds and experiment settings.

Everything path-related is derived from the repository root so the project is
portable across machines (important for the two-laptop deployment). No absolute
paths are hard-coded.

Usage
-----
>>> from pathsnap.config import CONFIG
>>> CONFIG.paths.train_dir
>>> CONFIG.seed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# config.py lives at <root>/pathsnap/config.py -> parents[1] is the repo root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem locations used throughout the project."""

    root: Path = PROJECT_ROOT

    # Raw dataset (read-only — never written to).
    data_raw: Path = PROJECT_ROOT / "data" / "raw" / "Student folder"

    # Derived / generated artifacts (git-ignored).
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    outputs: Path = PROJECT_ROOT / "outputs"
    outputs_eda: Path = PROJECT_ROOT / "outputs" / "eda"
    outputs_maps: Path = PROJECT_ROOT / "outputs" / "maps"
    outputs_experiments: Path = PROJECT_ROOT / "outputs" / "experiments"
    outputs_hpo: Path = PROJECT_ROOT / "outputs" / "hpo"
    models: Path = PROJECT_ROOT / "outputs" / "models"
    logs: Path = PROJECT_ROOT / "outputs" / "logs"
    reports: Path = PROJECT_ROOT / "reports"

    @property
    def train_dir(self) -> Path:
        """Directory holding the 80 training trips."""
        return self.data_raw / "train"

    @property
    def student_test_dir(self) -> Path:
        """Directory holding the 10 student-testing trips (dev evaluation)."""
        return self.data_raw / "student_test"

    def partition_dir(self, partition: str) -> Path:
        """Return the directory for a named partition.

        Parameters
        ----------
        partition:
            One of ``"train"`` or ``"student_test"``. The evaluator-testing
            partition is intentionally unavailable locally (leakage prevention).
        """
        mapping = {"train": self.train_dir, "student_test": self.student_test_dir}
        if partition not in mapping:
            raise ValueError(
                f"Unknown partition {partition!r}. "
                f"Allowed: {sorted(mapping)}. "
                "The evaluator-testing partition must not be used locally."
            )
        return mapping[partition]

    def ensure_output_dirs(self) -> None:
        """Create all writable output directories if missing (idempotent)."""
        for p in (
            self.data_processed,
            self.outputs,
            self.outputs_eda,
            self.outputs_maps,
            self.outputs_experiments,
            self.outputs_hpo,
            self.models,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)


# Raw file extensions (schema documented in docs/DATASET_SCHEMA.md).
TRACK_EXT = ".track"   # lon, lat, timestamp(s)
NODES_EXT = ".nodes"   # lon, lat   (row index = node_id)
ARCS_EXT = ".arcs"     # from_node_id, to_node_id (row index = arc_id)
ROUTE_EXT = ".route"   # arc_id sequence (ground truth)

# Only these partitions may be touched during development.
PERMITTED_PARTITIONS = ("train", "student_test")


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducibility and experiment-wide settings.

    Centralising the seed guarantees that every stochastic component
    (CV splits, model init, HPO sampler) can be made deterministic.
    """

    seed: int = 42
    # Group-aware CV: the group key is always the trip ID (no trip may be split
    # across folds — see docs/PROJECT_ANALYSIS.md R2).
    cv_folds: int = 5
    group_key: str = "trip_id"

    # Geography: the dataset is globally distributed, so distances must be
    # computed per trip (local projection or geodesic), never with a single
    # global CRS (see docs/DATASET_SCHEMA.md §5). WGS84 is the source CRS.
    source_crs: str = "EPSG:4326"

    paths: Paths = field(default_factory=Paths)


# Singleton used across the codebase.
CONFIG = ExperimentConfig()


if __name__ == "__main__":  # Quick manual sanity check.
    print("PROJECT_ROOT :", CONFIG.paths.root)
    print("train_dir    :", CONFIG.paths.train_dir, "exists:", CONFIG.paths.train_dir.exists())
    print("student_test :", CONFIG.paths.student_test_dir, "exists:", CONFIG.paths.student_test_dir.exists())
    print("seed         :", CONFIG.seed)
