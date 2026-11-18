"""Data ingestion: readers for the raw PathSnap files.

Parses the four whitespace-separated, header-less file types documented in
``docs/DATASET_SCHEMA.md`` into typed NumPy arrays. These readers are read-only
and never modify the raw dataset.

File types (0-based row index is the implicit ID where noted)
    .track : lon, lat, timestamp(s)           -> one GPS observation per row
    .nodes : lon, lat                          -> row index = node_id
    .arcs  : from_node_id, to_node_id (int)    -> row index = arc_id (directed)
    .route : arc_id (int)                      -> chronological ground-truth seq

This module intentionally performs *parsing only*. Cleaning, projection,
candidate generation and label derivation live in later-phase modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from pathsnap.config import (
    ARCS_EXT,
    CONFIG,
    NODES_EXT,
    ROUTE_EXT,
    TRACK_EXT,
)


@dataclass(frozen=True)
class Trip:
    """In-memory representation of a single trip's raw data.

    Attributes
    ----------
    trip_id:
        8-digit zero-padded trip identifier (also the group key for CV).
    partition:
        Partition the trip was loaded from (``"train"`` / ``"student_test"``).
    track:
        Array of shape ``(n_obs, 3)`` = [longitude, latitude, timestamp_s].
    nodes:
        Array of shape ``(n_nodes, 2)`` = [longitude, latitude]; row index =
        node_id.
    arcs:
        Integer array of shape ``(n_arcs, 2)`` = [from_node_id, to_node_id];
        row index = arc_id. Arcs are directed.
    route:
        Integer array of shape ``(route_len,)`` = chronological arc_id sequence
        (ground-truth reference route).
    """

    trip_id: str
    partition: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray

    @property
    def n_observations(self) -> int:
        return int(self.track.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])

    @property
    def route_length(self) -> int:
        return int(self.route.shape[0])


# --------------------------------------------------------------------------- #
# Low-level file readers
# --------------------------------------------------------------------------- #
def read_track(path: Path) -> np.ndarray:
    """Read a ``.track`` file into a float array ``(n_obs, 3)`` [lon, lat, t]."""
    arr = np.loadtxt(path, dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns (lon,lat,t), got {arr.shape[1]}")
    return arr


def read_nodes(path: Path) -> np.ndarray:
    """Read a ``.nodes`` file into a float array ``(n_nodes, 2)`` [lon, lat]."""
    arr = np.loadtxt(path, dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 2:
        raise ValueError(f"{path}: expected 2 columns (lon,lat), got {arr.shape[1]}")
    return arr


def read_arcs(path: Path) -> np.ndarray:
    """Read an ``.arcs`` file into an int array ``(n_arcs, 2)`` [from, to]."""
    arr = np.loadtxt(path, dtype=np.int64)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 2:
        raise ValueError(f"{path}: expected 2 columns (from,to), got {arr.shape[1]}")
    return arr


def read_route(path: Path) -> np.ndarray:
    """Read a ``.route`` file into a 1-D int array of arc_ids."""
    arr = np.loadtxt(path, dtype=np.int64)
    return np.atleast_1d(arr)


# --------------------------------------------------------------------------- #
# Trip-level helpers
# --------------------------------------------------------------------------- #
def list_trips(partition: str) -> List[str]:
    """Return the sorted list of trip IDs available in a partition."""
    pdir = CONFIG.paths.partition_dir(partition)
    if not pdir.exists():
        raise FileNotFoundError(f"Partition directory not found: {pdir}")
    return sorted(d.name for d in pdir.iterdir() if d.is_dir())


def load_trip(trip_id: str, partition: str) -> Trip:
    """Load all four raw files for a trip into a :class:`Trip`.

    Parameters
    ----------
    trip_id:
        8-digit zero-padded trip identifier.
    partition:
        ``"train"`` or ``"student_test"`` (evaluator partition is off-limits).
    """
    pdir = CONFIG.paths.partition_dir(partition)
    base = pdir / trip_id / trip_id
    return Trip(
        trip_id=trip_id,
        partition=partition,
        track=read_track(base.with_suffix(TRACK_EXT)),
        nodes=read_nodes(base.with_suffix(NODES_EXT)),
        arcs=read_arcs(base.with_suffix(ARCS_EXT)),
        route=read_route(base.with_suffix(ROUTE_EXT)),
    )


def iter_trips(partition: str):
    """Lazily yield every :class:`Trip` in a partition (memory-friendly)."""
    for trip_id in list_trips(partition):
        yield load_trip(trip_id, partition)
