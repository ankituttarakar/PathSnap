"""Exploratory Data Analysis for PathSnap (Phase 1).

Reusable, modular analysis functions over the raw trips (read-only). Nothing
here trains a model, derives observation->arc labels, or mixes partitions — it
produces descriptive evidence to guide Phases 2-8.

Design notes / assumptions
--------------------------
* Distances use a vectorised haversine (great-circle) in metres. The dataset is
  globally distributed, so a single projected CRS is inappropriate here; the
  geodesic is adequate for the short consecutive-point distances used in speed
  estimation (committing to a projection is a Phase-2 decision, see geo.py).
* "Speed" is a descriptive estimate = geodesic(consecutive points) / Δt. It is
  only computed between consecutive observations with Δt > 0.
* Connectivity of the road network uses a scalable sparse weakly-connected-
  component analysis (scipy.csgraph) so it runs on *all* trips, not just a few.
* Representative trips for mapping are chosen by documented, deterministic
  criteria (see :func:`select_representative_trips`) — never arbitrarily.

Outputs
-------
Per-trip stat tables (CSV) + summary charts under ``outputs/eda`` and Folium
maps under ``outputs/maps`` (written by the orchestration helpers).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.data_io import Trip, iter_trips, load_trip
from pathsnap.logging_utils import get_logger

logger = get_logger("pathsnap.eda")

# --------------------------------------------------------------------------- #
# Thresholds (documented, not magic). Distinguish *suspicious* from *confirmed*.
# --------------------------------------------------------------------------- #
GAP_THRESHOLD_S = 1.0          # a "gap" is any inter-sample Δt strictly > 1 s
LONG_GAP_S = 60.0              # inter-sample gaps above this are flagged "long"
# Speed-based GPS-quality thresholds (metres/second).
SPEED_SUSPICIOUS_MS = 55.0     # ~198 km/h: plausible only on fast motorways
SPEED_IMPLAUSIBLE_MS = 150.0   # ~540 km/h: not a ground vehicle -> confirmed error
# A single consecutive jump this large (metres) is a confirmed error regardless
# of Δt (no road vehicle moves >5 km between two ~1 Hz samples).
JUMP_CONFIRMED_M = 5000.0

EARTH_RADIUS_M = 6_371_000.0


# --------------------------------------------------------------------------- #
# Geodesic helper (local to EDA; not the Phase-2 projection in geo.py)
# --------------------------------------------------------------------------- #
def haversine_m(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Vectorised great-circle distance in metres between coordinate arrays."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# --------------------------------------------------------------------------- #
# Per-trip analyses. Each returns a flat dict of scalar summaries plus, where
# useful, raw arrays for aggregate charts.
# --------------------------------------------------------------------------- #
@dataclass
class TrackRecord:
    """Summary statistics for one trip's GPS observations (.track)."""

    n_obs: int
    duration_s: float
    frac_1s_steps: float
    n_gaps_gt_1s: int
    n_long_gaps: int
    max_gap_s: float
    n_dup_timestamps: int
    n_nonmonotonic: int
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    bbox_diag_km: float
    max_jump_m: float
    median_speed_ms: float
    p95_speed_ms: float
    max_speed_ms: float
    n_suspicious_speed: int
    n_confirmed_speed_error: int
    n_confirmed_jump_error: int


def analyze_track(trip: Trip) -> Tuple[TrackRecord, np.ndarray, np.ndarray]:
    """Analyse ``.track`` for one trip.

    Returns the :class:`TrackRecord` plus the raw inter-sample interval array
    (seconds) and the raw consecutive-speed array (m/s) for aggregate charts.
    """
    track = trip.track
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]

    dt = np.diff(ts)
    intervals = dt
    n_dup = int(np.sum(dt == 0))
    n_nonmono = int(np.sum(dt < 0))
    # Fraction of steps that are exactly 1 s (dataset is nominally 1 Hz).
    frac_1s = float(np.mean(np.isclose(dt, 1.0))) if dt.size else float("nan")
    gaps = dt[dt > GAP_THRESHOLD_S]
    max_gap = float(dt.max()) if dt.size else 0.0

    # Consecutive geodesic distances and speeds (only where Δt > 0).
    seg_m = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    valid = dt > 0
    speeds = np.full(seg_m.shape, np.nan)
    speeds[valid] = seg_m[valid] / dt[valid]
    finite_speeds = speeds[np.isfinite(speeds)]

    bbox_diag_km = haversine_m(lon.min(), lat.min(), lon.max(), lat.max()) / 1000.0

    rec = TrackRecord(
        n_obs=trip.n_observations,
        duration_s=float(ts[-1] - ts[0]) if ts.size else 0.0,
        frac_1s_steps=frac_1s,
        n_gaps_gt_1s=int(gaps.size),
        n_long_gaps=int(np.sum(dt > LONG_GAP_S)),
        max_gap_s=max_gap,
        n_dup_timestamps=n_dup,
        n_nonmonotonic=n_nonmono,
        lon_min=float(lon.min()),
        lon_max=float(lon.max()),
        lat_min=float(lat.min()),
        lat_max=float(lat.max()),
        bbox_diag_km=float(bbox_diag_km),
        max_jump_m=float(seg_m.max()) if seg_m.size else 0.0,
        median_speed_ms=float(np.median(finite_speeds)) if finite_speeds.size else float("nan"),
        p95_speed_ms=float(np.percentile(finite_speeds, 95)) if finite_speeds.size else float("nan"),
        max_speed_ms=float(finite_speeds.max()) if finite_speeds.size else float("nan"),
        n_suspicious_speed=int(np.sum(finite_speeds > SPEED_SUSPICIOUS_MS)),
        n_confirmed_speed_error=int(np.sum(finite_speeds > SPEED_IMPLAUSIBLE_MS)),
        n_confirmed_jump_error=int(np.sum(seg_m > JUMP_CONFIRMED_M)),
    )
    return rec, intervals, finite_speeds


@dataclass
class NetworkRecord:
    """Summary statistics for one trip's road network (.nodes / .arcs)."""

    n_nodes: int
    n_arcs: int
    n_self_loops: int
    n_duplicate_arcs: int
    frac_arcs_with_reverse: float
    n_isolated_nodes: int
    n_invalid_endpoint_refs: int
    n_invalid_node_coords: int
    mean_out_degree: float
    max_out_degree: int
    density_arcs_per_node: float
    n_weak_components: int
    largest_component_frac: float


def analyze_network(trip: Trip) -> NetworkRecord:
    """Analyse the road network for one trip using scalable sparse operations."""
    nodes = trip.nodes
    arcs = trip.arcs
    n_nodes = trip.n_nodes
    n_arcs = trip.n_arcs
    frm, to = arcs[:, 0], arcs[:, 1]

    # Validity checks.
    invalid_ref = int(
        np.sum((frm < 0) | (frm >= n_nodes) | (to < 0) | (to >= n_nodes))
    )
    bad_coords = int(
        np.sum(
            (nodes[:, 0] < -180) | (nodes[:, 0] > 180)
            | (nodes[:, 1] < -90) | (nodes[:, 1] > 90)
            | ~np.isfinite(nodes).all(axis=1)
        )
    )

    n_self_loops = int(np.sum(frm == to))

    # Duplicate *directed* arcs (same from,to appearing more than once).
    uniq_pairs = np.unique(arcs, axis=0)
    n_duplicate = int(n_arcs - uniq_pairs.shape[0])

    # Reverse-edge availability: fraction of directed arcs (u,v) whose reverse
    # (v,u) is also present. Uses a hashed set of unique pairs.
    pair_set = set(map(tuple, uniq_pairs.tolist()))
    rev_present = sum(1 for u, v in pair_set if (v, u) in pair_set)
    frac_reverse = float(rev_present / len(pair_set)) if pair_set else float("nan")

    # Degrees (out-degree per source node over valid refs).
    valid_mask = (frm >= 0) & (frm < n_nodes)
    out_deg = np.bincount(frm[valid_mask], minlength=n_nodes)
    referenced = np.zeros(n_nodes, dtype=bool)
    referenced[frm[valid_mask]] = True
    referenced[to[(to >= 0) & (to < n_nodes)]] = True
    n_isolated = int(n_nodes - int(referenced.sum()))

    # Weakly-connected components via sparse CSR (scales to 300k+ nodes).
    data = np.ones(frm[valid_mask].shape[0], dtype=np.int8)
    adj = csr_matrix(
        (data, (frm[valid_mask], to[valid_mask])), shape=(n_nodes, n_nodes)
    )
    n_comp, labels = connected_components(adj, directed=True, connection="weak")
    if n_nodes:
        comp_sizes = np.bincount(labels)
        largest_frac = float(comp_sizes.max() / n_nodes)
    else:
        largest_frac = float("nan")

    return NetworkRecord(
        n_nodes=n_nodes,
        n_arcs=n_arcs,
        n_self_loops=n_self_loops,
        n_duplicate_arcs=n_duplicate,
        frac_arcs_with_reverse=frac_reverse,
        n_isolated_nodes=n_isolated,
        n_invalid_endpoint_refs=invalid_ref,
        n_invalid_node_coords=bad_coords,
        mean_out_degree=float(out_deg.mean()) if n_nodes else float("nan"),
        max_out_degree=int(out_deg.max()) if n_nodes else 0,
        density_arcs_per_node=float(n_arcs / n_nodes) if n_nodes else float("nan"),
        n_weak_components=int(n_comp),
        largest_component_frac=largest_frac,
    )


@dataclass
class RouteRecord:
    """Summary statistics for one trip's ground-truth route (.route)."""

    route_len: int
    n_unique_arcs: int
    n_repeated_arcs: int
    n_invalid_arc_refs: int
    n_consecutive_pairs: int
    n_connected_pairs: int
    frac_connected: float
    n_broken_links: int
    route_len_over_obs: float


def analyze_route(trip: Trip) -> RouteRecord:
    """Analyse route integrity and consecutive-arc connectivity.

    Connectivity is defined as: for consecutive route arcs ``a_i`` and
    ``a_{i+1}``, the destination node of ``a_i`` equals the source node of
    ``a_{i+1}`` (``arcs[a_i].to == arcs[a_{i+1}].from``). This does NOT assume
    the route provides per-observation labels.
    """
    route = trip.route
    arcs = trip.arcs
    n_arcs = trip.n_arcs

    route_len = trip.route_length
    n_unique = int(np.unique(route).size)
    invalid = (route < 0) | (route >= n_arcs)
    n_invalid = int(invalid.sum())

    # Connectivity only over consecutive pairs whose BOTH arc ids are valid.
    n_pairs = max(route_len - 1, 0)
    connected = 0
    broken = 0
    if route_len >= 2:
        a = route[:-1]
        b = route[1:]
        valid_pair = (~invalid[:-1]) & (~invalid[1:])
        to_a = np.where(valid_pair, arcs[np.clip(a, 0, n_arcs - 1), 1], -1)
        from_b = np.where(valid_pair, arcs[np.clip(b, 0, n_arcs - 1), 0], -2)
        connected = int(np.sum((to_a == from_b) & valid_pair))
        broken = int(np.sum((to_a != from_b) & valid_pair))

    return RouteRecord(
        route_len=route_len,
        n_unique_arcs=n_unique,
        n_repeated_arcs=int(route_len - n_unique),
        n_invalid_arc_refs=n_invalid,
        n_consecutive_pairs=int(n_pairs),
        n_connected_pairs=connected,
        frac_connected=float(connected / n_pairs) if n_pairs else float("nan"),
        n_broken_links=broken,
        route_len_over_obs=float(route_len / trip.n_observations)
        if trip.n_observations
        else float("nan"),
    )


# --------------------------------------------------------------------------- #
# Partition-level orchestration
# --------------------------------------------------------------------------- #
def analyze_partition(
    partition: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run all per-trip analyses over a partition.

    Returns
    -------
    df:
        One row per trip (trip_id + all track/network/route summaries).
    all_intervals:
        Concatenated inter-sample Δt (seconds) across the partition (for charts).
    all_speeds:
        Concatenated finite consecutive speeds (m/s) across the partition.
    """
    rows: List[Dict] = []
    intervals_accum: List[np.ndarray] = []
    speeds_accum: List[np.ndarray] = []

    for trip in iter_trips(partition):
        track_rec, intervals, speeds = analyze_track(trip)
        net_rec = analyze_network(trip)
        route_rec = analyze_route(trip)
        row = {"trip_id": trip.trip_id, "partition": partition}
        row.update(asdict(track_rec))
        row.update(asdict(net_rec))
        row.update(asdict(route_rec))
        rows.append(row)
        intervals_accum.append(intervals)
        speeds_accum.append(speeds)
        logger.info(
            "%s/%s: obs=%d nodes=%d arcs=%d route=%d connected=%.3f",
            partition,
            trip.trip_id,
            track_rec.n_obs,
            net_rec.n_nodes,
            net_rec.n_arcs,
            route_rec.route_len,
            route_rec.frac_connected,
        )

    df = pd.DataFrame(rows).sort_values("trip_id").reset_index(drop=True)
    all_intervals = np.concatenate(intervals_accum) if intervals_accum else np.array([])
    all_speeds = np.concatenate(speeds_accum) if speeds_accum else np.array([])
    return df, all_intervals, all_speeds


def select_representative_trips(df: pd.DataFrame) -> Dict[str, str]:
    """Deterministically pick representative trips for mapping.

    Criteria (all ties broken by ascending trip_id for reproducibility):
        * ``typical``       : n_obs closest to the partition median.
        * ``dense_network`` : largest arc count (densest road network).
        * ``gps_anomaly``   : largest single consecutive jump (worst outlier).

    If two criteria select the same trip, the later one falls back to the next
    best distinct candidate so three distinct trips are always returned.
    """
    chosen: Dict[str, str] = {}
    used: set = set()

    median_obs = df["n_obs"].median()
    typical_order = (
        df.assign(_d=(df["n_obs"] - median_obs).abs())
        .sort_values(["_d", "trip_id"])["trip_id"]
        .tolist()
    )
    dense_order = df.sort_values(["n_arcs", "trip_id"], ascending=[False, True])[
        "trip_id"
    ].tolist()
    anomaly_order = df.sort_values(["max_jump_m", "trip_id"], ascending=[False, True])[
        "trip_id"
    ].tolist()

    for role, order in (
        ("typical", typical_order),
        ("dense_network", dense_order),
        ("gps_anomaly", anomaly_order),
    ):
        for tid in order:
            if tid not in used:
                chosen[role] = tid
                used.add(tid)
                break
    return chosen


def run_eda(*args, **kwargs):  # noqa: ANN002, ANN003
    """Deprecated placeholder kept for API stability.

    Use :func:`analyze_partition` and the orchestration in ``scripts/run_eda.py``.
    """
    raise NotImplementedError(
        "Use analyze_partition()/scripts/run_eda.py instead of run_eda()."
    )
