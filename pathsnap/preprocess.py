"""Preprocessing: projection, segmentation, outlier flagging, arc geometry.

Produces a :class:`ProcessedTrip` holding planar (metre) geometry derived from a
raw :class:`pathsnap.data_io.Trip`. Nothing is deleted from the raw data — GPS
outliers are *flagged*, not removed, and every arc keeps its original 0-based
``arc_id`` so predictions remain traceable to the raw network.

Decisions (justified) are recorded in docs via the first-evaluation report:
  * Projection: per-trip local UTM (metres) — global dataset precludes one CRS.
  * Segmentation: a new segment starts when Δt exceeds ``gap_split_s`` (default
    30 s). Motion features are undefined across a segment break.
  * Outliers: flagged when the implied consecutive speed exceeds
    ``speed_flag_ms`` (default 55 m/s). Flagged points are retained and excluded
    only from speed-feature smoothing, never deleted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathsnap.data_io import InferenceTrip, Trip
from pathsnap.geo import Projector, haversine_m, segment_bearings

GAP_SPLIT_S = 30.0       # Δt above this starts a new trajectory segment
SPEED_FLAG_MS = 55.0     # consecutive speed above this flags a suspicious point


@dataclass
class ProcessedTrip:
    """Projected, segmented view of a trip plus directed arc geometry."""

    trip_id: str
    partition: str
    epsg: int
    # Observations (order preserved from raw).
    lonlat: np.ndarray        # (N, 2) original lon/lat (unchanged)
    ts: np.ndarray            # (N,) timestamps (s)
    xy: np.ndarray            # (N, 2) projected metres
    dt: np.ndarray            # (N,) Δt to previous obs (first = nan)
    speed_ms: np.ndarray      # (N,) consecutive geodesic speed (first = nan)
    is_outlier: np.ndarray    # (N,) bool flag (NOT removed)
    segment_id: np.ndarray    # (N,) int segment index
    heading: np.ndarray       # (N,) planar heading to previous point (rad)
    # Network (original arc ids == row index preserved).
    node_xy: np.ndarray       # (K, 2) projected node coords
    arcs: np.ndarray          # (M, 2) [from_node, to_node]
    arc_ax: np.ndarray        # (M,) from-node x
    arc_ay: np.ndarray
    arc_bx: np.ndarray        # (M,) to-node x
    arc_by: np.ndarray
    arc_len: np.ndarray       # (M,) metres
    arc_bearing: np.ndarray   # (M,) radians
    arc_mid: np.ndarray       # (M, 2) midpoint (for spatial index)

    @property
    def n_obs(self) -> int:
        return int(self.xy.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])


def preprocess_trip(
    trip: Trip | InferenceTrip,
    gap_split_s: float = GAP_SPLIT_S,
    speed_flag_ms: float = SPEED_FLAG_MS,
) -> ProcessedTrip:
    """Build a projected view from raw trip data (read-only).

    ``InferenceTrip`` is supported so prediction never needs a reference route.
    """
    lon, lat, ts = trip.track[:, 0], trip.track[:, 1], trip.track[:, 2]
    proj = Projector.for_points(lon, lat)
    xy = proj.xy(lon, lat)

    # Δt and consecutive geodesic speed (first obs has no predecessor).
    dt = np.full(len(ts), np.nan)
    dt[1:] = np.diff(ts)
    seg_m = np.full(len(ts), np.nan)
    seg_m[1:] = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = seg_m / dt

    # Segmentation at large gaps.
    segment_id = np.zeros(len(ts), dtype=int)
    breaks = np.zeros(len(ts), dtype=bool)
    if len(ts) > 1:
        breaks = np.nan_to_num(dt, nan=0.0) > gap_split_s
        segment_id = np.cumsum(breaks).astype(int)
        speed[breaks] = np.nan

    is_outlier = np.zeros(len(ts), dtype=bool)
    is_outlier[1:] = np.nan_to_num(speed[1:], nan=0.0) > speed_flag_ms

    # Planar heading to previous point.
    heading = np.full(len(ts), np.nan)
    if len(ts) > 1:
        heading[1:] = np.arctan2(xy[1:, 1] - xy[:-1, 1], xy[1:, 0] - xy[:-1, 0])
        heading[breaks] = np.nan

    # Directed arc geometry (original arc_id == row index preserved).
    node_xy = proj.xy(trip.nodes[:, 0], trip.nodes[:, 1])
    frm, to = trip.arcs[:, 0], trip.arcs[:, 1]
    ax, ay = node_xy[frm, 0], node_xy[frm, 1]
    bx, by = node_xy[to, 0], node_xy[to, 1]
    arc_len = np.hypot(bx - ax, by - ay)
    arc_bearing = segment_bearings(ax, ay, bx, by)
    arc_mid = np.column_stack([(ax + bx) / 2.0, (ay + by) / 2.0])

    return ProcessedTrip(
        trip_id=trip.trip_id, partition=trip.partition, epsg=proj.epsg,
        lonlat=trip.track[:, :2].copy(), ts=ts.copy(), xy=xy, dt=dt,
        speed_ms=speed, is_outlier=is_outlier, segment_id=segment_id, heading=heading,
        node_xy=node_xy, arcs=trip.arcs.copy(),
        arc_ax=ax, arc_ay=ay, arc_bx=bx, arc_by=by,
        arc_len=arc_len, arc_bearing=arc_bearing, arc_mid=arc_mid,
    )


def clean_trip(*args, **kwargs):  # noqa: ANN002, ANN003
    """Deprecated alias retained for docs; use :func:`preprocess_trip`."""
    raise NotImplementedError("Use preprocess_trip(trip) instead of clean_trip().")
