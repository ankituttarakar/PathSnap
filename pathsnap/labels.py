"""Observation->arc label derivation by monotonic route projection (Phase 2).

The raw data gives an *ordered* ground-truth arc sequence (``.route``) but does
NOT align GPS rows to route rows. We derive a per-observation arc label by
projecting each GPS point onto the route **polyline** while enforcing forward
progression along the route. This is deliberately NOT a blind nearest-road
assignment:

  * The route order is respected (a windowed, monotonically non-decreasing
    pointer), which preserves direction and disambiguates **repeated arcs**
    (each occurrence in the route is a distinct position).
  * **Broken route transitions** are tolerated: the window may advance across a
    non-node-connected pair without inventing connector arcs.
  * **Off-route** observations (perpendicular distance above ``off_route_m``) are
    flagged ``confident=False`` and must be handled by an explicit policy
    downstream (e.g. excluded from supervised label training), not trusted.

Labels are reference information for *training only*; they must never become a
model input feature (leakage rule).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathsnap.geo import project_point_to_segments
from pathsnap.preprocess import ProcessedTrip

OFF_ROUTE_M = 50.0       # perpendicular distance above which an obs is "off route"
FORWARD_WINDOW = 25      # max route-arcs to search ahead of the current pointer
BACK_TOLERANCE = 1       # allow at most this many arcs of backward slack (GPS noise)


@dataclass
class LabelResult:
    """Per-observation route-association result for one trip."""

    label_arc_id: np.ndarray    # (N,) assigned arc_id (original id), -1 if none
    route_pos: np.ndarray       # (N,) index into the route sequence
    proj_dist_m: np.ndarray     # (N,) perpendicular distance to the route (m)
    along_route_m: np.ndarray   # (N,) cumulative distance along the route (m)
    confident: np.ndarray       # (N,) bool: proj_dist <= off_route_m

    @property
    def coverage(self) -> float:
        """Fraction of observations confidently placed on the route."""
        return float(np.mean(self.confident)) if self.confident.size else float("nan")


def derive_labels(
    pt: ProcessedTrip,
    route: np.ndarray,
    off_route_m: float = OFF_ROUTE_M,
    window: int = FORWARD_WINDOW,
) -> LabelResult:
    """Assign each observation to a route arc by monotonic projection.

    Parameters
    ----------
    pt:
        Processed trip (provides projected obs + arc geometry).
    route:
        Ordered array of ground-truth arc_ids (``Trip.route``).
    """
    n = pt.n_obs
    # Route-arc geometry in order (endpoints + cumulative start length).
    valid = (route >= 0) & (route < pt.n_arcs)
    r = route[valid]
    rax, ray = pt.arc_ax[r], pt.arc_ay[r]
    rbx, rby = pt.arc_bx[r], pt.arc_by[r]
    rlen = pt.arc_len[r]
    cum_start = np.concatenate([[0.0], np.cumsum(rlen)[:-1]]) if rlen.size else np.array([])

    label_arc = np.full(n, -1, dtype=np.int64)
    route_pos = np.full(n, -1, dtype=np.int64)
    proj_dist = np.full(n, np.nan)
    along = np.full(n, np.nan)

    if r.size == 0:
        return LabelResult(label_arc, route_pos, proj_dist, along, np.zeros(n, bool))

    cur = 0  # monotonic pointer into the route
    for i in range(n):
        lo = max(0, cur - BACK_TOLERANCE)
        hi = min(r.size, cur + window)
        idx = np.arange(lo, hi)
        d, t, _, _ = project_point_to_segments(
            pt.xy[i, 0], pt.xy[i, 1], rax[idx], ray[idx], rbx[idx], rby[idx]
        )
        j = int(idx[int(np.argmin(d))])
        label_arc[i] = int(r[j])
        route_pos[i] = j
        proj_dist[i] = float(d[int(np.argmin(d))])
        along[i] = float(cum_start[j] + t[int(np.argmin(d))] * rlen[j])
        cur = max(cur, j)  # never regress past the furthest confirmed position

    confident = proj_dist <= off_route_m
    return LabelResult(label_arc, route_pos, proj_dist, along, confident)


def derive_observation_arc_labels(*args, **kwargs):  # noqa: ANN002, ANN003
    """Deprecated alias retained for docs; use :func:`derive_labels`."""
    raise NotImplementedError("Use derive_labels(processed_trip, route) instead.")
