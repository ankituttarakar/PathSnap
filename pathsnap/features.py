"""Feature engineering for candidate classification (Phase 4).

Builds one feature row per (observation, candidate-arc). The target is whether
the candidate arc is the observation's derived label arc.

LEAKAGE RULE (enforced here): every feature is computable from the GPS
observation and the road-network geometry alone. **No feature uses the ``.route``
sequence.** The label column (``y``) is derived from the route for *training
only* and is never fed back as a feature. Along-route ground-truth progression is
intentionally excluded; only inference-time-available context (cumulative GPS
distance, Δt) is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.candidates import Candidates
from pathsnap.geo import angle_diff
from pathsnap.labels import LabelResult
from pathsnap.preprocess import ProcessedTrip

# Feature columns produced (kept explicit for reproducibility / the report).
FEATURE_COLUMNS = [
    "dist_m",          # perpendicular GPS->arc distance
    "t",               # fractional position along arc
    "rank",            # candidate distance rank within observation
    "n_cand",          # number of candidates for the observation
    "dist_margin_m",   # distance gap to the closest candidate
    "arc_len",         # candidate arc length
    "heading_diff",    # |GPS heading - arc bearing| folded to [0, pi/2]
    "speed_ms",        # GPS consecutive speed
    "dt_s",            # Δt to previous observation
    "is_outlier",      # GPS outlier flag for the observation
    "seg_start",       # 1 if observation starts a new trajectory segment
    "cum_gps_frac",    # cumulative GPS distance fraction (inference-available)
    "heading_diff_directed_rad",  # GPS course vs directed arc bearing, [0, pi]
    "signed_cross_track_m",  # signed offset left/right of directed candidate arc
    "previous_point_dist_m",  # previous fix distance to this candidate arc
    "projected_speed_along_arc_ms",  # signed candidate progress per second
    "projected_speed_error_ms",  # observed speed vs candidate-projected speed
    "gps_heading_change_rad",  # course change between consecutive GPS steps
]


def build_feature_frame(
    pt: ProcessedTrip, cand: Candidates, labels: LabelResult | None = None
) -> pd.DataFrame:
    """Assemble the (obs, candidate) feature frame for one trip.

    If ``labels`` is given, a ``y`` column (1 = candidate is the label arc) and a
    ``confident`` column are added for supervised training; rows for non-confident
    observations should be filtered by the caller before fitting.
    """
    o = cand.obs_idx
    a = cand.arc_id

    # Per-observation cumulative GPS distance fraction (inference-available).
    step = np.nan_to_num(
        np.r_[0.0, np.hypot(np.diff(pt.xy[:, 0]), np.diff(pt.xy[:, 1]))], nan=0.0
    )
    cum = np.cumsum(step)
    cum_frac = cum / cum[-1] if cum[-1] > 0 else np.zeros_like(cum)

    # Distance margin: gap to the closest candidate of the same observation.
    closest = pd.Series(cand.dist_m).groupby(o).transform("min").to_numpy()
    dist_margin = cand.dist_m - closest

    # Keep this legacy undirected angle for compatibility. Direction-sensitive
    # alignment is represented by heading_diff_directed_rad below.
    obs_head = pt.heading[o]
    arc_brg = pt.arc_bearing[a]
    hd = np.where(
        np.isfinite(obs_head),
        angle_diff(np.nan_to_num(obs_head, nan=0.0), arc_brg),
        np.pi / 2.0,
    )
    hd = np.minimum(hd, np.pi - hd)

    seg = pt.segment_id
    seg_start = np.r_[1, (np.diff(seg) != 0).astype(int)][o]
    previous_obs = np.maximum(o - 1, 0)
    has_previous = (
        (o > 0)
        & (seg[o] == seg[previous_obs])
        & np.isfinite(pt.dt[o])
        & (pt.dt[o] > 0)
    )

    arc_dx = pt.arc_bx[a] - pt.arc_ax[a]
    arc_dy = pt.arc_by[a] - pt.arc_ay[a]
    arc_len = pt.arc_len[a]
    safe_arc_len = np.where(arc_len > 0, arc_len, 1.0)

    # Signed cross-track distance is measured relative to the directed arc.
    signed_cross_track = (
        arc_dx * (pt.xy[o, 1] - pt.arc_ay[a])
        - arc_dy * (pt.xy[o, 0] - pt.arc_ax[a])
    ) / safe_arc_len
    signed_cross_track = np.where(arc_len > 0, signed_cross_track, 0.0)

    # Project the previous fix onto each current candidate segment. These
    # temporal features are neutral at segment boundaries and never cross gaps.
    previous_xy = pt.xy[previous_obs]
    arc_len_squared = arc_dx * arc_dx + arc_dy * arc_dy
    safe_arc_len_squared = np.where(arc_len_squared > 0, arc_len_squared, 1.0)
    previous_t = (
        (previous_xy[:, 0] - pt.arc_ax[a]) * arc_dx
        + (previous_xy[:, 1] - pt.arc_ay[a]) * arc_dy
    ) / safe_arc_len_squared
    previous_t = np.clip(previous_t, 0.0, 1.0)
    previous_foot_x = pt.arc_ax[a] + previous_t * arc_dx
    previous_foot_y = pt.arc_ay[a] + previous_t * arc_dy
    previous_dist = np.hypot(
        previous_xy[:, 0] - previous_foot_x,
        previous_xy[:, 1] - previous_foot_y,
    )
    previous_dist = np.where(has_previous, previous_dist, 0.0)

    safe_dt = np.where(has_previous, pt.dt[o], 1.0)
    projected_speed = ((cand.t - previous_t) * arc_len) / safe_dt
    projected_speed = np.where(has_previous, projected_speed, 0.0)
    observed_speed = np.nan_to_num(pt.speed_ms[o], nan=0.0, posinf=0.0, neginf=0.0)
    projected_speed_error = np.where(
        has_previous, np.abs(observed_speed - projected_speed), 0.0
    )

    directed_heading_diff = np.where(
        np.isfinite(obs_head),
        angle_diff(np.nan_to_num(obs_head, nan=0.0), arc_brg),
        np.pi / 2.0,
    )
    previous_heading = pt.heading[previous_obs]
    has_heading_pair = (
        has_previous
        & np.isfinite(pt.heading[o])
        & np.isfinite(previous_heading)
    )
    heading_change = np.where(
        has_heading_pair,
        angle_diff(
            np.nan_to_num(pt.heading[o], nan=0.0),
            np.nan_to_num(previous_heading, nan=0.0),
        ),
        0.0,
    )

    df = pd.DataFrame({
        "trip_id": pt.trip_id,
        "obs_idx": o,
        "arc_id": a,
        "dist_m": cand.dist_m,
        "t": cand.t,
        "rank": cand.rank,
        "n_cand": cand.n_cand,
        "dist_margin_m": dist_margin,
        "arc_len": pt.arc_len[a],
        "heading_diff": hd,
        "speed_ms": np.nan_to_num(pt.speed_ms[o], nan=0.0),
        "dt_s": np.nan_to_num(pt.dt[o], nan=0.0),
        "is_outlier": pt.is_outlier[o].astype(int),
        "seg_start": seg_start,
        "cum_gps_frac": cum_frac[o],
        "heading_diff_directed_rad": directed_heading_diff,
        "signed_cross_track_m": signed_cross_track,
        "previous_point_dist_m": previous_dist,
        "projected_speed_along_arc_ms": projected_speed,
        "projected_speed_error_ms": projected_speed_error,
        "gps_heading_change_rad": heading_change,
    })

    if labels is not None:
        df["y"] = (a == labels.label_arc_id[o]).astype(int)
        df["confident"] = labels.confident[o].astype(int)
    return df


def build_features(*args, **kwargs):  # noqa: ANN002, ANN003
    """Deprecated alias retained for docs; use :func:`build_feature_frame`."""
    raise NotImplementedError("Use build_feature_frame(pt, cand, labels) instead.")
