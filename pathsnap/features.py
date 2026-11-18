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

    # Heading difference (undirected: fold to [0, pi/2] so arc direction ambiguity
    # does not dominate; direction is modelled later in the HMM).
    obs_head = pt.heading[o]
    arc_brg = pt.arc_bearing[a]
    hd = angle_diff(np.nan_to_num(obs_head, nan=arc_brg), arc_brg)
    hd = np.minimum(hd, np.pi - hd)

    seg = pt.segment_id
    seg_start = np.r_[1, (np.diff(seg) != 0).astype(int)][o]

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
    })

    if labels is not None:
        df["y"] = (a == labels.label_arc_id[o]).astype(int)
        df["confident"] = labels.confident[o].astype(int)
    return df


def build_features(*args, **kwargs):  # noqa: ANN002, ANN003
    """Deprecated alias retained for docs; use :func:`build_feature_frame`."""
    raise NotImplementedError("Use build_feature_frame(pt, cand, labels) instead.")
