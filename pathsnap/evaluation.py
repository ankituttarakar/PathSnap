"""Evaluation metrics for candidate classification and routing (Phase 5).

Two levels:
  * candidate-level (binary): accuracy, precision, recall, macro F1, log loss —
    over all (obs, candidate) rows.
  * observation-level: Top-1 accuracy — per observation, is the highest-scoring
    candidate the true arc? Plus route continuity over the Top-1 arc sequence.

All metrics are computed from actual model outputs; none are fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Binary candidate-level metrics at threshold 0.5 (+ log loss)."""
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": float((pred == y_true).mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    try:
        out["log_loss"] = float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except ValueError:
        out["log_loss"] = float("nan")
    return out


def top1_accuracy(df: pd.DataFrame, proba: np.ndarray) -> float:
    """Observation-level Top-1 accuracy.

    ``df`` must contain ``trip_id``, ``obs_idx`` and ``y``; ``proba`` is the
    per-row positive-class probability. For each observation the highest-proba
    candidate is chosen; the observation is correct if that candidate is the
    true arc (``y == 1``).
    """
    tmp = df[["trip_id", "obs_idx", "y"]].copy()
    tmp["p"] = proba
    idx = tmp.groupby(["trip_id", "obs_idx"])["p"].idxmax()
    chosen = tmp.loc[idx]
    return float(chosen["y"].mean())


def route_continuity(df: pd.DataFrame, proba: np.ndarray, arcs_by_trip: dict) -> float:
    """Fraction of consecutive Top-1 predicted arcs that are node-connected.

    For each trip the Top-1 arc per observation is taken in observation order;
    continuity = mean over consecutive pairs of [to_node(prev) == from_node(next)].
    ``arcs_by_trip`` maps trip_id -> arcs array (M,2).
    """
    tmp = df[["trip_id", "obs_idx", "arc_id"]].copy()
    tmp["p"] = proba
    idx = tmp.groupby(["trip_id", "obs_idx"])["p"].idxmax()
    chosen = tmp.loc[idx].sort_values(["trip_id", "obs_idx"])
    fracs = []
    for tid, g in chosen.groupby("trip_id"):
        arcs = arcs_by_trip[tid]
        seq = g["arc_id"].to_numpy()
        if len(seq) < 2:
            continue
        to_prev = arcs[seq[:-1], 1]
        from_next = arcs[seq[1:], 0]
        fracs.append(float(np.mean(to_prev == from_next)))
    return float(np.mean(fracs)) if fracs else float("nan")
