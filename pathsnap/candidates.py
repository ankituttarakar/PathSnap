"""Candidate road generation via spatial indexing (Phase 3).

For each GPS observation we retrieve a small set of nearby arcs instead of
scanning the whole network (up to ~307k arcs/trip). A KD-tree over arc midpoints
gives an initial neighbour set; those are then refined by the true
point-to-segment distance and capped.

Candidate generation uses ONLY observation coordinates and network geometry —
never the ``.route`` — so it is valid at inference time. Labels are used solely
to *measure* candidate recall / miss rate, not to build the candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from pathsnap.geo import project_point_to_segments
from pathsnap.preprocess import ProcessedTrip

K_CANDIDATES = 8         # max candidates kept per observation
QUERY_K = 40             # nearest arc-midpoints queried before refinement
RADIUS_M = 60.0          # keep refined candidates within this perpendicular distance


@dataclass
class Candidates:
    """Flattened candidate table for one trip (one row per obs-candidate)."""

    obs_idx: np.ndarray      # (R,) observation index
    arc_id: np.ndarray       # (R,) candidate arc id (original)
    dist_m: np.ndarray       # (R,) perpendicular distance obs->arc (m)
    t: np.ndarray            # (R,) fractional position along the arc [0,1]
    rank: np.ndarray         # (R,) distance rank within the observation (0=closest)
    n_cand: np.ndarray       # (R,) number of candidates for that observation


def generate_candidates(
    pt: ProcessedTrip, k: int = K_CANDIDATES, query_k: int = QUERY_K, radius_m: float = RADIUS_M
) -> Candidates:
    """Generate up to ``k`` candidate arcs per observation using a KD-tree."""
    tree = cKDTree(pt.arc_mid)
    qk = int(min(query_k, pt.n_arcs))
    _, nn = tree.query(pt.xy, k=qk)       # (N, qk) nearest arc indices by midpoint
    nn = np.asarray(nn).reshape(pt.n_obs, qk)

    obs_list, arc_list, dist_list, t_list, rank_list, ncand_list = [], [], [], [], [], []
    for i in range(pt.n_obs):
        cand = np.unique(nn[i])
        d, t, _, _ = project_point_to_segments(
            pt.xy[i, 0], pt.xy[i, 1],
            pt.arc_ax[cand], pt.arc_ay[cand], pt.arc_bx[cand], pt.arc_by[cand],
        )
        order = np.argsort(d)
        cand, d, t = cand[order], d[order], t[order]
        keep = d <= radius_m
        if keep.sum() < 1:          # always keep at least the closest arc
            keep[0] = True
        cand, d, t = cand[keep][:k], d[keep][:k], t[keep][:k]
        m = len(cand)
        obs_list.append(np.full(m, i))
        arc_list.append(cand)
        dist_list.append(d)
        t_list.append(t)
        rank_list.append(np.arange(m))
        ncand_list.append(np.full(m, m))

    return Candidates(
        obs_idx=np.concatenate(obs_list).astype(np.int64),
        arc_id=np.concatenate(arc_list).astype(np.int64),
        dist_m=np.concatenate(dist_list),
        t=np.concatenate(t_list),
        rank=np.concatenate(rank_list).astype(np.int64),
        n_cand=np.concatenate(ncand_list).astype(np.int64),
    )


def candidate_recall(cand: Candidates, label_arc_id: np.ndarray, confident: np.ndarray) -> dict:
    """Recall / miss-rate of the candidate set against confident labels.

    An observation is "hit" if its (confident) label arc appears among its
    candidates. Returns counts and rates; the miss rate is the fraction of
    confident observations whose true arc is absent from the candidate set.
    """
    # Build per-observation candidate arc sets.
    by_obs: dict[int, set] = {}
    for o, a in zip(cand.obs_idx.tolist(), cand.arc_id.tolist()):
        by_obs.setdefault(o, set()).add(a)
    hit = miss = 0
    for i in np.where(confident)[0]:
        lab = int(label_arc_id[i])
        if lab < 0:
            continue
        if lab in by_obs.get(int(i), set()):
            hit += 1
        else:
            miss += 1
    total = hit + miss
    return {
        "confident_obs": int(total),
        "recall": float(hit / total) if total else float("nan"),
        "miss_rate": float(miss / total) if total else float("nan"),
        "hits": int(hit),
        "misses": int(miss),
    }
