"""Candidate road generation (Phase 3 — NOT yet implemented).

Planned responsibilities:
    * Build a per-trip spatial index (shapely STRtree / scipy KD-tree) over arc
      geometries — all-arc scoring is intractable (up to ~307k arcs/trip).
    * For each observation, return arcs within a radius r.
    * Choose r to minimise Candidate Miss Rate while keeping the set small.

Explicitly out of scope for Phase 0.
"""

from __future__ import annotations


def generate_candidates(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: per-observation candidate arcs (Phase 3)."""
    raise NotImplementedError(
        "Candidate generation is a Phase 3 task and has intentionally not started."
    )
