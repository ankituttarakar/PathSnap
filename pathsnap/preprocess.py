"""Preprocessing (Phase 2 — not yet implemented).

Planned responsibilities:
    * GPS outlier removal (speed/jump thresholds) — 10 train trips contain
      physically impossible jumps up to ~887 km (docs/DATASET_SCHEMA.md §6).
    * Trip segmentation at large timestamp gaps (max observed ~42 h).
    * Caching of cleaned per-trip tables under data/processed/.

Note: label derivation (observation -> arc) is deliberately NOT part of Phase 0
and is handled in pathsnap.labels during Phase 2.
"""

from __future__ import annotations


def clean_trip(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: outlier filtering + gap segmentation (Phase 2)."""
    raise NotImplementedError("Preprocessing is implemented in Phase 2 (not started).")
