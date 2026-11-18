"""Geographic helpers (Phase 2 — not yet implemented).

Planned responsibilities:
    * Per-trip projection to a local CRS (e.g. local UTM via pyproj) because the
      dataset is globally distributed — a single global CRS would distort
      distances (docs/DATASET_SCHEMA.md §5).
    * Geodesic (haversine) distance utilities.
    * Projection of a GPS point onto an arc / onto the ordered route polyline.

No geographic computation is implemented yet; coordinate handling starts in
Phase 2 to avoid committing to a distance metric before EDA validates it.
"""

from __future__ import annotations


def project_trip(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: per-trip projection to a local CRS (Phase 2)."""
    raise NotImplementedError("Projection is implemented in Phase 2 (not started).")
