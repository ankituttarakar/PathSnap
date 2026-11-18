"""Geographic utilities: per-trip projection and segment geometry (Phase 2).

The dataset is globally distributed, so a single projected CRS is inappropriate
(see docs/DATASET_SCHEMA.md §5). For each trip we build a local metric
projection (the appropriate UTM zone for the trip centroid) so that distances,
headings and point-to-segment projections are computed in metres on a plane.

All functions are pure/geometric and never touch the raw files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

EARTH_RADIUS_M = 6_371_000.0


def utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Projector:
    """Local planar projection for one trip (WGS84 -> local UTM, metres)."""

    epsg: int
    _fwd: Transformer

    @classmethod
    def for_points(cls, lon: np.ndarray, lat: np.ndarray) -> "Projector":
        """Build a projector using the centroid of the given coordinates."""
        epsg = utm_epsg(float(np.mean(lon)), float(np.mean(lat)))
        fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        return cls(epsg=epsg, _fwd=fwd)

    def xy(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Project lon/lat arrays to an (N, 2) array of metres [x, y]."""
        x, y = self._fwd.transform(np.asarray(lon), np.asarray(lat))
        return np.column_stack([np.asarray(x), np.asarray(y)])


def haversine_m(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Vectorised great-circle distance in metres (mirrors pathsnap.eda)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def project_point_to_segments(
    px: float, py: float, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray
):
    """Project one point onto many segments A->B (all in planar metres).

    Returns (distance, t, footx, footy) arrays where ``t`` in [0, 1] is the
    fractional position of the foot of the perpendicular along each segment.
    """
    abx = bx - ax
    aby = by - ay
    seg_len2 = abx * abx + aby * aby
    seg_len2 = np.where(seg_len2 == 0.0, 1e-12, seg_len2)
    t = ((px - ax) * abx + (py - ay) * aby) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    fx = ax + t * abx
    fy = ay + t * aby
    dist = np.hypot(px - fx, py - fy)
    return dist, t, fx, fy


def segment_bearings(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Bearing (radians) of each planar segment A->B, measured from +x (east)."""
    return np.arctan2(by - ay, bx - ax)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between two angles (radians), in [0, pi]."""
    d = np.abs(a - b) % (2.0 * np.pi)
    return np.where(d > np.pi, 2.0 * np.pi - d, d)


# Phase-2 stub retained name for backward references in docs.
def project_trip(*args, **kwargs):  # noqa: ANN002, ANN003
    """Deprecated placeholder; use :class:`Projector`."""
    raise NotImplementedError("Use Projector.for_points(...) instead of project_trip().")
