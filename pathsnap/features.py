"""Feature engineering (Phase 4 — not yet implemented).

Planned per (observation, candidate-arc) features:
    * geodesic/perpendicular distance observation->arc; projection fraction.
    * heading difference (GPS course vs arc bearing); speed; acceleration.
    * arc length; arc/node connectivity degree.
    * temporal context (Δt to neighbours; gap flag).
    * continuity vs previous observation's best candidate.

Scalers/encoders must be fit on the training partition only (leakage rule).
Out of scope for Phase 0.
"""

from __future__ import annotations


def build_features(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: feature matrix construction (Phase 4)."""
    raise NotImplementedError("Feature engineering is implemented in Phase 4 (not started).")
