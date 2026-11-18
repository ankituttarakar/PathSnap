"""End-to-end prediction pipeline (Phase 10 — not yet implemented).

``predict(trip_id)`` will tie together cleaning, candidate generation, feature
engineering, the tuned classifier and HMM/Viterbi decoding to return a predicted
arc sequence for a trip. The FastAPI backend (Laptop 1) will call this function.

Out of scope for Phase 0.
"""

from __future__ import annotations


def predict(trip_id: str, *args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: full predict(trip_id) pipeline (Phase 10)."""
    raise NotImplementedError("The prediction pipeline is implemented in Phase 10 (not started).")
