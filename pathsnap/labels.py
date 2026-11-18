"""Observation -> arc label derivation (Phase 2 — NOT yet implemented).

This is the project's highest-leverage decision (audit risk R1): the raw data
does not provide a direct observation->arc label. The true arc for each GPS
observation must be derived from the ordered ``.route`` polyline (e.g. by a
monotonic arc-length projection of each observation onto the route geometry).

Per the Phase 0 instructions, labels are intentionally NOT derived yet. The
rule will be designed and visually validated during Phase 2 before any model is
trained.
"""

from __future__ import annotations


def derive_observation_arc_labels(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: derive per-observation ground-truth arc_id (Phase 2)."""
    raise NotImplementedError(
        "Label derivation is a Phase 2 task and has intentionally not started."
    )
