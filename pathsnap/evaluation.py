"""Evaluation metrics (Phases 5/8 — not yet implemented).

Planned metrics (docs: Models and Training Guidelines §5):
    * Observation-level Top-1 Accuracy (proposed primary metric).
    * Macro F1-score, Precision, Recall, Log Loss, Confusion Matrix.
    * Candidate Miss Rate (true arc absent from the candidate set).
    * Route Continuity and sequence-level performance.
    * Training / inference time.

All metrics will be computed from actual predictions only — never fabricated.
Out of scope for Phase 0.
"""

from __future__ import annotations


def top1_accuracy(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: observation-level Top-1 accuracy (Phase 5)."""
    raise NotImplementedError("Evaluation metrics are implemented in Phase 5 (not started).")
