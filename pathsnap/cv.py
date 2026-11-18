"""Group-aware cross-validation helpers (Phase 6 — not yet implemented).

Planned responsibilities:
    * Wrap scikit-learn ``GroupKFold`` / ``StratifiedGroupKFold`` keyed on the
      trip ID so that no trip's observations are split across folds (audit risk
      R2 — the hardest leakage rule in this project).
    * Provide a guard that asserts no group appears in two folds.

Out of scope for Phase 0; the group key is already fixed in
``pathsnap.config.ExperimentConfig.group_key``.
"""

from __future__ import annotations


def group_kfold_split(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: trip-grouped CV split generator (Phase 6)."""
    raise NotImplementedError("Group-aware CV is implemented in Phase 6 (not started).")
