"""Group-aware cross-validation keyed on trip_id (Phase 6, used from Phase 5).

No trip's observations may be split across folds (audit risk R2). This module
wraps scikit-learn's grouped splitters and asserts that property so leakage
cannot pass silently.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold

from pathsnap.config import CONFIG


def group_kfold_split(
    groups: np.ndarray, n_splits: int | None = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) folds grouped by ``groups`` (trip_id).

    Asserts that no group appears in both the train and validation side of any
    fold.
    """
    n_splits = n_splits or CONFIG.cv_folds
    gkf = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(groups), 1))
    for tr, va in gkf.split(dummy_x, groups=groups):
        tr_groups = set(np.asarray(groups)[tr].tolist())
        va_groups = set(np.asarray(groups)[va].tolist())
        assert tr_groups.isdisjoint(va_groups), "Leakage: a trip spans train and val folds"
        yield tr, va


def assert_no_group_overlap(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
    """Assert two group sets are disjoint (e.g. train vs student_test)."""
    overlap = set(np.asarray(train_groups).tolist()) & set(np.asarray(test_groups).tolist())
    assert not overlap, f"Group overlap between partitions: {sorted(overlap)[:5]}"
