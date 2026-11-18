"""Model experimentation (Phase 5) and Bayesian HPO (Phase 7) — not implemented.

Planned responsibilities:
    * A model zoo spanning >=5 algorithm families (linear/probabilistic,
      instance-based, ensemble, boosting, optionally MLP) framed as per-candidate
      probabilistic classification.
    * A training/evaluation harness using trip-grouped CV with recorded configs,
      seeds and timings.
    * Optuna-based Bayesian hyperparameter optimisation of the chosen model
      (Phase 7), evaluated with group-aware CV on the 80 training trips only.

No model is trained in Phase 0 — doing so is explicitly prohibited here.
"""

from __future__ import annotations


def build_model_zoo(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: construct the comparison model set (Phase 5)."""
    raise NotImplementedError("Model experimentation is implemented in Phase 5 (not started).")


def run_bayesian_optimisation(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: Optuna Bayesian HPO on the chosen model (Phase 7)."""
    raise NotImplementedError("Bayesian HPO is implemented in Phase 7 (not started).")
