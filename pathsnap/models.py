
"""Model zoo and training harness (Phase 5).

The map-matching task is framed as **per-candidate probabilistic
classification**: for each (observation, candidate-arc) row the model predicts
P(candidate is the correct arc). At inference the highest-probability candidate
per observation is the Top-1 prediction.

Five models from distinct algorithm families are provided. Linear/probabilistic
models are wrapped in a StandardScaler; tree/boosting models are scale-invariant
and are left unscaled. All randomness is seeded from ``CONFIG.seed``.

Bayesian HPO (Phase 7) and HMM/Viterbi (Phase 8) are intentionally NOT here.
"""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.config import CONFIG

# Algorithm family tags for the report's comparison table.
MODEL_FAMILIES: Dict[str, str] = {
    "logistic_regression": "Linear / probabilistic",
    "gaussian_nb": "Probabilistic (generative)",
    "random_forest": "Ensemble (bagging)",
    "extra_trees": "Ensemble (randomised trees)",
    "hist_gradient_boosting": "Boosting",
}


def build_model_zoo(seed: int | None = None) -> Dict[str, Pipeline]:
    """Return the five candidate models, each as a fitted-ready estimator.

    Tree/boosting models are capped (depth / estimators) to stay trainable on
    ~1.1M rows within a reasonable time; these are the *initial* configurations
    recorded for the untuned comparison (Phase 5), before Bayesian HPO (Phase 7).
    """
    seed = CONFIG.seed if seed is None else seed
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]),
        "gaussian_nb": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=80, max_depth=18, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=80, max_depth=18, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=None,
            l2_regularization=1.0, random_state=seed,
        ),
    }


def run_bayesian_optimisation(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: Optuna Bayesian HPO on the chosen model (Phase 7)."""
    raise NotImplementedError("Bayesian HPO is implemented in Phase 7 (not started).")
