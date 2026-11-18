"""Train and compare >=5 models for PathSnap candidate classification (Phase 5).

Pipeline:
  1. Load the confident candidate rows for TRAIN (features_train.pkl).
  2. For each model: group-aware (trip_id) 5-fold OOF predictions -> candidate
     metrics (macro F1, log loss, precision/recall) + observation Top-1 accuracy.
  3. Refit on ALL train, evaluate on STUDENT_TEST (dev comparison only):
     Top-1, macro F1, log loss, route continuity.
  4. Save comparison table (CSV/JSON), comparison plots, run config/seeds, and
     the refit model artifacts.

Rules enforced: group-aware CV (no trip split across folds); student_test is
NEVER used for fitting; all metrics come from real predictions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pathsnap.config import CONFIG  # noqa: E402
from pathsnap.cv import assert_no_group_overlap, group_kfold_split  # noqa: E402
from pathsnap.data_io import load_trip  # noqa: E402
from pathsnap.evaluation import candidate_metrics, route_continuity, top1_accuracy  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS  # noqa: E402
from pathsnap.logging_utils import get_logger  # noqa: E402
from pathsnap.models import MODEL_FAMILIES, build_model_zoo  # noqa: E402

logger = get_logger("pathsnap.train", log_file="run_training.log")


def _pos_proba(est, X) -> np.ndarray:
    """Positive-class (y==1) probability column."""
    p = est.predict_proba(X)
    pos = list(est.classes_).index(1)
    return p[:, pos]


def main() -> int:
    CONFIG.paths.ensure_output_dirs()
    exp = CONFIG.paths.outputs_experiments
    mdir = CONFIG.paths.models
    seed = CONFIG.seed

    # ---- Load confident candidate rows ----
    train = joblib.load(CONFIG.paths.data_processed / "features_train.pkl")
    stud = joblib.load(CONFIG.paths.data_processed / "features_student_test.pkl")
    train = train[train["confident"] == 1].reset_index(drop=True)
    stud = stud[stud["confident"] == 1].reset_index(drop=True)
    assert_no_group_overlap(train["trip_id"].unique(), stud["trip_id"].unique())

    Xtr = train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    ytr = train["y"].to_numpy(dtype=int)
    groups = train["trip_id"].to_numpy()
    Xst = stud[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    yst = stud["y"].to_numpy(dtype=int)
    logger.info("train rows=%d (pos=%d) | student rows=%d (pos=%d)",
                len(train), int(ytr.sum()), len(stud), int(yst.sum()))

    # arcs for route continuity on student_test
    arcs_by_trip = {t: load_trip(t, "student_test").arcs for t in stud["trip_id"].unique()}

    folds = list(group_kfold_split(groups, CONFIG.cv_folds))
    results, run_cfg = [], {"seed": seed, "cv_folds": CONFIG.cv_folds,
                            "features": FEATURE_COLUMNS, "models": {}}

    for name, model in build_model_zoo(seed).items():
        logger.info("=== %s ===", name)
        oof = np.full(len(train), np.nan)
        t_fit = 0.0
        for fi, (tr, va) in enumerate(folds):
            est = build_model_zoo(seed)[name]
            t0 = time.perf_counter()
            est.fit(Xtr[tr], ytr[tr])
            t_fit += time.perf_counter() - t0
            oof[va] = _pos_proba(est, Xtr[va])
            logger.info("  fold %d done", fi + 1)
        cv = candidate_metrics(ytr, oof)
        cv_top1 = top1_accuracy(train, oof)

        # Refit on all train, evaluate on student_test (dev only).
        final = build_model_zoo(seed)[name]
        t0 = time.perf_counter(); final.fit(Xtr, ytr); refit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p_st = _pos_proba(final, Xst); infer_s = time.perf_counter() - t0
        st = candidate_metrics(yst, p_st)
        st_top1 = top1_accuracy(stud, p_st)
        st_cont = route_continuity(stud, p_st, arcs_by_trip)

        joblib.dump(final, mdir / f"model_{name}.joblib", compress=3)
        row = {
            "model": name, "family": MODEL_FAMILIES[name],
            "cv_macro_f1": cv["macro_f1"], "cv_top1": cv_top1, "cv_log_loss": cv["log_loss"],
            "cv_precision": cv["precision"], "cv_recall": cv["recall"],
            "student_top1": st_top1, "student_macro_f1": st["macro_f1"],
            "student_log_loss": st["log_loss"], "student_route_continuity": st_cont,
            "fit_time_cv_s": round(t_fit, 1), "refit_time_s": round(refit_s, 1),
            "infer_time_student_s": round(infer_s, 3),
        }
        results.append(row)
        run_cfg["models"][name] = {"family": MODEL_FAMILIES[name], "params": str(final)}
        logger.info("  cv_macro_f1=%.4f cv_top1=%.4f student_top1=%.4f cont=%.4f",
                    cv["macro_f1"], cv_top1, st_top1, st_cont)
        print(f"{name:24s} cvF1={cv['macro_f1']:.3f} cvTop1={cv_top1:.3f} "
              f"stTop1={st_top1:.3f} stF1={st['macro_f1']:.3f} "
              f"logloss={st['log_loss']:.3f} cont={st_cont:.3f} fit={t_fit:.0f}s")

    res = pd.DataFrame(results).sort_values("cv_macro_f1", ascending=False)
    res.to_csv(exp / "model_comparison.csv", index=False)
    with open(exp / "model_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    with open(exp / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(run_cfg, fh, indent=2)

    # ---- Plots (from real results) ----
    def _bar(col, title, fname):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        d = res.sort_values(col, ascending=False)
        ax.bar(d["model"], d[col], color="#3b76af")
        ax.set_title(title); ax.set_ylabel(col); plt.xticks(rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(exp / fname, dpi=110); plt.close(fig)

    _bar("cv_macro_f1", "CV Macro F1 (group-aware, train)", "cv_macro_f1.png")
    _bar("student_top1", "Student-test Top-1 accuracy", "student_top1.png")
    _bar("student_macro_f1", "Student-test Macro F1", "student_macro_f1.png")
    _bar("cv_log_loss", "CV Log Loss (lower better)", "cv_log_loss.png")

    best = res.iloc[0]["model"]
    with open(mdir / "feature_columns.json", "w", encoding="utf-8") as fh:
        json.dump({"features": FEATURE_COLUMNS, "best_model_by_cv_macro_f1": best}, fh, indent=2)

    print("\n=== SUMMARY (sorted by CV Macro F1) ===")
    print(res[["model", "cv_macro_f1", "cv_top1", "student_top1",
               "student_macro_f1", "student_log_loss", "student_route_continuity"]]
          .to_string(index=False))
    print(f"\nBest by CV Macro F1: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
