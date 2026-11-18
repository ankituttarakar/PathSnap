"""Build the candidate feature dataset for train and student_test (Phase 2-4).

Runs preprocessing -> label derivation -> candidate generation -> feature
engineering over every trip in a partition and writes a single parquet/CSV
feature table plus a per-trip diagnostics table (label coverage, candidate
recall/miss-rate). student_test is processed with the SAME code but written to a
separate file and never mixed into train.

Usage:
    python scripts/build_dataset.py            # builds train + student_test
No raw data is modified; labels are derived from .route for training only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathsnap.candidates import candidate_recall, generate_candidates  # noqa: E402
from pathsnap.config import CONFIG  # noqa: E402
from pathsnap.data_io import load_trip, list_trips  # noqa: E402
from pathsnap.features import build_feature_frame  # noqa: E402
from pathsnap.labels import derive_labels  # noqa: E402
from pathsnap.logging_utils import get_logger  # noqa: E402
from pathsnap.preprocess import preprocess_trip  # noqa: E402

logger = get_logger("pathsnap.build_dataset", log_file="build_dataset.log")


def build_partition(partition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature frame and per-trip diagnostics for a partition."""
    frames, diag = [], []
    for tid in list_trips(partition):
        trip = load_trip(tid, partition)
        pt = preprocess_trip(trip)
        labels = derive_labels(pt, trip.route)
        cand = generate_candidates(pt)
        rec = candidate_recall(cand, labels.label_arc_id, labels.confident)
        df = build_feature_frame(pt, cand, labels)
        frames.append(df)
        diag.append({
            "trip_id": tid, "partition": partition, "n_obs": pt.n_obs,
            "n_segments": int(pt.segment_id.max() + 1), "n_outliers": int(pt.is_outlier.sum()),
            "label_coverage": labels.coverage,
            "median_proj_dist_m": float(np.nanmedian(labels.proj_dist_m)),
            "candidate_recall": rec["recall"], "candidate_miss_rate": rec["miss_rate"],
            "n_candidate_rows": int(len(df)),
        })
        logger.info("%s/%s: cov=%.3f recall=%.3f rows=%d",
                    partition, tid, labels.coverage, rec["recall"], len(df))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(diag)


def main() -> int:
    CONFIG.paths.ensure_output_dirs()
    out = CONFIG.paths.data_processed
    out.mkdir(parents=True, exist_ok=True)
    eda = CONFIG.paths.outputs_eda

    for partition in ("train", "student_test"):
        feats, diag = build_partition(partition)
        joblib.dump(feats, out / f"features_{partition}.pkl", compress=3)
        diag.to_csv(eda / f"dataset_diagnostics_{partition}.csv", index=False)
        conf = feats[feats["confident"] == 1]
        logger.info(
            "%s: %d trips, %d candidate rows (%d confident), positives=%d, "
            "mean label coverage=%.3f, mean candidate recall=%.3f",
            partition, diag.shape[0], len(feats), len(conf), int(conf["y"].sum()),
            diag["label_coverage"].mean(), diag["candidate_recall"].mean(),
        )
        print(f"[{partition}] trips={diag.shape[0]} rows={len(feats)} "
              f"confident_rows={len(conf)} positives={int(conf['y'].sum())} "
              f"label_cov={diag['label_coverage'].mean():.3f} "
              f"cand_recall={diag['candidate_recall'].mean():.3f} "
              f"cand_miss={diag['candidate_miss_rate'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
