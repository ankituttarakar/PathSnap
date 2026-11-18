"""Train-only GroupKFold and one final held-out HMM comparison.

The script fits only the existing Random Forest configuration. It builds OOF
probabilities for every train-trip candidate so Viterbi sees the complete GPS
sequence, while fitting on confident train candidate rows only. Student-test
features and labels are loaded only after CV, the model artifact, and decoder
configuration are fixed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathsnap.config import CONFIG
from pathsnap.cv import assert_no_group_overlap, group_kfold_split
from pathsnap.data_io import load_inference_trip
from pathsnap.evaluation import candidate_metrics, top1_accuracy
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.hmm import viterbi_decode
from pathsnap.models import build_model_zoo
from pathsnap.preprocess import preprocess_trip


def _positive_probability(estimator, features: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    classes = list(estimator.classes_)
    return probabilities[:, classes.index(1)]


def _route_continuity(
    predicted_arcs: np.ndarray,
    segment_ids: np.ndarray,
    network_arcs: np.ndarray,
) -> tuple[float | None, int, int]:
    pairs = [
        (observation - 1, observation)
        for observation in range(1, len(predicted_arcs))
        if (
            segment_ids[observation] == segment_ids[observation - 1]
            and predicted_arcs[observation - 1] >= 0
            and predicted_arcs[observation] >= 0
        )
    ]
    if not pairs:
        return None, 0, 0
    connected = sum(
        network_arcs[int(predicted_arcs[previous]), 1]
        == network_arcs[int(predicted_arcs[current]), 0]
        for previous, current in pairs
    )
    return connected / len(pairs), connected, len(pairs)


def _directed_transition_counts(
    predicted_arcs: np.ndarray,
    projected_positions: np.ndarray,
    segment_ids: np.ndarray,
    network_arcs: np.ndarray,
) -> tuple[int, int]:
    valid = disconnected = 0
    for observation in range(1, len(predicted_arcs)):
        previous_arc = int(predicted_arcs[observation - 1])
        current_arc = int(predicted_arcs[observation])
        if (
            previous_arc < 0
            or current_arc < 0
            or segment_ids[observation] != segment_ids[observation - 1]
        ):
            continue
        same_arc_forward = (
            previous_arc == current_arc
            and projected_positions[observation] + 1e-12
            >= projected_positions[observation - 1]
        )
        directly_connected = (
            network_arcs[previous_arc, 1] == network_arcs[current_arc, 0]
        )
        if same_arc_forward or directly_connected:
            valid += 1
        else:
            disconnected += 1
    return valid, disconnected


def _evaluate_sequence_partition(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    partition: str,
) -> tuple[dict, list[dict]]:
    """Evaluate classifier and Viterbi paths, scoring labels only after decoding."""
    frame = frame.reset_index(drop=True)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(frame) != len(probabilities) or not np.all(np.isfinite(probabilities)):
        raise ValueError("A finite probability is required for every candidate row")

    trip_results = []
    for trip_id in sorted(frame["trip_id"].unique()):
        row_indices = np.flatnonzero(frame["trip_id"].to_numpy() == trip_id)
        trip_frame = frame.iloc[row_indices].reset_index(drop=True)
        trip_probabilities = probabilities[row_indices]

        # Inference inputs are route-free. Ground-truth columns are deliberately
        # not passed to preprocessing or the decoder.
        raw_trip = load_inference_trip(str(trip_id), partition=partition)
        processed = preprocess_trip(raw_trip)
        selected_classifier_rows = np.full(processed.n_obs, -1, dtype=np.int64)
        candidate_scores = trip_frame.assign(_score=trip_probabilities)
        top_rows = candidate_scores.groupby("obs_idx", sort=True)["_score"].idxmax()
        for row in top_rows.to_numpy(dtype=np.int64):
            observation = int(trip_frame.iloc[row]["obs_idx"])
            selected_classifier_rows[observation] = row

        selected_classifier_arcs = np.full(processed.n_obs, -1, dtype=np.int64)
        selected_classifier_t = np.zeros(processed.n_obs, dtype=float)
        has_classifier = selected_classifier_rows >= 0
        selected_classifier_arcs[has_classifier] = trip_frame.iloc[
            selected_classifier_rows[has_classifier]
        ]["arc_id"].to_numpy(dtype=np.int64)
        selected_classifier_t[has_classifier] = trip_frame.iloc[
            selected_classifier_rows[has_classifier]
        ]["t"].to_numpy(dtype=float)

        decoder_started = time.perf_counter()
        decoded = viterbi_decode(
            obs_idx=trip_frame["obs_idx"].to_numpy(dtype=np.int64),
            arc_ids=trip_frame["arc_id"].to_numpy(dtype=np.int64),
            emission_probabilities=trip_probabilities,
            t_positions=trip_frame["t"].to_numpy(dtype=float),
            observation_xy=processed.xy,
            segment_ids=processed.segment_id,
            arcs=processed.arcs,
            arc_lengths=processed.arc_len,
        )
        decoder_seconds = time.perf_counter() - decoder_started
        selected_hmm_t = np.zeros(processed.n_obs, dtype=float)
        has_hmm = decoded.candidate_indices >= 0
        selected_hmm_t[has_hmm] = trip_frame.iloc[
            decoded.candidate_indices[has_hmm]
        ]["t"].to_numpy(dtype=float)

        # Labels are consulted only here, after the route-free decoder returns.
        confident_by_observation = (
            trip_frame.groupby("obs_idx", sort=True)["confident"].first().astype(bool)
        )
        confident_observations = [
            int(index) for index, value in confident_by_observation.items() if value
        ]
        true_arcs = (
            trip_frame.loc[trip_frame["y"] == 1]
            .groupby("obs_idx", sort=True)["arc_id"]
            .first()
            .to_dict()
        )
        candidate_hits = sum(
            observation in true_arcs for observation in confident_observations
        )
        classifier_correct = sum(
            selected_classifier_arcs[index] == int(true_arcs[index])
            for index in confident_observations
            if index in true_arcs
        )
        hmm_correct = sum(
            decoded.arc_ids[index] == int(true_arcs[index])
            for index in confident_observations
            if index in true_arcs
        )
        # A confident label missing from the candidate set is an incorrect
        # observation prediction and a candidate-recall miss.
        confident_count = len(confident_observations)
        classifier_correct_rate = (
            classifier_correct / confident_count if confident_count else None
        )
        hmm_correct_rate = hmm_correct / confident_count if confident_count else None
        classifier_continuity, classifier_connected, continuity_pairs = (
            _route_continuity(
                selected_classifier_arcs,
                processed.segment_id,
                processed.arcs,
            )
        )
        hmm_continuity, hmm_connected, hmm_pairs = _route_continuity(
            decoded.arc_ids,
            processed.segment_id,
            processed.arcs,
        )
        classifier_directed, classifier_disconnected = _directed_transition_counts(
            selected_classifier_arcs,
            selected_classifier_t,
            processed.segment_id,
            processed.arcs,
        )
        hmm_directed, hmm_disconnected = _directed_transition_counts(
            decoded.arc_ids,
            selected_hmm_t,
            processed.segment_id,
            processed.arcs,
        )

        candidate_observations = set(trip_frame["obs_idx"].astype(int).tolist())
        no_candidate_count = processed.n_obs - len(candidate_observations)
        trip_results.append({
            "trip_id": str(trip_id),
            "partition": partition,
            "total_observations": processed.n_obs,
            "confident_observations": confident_count,
            "candidate_rows": len(trip_frame),
            "candidate_misses": confident_count - candidate_hits,
            "candidate_recall": (
                candidate_hits / confident_count if confident_count else None
            ),
            "classifier_top1": classifier_correct_rate,
            "hmm_top1": hmm_correct_rate,
            "classifier_route_continuity": classifier_continuity,
            "hmm_route_continuity": hmm_continuity,
            "classifier_continuity_pairs": continuity_pairs,
            "hmm_continuity_pairs": hmm_pairs,
            "classifier_connected_pairs": classifier_connected,
            "hmm_connected_pairs": hmm_connected,
            "classifier_directed_transitions": classifier_directed,
            "classifier_disconnected_transitions": classifier_disconnected,
            "hmm_directed_transitions": hmm_directed,
            "hmm_disconnected_transitions": hmm_disconnected,
            "predicted_observations": int(np.count_nonzero(has_hmm)),
            "no_candidate_observations": no_candidate_count,
            "segment_resets": decoded.segment_resets,
            "disconnected_resets": decoded.disconnected_resets,
            "fallback_decisions": decoded.fallback_decisions,
            "hmm_runtime_s": decoder_seconds,
        })

    confident_total = sum(row["confident_observations"] for row in trip_results)
    all_observations = sum(row["total_observations"] for row in trip_results)
    candidate_hits_total = sum(
        row["confident_observations"] - row["candidate_misses"]
        for row in trip_results
    )
    classifier_correct_total = sum(
        round(row["classifier_top1"] * row["confident_observations"])
        for row in trip_results
        if row["classifier_top1"] is not None
    )
    hmm_correct_total = sum(
        round(row["hmm_top1"] * row["confident_observations"])
        for row in trip_results
        if row["hmm_top1"] is not None
    )
    classifier_directed = sum(
        row["classifier_directed_transitions"] for row in trip_results
    )
    classifier_disconnected = sum(
        row["classifier_disconnected_transitions"] for row in trip_results
    )
    hmm_directed = sum(row["hmm_directed_transitions"] for row in trip_results)
    hmm_disconnected = sum(
        row["hmm_disconnected_transitions"] for row in trip_results
    )

    def macro_trip_metric(name: str) -> float | None:
        values = [row[name] for row in trip_results if row[name] is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "partition": partition,
        "trip_count": len(trip_results),
        "total_observations": all_observations,
        "confident_observations": confident_total,
        "candidate_rows": int(sum(row["candidate_rows"] for row in trip_results)),
        "candidate_recall": (
            candidate_hits_total / confident_total if confident_total else None
        ),
        "classifier_top1": (
            classifier_correct_total / confident_total if confident_total else None
        ),
        "hmm_top1": hmm_correct_total / confident_total if confident_total else None,
        "classifier_route_continuity_macro_trip": macro_trip_metric(
            "classifier_route_continuity"
        ),
        "hmm_route_continuity_macro_trip": macro_trip_metric("hmm_route_continuity"),
        "classifier_directed_transition_validity": (
            classifier_directed / (classifier_directed + classifier_disconnected)
            if classifier_directed + classifier_disconnected else None
        ),
        "hmm_directed_transition_validity": (
            hmm_directed / (hmm_directed + hmm_disconnected)
            if hmm_directed + hmm_disconnected else None
        ),
        "classifier_disconnected_transitions": classifier_disconnected,
        "hmm_disconnected_transitions": hmm_disconnected,
        "decoded_observations": int(sum(row["predicted_observations"] for row in trip_results)),
        "decoded_observation_coverage": (
            sum(row["predicted_observations"] for row in trip_results) / all_observations
            if all_observations else None
        ),
        "no_candidate_observations": int(sum(
            row["no_candidate_observations"] for row in trip_results
        )),
        "no_candidate_percentage": (
            100.0 * sum(row["no_candidate_observations"] for row in trip_results)
            / all_observations if all_observations else None
        ),
        "segment_resets": int(sum(row["segment_resets"] for row in trip_results)),
        "disconnected_resets": int(sum(
            row["disconnected_resets"] for row in trip_results
        )),
        "fallback_decisions": int(sum(
            row["fallback_decisions"] for row in trip_results
        )),
        "hmm_runtime_total_s": float(sum(row["hmm_runtime_s"] for row in trip_results)),
        "hmm_runtime_mean_per_trip_s": float(np.mean(
            [row["hmm_runtime_s"] for row in trip_results]
        )) if trip_results else 0.0,
    }
    return summary, trip_results


def main() -> int:
    CONFIG.paths.ensure_output_dirs()
    processed = CONFIG.paths.data_processed
    train = joblib.load(processed / "features_train.pkl").reset_index(drop=True)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in train]
    if missing_columns:
        raise ValueError(
            "Train feature data is stale; rebuild only the train partition first. "
            f"Missing: {missing_columns}"
        )

    fit_mask = train["confident"].to_numpy(dtype=bool)
    X_all = train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_all = train["y"].to_numpy(dtype=int)
    groups_all = train["trip_id"].to_numpy()
    oof = np.full(len(train), np.nan, dtype=float)
    fold_by_trip: dict[str, int] = {}
    fold_metrics = []
    folds = list(group_kfold_split(groups_all, CONFIG.cv_folds))
    for fold_number, (train_rows, valid_rows) in enumerate(folds, start=1):
        train_trip_ids = np.unique(groups_all[train_rows])
        valid_trip_ids = np.unique(groups_all[valid_rows])
        assert_no_group_overlap(train_trip_ids, valid_trip_ids)
        fit_rows = train_rows[fit_mask[train_rows]]
        if not len(fit_rows):
            raise ValueError(f"Fold {fold_number} has no confident training rows")

        estimator = build_model_zoo(CONFIG.seed)["random_forest"]
        estimator.fit(X_all[fit_rows], y_all[fit_rows])
        oof[valid_rows] = _positive_probability(estimator, X_all[valid_rows])
        for trip_id in valid_trip_ids:
            fold_by_trip[str(trip_id)] = fold_number

        scored_rows = valid_rows[fit_mask[valid_rows]]
        fold_metrics.append({
            "fold": fold_number,
            "train_trips": int(len(train_trip_ids)),
            "validation_trips": int(len(valid_trip_ids)),
            "confident_candidate_rows": int(len(scored_rows)),
            **candidate_metrics(y_all[scored_rows], oof[scored_rows]),
            "observation_top1": top1_accuracy(
                train.iloc[scored_rows].reset_index(drop=True), oof[scored_rows]
            ),
        })
        print(
            f"fold {fold_number}/{len(folds)}: "
            f"candidate_macro_f1={fold_metrics[-1]['macro_f1']:.4f} "
            f"observation_top1={fold_metrics[-1]['observation_top1']:.4f}"
        )

    if not np.all(np.isfinite(oof)):
        raise ValueError("OOF probabilities are missing for one or more train candidates")
    cv_scored = np.flatnonzero(fit_mask)
    cv_candidate_metrics = candidate_metrics(y_all[cv_scored], oof[cv_scored])
    cv_top1 = top1_accuracy(train.iloc[cv_scored].reset_index(drop=True), oof[cv_scored])
    cv_sequence, cv_trip_rows = _evaluate_sequence_partition(train, oof, "train")
    for row in cv_trip_rows:
        row["fold"] = fold_by_trip[row["trip_id"]]
    cv_fold_sequence = []
    for fold_number in range(1, len(folds) + 1):
        trips_in_fold = [row for row in cv_trip_rows if row["fold"] == fold_number]
        valid_count = sum(row["confident_observations"] for row in trips_in_fold)
        cv_fold_sequence.append({
            "fold": fold_number,
            "validation_trips": len(trips_in_fold),
            "hmm_observation_top1": (
                sum(round(row["hmm_top1"] * row["confident_observations"])
                    for row in trips_in_fold if row["hmm_top1"] is not None) / valid_count
                if valid_count else None
            ),
            "classifier_observation_top1": (
                sum(round(row["classifier_top1"] * row["confident_observations"])
                    for row in trips_in_fold if row["classifier_top1"] is not None) / valid_count
                if valid_count else None
            ),
            "hmm_route_continuity_macro_trip": (
                float(np.mean([row["hmm_route_continuity"] for row in trips_in_fold
                               if row["hmm_route_continuity"] is not None]))
                if any(row["hmm_route_continuity"] is not None for row in trips_in_fold)
                else None
            ),
        })

    fit_rows = np.flatnonzero(fit_mask)
    final_model = build_model_zoo(CONFIG.seed)["random_forest"]
    refit_started = time.perf_counter()
    final_model.fit(X_all[fit_rows], y_all[fit_rows])
    refit_seconds = time.perf_counter() - refit_started
    model_path = CONFIG.paths.models / "model_random_forest.joblib"
    joblib.dump(final_model, model_path, compress=3)
    schema_path = CONFIG.paths.models / "feature_columns.json"
    with schema_path.open("w", encoding="utf-8") as file:
        json.dump({
            "features": FEATURE_COLUMNS,
            "best_model_by_cv_macro_f1": "random_forest",
        }, file, indent=2)

    # Freeze all model and decoder decisions before loading student-test labels.
    CONFIG.paths.outputs_experiments.mkdir(parents=True, exist_ok=True)
    run_config = {
        "seed": CONFIG.seed,
        "cv_folds": CONFIG.cv_folds,
        "group_key": CONFIG.group_key,
        "model": "random_forest",
        "model_params": str(final_model),
        "features": FEATURE_COLUMNS,
        "decoder": "directed_arc_hmm_viterbi",
        "transition_rule": (
            "same directed arc with forward progress, or directly connected "
            "directed arcs; no connector arcs"
        ),
        "transition_log_score": (
            "-abs(network_distance - gps_displacement) / "
            "max(gps_displacement, 1 metre), row-normalized over valid successors"
        ),
        "student_test_used_for_training_or_selection": False,
    }
    with (CONFIG.paths.outputs_experiments / "hmm_run_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(run_config, file, indent=2)

    # The only student-test load occurs after train-only CV and model freezing.
    student_all = joblib.load(processed / "features_student_test.pkl").reset_index(drop=True)
    missing_student_columns = [
        column for column in FEATURE_COLUMNS if column not in student_all
    ]
    if missing_student_columns:
        raise ValueError(
            "Student-test feature data is stale; build that partition once for final "
            f"evaluation. Missing: {missing_student_columns}"
        )
    assert_no_group_overlap(
        train["trip_id"].unique(),
        student_all["trip_id"].unique(),
    )
    student_confident = student_all["confident"].to_numpy(dtype=bool)
    X_student_all = student_all[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    student_infer_started = time.perf_counter()
    student_probabilities = _positive_probability(final_model, X_student_all)
    student_infer_seconds = time.perf_counter() - student_infer_started
    student_scored = np.flatnonzero(student_confident)
    student_candidate_metrics = candidate_metrics(
        student_all.iloc[student_scored]["y"].to_numpy(dtype=int),
        student_probabilities[student_scored],
    )
    student_top1 = top1_accuracy(
        student_all.iloc[student_scored].reset_index(drop=True),
        student_probabilities[student_scored],
    )
    student_sequence, student_trip_rows = _evaluate_sequence_partition(
        student_all,
        student_probabilities,
        "student_test",
    )
    student_y = student_all.iloc[student_scored]["y"].to_numpy(dtype=int)
    student_candidate_pred = (
        student_probabilities[student_scored] >= 0.5
    ).astype(int)
    student_sequence["candidate_confusion_matrix_labels_0_1"] = (
        confusion_matrix(student_y, student_candidate_pred, labels=[0, 1]).tolist()
    )
    student_sequence["candidate_classification_report"] = classification_report(
        student_y,
        student_candidate_pred,
        labels=[0, 1],
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    student_sequence["classifier_candidate_metrics"] = student_candidate_metrics
    student_sequence["classifier_model_predict_proba_s"] = student_infer_seconds

    cv_summary = {
        "candidate_metrics": cv_candidate_metrics,
        "observation_top1": cv_top1,
        "sequence_metrics": cv_sequence,
        "folds": fold_metrics,
        "hmm_sequence_metrics_by_fold": cv_fold_sequence,
    }
    report = {
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "cv_train_only": cv_summary,
        "student_test_final_held_out": student_sequence,
        "model_refit_seconds": refit_seconds,
    }
    report_path = CONFIG.paths.outputs_experiments / "hmm_evaluation.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    pd.DataFrame(cv_trip_rows).to_csv(
        CONFIG.paths.outputs_experiments / "hmm_cv_trip_metrics.csv", index=False
    )
    pd.DataFrame(student_trip_rows).to_csv(
        CONFIG.paths.outputs_experiments / "hmm_student_trip_metrics.csv", index=False
    )

    print("\n=== TRAIN-ONLY OOF ===")
    print(json.dumps({
        "candidate_metrics": cv_candidate_metrics,
        "observation_top1_classifier": cv_top1,
        "observation_top1_hmm": cv_sequence["hmm_top1"],
        "route_continuity_classifier": cv_sequence[
            "classifier_route_continuity_macro_trip"
        ],
        "route_continuity_hmm": cv_sequence["hmm_route_continuity_macro_trip"],
        "candidate_recall": cv_sequence["candidate_recall"],
    }, indent=2))
    print("\n=== FINAL HELD-OUT STUDENT TEST ===")
    print(json.dumps(student_sequence, indent=2))
    print(f"\nHMM evaluation artifacts: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
