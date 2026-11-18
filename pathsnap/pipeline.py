"""Batch classifier and HMM inference for a complete PathSnap trip.

The inference path reads only ``.track``, ``.nodes`` and ``.arcs``. It does not
load route labels. ``cum_gps_frac`` uses the complete trip, so inference is
batch-only, not streaming.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from pathsnap.candidates import generate_candidates
from pathsnap.config import CONFIG
from pathsnap.data_io import load_inference_trip
from pathsnap.features import FEATURE_COLUMNS, build_feature_frame
from pathsnap.hmm import viterbi_decode
from pathsnap.preprocess import preprocess_trip

CLASSIFIER_DECODER = "classifier_only"
HMM_DECODER = "hmm_viterbi"


@dataclass(frozen=True)
class ObservationPrediction:
    """Selected directed arc and classifier confidence for one GPS observation.

    In HMM mode the arc is Viterbi-decoded, while confidence remains the
    classifier's positive-class probability for that selected candidate.
    """

    observation_index: int
    longitude: float
    latitude: float
    predicted_arc_id: int | None
    confidence: float | None
    predicted_arc_geometry: tuple[tuple[float, float], tuple[float, float]] | None


@dataclass(frozen=True)
class PredictionResult:
    """Structured predictions and decoder diagnostics for a complete trip."""

    trip_id: str
    observations: tuple[ObservationPrediction, ...]
    total_observations: int
    successfully_predicted_observations: int
    inference_duration_s: float
    decoder: str = CLASSIFIER_DECODER
    segment_resets: int = 0
    disconnected_resets: int = 0
    fallback_decisions: int = 0
    missing_candidate_observations: int = 0
    directed_transitions: int = 0
    disconnected_transitions: int = 0


def _load_model(model_path: Path):
    """Load the fitted classifier and validate its feature contract."""
    schema_path = model_path.parent / "feature_columns.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Random Forest model not found: {model_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")

    with schema_path.open("r", encoding="utf-8") as file:
        schema = json.load(file)
    if not isinstance(schema, dict) or schema.get("features") != FEATURE_COLUMNS:
        raise ValueError(
            "Saved model feature schema does not match "
            "pathsnap.features.FEATURE_COLUMNS"
        )

    model = joblib.load(model_path)
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Saved Random Forest model does not provide predict_proba()")
    if 1 not in np.asarray(getattr(model, "classes_", [])).tolist():
        raise ValueError("Saved Random Forest model has no positive class (1)")
    n_features = getattr(model, "n_features_in_", len(FEATURE_COLUMNS))
    if n_features != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Saved model expects {n_features} features; "
            f"{len(FEATURE_COLUMNS)} are configured"
        )
    return model


@lru_cache(maxsize=4)
def load_model(model_path: Path | None = None):
    """Load and cache the validated inference model for API-process reuse."""
    configured_path = model_path or CONFIG.paths.models / "model_random_forest.joblib"
    return _load_model(Path(configured_path).resolve())


def predict(
    trip_id: str,
    partition: str | None = None,
    *,
    partition_path: Path | None = None,
    model_path: Path | None = None,
    decoder: str = CLASSIFIER_DECODER,
) -> PredictionResult:
    """Predict directed arcs with the classifier or HMM/Viterbi decoder.

    Exactly one of ``partition`` or ``partition_path`` is required. The
    configured model path defaults to
    ``outputs/models/model_random_forest.joblib``. No labels are loaded.
    """
    if decoder not in {CLASSIFIER_DECODER, HMM_DECODER}:
        raise ValueError(f"Unknown prediction decoder {decoder!r}")
    started = time.perf_counter()
    trip = load_inference_trip(
        trip_id,
        partition=partition,
        partition_path=partition_path,
    )
    if trip.n_observations == 0:
        raise ValueError(f"Trip {trip_id} has no GPS observations")
    if trip.n_nodes == 0 or trip.n_arcs == 0:
        raise ValueError(f"Trip {trip_id} has no road-network nodes or arcs")
    if np.any(trip.arcs < 0) or np.any(trip.arcs >= trip.n_nodes):
        raise ValueError(f"Trip {trip_id} has an arc referencing an invalid node")

    processed = preprocess_trip(trip)
    candidates = generate_candidates(processed)

    # Keep one result for every input observation, including any without a
    # candidate; disconnected predictions are deliberately left disconnected.
    best: dict[int, tuple[int, float]] = {}
    hmm_diagnostics = None
    if candidates.arc_id.size:
        features = build_feature_frame(processed, candidates)
        missing_columns = [column for column in FEATURE_COLUMNS if column not in features]
        if missing_columns:
            raise ValueError(f"Inference features are missing columns: {missing_columns}")

        model = load_model() if model_path is None else load_model(model_path)
        probabilities = np.asarray(
            model.predict_proba(features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)),
            dtype=float,
        )
        classes = np.asarray(model.classes_).tolist()
        positive_column = classes.index(1)
        if probabilities.shape != (len(features), len(classes)):
            raise ValueError("Model predict_proba() returned an unexpected shape")
        positive_probabilities = probabilities[:, positive_column]
        if (
            not np.all(np.isfinite(positive_probabilities))
            or np.any(positive_probabilities < 0.0)
            or np.any(positive_probabilities > 1.0)
        ):
            raise ValueError("Model predict_proba() returned invalid confidence values")

        if decoder == HMM_DECODER:
            hmm_diagnostics = viterbi_decode(
                obs_idx=features["obs_idx"].to_numpy(dtype=np.int64),
                arc_ids=features["arc_id"].to_numpy(dtype=np.int64),
                emission_probabilities=positive_probabilities,
                t_positions=features["t"].to_numpy(dtype=float),
                observation_xy=processed.xy,
                segment_ids=processed.segment_id,
                arcs=processed.arcs,
                arc_lengths=processed.arc_len,
            )
            for observation_index, row in enumerate(hmm_diagnostics.candidate_indices):
                if row >= 0:
                    best[observation_index] = (
                        int(features.iloc[row]["arc_id"]),
                        float(positive_probabilities[row]),
                    )
        else:
            for obs_idx, arc_id, confidence in zip(
                features["obs_idx"].to_numpy(),
                features["arc_id"].to_numpy(),
                positive_probabilities,
            ):
                observation_index = int(obs_idx)
                current = best.get(observation_index)
                if current is None or confidence > current[1]:
                    best[observation_index] = (int(arc_id), float(confidence))
    elif decoder == HMM_DECODER:
        hmm_diagnostics = viterbi_decode(
            obs_idx=np.empty(0, dtype=np.int64),
            arc_ids=np.empty(0, dtype=np.int64),
            emission_probabilities=np.empty(0, dtype=float),
            t_positions=np.empty(0, dtype=float),
            observation_xy=processed.xy,
            segment_ids=processed.segment_id,
            arcs=processed.arcs,
            arc_lengths=processed.arc_len,
        )

    observations = []
    for observation_index, (longitude, latitude, _) in enumerate(trip.track):
        selected = best.get(observation_index)
        if selected is None:
            observations.append(ObservationPrediction(
                observation_index=observation_index,
                longitude=float(longitude),
                latitude=float(latitude),
                predicted_arc_id=None,
                confidence=None,
                predicted_arc_geometry=None,
            ))
            continue

        arc_id, confidence = selected
        from_node, to_node = map(int, trip.arcs[arc_id])
        start_lonlat = tuple(map(float, trip.nodes[from_node]))
        end_lonlat = tuple(map(float, trip.nodes[to_node]))
        observations.append(ObservationPrediction(
            observation_index=observation_index,
            longitude=float(longitude),
            latitude=float(latitude),
            predicted_arc_id=arc_id,
            confidence=confidence,
            predicted_arc_geometry=(start_lonlat, end_lonlat),
        ))

    duration = time.perf_counter() - started
    return PredictionResult(
        trip_id=trip.trip_id,
        observations=tuple(observations),
        total_observations=trip.n_observations,
        successfully_predicted_observations=len(best),
        inference_duration_s=duration,
        decoder=decoder,
        segment_resets=(hmm_diagnostics.segment_resets if hmm_diagnostics else 0),
        disconnected_resets=(hmm_diagnostics.disconnected_resets if hmm_diagnostics else 0),
        fallback_decisions=(hmm_diagnostics.fallback_decisions if hmm_diagnostics else 0),
        missing_candidate_observations=(
            hmm_diagnostics.missing_candidate_observations if hmm_diagnostics else 0
        ),
        directed_transitions=(
            hmm_diagnostics.directed_transitions if hmm_diagnostics else 0
        ),
        disconnected_transitions=(
            hmm_diagnostics.disconnected_transitions if hmm_diagnostics else 0
        ),
    )
