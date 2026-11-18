from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pathsnap import data_io, pipeline
from pathsnap.candidates import Candidates
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.pipeline import HMM_DECODER


class _FakeClassifier:
    classes_ = np.array([0, 1])
    n_features_in_ = len(FEATURE_COLUMNS)

    def predict_proba(self, features):
        return np.tile([0.1, 0.9], (len(features), 1))


class InferenceLoaderTests(unittest.TestCase):
    def test_inference_loader_does_not_attempt_to_open_route(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            partition_path = Path(temporary_directory)
            trip_dir = partition_path / "00000083"
            trip_dir.mkdir()
            np.savetxt(trip_dir / "00000083.track", [[8.0, 50.0, 1.0]])
            np.savetxt(trip_dir / "00000083.nodes", [[8.0, 50.0], [8.001, 50.0]])
            np.savetxt(trip_dir / "00000083.arcs", [[0, 1]], fmt="%d")

            with patch.object(
                data_io,
                "read_route",
                side_effect=AssertionError("inference attempted to read .route"),
            ) as read_route:
                trip = data_io.load_inference_trip(
                    "00000083", partition_path=partition_path
                )

            read_route.assert_not_called()
            self.assertEqual(trip.n_observations, 1)
            self.assertEqual(trip.arcs.tolist(), [[0, 1]])
            self.assertFalse(hasattr(trip, "route"))

    def test_loader_rejects_invalid_trip_id(self):
        with self.assertRaisesRegex(ValueError, "Invalid trip ID"):
            data_io.load_inference_trip("..", partition="train")

    def test_loader_requires_exactly_one_location_selector(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            data_io.load_inference_trip("00000083")


class PredictionPipelineTests(unittest.TestCase):
    def test_predict_returns_directed_arc_and_all_observations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trip_dir = root / "00000083"
            trip_dir.mkdir()
            np.savetxt(
                trip_dir / "00000083.track",
                [[8.0, 50.0, 1.0], [8.0005, 50.0, 2.0]],
            )
            np.savetxt(
                trip_dir / "00000083.nodes",
                [[8.0, 50.0], [8.001, 50.0]],
            )
            np.savetxt(trip_dir / "00000083.arcs", [[0, 1]], fmt="%d")

            with patch.object(
                pipeline, "load_model", return_value=_FakeClassifier()
            ) as load_model:
                result = pipeline.predict(
                    "00000083",
                    partition_path=root,
                )

            load_model.assert_called_once_with()
            self.assertEqual(result.trip_id, "00000083")
            self.assertEqual(result.total_observations, 2)
            self.assertEqual(result.successfully_predicted_observations, 2)
            self.assertEqual(
                [observation.predicted_arc_id for observation in result.observations],
                [0, 0],
            )
            self.assertEqual(
                result.observations[0].predicted_arc_geometry,
                ((8.0, 50.0), (8.001, 50.0)),
            )
            self.assertEqual(result.observations[0].confidence, 0.9)
            self.assertGreaterEqual(result.inference_duration_s, 0.0)

            with patch.object(
                pipeline, "load_model", return_value=_FakeClassifier()
            ), patch.object(
                data_io,
                "read_route",
                side_effect=AssertionError("HMM inference attempted to read .route"),
            ) as read_route:
                hmm_result = pipeline.predict(
                    "00000083",
                    partition_path=root,
                    decoder=HMM_DECODER,
                )

            read_route.assert_not_called()
            self.assertEqual(hmm_result.decoder, HMM_DECODER)
            self.assertEqual(
                [observation.predicted_arc_id for observation in hmm_result.observations],
                [0, 0],
            )
            self.assertEqual(hmm_result.directed_transitions, 1)

    def test_schema_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            model_path = model_dir / "model_random_forest.joblib"
            model_path.touch()
            (model_dir / "feature_columns.json").write_text(
                json.dumps({"features": FEATURE_COLUMNS[:-1]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "feature schema"):
                pipeline._load_model(model_path)

    def test_missing_model_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "missing.joblib"
            with self.assertRaisesRegex(FileNotFoundError, "Random Forest model"):
                pipeline._load_model(model_path)

    def test_observation_without_candidates_is_returned_without_prediction(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trip_dir = root / "00000083"
            trip_dir.mkdir()
            np.savetxt(
                trip_dir / "00000083.track",
                [[8.0, 50.0, 1.0], [8.0005, 50.0, 2.0]],
            )
            np.savetxt(
                trip_dir / "00000083.nodes",
                [[8.0, 50.0], [8.001, 50.0], [8.002, 50.001]],
            )
            np.savetxt(
                trip_dir / "00000083.arcs",
                [[0, 1], [1, 2]],
                fmt="%d",
            )
            empty_candidates = Candidates(
                obs_idx=np.array([], dtype=np.int64),
                arc_id=np.array([], dtype=np.int64),
                dist_m=np.array([], dtype=float),
                t=np.array([], dtype=float),
                rank=np.array([], dtype=np.int64),
                n_cand=np.array([], dtype=np.int64),
            )

            with patch.object(
                pipeline, "generate_candidates", return_value=empty_candidates
            ):
                result = pipeline.predict(
                    "00000083", partition_path=root, decoder=HMM_DECODER
                )

            self.assertEqual(result.successfully_predicted_observations, 0)
            self.assertEqual(len(result.observations), 2)
            self.assertEqual(result.missing_candidate_observations, 2)
            self.assertTrue(all(
                observation.predicted_arc_id is None
                and observation.confidence is None
                and observation.predicted_arc_geometry is None
                for observation in result.observations
            ))


if __name__ == "__main__":
    unittest.main()
