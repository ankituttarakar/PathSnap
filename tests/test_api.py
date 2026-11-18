from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import main
from pathsnap.pipeline import (
    HMM_DECODER,
    ObservationPrediction,
    PredictionResult,
)


class PathSnapApiTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"PATHSNAP_DEMO_PARTITION": "train"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def _client(self, model=None):
        load_model = patch.object(main.pipeline, "load_model", return_value=model or object())
        load_model.start()
        self.addCleanup(load_model.stop)
        client = TestClient(main.app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_health_reports_model_and_mode(self):
        client = self._client(model=object())
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "ready",
            "ready": True,
            "model_loaded": True,
            "model_name": "random_forest",
            "prediction_mode": "classifier_plus_hmm_viterbi_batch",
            "demo_partition": "train",
            "demo_partition_configured": True,
            "demo_partition_available": True,
        })

    def test_trips_lists_only_configured_partition(self):
        client = self._client()
        with patch.object(main, "list_trips", return_value=["00000083"]) as list_partition:
            response = client.get("/trips")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "partition": "train",
            "trips": ["00000083"],
        })
        list_partition.assert_called_once_with("train")

    def test_trips_returns_configuration_error_for_unsupported_partition(self):
        client = self._client()
        for partition in ("student_test", "evaluator_test"):
            with self.subTest(partition=partition):
                with patch.dict(os.environ, {"PATHSNAP_DEMO_PARTITION": partition}):
                    response = client.get("/trips")
                self.assertEqual(response.status_code, 503)
                self.assertIn("evaluator data is not available", response.json()["detail"])

    def test_predict_returns_geojson_without_ground_truth_or_metrics(self):
        client = self._client()
        prediction = PredictionResult(
            trip_id="00000083",
            observations=(
                ObservationPrediction(
                    observation_index=0,
                    longitude=8.0,
                    latitude=50.0,
                    predicted_arc_id=17,
                    confidence=0.82,
                    predicted_arc_geometry=((8.0, 50.0), (8.001, 50.0)),
                ),
                ObservationPrediction(
                    observation_index=1,
                    longitude=8.0005,
                    latitude=50.0,
                    predicted_arc_id=None,
                    confidence=None,
                    predicted_arc_geometry=None,
                ),
            ),
            total_observations=2,
            successfully_predicted_observations=1,
            inference_duration_s=0.25,
            decoder=HMM_DECODER,
        )
        with patch.object(main.pipeline, "predict", return_value=prediction) as infer:
            response = client.post("/predict", json={"trip_id": "00000083"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["trip_id"], "00000083")
        self.assertEqual(body["decoder"], "random_forest_hmm_viterbi")
        self.assertEqual(body["predicted_arc_ids"], [17, None])
        self.assertEqual(body["confidence"], [0.82, None])
        self.assertEqual(body["gps"], {
            "type": "LineString",
            "coordinates": [[8.0, 50.0], [8.0005, 50.0]],
        })
        self.assertEqual(body["predicted_route"]["type"], "FeatureCollection")
        feature = body["predicted_route"]["features"][0]
        self.assertEqual(feature["properties"], {
            "arc_id": 17,
            "obs_idx": 0,
            "confidence": 0.82,
        })
        self.assertEqual(feature["geometry"], {
            "type": "LineString",
            "coordinates": [[8.0, 50.0], [8.001, 50.0]],
        })
        self.assertEqual(body["summary"]["total_observations"], 2)
        self.assertNotIn("ground_truth", body)
        self.assertNotIn("metrics", body)
        infer.assert_called_once_with(
            "00000083", partition="train", decoder=HMM_DECODER
        )

    def test_predict_rejects_bad_trip_id(self):
        client = self._client()
        response = client.post("/predict", json={"trip_id": "../"})
        self.assertEqual(response.status_code, 422)

    def test_predict_returns_not_found_for_missing_trip_files(self):
        client = self._client()
        with patch.object(
            main.pipeline,
            "predict",
            side_effect=FileNotFoundError("missing track"),
        ):
            response = client.post("/predict", json={"trip_id": "00000083"})
        self.assertEqual(response.status_code, 404)

    def test_predict_returns_503_when_model_did_not_load(self):
        load_model = patch.object(
            main.pipeline,
            "load_model",
            side_effect=FileNotFoundError("missing model"),
        )
        load_model.start()
        self.addCleanup(load_model.stop)
        client = TestClient(main.app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)

        response = client.post("/predict", json={"trip_id": "00000083"})
        self.assertEqual(response.status_code, 503)

    def test_health_is_not_ready_when_model_schema_is_invalid(self):
        load_model = patch.object(
            main.pipeline,
            "load_model",
            side_effect=ValueError("feature schema mismatch"),
        )
        load_model.start()
        self.addCleanup(load_model.stop)
        client = TestClient(main.app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)

        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ready"])
        self.assertFalse(response.json()["model_loaded"])

    def test_cors_allows_the_approved_local_frontend_origin(self):
        client = self._client()
        for origin in (
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ):
            with self.subTest(origin=origin):
                response = client.options(
                    "/predict",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.headers["access-control-allow-origin"],
                    origin,
                )


if __name__ == "__main__":
    unittest.main()
