from __future__ import annotations

import unittest

import numpy as np

from pathsnap.candidates import generate_candidates
from pathsnap.data_io import InferenceTrip
from pathsnap.features import FEATURE_COLUMNS, build_feature_frame
from pathsnap.preprocess import preprocess_trip


class FeatureSchemaTests(unittest.TestCase):
    def test_eighteen_features_are_inference_only_and_direction_aware(self):
        trip = InferenceTrip(
            trip_id="00000083",
            partition="train",
            track=np.asarray([
                [8.0, 50.0, 1.0],
                [8.0002, 50.0, 2.0],
                [8.0004, 50.0, 50.0],
            ]),
            nodes=np.asarray([[8.0, 50.0], [8.001, 50.0]]),
            arcs=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        )
        processed = preprocess_trip(trip)
        candidates = generate_candidates(processed, k=2, query_k=2, radius_m=100.0)
        features = build_feature_frame(processed, candidates)

        self.assertEqual(len(FEATURE_COLUMNS), 18)
        self.assertEqual(list(features[FEATURE_COLUMNS].columns), FEATURE_COLUMNS)
        self.assertNotIn("y", features)
        self.assertNotIn("confident", features)
        self.assertTrue(np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all())

        first_moving = features[features.obs_idx == 1].set_index("arc_id")
        self.assertLess(
            first_moving.loc[0, "heading_diff_directed_rad"],
            first_moving.loc[1, "heading_diff_directed_rad"],
        )
        self.assertGreater(
            first_moving.loc[0, "projected_speed_along_arc_ms"],
            first_moving.loc[1, "projected_speed_along_arc_ms"],
        )

        after_gap = features[features.obs_idx == 2]
        self.assertTrue((after_gap["seg_start"] == 1).all())
        self.assertTrue(np.isnan(processed.speed_ms[2]))
        self.assertTrue(np.isnan(processed.heading[2]))
        self.assertTrue((after_gap["heading_diff"] == np.pi / 2.0).all())
        self.assertTrue((after_gap["previous_point_dist_m"] == 0).all())
        self.assertTrue((after_gap["projected_speed_along_arc_ms"] == 0).all())
        self.assertTrue((after_gap["gps_heading_change_rad"] == 0).all())


if __name__ == "__main__":
    unittest.main()
