from __future__ import annotations

import unittest

import numpy as np

from pathsnap.cv import assert_no_group_overlap, group_kfold_split
from scripts.run_hmm_evaluation import _route_continuity


class HMMEvaluationLeakageTests(unittest.TestCase):
    def test_grouped_folds_keep_each_trip_in_one_partition(self):
        groups = np.repeat(
            np.asarray([f"trip_{index}" for index in range(6)]),
            3,
        )
        for train_rows, validation_rows in group_kfold_split(groups, n_splits=3):
            train_trips = np.unique(groups[train_rows])
            validation_trips = np.unique(groups[validation_rows])
            self.assertTrue(set(train_trips).isdisjoint(validation_trips))

    def test_overlapping_train_and_held_out_trip_ids_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "Group overlap"):
            assert_no_group_overlap(
                np.asarray(["train_trip_1", "train_trip_2"]),
                np.asarray(["student_trip_1", "train_trip_2"]),
            )

    def test_route_continuity_uses_all_predictions_but_skips_segment_boundaries(self):
        continuity, connected, pairs = _route_continuity(
            predicted_arcs=np.asarray([0, 0, 1, 2]),
            segment_ids=np.asarray([0, 0, 0, 1]),
            network_arcs=np.asarray([[0, 1], [1, 2], [3, 4]]),
        )

        self.assertEqual((continuity, connected, pairs), (0.5, 1, 2))


if __name__ == "__main__":
    unittest.main()
