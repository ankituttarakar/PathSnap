from __future__ import annotations

import unittest

import numpy as np

from pathsnap.hmm import viterbi_decode


class ViterbiDecoderTests(unittest.TestCase):
    def decode(self, obs, arcs, probabilities, t, xy, segments, network, lengths):
        return viterbi_decode(
            obs_idx=np.asarray(obs),
            arc_ids=np.asarray(arcs),
            emission_probabilities=np.asarray(probabilities, dtype=float),
            t_positions=np.asarray(t, dtype=float),
            observation_xy=np.asarray(xy, dtype=float),
            segment_ids=np.asarray(segments),
            arcs=np.asarray(network, dtype=np.int64),
            arc_lengths=np.asarray(lengths, dtype=float),
        )

    def test_viterbi_prefers_a_valid_directed_path_over_local_top_one(self):
        result = self.decode(
            obs=[0, 0, 1, 1],
            arcs=[0, 2, 1, 3],
            probabilities=[0.55, 0.45, 0.40, 0.60],
            t=[0.9, 0.1, 0.2, 0.5],
            xy=[[9.0, 0.0], [10.0, 0.0]],
            segments=[0, 0],
            network=[[0, 1], [1, 2], [1, 0], [10, 11]],
            lengths=[10.0, 10.0, 10.0, 10.0],
        )
        self.assertEqual(result.arc_ids.tolist(), [0, 1])
        self.assertEqual(result.directed_transitions, 1)
        self.assertEqual(result.disconnected_transitions, 0)

    def test_reciprocal_directed_arc_is_a_distinct_valid_state(self):
        result = self.decode(
            obs=[0, 1],
            arcs=[0, 1],
            probabilities=[0.8, 0.7],
            t=[0.8, 0.2],
            xy=[[8.0, 0.0], [6.0, 0.0]],
            segments=[0, 0],
            network=[[0, 1], [1, 0]],
            lengths=[10.0, 10.0],
        )
        self.assertEqual(result.arc_ids.tolist(), [0, 1])
        self.assertEqual(result.directed_transitions, 1)

    def test_same_arc_cannot_move_backward_and_disconnected_fallback_is_counted(self):
        result = self.decode(
            obs=[0, 1],
            arcs=[0, 1],
            probabilities=[0.9, 0.9],
            t=[0.8, 0.2],
            xy=[[8.0, 0.0], [2.0, 0.0]],
            segments=[0, 0],
            network=[[0, 1], [5, 6]],
            lengths=[10.0, 10.0],
        )
        self.assertEqual(result.arc_ids.tolist(), [0, 1])
        self.assertEqual(result.disconnected_resets, 1)
        self.assertEqual(result.fallback_decisions, 1)
        self.assertEqual(result.disconnected_transitions, 1)

    def test_segment_boundary_resets_without_a_fallback(self):
        result = self.decode(
            obs=[0, 1],
            arcs=[0, 1],
            probabilities=[0.2, 0.9],
            t=[0.5, 0.5],
            xy=[[0.0, 0.0], [100.0, 0.0]],
            segments=[0, 1],
            network=[[0, 1], [5, 6]],
            lengths=[10.0, 10.0],
        )
        self.assertEqual(result.arc_ids.tolist(), [0, 1])
        self.assertEqual(result.segment_resets, 1)
        self.assertEqual(result.fallback_decisions, 0)
        self.assertEqual(result.disconnected_transitions, 0)

    def test_missing_candidates_produce_none_and_restart_the_sequence(self):
        result = self.decode(
            obs=[0, 2],
            arcs=[0, 1],
            probabilities=[0.9, 0.8],
            t=[0.5, 0.5],
            xy=[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            segments=[0, 0, 0],
            network=[[0, 1], [1, 2]],
            lengths=[2.0, 2.0],
        )
        self.assertEqual(result.arc_ids.tolist(), [0, -1, 1])
        self.assertEqual(result.missing_candidate_observations, 1)
        self.assertEqual(result.fallback_decisions, 1)

    def test_single_candidate_decodes_and_tiny_emissions_are_stable(self):
        kwargs = dict(
            obs=[0, 1],
            arcs=[0, 0],
            probabilities=[1e-300, 1e-300],
            t=[0.2, 0.8],
            xy=[[2.0, 0.0], [8.0, 0.0]],
            segments=[0, 0],
            network=[[0, 1]],
            lengths=[10.0],
        )
        first = self.decode(**kwargs)
        second = self.decode(**kwargs)
        self.assertEqual(first.arc_ids.tolist(), [0, 0])
        self.assertTrue(np.array_equal(first.arc_ids, second.arc_ids))
        self.assertEqual(first.disconnected_transitions, 0)

    def test_impossible_transition_does_not_create_a_connector(self):
        result = self.decode(
            obs=[0, 1],
            arcs=[0, 1],
            probabilities=[0.7, 0.9],
            t=[0.5, 0.5],
            xy=[[0.0, 0.0], [1.0, 0.0]],
            segments=[0, 0],
            network=[[0, 1], [4, 5]],
            lengths=[1.0, 1.0],
        )
        self.assertEqual(result.arc_ids.tolist(), [0, 1])
        self.assertEqual(result.directed_transitions, 0)
        self.assertEqual(result.disconnected_transitions, 1)
        self.assertEqual(result.disconnected_resets, 1)


if __name__ == "__main__":
    unittest.main()
