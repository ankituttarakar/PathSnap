"""Directed-arc HMM and deterministic Viterbi decoding.

Candidate classifier probabilities are normalized within each observation and
used as emissions. Transitions are allowed only for forward movement on the
same directed arc or directly connected directed arcs. Transition plausibility
penalizes disagreement between directed network distance and GPS displacement.
No missing road links are invented: an impossible transition starts a new
emission-only chain and is counted as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class ViterbiResult:
    """Selected candidate-row indexes and route-decoding diagnostics."""

    candidate_indices: np.ndarray
    arc_ids: np.ndarray
    segment_resets: int
    disconnected_resets: int
    fallback_decisions: int
    missing_candidate_observations: int
    directed_transitions: int
    disconnected_transitions: int


def _transition_distance(
    previous_arc: int,
    current_arc: int,
    previous_t: float,
    current_t: float,
    arcs: np.ndarray,
    arc_lengths: np.ndarray,
) -> float | None:
    """Return along-network distance for a valid directed one-step transition."""
    if previous_arc == current_arc:
        if current_t + 1e-12 < previous_t:
            return None
        return max(0.0, current_t - previous_t) * arc_lengths[previous_arc]

    if arcs[previous_arc, 1] != arcs[current_arc, 0]:
        return None
    return (
        (1.0 - previous_t) * arc_lengths[previous_arc]
        + current_t * arc_lengths[current_arc]
    )


def _emission_log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Normalize positive classifier scores within an observation in log space."""
    scores = np.maximum(probabilities, np.finfo(float).tiny)
    total = scores.sum()
    if not np.isfinite(total) or total <= 0.0:
        scores = np.full(len(probabilities), 1.0 / len(probabilities))
    else:
        scores = scores / total
    return np.log(scores)


def viterbi_decode(
    obs_idx: np.ndarray,
    arc_ids: np.ndarray,
    emission_probabilities: np.ndarray,
    t_positions: np.ndarray,
    observation_xy: np.ndarray,
    segment_ids: np.ndarray,
    arcs: np.ndarray,
    arc_lengths: np.ndarray,
) -> ViterbiResult:
    """Decode candidate directed arcs for one complete trip.

    ``candidate_indices[i]`` is the row index into the supplied candidate
    arrays selected for observation ``i``; it is -1 when that observation has
    no candidates. Equal scores are resolved by ascending arc ID.

    A graph transition exists only when the same directed arc advances
    monotonically, or the previous arc's destination is the current arc's
    source. There are no inferred connector arcs or hidden shortest paths.
    """
    obs_idx = np.asarray(obs_idx, dtype=np.int64)
    arc_ids = np.asarray(arc_ids, dtype=np.int64)
    emissions = np.asarray(emission_probabilities, dtype=float)
    t_positions = np.asarray(t_positions, dtype=float)
    xy = np.asarray(observation_xy, dtype=float)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    arcs = np.asarray(arcs, dtype=np.int64)
    arc_lengths = np.asarray(arc_lengths, dtype=float)

    n_candidates = len(obs_idx)
    if any(len(values) != n_candidates for values in (arc_ids, emissions, t_positions)):
        raise ValueError("Candidate arrays must have matching lengths")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("observation_xy must have shape (n_observations, 2)")
    n_observations = len(xy)
    if len(segment_ids) != n_observations:
        raise ValueError("segment_ids must contain one value per observation")
    if arcs.ndim != 2 or arcs.shape[1] != 2 or len(arc_lengths) != len(arcs):
        raise ValueError("arcs and arc_lengths must describe the directed network")
    if (
        np.any(obs_idx < 0)
        or np.any(obs_idx >= n_observations)
        or np.any(arc_ids < 0)
        or np.any(arc_ids >= len(arcs))
    ):
        raise ValueError("Candidate observation or arc index is out of range")
    if (
        not np.all(np.isfinite(emissions))
        or np.any(emissions < 0.0)
        or np.any(emissions > 1.0)
    ):
        raise ValueError("Emission probabilities must be finite values in [0, 1]")
    if (
        not np.all(np.isfinite(t_positions))
        or np.any(t_positions < 0.0)
        or np.any(t_positions > 1.0)
        or not np.all(np.isfinite(xy))
        or not np.all(np.isfinite(arc_lengths))
        or np.any(arc_lengths < 0.0)
    ):
        raise ValueError("Positions and geometry must be finite and in range")

    rows_by_observation: list[list[int]] = [[] for _ in range(n_observations)]
    for row, observation_index in enumerate(obs_idx.tolist()):
        rows_by_observation[observation_index].append(row)
    for rows in rows_by_observation:
        rows.sort(key=lambda row: (int(arc_ids[row]), row))

    selected_rows = np.full(n_observations, -1, dtype=np.int64)
    chain_rows: list[np.ndarray] = []
    chain_backpointers: list[np.ndarray] = []
    chain_scores = np.empty(0, dtype=float)
    previous_observation: int | None = None
    previous_was_missing = False
    segment_resets = 0
    disconnected_resets = 0
    fallback_decisions = 0
    missing_count = 0

    def finish_chain() -> None:
        nonlocal chain_rows, chain_backpointers, chain_scores
        if not chain_rows:
            return
        state = int(np.argmax(chain_scores))
        for layer in range(len(chain_rows) - 1, -1, -1):
            selected_rows[chain_observation_indices[layer]] = chain_rows[layer][state]
            if layer:
                state = int(chain_backpointers[layer][state])
        chain_rows = []
        chain_backpointers = []
        chain_scores = np.empty(0, dtype=float)

    chain_observation_indices: list[int] = []

    def start_chain(observation_index: int, rows: np.ndarray) -> None:
        nonlocal chain_rows, chain_backpointers, chain_scores, chain_observation_indices
        chain_rows = [rows]
        chain_backpointers = [np.full(len(rows), -1, dtype=np.int64)]
        chain_scores = _emission_log_probabilities(emissions[rows])
        chain_observation_indices = [observation_index]

    for observation_index in range(n_observations):
        if (
            previous_observation is not None
            and segment_ids[observation_index] != segment_ids[previous_observation]
        ):
            finish_chain()
            segment_resets += 1
            previous_was_missing = False

        rows = np.asarray(rows_by_observation[observation_index], dtype=np.int64)
        if not len(rows):
            finish_chain()
            missing_count += 1
            previous_was_missing = True
            previous_observation = observation_index
            continue

        if not chain_rows:
            if previous_was_missing:
                fallback_decisions += 1
            start_chain(observation_index, rows)
            previous_was_missing = False
            previous_observation = observation_index
            continue

        previous_rows = chain_rows[-1]
        transition_scores = np.full((len(previous_rows), len(rows)), -np.inf)
        gps_distance = float(np.linalg.norm(
            xy[observation_index] - xy[observation_index - 1]
        ))
        distance_scale = max(gps_distance, 1.0)

        for previous_position, previous_row in enumerate(previous_rows):
            raw_scores = np.full(len(rows), -np.inf)
            for current_position, current_row in enumerate(rows):
                network_distance = _transition_distance(
                    int(arc_ids[previous_row]),
                    int(arc_ids[current_row]),
                    float(t_positions[previous_row]),
                    float(t_positions[current_row]),
                    arcs,
                    arc_lengths,
                )
                if network_distance is not None:
                    raw_scores[current_position] = (
                        -abs(network_distance - gps_distance) / distance_scale
                    )
            finite = np.isfinite(raw_scores)
            if np.any(finite):
                transition_scores[previous_position, finite] = (
                    raw_scores[finite] - logsumexp(raw_scores[finite])
                )

        joint_scores = chain_scores[:, None] + transition_scores
        next_scores = _emission_log_probabilities(emissions[rows]) + logsumexp(
            joint_scores, axis=0
        )
        if not np.any(np.isfinite(next_scores)):
            finish_chain()
            disconnected_resets += 1
            fallback_decisions += 1
            start_chain(observation_index, rows)
        else:
            backpointer = np.argmax(joint_scores, axis=0).astype(np.int64)
            chain_rows.append(rows)
            chain_backpointers.append(backpointer)
            chain_scores = next_scores
            chain_observation_indices.append(observation_index)

        previous_was_missing = False
        previous_observation = observation_index

    finish_chain()
    decoded_arcs = np.full(n_observations, -1, dtype=np.int64)
    has_prediction = selected_rows >= 0
    decoded_arcs[has_prediction] = arc_ids[selected_rows[has_prediction]]

    directed_transitions = 0
    disconnected_transitions = 0
    for observation_index in range(1, n_observations):
        previous_row = selected_rows[observation_index - 1]
        current_row = selected_rows[observation_index]
        if (
            previous_row < 0
            or current_row < 0
            or segment_ids[observation_index] != segment_ids[observation_index - 1]
        ):
            continue
        distance = _transition_distance(
            int(arc_ids[previous_row]),
            int(arc_ids[current_row]),
            float(t_positions[previous_row]),
            float(t_positions[current_row]),
            arcs,
            arc_lengths,
        )
        if distance is None:
            disconnected_transitions += 1
        else:
            directed_transitions += 1

    return ViterbiResult(
        candidate_indices=selected_rows,
        arc_ids=decoded_arcs,
        segment_resets=segment_resets,
        disconnected_resets=disconnected_resets,
        fallback_decisions=fallback_decisions,
        missing_candidate_observations=missing_count,
        directed_transitions=directed_transitions,
        disconnected_transitions=disconnected_transitions,
    )
