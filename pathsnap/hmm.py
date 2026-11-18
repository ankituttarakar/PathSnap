"""HMM + Viterbi route reconstruction (Phase 8 — not yet implemented).

Planned responsibilities:
    * Hidden states = candidate arcs per observation (time step = observation).
    * Emission probabilities from the tuned classifier (+ distance likelihood).
    * Transition probabilities from road-network connectivity (networkx), with
      penalties for disconnected / over-long transitions and special handling of
      large GPS gaps.
    * Viterbi decoding to the most likely connected arc sequence.
    * Comparison of classifier-only vs classifier+HMM (pointwise + sequence-level
      metrics, route continuity).

Decoder parameters are tuned on train + student_test only. Out of scope here.
"""

from __future__ import annotations


def viterbi_decode(*args, **kwargs):  # noqa: ANN002, ANN003
    """Placeholder: Viterbi decoding over candidate arcs (Phase 8)."""
    raise NotImplementedError("HMM/Viterbi is implemented in Phase 8 (not started).")
