"""PathSnap — distributed GPS map-matching & route reconstruction.

A modular pipeline that snaps noisy 1 Hz GPS observations onto a road network
and reconstructs the traversed route. See ``docs/IMPLEMENTATION_ROADMAP.md`` for
the phase plan and ``docs/PROJECT_ANALYSIS.md`` for the audit.

Only Phase 0 (foundation/skeleton) is implemented. Modules for candidate
generation, labelling, modelling, HMM/Viterbi, API and UI are declared as
documented stubs and raise ``NotImplementedError`` until their phase.
"""

from __future__ import annotations

__version__ = "0.0.1"
__all__ = ["config", "logging_utils", "data_io"]
