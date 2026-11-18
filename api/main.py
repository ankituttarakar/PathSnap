"""Minimal FastAPI application placeholder (Laptop 1).

Phase 0 provides only a health-check endpoint so the backend scaffold is
verifiable (``uvicorn api.main:app --reload``). The real endpoints —
``GET /trips`` (list the 10 evaluation trips) and ``POST /predict`` (run the full
pipeline and compute accuracy server-side) — are implemented in Phase 10.

Leakage rule (enforced from the start): ground-truth labels must stay
server-side and must never be returned to the UI.
"""

from __future__ import annotations

from fastapi import FastAPI

from pathsnap import __version__

app = FastAPI(
    title="PathSnap API",
    version=__version__,
    description="Map-matching & route reconstruction backend (skeleton).",
)


@app.get("/health")
def health() -> dict:
    """Liveness probe used to confirm the backend scaffold boots."""
    return {"status": "ok", "version": __version__, "phase": "0-skeleton"}


# NOTE: /trips and /predict are intentionally not implemented in Phase 0.
