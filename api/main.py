"""PathSnap inference API for the Laptop 1 backend."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pathsnap import __version__
from pathsnap import pipeline
from pathsnap.config import CONFIG
from pathsnap.data_io import list_trips

logger = logging.getLogger("pathsnap.api")

MODEL_NAME = "random_forest"
PREDICTION_MODE = "classifier_plus_hmm_viterbi_batch"
DEFAULT_DEMO_PARTITION = "train"
ALLOWED_DEMO_PARTITIONS = frozenset({"train"})
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


class DemoPartitionConfigurationError(ValueError):
    """Raised when the demo partition is missing or not explicitly supported."""


def _demo_partition() -> str:
    partition = os.getenv("PATHSNAP_DEMO_PARTITION", DEFAULT_DEMO_PARTITION).strip()
    if partition not in ALLOWED_DEMO_PARTITIONS:
        raise DemoPartitionConfigurationError(
            "PATHSNAP_DEMO_PARTITION must be explicitly configured as "
            "'train'; student_test is not available and evaluator data is not "
            "available to this API"
        )
    return partition


def _cors_origins() -> list[str]:
    configured = os.getenv("PATHSNAP_CORS_ORIGINS")
    if configured is None:
        origins = list(DEFAULT_CORS_ORIGINS)
    else:
        origins = [origin.strip() for origin in configured.split(",") if origin.strip()]

    for origin in origins:
        parsed = urlsplit(origin)
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"Invalid CORS origin {origin!r}; use an exact HTTP(S) origin")
    return origins


class HealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    ready: bool
    model_loaded: bool
    model_name: str
    prediction_mode: str
    demo_partition: str | None
    demo_partition_configured: bool
    demo_partition_available: bool


class TripsResponse(BaseModel):
    partition: str
    trips: list[str]


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=r"^[0-9]{8}$")


class GeoJSONLineString(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[tuple[float, float]]


class ArcProperties(BaseModel):
    arc_id: int
    obs_idx: int
    confidence: float


class ArcFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    properties: ArcProperties
    geometry: GeoJSONLineString


class GeoJSONFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[ArcFeature]


class PredictionSummary(BaseModel):
    total_observations: int
    successfully_predicted_observations: int
    inference_duration_s: float


class PredictResponse(BaseModel):
    trip_id: str
    decoder: str
    predicted_arc_ids: list[int | None]
    confidence: list[float | None]
    gps: GeoJSONLineString
    predicted_route: GeoJSONFeatureCollection
    summary: PredictionSummary


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the validated model once and retain it for this API process."""
    app.state.model = None
    app.state.model_error = None
    try:
        app.state.model = pipeline.load_model()
    except (FileNotFoundError, ValueError) as exc:
        app.state.model_error = str(exc)
        logger.error("PathSnap model is unavailable or invalid: %s", exc)
    yield


app = FastAPI(
    title="PathSnap API",
    version=__version__,
    description="Classifier and HMM/Viterbi batch map-matching inference.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report model and demo-partition readiness."""
    partition = None
    partition_configured = True
    try:
        partition = _demo_partition()
    except DemoPartitionConfigurationError:
        partition_configured = False

    partition_available = (
        CONFIG.paths.partition_dir(partition).is_dir()
        if partition is not None
        else False
    )
    model_loaded = app.state.model is not None
    ready = model_loaded and partition_configured and partition_available
    return HealthResponse(
        status="ready" if ready else "not_ready",
        ready=ready,
        model_loaded=model_loaded,
        model_name=MODEL_NAME,
        prediction_mode=PREDICTION_MODE,
        demo_partition=partition,
        demo_partition_configured=partition_configured,
        demo_partition_available=partition_available,
    )


@app.get("/trips", response_model=TripsResponse)
def trips() -> TripsResponse:
    """List trip IDs only from the explicitly configured demo partition."""
    try:
        partition = _demo_partition()
        trip_ids = list_trips(partition)
    except DemoPartitionConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error("Configured demo partition is unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Configured demo partition directory is unavailable",
        ) from exc
    return TripsResponse(partition=partition, trips=trip_ids)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Return HMM-decoded predictions without labels or evaluation metrics."""
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded; check model artifact and feature schema",
        )
    try:
        partition = _demo_partition()
    except DemoPartitionConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = pipeline.predict(
            request.trip_id,
            partition=partition,
            decoder=pipeline.HMM_DECODER,
        )
    except FileNotFoundError as exc:
        logger.info("Trip %s is unavailable in %s: %s", request.trip_id, partition, exc)
        raise HTTPException(
            status_code=404,
            detail=f"Trip {request.trip_id} is unavailable in the configured demo partition",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed for trip %s", request.trip_id)
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    arc_features = []
    for observation in result.observations:
        if (
            observation.predicted_arc_id is None
            or observation.confidence is None
            or observation.predicted_arc_geometry is None
        ):
            continue
        arc_features.append(ArcFeature(
            properties=ArcProperties(
                arc_id=observation.predicted_arc_id,
                obs_idx=observation.observation_index,
                confidence=observation.confidence,
            ),
            geometry=GeoJSONLineString(
                coordinates=list(observation.predicted_arc_geometry)
            ),
        ))

    return PredictResponse(
        trip_id=result.trip_id,
        decoder=f"random_forest_{result.decoder}",
        predicted_arc_ids=[
            observation.predicted_arc_id for observation in result.observations
        ],
        confidence=[
            observation.confidence for observation in result.observations
        ],
        gps=GeoJSONLineString(
            coordinates=[
                (observation.longitude, observation.latitude)
                for observation in result.observations
            ]
        ),
        predicted_route=GeoJSONFeatureCollection(features=arc_features),
        summary=PredictionSummary(
            total_observations=result.total_observations,
            successfully_predicted_observations=(
                result.successfully_predicted_observations
            ),
            inference_duration_s=result.inference_duration_s,
        ),
    )


def main() -> None:
    """Run Uvicorn using environment-configurable bind host and port."""
    host = os.getenv("PATHSNAP_API_HOST", "0.0.0.0")
    port_value = os.getenv("PATHSNAP_API_PORT", "8000")
    try:
        port = int(port_value)
    except ValueError as exc:
        raise ValueError("PATHSNAP_API_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PATHSNAP_API_PORT must be between 1 and 65535")
    uvicorn.run("api.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
