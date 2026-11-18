import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowDownToLine,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Gauge,
  Layers,
  MapPinned,
  Radio,
  RefreshCw,
  Route,
  Server,
  Sparkles,
  Wifi,
  WifiOff,
} from "lucide-react";
import { getHealth, getTrips, hasApiBaseUrl, predictTrip } from "./api.js";
import RouteMap from "./components/RouteMap.jsx";

function formatDuration(seconds) {
  return Number.isFinite(seconds) ? `${seconds.toFixed(2)} s` : "—";
}

function getConfidenceSummary(values) {
  const finite = (values ?? []).filter(Number.isFinite);
  if (finite.length === 0) return null;
  const total = finite.reduce((sum, value) => sum + value, 0);
  return {
    mean: total / finite.length,
    min: Math.min(...finite),
    max: Math.max(...finite),
  };
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [isChecking, setIsChecking] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [backendError, setBackendError] = useState("");
  const [predictionError, setPredictionError] = useState("");
  const [fitToken, setFitToken] = useState(0);

  const configured = hasApiBaseUrl();
  const backendReady = Boolean(health?.ready && trips.length > 0);
  const decoderIsHmm = health?.prediction_mode?.toLowerCase().includes("hmm");

  const refreshBackend = useCallback(async () => {
    setIsChecking(true);
    setBackendError("");
    const [healthResult, tripsResult] = await Promise.allSettled([
      getHealth(),
      getTrips(),
    ]);

    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
    } else {
      setHealth(null);
    }

    if (tripsResult.status === "fulfilled") {
      const nextTrips = tripsResult.value.trips;
      setTrips(nextTrips);
      setSelectedTrip((current) =>
        nextTrips.includes(current) ? current : (nextTrips[0] ?? ""),
      );
    } else {
      setTrips([]);
      setSelectedTrip("");
    }

    const failures = [healthResult, tripsResult]
      .filter((result) => result.status === "rejected")
      .map((result) => result.reason?.message ?? "Backend request failed.");
    if (failures.length > 0) {
      setBackendError([...new Set(failures)].join(" "));
    } else if (!healthResult.value.ready) {
      setBackendError(
        "The API responded, but reports that the model or demo partition is not ready.",
      );
    } else if (tripsResult.value.trips.length === 0) {
      setBackendError("The configured demo partition contains no trips.");
    }
    setIsChecking(false);
  }, []);

  useEffect(() => {
    refreshBackend();
  }, [refreshBackend]);

  const confidence = useMemo(
    () => getConfidenceSummary(prediction?.confidence),
    [prediction],
  );

  async function handlePredict(event) {
    event.preventDefault();
    if (!selectedTrip || !backendReady || isPredicting) return;

    setIsPredicting(true);
    setPredictionError("");
    setPrediction(null);
    try {
      const result = await predictTrip(selectedTrip);
      setPrediction(result);
    } catch (error) {
      setPredictionError(error?.message ?? "Prediction failed. Please retry.");
    } finally {
      setIsPredicting(false);
    }
  }

  function handleTripChange(event) {
    setSelectedTrip(event.target.value);
    setPrediction(null);
    setPredictionError("");
  }

  const connectionLabel = !configured
    ? "API URL required"
    : isChecking
      ? "Checking backend"
      : backendReady
        ? "Backend connected"
        : health
          ? "Backend not ready"
          : "Backend offline";
  const decoderLabel = prediction?.decoder?.toLowerCase().includes("hmm")
    ? "HMM / Viterbi"
    : "Random Forest";

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="PathSnap home">
          <span className="brand-mark">
            <Route size={21} strokeWidth={2.2} />
          </span>
          <span className="brand-copy">
            <strong>PathSnap<span>ML</span></strong>
            <small>MAP MATCHING LAB</small>
          </span>
        </a>

        <div className="topbar-center">
          <span className="environment-pill">
            <span className="environment-dot" />
            LIVE INFERENCE
          </span>
        </div>

        <div className={`connection-pill ${backendReady ? "is-online" : "is-offline"}`}>
          {backendReady ? <Wifi size={15} /> : <WifiOff size={15} />}
          <span>{connectionLabel}</span>
          <button
            type="button"
            className="icon-button connection-refresh"
            onClick={refreshBackend}
            disabled={isChecking}
            aria-label="Refresh backend status"
            title="Refresh backend status"
          >
            <RefreshCw size={14} className={isChecking ? "spin" : ""} />
          </button>
        </div>
      </header>

      <section className="page-heading">
        <div>
          <div className="eyebrow"><Activity size={14} /> INFERENCE WORKSPACE</div>
          <h1>GPS map matching</h1>
          <p>Match a trip to the road network with the Random Forest + HMM pipeline.</p>
        </div>
        <div className={`decoder-status ${decoderIsHmm ? "decoder-ready" : ""}`}>
          <span className="decoder-icon"><Sparkles size={16} /></span>
          <span>
            <strong>{decoderIsHmm ? "HMM / Viterbi" : "Decoder status"}</strong>
            <small>{decoderIsHmm ? "Sequence decoding available" : "Awaiting backend status"}</small>
          </span>
          <span className="status-indicator" />
        </div>
      </section>

      {!configured && (
        <div className="notice-banner" role="alert">
          <AlertCircle size={18} />
          <div>
            <strong>Backend URL is not configured</strong>
            <span>
              Copy <code>.env.example</code> to <code>.env</code> and set{" "}
              <code>VITE_API_BASE_URL</code> to Laptop 1&apos;s LAN address.
            </span>
          </div>
        </div>
      )}

      {backendError && configured && (
        <div className="notice-banner" role="alert">
          <AlertCircle size={18} />
          <div>
            <strong>Backend connection issue</strong>
            <span>{backendError}</span>
          </div>
          <button
            type="button"
            className="text-button"
            onClick={refreshBackend}
            disabled={isChecking}
          >
            Retry
          </button>
        </div>
      )}

      <div className="workspace-grid">
        <aside className="control-column">
          <section className="panel trip-panel">
            <div className="panel-heading">
              <div className="panel-icon"><MapPinned size={17} /></div>
              <div>
                <h2>Trip selection</h2>
                <p>Choose a trip available on the backend</p>
              </div>
            </div>

            <form onSubmit={handlePredict}>
              <label className="field-label" htmlFor="trip-select">TRIP ID</label>
              <div className="select-wrap">
                <select
                  id="trip-select"
                  value={selectedTrip}
                  onChange={handleTripChange}
                  disabled={!backendReady || isChecking || isPredicting}
                >
                  {trips.length === 0 ? (
                    <option value="">
                      {isChecking ? "Loading trips…" : "No trips available"}
                    </option>
                  ) : (
                    trips.map((tripId) => (
                      <option key={tripId} value={tripId}>{tripId}</option>
                    ))
                  )}
                </select>
                <ChevronDown size={16} />
              </div>

              <div className="partition-note">
                <Server size={14} />
                <span>Partition: <strong>{health?.demo_partition ?? "—"}</strong></span>
                <span className="trip-count">{trips.length} trips</span>
              </div>

              <button
                type="submit"
                className="run-button"
                disabled={!backendReady || !selectedTrip || isPredicting}
              >
                {isPredicting ? (
                  <>
                    <span className="spinner" />
                    Running HMM/Viterbi…
                  </>
                ) : (
                  <>
                    <ArrowDownToLine size={17} />
                    Run prediction
                  </>
                )}
              </button>
            </form>

            {predictionError && (
              <div className="inline-error" role="alert">
                <AlertCircle size={16} />
                <span>{predictionError}</span>
              </div>
            )}
          </section>

          <section className="panel pipeline-panel">
            <div className="panel-heading compact-heading">
              <div className="panel-icon"><Radio size={17} /></div>
              <div>
                <h2>Pipeline status</h2>
                <p>Live backend readiness</p>
              </div>
            </div>
            <div className="pipeline-row">
              <span className={`pipeline-status-dot ${health?.model_loaded ? "active" : ""}`} />
              <span>Random Forest model</span>
              <strong>{health?.model_loaded ? "Loaded" : "Unavailable"}</strong>
            </div>
            <div className="pipeline-row">
              <span className={`pipeline-status-dot ${decoderIsHmm ? "active" : ""}`} />
              <span>HMM / Viterbi</span>
              <strong>{decoderIsHmm ? "Ready" : "Unavailable"}</strong>
            </div>
            <div className="pipeline-row">
              <span className={`pipeline-status-dot ${backendReady ? "active" : ""}`} />
              <span>Demo trip partition</span>
              <strong>{backendReady ? "Connected" : "Waiting"}</strong>
            </div>
          </section>

          <div className="privacy-note">
            <CheckCircle2 size={15} />
            <span>Predictions are served by the backend. No reference route or accuracy is shown.</span>
          </div>
        </aside>

        <section className="main-column">
          <section className="panel map-panel">
            <div className="map-panel-header">
              <div className="panel-heading compact-heading">
                <div className="panel-icon map-panel-icon"><MapPinned size={17} /></div>
                <div>
                  <h2>Route visualization</h2>
                  <p>
                    {prediction
                      ? `Trip ${prediction.trip_id} · HMM/Viterbi prediction`
                      : "Run a prediction to display GPS and decoded road arcs"}
                  </p>
                </div>
              </div>
              {prediction && (
                <span className="result-badge">
                  <span className="result-badge-dot" />
                  PREDICTION COMPLETE
                </span>
              )}
            </div>

            <div className="map-frame">
              {prediction ? (
                <RouteMap
                  prediction={prediction}
                  fitToken={fitToken}
                  onFit={() => setFitToken((value) => value + 1)}
                />
              ) : (
                <div className="map-placeholder">
                  <div className="placeholder-rings"><MapPinned size={27} /></div>
                  <strong>{isPredicting ? "Matching GPS to the road network" : "Map awaits a trip"}</strong>
                  <span>
                    {isPredicting
                      ? "Generating candidates and decoding the route sequence."
                      : "Select an available trip and run an HMM/Viterbi prediction."}
                  </span>
                  {isPredicting && <span className="loading-track"><i /></span>}
                </div>
              )}
            </div>
            <div className="map-footer">
              <span><span className="footer-dot gps-footer-dot" /> GPS observations</span>
              <span><span className="footer-dot arc-footer-dot" /> Predicted directed arcs</span>
              <span className="map-source">OpenStreetMap tiles</span>
            </div>
          </section>

          <section className="results-section">
            <div className="results-title">
              <div>
                <div className="eyebrow"><Gauge size={14} /> PREDICTION OUTPUT</div>
                <h2>Run summary</h2>
              </div>
              {prediction && (
                <span className="hmm-tag"><Sparkles size={13} /> HMM / VITERBI</span>
              )}
            </div>

            <div className="metrics-grid">
              <MetricCard
                icon={<MapPinned size={17} />}
                label="Trip ID"
                value={prediction?.trip_id ?? "—"}
                detail="Selected input trip"
              />
              <MetricCard
                icon={<Activity size={17} />}
                label="GPS observations"
                value={prediction?.summary?.total_observations?.toLocaleString() ?? "—"}
                detail="Input trajectory points"
              />
              <MetricCard
                icon={<Route size={17} />}
                label="Decoded observations"
                value={prediction?.summary?.successfully_predicted_observations?.toLocaleString() ?? "—"}
                detail="Observations with a selected arc"
              />
              <MetricCard
                icon={<Layers size={17} />}
                label="Predicted road arcs"
                value={prediction?.predicted_route?.features?.length?.toLocaleString() ?? "—"}
                detail="Separate directed features"
              />
              <MetricCard
                icon={<Clock3 size={17} />}
                label="Inference duration"
                value={formatDuration(prediction?.summary?.inference_duration_s)}
                detail="Backend-reported runtime"
              />
              <MetricCard
                icon={<Sparkles size={17} />}
                label="Mean confidence"
                value={confidence ? `${(confidence.mean * 100).toFixed(1)}%` : "—"}
                detail={
                  confidence
                    ? `Range ${(confidence.min * 100).toFixed(1)}–${(confidence.max * 100).toFixed(1)}%`
                    : "Classifier probability for selected arcs"
                }
              />
            </div>

            {prediction && (
              <div className="result-footnote">
                <CheckCircle2 size={15} />
                <span>
                  {decoderLabel} output from the live API. Confidence is the classifier&apos;s
                  probability for the selected candidate, not an accuracy score.
                </span>
              </div>
            )}
          </section>
        </section>
      </div>

      <footer className="app-footer">
        <span>PathSnap-ML <span className="footer-separator">/</span> Two-laptop inference demo</span>
        <span><span className="footer-dot active-footer-dot" /> Batch prediction · Ground truth stays server-side</span>
      </footer>
    </main>
  );
}

function MetricCard({ icon, label, value, detail }) {
  return (
    <article className="metric-card">
      <div className="metric-card-top">
        <span className="metric-icon">{icon}</span>
        <span className="metric-label">{label}</span>
      </div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-detail">{detail}</span>
    </article>
  );
}
