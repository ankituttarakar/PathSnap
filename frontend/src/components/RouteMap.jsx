import { useEffect, useMemo } from "react";
import { Layers, LocateFixed, MapPinned } from "lucide-react";
import L from "leaflet";
import {
  CircleMarker,
  MapContainer,
  Polyline,
  TileLayer,
  Tooltip,
  useMap,
} from "react-leaflet";
import { buildMapLayers } from "../mapUtils.js";

function FitRoute({ positions, fitToken }) {
  const map = useMap();

  useEffect(() => {
    if (positions.length === 1) {
      map.setView(positions[0], 15);
      return;
    }
    if (positions.length > 1) {
      map.fitBounds(L.latLngBounds(positions), {
        padding: [42, 42],
        maxZoom: 16,
      });
    }
  }, [fitToken, map, positions]);

  return null;
}

export default function RouteMap({ prediction, fitToken, onFit }) {
  const { gpsPositions, arcSegments, fitPositions } = useMemo(
    () => buildMapLayers(prediction),
    [prediction],
  );

  if (gpsPositions.length === 0) {
    return (
      <div className="map-empty" role="status">
        <MapPinned size={30} />
        <strong>No map data to display</strong>
        <span>The prediction response did not contain usable GPS coordinates.</span>
      </div>
    );
  }

  const start = gpsPositions[0];
  const finish = gpsPositions[gpsPositions.length - 1];

  return (
    <div className="map-shell">
      <MapContainer
        center={start}
        zoom={13}
        scrollWheelZoom
        className="route-map"
        preferCanvas
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FitRoute positions={fitPositions} fitToken={fitToken} />

        <Polyline
          positions={gpsPositions}
          pathOptions={{
            color: "#38bdf8",
            weight: 4,
            opacity: 0.9,
            dashArray: "9 8",
          }}
        />

        {arcSegments.map((segment) => (
          <Polyline
            key={segment.key}
            positions={segment.positions}
            pathOptions={{
              color: "#f6ad55",
              weight: 5,
              opacity: Number.isFinite(segment.confidence)
                ? 0.48 + 0.48 * segment.confidence
                : 0.85,
            }}
          >
            <Tooltip>
              Directed arc {segment.arcId} · observation{" "}
              {segment.observationIndex}
              {Number.isFinite(segment.confidence)
                ? ` · classifier confidence ${(segment.confidence * 100).toFixed(1)}%`
                : ""}
            </Tooltip>
          </Polyline>
        ))}

        <CircleMarker
          center={start}
          radius={8}
          pathOptions={{
            color: "#07120d",
            weight: 2,
            fillColor: "#4ade80",
            fillOpacity: 1,
          }}
        >
          <Tooltip permanent direction="top" offset={[0, -7]}>
            Start
          </Tooltip>
        </CircleMarker>
        <CircleMarker
          center={finish}
          radius={8}
          pathOptions={{
            color: "#1c0c10",
            weight: 2,
            fillColor: "#fb7185",
            fillOpacity: 1,
          }}
        >
          <Tooltip permanent direction="top" offset={[0, -7]}>
            Finish
          </Tooltip>
        </CircleMarker>
      </MapContainer>

      <button
        className="map-fit-button"
        type="button"
        onClick={onFit}
        aria-label="Fit map to route"
        title="Fit map to route"
      >
        <LocateFixed size={16} />
        Fit route
      </button>

      <div className="map-legend" aria-label="Map legend">
        <div className="legend-title">
          <Layers size={14} />
          Layers
        </div>
        <div className="legend-item">
          <span className="legend-line gps-line" />
          GPS trajectory
        </div>
        <div className="legend-item">
          <span className="legend-line arc-line" />
          HMM/Viterbi road arcs
        </div>
        <div className="legend-item">
          <span className="legend-dot start-dot" />
          Start
          <span className="legend-dot finish-dot" />
          Finish
        </div>
      </div>
      <div className="map-attribution-note">Map tiles © OpenStreetMap contributors</div>
    </div>
  );
}
