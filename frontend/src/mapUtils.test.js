import { describe, expect, it } from "vitest";
import { buildMapLayers, geoJsonCoordinateToLatLng } from "./mapUtils.js";

describe("GeoJSON map conversion", () => {
  it("converts [longitude, latitude] into Leaflet [latitude, longitude]", () => {
    expect(geoJsonCoordinateToLatLng([37.62, 55.75])).toEqual([55.75, 37.62]);
  });

  it("keeps each returned road arc as a separate line segment", () => {
    const layers = buildMapLayers({
      gps: {
        type: "LineString",
        coordinates: [[10, 20], [11, 21]],
      },
      predicted_route: {
        type: "FeatureCollection",
        features: [
          {
            properties: { arc_id: 17, obs_idx: 0, confidence: 0.8 },
            geometry: { type: "LineString", coordinates: [[10, 20], [10.5, 20.5]] },
          },
          {
            properties: { arc_id: 91, obs_idx: 1, confidence: 0.7 },
            geometry: { type: "LineString", coordinates: [[12, 22], [12.5, 22.5]] },
          },
        ],
      },
    });

    expect(layers.gpsPositions).toEqual([[20, 10], [21, 11]]);
    expect(layers.arcSegments).toHaveLength(2);
    expect(layers.arcSegments[0].positions).toEqual([[20, 10], [20.5, 10.5]]);
    expect(layers.arcSegments[1].positions).toEqual([[22, 12], [22.5, 12.5]]);
    expect(layers.arcSegments[0].arcId).toBe(17);
    expect(layers.arcSegments[1].arcId).toBe(91);
  });

  it("rejects coordinates outside WGS84 bounds", () => {
    expect(geoJsonCoordinateToLatLng([181, 30])).toBeNull();
    expect(geoJsonCoordinateToLatLng([10, 91])).toBeNull();
  });
});
