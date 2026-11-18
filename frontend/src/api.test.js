import { afterEach, describe, expect, it, vi } from "vitest";
import { getHealth, getTrips, predictTrip } from "./api.js";

function jsonResponse(body, options = {}) {
  return new Response(JSON.stringify(body), {
    status: options.status ?? 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("PathSnap API client", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("uses the configured LAN base URL for readiness and trips", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "http://192.168.1.20:8000/");
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(
          jsonResponse({
            ready: true,
            prediction_mode: "classifier_plus_hmm_viterbi_batch",
          }),
        )
        .mockResolvedValueOnce(
          jsonResponse({ partition: "train", trips: ["00000083"] }),
        ),
    );

    await expect(getHealth()).resolves.toMatchObject({ ready: true });
    await expect(getTrips()).resolves.toEqual({
      partition: "train",
      trips: ["00000083"],
    });
    expect(fetch.mock.calls.map(([url]) => url)).toEqual([
      "http://192.168.1.20:8000/health",
      "http://192.168.1.20:8000/trips",
    ]);
  });

  it("posts the selected trip and accepts the backend GeoJSON contract", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "http://192.168.1.20:8000");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          trip_id: "00000083",
          decoder: "random_forest_hmm_viterbi",
          predicted_arc_ids: [4],
          confidence: [0.91],
          gps: { type: "LineString", coordinates: [[10, 20]] },
          predicted_route: {
            type: "FeatureCollection",
            features: [],
          },
          summary: {
            total_observations: 1,
            successfully_predicted_observations: 1,
            inference_duration_s: 0.1,
          },
        }),
      ),
    );

    const prediction = await predictTrip("00000083");
    expect(prediction.decoder).toBe("random_forest_hmm_viterbi");
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      trip_id: "00000083",
    });
    expect(fetch.mock.calls[0][0]).toBe(
      "http://192.168.1.20:8000/predict",
    );
  });

  it("surfaces API errors instead of treating them as predictions", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "http://192.168.1.20:8000");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: "Trip is unavailable" }, { status: 404 }),
      ),
    );

    await expect(predictTrip("00000083")).rejects.toThrow(
      "Trip is unavailable",
    );
  });
});
