import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";
import { getHealth, getTrips, predictTrip } from "./api.js";

vi.mock("./api.js", () => ({
  getHealth: vi.fn(),
  getTrips: vi.fn(),
  hasApiBaseUrl: vi.fn(() => true),
  predictTrip: vi.fn(),
}));

vi.mock("./components/RouteMap.jsx", () => ({
  default: () => <div data-testid="route-map">Map rendered</div>,
}));

const predictionResponse = {
  trip_id: "00000083",
  decoder: "random_forest_hmm_viterbi",
  predicted_arc_ids: [17],
  confidence: [0.91],
  gps: { type: "LineString", coordinates: [[10, 20]] },
  predicted_route: { type: "FeatureCollection", features: [{}, {}] },
  summary: {
    total_observations: 1,
    successfully_predicted_observations: 1,
    inference_duration_s: 0.25,
  },
};

describe("PathSnap dashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getHealth.mockResolvedValue({
      ready: true,
      model_loaded: true,
      prediction_mode: "classifier_plus_hmm_viterbi_batch",
      demo_partition: "train",
    });
    getTrips.mockResolvedValue({
      partition: "train",
      trips: ["00000083", "00000084"],
    });
  });

  it("loads real trip choices and renders the returned prediction summary", async () => {
    predictTrip.mockResolvedValue(predictionResponse);
    render(<App />);

    await screen.findByRole("option", { name: "00000083" });
    expect(screen.getByText("Backend connected")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /run prediction/i }));

    await screen.findByTestId("route-map");
    expect(predictTrip).toHaveBeenCalledWith("00000083");
    expect(screen.getByText("HMM / Viterbi output from the live API.", { exact: false }))
      .toBeInTheDocument();
    expect(screen.getByText("91.0%")).toBeInTheDocument();
    expect(screen.queryByText("Accuracy")).not.toBeInTheDocument();
  });

  it("shows loading state while prediction is pending", async () => {
    let resolvePrediction;
    predictTrip.mockReturnValue(
      new Promise((resolve) => {
        resolvePrediction = resolve;
      }),
    );
    render(<App />);
    await screen.findByRole("option", { name: "00000083" });

    fireEvent.click(screen.getByRole("button", { name: /run prediction/i }));
    expect(
      screen.getByRole("button", { name: /running hmm\/viterbi/i }),
    ).toBeDisabled();

    resolvePrediction(predictionResponse);
    await waitFor(() => expect(screen.getByTestId("route-map")).toBeInTheDocument());
  });

  it("shows backend prediction errors and allows retry", async () => {
    predictTrip.mockRejectedValueOnce(new Error("Trip is unavailable"));
    render(<App />);
    await screen.findByRole("option", { name: "00000083" });

    fireEvent.click(screen.getByRole("button", { name: /run prediction/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Trip is unavailable",
    );
    expect(screen.queryByTestId("route-map")).not.toBeInTheDocument();
  });
});
