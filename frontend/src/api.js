const REQUEST_TIMEOUT_MS = 15_000;

export function getApiBaseUrl() {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  return configured ? configured.replace(/\/+$/, "") : "";
}

export function hasApiBaseUrl() {
  return getApiBaseUrl().length > 0;
}

async function request(path, options = {}) {
  const baseUrl = getApiBaseUrl();
  if (!baseUrl) {
    throw new Error(
      "Backend URL is not configured. Set VITE_API_BASE_URL in frontend/.env.",
    );
  }

  const controller = new AbortController();
  const timeoutId = window.setTimeout(
    () => controller.abort(),
    REQUEST_TIMEOUT_MS,
  );

  let response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...options.headers,
      },
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("The backend request timed out. Check the network and retry.");
    }
    throw new Error(
      `Could not reach the backend at ${baseUrl}. Check its LAN address, port, and CORS settings.`,
    );
  } finally {
    window.clearTimeout(timeoutId);
  }

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof payload?.detail === "string"
        ? payload.detail
        : `Backend request failed with status ${response.status}.`;
    throw new Error(detail);
  }

  return payload;
}

export async function getHealth() {
  const health = await request("/health");
  if (
    typeof health?.ready !== "boolean" ||
    typeof health?.prediction_mode !== "string"
  ) {
    throw new Error("The backend returned an unexpected health response.");
  }
  return health;
}

export async function getTrips() {
  const result = await request("/trips");
  if (
    !Array.isArray(result?.trips) ||
    result.trips.some((tripId) => typeof tripId !== "string")
  ) {
    throw new Error("The backend returned an unexpected trips response.");
  }
  return result;
}

export async function predictTrip(tripId) {
  const result = await request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trip_id: tripId }),
  });

  if (
    result?.predicted_route?.type !== "FeatureCollection" ||
    !Array.isArray(result.predicted_route.features) ||
    result?.gps?.type !== "LineString" ||
    !Array.isArray(result.gps.coordinates) ||
    !Array.isArray(result?.confidence) ||
    !result?.summary
  ) {
    throw new Error("The backend returned an unexpected prediction response.");
  }
  return result;
}
