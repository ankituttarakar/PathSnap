export function geoJsonCoordinateToLatLng(coordinate) {
  if (
    !Array.isArray(coordinate) ||
    coordinate.length < 2 ||
    !Number.isFinite(coordinate[0]) ||
    !Number.isFinite(coordinate[1]) ||
    coordinate[0] < -180 ||
    coordinate[0] > 180 ||
    coordinate[1] < -90 ||
    coordinate[1] > 90
  ) {
    return null;
  }

  const [longitude, latitude] = coordinate;
  return [latitude, longitude];
}

export function buildMapLayers(prediction) {
  const gpsPositions = (prediction?.gps?.coordinates ?? [])
    .map(geoJsonCoordinateToLatLng)
    .filter(Boolean);

  const arcSegments = (prediction?.predicted_route?.features ?? [])
    .map((feature) => {
      const positions = (feature?.geometry?.coordinates ?? [])
        .map(geoJsonCoordinateToLatLng)
        .filter(Boolean);
      return {
        key: `${feature?.properties?.obs_idx}-${feature?.properties?.arc_id}`,
        arcId: feature?.properties?.arc_id,
        observationIndex: feature?.properties?.obs_idx,
        confidence: feature?.properties?.confidence,
        positions,
      };
    })
    .filter((segment) => segment.positions.length >= 2);

  return {
    gpsPositions,
    arcSegments,
    fitPositions: [
      ...gpsPositions,
      ...arcSegments.flatMap((segment) => segment.positions),
    ],
  };
}
