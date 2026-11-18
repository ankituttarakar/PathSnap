"""Visualisation for PathSnap EDA (Phase 1) and later route maps (Phase 9).

Two groups of helpers:
    * matplotlib summary charts (:func:`make_eda_charts`) — written as PNGs.
    * Folium interactive maps (:func:`folium_trip_map`) — written as HTML.

All figures are generated from actual computed data; nothing is fabricated.
matplotlib uses the non-interactive ``Agg`` backend so charts render headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless, must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pathsnap.data_io import Trip  # noqa: E402
from pathsnap.eda import haversine_m  # noqa: E402

# Folium map context limits (keep HTML readable and reasonably sized).
_MAX_CONTEXT_ARCS = 1500
_GPS_MARKER_STRIDE = 10  # draw every Nth GPS point as a marker (polyline is full)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def make_eda_charts(
    train_df,
    intervals_s: np.ndarray,
    speeds_ms: np.ndarray,
    out_dir: Path,
) -> List[Path]:
    """Create the five required EDA charts from the TRAIN partition.

    Returns the list of written PNG paths. ``intervals_s`` and ``speeds_ms`` are
    the concatenated train arrays returned by :func:`pathsnap.eda.analyze_partition`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # 1. Observations per trip.
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(train_df)), train_df["n_obs"].to_numpy(), color="#3b76af")
    ax.set_title("GPS observations per trip (train, n=%d)" % len(train_df))
    ax.set_xlabel("trip (sorted by id)")
    ax.set_ylabel("observations")
    p = out_dir / "observations_per_trip.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    # 2. Timestamp interval distribution (clip tail for readability; log count).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    clipped = np.clip(intervals_s, 0, 15)
    ax.hist(clipped, bins=np.arange(-0.5, 16.5, 1.0), color="#3b76af", edgecolor="white")
    ax.set_yscale("log")
    frac_1s = float(np.mean(np.isclose(intervals_s, 1.0))) if intervals_s.size else float("nan")
    ax.set_title(f"Inter-sample Δt distribution (train)\n{frac_1s:.1%} of steps are exactly 1 s; tail clipped at 15 s")
    ax.set_xlabel("Δt between consecutive observations (s)")
    ax.set_ylabel("count (log scale)")
    p = out_dir / "timestamp_interval_distribution.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    # 3. GPS speed distribution (clip to vehicle range; note outliers beyond).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    finite = speeds_ms[np.isfinite(speeds_ms)]
    beyond = int(np.sum(finite > 60))
    ax.hist(np.clip(finite, 0, 60), bins=60, color="#4a9b5e", edgecolor="white")
    ax.set_title(f"Estimated consecutive-point speed (train)\n{beyond} segments exceed 60 m/s (shown clipped)")
    ax.set_xlabel("speed (m/s)  [geodesic / Δt]")
    ax.set_ylabel("count")
    p = out_dir / "gps_speed_distribution.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    # 4. Network size distribution (nodes & arcs).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(train_df["n_nodes"].to_numpy(), bins=20, color="#8a63b5", edgecolor="white")
    axes[0].set_title("Nodes per trip (train)"); axes[0].set_xlabel("nodes"); axes[0].set_ylabel("trips")
    axes[1].hist(train_df["n_arcs"].to_numpy(), bins=20, color="#b5636b", edgecolor="white")
    axes[1].set_title("Arcs per trip (train)"); axes[1].set_xlabel("arcs"); axes[1].set_ylabel("trips")
    p = out_dir / "network_size_distribution.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    # 5. Route length distribution.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(train_df["route_len"].to_numpy(), bins=20, color="#c08437", edgecolor="white")
    ax.set_title("Ground-truth route length per trip (train)")
    ax.set_xlabel("route length (number of arcs)")
    ax.set_ylabel("trips")
    p = out_dir / "route_length_distribution.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    return paths


# --------------------------------------------------------------------------- #
# Folium maps
# --------------------------------------------------------------------------- #
def _legend_html(title: str) -> str:
    """Return a small fixed HTML legend for a trip map."""
    return f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: white; padding: 8px 12px; border: 1px solid #888;
                border-radius: 4px; font-size: 12px; font-family: sans-serif;">
      <b>{title}</b><br>
      <span style="color:#1f6fd6;">&#9679;</span> Raw GPS observations<br>
      <span style="color:#2e8b3d;">&#9644;</span> Reference route (ground truth)<br>
      <span style="color:#999;">&#9644;</span> Road-network context (sampled)
    </div>
    """


def folium_trip_map(trip: Trip, out_path: Path, role: str = "") -> Path:
    """Render an interactive Folium map for a trip and write it to ``out_path``.

    Layers (with a layer control and legend):
        * raw GPS observations (polyline + sampled markers);
        * the ground-truth reference route (arc geometry from ``.route``);
        * a sampled subset of the surrounding road network for context.

    The road-network context is capped at ``_MAX_CONTEXT_ARCS`` arcs within the
    track bounding box to keep the HTML readable. This is visualisation only —
    it does not derive labels.
    """
    import folium  # local import keeps module import light

    out_path.parent.mkdir(parents=True, exist_ok=True)
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    lon, lat = track[:, 0], track[:, 1]
    center = [float(lat.mean()), float(lon.mean())]

    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    title = f"Trip {trip.trip_id} ({trip.partition})" + (f" — {role}" if role else "")

    # --- Road-network context (sampled arcs within padded bbox) ---
    net_layer = folium.FeatureGroup(name="Road network (sampled)", show=True)
    pad = 0.002
    lo_lon, hi_lon = lon.min() - pad, lon.max() + pad
    lo_lat, hi_lat = lat.min() - pad, lat.max() + pad
    frm, to = arcs[:, 0], arcs[:, 1]
    f_in = (
        (nodes[frm, 0] >= lo_lon) & (nodes[frm, 0] <= hi_lon)
        & (nodes[frm, 1] >= lo_lat) & (nodes[frm, 1] <= hi_lat)
    )
    t_in = (
        (nodes[to, 0] >= lo_lon) & (nodes[to, 0] <= hi_lon)
        & (nodes[to, 1] >= lo_lat) & (nodes[to, 1] <= hi_lat)
    )
    in_box = np.where(f_in & t_in)[0]
    drawn_context = int(min(in_box.size, _MAX_CONTEXT_ARCS))
    for aid in in_box[:_MAX_CONTEXT_ARCS]:
        a, b = int(frm[aid]), int(to[aid])
        folium.PolyLine(
            [(nodes[a, 1], nodes[a, 0]), (nodes[b, 1], nodes[b, 0])],
            color="#999999", weight=1, opacity=0.5,
        ).add_to(net_layer)
    net_layer.add_to(fmap)

    # --- Reference route (ground truth) ---
    route_layer = folium.FeatureGroup(name="Reference route (ground truth)", show=True)
    for aid in trip.route:
        if 0 <= aid < arcs.shape[0]:
            a, b = int(frm[aid]), int(to[aid])
            folium.PolyLine(
                [(nodes[a, 1], nodes[a, 0]), (nodes[b, 1], nodes[b, 0])],
                color="#2e8b3d", weight=4, opacity=0.9,
            ).add_to(route_layer)
    route_layer.add_to(fmap)

    # --- Raw GPS observations ---
    gps_layer = folium.FeatureGroup(name="Raw GPS observations", show=True)
    folium.PolyLine(
        list(zip(lat.tolist(), lon.tolist())), color="#1f6fd6", weight=2, opacity=0.7
    ).add_to(gps_layer)
    for i in range(0, len(track), _GPS_MARKER_STRIDE):
        folium.CircleMarker(
            [float(lat[i]), float(lon[i])], radius=2, color="#1f6fd6",
            fill=True, fill_opacity=0.8,
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_legend_html(title)))

    # Fit bounds to the GPS track.
    fmap.fit_bounds([[lo_lat, lo_lon], [hi_lat, hi_lon]])
    fmap.save(str(out_path))
    return out_path
