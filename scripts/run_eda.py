"""Run PathSnap Phase 1 EDA reproducibly.

Analyses all 80 training trips and, separately, the 10 student_test trips
(descriptive only — never mixed into train). Writes per-trip stat tables, a
partition comparison, summary charts (train), three representative Folium maps
(train), and a machine-readable aggregate summary.

Usage (from the repository root):

    python scripts/run_eda.py

No raw data is modified; no model is trained; no observation labels are derived.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathsnap.config import CONFIG  # noqa: E402
from pathsnap.data_io import load_trip  # noqa: E402
from pathsnap.eda import (  # noqa: E402
    analyze_partition,
    select_representative_trips,
)
from pathsnap.logging_utils import get_logger  # noqa: E402
from pathsnap.viz import folium_trip_map, make_eda_charts  # noqa: E402

logger = get_logger("pathsnap.run_eda", log_file="run_eda.log")

# Aggregate columns to summarise in the comparison table.
_NUMERIC_SUMMARY_COLS = [
    "n_obs", "duration_s", "frac_1s_steps", "n_gaps_gt_1s", "max_gap_s",
    "n_nodes", "n_arcs", "density_arcs_per_node", "frac_arcs_with_reverse",
    "route_len", "route_len_over_obs", "frac_connected", "bbox_diag_km",
    "max_jump_m", "median_speed_ms",
]


def _agg(df: pd.DataFrame) -> dict:
    """Aggregate per-trip stats for a partition into a compact dict."""
    out = {"n_trips": int(len(df)), "total_observations": int(df["n_obs"].sum())}
    for col in ["n_obs", "n_nodes", "n_arcs", "route_len", "bbox_diag_km"]:
        out[f"{col}_min"] = float(df[col].min())
        out[f"{col}_median"] = float(df[col].median())
        out[f"{col}_mean"] = float(df[col].mean())
        out[f"{col}_max"] = float(df[col].max())
        out[f"{col}_std"] = float(df[col].std())
    return out


def main() -> int:
    CONFIG.paths.ensure_output_dirs()
    eda_dir = CONFIG.paths.outputs_eda
    maps_dir = CONFIG.paths.outputs_maps

    # ----- TRAIN (full analysis) -----
    logger.info("Analysing TRAIN partition (80 trips)...")
    train_df, train_intervals, train_speeds = analyze_partition("train")
    train_df.to_csv(eda_dir / "train_trip_stats.csv", index=False)

    # ----- STUDENT_TEST (descriptive only, kept separate) -----
    logger.info("Analysing STUDENT_TEST partition (10 trips, descriptive only)...")
    st_df, _, _ = analyze_partition("student_test")
    st_df.to_csv(eda_dir / "student_test_trip_stats.csv", index=False)

    # ----- Partition comparison table -----
    comp = pd.DataFrame(
        {
            "train": train_df[_NUMERIC_SUMMARY_COLS].mean(),
            "student_test": st_df[_NUMERIC_SUMMARY_COLS].mean(),
        }
    )
    comp.to_csv(eda_dir / "partition_comparison.csv")

    # ----- Charts (TRAIN only) -----
    chart_paths = make_eda_charts(train_df, train_intervals, train_speeds, eda_dir)

    # ----- Representative Folium maps (TRAIN only) -----
    reps = select_representative_trips(train_df)
    map_paths = {}
    for role, tid in reps.items():
        trip = load_trip(tid, "train")
        out = maps_dir / f"trip_{tid}_{role}.html"
        folium_trip_map(trip, out, role=role)
        map_paths[role] = {"trip_id": tid, "path": str(out.relative_to(ROOT))}

    # ----- Machine-readable summary -----
    summary = {
        "train": _agg(train_df),
        "student_test": _agg(st_df),
        "representative_trips": map_paths,
        "route_connectivity": {
            "train_trips_fully_connected": int((train_df["frac_connected"] >= 0.999).sum()),
            "train_min_frac_connected": float(train_df["frac_connected"].min()),
            "train_mean_frac_connected": float(train_df["frac_connected"].mean()),
            "train_total_broken_links": int(train_df["n_broken_links"].sum()),
            "st_trips_fully_connected": int((st_df["frac_connected"] >= 0.999).sum()),
            "st_mean_frac_connected": float(st_df["frac_connected"].mean()),
        },
        "data_quality": {
            "train_total_gaps_gt_1s": int(train_df["n_gaps_gt_1s"].sum()),
            "train_max_gap_s": float(train_df["max_gap_s"].max()),
            "train_total_dup_timestamps": int(train_df["n_dup_timestamps"].sum()),
            "train_total_nonmonotonic": int(train_df["n_nonmonotonic"].sum()),
            "train_trips_with_confirmed_jump_error": int((train_df["n_confirmed_jump_error"] > 0).sum()),
            "train_total_confirmed_jump_errors": int(train_df["n_confirmed_jump_error"].sum()),
            "train_total_invalid_endpoint_refs": int(train_df["n_invalid_endpoint_refs"].sum()),
            "train_total_invalid_arc_refs": int(train_df["n_invalid_arc_refs"].sum()),
            "train_total_self_loops": int(train_df["n_self_loops"].sum()),
            "train_total_isolated_nodes": int(train_df["n_isolated_nodes"].sum()),
            "train_trips_single_component": int((train_df["n_weak_components"] == 1).sum()),
        },
        "charts": [str(p.relative_to(ROOT)) for p in chart_paths],
    }
    with open(eda_dir / "eda_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ----- Console validation summary -----
    print("\n" + "=" * 68)
    print("PHASE 1 EDA — SUMMARY")
    print("=" * 68)
    print(f"TRAIN trips analysed       : {summary['train']['n_trips']}  "
          f"(obs total {summary['train']['total_observations']:,})")
    print(f"STUDENT_TEST trips analysed: {summary['student_test']['n_trips']}  "
          f"(obs total {summary['student_test']['total_observations']:,})")
    print(f"Representative trips        : {reps}")
    print(f"Route connectivity (train)  : "
          f"{summary['route_connectivity']['train_trips_fully_connected']}/80 fully connected, "
          f"mean frac={summary['route_connectivity']['train_mean_frac_connected']:.4f}, "
          f"broken links total={summary['route_connectivity']['train_total_broken_links']}")
    print(f"Data quality (train)        : gaps>1s={summary['data_quality']['train_total_gaps_gt_1s']:,}, "
          f"max gap={summary['data_quality']['train_max_gap_s']:.0f}s, "
          f"dup ts={summary['data_quality']['train_total_dup_timestamps']}, "
          f"nonmono={summary['data_quality']['train_total_nonmonotonic']}")
    print(f"Confirmed GPS jump errors   : {summary['data_quality']['train_total_confirmed_jump_errors']} "
          f"across {summary['data_quality']['train_trips_with_confirmed_jump_error']} trips")
    print(f"Charts written              : {len(chart_paths)} -> {eda_dir.relative_to(ROOT)}")
    print(f"Maps written                : {len(map_paths)} -> {maps_dir.relative_to(ROOT)}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
