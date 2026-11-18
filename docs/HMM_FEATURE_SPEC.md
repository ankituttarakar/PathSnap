# PathSnap feature specification for HMM map matching

The candidate classifier will use 18 features: the existing 12 plus six
inference-time features. None is derived from `.route`, reference labels, or a
student/evaluator target. `y` and `confident` remain training/evaluation columns,
not model features.

| Feature | Measurement and calculation | Data used | Inference available | Future observations | Ground-truth leakage risk |
|---|---|---|---|---|---|
| `dist_m` | Perpendicular distance from the current GPS fix to the candidate segment, metres | Current GPS, candidate geometry | Yes | No | None |
| `t` | Clamped fractional projection position along the candidate segment, 0 to 1 | Current GPS, candidate geometry | Yes | No | None |
| `rank` | Candidate order by point-to-segment distance within the current observation; 0 is nearest | Candidate set | Yes | No | None |
| `n_cand` | Number of retained candidates for the current observation | Candidate set | Yes | No | None |
| `dist_margin_m` | Candidate `dist_m` minus the nearest candidate distance for the observation | Current GPS, candidate set | Yes | No | None |
| `arc_len` | Straight-line length of the directed candidate segment, metres | Candidate geometry | Yes | No | None |
| `heading_diff` | Absolute GPS-course/arc-bearing difference folded to `[0, pi/2]`; neutral `pi/2` when course is unavailable | Previous/current GPS, candidate geometry | Yes | No | None |
| `speed_ms` | Consecutive geodesic GPS distance divided by elapsed time, metres/second | Previous/current GPS and timestamps | Yes | No | None |
| `dt_s` | Elapsed time since the preceding GPS fix, seconds | Previous/current timestamps | Yes | No | None |
| `is_outlier` | Existing binary flag when computed GPS speed exceeds the existing 55 m/s flag value | Previous/current GPS and timestamps | Yes | No | None |
| `seg_start` | 1 at the first fix or after the existing `dt > 30 s` segment boundary | Previous/current timestamps | Yes | No | None |
| `cum_gps_frac` | Cumulative trip GPS distance divided by total trip GPS distance | All GPS fixes in the trip | Yes, in batch | **Yes**; total distance needs the complete trip | None |
| `heading_diff_directed_rad` | Smallest absolute angular difference in `[0, pi]` between current GPS course and the directed candidate bearing; neutral `pi/2` when course is unavailable | Previous/current GPS, candidate geometry | Yes | No | None |
| `signed_cross_track_m` | Signed perpendicular offset from the directed segment; positive is left of the from-node-to-to-node direction | Current GPS, candidate geometry | Yes | No | None |
| `previous_point_dist_m` | Distance from the previous fix to the current candidate segment; 0 at a segment start | Previous/current GPS, candidate geometry | Yes | No | None |
| `projected_speed_along_arc_ms` | Signed change in projection distance on the current candidate segment divided by `dt`; 0 at a segment start or invalid interval | Previous/current GPS, timestamps, candidate geometry | Yes | No | None |
| `projected_speed_error_ms` | Absolute difference between `speed_ms` and signed `projected_speed_along_arc_ms`; 0 where the projected speed is unavailable | Previous/current GPS, timestamps, candidate geometry | Yes | No | None |
| `gps_heading_change_rad` | Absolute wrapped angle between the current and preceding GPS course, within the same segment; 0 when either course is unavailable | Previous GPS fixes and timestamps | Yes | No | None |

The six added values are candidate-row classifier inputs. HMM transition
compatibility (directed connectivity and distance between candidate states) is
computed separately and is **not** counted among the 18 features.

This is a fixed specification, not a feature-selection result. Any future
feature selection or transition parameter fitting must use train-only
GroupKFold. Student-test labels are reserved for a single final held-out
evaluation after implementation decisions are frozen.
