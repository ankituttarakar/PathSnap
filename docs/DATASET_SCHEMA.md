# PathSnap — Dataset Schema (Verified)

> All facts below were verified directly against the files in
> `data/raw/Student folder/` on 2026-11-18 using NumPy over **all 90 local trips**
> (80 train + 10 student_test). Nothing here is assumed from the PDFs alone.

## 1. Directory layout

```
data/raw/Student folder/
├── train/            80 trip folders  (00000001 … 00000098, non-contiguous)
│   └── <tripid>/
│        ├── <tripid>.track   GPS observations
│        ├── <tripid>.nodes   road-network node coordinates
│        ├── <tripid>.arcs    road-network directed arcs
│        └── <tripid>.route   ground-truth arc sequence
└── student_test/     10 trip folders  (00000000, 00000007, 00000009, 00000040,
     └── <tripid>/ …            00000052, 00000056, 00000065, 00000076, 00000077, 00000009)
```

- Trip IDs are **8-digit zero-padded** strings and are **not contiguous** (gaps in
  numbering: e.g. train has 01–14,16,18–22,24–29,31… ; the missing IDs are the
  student_test and evaluator-test trips drawn from the same global numbering).
- Every trip folder contains **exactly four files**, one of each type. Verified:
  train has 80/80/80/80 of `.track/.arcs/.nodes/.route`; student_test has 10 each.
- **The evaluator-test partition (10 trips) is NOT present in this repo.** Only
  `train` and `student_test` exist locally. This is correct and expected.

## 2. File schemas (whitespace-separated, **no headers**)

### `.track` — GPS observations (one row per observation)
| col | meaning | type | notes |
|-----|---------|------|-------|
| 0 | **longitude** | float | degrees, WGS84 |
| 1 | **latitude**  | float | degrees, WGS84 |
| 2 | **timestamp** | float | **seconds**, printed as `N.000000` |

- Column order is **lon, lat** (X, Y) — NOT lat, lon. Confirmed against the
  node/arc geometry.
- Timestamps are integers stored as floats, starting near `1.0`, nominally **1 Hz**.
- Verified: **0 NaN values**, **0 duplicate timestamps**, **0 non-monotonic
  timestamps** across all 90 trips. Timestamps are strictly increasing.
- Row count = number of observations for that trip (see §4). The timestamp of the
  last row may exceed the row count because of dropped samples (gaps).

### `.nodes` — road-network nodes
| col | meaning | type |
|-----|---------|------|
| 0 | longitude | float |
| 1 | latitude  | float |

- **The 0-based row index IS the `node_id`.** There is no explicit ID column.
- Same lon,lat order as `.track`.

### `.arcs` — directed road segments
| col | meaning | type |
|-----|---------|------|
| 0 | `from_node_id` | int |
| 1 | `to_node_id`   | int |

- **The 0-based row index IS the `arc_id`.**
- Arcs are **directed**: both `0 1` and `1 0` appear as separate rows, so a
  two-way street is represented by two arcs. (Verified in trip 00000001 head.)
- Verified: for every trip, `max(from/to node id) < number_of_nodes` and
  `min >= 0` → **no arc references a non-existent node** (0 violations / 90 trips).

### `.route` — ground-truth reference route
| col | meaning | type |
|-----|---------|------|
| 0 | `arc_id` | int |

- A single column: the **chronological sequence of arc_ids** the vehicle traversed.
- Verified: for every trip, `max(route arc_id) < number_of_arcs` and `min >= 0`
  → **every route entry indexes a valid arc** (0 violations / 90 trips).
- **student_test trips DO ship with `.route` files** (ground truth), because that
  partition is for development-stage evaluation/model comparison. The evaluator
  partition's routes are withheld server-side and are not in this repo.

## 3. ID correspondence (the key relationships)

```
.track row            = one GPS observation (lon,lat,t)   — NOT aligned to any other file's rows
.nodes row index      = node_id            → (lon, lat)
.arcs  row index      = arc_id             → (from_node_id, to_node_id)
.route row            = arc_id (into .arcs) → chronological ground-truth segment sequence
```

- **Do NOT assume row-alignment between `.track` and the network files.** A GPS
  observation is matched to an arc by geometry/model, not by row index. The number
  of track rows, route rows, node rows and arc rows are all different.
- The modelling target is: **for each GPS observation, predict the arc_id it lies
  on.** The `.route` gives the *sequence* of true arcs but is **not** pre-aligned to
  observations — observation→arc labels must be derived (e.g. by projecting each
  observation to the nearest point along the ordered route polyline). This
  derivation is a design decision, documented as a known unknown in PROJECT_ANALYSIS.

## 4. Verified quantities

### Partition counts
| partition | trips | total observations | obs min / median / max |
|-----------|-------|--------------------|------------------------|
| train | 80 | **208,417** | 361 / 1,529 / 19,313 |
| student_test | 10 | **19,160** | 262 / 1,420 / 3,958 |
| evaluator_test | 10 | (not in repo) | — |

### Per-trip network size ranges
| | nodes | arcs | route length |
|-|-------|------|--------------|
| train | 299 – 155,003 | 649 – 307,392 | 59 – 2,266 |
| student_test | 1,770 – 26,635 | 3,577 – 56,027 | 95 – 1,323 |

Network size varies by ~500× across trips — the per-trip road network is a local
subgraph, not a single shared map. Arc count ≈ 2× node count (consistent with
mostly bidirectional streets).

## 5. Geography

- Individual student_test trips are geographically compact (all bbox < 50 km).
- Example: train trip 00000001 spans lon≈8.58, lat≈49.84→50.19 → **Darmstadt /
  Rhine-Main, Germany** region. Street labels in the guide image are German/Dutch.
- **The dataset as a whole is global**: aggregate train extent lon [−100.4, 137.2],
  lat [−34.6, 61.3]. Different trips sit in different countries/continents, so any
  projection or distance metric must be chosen **per trip** (local UTM or geodesic),
  never a single global projection.
- Degrees are angular — use haversine/geodesic or a per-trip projected CRS, never
  raw Euclidean distance in degrees (per the Dataset Exploration doc §5).

## 6. Data-quality issues (verified counts)

| issue | train | student_test |
|-------|-------|--------------|
| NaN / missing GPS values | 0 | 0 |
| duplicate timestamps | 0 | 0 |
| non-monotonic timestamps | 0 | 0 |
| timestamp gaps > 1 s (count of steps) | **7,286** | 264 |
| max single timestamp gap | **153,064 s (~42 h)** | 210 s |
| trips with bbox > 50 km (long trips) | **10** | 0 |
| largest consecutive-observation jump | **887.5 m** (trip 00000055) | ~54 m |

> **Correction (Phase 1 EDA, 2026-11-18):** an earlier draft of this table
> reported an "~887 km" jump and "10 trips with physically impossible jumps".
> That was a units error — the value is **887.5 metres** (over a 184 s dropout
> ≈ 4.8 m/s). Re-computation finds **0** segments > 5 km. The 10 bbox > 50 km
> trips are **legitimately long intercity trips** (bbox 50–78.5 km), not
> corrupted. See `reports/phase1_eda_report.md` §3.2.

- ~94% of train steps are exactly 1 s apart (per-trip min 53%); the remaining
  steps are gaps of 2 s up to ~42.5 h.
- No **confirmed** impossible position jumps. There are suspicious high speeds
  (25 trips, 981 segments > 55 m/s; 1 trip > 150 m/s), clustered around
  timestamp gaps — GPS noise to smooth in Phase 2, not whole-trip errors.
- At least one train trip has a ~42-hour timestamp gap → either a parked/stopped
  interval or two trips concatenated; must be handled by the gap logic, not
  interpolated across.
- **student_test is clean** — convenient for development-stage evaluation, but the
  model/cleaning pipeline must still be robust to the messy train trips and to
  unknown evaluator trips.

## 7. Open schema questions (do not guess — see PROJECT_ANALYSIS §Risks)
1. **Observation→arc label derivation** is not provided explicitly; it must be
   constructed from the ordered `.route` polyline. The exact snapping rule
   (nearest arc vs. monotonic projection along the route) is a modelling choice.
2. Whether the `.route` arc sequence is node-connected end-to-end
   (each arc's `to_node` = next arc's `from_node`) — **verified in Phase 1**:
   71/80 train routes are fully connected; 9 trips have 1–3 broken links (13
   total, ≈0.07% of pairs). Near-connected but not perfect; the nature of the
   breaks (U-turn / one-way / noise) is an open question for Phase 2.
3. Units of timestamp confirmed as seconds; absolute epoch/origin is unknown and
   irrelevant (only deltas matter).
