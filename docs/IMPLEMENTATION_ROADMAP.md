# PathSnap — Implementation Roadmap

> Sequenced plan to deliver the full hackathon system. Each phase lists goal,
> concrete tasks, outputs, and the leakage/quality guards that apply. Cross-refs:
> `PROJECT_ANALYSIS.md` (risks Rn), `DATASET_SCHEMA.md` (data facts).
> **Implementation has not started** — this is the plan only.

## Guiding principles
- **Group = trip** everywhere: `GroupKFold(groups=trip_id)` for all CV (R2).
- Fit preprocessing on **train only**; `student_test` = dev comparison; evaluator
  partition untouched (not in repo).
- Build the end-to-end system **thin first** (trivial model behind API+UI), then
  improve the model behind a stable contract (R8).
- Log seeds, configs, timings, and per-trip metrics to CSV for the report & viva.

---

## Phase 0 — Environment & project skeleton
- Verify `.venv` has all `requirements.txt` packages (install state was unverified);
  add `xgboost`/`lightgbm` only if we commit to boosting beyond sklearn's
  `HistGradientBoosting` (R10). Avoid unnecessary installs.
- Create package layout under `src/`:
  ```
  src/pathsnap/
    io.py            # readers for .track/.nodes/.arcs/.route
    geo.py           # per-trip projection, haversine, projection-onto-polyline
    clean.py         # outlier filter, gap segmentation
    candidates.py    # spatial index + per-observation candidate arcs
    labels.py        # derive observation→arc ground truth from .route polyline
    features.py      # feature engineering
    cv.py            # GroupKFold helpers + leakage guard
    models.py        # model zoo + training/eval harness
    hpo.py           # Optuna Bayesian search
    hmm.py           # emission/transition + Viterbi decoder
    viz.py           # folium + matplotlib
    pipeline.py      # end-to-end predict(trip_id) used by API
  api/  (FastAPI)   frontend/  notebooks/  outputs/  reports/
  ```
- **Output:** importable package; initial commit.

## Phase 1 — EDA (Exploratory Data Analysis)  — *Report §2.2*
- Reproduce verified inventory (counts, obs stats, network sizes) as code + CSV.
- Missing-value / timestamp-gap / outlier analysis per trip; flag the 10
  outlier trips and the ~42 h-gap trip (R4).
- Per-trip geographic extent; confirm global distribution → per-trip CRS (R3).
- **Verify the two known unknowns**: (a) observation→arc label derivation feasibility,
  (b) whether each `.route` is a connected arc chain (R1, R7).
- Plots: trajectory scatter, sampling-interval histogram, network-size distribution,
  **≥1 Folium map** (raw GPS + road network + reference route), as in the guide image.
- **Output:** `notebooks/01_eda.ipynb`, `outputs/eda/*.csv`, `outputs/maps/*.html`.
- **Guard:** label EDA outputs by partition.

## Phase 2 — Cleaning, projection & observation→arc labels
- `clean.py`: speed/jump-based outlier removal; segment trips at large time gaps.
- `geo.py`: per-trip projection (local UTM via `pyproj`) or geodesic distance.
- `labels.py` **(highest-leverage step, R1)**: project each observation onto the
  ordered `.route` polyline (monotonic arc-length projection) to assign the true
  arc_id; validate on Folium against several trips before trusting it.
- **Output:** cached cleaned+labelled per-trip tables (`outputs/processed/`).
- **Guard:** labels derived from `.route` only; no eval data involved.

## Phase 3 — Candidate road generation
- Build a per-trip spatial index (shapely `STRtree` / scipy KD-tree on arc
  geometries) (R6).
- For each observation, retrieve arcs within radius **r**; r chosen to minimise
  **Candidate Miss Rate** (true arc present in candidate set) on train while keeping
  the set small. Sweep r and report the miss-rate/size trade-off (R5).
- **Output:** per-observation candidate lists; miss-rate-vs-radius curve.

## Phase 4 — Feature engineering  — *Report §2.2*
- Per (observation, candidate-arc) features:
  - perpendicular/geodesic distance obs→arc; projection fraction along arc.
  - heading difference (GPS course vs arc bearing); GPS speed; acceleration.
  - arc length; arc degree/connectivity; node degrees.
  - temporal context (Δt to prev/next obs; gap flag).
  - continuity features vs previous observation's best candidate.
- Fit scalers on **train only**; persist transformers as artifacts.
- **Output:** feature matrix + feature dictionary; `features.py`.

## Phase 5 — Model experimentation (≥5 models)  — *Report §3*
Frame as **per-candidate binary/probabilistic classification** ("is this the correct
arc for this observation?"), then pick the top-scoring candidate per observation.
Train **≥5 across families** (Models & Training doc §3):
1. Logistic Regression (linear/probabilistic baseline).
2. Gaussian Naive Bayes **or** Ridge/SGD classifier.
3. k-NN (instance-based).
4. Random Forest / Extra Trees (ensemble).
5. Gradient boosting — sklearn `HistGradientBoosting` (and/or XGBoost/LightGBM).
6. (optional) MLP.
- Harness: `GroupKFold` CV on 80 train trips; consistent preprocessing; record
  configs, **seeds**, train/inference time.
- Metrics per model: Macro F1, Top-1 Accuracy, Log Loss, Precision/Recall, Candidate
  Miss Rate, Route Continuity; evaluate on **student_test** for dev comparison.
- **Output:** `outputs/experiments/model_comparison.csv`, confusion matrices,
  comparison plots; declare **primary metric** (proposed: Observation-level Top-1
  Accuracy, with Macro F1 + Route Continuity as tie-breakers).
- **Guard:** never select on training accuracy alone; GroupKFold enforced.

## Phase 6 — Group-aware cross-validation (formalised)
- `cv.py`: `GroupKFold`/`StratifiedGroupKFold` wrapper keyed on trip ID, plus an
  assertion that no trip ID appears in two folds (R2). Used by Phases 5 & 7.
- Report per-fold variance (stability) for each model.

## Phase 7 — Bayesian hyperparameter optimisation  — *Report §4*
- **Optuna** (TPE/Bayesian) on the most promising model from Phase 5.
- Justified search space; objective = CV Macro F1 (or primary metric) under
  **group-aware CV on the 80 train trips** only.
- Log all trials; refit tuned model; evaluate tuned-vs-untuned on student_test;
  report compute cost.
- **Output:** `outputs/hpo/study.db` + trials CSV + importance plots.
- **Guard:** evaluator partition excluded from search space/objective/thresholds.

## Phase 8 — HMM + Viterbi route reconstruction  — *Report §5*
- **Hidden states:** candidate arcs per observation (time step = observation).
- **Emission:** normalized per-observation candidate probabilities from the
  classifier; the classifier includes 18 inference-time features.
- **Transition:** stay on the same directed arc with forward progress, or move to
  a directly connected directed arc. Penalize mismatch between network distance
  and GPS displacement.
- **Viterbi** decode; reset at segment gaps, report missing candidates and
  disconnected-transition fallbacks, and never invent connector arcs.
- Compare **classifier-only vs classifier+HMM**: pointwise Top-1 Accuracy,
  sequence-level performance, **Route Continuity**; show where HMM helps / hurts.
- **Output:** `pathsnap/hmm.py`, `scripts/run_hmm_evaluation.py`, and comparison
  tables/per-trip results under `outputs/experiments/`.
- **Guard:** train-only GroupKFold for development; student_test is a single
  final held-out evaluation and is never used to tune decoder parameters.

## Phase 9 — Folium visualisation  — *Report §6 (mandatory)*
- Reusable `viz.py`: layers for raw GPS, road network, reference route, predicted
  route; distinguishable colours, legend, layer controls, sensible zoom.
- Export interactive HTML + static screenshots for the report.
- **Output:** `outputs/maps/*.html`.

## Phase 10 — FastAPI backend (Laptop 1)  — *Report §7*
- Endpoint e.g. `POST /predict {trip_id}` → runs `pipeline.predict(trip_id)` →
  returns predicted route (arc sequence + geometry) + metrics.
- Server-side reference data computes accuracy; **ground truth never returned to UI**.
- Loads tuned model + preprocessing + HMM config artifacts at startup; visible
  request logs for the demo. `GET /trips` lists the 10 evaluation trips.
- **Output:** `api/` app, `uvicorn` run command, artifact-loading logic.
- **Guard:** no hardcoded outputs; works for all 10 trips.

## Phase 11 — Frontend (Laptop 2)  — *Report §7*
- UI lists the 10 evaluation trips, sends selected trip to the API (configurable
  Laptop-1 host), renders returned route on a map + shows evaluation results.
- Lightweight (static HTML/JS or Streamlit) talking to the API over the network.
- **Output:** `frontend/`.
- **Guard:** UI never sees ground-truth labels; no bypassing the API.

## Phase 12 — Integration, final evaluation & technical report
- End-to-end rehearsal across two machines for all 10 trips; capture API-log and UI
  screenshots.
- Assemble consolidated **PDF report** per prescribed sections: EDA, ≥5-model
  comparison + per-model Q&A, Bayesian HPO, HMM/Viterbi, mandatory Folium maps,
  API/UI/deployment, conclusions. Every number traceable to repo outputs.
- README (install/run/deploy), clearly named folders, no secrets; push public repo.

---

## Dependency / sequencing graph
```
P0 → P1 → P2 → P3 → P4 → P5 ─┬─ P6 (CV, used by P5 & P7)
                             └─ P7 → P8 → P9
P10 (API) can start thin right after P2 with a trivial model, then swap in P7/P8.
P11 (UI) parallels P10 against the API contract.
P12 integrates everything.
```

## Milestones (hackathon-pace, "first-phase eval before lunch")
- **M1 (early):** EDA done + thin API+UI integrated end-to-end with a baseline model
  (de-risks integration, R8) — enough to pass a first-phase check.
- **M2:** ≥5 models compared under GroupKFold; primary metric declared.
- **M3:** Bayesian HPO + HMM/Viterbi improving route continuity; Folium maps.
- **M4:** Full two-laptop demo for all 10 trips + report finalised + repo pushed.
