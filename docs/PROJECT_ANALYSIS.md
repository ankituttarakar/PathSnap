# PathSnap — Project Analysis & Audit

> Author: ML lead. Date: 2026-11-18. Status: **Audit complete — implementation not
> yet started.** Every quantitative claim here was verified against the actual
> files; see `DATASET_SCHEMA.md` for the raw evidence.

## 1. Problem statement

Build **PathSnap**, a distributed map-matching system that reconstructs a realistic
vehicle trajectory by snapping noisy 1 Hz GPS observations onto a road network.
For each GPS observation we must predict the road **arc** (segment) the vehicle was
on; the full sequence of arcs is the reconstructed route.

Core pipeline required by the hackathon:
```
raw .track/.nodes/.arcs/.route
  → clean + project
  → candidate arc generation (per observation)
  → feature engineering
  → supervised classifier (score each candidate arc)  [≥5 models compared]
  → Bayesian hyperparameter optimisation of the chosen model
  → HMM + Viterbi decoding over candidates (enforce connectivity/sequence)
  → predicted route
  → FastAPI backend (Laptop 1) + evaluator UI (Laptop 2)
  → Folium maps + technical report
```

## 2. Repository audit

| Area | State |
|------|-------|
| Git | Branch `main`, remote `github.com/ankituttarakar/PathSnap-ML`, **no commits yet** (clean working tree, empty history). |
| `docs/` | 5 official PDFs (all read) + the 3 analysis docs produced by this audit. |
| `data/raw/Student folder/` | `train/` (80 trips) + `student_test/` (10 trips). Evaluator partition absent (correct). Also present: two source ZIPs (`Student folder.zip`, `PathSnap_student_materials.zip`). |
| `src/`, `frontend/`, `notebooks/`, `outputs/`, `reports/` | **Empty scaffolding dirs** — nothing implemented. |
| `requirements.txt` | Full intended stack present (numpy, pandas, scikit-learn, scipy, networkx, folium, pyproj, shapely, geopandas, haversine, optuna, fastapi, uvicorn, joblib, tqdm, jupyter, matplotlib, seaborn). No XGBoost/LightGBM/CatBoost listed yet. |
| `.venv/` | Python virtual-env present (package install state not fully enumerated — pip listing was interrupted). |

**Bottom line:** clean greenfield repo. Data and dependency list are in place; no
code, no commits, no models.

## 3. Documentation summary (5 PDFs)

- **Main Guidelines** — challenge overview; two-laptop architecture (Laptop 1 =
  model+API+server-side eval; Laptop 2 = UI showing 10 eval trips); 12-stage work
  process.
- **Dataset Exploration** — file types & meanings; partitions (80/10/10); strict
  **leakage rules**; geographic-distance cautions; required EDA outputs.
- **Models & Training** — **≥5 models from multiple families**; experimental
  design; metric list; **group-aware CV**; **Bayesian** HPO; **HMM/Viterbi**
  requirements; final-evaluation protocol.
- **Deliverables / Technical Report** — exactly 2 deliverables (public GitHub repo +
  one consolidated PDF report); mandatory report sections; **mandatory Folium**;
  API/UI screenshots; independent-work rules.
- **Assessment & Plagiarism** — integrity, cross-team similarity checks, **data-leakage
  rules**, no hardcoded predictions, viva/defence expectations.

## 4. Mandatory requirements checklist (what we are graded on)

**Data & EDA**
- [ ] Dataset inventory (trip/file/obs counts), summary stats, missing-value analysis.
- [ ] GPS trajectory plots, road-network viz, **≥1 Folium map**.
- [ ] State which **partition** each analysis used.

**Modelling**
- [ ] Train & compare **≥5 distinct models** from multiple algorithm families, with
  recorded configs, seeds, and results (not training-accuracy-only selection).
- [ ] Candidate generation per observation; feature engineering.
- [ ] Metrics: Accuracy, Precision, Recall, **Macro F1**, Confusion Matrix, **Log
  Loss**, **Observation-level Top-1 Accuracy**, **Candidate Miss Rate**, **Route
  Continuity**, sequence-level perf, train/inference time.
- [ ] Declare and justify the **primary selection metric**.

**Optimisation**
- [ ] **Bayesian** HPO (Optuna) on the chosen model, **group-aware CV on the 80 train
  trips**, logged trials, tuned-vs-untuned comparison, compute cost.

**Route reconstruction**
- [ ] **HMM + Viterbi**: defined hidden states, emission & transition models, handling
  of missing candidates / invalid transitions / GPS gaps; **classifier-only vs
  classifier+HMM** comparison; route-continuity analysis.

**System**
- [ ] **FastAPI** backend (Laptop 1): takes a trip ID, runs full pipeline, computes
  accuracy server-side against reference data, returns predicted route. API logs
  must be shown live. Ground truth never leaves the server.
- [ ] **Frontend** (Laptop 2): lists the 10 evaluation trips, sends selected trip to
  API, renders returned route + results. Must work for all 10, no hardcoding.

**Submission**
- [ ] Public GitHub repo: all code (data proc, features, every model, HPO, HMM, API,
  UI, deployment, Folium), saved model + config artifacts, README with
  install/run/deploy, clearly named folders, no secrets.
- [ ] One consolidated **PDF technical report** following the prescribed sections;
  every number verifiable from the repo.

## 5. Leakage / integrity constraints (hard rules)

1. **No trip may be split across partitions** → all CV must be **GroupKFold by trip
   ID** (group = trip). Never split a trip's observations across folds.
2. **Evaluator-test data/labels** must not touch training, feature selection, model
   choice, HPO, or thresholds. (It isn't even in the repo — keep it that way.)
3. Fit all learned preprocessing (scalers, encoders) on **train only**.
4. **student_test** is for development-stage comparison **only** — not for training;
   using it to pick the final model is allowed (it's dev data), but it must not be
   mixed into fitting.
5. **No hardcoded predictions**, no manual edits to eval outputs. Server keeps ground
   truth; UI/model never see eval labels.
6. Independent implementation; results must be reproducible and defensible in viva.

## 6. Key verified data facts (see DATASET_SCHEMA.md)

- 80 train trips / 208,417 obs; 10 student_test trips / 19,160 obs. Clean of NaNs,
  dup/non-monotonic timestamps.
- IDs fully consistent: route→arc and arc→node indices valid in 90/90 trips.
- Data-quality issues to engineer around: **7,286** >1 s gaps in train (max
  ~42.5 h). *(Correction, Phase 1: there are NO impossible GPS jumps — the
  earlier "887 km" was a units error for 887.5 **m**; the 10 bbox>50 km trips are
  legitimate long trips. See `reports/phase1_eda_report.md`.)* Suspicious high
  speeds near gaps (25 trips) to smooth in Phase 2.
- Dataset is **globally distributed** → per-trip local projection / geodesic
  distances only.

## 7. Risks & pitfalls

| # | Risk | Mitigation |
|---|------|------------|
| R1 | **Observation→arc labels are not given** — must be derived from the ordered `.route` polyline; a wrong snapping rule poisons all supervised labels. | Spend EDA time on a principled, monotonic projection of observations onto the route polyline; validate visually on Folium before training. Treat as the project's highest-leverage decision. |
| R2 | **Leakage via non-grouped CV.** | Enforce `GroupKFold(groups=trip_id)` everywhere; write a guard that asserts no trip spans folds. |
| R3 | **Global geography** → a single projection distorts distances. | Choose CRS/geodesic **per trip** (local UTM zone or haversine). |
| R4 | **GPS outliers & 42 h gaps** break distance features and HMM transitions. | Robust outlier filter (speed/jump threshold); segment trips at large gaps; HMM must allow "gap" transitions rather than interpolating. |
| R5 | **Candidate Miss Rate** — if true arc isn't in the candidate set, it's unrecoverable downstream. | Tune candidate radius to drive miss rate low; report it as a first-class metric. |
| R6 | **Huge network variance** (up to 307k arcs/trip) → per-observation all-arc scoring is intractable. | Spatial index (KD-tree/STRtree) for candidate arcs within radius; never score all arcs. |
| R7 | **Route connectivity not yet verified** — HMM transition design assumes arc adjacency. | Verify route arc-chain connectivity in EDA (known unknown, DATASET_SCHEMA §7.2). |
| R8 | **Two-laptop integration** is often where hackathon points are lost. | Build API+UI early with a trivial model, integrate end-to-end on day 1, then improve the model behind a stable contract. |
| R9 | **Plagiarism / AI-content checks.** | Keep our own experiment logs, seeds, commit history; write report in our own words; be able to defend every number. |
| R10 | Missing boosting libs (xgboost/lightgbm) in requirements. | Add them if used, or stay within sklearn's HistGradientBoosting to avoid install churn. |

## 8. Known unknowns (explicitly not guessed)
- Exact observation→arc labelling rule (R1). **To be decided in EDA.**
- Whether every `.route` is a fully node-connected arc chain (R7). **To be verified.**
- Evaluator's exact scoring formula / primary metric weighting (server-side). We
  will optimise Observation-level Top-1 Accuracy + Macro F1 + Route Continuity as
  proxies, and state this assumption.
- Package install state inside `.venv` (pip enumeration was interrupted) — verify
  before first run.

## 9. Audit conclusion
The repo is a clean starting point with verified, well-structured data and a
complete requirements picture. The single most important pre-implementation decision
is **how to derive per-observation arc labels from the route polyline** (R1). No
implementation has begun, per instructions. Proceed to the roadmap.
