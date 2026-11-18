# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap reconstructs realistic vehicle trajectories by matching noisy 1 Hz GPS
observations to a road network. For each GPS observation it predicts the road
**arc** (segment) the vehicle was on; the sequence of arcs is the reconstructed
route. A supervised classifier scores candidate arcs and an **HMM + Viterbi**
decoder enforces road-network connectivity to produce a coherent route, served
over a **FastAPI** backend to an evaluator UI.

> Hackathon: PES University UE24CS352A — Machine Learning.
> Full problem analysis and plan live in [`docs/`](docs/).

## Status (honest, as of this commit)

**Completed and runnable:**
- **Phase 0 — foundation:** package skeleton, `config`, `data_io` (raw readers),
  `logging_utils`, environment verification (`scripts/verify_env.py`), FastAPI
  health scaffold.
- **Phase 1 — Exploratory Data Analysis:** `pathsnap/eda.py` + `pathsnap/viz.py`,
  run via `scripts/run_eda.py`. Produces real per-trip statistics, partition
  comparison, 5 charts and 3 Folium maps under `outputs/`, and a written report
  at [`reports/phase1_eda_report.md`](reports/phase1_eda_report.md). All 90 trips
  analysed (80 train fully; 10 student_test kept strictly separate).

- **Phase 2–5 — preprocessing → labels → candidates → features → 5-model
  comparison (runnable):** `pathsnap/{geo,preprocess,labels,candidates,features,
  cv,models,evaluation}.py`, driven by `scripts/build_dataset.py` and
  `scripts/run_training.py`. Per-trip UTM projection, gap segmentation, outlier
  flagging, **monotonic route-projection labels** (coverage 96%), KD-tree
  candidate generation (recall 97%), 12 leakage-safe features, and **5 trained
  models** compared under `GroupKFold(trip_id)`. Results in
  `outputs/experiments/`; report at
  [`reports/first_evaluation_report.md`](reports/first_evaluation_report.md).
  Best so far: Random Forest (CV macro-F1 0.779, student Top-1 0.558). These are
  **real, reproducible** numbers — classifier-only Top-1 is modest (~0.55, close
  to a nearest-arc baseline); the sequential gain is expected from the HMM.

**Not yet implemented (honest — do NOT assume these exist):** Bayesian HPO
(Phase 7), HMM/Viterbi sequential decoding (Phase 8), the full prediction
pipeline (Phase 10), and the frontend (Phase 11). These remain documented stubs.
Nothing is fabricated; no evaluator data was used.

## Architecture

```
GPS (.track) + road network (.nodes/.arcs) + reference route (.route)
   → ingestion        pathsnap/data_io.py
   → preprocessing    pathsnap/preprocess.py , pathsnap/geo.py      [Phase 2]
   → labels           pathsnap/labels.py                            [Phase 2]
   → candidates       pathsnap/candidates.py                        [Phase 3]
   → features         pathsnap/features.py                          [Phase 4]
   → models (>=5)     pathsnap/models.py + pathsnap/cv.py           [Phase 5-7]
   → HMM/Viterbi      pathsnap/hmm.py                               [Phase 8]
   → evaluation       pathsnap/evaluation.py                        [Phase 5/8]
   → visualisation    pathsnap/viz.py (Folium)                      [Phase 1/9]
   → pipeline         pathsnap/pipeline.py  →  api/ (Laptop 1)      [Phase 10]
                                             →  frontend/ (Laptop 2)[Phase 11]
```

Two-laptop deployment: **Laptop 1** runs the model + FastAPI + server-side
evaluation; **Laptop 2** runs the UI that lists the 10 evaluation trips and
displays returned routes.

## Repository layout

```
PathSnap-ML/
├── pathsnap/              # core Python package (importable from repo root)
│   ├── config.py          # paths, random seed, experiment settings
│   ├── logging_utils.py   # logging factory
│   ├── data_io.py         # raw-file readers + Trip dataclass   [implemented]
│   ├── eda.py             # exploratory analysis                [Phase 1 stub]
│   ├── geo.py             # projection / geodesic helpers        [Phase 2 stub]
│   ├── preprocess.py      # outlier filtering, gap segmentation  [Phase 2 stub]
│   ├── labels.py          # observation→arc label derivation     [Phase 2 stub]
│   ├── candidates.py      # candidate arc generation             [Phase 3 stub]
│   ├── features.py        # feature engineering                  [Phase 4 stub]
│   ├── cv.py              # group-aware cross-validation         [Phase 6 stub]
│   ├── models.py          # model zoo + Bayesian HPO             [Phase 5/7 stub]
│   ├── hmm.py             # HMM + Viterbi decoding               [Phase 8 stub]
│   ├── evaluation.py      # metrics                              [Phase 5/8 stub]
│   ├── viz.py             # Folium + matplotlib                  [Phase 1/9 stub]
│   └── pipeline.py        # end-to-end predict(trip_id)          [Phase 10 stub]
├── api/                   # FastAPI backend (Laptop 1)           [skeleton]
├── frontend/              # evaluator UI (Laptop 2)              [placeholder]
├── scripts/
│   └── verify_env.py      # environment + dataset verification
├── notebooks/             # EDA / experiment notebooks
├── outputs/               # generated artifacts (git-ignored)
├── reports/               # technical report
├── data/raw/Student folder/   # raw dataset (git-ignored, read-only)
├── docs/                  # official PDFs + audit documents
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the environment and dataset (run from the repo root)
python scripts/verify_env.py
```

`verify_env.py` confirms the Python version, that every required package
imports, that `pathsnap` imports, and that the raw dataset is present with the
expected trip counts (80 train / 10 student_test).

## Data

The raw dataset lives under `data/raw/Student folder/` and is **read-only** —
never modified or committed. Partitions:

| partition | trips | use |
|-----------|-------|-----|
| `train` | 80 | training, cross-validation, HPO |
| `student_test` | 10 | development-stage evaluation / model comparison |
| evaluator_test | 10 | final evaluation only — **not present locally** |

See [`docs/DATASET_SCHEMA.md`](docs/DATASET_SCHEMA.md) for the verified file
schemas and [`docs/PROJECT_ANALYSIS.md`](docs/PROJECT_ANALYSIS.md) for the audit.

## Reproducibility & integrity

- A single random seed is centralised in `pathsnap/config.py`.
- All cross-validation is **group-aware by trip ID** — no trip is split across
  folds, and evaluator data/labels never enter training, feature selection, or
  tuning.
- No experimental results are fabricated; every reported number will be produced
  by committed code.
