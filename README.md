# Spotify Track Popularity Prediction

An end-to-end, production-style **Machine Learning system** that predicts a Spotify track's
`track_popularity` score (0–100) from its metadata — release date, artist stats, genre,
duration, and album context — served through a modern **Flask** dashboard.

Built to demonstrate real-world **MLOps practices**: modular pipelines, config-driven design,
experiment tracking (MLflow), dataset versioning (DVC), automated CI, and a polished web UI.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Architecture Diagram](#-architecture-diagram)
- [Dataset Description](#-dataset-description)
- [Folder Structure](#-folder-structure)
- [Installation Steps](#-installation-steps)
- [Running Locally](#-running-locally)
- [Docker Instructions](#-docker-instructions)
- [ML Pipeline Explanation](#-ml-pipeline-explanation)
- [Model Performance](#-model-performance)
- [Web Application](#-web-application)
- [Screenshots](#-screenshots)
- [Technologies Used](#-technologies-used)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## Project Overview

Spotify's `track_popularity` field is one of the most requested-but-elusive metrics for
artists, labels and data scientists to predict. This project builds a full ML system —
not just a notebook — around that target:

- A **six-model bake-off** (Linear Regression, Random Forest, Gradient Boosting, XGBoost,
  CatBoost, LightGBM) with automatic best-model selection by **R² Score**.
- A **config-driven**, strongly-typed pipeline (`config.yaml` + `params.yaml`) — no hard-coded
  paths or hyperparameters anywhere in the code.
- **Custom logging & exception handling** used consistently across every module.
- **MLflow** experiment tracking and **DVC** dataset/pipeline versioning.
- A **Flask** dashboard with dark/light themes, live predictions, a model metrics page,
  feature importance chart, and locally-stored prediction history.
- **Dockerized**, with a GitHub Actions CI workflow that lints and re-runs the full
  training pipeline on every push.

---

## Architecture Diagram

```
                     ┌────────────────────────┐
                     │   Raw CSV source(s)    │
                     │  data/raw/*.csv (DVC)  │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │   1. Data Ingestion    │  merge + de-dup + train/test split
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │  2. Data Validation    │  schema check, null/target sanity
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ 3. Data Transformation │  FeatureEngineer + OutlierCapper +
                     │   & Feature Engineering│  Imputer + OneHotEncoder + Scaler
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │   4. Model Training    │  6 regressors, MLflow-tracked
                     │   (best R² auto-pick)  │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │  5. Model Evaluation   │  final metrics + feature importance
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ best_model.joblib +    │
                     │ preprocessor.joblib    │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐        ┌─────────────────────────┐
                     │  Prediction Pipeline   │◀──────▶│   Flask Web Dashboard   │
                     │  (CustomData → predict)│        │  Predict / Metrics /    │
                     └────────────────────────┘        │  History / About       │
                                                        └─────────────────────────┘
```

---

## Dataset Description

The raw dataset (`data/raw/track_data_final.csv`) is a **merged and de-duplicated**
combination of two Spotify metadata exports (`spotify_data clean.csv` and
`track_data_final.csv`), unified by `data_ingestion.py` and reduced from
`17,360 → 8,494` unique rows keyed on `track_id`.

| Column                | Type    | Description                                          |
|-----------------------|---------|-------------------------------------------------------|
| `track_id`            | string  | Unique Spotify track identifier                       |
| `track_name`          | string  | Track title                                           |
| `track_number`        | int     | Position of the track within its album                |
| `track_popularity`    | int     | **Target.** Spotify popularity score (0–100)          |
| `track_duration_ms`   | int     | Track duration in milliseconds                        |
| `explicit`            | bool    | Explicit content flag                                 |
| `artist_name`         | string  | Primary artist name                                   |
| `artist_popularity`   | float   | Artist's own popularity score (0–100)                 |
| `artist_followers`    | float   | Number of followers the artist has                    |
| `artist_genres`       | string  | Comma-separated genre list for the artist              |
| `album_id`            | string  | Unique album identifier                                |
| `album_name`          | string  | Album title                                            |
| `album_release_date`  | string  | Release date (`YYYY-MM-DD`, `YYYY-MM` or `YYYY`)       |
| `album_total_tracks`  | int     | Number of tracks in the album                          |
| `album_type`          | string  | `album`, `single`, or `compilation`                    |

---

## Folder Structure

```
spotify-track-popularity/
├── app.py                          # Flask application (thin presentational layer)
├── main.py                         # Entry point: runs the full training pipeline
├── config/
│   ├── config.yaml                 # All file paths (single source of truth)
│   └── params.yaml                 # Model hyperparameters
├── data/raw/                        # DVC-tracked raw CSV(s)
│   └── track_data_final.csv.dvc
├── src/spotify_popularity/
│   ├── logger.py                    # Centralised logger
│   ├── exception.py                 # Custom exception class
│   ├── entity/
│   │   └── config_entity.py         # Dataclass configs
│   ├── config/
│   │   └── configuration.py         # Reads yaml -> typed config objects
│   ├── utils/
│   │   └── common.py                # yaml/json/joblib helpers
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── feature_engineering.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   └── pipeline/
│       ├── training_pipeline.py     # Orchestrates all 5 stages
│       └── prediction_pipeline.py   # Inference-time pipeline + CustomData
├── notebook/
│   └── EDA.ipynb                    # Exploratory Data Analysis
├── templates/                       # Jinja2 templates (dashboard UI)
├── static/css/style.css             # Design system, dark/light themes
├── static/js/script.js              # Theme toggle, form UX, animations
├── tests/                           # Pytest unit tests
├── artifacts/                       # Generated at runtime (ingestion, model, metrics...)
├── .github/workflows/ci.yaml        # Lint + train on every push
├── dvc.yaml / dvc.lock              # DVC pipeline definition
├── Dockerfile / .dockerignore
├── requirements.txt / setup.py / setup.cfg
└── README.md
```

---

## Installation Steps

**Prerequisites:** Python 3.10+, pip, (optionally) Docker & DVC.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/spotify-track-popularity.git
cd spotify-track-popularity

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies (also installs the project itself, editable)
pip install -r requirements.txt

# 4. (Optional) Pull the versioned raw dataset via DVC
dvc pull
```

---

## Running Locally

**1. Train the model** (runs Ingestion → Validation → Transformation → Training → Evaluation):

```bash
python main.py
```

Every stage is also independently executable, e.g.:

```bash
python -m src.spotify_popularity.components.data_ingestion
python -m src.spotify_popularity.components.model_trainer
```

**2. Launch the Flask dashboard:**

```bash
python app.py
# Visit http://127.0.0.1:5000
```

**3. Run the test suite / linting (same checks as CI):**

```bash
pytest tests/ -v
flake8 src --max-line-length=120
black --check src
```

**4. Inspect experiments with MLflow:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Docker Instructions

```bash
# Build the image
docker build -t spotify-popularity:latest .

# Run the container (serves on port 5000 via Gunicorn)
docker run -p 5000:5000 spotify-popularity:latest
```

> Note: run `python main.py` at least once before building the image (or mount a volume with
> pre-trained `artifacts/`) so the Flask app has a `best_model.joblib` and `preprocessor.joblib`
> to load at startup.

---

## ML Pipeline Explanation

| Stage | Module | Responsibility |
|---|---|---|
| **1. Data Ingestion** | `data_ingestion.py` | Merges raw CSV source(s), de-duplicates by `track_id`, splits train/test |
| **2. Data Validation** | `data_validation.py` | Schema check (required columns), null/target sanity report |
| **3. Feature Engineering** | `feature_engineering.py` | Date parsing (year/month/day-of-week/recency), genre bucketing (top-N + "other"), log-transform of followers, text-length feature |
| **3. Data Transformation** | `data_transformation.py` | `OutlierCapper` (IQR clipping) → `SimpleImputer` → `OneHotEncoder` / `StandardScaler`, bundled into one persisted `SpotifyPreprocessor` |
| **4. Model Training** | `model_trainer.py` | Trains 6 regressors, logs params/metrics to MLflow, **auto-selects the best model by test R²** |
| **5. Model Evaluation** | `model_evaluation.py` | Final R² / MAE / RMSE / MAPE report + feature importance extraction |
| **Prediction Pipeline** | `prediction_pipeline.py` | `CustomData` → same preprocessor → model → bounded [0, 100] prediction, logged to local history |

All handling of **missing values, duplicates, outliers, encoding, date conversion and
scaling** required by the project spec lives inside stages 3–4 above, driven entirely by
`config.yaml` / `params.yaml`.

---

## Model Performance

Automatically selected best model: **RandomForestRegressor**

| Model                     | R² Score | MAE    | RMSE   |
|---------------------------|:--------:|:------:|:------:|
| **RandomForestRegressor** ⭐ | **0.372** | **13.38** | **18.75** |
| LGBMRegressor              | 0.369    | 13.45  | 18.79  |
| XGBRegressor               | 0.359    | 13.47  | 18.95  |
| CatBoostRegressor          | 0.355    | 13.78  | 19.00  |
| GradientBoostingRegressor  | 0.349    | 13.85  | 19.10  |
| LinearRegression           | 0.256    | 15.29  | 20.41  |

*(Metrics regenerate automatically on every `python main.py` run — see
`artifacts/metrics/metrics.json` and `artifacts/model/training_report.json` for the current
numbers, also displayed live on the app's **Model Metrics** page.)*

---

## Web Application

The Flask app (`app.py`) is a thin presentational layer over the prediction pipeline, with:

- **`/`** — Prediction form with client + server-side input validation
- **`/predict`** — Runs inference and shows the result
- **`/metrics`** — Model comparison table + feature importance chart
- **`/history`** — Local JSON-backed prediction history (last 200 predictions)
- **`/about`** — Project background
- **`/api/health`** — Health-check endpoint for Docker/orchestrators

UI highlights: dark/light theme toggle (persisted, flash-free), responsive cards, smooth
CSS animations, and a cohesive dashboard design system (`static/css/style.css`).

---

## Screenshots

> _Add screenshots after running the app locally:_

| Prediction Form | Model Metrics | Prediction History |
|---|---|---|
| `docs/screenshots/predict.png` | `docs/screenshots/metrics.png` | `docs/screenshots/history.png` |

---

## Technologies Used

**ML / Data:** pandas, NumPy, scikit-learn, XGBoost, CatBoost, LightGBM
**MLOps:** MLflow (experiment tracking), DVC (data & pipeline versioning)
**Backend:** Flask, Gunicorn
**Frontend:** HTML5, CSS3 (custom design system), vanilla JavaScript
**Quality/CI:** pytest, flake8, black, GitHub Actions
**Packaging:** Docker, setuptools

---

## Future Improvements

- Hyperparameter tuning via Optuna / GridSearchCV, tracked in MLflow.
- Audio-feature enrichment (danceability, energy, tempo) via the Spotify Web API.
- Model registry + staged promotion (staging → production) with MLflow Model Registry.
- Async task queue (Celery/RQ) for batch prediction jobs.
- Authentication + per-user prediction history instead of a single local JSON file.
- Kubernetes deployment manifests for horizontal scaling.

---

[GitHub](https://github.com/<dilpreetkaur06>) · [LinkedIn](www.linkedin.com/in/dilpreet-kaur2004/)
