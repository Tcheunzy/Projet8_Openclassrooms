C# Credit Scoring — Prêt à dépenser

[![CI](https://github.com/Tcheunzy/Projet8_Openclassrooms/actions/workflows/ci.yml/badge.svg)](https://github.com/Tcheunzy/Projet8_Openclassrooms/actions/workflows/ci.yml)

Productionising a credit-scoring model: FastAPI service, Gradio demo interface,
Docker image and CI/CD pipeline.

The model (LightGBM, trained on the Home Credit dataset) estimates the
probability that an applicant will default, and turns that probability into a
lending decision using a threshold optimised on a business cost function.

## Live demo

| Interface | URL | Intended audience |
|---|---|---|
| Scoring interface | [/gradio](https://projet8-scoring-credit.onrender.com/gradio) | Loan officer |
| API documentation | [/docs](https://projet8-scoring-credit.onrender.com/docs) | Integrating developer |
| Health endpoint | [/health](https://projet8-scoring-credit.onrender.com/health) | Operations |

> **The first request is slow.** The service runs on a free instance that spins
> down after 15 minutes of inactivity. The first call after an idle period can
> take up to a minute while the instance restarts and reloads the model.
> Subsequent calls answer in tens of milliseconds.

## The problem this solves

A model trained in a notebook cannot be deployed as is. Between the raw fields of
a loan application and the 575 features the model expects lies a chain of
transformations: anomaly handling, aggregation of six banking-history tables,
business ratios, imputation, encoding.

In production that chain must be replayed **identically**. Any divergence — a
recomputed threshold, a missing column, a reordered step — yields a prediction
that looks normal but is wrong. This is *training-serving skew*, and it is
invisible: the model raises no error.

Three mechanisms address it, detailed further down: **a single transformation
codebase**, **frozen artefacts**, and **two column contracts**.

## Architecture

The full path of a request:

```
 1.  JSON request                    POST /predict
 2.  Pydantic validation             -> 422 on missing field, out-of-range
                                        value or wrong type
 3.  Input contract                  realign on the 121 raw columns
                                        (unsupplied fields become NaN)
 4.  Cleaning                        DAYS_* sentinels, capping,
                                        missing-value ratio,
                                        document-flag grouping
 5.  Client history                  lookup in precomputed aggregations
                                        (6 parquet tables, keyed on SK_ID_CURR)
 6.  Feature engineering             business ratios, external scores,
                                        payment behaviour
 7.  Output contract                 realign on the 456 columns the
                                        preprocessor expects
 8.  Preprocessing                   imputation, one-hot, ordinal encoding
                                        -> 575 features
 9.  Model                           LightGBM -> default probability
10.  Business decision               probability > 0.24 -> declined
```

Steps 4, 6 and 8 run **exactly the same code** as training, imported from `src/`.

## Repository layout

```
api/
  main.py                  endpoints, artefact loading at startup
  schemas.py               input/output contract (Pydantic)
  gradio_app.py            demo interface, an HTTP client of the API
src/
  cleaning.py              anomalies, capping, column grouping
  aggregation.py           aggregation of the six auxiliary tables
  feature_engineering.py   business ratios and derived variables
  pipeline.py              orchestration and column contracts
  precompute_history.py    offline aggregation job (re-run when history changes)
  export_model.py          exports the model out of the MLflow registry
models/
  model.joblib                 production model (frozen copy of version 4)
  preprocessor.joblib          fitted ColumnTransformer
  preprocessing_params.json    frozen thresholds and column lists
  application_columns.json     input contract
  history/*.parquet            per-client aggregations
tests/                     28 unit and functional tests
notebook/                  exploratory analysis and training (upstream project)
Dockerfile                 production image
```

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Tcheunzy/Projet8_Openclassrooms.git
cd Projet8_Openclassrooms
uv sync
```

Three dependency groups are kept apart:

| Group | Contents | Command |
|---|---|---|
| main | what the API loads to serve a prediction | `uv sync --no-dev` |
| `dev` | pytest, coverage (installed by default) | `uv sync` |
| `train` | MLflow, Optuna, SHAP — never deployed | `uv sync --group train` |

## Running the API

```bash
uv run uvicorn api.main:app --reload --port 8000
```

- Interactive documentation: http://localhost:8000/docs
- Scoring interface: http://localhost:8000/gradio

## Calling the API

```bash
curl -X POST https://projet8-scoring-credit.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SK_ID_CURR": 100002,
    "AMT_INCOME_TOTAL": 150000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "AMT_GOODS_PRICE": 450000,
    "DAYS_BIRTH": -16000,
    "DAYS_EMPLOYED": -2000,
    "CNT_FAM_MEMBERS": 2,
    "CNT_CHILDREN": 0,
    "CODE_GENDER": "F",
    "NAME_CONTRACT_TYPE": "Cash loans",
    "FLAG_OWN_CAR": "Y",
    "FLAG_OWN_REALTY": "Y",
    "NAME_EDUCATION_TYPE": "Higher education"
  }'
```

Response:

```json
{
  "sk_id_curr": 100002,
  "probability": 0.0808,
  "threshold": 0.24,
  "decision": "accordé",
  "mlflow_model_version": "4"
}
```

Only 14 fields are required — those without which a prediction would be
meaningless. Any other column of the dataset can be passed through
`extra_fields`, and whatever is still absent is imputed with the median learned
at training time: the same mechanism already applied to the thousands of missing
values in the training set.

Durations follow the dataset convention: `DAYS_BIRTH` and `DAYS_EMPLOYED` are
expressed in **negative days** before the application. The Gradio interface takes
years and performs the conversion itself.

## Tests

```bash
uv run pytest -v
uv run pytest --cov=src --cov=api --cov-report=term-missing
```

28 tests, roughly 93% coverage (the CI enforces a 90% floor). No test depends on
the `data/` folder or on a running MLflow server: the suite runs on a bare
machine, which is what makes continuous integration possible at all.

| File | Scope |
|---|---|
| `test_cleaning.py` | cleaning functions on hand-built DataFrames |
| `test_aggregation.py` | aggregations, including the two-level `bureau` chain |
| `test_feature_engineering.py` | late-payment detection |
| `test_pipeline.py` | orchestration, and ordering of operations |
| `test_schemas.py` | validation: missing field, out-of-range value, wrong type |
| `test_api.py` | endpoints, unknown client, internal failure |
| `test_gradio.py` | unit conversion and error handling, with HTTP mocked |

## Docker

```bash
docker build -t projet8-api .
docker run --rm -p 8000:8000 projet8-api
```

The image weighs 443 MB and the running container uses about 410 MiB. The listen
port is configurable through the `PORT` variable, as most hosting providers
require:

```bash
docker run --rm -e PORT=10000 -p 10000:10000 projet8-api
```

## CI/CD

On every pull request and every push to `main`, GitHub Actions:

1. rebuilds the environment from `uv.lock`;
2. runs the 28 tests and enforces the coverage floor;
3. builds the Docker image;
4. starts the container and queries `/health`.

The `main` branch is protected: nothing merges unless these checks pass. Render
redeploys automatically on merge.

Building an image only proves it is syntactically valid; it is the fourth step —
starting the container and getting an answer — that proves the application
actually works inside it.

## Design decisions

### A single transformation codebase

`src/` is called both at training time and at serving time. Training-serving skew
becomes structurally impossible, because the logic exists in exactly one place.

This refactor surfaced a defect in the original notebook: `PAYMENT_RATE` was
computed on the training set only, never on the test set. Once the logic lives in
functions taking a single DataFrame, that omission cannot happen.

### Frozen artefacts

Capping thresholds, drop lists and imputation medians were computed **once**, on
the training data, and live in `models/`. They are never recomputed.

The reason is concrete: production handles one client at a time. The 99.5th
percentile of a single income is that income — capping would be meaningless and
every applicant would be treated differently.

### Two column contracts

This is the only part with no counterpart in the notebook, because it answers a
question training never asks: *what happens when you process one client instead
of 300,000?*

**On input**, the request DataFrame is realigned on the 121 columns of
`application_train`. Without it, the absent `FLAG_DOCUMENT_*` columns would raise
an error, and the missing-value ratio — itself a predictive feature — would be
measured over a different set of columns than at training time.

**On output**, the DataFrame is realigned on the columns the preprocessor
expects. A client with no microloan in their history produces no such one-hot
column, yet the model expects one; realignment recreates it as `NaN`, which
imputation handles normally.

That expected list is not an extra artefact to maintain: it is read straight out
of `preprocessor.joblib`, which remembers the columns it was fitted on. One
source of truth, impossible to desynchronise.

### Precomputed history

The six banking-history tables hold over 20 million rows. Shipping them in the
image and aggregating them per request is not viable.

Aggregations only change when history changes, so they are computed offline by
`src/precompute_history.py` and stored as parquet, keyed by client id. A request
simply reads one row.

A client absent from that store is not an error: their history features are
`NaN` and imputation takes over. A new applicant with no banking record is a
legitimate business case.

### A 0.24 business threshold, not 0.5

The threshold was optimised on an asymmetric cost function: a missed default
costs roughly ten times more than an unwarranted refusal. The model therefore
flags more often, accepting false alarms on sound files.

The threshold is returned with every response so the caller can justify the
decision, and read from the `THRESHOLD` environment variable so it can be tuned
without rebuilding the image.

### MLflow removed from the serving path

MLflow remains the source of truth during development — it versions and traces
training runs. But the API used it for a single file load, at the cost of
SQLAlchemy, Alembic and Flask being imported on every startup.

The model is now exported once to `models/model.joblib` by `src/export_model.py`.
Measured gains:

| Metric | Before | After |
|---|---|---|
| Container memory | 637 MiB | 410 MiB |
| Image size | 532 MB | 443 MB |
| Test suite runtime | 3.96 s | 1.03 s |

The dependency reorganisation that came with it also took Dependabot alerts from
73 down to 1: most of the attack surface came from training libraries that had no
business being in production.

## Updating the model

```bash
# 1. Retrain and register in MLflow (notebook)
# 2. Export the new version
uv run --group train python -m src.export_model
# 3. Recompute history if the source data changed
uv run python -m src.precompute_history
# 4. Verify, then commit
uv run pytest
```

The `MODEL_VERSION` environment variable must be updated accordingly: it is
returned with every response and is what ties a decision back to the version that
produced it.

## Known limitations

- **`build_features` is not covered by automated tests.** It is verified by
  `uv run python -m src.pipeline`, which replays the full pipeline on 5,000 real
  clients. A synthetic test would have been more brittle and less convincing.
- **The API is unauthenticated.** API-key protection is planned.
- **The bundled history covers 20,000 clients**, a demonstration subset. In
  production these aggregations would live in a database.
- **Free hosting spins down**, with the cold-start delay described above.

## Roadmap

**Step 3 — Production monitoring.** Prediction logging to PostgreSQL, data-drift
analysis with Evidently, Streamlit monitoring dashboard.

**Step 4 — Performance.** Pipeline profiling, inference optimisation (ONNX
Runtime), further memory reduction.

## Author

**Tcheunzy** — Project 8, OpenClassrooms Data Scientist track.

The model and exploratory analysis come from the upstream project
(*Implémentez un modèle de scoring*); its notebook is kept in `notebook/`.
