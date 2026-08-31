# Credit Scoring — Prêt à dépenser

[![CI](https://github.com/Tcheunzy/Projet8_Openclassrooms/actions/workflows/ci.yml/badge.svg)](https://github.com/Tcheunzy/Projet8_Openclassrooms/actions/workflows/ci.yml)

Productionising a credit-scoring model: FastAPI service, Gradio demo interface,
Docker image, CI/CD pipeline, production monitoring with drift detection, and a
profiled prediction path.

The model (LightGBM, trained on the Home Credit dataset) estimates the
probability that an applicant will default, and turns that probability into a
lending decision using a threshold optimised on a business cost function.

## Live demo

| Interface | URL | Intended audience |
|---|---|---|
| Scoring interface | [/gradio](https://projet8-scoring-credit.onrender.com/gradio) | Loan officer |
| API documentation | [/docs](https://projet8-scoring-credit.onrender.com/docs) | Integrating developer |
| Health endpoint | [/health](https://projet8-scoring-credit.onrender.com/health) | Operations |

The monitoring dashboard runs locally: `uv run streamlit run monitoring/dashboard.py`.

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

Once the model is live, a second problem appears: the world moves and the model
does not. Every prediction is therefore logged, and production inputs are
compared against the training distribution — see *Production monitoring*.

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
11.  Response returned               the caller waits no longer than this
12.  Background logging              inputs + decision + latency -> PostgreSQL
```

Steps 4, 6 and 8 run **exactly the same code** as training, imported from `src/`.
Step 12 runs *after* the response is sent, and never fails the request.

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
  precompute_history.py    offline aggregation job
  export_model.py          exports the model out of the MLflow registry
database/
  predictions.py           connection pool, schema, insert and read
  init_db.py               one-off schema creation
  comparer_latences.py     production latency, before and after optimisation
benchmarks/
  profile_prediction.py    per-stage timing and cProfile of one prediction
monitoring/
  build_reference.py       freezes the training-distribution reference
  drift.py                 Evidently analysis and result summary
  dashboard.py             Streamlit monitoring dashboard
  simulate_traffic.py      traffic generator, with a drift mode
models/
  model.joblib                 production model (frozen copy of version 4)
  preprocessor.joblib          fitted ColumnTransformer
  preprocessing_params.json    frozen thresholds and column lists
  application_columns.json     input contract
  reference.parquet            drift-detection reference sample
  history/*.parquet            per-client aggregations
tests/                     36 unit and functional tests
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

Four dependency groups are kept apart:

| Group | Contents | Command |
|---|---|---|
| main | what the API loads to serve a prediction | `uv sync --no-dev` |
| `dev` | pytest, coverage (installed by default) | `uv sync` |
| `monitoring` | Evidently, Streamlit — the dashboard only | `uv sync --group monitoring` |
| `train` | MLflow, Optuna, SHAP — never deployed | `uv sync --group train` |

Prediction logging needs a `DATABASE_URL`. Copy `.env.example` to `.env` and fill
it in; without it the API runs normally and simply logs nothing.

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

## Production monitoring

Every prediction is written to PostgreSQL: the inputs received, the decision, the
model version, the threshold applied and the end-to-end latency. Logging the
**inputs** — not just the outcome — is what makes drift detection possible at
all.

```bash
uv run python -m database.init_db                 # create the table (once)
uv run python -m monitoring.build_reference       # freeze the reference (once)
uv run streamlit run monitoring/dashboard.py      # open the dashboard
```

To generate traffic against a running API:

```bash
uv run python -m monitoring.simulate_traffic --n 300                  # unknown clients
uv run python -m monitoring.simulate_traffic --n 300 --source train   # known clients
uv run python -m monitoring.simulate_traffic --n 800 --derive         # shifted population
```

### What the dashboard shows

**Activity** — volume, refusal rate, median and 95th-percentile latency, share of
clients found in the history store.

**Drift** — how many input columns have moved away from the training
distribution, which ones, and by how much. Evidently picks the statistical test
per variable type: Wasserstein distance for numerical columns, Jensen-Shannon
distance for categorical ones.

**Distribution comparison** — reference against production for any column. The
table says *which* variables drifted; this chart says *in which direction*.

### What it found

On roughly 1,300 production predictions, 9 of 22 columns were flagged. To check
that this was not an artefact, `application_train` and `application_test` were
compared directly at 10,000 rows each — where sampling noise is negligible. Five
columns still drifted: `AMT_GOODS_PRICE`, `AMT_CREDIT`, `AMT_ANNUITY`,
`NAME_CONTRACT_TYPE` and `DAYS_BIRTH`.

**The drift is real.** The applications the model scores today come from a
younger population borrowing larger amounts than the one it learned from. That is
a retraining signal, and the monitoring surfaced it.

## Performance

Latency was profiled before anything was optimised. `benchmarks/profile_prediction.py`
times the nine stages of a prediction over 100 runs, and `cProfile` confirms the
ranking function by function.

```bash
uv run python -m benchmarks.profile_prediction
```

The first measurement contradicted the intuition it was meant to confirm. The
history lookup — six parquet tables scanned per request — was the suspected
bottleneck; it accounted for 1% of the time. A single line of feature
engineering accounted for more than the model itself.

### What was changed

**Infinity replacement, column by column, cost 29 ms.** Division-derived ratios
can produce infinities, which were replaced through a loop over the 447 numeric
columns. Applied to the whole frame instead, the same result costs 0.4 ms — pandas
dispatches one vectorised pass rather than 447 assignments, each of which
reallocated a column.

**The output column contract was rebuilt on every request.** It only depends on
`preprocessor.joblib`, so it is now read once at startup and stored alongside the
model.

**Feature names were recomputed on every request.** `get_feature_names_out()`
followed by a regular expression over 575 names is not free, and the answer never
changes. Same treatment: computed once in `lifespan`.

None of the three touches the arithmetic. The control probability for client
100002 is `0.107644` before and after, and the 36 tests pass unchanged — which is
the only reason the gain is worth anything.

### Result

| | Local (100 runs) | Production median | Production p95 |
|---|---|---|---|
| Before | 27.9 ms | 961 ms | 1,925 ms |
| After | 10.7 ms | 280 ms | 357 ms |
| Gain | −62% | −71% | −81% |

Production figures come from real logged calls — 154 before, 403 after — read
back with `uv run python -m database.comparer_latences`.

The p95 matters more than the median here: one call in twenty used to take over
two seconds, and none now exceeds 360 ms. The spread collapsed further than the
average did, which is what an API is judged on.

### Why production gained forty times more than the laptop

Locally the pipeline saved 17 ms per prediction. In production it saved 681 — a
factor of about forty, which is exactly the CPU allowance of the free Render
instance: 0.1 of a core.

The two measurements are independent, one from a profiler and one from live
traffic, and they agree. That agreement is the actual finding: **on constrained
infrastructure, code optimisation is not a refinement, it is what makes the
service usable.** The same 17 ms would have been invisible on a dedicated core.

### Why ONNX Runtime was not adopted

Converting the model to ONNX was the planned optimisation. Profiling made the
case against it: after the three changes above, LightGBM inference is 0.86 ms of
the remaining 10.7 — 8%. Even a free inference would cut a tenth of the latency.

The remaining 92% is pandas and scikit-learn: cleaning, merging, feature
engineering, encoding. ONNX does not cover that path, and exporting the model
alone would add a conversion step, a second artefact to keep in sync with
`preprocessor.joblib`, and a numerical-equivalence risk — for a gain smaller than
the one already obtained by deleting a loop.

The decision is therefore documented rather than implemented. Measuring first is
what turned a plausible optimisation into a demonstrably wrong one.

## Tests

```bash
uv run pytest -v
uv run pytest --cov=src --cov=api --cov-report=term-missing
```

36 tests, roughly 92% coverage on `src/` and `api/` (the CI enforces a 90%
floor). No test depends on the `data/` folder, on a database, or on a running
MLflow server: the suite runs on a bare machine, which is what makes continuous
integration possible at all.

| File | Scope |
|---|---|
| `test_cleaning.py` | cleaning functions on hand-built DataFrames |
| `test_aggregation.py` | aggregations, including the two-level `bureau` chain |
| `test_feature_engineering.py` | late-payment detection |
| `test_pipeline.py` | orchestration, and ordering of operations |
| `test_schemas.py` | validation: missing field, out-of-range value, wrong type |
| `test_api.py` | endpoints, unknown client, internal failure, DB outage |
| `test_gradio.py` | unit conversion and error handling, with HTTP mocked |
| `test_database.py` | parameter ordering of the insert, with a stub pool |
| `test_drift.py` | drift detection signals real shifts and stays silent otherwise |

Two of these deserve a mention. `test_journalisation_absorbe_une_panne_de_base`
has no assertion at all: what it verifies is the *absence* of an exception when
the database is unreachable. And `test_build_aggregations_produit_les_six_tables`
checks an *ordering*, not a value — if sentinel handling ran after aggregation
instead of before, one mean would be 182,371 instead of −500.

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
2. runs the 36 tests and enforces the coverage floor;
3. builds the Docker image;
4. starts the container and queries `/health`.

The `main` branch is protected: nothing merges unless these checks pass.

Building an image only proves it is syntactically valid; it is the fourth step —
starting the container and getting an answer — that proves the application
actually works inside it.

### Deployment is triggered by the pipeline, not by the commit

A fifth job calls Render's deploy hook, guarded by two conditions:

```yaml
deploy:
  needs: tests
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
```

`needs: tests` is the point of the whole arrangement. Render's own auto-deploy
setting reacts to commits and knows nothing about test results: a broken merge
would reach production regardless. Routing the deployment through the pipeline
means **nothing is deployed that has not been tested**.

The second condition excludes pull requests. Without it, opening a PR against
`main` would deploy unreviewed code, bypassing branch protection.

The hook URL is a `RENDER_DEPLOY_HOOK` repository secret, so it never appears in
the repository or in the workflow logs. Render's `Auto-Deploy` is set to `Off`,
leaving a single path to production.

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

### Monitoring must never break what it monitors

Prediction logging runs as a FastAPI background task — after the response has
been sent, so it costs the caller nothing — and every database error is caught
and swallowed. If `DATABASE_URL` is unset or the database is down, the API serves
predictions exactly as before and simply records nothing.

This is deliberate and tested: `test_predict_fonctionne_sans_base` and
`test_journalisation_absorbe_une_panne_de_base` both exist to keep it true. It is
also what lets the CI validate the Docker image with no database in sight.

Swallowing an exception has a cost, though, and it was paid before it was
noticed: a batch of predictions went unrecorded while the API kept answering
`healthy`, and nothing anywhere said so. Two changes closed that gap. Failures
are now logged with their traceback, and `/health` reports the state of the
logging path:

```json
{"status": "healthy", "model_loaded": true, "model_version": "4", "journalisation": true}
```

A degraded service that reports itself as healthy is worse than one that fails
loudly, because it removes the only signal an operator has. Fault tolerance and
silence are not the same thing.

### Drift needs volume, and the dashboard says so

A simulation with **no** real drift — reference and current drawn from the same
distribution — showed Evidently flagging 20% to 80% of columns at 157
observations, and 0% from 2,000 onwards. Below a few hundred rows, a drift report
mostly measures sampling noise.

The dashboard therefore displays the sample size and warns explicitly below 500
observations. A drift figure without its sample size misleads its reader, and a
team that learns to ignore false alarms will ignore the real one too.

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
  production these aggregations would live in a database alongside the logs.
- **The monitoring dashboard runs locally**, not as a deployed service.
- **The free PostgreSQL instance expires 30 days after creation.** The schema and
  the traffic generator make the setup reproducible in minutes.
- **Free hosting spins down**, with the cold-start delay described above.
- **The `predictions` table does not record where a call came from.** Separating
  local calls from production ones in the latency comparison relies on a
  threshold — 150 ms — which works only because the two populations are an order
  of magnitude apart. A column would have been better than a heuristic, and the
  lesson is that a logging table should carry the call's origin from day one.

## Next steps

- **API-key authentication**, the last gap in the service itself.
- **Deploying the monitoring dashboard**, so drift is visible without a laptop.
- **Scheduled drift reports**, turning the dashboard from something someone opens
  into something that raises its hand.

## Author

**Tcheunzy** — Project 8, OpenClassrooms Data Scientist track.

The model and exploratory analysis come from the upstream project
(*Implémentez un modèle de scoring*); its notebook is kept in `notebook/`.
