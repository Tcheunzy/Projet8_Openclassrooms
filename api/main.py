"""API de scoring crédit — Prêt à dépenser."""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from api.gradio_app import build_demo
from api.schemas import ClientPredictionInput, PredictionResponse
from api.security import verifier_cle
from database.predictions import (
    STATUT_ERREUR,
    STATUT_SUCCES,
    STATUT_VALIDATION,
    create_pool,
    log_prediction,
)
from src.pipeline import (
    add_features,
    clean_application,
    get_expected_columns,
    load_params,
    merge_aggregations,
    noms_de_features,
    transform_for_model,
)

load_dotenv()

# Sans cette configuration, le logger « api » hérite du niveau par défaut
# (WARNING) et tous les messages d'information sont écartés avant d'atteindre
# la sortie standard — donc avant d'arriver dans les journaux de l'hébergeur.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s",
)
logger = logging.getLogger("api")

ROOT = Path(__file__).resolve().parent.parent
THRESHOLD = float(os.getenv("THRESHOLD", "0.24"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "4")
ORIGINE = os.getenv("ENVIRONNEMENT", "local")

HISTORY_TABLES = [
    "bureau_agg", "previous_agg", "cc_agg",
    "inst_agg", "pos_agg", "payment_behavior_agg",
]

ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge modèle et artefacts une seule fois, au démarrage du serveur."""
    ml["model"] = joblib.load(ROOT / "models" / "model.joblib")
    ml["preprocessor"] = joblib.load(ROOT / "models" / "preprocessor.joblib")

    # Ces deux listes ne dépendent que du préprocesseur : les recalculer
    # à chaque requête est du travail perdu.
    ml["expected_columns"] = get_expected_columns(ml["preprocessor"])
    ml["feature_names"] = noms_de_features(ml["preprocessor"])
    ml["params"] = load_params(ROOT / "models" / "preprocessing_params.json")

    with open(ROOT / "models" / "application_columns.json") as f:
        ml["app_columns"] = json.load(f)

    ml["history"] = {
        name: pd.read_parquet(ROOT / "models" / "history" / f"{name}.parquet")
        for name in HISTORY_TABLES
    }

    # La journalisation est optionnelle : sans base, l'API fonctionne normalement.
    ml["pool"] = None
    if os.getenv("DATABASE_URL"):
        try:
            ml["pool"] = create_pool()
            logger.info("Journalisation des predictions activee.")
        except Exception as exc:
            logger.warning("Base injoignable, journalisation desactivee : %s", exc)
    else:
        logger.warning("DATABASE_URL absente : journalisation desactivee.")

    if os.getenv("API_KEY"):
        logger.info("Authentification par cle d'API activee sur /predict.")
    else:
        logger.warning("API_KEY absente : /predict est accessible sans authentification.")

    yield

    if ml.get("pool") is not None:
        ml["pool"].close()
    ml.clear()


app = FastAPI(
    title="Prédicteur de remboursement crédit",
    description="Expose un modèle LightGBM entraîné sur le dataset Home Credit.",
    version="0.1.0",
    lifespan=lifespan,
)


def _journaliser(**kwargs) -> None:
    """Enregistre un appel à /predict. N'échoue jamais : une panne de la base
    ne doit pas remonter jusqu'à l'utilisateur — mais elle doit laisser une
    trace exploitable dans les journaux."""
    if ml.get("pool") is None:
        logger.warning("Journalisation ignoree : aucun pool de connexions.")
        return
    try:
        log_prediction(ml["pool"], **kwargs)
    except Exception as exc:
        logger.warning("Journalisation echouee : %s", exc, exc_info=True)


@app.exception_handler(RequestValidationError)
async def journaliser_les_rejets(request: Request, exc: RequestValidationError):
    """Enregistre les requêtes refusées par Pydantic.

    Ces rejets se produisent AVANT l'entrée dans la route : FastAPI valide le
    corps et répond 422 sans jamais appeler `predict`. Sans ce gestionnaire,
    aucune requête invalide ne laisse de trace et le taux d'erreur mesuré est
    structurellement nul.
    """
    erreurs = exc.errors()

    try:
        corps = await request.json()
    except Exception:
        corps = {}
    if not isinstance(corps, dict):
        corps = {}

    premiere = erreurs[0] if erreurs else {}
    champ = ".".join(str(p) for p in premiere.get("loc", [])[1:])
    type_erreur = premiere.get("type", "inconnu")
    error_type = f"{type_erreur}:{champ}" if champ else type_erreur

    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(erreurs)},
        # La journalisation part après l'envoi de la réponse, comme sur le
        # chemin nominal : le client n'attend jamais l'écriture en base.
        background=BackgroundTask(
            _journaliser,
            model_version=MODEL_VERSION,
            threshold=THRESHOLD,
            features=corps,
            sk_id_curr=corps.get("SK_ID_CURR"),
            status=STATUT_VALIDATION,
            error_type=error_type,
            origine=ORIGINE,
        ),
    )


@app.get("/", tags=["Monitoring"])
def root():
    return {"message": "API Prêt à dépenser", "status": "online"}


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Point de contrôle, volontairement non authentifié.

    La supervision de l'hébergeur l'interroge sans clé ; l'exiger ici ferait
    échouer les vérifications de démarrage de Render et de la CI.
    """
    return {
        "status": "healthy",
        "model_loaded": "model" in ml,
        "model_version": MODEL_VERSION,
        "journalisation": ml.get("pool") is not None,
        "authentification": bool(os.getenv("API_KEY")),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"],
          dependencies=[Depends(verifier_cle)])
def predict(payload: ClientPredictionInput, background_tasks: BackgroundTasks):
    debut = time.perf_counter()
    try:
        df = payload.to_dataframe()
        df = df.reindex(columns=ml["app_columns"])
        df = clean_application(df, ml["params"])

        # ÉTAPE 1 — récupérer l'historique du client
        aggs = {
            name: table[table["SK_ID_CURR"] == payload.SK_ID_CURR]
            for name, table in ml["history"].items()
        }
        history_found = any(not table.empty for table in aggs.values())

        # ÉTAPE 2 — enrichir
        df = merge_aggregations(df, aggs, ml["params"])
        df = add_features(df, aggs, ml["params"])

        # ÉTAPE 3 — réaligner sur le contrat du modèle
        df = df.reindex(columns=ml["expected_columns"])

        # ÉTAPE 4 — prédire
        X = transform_for_model(df, ml["preprocessor"], ml["feature_names"])
        proba = float(ml["model"].predict_proba(X)[0, 1])
        decision = "refusé" if proba > THRESHOLD else "accordé"

        # ÉTAPE 5 — programmer la journalisation (exécutée après la réponse)
        features = payload.model_dump(exclude={"extra_fields"})
        features.update(payload.extra_fields)

        background_tasks.add_task(
            _journaliser,
            sk_id_curr=payload.SK_ID_CURR,
            model_version=MODEL_VERSION,
            threshold=THRESHOLD,
            probability=proba,
            decision=decision,
            history_found=history_found,
            latency_ms=int((time.perf_counter() - debut) * 1000),
            status=STATUT_SUCCES,
            origine=ORIGINE,
            features=features,
        )

        # ÉTAPE 6 — répondre
        return PredictionResponse(
            sk_id_curr=payload.SK_ID_CURR,
            probability=proba,
            threshold=THRESHOLD,
            decision=decision,
            mlflow_model_version=MODEL_VERSION,
        )

    except Exception as exc:
        logger.exception("Echec de la prediction pour %s", payload.SK_ID_CURR)
        # Journalisation synchrone ici, et non par background_tasks : lorsque
        # la route lève une exception, la réponse est construite par le
        # gestionnaire d'erreurs de FastAPI et les tâches de fond enregistrées
        # sur la requête ne sont jamais exécutées.
        _journaliser(
            model_version=MODEL_VERSION,
            threshold=THRESHOLD,
            features=payload.model_dump(exclude={"extra_fields"}),
            sk_id_curr=payload.SK_ID_CURR,
            latency_ms=int((time.perf_counter() - debut) * 1000),
            status=STATUT_ERREUR,
            error_type=type(exc).__name__,
            origine=ORIGINE,
        )
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {exc}")


app = gr.mount_gradio_app(app, build_demo(), path="/gradio")