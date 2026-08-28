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
from fastapi import BackgroundTasks, FastAPI, HTTPException

from api.gradio_app import build_demo
from api.schemas import ClientPredictionInput, PredictionResponse
from database.predictions import create_pool, log_prediction
from src.pipeline import (
    add_features,
    clean_application,
    get_expected_columns,
    load_params,
    merge_aggregations,
    transform_for_model,
)

load_dotenv()
logger = logging.getLogger("api")

ROOT = Path(__file__).resolve().parent.parent
THRESHOLD = float(os.getenv("THRESHOLD", "0.24"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "4")

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
    """Enregistre une prédiction. N'échoue jamais : une panne de la base
    ne doit pas remonter jusqu'à l'utilisateur."""
    if ml.get("pool") is None:
        return
    try:
        log_prediction(ml["pool"], **kwargs)
    except Exception as exc:
        logger.warning("Journalisation echouee : %s", exc)


@app.get("/", tags=["Monitoring"])
def root():
    return {"message": "API Prêt à dépenser", "status": "online"}


@app.get("/health", tags=["Monitoring"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "model" in ml,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
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
        df = df.reindex(columns=get_expected_columns(ml["preprocessor"]))

        # ÉTAPE 4 — prédire
        X = transform_for_model(df, ml["preprocessor"])
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
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {exc}")


app = gr.mount_gradio_app(app, build_demo(), path="/gradio")