"""API de scoring crédit — Prêt à dépenser."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow.lightgbm
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import ClientPredictionInput, PredictionResponse
from src.pipeline import (
    add_features,
    clean_application,
    get_expected_columns,
    load_params,
    merge_aggregations,
    transform_for_model,
)

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
    ml["model"] = mlflow.lightgbm.load_model(str(ROOT / "models" / "mlflow_model"))
    ml["preprocessor"] = joblib.load(ROOT / "models" / "preprocessor.joblib")
    ml["params"] = load_params(ROOT / "models" / "preprocessing_params.json")

    with open(ROOT / "models" / "application_columns.json") as f:
        ml["app_columns"] = json.load(f)

    ml["history"] = {
        name: pd.read_parquet(ROOT / "models" / "history" / f"{name}.parquet")
        for name in HISTORY_TABLES
    }
    yield
    ml.clear()


app = FastAPI(
    title="Prédicteur de remboursement crédit",
    description="Expose un modèle LightGBM entraîné sur le dataset Home Credit.",
    version="0.1.0",
    lifespan=lifespan,
)


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
def predict(payload: ClientPredictionInput):
    try:
        df = payload.to_dataframe()
        df = df.reindex(columns=ml["app_columns"])
        df = clean_application(df, ml["params"])

        # ÉTAPE 1 — récupérer l'historique du client
        aggs = {
            name: table[table["SK_ID_CURR"] == payload.SK_ID_CURR]
            for name, table in ml["history"].items()
        }
        # ÉTAPE 2 — enrichir
        df = merge_aggregations(df, aggs, ml["params"])
        df = add_features(df, aggs, ml["params"])

        # ÉTAPE 3 — réaligner sur le contrat du modèle
        df = df.reindex(columns=get_expected_columns(ml["preprocessor"]))

        # ÉTAPE 4 — prédire
        X = transform_for_model(df, ml["preprocessor"])
        proba = float(ml["model"].predict_proba(X)[0, 1])

        # ÉTAPE 5 — return
        return PredictionResponse(
            sk_id_curr=payload.SK_ID_CURR,
            probability=proba,
            threshold=THRESHOLD,
            decision="refusé" if proba > THRESHOLD else "accordé",
            mlflow_model_version=MODEL_VERSION,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {exc}")