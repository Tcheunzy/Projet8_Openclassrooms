"""Profilage d'une prédiction, hors HTTP et hors base de données.

Mesure le pipeline seul : ni le réseau, ni la journalisation, ni le
démarrage du serveur ne viennent brouiller le résultat.
"""
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import joblib
import pandas as pd

from api.schemas import ClientPredictionInput
from src.pipeline import (add_features, clean_application, get_expected_columns,
                          load_params, merge_aggregations, transform_for_model)

ROOT = Path(__file__).resolve().parent.parent

HISTORY_TABLES = ["bureau_agg", "previous_agg", "cc_agg",
                  "inst_agg", "pos_agg", "payment_behavior_agg"]

# Un client présent dans le magasin d'historique : le chemin le plus coûteux.
DOSSIER = {
    "SK_ID_CURR": 100002,
    "AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000, "AMT_GOODS_PRICE": 450000,
    "DAYS_BIRTH": -16000, "DAYS_EMPLOYED": -2000,
    "CNT_FAM_MEMBERS": 2, "CNT_CHILDREN": 0,
    "CODE_GENDER": "F", "NAME_CONTRACT_TYPE": "Cash loans",
    "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "Y",
    "NAME_EDUCATION_TYPE": "Higher education",
}


def charger_artefacts() -> tuple[dict, float]:
    """Reproduit le lifespan de l'API. Retourne aussi sa durée : c'est elle
    que paie un démarrage à froid sur l'hébergement."""
    debut = time.perf_counter()
    ml = {
        "model": joblib.load(ROOT / "models" / "model.joblib"),
        "preprocessor": joblib.load(ROOT / "models" / "preprocessor.joblib"),
        "params": load_params(ROOT / "models" / "preprocessing_params.json"),
    }
    with open(ROOT / "models" / "application_columns.json") as f:
        ml["app_columns"] = json.load(f)
    ml["history"] = {
        nom: pd.read_parquet(ROOT / "models" / "history" / f"{nom}.parquet")
        for nom in HISTORY_TABLES
    }
    return ml, time.perf_counter() - debut


def predire_chronometre(ml: dict, payload: ClientPredictionInput):
    """Rejoue l'endpoint étape par étape, en chronométrant chacune.

    /!\\ Cette fonction duplique volontairement la logique de api/main.py :
    on ne peut pas chronométrer l'intérieur d'une fonction depuis l'extérieur.
    Toute modification de l'endpoint doit être reportée ici.
    """
    temps, t = {}, time.perf_counter

    t0 = t(); df = payload.to_dataframe()
    temps["1. conversion en DataFrame"] = t() - t0

    t0 = t(); df = df.reindex(columns=ml["app_columns"])
    temps["2. contrat d'entrée"] = t() - t0

    t0 = t(); df = clean_application(df, ml["params"])
    temps["3. nettoyage"] = t() - t0

    t0 = t()
    aggs = {nom: table[table["SK_ID_CURR"] == payload.SK_ID_CURR]
            for nom, table in ml["history"].items()}
    temps["4. recherche d'historique"] = t() - t0

    t0 = t(); df = merge_aggregations(df, aggs, ml["params"])
    temps["5. fusion"] = t() - t0

    t0 = t(); df = add_features(df, aggs, ml["params"])
    temps["6. feature engineering"] = t() - t0

    t0 = t(); df = df.reindex(columns=get_expected_columns(ml["preprocessor"]))
    temps["7. contrat de sortie"] = t() - t0

    t0 = t(); X = transform_for_model(df, ml["preprocessor"])
    temps["8. prétraitement"] = t() - t0

    t0 = t(); proba = float(ml["model"].predict_proba(X)[0, 1])
    temps["9. prédiction du modèle"] = t() - t0

    return proba, temps


def mesurer_par_etape(ml: dict, payload, n: int = 100) -> pd.DataFrame:
    cumuls: dict[str, float] = {}
    for _ in range(n):
        _, temps = predire_chronometre(ml, payload)
        for etape, duree in temps.items():
            cumuls[etape] = cumuls.get(etape, 0.0) + duree

    tableau = pd.DataFrame([{"étape": e, "ms": 1000 * d / n}
                            for e, d in cumuls.items()])
    tableau["part"] = tableau["ms"] / tableau["ms"].sum()
    return tableau.sort_values("ms", ascending=False)


def profiler(ml: dict, payload, n: int = 100, lignes: int = 20) -> str:
    profil = cProfile.Profile()
    profil.enable()
    for _ in range(n):
        predire_chronometre(ml, payload)
    profil.disable()

    flux = io.StringIO()
    pstats.Stats(profil, stream=flux).sort_stats("cumulative").print_stats(lignes)
    return flux.getvalue()


def main():
    ml, duree_chargement = charger_artefacts()
    payload = ClientPredictionInput(**DOSSIER)

    proba, _ = predire_chronometre(ml, payload)
    print(f"Chargement des artefacts : {duree_chargement:.2f} s")
    print(f"Probabilité de contrôle  : {proba:.6f}")
    print()

    tableau = mesurer_par_etape(ml, payload)
    total = tableau["ms"].sum()
    print(f"Répartition du temps sur 100 prédictions ({total:.1f} ms au total)")
    print(tableau.to_string(index=False, formatters={
        "ms": "{:.2f}".format, "part": "{:.0%}".format}))
    print()

    print("Détail par fonction (cProfile, 20 premières lignes)")
    print(profiler(ml, payload))


if __name__ == "__main__":
    main()