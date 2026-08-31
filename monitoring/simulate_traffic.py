"""Envoie des dossiers réels à l'API, pour alimenter la base de production.

Deux modes :
  normal  — dossiers tirés de application_test, population conforme à l'entraînement
  dérive  — mêmes dossiers, distributions volontairement décalées
"""
import argparse
import os
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from api.schemas import ClientPredictionInput

# Sans ce chargement, API_KEY vaudrait None même quand le .env la définit,
# et le générateur recevrait des 401 dès que l'API est protégée.
load_dotenv()

API_KEY = os.getenv("API_KEY")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

OBLIGATOIRES = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED",
                "CNT_FAM_MEMBERS", "CNT_CHILDREN", "CODE_GENDER",
                "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                "NAME_EDUCATION_TYPE"]

ENTIERS = ["SK_ID_CURR", "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_CHILDREN"]


def colonnes_suivies() -> list[str]:
    return [nom for nom in ClientPredictionInput.model_fields if nom != "extra_fields"]


def construire_payload(ligne: pd.Series) -> dict | None:
    """Convertit une ligne du CSV en requête JSON valide, ou None si inutilisable."""
    if ligne[OBLIGATOIRES].isnull().any():
        return None
    if ligne["DAYS_EMPLOYED"] > 0:          # code sentinelle 365243
        return None

    payload = {}
    for nom, valeur in ligne.items():
        if pd.isna(valeur):
            continue                        # champ optionnel absent : on l'omet
        if nom in ENTIERS:
            payload[nom] = int(valeur)
        elif isinstance(valeur, (np.floating, float)):
            payload[nom] = float(valeur)
        elif isinstance(valeur, (np.integer, int)):
            payload[nom] = int(valeur)
        else:
            payload[nom] = str(valeur)
    return payload


def appliquer_derive(df: pd.DataFrame) -> pd.DataFrame:
    """Décale volontairement la population, pour démontrer la détection."""
    df = df.copy()
    df["AMT_CREDIT"] = df["AMT_CREDIT"] * 1.8
    df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"] * 0.6
    # Population plus jeune, en restant dans les bornes acceptées par l'API
    df["DAYS_BIRTH"] = (df["DAYS_BIRTH"] * 0.75).clip(-25000, -6600).astype(int)
    df["CODE_GENDER"] = "F"
    return df


def main():
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--n", type=int, default=150, help="nombre de dossiers")
    parseur.add_argument("--derive", action="store_true",
                         help="décale les distributions pour provoquer une dérive")
    parseur.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"))
    parseur.add_argument("--source", choices=["test", "train"], default="test",
                         help="test = clients inconnus du magasin ; "
                              "train = clients dont l'historique est présent")
    args = parseur.parse_args()

    if args.source == "train":
        # Les 20 000 premiers clients : ceux dont precompute_history a
        # calculé les agrégations.
        df = pd.read_csv(DATA / "application_train.csv",
                         usecols=colonnes_suivies(), nrows=20000)
    else:
        df = pd.read_csv(DATA / "application_test.csv", usecols=colonnes_suivies())

    df = df.sample(n=min(args.n * 2, len(df)), random_state=42)

    if args.derive:
        df = appliquer_derive(df)
        print("Mode dérive : crédits x1.8, revenus x0.6, population rajeunie, genre F")

    envoyes = ignores = echecs = 0

    # La clé appartient à la session, pas à chaque requête : la placer sur le
    # client évite d'avoir à y penser à chaque appel, et un oubli produirait
    # ici 150 rejets d'affilée.
    entetes = {"X-API-Key": API_KEY} if API_KEY else {}

    with httpx.Client(timeout=60.0, headers=entetes) as client:
        for _, ligne in df.iterrows():
            if envoyes >= args.n:
                break

            payload = construire_payload(ligne)
            if payload is None:
                ignores += 1
                continue

            try:
                reponse = client.post(f"{args.url}/predict", json=payload)

                if reponse.status_code == 200:
                    envoyes += 1
                    if envoyes % 25 == 0:
                        print(f"  {envoyes} envoyés…")
                else:
                    echecs += 1
                    if echecs <= 3:
                        print(f"  {reponse.status_code} : {reponse.text[:200]}")
            except httpx.RequestError as exc:
                echecs += 1
                print(f"  connexion : {exc}")

    print(f"\n{envoyes} prédictions envoyées, {ignores} dossiers ignorés "
          f"(champ obligatoire manquant ou code sentinelle), {echecs} échecs")

if __name__ == "__main__":
    main()