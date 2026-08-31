"""Fige un echantillon des donnees d'entrainement comme jeu de reference.

Ce fichier sert de point de comparaison pour la detection de derive :
il represente ce que le modele a vu pendant son apprentissage.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from api.schemas import ClientPredictionInput

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CIBLE = ROOT / "models" / "reference.parquet"

N_LIGNES = 10000


def colonnes_journalisees() -> list[str]:
    """Les colonnes que l'API enregistre reellement, lues depuis le schema.

    Deriver la liste du modele Pydantic plutot que la recopier garantit
    qu'elle ne pourra jamais se desynchroniser du contrat de l'API.
    """
    return [nom for nom in ClientPredictionInput.model_fields if nom != "extra_fields"]


def main():
    colonnes = colonnes_journalisees()
    print(f"{len(colonnes)} colonnes suivies")

    df = pd.read_csv(DATA / "application_train.csv", usecols=colonnes)
    df = df.sample(n=min(N_LIGNES, len(df)), random_state=42)

    # Le code sentinelle n'existe pas cote production : l'API refuse les
    # valeurs positives. Le neutraliser evite une fausse alerte de derive.
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})

    df.to_parquet(CIBLE, index=False)
    print(f"Reference figee : {CIBLE} ({CIBLE.stat().st_size / 1e6:.2f} Mo)")
    print(df.describe().T[["mean", "min", "max"]].head(10))


if __name__ == "__main__":
    main()