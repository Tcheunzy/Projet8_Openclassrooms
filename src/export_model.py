"""Exporte le modèle du registre MLflow vers un format chargeable sans MLflow.

À relancer uniquement lors d'un changement de version du modèle.
MLflow reste la source de vérité ; ce fichier en est une copie figée.
"""
from pathlib import Path

import joblib
import mlflow.lightgbm

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "models" / "mlflow_model"
CIBLE = ROOT / "models" / "model.joblib"


def main():
    model = mlflow.lightgbm.load_model(str(SOURCE))
    joblib.dump(model, CIBLE)
    print(f"Modèle exporté : {CIBLE} ({CIBLE.stat().st_size / 1e6:.1f} Mo)")
    print(f"Features attendues : {model.n_features_in_}")


if __name__ == "__main__":
    main()