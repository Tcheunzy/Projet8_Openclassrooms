"""Pré-calcule les agrégations historiques par client, une fois, hors ligne.

Ce script n'est pas appelé par l'API : il produit les fichiers parquet que
l'API se contentera de lire au démarrage. À relancer uniquement quand les
données historiques changent.
"""
from pathlib import Path

import pandas as pd

from src.pipeline import build_aggregations, load_params
import json

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STORE = ROOT / "models" / "history"

N_CLIENTS = 20000  # sous-ensemble de démonstration


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    params = load_params(ROOT / "models" / "preprocessing_params.json")

    app_ids = pd.read_csv(DATA / "application_train.csv", usecols=["SK_ID_CURR"], nrows=N_CLIENTS)
    ids = set(app_ids["SK_ID_CURR"])
    print(f"{len(ids)} clients retenus")

    def load_filtered(name):
        df = pd.read_csv(DATA / f"{name}.csv")
        return df[df["SK_ID_CURR"].isin(ids)]

    bureau = load_filtered("bureau")
    bureau_balance = pd.read_csv(DATA / "bureau_balance.csv")
    bureau_balance = bureau_balance[bureau_balance["SK_ID_BUREAU"].isin(bureau["SK_ID_BUREAU"])]

    aggs = build_aggregations(
        bureau, bureau_balance,
        load_filtered("previous_application"),
        load_filtered("credit_card_balance"),
        load_filtered("installments_payments"),
        load_filtered("POS_CASH_balance"),
        params,
    )

    for name, df in aggs.items():
        path = STORE / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"{name:24s} {df.shape}  ->  {path.stat().st_size / 1e6:.1f} Mo")

    columns = [c for c in pd.read_csv(DATA / "application_train.csv", nrows=0).columns if c != "TARGET"]
    with open(ROOT / "models" / "application_columns.json", "w") as f:
        json.dump(columns, f, indent=2)

if __name__ == "__main__":
    main()