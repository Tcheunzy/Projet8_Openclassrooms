import json

import joblib
import pandas as pd

from src.cleaning import handle_days_anomaly, apply_document_grouping, cap_column
from src.aggregation import aggregate_table, aggregate_bureau, aggregate_previous_application
from src.feature_engineering import (
    add_application_ratios,
    add_debt_to_income,
    add_ext_source_features,
    aggregate_payment_behavior,
    replace_infinities,
)

# Colonnes supprimées par le clustering de corrélation mais indispensables au
# feature engineering. Le notebook les supprimait puis les rechargeait depuis le
# CSV ; ici on les protège simplement de la suppression (résultat identique).
PROTECTED_FROM_DROP = ['AMT_CREDIT', 'CNT_FAM_MEMBERS']

# Supprimée explicitement en cellule 98 du notebook
EXTRA_DROP = ['PREV_NAME_CASH_LOAN_PURPOSE_XAP_mean']


def load_params(path: str) -> dict:
    """Charge les artefacts de prétraitement figés lors de l'entraînement."""
    with open(path) as f:
        return json.load(f)


def clean_application(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Nettoyage de la table application, dans l'ordre exact du notebook (Blocs 2 et 4).
    L'ordre importe : pct_nan_row est calculé sur l'ensemble de colonnes existant
    à ce moment précis, donc avant les suppressions.
    """
    df = df.copy()

    # Bloc 2.2 : code sentinelle 365243 -> NaN + flag
    df = handle_days_anomaly(df, columns=['DAYS_EMPLOYED'])

    # Bloc 2.4 : capping du revenu au 99.5e percentile figé à l'entraînement
    df = cap_column(df, 'AMT_INCOME_TOTAL', upper=params['amt_income_total_cap'])

    # Bloc 3 : proportion de valeurs manquantes par client (variable prédictive)
    cols_to_check = [c for c in df.columns if c not in ['SK_ID_CURR', 'TARGET']]
    df['pct_nan_row'] = df[cols_to_check].isnull().mean(axis=1)

    # Bloc 4.1 : regroupement des FLAG_DOCUMENT_* + suppressions pures
    df = apply_document_grouping(df, params['flag_document_quasi_constant'])
    df = df.drop(columns=params['cols_to_drop_pure'], errors='ignore')

    # Bloc 4.2 : bloc logement (on ne garde que les versions _AVG)
    df = df.drop(columns=params['cols_to_drop_housing'], errors='ignore')

    # Bloc 4.3 : clusters de variables corrélées
    df = df.drop(columns=params['cols_to_drop_clusters'], errors='ignore')

    return df


def build_aggregations(bureau, bureau_balance, previous_app,
                       credit_card_balance, installments_payments,
                       pos_cash_balance, params: dict) -> dict:
    """
    Construit toutes les tables agrégées au niveau SK_ID_CURR (Bloc 5).
    Retourne un dictionnaire nom -> DataFrame prêt à fusionner.
    """
    # Les codes sentinelles de previous_application doivent être traités
    # AVANT l'agrégation, sinon 365243 pollue les moyennes.
    previous_app = handle_days_anomaly(previous_app, columns=params['previous_days_cols'])

    return {
        'bureau_agg': aggregate_bureau(bureau, bureau_balance),
        'previous_agg': aggregate_previous_application(previous_app),
        'cc_agg': aggregate_table(credit_card_balance, 'SK_ID_CURR', 'CC', ['SK_ID_PREV']),
        'inst_agg': aggregate_table(installments_payments, 'SK_ID_CURR', 'INST', ['SK_ID_PREV']),
        'pos_agg': aggregate_table(pos_cash_balance, 'SK_ID_CURR', 'POS', ['SK_ID_PREV']),
        'payment_behavior_agg': aggregate_payment_behavior(installments_payments),
    }


def merge_aggregations(df: pd.DataFrame, aggs: dict, params: dict) -> pd.DataFrame:
    """
    Fusionne les tables agrégées dans l'ordre du notebook, puis applique les
    suppressions post-fusion.
    """
    df = df.copy()
    n_rows = len(df)

    for name in ['bureau_agg', 'previous_agg', 'cc_agg', 'inst_agg', 'pos_agg']:
        df = df.merge(aggs[name], on='SK_ID_CURR', how='left')
        assert len(df) == n_rows, f"Duplication de lignes lors de la fusion de {name}"

    df = df.drop(columns=EXTRA_DROP, errors='ignore')

    # Suppression des variables trop corrélées, en protégeant celles dont le
    # feature engineering a besoin
    to_drop = [c for c in params['drop_list_restricted']
               if c in df.columns and c not in PROTECTED_FROM_DROP]
    df = df.drop(columns=to_drop)

    return df


def add_features(df: pd.DataFrame, aggs: dict, params: dict) -> pd.DataFrame:
    """Feature engineering (Bloc 6), dans l'ordre du notebook."""
    df = df.copy()

    df = add_ext_source_features(df)
    df = add_application_ratios(df)
    df = add_debt_to_income(df, cap_debt=params['debt_to_income_cap'])

    n_rows = len(df)
    df = df.merge(aggs['payment_behavior_agg'], on='SK_ID_CURR', how='left')
    assert len(df) == n_rows, "Duplication de lignes lors de la fusion de payment_behavior_agg"

    df = replace_infinities(df)

    return df


def get_expected_columns(preprocessor) -> list:
    """
    Extrait du préprocesseur ajusté la liste exacte des colonnes qu'il attend,
    dans l'ordre. C'est le contrat d'entrée du modèle : il est figé dans
    preprocessor.joblib, donc pas besoin de le stocker séparément.
    """
    expected = []
    for _, _, columns in preprocessor.transformers_:
        if isinstance(columns, list):
            expected.extend(columns)
    return expected


def build_features(app, bureau, bureau_balance, previous_app,
                   credit_card_balance, installments_payments, pos_cash_balance,
                   params: dict, preprocessor=None) -> pd.DataFrame:
    """
    Pipeline complet : données brutes -> DataFrame prêt pour preprocessor.transform().

    Si `preprocessor` est fourni, le résultat est réaligné sur les colonnes
    exactes attendues par le modèle. C'est indispensable en production : quand on
    agrège l'historique d'UN seul client, les catégories absentes de son
    historique (par ex. CREDIT_TYPE_Microloan) ne génèrent aucune colonne one-hot,
    alors que le modèle l'attend. Le reindex les recrée à NaN, que l'imputer
    traitera comme n'importe quelle valeur manquante.
    """
    df = clean_application(app, params)
    aggs = build_aggregations(bureau, bureau_balance, previous_app,
                              credit_card_balance, installments_payments,
                              pos_cash_balance, params)
    df = merge_aggregations(df, aggs, params)
    df = add_features(df, aggs, params)

    if preprocessor is not None:
        expected = get_expected_columns(preprocessor)
        sk_id = df['SK_ID_CURR'] if 'SK_ID_CURR' in df.columns else None
        df = df.reindex(columns=expected)
        df = df.copy()          # défragmente le DataFrame
        if sk_id is not None:
            df.insert(0, 'SK_ID_CURR', sk_id.values)

    return df
import re


def transform_for_model(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """
    Applique le préprocesseur ajusté et restitue un DataFrame dont les noms de
    colonnes sont nettoyés exactement comme à l'entraînement.
    LightGBM refuse les caractères spéciaux JSON dans les noms de features
    (issus des modalités one-hot du type "Trade: type 3"), d'où ce nettoyage —
    qui doit être rigoureusement identique côté entraînement et côté service.
    """
    X = df.drop(columns=['SK_ID_CURR'], errors='ignore')
    transformed = preprocessor.transform(X)
    columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c))
               for c in preprocessor.get_feature_names_out()]
    return pd.DataFrame(transformed, columns=columns, index=X.index)





if __name__ == "__main__":
    import joblib

    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data"

    params = load_params(ROOT / "models" / "preprocessing_params.json")
    preprocessor = joblib.load(ROOT / "models" / "preprocessor.joblib")

    app = pd.read_csv(DATA / "application_train.csv", nrows=5000)
    ids = set(app["SK_ID_CURR"])

    def load_filtered(name):
        d = pd.read_csv(DATA / f"{name}.csv")
        return d[d["SK_ID_CURR"].isin(ids)] if "SK_ID_CURR" in d.columns else d

    bureau = load_filtered("bureau")
    bureau_balance = pd.read_csv(DATA / "bureau_balance.csv")
    bureau_balance = bureau_balance[bureau_balance["SK_ID_BUREAU"].isin(bureau["SK_ID_BUREAU"])]

    result = build_features(
        app, bureau, bureau_balance,
        load_filtered("previous_application"),
        load_filtered("credit_card_balance"),
        load_filtered("installments_payments"),
        load_filtered("POS_CASH_balance"),
        params, preprocessor,
    )

    print("Shape :", result.shape)

    X = transform_for_model(result, preprocessor)
    print("Transform OK :", X.shape)

    import mlflow.lightgbm
    model = mlflow.lightgbm.load_model(str(ROOT / "models" / "mlflow_model"))
    proba = model.predict_proba(X)[:, 1]
    print("Probabilités de défaut — 5 premiers clients :", proba[:5].round(4))
    print("Taux de défaut prédit (seuil 0.24) :", (proba > 0.24).mean().round(4))