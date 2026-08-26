"""Tests de l'orchestration du pipeline."""
import pandas as pd

from src.pipeline import build_aggregations


def test_build_aggregations_produit_les_six_tables():
    """build_aggregations doit retourner le dictionnaire attendu par merge_aggregations,
    et traiter les codes sentinelles AVANT d'agréger."""
    params = {"previous_days_cols": ["DAYS_FIRST_DUE"]}

    bureau = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_BUREAU": [100, 101],
        "AMT_CREDIT_SUM": [5000, 15000],
        "CREDIT_ACTIVE": ["Active", "Closed"],
        "CREDIT_CURRENCY": ["currency 1", "currency 1"],
        "CREDIT_TYPE": ["Consumer credit", "Car loan"],
    })
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": [100, 101],
        "MONTHS_BALANCE": [-1, -1],
        "STATUS": ["C", "0"],
    })
    previous_app = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_PREV": [10, 11],
        "AMT_APPLICATION": [1000, 3000],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused"],
        "DAYS_FIRST_DUE": [-500, 365243],
    })
    credit_card_balance = pd.DataFrame({
        "SK_ID_CURR": [1], "SK_ID_PREV": [10], "AMT_BALANCE": [250],
    })
    installments_payments = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_PREV": [10, 11],
        "AMT_INSTALMENT": [100, 100],
        "AMT_PAYMENT": [100, 80],
        "DAYS_INSTALMENT": [-30, -60],
        "DAYS_ENTRY_PAYMENT": [-35, -55],
    })
    pos_cash_balance = pd.DataFrame({
        "SK_ID_CURR": [1], "SK_ID_PREV": [10], "CNT_INSTALMENT": [12],
    })

    aggs = build_aggregations(
        bureau, bureau_balance, previous_app,
        credit_card_balance, installments_payments, pos_cash_balance,
        params,
    )

    assert set(aggs) == {
        "bureau_agg", "previous_agg", "cc_agg",
        "inst_agg", "pos_agg", "payment_behavior_agg",
    }

    # Chaque table est indexable par client
    for table in aggs.values():
        assert "SK_ID_CURR" in table.columns

    # Le code sentinelle 365243 a bien été neutralisé AVANT l'agrégation :
    # la moyenne ne doit pas être polluée, et le drapeau doit exister
    assert "PREV_DAYS_FIRST_DUE_ANOM_sum" in aggs["previous_agg"].columns
    assert aggs["previous_agg"]["PREV_DAYS_FIRST_DUE_ANOM_sum"].iloc[0] == 1
    assert aggs["previous_agg"]["PREV_DAYS_FIRST_DUE_mean"].iloc[0] == -500

    assert aggs["payment_behavior_agg"]["LATE_PAYMENT_RATE"].iloc[0] == 0.5