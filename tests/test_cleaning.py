import pandas as pd

from src.cleaning import apply_document_grouping, cap_column, handle_days_anomaly


def test_handle_days_anomaly_remplace_le_code_sentinelle():
    df = pd.DataFrame({"DAYS_EMPLOYED": [-500, 365243, -1200]})
    result = handle_days_anomaly(df, columns=["DAYS_EMPLOYED"])

    assert result["DAYS_EMPLOYED_ANOM"].tolist() == [False, True, False]
    assert result["DAYS_EMPLOYED"].isna().sum() == 1
    assert result["DAYS_EMPLOYED"].tolist()[0] == -500

def test_apply_document_grouping_somme_et_supprime_les_colonnes():
    df = pd.DataFrame({
        "FLAG_DOCUMENT_2": [1, 0],
        "FLAG_DOCUMENT_4": [0, 0],
        "FLAG_DOCUMENT_5": [1, 1],
    })
    flag_cols = ["FLAG_DOCUMENT_2", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5"]

    result = apply_document_grouping(df, flag_cols=flag_cols)

    assert result["TOTAL_DOCUMENTS_PROVIDED"].tolist() == [2, 1]
    for col in flag_cols:
        assert col not in result.columns
    

def test_cap_column_plafonne_avec_borne_haute():
    df = pd.DataFrame({"AMT_INCOME_TOTAL": [50000, 999999999]})

    result = cap_column(df, "AMT_INCOME_TOTAL", upper=500000)

    assert result["AMT_INCOME_TOTAL"].tolist() == [50000, 500000]
    assert result["AMT_INCOME_TOTAL_CAPPED_FLAG"].tolist() == [False, True]


def test_cap_column_plafonne_avec_deux_bornes():
    df = pd.DataFrame({"DEBT_TO_INCOME": [-0.5, 0.3, 50]})

    result = cap_column(df, "DEBT_TO_INCOME", lower=0, upper=10)

    assert result["DEBT_TO_INCOME"].tolist() == [0.0, 0.3, 10.0]
    assert result["DEBT_TO_INCOME_CAPPED_FLAG"].tolist() == [True, False, True]