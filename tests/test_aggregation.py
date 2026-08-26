"""Tests unitaires des fonctions d'agrégation (chemin hors ligne du pipeline)."""
import pandas as pd

from src.aggregation import (
    aggregate_bureau,
    aggregate_previous_application,
    aggregate_table,
)


def test_aggregate_table_resume_par_client():
    """Chaque colonne numérique doit produire min, max, mean, sum, plus un compteur."""
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_PREV": [10, 11, 12],
        "AMT_BALANCE": [100, 200, 50],
    })

    result = aggregate_table(df, group_key="SK_ID_CURR", prefix="CC",
                             exclude_cols=["SK_ID_PREV"]).set_index("SK_ID_CURR")

    assert result.loc[1, "CC_AMT_BALANCE_sum"] == 300
    assert result.loc[1, "CC_AMT_BALANCE_mean"] == 150
    assert result.loc[1, "CC_COUNT"] == 2
    assert result.loc[2, "CC_AMT_BALANCE_sum"] == 50
    assert result.loc[2, "CC_COUNT"] == 1
    assert "SK_ID_PREV_sum" not in result.columns


def test_aggregate_previous_application_encode_les_categories_en_proportions():
    """Une colonne catégorielle doit devenir une proportion, pas une statistique brute."""
    previous_app = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_PREV": [10, 11, 12],
        "AMT_APPLICATION": [1000, 3000, 500],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
        "DAYS_FIRST_DUE_ANOM": [False, True, False],
    })

    result = aggregate_previous_application(previous_app).set_index("SK_ID_CURR")

    assert result.loc[1, "PREV_AMT_APPLICATION_mean"] == 2000
    assert result.loc[1, "PREV_APPLICATION_COUNT"] == 2
    assert result.loc[2, "PREV_APPLICATION_COUNT"] == 1

    # Une demande approuvée sur deux pour le client 1
    assert result.loc[1, "PREV_NAME_CONTRACT_STATUS_Approved_mean"] == 0.5

    # Les drapeaux d'anomalie survivent à l'agrégation
    assert result.loc[1, "PREV_DAYS_FIRST_DUE_ANOM_sum"] == 1


def test_aggregate_bureau_enchaine_les_deux_niveaux():
    """L'historique mensuel est résumé par crédit, puis les crédits par client."""
    bureau = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_BUREAU": [100, 101],
        "AMT_CREDIT_SUM": [5000, 15000],
        "CREDIT_ACTIVE": ["Active", "Closed"],
        "CREDIT_CURRENCY": ["currency 1", "currency 1"],
        "CREDIT_TYPE": ["Consumer credit", "Car loan"],
    })
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": [100, 100, 101],
        "MONTHS_BALANCE": [-1, -2, -1],
        "STATUS": ["C", "0", "C"],
    })

    result = aggregate_bureau(bureau, bureau_balance).set_index("SK_ID_CURR")

    assert len(result) == 1
    assert result.loc[1, "BUREAU_CREDIT_COUNT"] == 2
    assert result.loc[1, "AMT_CREDIT_SUM_sum"] == 20000

    # Un crédit actif sur les deux
    assert result.loc[1, "CREDIT_ACTIVE_Active_mean"] == 0.5

    # Le crédit 100 a deux mois d'historique, le crédit 101 un seul
    assert result.loc[1, "MONTHS_BALANCE_count_sum"] == 3


def test_aggregate_table_sans_colonne_categorielle():
    """Une table purement numérique doit s'agréger sans passer par l'encodage one-hot.

    C'est le cas réel d'installments_payments, qui ne contient aucune colonne texte.
    Les booléens sont convertis en entiers pour pouvoir être sommés.
    """
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_PREV": [10, 11],
        "AMT_PAYMENT": [100, 300],
        "EST_EN_RETARD": [True, False],
    })

    result = aggregate_table(df, group_key="SK_ID_CURR", prefix="INST",
                             exclude_cols=["SK_ID_PREV"]).set_index("SK_ID_CURR")

    assert result.loc[1, "INST_AMT_PAYMENT_mean"] == 200
    assert result.loc[1, "INST_EST_EN_RETARD_sum"] == 1
    assert result.loc[1, "INST_COUNT"] == 2

def test_aggregate_table_encode_les_colonnes_categorielles():
    """Une colonne texte devient une proportion par client, pas une statistique brute."""
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1],
        "SK_ID_PREV": [10, 11, 12],
        "AMT_BALANCE": [100, 200, 300],
        "NAME_CONTRACT_STATUS": ["Active", "Active", "Completed"],
    })

    result = aggregate_table(df, group_key="SK_ID_CURR", prefix="POS",
                             exclude_cols=["SK_ID_PREV"]).set_index("SK_ID_CURR")

    # Deux contrats actifs sur trois
    assert result.loc[1, "POS_NAME_CONTRACT_STATUS_Active_mean"] == 2 / 3
    assert result.loc[1, "POS_NAME_CONTRACT_STATUS_Completed_mean"] == 1 / 3
    assert result.loc[1, "POS_AMT_BALANCE_sum"] == 600