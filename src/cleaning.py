import pandas as pd
import numpy as np


def handle_days_anomaly(df: pd.DataFrame, columns: list, anomaly_value: int = 365243) -> pd.DataFrame:
    """
    Pour chaque colonne DAYS_* fournie :
    - crée un flag booléen indiquant si la valeur était le code sentinelle
    - remplace le code sentinelle par NaN
    """
    df = df.copy()  # ne jamais modifier le DataFrame reçu en argument
    for col in columns:
        if col not in df.columns:
            continue  # ignore silencieusement les colonnes absentes de cette table
        flag_col = f"{col}_ANOM"
        df[flag_col] = df[col] == anomaly_value          # True là où la valeur est le code sentinelle
        df[col] = df[col].replace({anomaly_value: np.nan})  # remplace le code par une vraie valeur manquante
    return df


def apply_document_grouping(df: pd.DataFrame, flag_cols: list) -> pd.DataFrame:
    """
    Regroupe plusieurs colonnes FLAG_DOCUMENT_* quasi-constantes en une seule
    colonne TOTAL_DOCUMENTS_PROVIDED (nombre de documents fournis), puis supprime
    les colonnes d'origine.
    """
    df = df.copy()
    df["TOTAL_DOCUMENTS_PROVIDED"] = df[flag_cols].sum(axis=1)  # somme ligne par ligne des flags (0/1)
    df = df.drop(columns=flag_cols)
    return df


def cap_column(df: pd.DataFrame, column: str, upper: float = None, lower: float = None,
               flag_suffix: str = "_CAPPED_FLAG") -> pd.DataFrame:
    """
    Plafonne (capping) une colonne numérique entre lower et upper (bornes optionnelles),
    et crée un flag indiquant les lignes qui ont été modifiées.
    """
    df = df.copy()

    flag = pd.Series(False, index=df.index)   # flag initial : personne n'est hors bornes
    if upper is not None:
        flag = flag | (df[column] > upper)
    if lower is not None:
        flag = flag | (df[column] < lower)
    df[f"{column}{flag_suffix}"] = flag        # on enregistre le flag AVANT de modifier la colonne

    df[column] = df[column].clip(lower=lower, upper=upper)  # applique le plafonnement
    return df


if __name__ == "__main__":
    # Fonction 1
    test_df = pd.DataFrame({"DAYS_EMPLOYED": [-500, 365243, -1200]})
    print(handle_days_anomaly(test_df, columns=["DAYS_EMPLOYED"]))
    # Attendu : DAYS_EMPLOYED_ANOM = [False, True, False], DAYS_EMPLOYED = [-500, NaN, -1200]

    # Fonction 2
    test_df2 = pd.DataFrame({"FLAG_DOCUMENT_2": [1, 0], "FLAG_DOCUMENT_4": [0, 0], "FLAG_DOCUMENT_5": [1, 1]})
    print(apply_document_grouping(test_df2, flag_cols=["FLAG_DOCUMENT_2", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5"]))
    # Attendu : TOTAL_DOCUMENTS_PROVIDED = [2, 1], plus de colonnes FLAG_DOCUMENT_*

    # Fonction 3
    test_df3 = pd.DataFrame({"AMT_INCOME_TOTAL": [50000, 999999999]})
    print(cap_column(test_df3, "AMT_INCOME_TOTAL", upper=500000))

    test_df4 = pd.DataFrame({"DEBT_TO_INCOME": [-0.5, 0.3, 50]})
    print(cap_column(test_df4, "DEBT_TO_INCOME", lower=0, upper=10))