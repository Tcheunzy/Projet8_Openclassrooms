import numpy as np
import pandas as pd

from src.cleaning import cap_column


def replace_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs infinies (issues de divisions par 0) par NaN, afin que
    l'imputation du préprocesseur puisse les traiter. Sur app_train ce remplacement
    est un no-op (0 valeur infinie constatée), mais en production un client avec
    AMT_GOODS_PRICE = 0 ou AMT_ANNUITY = 0 produirait un inf qui ferait planter
    l'imputer.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def add_application_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features de ratio calculables à partir de la seule table application.
    Aucune dépendance aux tables annexes : peut donc être appelée avant les merges.
    """
    df = df.copy()

    # Nombre de mensualités nécessaires pour rembourser le crédit
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    # Âge approximatif du client à la fin du prêt (CREDIT_TERM converti en années)
    df['AGE_AT_LOAN_END'] = (df['DAYS_BIRTH'] / -365) + (df['CREDIT_TERM'] / 12)

    # Poids de la mensualité dans le revenu
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # Montant emprunté rapporté au revenu
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # Revenu du foyer réparti par membre
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # Part du bien financée par le crédit (proche de 1 = pas d'apport personnel)
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

    # Proportion de la vie du client passée dans son emploi actuel
    df['EMPLOYED_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # Taux de remboursement (inverse de CREDIT_TERM)
    # /!\ Dans le notebook d'origine, cette feature n'était calculée que sur app_train.
    #     Ici elle est calculée pour tout DataFrame passé en entrée.
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    return df


def add_debt_to_income(df: pd.DataFrame, cap_debt: float = None) -> pd.DataFrame:
    """
    Ajoute DEBT_TO_INCOME (dette totale déclarée au bureau de crédit / revenu).
    Dépend de AMT_CREDIT_SUM_DEBT_sum : doit être appelée APRÈS le merge avec
    l'agrégation bureau.
    Si cap_debt est fourni, applique le plafonnement figé à l'entraînement.
    """
    df = df.copy()
    df['DEBT_TO_INCOME'] = df['AMT_CREDIT_SUM_DEBT_sum'] / df['AMT_INCOME_TOTAL']

    if cap_debt is not None:
        df = cap_column(df, 'DEBT_TO_INCOME', upper=cap_debt, lower=0)

    return df


def aggregate_payment_behavior(installments_payments: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le comportement de paiement du client à partir de l'historique
    d'échéances (installments_payments), puis l'agrège au niveau SK_ID_CURR.
    Retourne une table à fusionner sur SK_ID_CURR.
    """
    ip = installments_payments.copy()

    # Écart entre ce qui était dû et ce qui a été payé (positif = sous-paiement)
    ip['PAYMENT_DIFF'] = ip['AMT_INSTALMENT'] - ip['AMT_PAYMENT']

    # Retard en jours (positif = paiement après l'échéance)
    ip['DAYS_PAST_DUE'] = ip['DAYS_ENTRY_PAYMENT'] - ip['DAYS_INSTALMENT']

    ip['LATE_PAYMENT_FLAG'] = (ip['DAYS_PAST_DUE'] > 0).astype(int)

    payment_behavior_agg = ip.groupby('SK_ID_CURR').agg(
        PAYMENT_DIFF_mean=('PAYMENT_DIFF', 'mean'),
        PAYMENT_DIFF_sum=('PAYMENT_DIFF', 'sum'),
        DAYS_PAST_DUE_mean=('DAYS_PAST_DUE', 'mean'),
        DAYS_PAST_DUE_max=('DAYS_PAST_DUE', 'max'),
        LATE_PAYMENT_RATE=('LATE_PAYMENT_FLAG', 'mean'),
        LATE_PAYMENT_COUNT=('LATE_PAYMENT_FLAG', 'sum'),
    ).reset_index()

    return payment_behavior_agg

def add_ext_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Résume les trois scores externes (EXT_SOURCE_1/2/3) en trois features :
    - le score minimum et maximum parmi ceux disponibles (min/max ignorent les NaN)
    - le nombre de scores effectivement renseignés (0 à 3), qui est en soi un signal :
      un client sans aucun score externe n'a pas le même profil de risque qu'un
      client noté par trois organismes.
    """
    df = df.copy()
    liste_ext_source = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    df['EXT_SOURCE_combined_min'] = df[liste_ext_source].min(axis=1)
    df['EXT_SOURCE_combined_max'] = df[liste_ext_source].max(axis=1)
    df['EXT_SOURCE_combined_count_available'] = (~df[liste_ext_source].isnull()).sum(axis=1)

    return df

if __name__ == "__main__":
    
    test_df = pd.DataFrame({
        "AMT_CREDIT": [100000, 200000],
        "AMT_ANNUITY": [10000, 0],
        "AMT_INCOME_TOTAL": [50000, 100000],
        "AMT_GOODS_PRICE": [90000, 200000],
        "DAYS_BIRTH": [-10950, -14600],
        "DAYS_EMPLOYED": [-1000, -2000],
        "CNT_FAM_MEMBERS": [2, 4],
    })
    result = add_application_ratios(test_df)
    print(result[['CREDIT_TERM', 'PAYMENT_RATE', 'CREDIT_GOODS_RATIO', 'INCOME_PER_PERSON']])
    result = replace_infinities(result)
    print("\nAprès replace_infinities, CREDIT_TERM :", result['CREDIT_TERM'].tolist())