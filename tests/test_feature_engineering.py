"""Tests unitaires du feature engineering."""
import pandas as pd

from src.feature_engineering import aggregate_payment_behavior


def test_aggregate_payment_behavior_detecte_les_retards():
    """Un paiement postérieur à l'échéance doit être compté comme retard."""
    installments = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "AMT_INSTALMENT": [100, 100],
        "AMT_PAYMENT": [100, 80],
        "DAYS_INSTALMENT": [-30, -60],
        "DAYS_ENTRY_PAYMENT": [-35, -55],
    })

    result = aggregate_payment_behavior(installments).set_index("SK_ID_CURR")

    # La 2e échéance est payée 5 jours après la date due
    assert result.loc[1, "LATE_PAYMENT_RATE"] == 0.5
    assert result.loc[1, "LATE_PAYMENT_COUNT"] == 1
    assert result.loc[1, "DAYS_PAST_DUE_max"] == 5

    # 20 de sous-paiement au total
    assert result.loc[1, "PAYMENT_DIFF_sum"] == 20