"""Tests de validation des schémas Pydantic (aucune dépendance externe)."""
import pytest
from pydantic import ValidationError

from api.schemas import ClientPredictionInput


def test_champ_obligatoire_manquant_est_rejete(valid_payload):
    """Un champ obligatoire absent doit lever une erreur de validation."""
    del valid_payload["SK_ID_CURR"]

    with pytest.raises(ValidationError):
        ClientPredictionInput(**valid_payload)


def test_valeur_hors_bornes_est_rejetee(valid_payload):
    """Un montant de crédit négatif viole la contrainte gt=0."""
    valid_payload["AMT_CREDIT"] = -1000

    with pytest.raises(ValidationError):
        ClientPredictionInput(**valid_payload)


def test_mauvais_type_est_rejete(valid_payload):
    """Une chaîne non numérique dans un champ float doit être rejetée."""
    valid_payload["AMT_CREDIT"] = "beaucoup"

    with pytest.raises(ValidationError):
        ClientPredictionInput(**valid_payload)


def test_to_dataframe_produit_une_ligne_a_plat(valid_payload):
    """Les extra_fields doivent être fusionnés au même niveau que les autres colonnes."""
    valid_payload["extra_fields"] = {"AMT_REQ_CREDIT_BUREAU_YEAR": 2}
    payload = ClientPredictionInput(**valid_payload)

    df = payload.to_dataframe()

    assert df.shape[0] == 1
    assert "AMT_REQ_CREDIT_BUREAU_YEAR" in df.columns
    assert "extra_fields" not in df.columns
    assert df["AMT_CREDIT"].iloc[0] == 500000