"""Fixtures partagées par tous les tests."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Client de test FastAPI. Le `with` déclenche le lifespan (chargement du modèle)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Requête valide minimale, réutilisée comme base dans plusieurs tests."""
    return {
        "SK_ID_CURR": 100002,
        "AMT_INCOME_TOTAL": 150000,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 450000,
        "DAYS_BIRTH": -16000,
        "DAYS_EMPLOYED": -2000,
        "CNT_FAM_MEMBERS": 2,
        "CNT_CHILDREN": 0,
        "CODE_GENDER": "F",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "NAME_EDUCATION_TYPE": "Higher education",
    }
