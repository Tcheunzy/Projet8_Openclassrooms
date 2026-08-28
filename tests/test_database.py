"""Tests de la couche d'acces, sans base reelle."""
from database.predictions import log_prediction


class FausseConnexion:
    """Enregistre les requetes au lieu de les executer."""

    def __init__(self, journal):
        self.journal = journal

    def execute(self, sql, params=None):
        self.journal.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FauxPool:
    def __init__(self):
        self.journal = []

    def connection(self):
        return FausseConnexion(self.journal)


def test_log_prediction_transmet_les_parametres_dans_le_bon_ordre():
    pool = FauxPool()

    log_prediction(pool, sk_id_curr=100002, model_version="4", threshold=0.24,
                   probability=0.61, decision="refusé", history_found=True,
                   latency_ms=32, features={"AMT_CREDIT": 500000})

    assert len(pool.journal) == 1
    sql, params = pool.journal[0]

    assert "INSERT INTO predictions" in sql
    assert params[0] == 100002        # sk_id_curr
    assert params[1] == "4"           # model_version
    assert params[2] == 0.24          # threshold
    assert params[3] == 0.61          # probability
    assert params[4] == "refusé"      # decision
    assert params[5] is True          # history_found
    assert params[6] == 32            # latency_ms