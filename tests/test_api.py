"""Tests fonctionnels des endpoints FastAPI.

Le client de test charge le modèle et les artefacts via le lifespan ;
aucun serveur MLflow ni fichier de data/ n'est nécessaire.
"""


def test_root_renvoie_200(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_health_confirme_le_chargement_du_modele(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_renvoie_une_prediction_valide(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["sk_id_curr"] == valid_payload["SK_ID_CURR"]
    assert 0 <= body["probability"] <= 1
    assert body["decision"] in {"accordé", "refusé"}
    assert body["threshold"] == 0.24


def test_predict_accepte_un_client_inconnu(client, valid_payload):
    """Un client sans historique est un cas métier légitime, pas une erreur."""
    valid_payload["SK_ID_CURR"] = 999999999

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1


def test_predict_rejette_une_valeur_invalide(client, valid_payload):
    """Un montant négatif doit produire un 422, pas une prédiction."""
    valid_payload["AMT_CREDIT"] = -1000

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422

def test_predict_renvoie_500_si_le_pipeline_echoue(client, valid_payload, monkeypatch):
    """Une panne interne doit produire un 500 explicite, pas un plantage du serveur."""
    def pipeline_en_panne(*args, **kwargs):
        raise ValueError("panne simulée")

    monkeypatch.setattr("api.main.clean_application", pipeline_en_panne)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500
    assert "panne simulée" in response.json()["detail"]

def test_interface_gradio_est_montee(client):
    """L'interface doit rester accessible après toute évolution de l'API."""
    response = client.get("/gradio")

    assert response.status_code == 200

def test_predict_fonctionne_sans_base(client, valid_payload):
    """L'API doit predire normalement quand aucune base n'est configuree."""
    from api import main

    assert main.ml["pool"] is None

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200


def test_journalisation_absorbe_une_panne_de_base(monkeypatch):
    """Une base injoignable ne doit jamais faire remonter d'exception."""
    from api import main

    def echec(*args, **kwargs):
        raise RuntimeError("base injoignable")

    monkeypatch.setitem(main.ml, "pool", object())   # un pool factice, non nul
    monkeypatch.setattr(main, "log_prediction", echec)

    main._journaliser(sk_id_curr=1, model_version="4", threshold=0.24,
                      probability=0.5, decision="refusé", history_found=False,
                      latency_ms=10, features={})