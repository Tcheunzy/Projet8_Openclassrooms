"""Tests de l'interface Gradio.

`httpx.post` est remplacé par une fausse version : aucun appel réseau n'est
émis, et on peut inspecter le payload que l'interface aurait envoyé.
"""
import httpx
import pytest

from api.gradio_app import predire


class FakeResponse:
    """Réponse HTTP minimale, suffisante pour ce que `predire` en fait."""

    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def appeler_predire(**surcharges):
    """Appelle `predire` avec un dossier valide, modifiable champ par champ."""
    valeurs = {
        "sk_id_curr": 100002, "age": 44, "anciennete": 6,
        "genre": "F", "contrat": "Cash loans", "education": "Higher education",
        "possede_voiture": "Y", "possede_bien": "Y",
        "nb_enfants": 0, "nb_personnes": 2,
        "revenu": 150000, "montant_credit": 500000,
        "mensualite": 25000, "prix_bien": 450000,
        "ext_1": 0.5, "ext_2": 0.5, "ext_3": 0.5,
    }
    valeurs.update(surcharges)
    return predire(**valeurs)


def test_predire_convertit_les_annees_en_jours_negatifs(monkeypatch):
    """44 ans doivent devenir -16071 jours : une erreur ici fausserait
    silencieusement toutes les prédictions de l'interface."""
    envoye = {}

    def faux_post(url, json, timeout):
        envoye.update(json)
        return FakeResponse(200, {
            "sk_id_curr": 100002, "probability": 0.08, "threshold": 0.24,
            "decision": "accordé", "mlflow_model_version": "4",
        })

    monkeypatch.setattr("api.gradio_app.httpx.post", faux_post)

    appeler_predire(age=44, anciennete=6)

    assert envoye["DAYS_BIRTH"] == -16071
    assert envoye["DAYS_EMPLOYED"] == -2191


def test_predire_omet_les_scores_externes_non_renseignes(monkeypatch):
    """Un champ vidé dans l'interface ne doit pas être envoyé à l'API."""
    envoye = {}

    def faux_post(url, json, timeout):
        envoye.update(json)
        return FakeResponse(200, {
            "sk_id_curr": 1, "probability": 0.5, "threshold": 0.24,
            "decision": "refusé", "mlflow_model_version": "4",
        })

    monkeypatch.setattr("api.gradio_app.httpx.post", faux_post)

    appeler_predire(ext_1=0.7, ext_2=None, ext_3=None)

    assert envoye["EXT_SOURCE_1"] == 0.7
    assert "EXT_SOURCE_2" not in envoye
    assert "EXT_SOURCE_3" not in envoye


def test_predire_met_en_forme_la_decision(monkeypatch):
    """Le résultat affiché doit reprendre la décision et la probabilité."""
    def faux_post(url, json, timeout):
        return FakeResponse(200, {
            "sk_id_curr": 100002, "probability": 0.6033, "threshold": 0.24,
            "decision": "refusé", "mlflow_model_version": "4",
        })

    monkeypatch.setattr("api.gradio_app.httpx.post", faux_post)

    resultat = appeler_predire()

    assert "refusé" in resultat
    assert "60.3%" in resultat
    assert "100002" in resultat


def test_predire_affiche_le_champ_fautif_sur_un_422(monkeypatch):
    """Une erreur de validation doit nommer le champ en cause, pas rester opaque."""
    def faux_post(url, json, timeout):
        return FakeResponse(422, {
            "detail": [{
                "loc": ["body", "AMT_CREDIT"],
                "msg": "Input should be greater than 0",
            }]
        })

    monkeypatch.setattr("api.gradio_app.httpx.post", faux_post)

    resultat = appeler_predire(montant_credit=-1000)

    assert "Saisie invalide" in resultat
    assert "AMT_CREDIT" in resultat
    assert "greater than 0" in resultat


def test_predire_signale_une_api_injoignable(monkeypatch):
    """Si l'API ne répond pas, l'utilisateur doit le savoir explicitement."""
    def faux_post(url, json, timeout):
        raise httpx.RequestError("connexion refusée")

    monkeypatch.setattr("api.gradio_app.httpx.post", faux_post)

    resultat = appeler_predire()

    assert "Erreur de connexion" in resultat