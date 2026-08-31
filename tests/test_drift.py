"""Tests de l'analyse de dérive. Ni base ni API requises."""
import numpy as np
import pandas as pd

from monitoring.drift import (construire_rapport, preparer_courant,
                              resume_derive, types_de_colonnes)


def reference_factice(n=2000, decalage=0.0):
    rng = np.random.default_rng(0 if decalage == 0 else 1)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(100002, 100002 + n),
        "AMT_CREDIT": rng.normal(600000 + decalage, 150000, n),
        "DAYS_BIRTH": rng.normal(-16000, 4300, n).astype(int),
        "CODE_GENDER": rng.choice(["F", "M"], n),
    })


def test_preparer_courant_sur_une_base_vide():
    """Sans prédiction, on renvoie un cadre vide mais correctement colonné."""
    colonnes = ["SK_ID_CURR", "AMT_CREDIT", "CODE_GENDER"]

    resultat = preparer_courant(pd.DataFrame(), colonnes)

    assert resultat.empty
    assert list(resultat.columns) == colonnes


def test_preparer_courant_deplie_le_jsonb_et_realigne():
    """La colonne features contient des dictionnaires : elle doit devenir
    des colonnes, alignées sur celles de la référence."""
    predictions = pd.DataFrame({"features": [
        {"SK_ID_CURR": 1, "AMT_CREDIT": 500000, "CHAMP_INCONNU": 42},
        {"SK_ID_CURR": 2, "AMT_CREDIT": 700000, "CHAMP_INCONNU": 43},
    ]})
    colonnes = ["SK_ID_CURR", "AMT_CREDIT", "CODE_GENDER"]

    resultat = preparer_courant(predictions, colonnes)

    assert list(resultat.columns) == colonnes
    assert resultat["AMT_CREDIT"].tolist() == [500000, 700000]
    assert resultat["CODE_GENDER"].isna().all()      # absente -> NaN
    assert "CHAMP_INCONNU" not in resultat.columns   # en trop -> écartée


def test_types_de_colonnes_exclut_l_identifiant():
    """SK_ID_CURR dérive par construction : il ne doit pas être analysé."""
    numeriques, categorielles = types_de_colonnes(reference_factice(50))

    assert "SK_ID_CURR" not in numeriques + categorielles
    assert set(numeriques) == {"AMT_CREDIT", "DAYS_BIRTH"}
    assert categorielles == ["CODE_GENDER"]


def test_resume_derive_detecte_un_decalage_reel():
    """Un décalage franc sur AMT_CREDIT doit être signalé, les autres non."""
    reference = reference_factice()
    courant = reference_factice(decalage=400000)

    bilan, detail = resume_derive(construire_rapport(reference, courant))

    assert set(detail.columns) == {"colonne", "methode", "score", "seuil", "derive"}
    assert bilan["colonnes_derivees"] >= 1

    scores = detail.set_index("colonne")["score"]
    assert scores["AMT_CREDIT"] > 0.1
    assert scores["CODE_GENDER"] < 0.1


def test_resume_derive_ne_signale_rien_sur_deux_jeux_identiques():
    """Sans décalage, aucune colonne ne doit être signalée."""
    reference = reference_factice()

    bilan, detail = resume_derive(construire_rapport(reference, reference.copy()))

    assert bilan["colonnes_derivees"] == 0
    assert not detail["derive"].any()