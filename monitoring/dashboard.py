"""Tableau de bord de supervision du modèle de scoring.

Lit la base des prédictions de production, affiche les indicateurs
d'exploitation et l'analyse de dérive. Ne charge ni le modèle ni le pipeline :
c'est un observateur, pas un acteur.
"""
import sys
from pathlib import Path

# Streamlit n'ajoute que le dossier du script au chemin d'import :
# on y ajoute la racine du projet pour retrouver database/ et monitoring/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta, timezone

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from database.predictions import create_pool, fetch_predictions
from monitoring.drift import (charger_reference, construire_rapport,
                              preparer_courant, resume_derive)

ROOT = Path(__file__).resolve().parent.parent

# En dessous de ce volume, les tests statistiques signalent des dérives
# inexistantes — vérifié par simulation lors de la conception.
SEUIL_ECHANTILLON = 500

# Paire validée pour la lisibilité, daltonismes protan et tritan compris.
BLEU, ORANGE, ROUGE = "#2563EB", "#D97706", "#B42318"

FENETRES = {
    "Dernières 24 heures": timedelta(hours=24),
    "7 derniers jours": timedelta(days=7),
    "30 derniers jours": timedelta(days=30),
    "Tout l'historique": None,
}

st.set_page_config(page_title="Supervision — scoring crédit",
                   page_icon="📊", layout="wide")


# --------------------------------------------------------------- données
@st.cache_resource
def obtenir_pool():
    """Ressource partagée : une seule connexion pour toutes les exécutions."""
    load_dotenv(ROOT / ".env")
    return create_pool()


@st.cache_data(ttl=300, show_spinner="Analyse en cours…")
def analyser(depuis):
    """Lecture en base et calcul de dérive, mis en cache 5 minutes.

    Les deux opérations sont regroupées pour n'avoir qu'une seule clé de
    cache — `depuis` — plutôt que d'avoir à hacher un DataFrame.
    """
    predictions = fetch_predictions(obtenir_pool(), since=depuis)
    reference = charger_reference(ROOT / "models" / "reference.parquet")
    courant = preparer_courant(predictions, list(reference.columns))

    if len(courant) < 2:
        return predictions, reference, courant, None, None, None

    instantane = construire_rapport(reference, courant)
    bilan, detail = resume_derive(instantane)
    return (predictions, reference, courant, bilan, detail,
            instantane.get_html_str(as_iframe=False))


def couleur_source():
    """Encodage couleur commun aux comparaisons référence / production."""
    return alt.Color(
        "source:N",
        scale=alt.Scale(domain=["Référence", "Production"],
                        range=[BLEU, ORANGE]),
        legend=alt.Legend(title=None, orient="top"),
    )

def courbe_temporelle(serie, titre_y, couleur=BLEU, format_y=None):
    """Courbe avec points visibles : reste lisible même sur un seul intervalle."""
    donnees = serie.reset_index()
    donnees.columns = ["date", "valeur"]
    return (
        alt.Chart(donnees)
        .mark_line(point=True, strokeWidth=2, color=couleur)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("valeur:Q", title=titre_y,
                    axis=alt.Axis(format=format_y) if format_y else alt.Axis()),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("valeur:Q")],
        )
    )

def densite_comparee(serie_ref, serie_cur, nom, bins=60):
    """Densités calculées en Python, sur des intervalles communs.

    Évite d'envoyer 11 000 valeurs au navigateur : 120 points suffisent
    à tracer deux courbes lissées.
    """
    valeurs = pd.concat([serie_ref, serie_cur]).dropna()
    bornes = np.linspace(valeurs.min(), valeurs.max(), bins + 1)
    centres = (bornes[:-1] + bornes[1:]) / 2

    morceaux = []
    for libelle, serie in (("Référence", serie_ref), ("Production", serie_cur)):
        densite, _ = np.histogram(serie.dropna(), bins=bornes, density=True)
        morceaux.append(pd.DataFrame({nom: centres, "densité": densite,
                                      "source": libelle}))
    return pd.concat(morceaux, ignore_index=True)
# --------------------------------------------------------------- filtres
st.sidebar.title("Filtres")
libelle = st.sidebar.selectbox("Période observée", list(FENETRES))
duree = FENETRES[libelle]
depuis = datetime.now(timezone.utc) - duree if duree else None

if st.sidebar.button("Actualiser les données"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Les données sont mises en cache 5 minutes.")


# --------------------------------------------------------------- en-tête
st.title("Supervision du modèle de scoring crédit")
st.caption("Prêt à dépenser — suivi des prédictions et détection de dérive")

predictions, reference, courant, bilan, detail, rapport_html = analyser(depuis)

if predictions.empty:
    st.info("Aucune prédiction enregistrée sur la période sélectionnée.")
    st.stop()

# Pas de temps adapté à la fenêtre : un graphique horaire sur 30 jours
# serait illisible.
pas = "1h" if (duree and duree <= timedelta(days=2)) else "1D"
horodate = predictions.set_index("created_at").sort_index()


# --------------------------------------------------------------- activité
st.header("Activité")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prédictions", f"{len(predictions):,}".replace(",", " "))
c2.metric("Taux de refus", f"{(predictions['decision'] == 'refusé').mean():.1%}")
c3.metric("Latence médiane", f"{predictions['latency_ms'].median():.0f} ms")
c4.metric("Clients connus", f"{predictions['history_found'].mean():.0%}")

g1, g2 = st.columns(2)

with g1:
    st.subheader("Volume dans le temps")
    st.bar_chart(horodate["id"].resample(pas).count().rename("prédictions"))

with g2:
    st.subheader("Taux de refus dans le temps")
    refus = (horodate.assign(refuse=lambda d: d["decision"] == "refusé")["refuse"]
             .resample(pas).mean())
    st.altair_chart(courbe_temporelle(refus, "taux de refus", format_y="%"),
                    use_container_width=True)
    st.caption("Un décrochage se voit ici avant que la dérive statistique "
               "ne devienne significative.")

g3, g4 = st.columns(2)

with g3:
    st.subheader("Distribution des probabilités de défaut")
    effectifs, bornes = np.histogram(predictions["probability"],
                                     bins=20, range=(0, 1))
    st.bar_chart(pd.DataFrame({"effectif": effectifs},
                              index=np.round(bornes[:-1], 2)))
    st.caption("Seuil de décision : 0,24 — au-delà, le dossier est refusé.")

with g4:
    st.subheader("Latence : médiane et 95ᵉ centile")
    latences = (horodate["latency_ms"].resample(pas)
                .agg(médiane="median", p95=lambda s: s.quantile(0.95))
                .reset_index()
                .melt("created_at", var_name="mesure", value_name="ms"))
    st.altair_chart(
        alt.Chart(latences).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("created_at:T", title=None),
            y=alt.Y("ms:Q", title="millisecondes"),
            color=alt.Color("mesure:N",
                            scale=alt.Scale(domain=["médiane", "p95"],
                                            range=[BLEU, ORANGE]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["created_at:T", "mesure:N", "ms:Q"],
        ),
        use_container_width=True,
    )
    st.caption("La médiane seule masque les cas lents : le 95ᵉ centile dit "
               "ce que vit l'utilisateur le plus mal servi.")


# --------------------------------------------------------------- dérive
st.header("Dérive des données")
st.markdown(
    "Comparaison entre les dossiers reçus en production et un échantillon "
    "des données d'entraînement. Une dérive signale que la population a "
    "évolué depuis l'apprentissage du modèle."
)

if bilan is None:
    st.info("Pas assez de données pour analyser la dérive.")
    st.stop()

if len(courant) < SEUIL_ECHANTILLON:
    st.warning(
        f"**{len(courant)} observations seulement.** En dessous d'environ "
        f"{SEUIL_ECHANTILLON}, les tests statistiques signalent des dérives "
        "qui n'existent pas. Interprétez ce bilan avec prudence."
    )
else:
    st.success(f"{len(courant)} observations : échantillon suffisant "
               "pour conclure.")

d1, d2 = st.columns([1, 3])

with d1:
    st.metric("Colonnes dérivées", f"{bilan['colonnes_derivees']} / {len(detail)}")
    st.metric("Proportion", f"{bilan['proportion']:.0%}")

with d2:
    barres = alt.Chart(detail).mark_bar(cornerRadius=3, color=BLEU).encode(
        y=alt.Y("colonne:N", sort="-x", title=None),
        x=alt.X("score:Q", title="score de dérive"),
        tooltip=["colonne", "methode", "score"],
    )
    seuil = (alt.Chart(pd.DataFrame({"x": [0.1]}))
             .mark_rule(color=ROUGE, strokeDash=[4, 4]).encode(x="x:Q"))
    st.altair_chart(barres + seuil, use_container_width=True)

st.dataframe(detail, use_container_width=True, hide_index=True)
st.caption(
    "La méthode dépend du type de variable : distance de Wasserstein pour les "
    "numériques, distance de Jensen-Shannon pour les catégorielles."
)


# ------------------------------------------------- comparaison détaillée
st.subheader("Comparaison des distributions")
st.caption("Le tableau dit *quelles* variables ont dérivé ; ce graphique dit "
           "*dans quel sens*.")

colonne = st.selectbox("Variable à inspecter", detail["colonne"].tolist())

comparaison = pd.concat([
    reference[[colonne]].assign(source="Référence"),
    courant[[colonne]].assign(source="Production"),
])

if pd.api.types.is_numeric_dtype(reference[colonne]):
    donnees = densite_comparee(reference[colonne], courant[colonne], colonne, bins=30)
    graphique = (
        alt.Chart(donnees)
        .mark_area(opacity=0.55, interpolate="monotone")
        .encode(x=alt.X(f"{colonne}:Q"),
                y=alt.Y("densité:Q", title="densité", stack=None),
                color=couleur_source(),
                tooltip=[colonne, "densité:Q", "source:N"])
    )
else:
    parts = (comparaison.groupby(["source", colonne]).size()
             .rename("effectif").reset_index())
    parts["part"] = (parts.groupby("source")["effectif"]
                     .transform(lambda s: s / s.sum()))
    graphique = (
        alt.Chart(parts).mark_bar()
        .encode(y=alt.Y(f"{colonne}:N", sort="-x", title=None),
                x=alt.X("part:Q", title="proportion",
                        axis=alt.Axis(format="%")),
                yOffset="source:N", color=couleur_source())
    )

st.altair_chart(graphique, use_container_width=True)


# --------------------------------------------------------- rapport brut
with st.expander("Rapport Evidently détaillé"):
    components.html(rapport_html, height=1000, scrolling=True)