"""Analyse de dérive entre le jeu de référence et les données de production."""
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

# L'identifiant client n'est pas une variable explicative : il dérive par
# construction (les identifiants augmentent) et polluerait l'analyse.
COLONNES_EXCLUES = ["SK_ID_CURR"]


def charger_reference(chemin: Path) -> pd.DataFrame:
    return pd.read_parquet(chemin)


def preparer_courant(predictions: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    """Déplie la colonne JSONB `features` en colonnes, alignées sur la référence."""
    if predictions.empty:
        return pd.DataFrame(columns=colonnes)
    depliees = pd.json_normalize(predictions["features"])
    return depliees.reindex(columns=colonnes)


def types_de_colonnes(reference: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Sépare numériques et catégorielles d'après la référence, qui fait foi."""
    colonnes = [c for c in reference.columns if c not in COLONNES_EXCLUES]
    numeriques = [c for c in colonnes if pd.api.types.is_numeric_dtype(reference[c])]
    categorielles = [c for c in colonnes if c not in numeriques]
    return numeriques, categorielles


def construire_rapport(reference: pd.DataFrame, courant: pd.DataFrame):
    numeriques, categorielles = types_de_colonnes(reference)
    definition = DataDefinition(numerical_columns=numeriques,
                                categorical_columns=categorielles)

    jeu_reference = Dataset.from_pandas(reference[numeriques + categorielles],
                                        data_definition=definition)
    jeu_courant = Dataset.from_pandas(courant[numeriques + categorielles],
                                      data_definition=definition)

    rapport = Report([DataDriftPreset()])
    return rapport.run(reference_data=jeu_reference, current_data=jeu_courant)


def resume_derive(instantane) -> tuple[dict, pd.DataFrame]:
    """Extrait le bilan global et le détail colonne par colonne."""
    metriques = instantane.dict()["metrics"]
    global_, lignes = {}, []

    for m in metriques:
        config = m.get("config", {})
        if config.get("type", "").endswith("DriftedColumnsCount"):
            global_ = {"colonnes_derivees": int(m["value"]["count"]),
                       "proportion": float(m["value"]["share"])}
        elif config.get("type", "").endswith("ValueDrift"):
            score = float(m["value"])
            seuil = float(config["threshold"])
            lignes.append({"colonne": config["column"],
                           "methode": config["method"],
                           "score": round(score, 4),
                           "seuil": seuil,
                           "derive": score > seuil})

    detail = pd.DataFrame(lignes).sort_values("score", ascending=False)
    return global_, detail