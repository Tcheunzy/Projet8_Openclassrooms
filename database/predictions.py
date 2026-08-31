"""Accès à la base des prédictions de production."""
import os

import pandas as pd
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

# Une ligne représente un appel à /predict, réussi ou non. Les colonnes du
# résultat sont donc facultatives : un appel rejeté n'a ni probabilité, ni
# décision, ni parfois même d'identifiant client exploitable.
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    sk_id_curr      BIGINT,
    model_version   TEXT             NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    probability     DOUBLE PRECISION,
    decision        TEXT,
    history_found   BOOLEAN,
    latency_ms      INTEGER,
    status          TEXT             NOT NULL DEFAULT 'succes',
    error_type      TEXT,
    features        JSONB            NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_sk_id_curr
    ON predictions (sk_id_curr);
CREATE INDEX IF NOT EXISTS idx_predictions_status
    ON predictions (created_at DESC, status);
"""

STATUT_SUCCES = "succes"
STATUT_VALIDATION = "validation"
STATUT_ERREUR = "erreur"


def create_pool(min_size: int = 1, max_size: int = 4) -> ConnectionPool:
    """Ouvre un réservoir de connexions réutilisables."""
    return ConnectionPool(
        os.environ["DATABASE_URL"],
        min_size=min_size,
        max_size=max_size,
        open=True,
        # Vérifie qu'une connexion est vivante avant de la prêter. Sans cela,
        # une connexion restée inactive et coupée par le réseau provoque une
        # OperationalError au premier usage.
        check=ConnectionPool.check_connection,
        # Ferme les connexions inactives au-delà de 5 minutes plutôt que de
        # les laisser vieillir.
        max_idle=300,
    )


def init_schema(pool: ConnectionPool) -> None:
    """Crée la table et ses index. Sans effet si elle existe déjà."""
    with pool.connection() as conn:
        conn.execute(SCHEMA)


def log_prediction(pool: ConnectionPool, *, model_version, threshold, features,
                   sk_id_curr=None, probability=None, decision=None,
                   history_found=None, latency_ms=None,
                   status=STATUT_SUCCES, error_type=None) -> None:
    """Enregistre un appel à /predict, réussi ou non.

    Les arguments sont nommés obligatoirement. Seuls la version du modèle, le
    seuil et les données reçues sont exigés : tout le reste dépend de l'issue
    de l'appel.
    """
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (sk_id_curr, model_version, threshold, probability,
                                     decision, history_found, latency_ms,
                                     status, error_type, features)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (sk_id_curr, model_version, threshold, probability, decision,
             history_found, latency_ms, status, error_type, Jsonb(features)),
        )


def fetch_predictions(pool: ConnectionPool, since=None, limit=None,
                      status=None) -> pd.DataFrame:
    """Relit les appels, pour le tableau de bord et l'analyse de dérive.

    `status` filtre sur l'issue. L'analyse de dérive doit passer
    `status=STATUT_SUCCES` : les données d'un appel refusé sont par définition
    invalides, les inclure dans une comparaison de distributions reviendrait à
    mesurer la dérive sur du bruit.
    """
    sql = "SELECT * FROM predictions"
    conditions = []
    params = []

    if since is not None:
        conditions.append("created_at >= %s")
        params.append(since)

    if status is not None:
        conditions.append("status = %s")
        params.append(status)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY created_at DESC"

    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return pd.DataFrame(rows)