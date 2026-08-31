"""Accès à la base des prédictions de production."""
import os

import pandas as pd
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    sk_id_curr      BIGINT           NOT NULL,
    model_version   TEXT             NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    probability     DOUBLE PRECISION NOT NULL,
    decision        TEXT             NOT NULL,
    history_found   BOOLEAN          NOT NULL,
    latency_ms      INTEGER,
    features        JSONB            NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_sk_id_curr
    ON predictions (sk_id_curr);
"""


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


def log_prediction(pool: ConnectionPool, *, sk_id_curr, model_version, threshold,
                   probability, decision, history_found, latency_ms, features) -> None:
    """Enregistre une prédiction. Les arguments sont nommés obligatoirement."""
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (sk_id_curr, model_version, threshold, probability,
                                     decision, history_found, latency_ms, features)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (sk_id_curr, model_version, threshold, probability, decision,
             history_found, latency_ms, Jsonb(features)),
        )


def fetch_predictions(pool: ConnectionPool, since=None, limit=None) -> pd.DataFrame:
    """Relit les prédictions, pour le tableau de bord et l'analyse de dérive."""
    sql = "SELECT * FROM predictions"
    params = []

    if since is not None:
        sql += " WHERE created_at >= %s"
        params.append(since)

    sql += " ORDER BY created_at DESC"

    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return pd.DataFrame(rows)