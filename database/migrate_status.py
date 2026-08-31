"""Ajoute le suivi des erreurs à la table des prédictions.

Migration ponctuelle : à exécuter une fois sur une base existante. Le schéma
complet de `predictions.py` intègre déjà ces colonnes pour toute base créée
à partir de maintenant.
"""
import os

import psycopg
from dotenv import load_dotenv

MIGRATION = """
ALTER TABLE predictions
    ALTER COLUMN sk_id_curr    DROP NOT NULL,
    ALTER COLUMN probability   DROP NOT NULL,
    ALTER COLUMN decision      DROP NOT NULL,
    ALTER COLUMN history_found DROP NOT NULL,
    ADD COLUMN IF NOT EXISTS status     TEXT NOT NULL DEFAULT 'succes',
    ADD COLUMN IF NOT EXISTS error_type TEXT;

CREATE INDEX IF NOT EXISTS idx_predictions_status
    ON predictions (created_at DESC, status);
"""

def main() -> None:
    load_dotenv(".env")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute(MIGRATION)
        conn.commit()
    print("Migration appliquée.")


if __name__ == "__main__":
    main()