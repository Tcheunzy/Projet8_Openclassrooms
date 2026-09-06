"""Ajoute l'origine de l'appel à la table des prédictions.

Migration ponctuelle. Les lignes antérieures sont reclassées une fois par
la latence — au-dessus de 150 ms, seule la production peut répondre — puis
la colonne fait foi pour tout ce qui suit.
"""
import os

import psycopg
from dotenv import load_dotenv

MIGRATION = """
ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS origine TEXT NOT NULL DEFAULT 'local';

CREATE INDEX IF NOT EXISTS idx_predictions_origine
    ON predictions (origine, created_at DESC);
"""

RECLASSEMENT = """
UPDATE predictions
SET origine = 'production'
WHERE latency_ms > 150 AND origine = 'local';
"""


def main() -> None:
    load_dotenv(".env")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute(MIGRATION)
        reclassees = conn.execute(RECLASSEMENT).rowcount
        conn.commit()
    print(f"Migration appliquée. {reclassees} lignes reclassées en production.")


if __name__ == "__main__":
    main()