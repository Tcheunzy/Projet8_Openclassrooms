"""Compare les latences de production avant et après l'optimisation du pipeline."""
import os

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

# Mise en production du pipeline optimisé, en UTC : Render affiche ses
# horodatages dans le fuseau du navigateur, la base stocke en UTC.
BASCULE = "2026-08-31 13:31:00+00"

# La table ne conserve pas l'origine de l'appel. En local le pipeline répond
# en une trentaine de millisecondes, sur Render en plusieurs centaines : le
# seuil sépare donc proprement les deux populations, sans recouvrement.


REQUETE = """
SELECT
  CASE WHEN created_at < %(bascule)s THEN 'avant' ELSE 'apres' END AS version,
  COUNT(*) AS n,
  ROUND(MIN(latency_ms)::numeric, 0) AS min_ms,
  ROUND(PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY latency_ms)::numeric, 0)
        AS mediane_ms,
  ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric, 0)
        AS p95_ms
FROM predictions
WHERE origine = 'production'
GROUP BY 1
ORDER BY 1 DESC;
"""


def main() -> None:
    load_dotenv(".env")
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as conn:
        lignes = conn.execute(REQUETE, {"bascule": BASCULE}).fetchall()

    print("Appels de production\n")
    print(f"{'version':<8} {'n':>6} {'min':>9} {'médiane':>10} {'p95':>10}")
    for ligne in lignes:
        print(f"{ligne['version']:<8} {ligne['n']:>6} "
              f"{ligne['min_ms']:>7} ms {ligne['mediane_ms']:>8} ms "
              f"{ligne['p95_ms']:>8} ms")


if __name__ == "__main__":
    main()