"""Crée le schéma de la base. À lancer une fois."""
from dotenv import load_dotenv

from database.predictions import create_pool, init_schema


def main():
    load_dotenv()
    pool = create_pool()
    init_schema(pool)
    print("Schema cree.")
    pool.close()


if __name__ == "__main__":
    main()