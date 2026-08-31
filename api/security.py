"""Authentification par clé d'API.

Il ne s'agit pas d'authentifier un utilisateur — pas de comptes, pas de mots
de passe — mais un système appelant. C'est le mécanisme adapté à une API
interne consommée par une application métier.
"""
import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

NOM_ENTETE = "X-API-Key"

# auto_error=False laisse la dépendance décider. Avec le comportement par
# défaut, FastAPI renverrait 403 dès que l'en-tête manque, sans nous laisser
# le cas « aucune clé configurée, service ouvert ».
entete_cle = APIKeyHeader(name=NOM_ENTETE, auto_error=False)


def cle_attendue() -> str | None:
    """Clé de référence, ou None si l'API doit rester ouverte.

    Lue à chaque appel et non au chargement du module : une valeur figée à
    l'import serait intestable, et empêcherait de changer la clé sans
    redémarrer le service.
    """
    return os.getenv("API_KEY") or None


def verifier_cle(cle: str | None = Security(entete_cle)) -> None:
    """Refuse la requête si la clé est absente ou incorrecte."""
    attendue = cle_attendue()

    # Aucune clé configurée : l'API reste ouverte. Même principe que pour la
    # base de données — l'absence de configuration ne doit pas empêcher le
    # service de fonctionner, ni la CI de valider l'image sans secret.
    if attendue is None:
        return

    # compare_digest plutôt que == : une comparaison classique s'interrompt au
    # premier caractère différent, et son temps d'exécution révèle alors
    # combien de caractères sont corrects. Comparer des octets évite en outre
    # l'erreur que lève compare_digest sur une chaîne non ASCII.
    if cle is None or not secrets.compare_digest(cle.encode(), attendue.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Clé d'API absente ou invalide. En-tête attendu : {NOM_ENTETE}.",
        )