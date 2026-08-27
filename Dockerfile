# Image de base légère : Python sans les outils de compilation inutiles
FROM python:3.12-slim

# uv est récupéré depuis son image officielle plutôt qu'installé via pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# LightGBM est une bibliothèque C++ : elle a besoin du runtime OpenMP,
# absent des images "slim".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Couche dépendances : ne se reconstruit que si pyproject/uv.lock changent ---
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Couche application : se reconstruit à chaque modification du code ---
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Les exécutables de l'environnement virtuel deviennent accessibles directement
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Exécution sous un utilisateur non privilégié
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Docker interroge l'endpoint /health pour savoir si le conteneur est réellement prêt
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]