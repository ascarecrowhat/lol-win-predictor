# Serving image for the public demo. Deliberately carries no match corpus: the
# 27 KB state snapshot plus the model is all that is needed, and each player's
# history is fetched from Riot on demand and cached in an ephemeral DuckDB file.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVE_DB_PATH=/tmp/cache.duckdb \
    PORT=8000

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY lolpred/ ./lolpred/
COPY web/ ./web/
COPY models/lolpred.joblib models/state.joblib ./models/

# RIOT_API_KEY must be supplied as a secret at run time, never baked in.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn lolpred.api.app:app --host 0.0.0.0 --port ${PORT}"]
