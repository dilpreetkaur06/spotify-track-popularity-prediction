# =============================================================================
# Dockerfile - Spotify Track Popularity Prediction
# Multi-stage build: keeps the final image lean by not shipping build tools.
# =============================================================================

FROM python:3.10-slim AS base

# Prevents Python from writing .pyc files & buffers stdout (better logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies required by lightgbm / catboost / xgboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first (leverages Docker layer caching)
COPY requirements.txt setup.py README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure runtime directories exist even before the pipeline has been run
RUN mkdir -p artifacts logs mlruns

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:5000/ || exit 1

# Gunicorn for a production-grade WSGI server
RUN pip install gunicorn==22.0.0

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
