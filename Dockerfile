# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Production image for the Brain Disease Detection web service.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Fail fast, no .pyc files, unbuffered logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    PORT=8000

WORKDIR /app

# System libraries required by Pillow / TensorFlow at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY src ./src
COPY configs ./configs
COPY wsgi.py ./

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# 2 workers is a sane default for a CPU-bound inference service; tune per host.
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
