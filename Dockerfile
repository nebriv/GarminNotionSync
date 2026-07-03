FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GARMIN_TOKEN_DIR=/data/garmin-tokens \
    DATA_DIR=/data \
    METRICS_PORT=9100

# ca-certificates for TLS to Garmin/Notion; tzdata so TZ env works for logs/scheduling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY sync.py notify.py runner.py healthcheck.py ./

# Run as a non-root user; /data is the mounted volume for tokens + state.
RUN useradd --uid 1000 --create-home appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app
USER appuser

VOLUME ["/data"]
EXPOSE 9100

HEALTHCHECK --interval=5m --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

CMD ["python", "runner.py"]
