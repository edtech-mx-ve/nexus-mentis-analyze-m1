FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    ARGOS_DEVICE_TYPE=cpu \
    NEXUS_ARTIFACT_DIR=/app/artifacts/sentiment140_real_200k \
    NEXUS_TRANSLATION_DIR=/app/artifacts/translation_models \
    ARGOS_PACKAGES_DIR=/app/artifacts/translation_models

WORKDIR /app

RUN apt-get update && apt-get install --no-install-recommends -y libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cloud.txt /app/requirements-cloud.txt
RUN python -m pip install --no-cache-dir -r requirements-cloud.txt

COPY --chown=10001:10001 nexus_mentis/ /app/nexus_mentis/
COPY --chown=10001:10001 artifacts/sentiment140_real_200k/ /app/artifacts/sentiment140_real_200k/
COPY --chown=10001:10001 cloud_entry.py /app/cloud_entry.py
COPY --chown=10001:10001 cloud_preflight.py /app/cloud_preflight.py

RUN useradd -m -u 10001 -s /bin/sh nexus \
    && mkdir -p /app/artifacts/translation_models \
    && chown -R nexus:nexus /app/artifacts

USER nexus

# Se instala ES->EN UNA VEZ durante build. Nunca en una peticion del usuario.
# El build falla si Argos no puede instalarse o verificarse.
RUN python -m nexus_mentis.cli setup-translation \
    && python cloud_preflight.py

EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn cloud_entry:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --no-access-log"]
