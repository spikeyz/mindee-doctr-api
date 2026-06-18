# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 – base with system deps
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCTR_CACHE_DIR=/app/.cache/doctr \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        poppler-utils \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 – install Python dependencies
# ──────────────────────────────────────────────────────────────────────────────
FROM base AS deps

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 – pre-download default docTR models so the first request is fast
# ──────────────────────────────────────────────────────────────────────────────
FROM deps AS model-download

# Download default detection + recognition weights at build time.
RUN python - <<'EOF'
from doctr.models import ocr_predictor
print("Downloading default models (db_resnet50 + crnn_vgg16_bn) …")
ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
print("Done.")
EOF

# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 – final runtime image
# ──────────────────────────────────────────────────────────────────────────────
FROM model-download AS runtime

WORKDIR /app

COPY app/ ./app/

EXPOSE 8000

# Number of uvicorn workers can be tuned via WORKERS env var.
ENV WORKERS=1

CMD uvicorn app.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers ${WORKERS} \
        --log-level info
