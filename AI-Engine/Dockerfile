FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=10
ENV HF_HOME=/models/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY AI-Engine/requirements /tmp/ai-requirements.txt
COPY Recommendation-system/requirements.txt /tmp/recommendation-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r /tmp/ai-requirements.txt -r /tmp/recommendation-requirements.txt

COPY AI-Engine ./AI-Engine
COPY Recommendation-system ./Recommendation-system

WORKDIR /app/AI-Engine

ENV OLLAMA_BASE_URL=http://ollama:11434
ENV RAG_DB_DIR=/app/AI-Engine/db_vigogne_bge_m3
ENV CHROMA_DIR=/app/Recommendation-system/RecommendationSystem/chroma_store
ENV EMBEDDING_CACHE_DIR=/models/huggingface

EXPOSE 8000

CMD ["uvicorn", "service_api:app", "--host", "0.0.0.0", "--port", "8000"]
