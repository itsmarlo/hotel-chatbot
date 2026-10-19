FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOTEL_API_HOST=0.0.0.0 \
    HOTEL_UI_PORT=7860 \
    HOTEL_RAG_MODE=semantic \
    HOTEL_RAG_LOCAL_ONLY=1 \
    HF_HOME=/app/.cache/huggingface \
    HOTEL_DATABASE_PATH=/app/data/synthetic/hotel.db

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY app.py ./app.py
COPY configs ./configs
COPY data ./data
COPY models/trained ./models/trained
COPY scripts/train_models.py ./scripts/train_models.py

RUN python -m pip install --no-cache-dir '.[rag]'
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', revision='e8f8c211226b894fcb81acc59f3b34ba3efd5f42')"
RUN python scripts/train_models.py
RUN useradd --system --uid 10001 hotel \
    && mkdir -p /app/data/synthetic \
    && chown -R hotel:hotel /app

USER hotel

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=5)" || exit 1

CMD ["python", "-m", "app.gradio_app"]
