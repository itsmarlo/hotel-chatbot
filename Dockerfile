FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOTEL_API_HOST=0.0.0.0 \
    HOTEL_UI_PORT=7860 \
    HOTEL_DATABASE_PATH=/app/data/synthetic/hotel.db

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY app.py ./app.py
COPY configs ./configs
COPY data ./data

RUN python -m pip install --no-cache-dir .

EXPOSE 7860

CMD ["python", "-m", "app.gradio_app"]
