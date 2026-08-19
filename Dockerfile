FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY static ./static
COPY artifacts ./artifacts
COPY reports/model_metadata.json ./reports/model_metadata.json

USER app
EXPOSE 8080
CMD ["sh", "-c", "uvicorn spotter_rate_intelligence.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
