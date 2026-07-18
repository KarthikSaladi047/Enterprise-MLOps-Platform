FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pydantic httpx gradio requests

COPY ui.py .
COPY api.py .

EXPOSE 8000 7860