FROM python:3.10-slim

WORKDIR /app

# Install deps first for better layer caching (only re-installs on requirements.txt change)
COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile -r requirements.txt \
    && pip cache purge

COPY train.py .

RUN useradd --create-home --uid 1000 trainer
USER trainer

ENTRYPOINT ["python3", "train.py"]