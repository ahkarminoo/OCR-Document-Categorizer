FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# System deps: Tesseract + OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY requirement.txt /app/requirement.txt
RUN pip install --no-cache-dir -r /app/requirement.txt

# Pre-download EasyOCR models so the first request isn't slow
RUN python -c "import easyocr; easyocr.Reader(['en'], verbose=False)" || true

# Copy application source
COPY . /app

# Persistent scan output storage
RUN mkdir -p /data/outputs
ENV OUTPUT_DIR=/data/outputs

EXPOSE 8000

# Production: no --reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
