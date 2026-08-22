FROM python:3.10-slim

# opencv (pulled in by paddleocr) needs these at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir paddlepaddle==3.0.0 paddleocr==3.7.0

COPY app.py pipeline.py ./
COPY fixtures ./fixtures

# The fine-tuned recognizer. 113 MB of weights cannot live in git, so mount
# them at run time (see docker-compose.yml) or bake them in with:
#     COPY models/rec /app/models/rec
COPY models/ ./models/

ENV MODEL_DIR=/app/models/rec \
    REC_MODEL_NAME=PP-OCRv5_mobile_rec \
    MOCK=0

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
