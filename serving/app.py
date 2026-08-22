"""
SNAPTOCK OCR service.

    POST /ocr/nota   multipart image -> validated line items
    GET  /health     readiness

Set MOCK=1 to serve a fixture with no model and no paddle installed.
"""

from __future__ import annotations

import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from pipeline import assemble

MOCK = os.environ.get("MOCK", "0") == "1"
MODEL_DIR = os.environ.get("MODEL_DIR", "models/rec")
REC_MODEL = os.environ.get("REC_MODEL_NAME", "PP-OCRv5_mobile_rec")
DET_MODEL = os.environ.get("DET_MODEL_NAME", "PP-OCRv5_mobile_det")
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 12 * 1024 * 1024))
PADDING = 0.05          # must match the padding used to build training crops

FIXTURE = Path(__file__).parent / "fixtures" / "sample_response.json"
models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load once at startup; loading per request would dominate latency."""
    if MOCK:
        print("MOCK=1 - serving fixtures, no model loaded")
    else:
        from paddleocr import TextDetection, TextRecognition
        models["det"] = TextDetection(model_name=DET_MODEL)
        # model_name is required: TextRecognition validates the directory's
        # config against its default (PP-OCRv6) and would reject our v5 export.
        models["rec"] = TextRecognition(model_name=REC_MODEL, model_dir=MODEL_DIR)
        print("models ready")
    yield
    models.clear()


app = FastAPI(title="SNAPTOCK OCR", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "mock": MOCK, "models_loaded": sorted(models)}


def crop(page: np.ndarray, poly) -> tuple[np.ndarray, dict]:
    """Bounding rect of a detected polygon, padded like the training crops."""
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    x0 = max(0, int(x - w * PADDING))
    y0 = max(0, int(y - h * PADDING))
    x1 = min(page.shape[1], int(x + w * (1 + PADDING)))
    y1 = min(page.shape[0], int(y + h * (1 + PADDING)))
    return page[y0:y1, x0:x1], {"x": x, "y": y, "w": w, "h": h}


@app.post("/ocr/nota")
async def ocr_nota(image: UploadFile = File(...)):
    started = time.perf_counter()
    raw = await image.read()

    if not raw:
        raise HTTPException(400, detail={"error": "empty_upload"})
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, detail={"error": "file_too_large",
                                         "max_bytes": MAX_BYTES})

    if MOCK:
        body = json.loads(FIXTURE.read_text())
        body["ms"] = int((time.perf_counter() - started) * 1000)
        return JSONResponse(body)

    try:
        page = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        raise HTTPException(415, detail={"error": "unreadable_image"})

    # PaddleOCR accepts numpy arrays or paths only - it silently ignores PIL
    # images and returns nothing, so everything below stays numpy.
    detected = models["det"].predict(page)
    polys = detected[0]["dt_polys"] if detected else []

    crops, geometry = [], []
    for poly in polys:
        patch, box = crop(page, poly)
        if patch.shape[0] >= 4 and patch.shape[1] >= 4:
            crops.append(patch)
            geometry.append(box)

    if not crops:
        raise HTTPException(422, detail={"error": "no_text_detected"})

    boxes = [
        {**box,
         "text": (pred.get("rec_text") or "").strip(),
         "conf": float(pred.get("rec_score") or 0.0)}
        for box, pred in zip(geometry, models["rec"].predict(crops))
    ]

    height, width = page.shape[:2]
    body = assemble([b for b in boxes if b["text"]], width, height)
    body["ms"] = int((time.perf_counter() - started) * 1000)
    return JSONResponse(body)
