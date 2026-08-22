"""
SNAPTOCK OCR service.

    POST /ocr/nota   multipart image -> validated line items
    GET  /health     readiness, and whether a real model is loaded

Run with MOCK=1 to serve a fixture with no model and no paddle installed.
That exists so the backend can be built against a fixed contract while the
model is still training -- the API shape is identical either way.
"""

from __future__ import annotations

import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pipeline import assemble

MOCK = os.environ.get("MOCK", "0") == "1"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/rec"))
REC_MODEL_NAME = os.environ.get("REC_MODEL_NAME", "PP-OCRv5_mobile_rec")
DET_MODEL_NAME = os.environ.get("DET_MODEL_NAME", "PP-OCRv5_mobile_det")
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 12 * 1024 * 1024))

FIXTURE = Path(__file__).parent / "fixtures" / "sample_response.json"

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load once at startup. Loading per request would dominate latency."""
    if MOCK:
        print("MOCK=1 — serving fixtures, no model loaded")
    else:
        from paddleocr import TextDetection, TextRecognition
        print(f"loading detection ({DET_MODEL_NAME}) ...")
        models["det"] = TextDetection(model_name=DET_MODEL_NAME)
        print(f"loading recognition ({REC_MODEL_NAME}) from {MODEL_DIR} ...")
        # model_name is required: TextRecognition validates the directory's
        # config against its default (PP-OCRv6_medium_rec) and rejects ours.
        models["rec"] = TextRecognition(model_name=REC_MODEL_NAME,
                                        model_dir=str(MODEL_DIR))
        print("models ready")
    yield
    models.clear()


app = FastAPI(title="SNAPTOCK OCR", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "mock": MOCK,
            "models_loaded": sorted(models) if not MOCK else []}


def crop_boxes(image, det_result, padding=0.05):
    """Crop each detected polygon's bounding rect, matching training padding."""
    boxes = []
    for poly in det_result.get("dt_polys", []):
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        px, py = w * padding, h * padding
        crop = image.crop((max(0, int(x - px)), max(0, int(y - py)),
                           min(image.width, int(x + w + px)),
                           min(image.height, int(y + h + py))))
        if crop.width >= 4 and crop.height >= 4:
            boxes.append({"crop": crop, "x": x, "y": y, "w": w, "h": h})
    return boxes


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

    from PIL import Image
    try:
        page = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(415, detail={"error": "unreadable_image"})

    det = models["det"].predict([page])[0]
    crops = crop_boxes(page, det)
    if not crops:
        raise HTTPException(422, detail={"error": "no_text_detected"})

    preds = models["rec"].predict([c["crop"] for c in crops])
    boxes = [{"text": (p.get("rec_text") or "").strip(),
              "conf": float(p.get("rec_score") or 0.0),
              "x": c["x"], "y": c["y"], "w": c["w"], "h": c["h"]}
             for c, p in zip(crops, preds)]

    body = assemble([b for b in boxes if b["text"]], page.width, page.height)
    body["ms"] = int((time.perf_counter() - started) * 1000)
    return JSONResponse(body)
