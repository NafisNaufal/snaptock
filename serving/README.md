# OCR service

Wraps the fine-tuned handwritten-nota recognizer behind one HTTP endpoint, so
the Express backend never has to know PaddlePaddle exists.

**API contract: [`CONTRACT.md`](CONTRACT.md)** — that is what the backend codes
against.

## Run with the mock (no model needed)

```bash
MOCK=1 docker compose up
curl -F image=@sample.jpg http://localhost:8001/ocr/nota
```

## Run for real

Drop the three exported files into `models/rec/`:

```
models/rec/inference.json        225 KB   architecture (frozen graph)
models/rec/inference.pdiparams   114 MB   weights
models/rec/inference.yml         145 KB   preprocessing + character dict
```

They are **not in git** — the weights exceed GitHub's 100 MB file limit. Get
them from the training run, or from wherever the team is hosting them.

```bash
docker compose up --build
```

## Local, without Docker

```bash
pip install -r requirements.txt paddlepaddle==3.0.0 paddleocr==3.7.0
MODEL_DIR=models/rec uvicorn app:app --port 8001
```

## What is in here

| file | |
|---|---|
| `app.py` | FastAPI: routes, upload limits, model lifecycle |
| `pipeline.py` | detection → crop → recognition → column geometry → reconciliation |
| `fixtures/` | the mock response, and the schema of record |

Only the recognition step is a model we trained. Field assignment is geometric
(the nota booklet is a fixed printed template) and reconciliation is
arithmetic. See `../docs/dataset-audit.html`.
