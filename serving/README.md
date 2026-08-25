# Nota OCR service

Send a photo of a nota, get back the line items. One HTTP call — no Python,
no PaddlePaddle, no model files on the backend side.

## Run it

This is a standalone Python service. Your backend just makes an HTTP call to it
— it does not matter what your backend is written in, or whether you use Docker
for it.

**With Docker** (recommended — you never install PaddlePaddle):

```bash
docker compose up --build
```

**Without Docker** (Python 3.10+):

```bash
pip install -r requirements.txt
pip install paddlepaddle==3.0.0 paddleocr==3.7.0
MODEL_DIR=models/rec uvicorn app:app --port 8001
```

Either way it listens on `http://localhost:8001`:

```bash
curl -F image=@nota.jpg http://localhost:8001/ocr/nota
```

Health check: `GET /health`

> If the direct install fails with a `paddle.pir` circular import, an unrelated
> PyPI package called `paddle` has shadowed PaddlePaddle. Fix:
> `pip uninstall -y paddle && pip install --force-reinstall paddlepaddle==3.0.0`
> This is the main reason Docker is recommended.

## Response

```json
{
  "items": [
    {"nama": "teh celup",  "qty": 2,  "harga": 5000,  "jumlah": 10000,
     "confidence": 1.0,   "reconciled": true,  "warnings": []},
    {"nama": "gula pasir", "qty": 1,  "harga": 15000, "jumlah": 15000,
     "confidence": 0.97,  "reconciled": true,  "warnings": []}
  ],
  "total": {"computed": 40000, "stated": 40000, "matches": true},
  "warnings": [],
  "needs_review": false,
  "ms": 2019
}
```

| field | meaning |
|---|---|
| `qty` `harga` `jumlah` | integers, rupiah. **`null` if unreadable** — check before saving |
| `confidence` | 0–1, the weakest character in that row |
| `reconciled` | `qty × harga == jumlah` held |
| `needs_review` | **the field to branch on** — true if any row needs a human |
| `total.stated` | the "Jumlah Rp." on the receipt. `null` if unreadable, never guessed |

Errors: `400` empty upload · `413` over 12 MB · `415` unreadable image ·
`422` no text detected.

## Three rules

1. **Branch on `needs_review`, not `confidence`.** Thresholds will change; the
   boolean won't.
2. **Never auto-save a row where `reconciled` is false.** The receipt's own
   arithmetic disagrees with what we read. We report it rather than guess.
3. **A `null` is not a zero.** Detection sometimes misses a small isolated
   digit. Those rows come back with `incomplete_row` — show them to the user.

Latency is 2–5 s per nota on CPU, synchronous.

## Accuracy

Measured on 1,023 held-out crops the model never trained on:

| | held-out real receipts | unseen layouts |
|---|---|---|
| digits (qty, harga, jumlah) | **0.13%** | **0.00%** |
| letters (product names) | **1.08%** | **0.16%** |
| character error, overall | **0.48%** | **0.06%** |
| whole crop exact match | **98.24%** | **99.86%** |

"Unseen layouts" means nota with column orders the model never trained on —
a different supplier's form.

## Files

| | |
|---|---|
| `app.py` | the API: routes, upload limits, model loading |
| `pipeline.py` | detection → crop → recognition → columns → reconciliation |
| `models/rec/` | the fine-tuned model (114 MB, not in git) |
| `fixtures/` | mock response, used when `MOCK=1` |

| `layout.py` | discovers the table: columns by clustering, roles by arithmetic |

Only recognition is a trained model. Table structure is **discovered per
document**, not assumed: columns are found by clustering box positions, and
which column is `harga` versus `jumlah` is decided by whichever assignment
makes `qty × harga = jumlah` hold across the rows. That means a different
supplier's template works without recalibration.
