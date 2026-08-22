# Nota OCR service

Send a photo of a nota, get back the line items. One HTTP call — no Python,
no PaddlePaddle, no model files on the backend side.

## Run it

```bash
docker compose up --build
```

First build takes a few minutes. Then:

```bash
curl -F image=@nota.jpg http://localhost:8001/ocr/nota
```

Health check: `GET /health`

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

| | error rate |
|---|---|
| digits (qty, harga, jumlah) | **0.21%** |
| letters (product names) | **1.28%** |
| whole crop exact match | **98.14%** |

## Files

| | |
|---|---|
| `app.py` | the API: routes, upload limits, model loading |
| `pipeline.py` | detection → crop → recognition → columns → reconciliation |
| `models/rec/` | the fine-tuned model (114 MB, not in git) |
| `fixtures/` | mock response, used when `MOCK=1` |

Only recognition is a trained model. Field assignment is geometric — the nota
booklet is a fixed printed template — and reconciliation is arithmetic.
