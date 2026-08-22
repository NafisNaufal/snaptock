# OCR service — API contract

The backend never loads a model. It makes one HTTP call.

Base URL in compose: `http://ocr:8001`

---

## `POST /ocr/nota`

`multipart/form-data`, field name **`image`** — a JPEG or PNG photo of a nota.
Max 12 MB.

### 200 — parsed

```json
{
  "items": [
    {"nama": "teh celup",   "qty": 2,  "harga": 5000,  "jumlah": 10000,
     "confidence": 0.961, "reconciled": true,  "warnings": []},
    {"nama": "gula pasir",  "qty": 1,  "harga": 15000, "jumlah": 15000,
     "confidence": 0.618, "reconciled": true,  "warnings": ["low_confidence"]}
  ],
  "total": {"computed": 40000, "stated": 40000, "matches": true},
  "warnings": [],
  "needs_review": true,
  "ms": 1840
}
```

| field | meaning |
|---|---|
| `qty`, `harga`, `jumlah` | integers in rupiah / units. `null` if unreadable |
| `confidence` | 0–1, the **weakest** character in that row. Not a probability of correctness — treat it as a ranking signal |
| `reconciled` | `qty × harga == jumlah` held for this row |
| `warnings` | `low_confidence`, `incomplete_row`, `arithmetic_mismatch: …` |
| `needs_review` | **true if any row needs a human.** The one field to branch on |

### Errors

| code | body `error` | when |
|---|---|---|
| 400 | `empty_upload` | no file |
| 413 | `file_too_large` | over 12 MB |
| 415 | `unreadable_image` | not a decodable image |
| 422 | `no_text_detected` | detection found nothing — blurry, or not a nota |

---

## `GET /health`

```json
{"status": "ok", "mock": false, "models_loaded": ["det", "rec"]}
```

---

## What the backend must do with this

**Branch on `needs_review`, not on `confidence`.** The thresholds live in the
service and will change as the model improves; the boolean will not.

**Never auto-commit a row where `reconciled` is false.** The receipt's own
arithmetic disagrees with what we read. `qty × harga = jumlah` is
under-determined, so the service refuses to guess — it reports the conflict and
leaves the decision to a person.

**Expect names to be wrong more often than numbers.** Measured on the held-out
test split: digit CER **1.34%**, letter CER **16.90%**. Prices and quantities
are close to reliable; product names are not. The UI should make correcting a
`nama` trivial and should never block on one.

**`ms` is wall-clock inference time.** Budget 2–5 s per nota on CPU. Synchronous
by design — the competition scope forbids background jobs.

---

## Building against it before the model exists

```bash
cd serving && MOCK=1 docker compose up
curl -F image=@any.jpg http://localhost:8001/ocr/nota
```

`MOCK=1` needs no model and no paddle. It returns the fixture above with the
identical schema, so the entire Express integration can be written and tested
now. Switching to the real model changes nothing on the backend side.
