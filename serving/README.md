# OCR service — for the backend

You send a photo of a nota. You get back the line items. That's the whole API.

You never install PaddlePaddle, never load a model, never see a `.pdiparams`.
One HTTP call.

---

## Start it (no model needed)

```bash
cd serving
MOCK=1 docker compose up
```

`MOCK=1` returns a realistic fixture with **exactly the same JSON shape** as the
real thing. Build the entire integration against it — when the trained model
drops in later, nothing on your side changes.

```bash
curl -F image=@nota.jpg http://localhost:8001/ocr/nota
```

---

## `POST /ocr/nota`

`multipart/form-data`, field name **`image`**. JPEG or PNG, max 12 MB.

```json
{
  "items": [
    {"nama": "teh celup",  "qty": 2,  "harga": 5000,  "jumlah": 10000,
     "confidence": 0.961, "reconciled": true, "warnings": []},
    {"nama": "gula pasir", "qty": 1,  "harga": 15000, "jumlah": 15000,
     "confidence": 0.618, "reconciled": true, "warnings": ["low_confidence"]}
  ],
  "total": {"computed": 40000, "stated": 40000, "matches": true},
  "warnings": [],
  "needs_review": true,
  "ms": 1840
}
```

| field | what it means |
|---|---|
| `qty` `harga` `jumlah` | integers, rupiah. **`null` if we couldn't read it** |
| `confidence` | 0–1, the weakest character in that row |
| `reconciled` | `qty × harga == jumlah` checked out |
| `needs_review` | **the one field to branch on** — true if any row needs a human |
| `total.stated` | the "Jumlah Rp." on the receipt. `null` if unreadable — never guessed |
| `total.matches` | `null` when `stated` is null, otherwise true/false |

### Errors

| code | `error` | when |
|---|---|---|
| 400 | `empty_upload` | no file sent |
| 413 | `file_too_large` | over 12 MB |
| 415 | `unreadable_image` | not a decodable image |
| 422 | `no_text_detected` | blurry, or not a nota |

### `GET /health`

```json
{"status": "ok", "mock": false, "models_loaded": ["det", "rec"]}
```

---

## Three things to know

**1. Branch on `needs_review`, not on `confidence`.**
The thresholds live inside the service and will move as the model improves.
The boolean won't.

**2. Never auto-save a row where `reconciled` is false.**
It means the receipt's own arithmetic disagrees with what we read. We report
the conflict instead of guessing, because `qty × harga = jumlah` has many
solutions and picking the wrong one silently writes a wrong price into stock.
Show the user that row and let them confirm.

**3. Product names are wrong far more often than numbers.**
Measured on held-out test data: digits are misread **1.3%** of the time,
letters **17%**. So prices and quantities are close to reliable; `nama` is not.
Make correcting a name one tap, and never block the flow on one.

Latency is 2–5 s per nota on CPU. Synchronous by design.

---

## Running it for real (later, not needed yet)

Drop three files into `models/rec/` — `inference.json`, `inference.pdiparams`,
`inference.yml` — then `docker compose up --build`. Nafis will send them; they
are not in git because the weights are 114 MB.
