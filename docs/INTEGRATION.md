# Integration handoff — nota OCR

Everything needed to put the OCR into the app. No research, no roadmap.

---

## 1. What you receive

| | |
|---|---|
| `snaptok-ocr.zip` (85 MB) | the service **and** the trained model, ready to run |
| `github.com/NafisNaufal/snaptock` | source, notebooks, docs |

The zip contains the model weights; the repo does not (114 MB exceeds GitHub's
file limit). **Use the zip.**

---

## 2. Run it

```bash
unzip snaptok-ocr.zip -d ocr && cd ocr
docker compose up --build
```

Live on `http://localhost:8001`. No Python setup, no PaddlePaddle install.

Without Docker:

```bash
pip install -r requirements.txt
pip install paddlepaddle==3.0.0 paddleocr==3.7.0
MODEL_DIR=models/rec uvicorn app:app --port 8001
```

> If the direct install fails with a `paddle.pir` circular import, an unrelated
> PyPI package named `paddle` has shadowed PaddlePaddle:
> `pip uninstall -y paddle && pip install --force-reinstall paddlepaddle==3.0.0`

---

## 3. The one endpoint

`POST /ocr/nota` — `multipart/form-data`, field name **`image`**, JPEG or PNG,
max 12 MB.

```json
{
  "items": [
    {"nama": "kopi sachet", "qty": 10, "harga": 1500, "jumlah": 15000,
     "confidence": 0.96, "reconciled": true,  "warnings": []},
    {"nama": "gula pasir",  "qty": 1,  "harga": 15000, "jumlah": 15000,
     "confidence": 0.96, "reconciled": true,  "warnings": []}
  ],
  "total": {"computed": 40000, "stated": 40000, "matches": true},
  "warnings": [],
  "needs_review": false,
  "ms": 2019
}
```

| field | meaning |
|---|---|
| `qty` `harga` `jumlah` | integers, rupiah. **`null` if unreadable — check before saving** |
| `confidence` | 0–1, the weakest character in that row |
| `reconciled` | `qty × harga == jumlah` held for this row |
| `warnings` | `incomplete_row`, `low_confidence`, `arithmetic_mismatch: …` |
| `needs_review` | **the field to branch on** — true if any row needs a human |
| `total.stated` | the "Jumlah Rp." on the receipt. `null` if unreadable, never guessed |
| `total.matches` | `null` when `stated` is null, else true/false |

**Errors:** `400` empty upload · `413` over 12 MB · `415` unreadable image ·
`422` no text detected. Body is `{"error": "<code>"}`.

**Health:** `GET /health` → `{"status":"ok","mock":false,"models_loaded":["det","rec"]}`

---

## 4. Four rules for the backend

**Branch on `needs_review`, not on `confidence`.** Thresholds live inside the
service and will move; the boolean will not.

**Never auto-save a row where `reconciled` is false.** The receipt's own
arithmetic disagrees with what we read. We report the conflict rather than
guess, because `qty × harga = jumlah` has many possible repairs and picking the
wrong one writes a wrong price into stock.

**A `null` is not a zero.** Detection sometimes misses a small isolated digit —
usually a lone `1` in the quantity column. Those rows return `qty: null` with
`incomplete_row`.

**Product names are wrong more often than numbers.** Measured on held-out
receipts: digits 0.13% character error, letters 1.08%. Make correcting a `nama`
one tap; never block the flow on one.

---

## 5. Build the mock first

```bash
MOCK=1 docker compose up
```

Returns a fixture with the **identical schema**, no model, no GPU. Write and
test the whole integration against it, then flip `MOCK=0`. Nothing on your side
changes.

---

## 6. Performance and limits

- **2–5 seconds per nota** on CPU, synchronous by design.
- Models load once at startup (~60–90 s). Do not restart per request.
- Built for the standard pre-printed Indonesian nota booklet
  (`BANYAKNYA / NAMA BARANG / HARGA / JUMLAH`). Column positions are
  **discovered per document**, so other suppliers' column orders work; a
  free-form handwritten note on blank paper does not.
- Thermal struk (printed cash-register receipts) are untested.

## 7. Accuracy

| | held-out real receipts | unseen layouts |
|---|---|---|
| digits | **0.13%** | **0.00%** |
| letters | **1.08%** | **0.16%** |
| character error | **0.48%** | **0.06%** |
| whole-crop exact match | **98.24%** | **99.86%** |

Measured on 1,023 real crops and 1,470 crops from column orders never trained
on. Caveat worth knowing: the corpus has no writer labels, so the test set
shares handwriting with training and these numbers flatter slightly on a
genuinely new writer.

---

## 8. Files

| | |
|---|---|
| `app.py` | routes, upload limits, model lifecycle |
| `pipeline.py` | boxes → rows → columns → line items → reconciliation |
| `layout.py` | column discovery; role assignment by arithmetic |
| `models/rec/` | the fine-tuned recogniser (114 MB) |
| `fixtures/` | the mock response |

Detection uses stock PP-OCRv5; only recognition is fine-tuned. Field assignment
is geometric and reconciliation is arithmetic — neither is a learned model.
