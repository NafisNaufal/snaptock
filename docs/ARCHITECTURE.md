# Architecture

How a photograph of a nota becomes validated line items, stage by stage, and
which part of the repository owns each stage.

The whole path is seven stages. **One of them is a trained model.** Everything
else is geometry and arithmetic, and that is a deliberate choice rather than an
accident of scope — the reasoning is in [DESIGN-DECISIONS.md](DESIGN-DECISIONS.md).

```
photo ──▶ DETECT ──▶ LEVEL ──▶ CUT ──▶ READ ──▶ GRID ──▶ ROLES ──▶ RECONCILE ──▶ JSON
          stock      geometry   geometry  FINE-   geometry  arithmetic  arithmetic
          PP-OCRv5              TUNED
```

---

## 1. Detect — where is the text?

`serving/app.py`

Stock **PP-OCRv5 mobile detection** (DB segmentation), unmodified. It returns
one quadrilateral per text line. This model is not fine-tuned; it locates text
well enough on handwriting as shipped.

## 2. Level — which way is up?

`serving/geometry.py` · `page_angle`, `rotate`, `deskew`, `upright`

A photographed nota is rarely upright. The recogniser only reads horizontal
text, so a sideways page yields empty strings on every line.

No second model is used. The detector's quadrilaterals already encode the answer:
text lines run along the writing direction, so the length-weighted average edge
angle of those quads **is** the page angle.

Angles are averaged **doubled and then halved**. A line direction is only defined
modulo 180°, so a page of vertical lines reads as −90° on some quads and +90° on
others; plain averaging cancels them to zero and leaves the page sideways.
Doubling maps both onto the same direction first.

That fixes tilt and all four quarter turns. It cannot distinguish upright from
upside down — both give the same angle — so `upright()` reads the longest few
lines both ways and keeps whichever the recogniser is more confident about. Long
lines are the item names, which carry real words; a confidence gap there means
something, where a gap on a two-digit quantity does not.

`deskew()` **iterates**. On a page lying at 86° the detector is working at its
worst and its angle is a couple of degrees out; only once the page is roughly
level does it report the rest. Each pass refines the running angle and rotates
the *original* again, so the pixels are only ever resampled once.

## 3. Cut — one rectangle per line

`serving/geometry.py` · `rectify`, `bounds`

Each quadrilateral is warped onto an upright rectangle. Cropping to the
axis-aligned bounding box instead is only correct when the line is level: on a
hand-held page a slanted line's bounding box also swallows the paper and the
neighbouring lines around it. A line the detector found standing on end is stood
back up. Padding matches the margin the training crops were cut with.

## 4. Read — the one trained model

`serving/app.py` · `serving/models/rec/`

**PP-OCRv5 mobile recognition, fine-tuned on handwritten nota.** CTC over a
sequence-to-sequence transformer or a document VLM, for three reasons: it trains
from far less data, it runs on CPU after quantisation, and **it cannot
hallucinate a value that is not on the page** — which is what matters when a
wrong digit silently corrupts inventory.

Character-level, not word-level. The corpus labels are a closed vocabulary of 739
literal strings; a 739-way classifier could never emit a product name it had not
seen.

## 5. Grid — where are the rows and columns?

`serving/layout.py` · `cluster_1d`, `row_pitch`, `build_grid`

Columns first: they are separated by whitespace far wider than any word gap, so
clustering box centres on the x-axis is reliable. Rows are then measured against
them.

**Row pitch is measured, not inferred from box size.** A column holds exactly one
entry per row, so the step between consecutive entries down a column *is* the row
pitch. Box height cannot stand in for it — the detector pads each line
generously, so on a real nota the boxes come out *taller than the line spacing*
and every row overlaps its neighbours.

**Groups are anchored on their own running centre.** Splitting on the gap between
neighbouring values lets one straddling box chain two rows into one, and on a
table that is how four item rows become a single row: every neighbouring pair sits
closer than the threshold while the two ends lie a couple of pitches apart.

## 6. Roles — which column is which?

`serving/layout.py` · `search_roles`, `assign_roles`, `split_count`, `unmerge_counts`

Not by position, and not by reading the printed headers. **By which assignment
makes the receipt's own arithmetic hold.**

```
qty × harga = jumlah
```

A wrong assignment fails that identity on every row at once, so every ordered
triple of columns is scored by how many rows reconcile and the best is kept. No
training, no template knowledge.

Multiplication is commutative, so `qty × harga == harga × qty` and the arithmetic
alone cannot separate the quantity column from the unit-price column. One weak
prior breaks the tie: quantities are small, prices are not.

Two structures the clustering cannot reach are handled here:

- **A quantity welded onto a name.** The detector sometimes swallows the quantity
  cell into the name cell, so `2 kg gula pasir` arrives as one text line. The
  parser splits the leading count off and scores the document *both ways*, keeping
  whichever adds up better. A tie goes to the split, because the unsplit reading
  reconciles just as well while leaving the receipt with no item names at all.
- **A form with no unit-price column at all.** Some booklets print only quantity
  and total.

If the arithmetic cannot arbitrate — too many cells unreadable on a poor photo —
the parser falls back on the one convention every Indonesian nota shares: the
money runs on the right, the running total to the right of the unit price. Those
rows still leave flagged, so the shopkeeper corrects pre-filled rows instead of
typing them.

## 7. Reconcile — is what we read self-consistent?

`serving/pipeline.py` · `reconcile`, `assemble`

The receipt carries a redundant third value, so it checks itself. The identity
holds on **98.1%** of line items in the corpus, which makes it usable as an error
detector.

**Nothing is silently repaired.** `qty × harga = jumlah` has many solutions, and
one that restores the arithmetic while inventing a price is worse than none. A
row that does not reconcile is returned with `reconciled: false` and the
discrepancy spelled out.

The grand total is read from the money column *below the last item row*. If it
cannot be read it is `null` — never guessed, and never scanned for when no items
were found, or the nota number at the top of the page gets reported as the total.

---

## Repository map

| Path | |
|---|---|
| `serving/app.py` | the API: routes, upload limits, model lifecycle |
| `serving/geometry.py` | stages 2–3: level the page, cut each line out square |
| `serving/layout.py` | stages 5–6: grid discovery, role assignment by arithmetic |
| `serving/pipeline.py` | stage 7: rows → reconciled line items → the response |
| `serving/models/rec/` | the fine-tuned recogniser (114 MB, not in git) |
| `serving/fixtures/` | mock response, served when `MOCK=1` |
| `ml/synth_nota.py` | rendered synthetic nota, layout-held-out split |
| `ml/composite_real.py` | synthetic line items written into real photographs |
| `ml/wordpool.py` | verification gate for generated handwriting |
| `ml/eval_layout_generalization.py` | the fixed-geometry vs discovered benchmark |
| `ml/sku_catalogue.csv` | 200 Indonesian warung SKUs with plausible prices |
| `notebooks/01_eda.ipynb` | corpus exploration |
| `notebooks/02_train_ocr.ipynb` | download → prepare → fine-tune → evaluate |
| `notebooks/03_forecasting.ipynb` | SBA vs Chronos-Bolt, inventory simulation |

Each notebook is self-contained: it generates or downloads whatever it needs and
defines its own functions. Nothing imports from anything else in the repository.

---

## Running the service

```bash
cd serving
docker compose up --build            # http://localhost:8001
```

Without Docker (Python 3.10+):

```bash
pip install -r requirements.txt
pip install paddlepaddle==3.0.0 paddleocr==3.7.0
MODEL_DIR=models/rec uvicorn app:app --port 8001
```

`MOCK=1` serves a fixture with the identical schema and needs neither the model
nor PaddlePaddle — the whole backend integration can be built against it.

The API contract is in [INTEGRATION.md](INTEGRATION.md).

### Environment variables

| | default | |
|---|---|---|
| `MODEL_DIR` | `models/rec` | fine-tuned recogniser |
| `REC_MODEL_NAME` | `PP-OCRv5_mobile_rec` | required; the loader validates the export against it |
| `DET_MODEL_NAME` | `PP-OCRv5_mobile_det` | stock detection |
| `DET_SIDE_LEN` | `960` | detection resolution; see DESIGN-DECISIONS §6 |
| `MAX_UPLOAD_BYTES` | `12582912` | 12 MB |
| `MOCK` | `0` | `1` serves fixtures, no model |
