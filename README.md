# SNAPTOCK — AI workstream

Restock recommendation for Indonesian UMKM. Photograph a supplier's purchase
note (*nota pembelian*), extract the line items, update inventory, forecast when
stock runs out.

This repository holds the **AI and data pipeline**, plus the OCR service the
backend calls. Backend, frontend and the top-level `docker compose` land here as
they are built.

---

## Start here

| If you are… | read |
|---|---|
| **writing the proposal** | [`docs/DESIGN-DECISIONS.md`](docs/DESIGN-DECISIONS.md) — every choice, its alternative, and the measurement that decided it |
| **integrating the OCR into the backend** | [`docs/INTEGRATION.md`](docs/INTEGRATION.md) — one endpoint, the response schema, four rules |
| **picking up the code** | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the seven stages and which module owns each |
| **checking the numbers** | [`docs/RESULTS.md`](docs/RESULTS.md) — the controlled experiments |
| **wondering if the data is trustworthy** | [`docs/dataset-audit.html`](docs/dataset-audit.html) — it was not, and this is why |

---

## The pipeline

```
photo ──▶ DETECT ──▶ LEVEL ──▶ CUT ──▶ READ ──▶ GRID ──▶ ROLES ──▶ RECONCILE ──▶ JSON
          stock      geometry   geometry  FINE-   geometry  arithmetic  arithmetic
          PP-OCRv5                        TUNED
```

**Exactly one stage is a trained model.** Detection is stock PP-OCRv5.
Orientation, line rectification, the table grid, column roles and validation are
geometry and arithmetic. That is what lets a different supplier's nota work
without recalibration, and it is why the system's failures are inspectable
rather than opaque.

The load-bearing idea is that **the receipt checks itself**:

```
qty × harga = jumlah
```

That identity holds on 98.1% of line items in the corpus. It is used to decide
which column is which — a wrong assignment fails on every row at once — and to
catch recognition errors, including the dropped zeros that would otherwise write
a 10× price into stock. It is never used to silently *repair* a row: a fix that
restores the arithmetic while inventing a price is worse than no fix.

---

## Headline numbers

**Reading handwriting** — 1,023 crops from receipts the model never trained on:

| | held-out real receipts | unseen column layouts |
|---|---:|---:|
| digits (qty, harga, jumlah) | **0.13%** CER | **0.00%** |
| letters (product names) | **1.08%** CER | **0.16%** |
| whole-crop exact match | **98.24%** | **99.86%** |

**Understanding the table** — 200 generated nota, column orders never trained on:

| | |
|---|---:|
| hard-coded column positions (what shipped first) | 37.0% |
| **roles discovered from the receipt's arithmetic** | **96.0%** |

**Surviving a real photograph** — the three failures that returned an *empty*
document to the backend, and what fixing them bought:

| | before | after |
|---|---:|---:|
| lines read on a sideways hand-held nota | 24 | **34** |
| item names recovered when qty is welded onto the name | 0% | **78.7%** |
| rows reconciling across 40 held-out real receipts | 89.6% | 89.3% *(unchanged)* |

---

## Repository map

| Path | |
|---|---|
| `serving/` | the OCR service — FastAPI, Docker, the API the backend calls |
| `serving/geometry.py` | levels the page, turns it upright, cuts each line out square |
| `serving/layout.py` | discovers the table; assigns column roles by arithmetic |
| `serving/pipeline.py` | rows → validated line items → the response |
| `ml/synth_nota.py` | rendered synthetic nota, layout-held-out split |
| `ml/composite_real.py` | synthetic line items written into real photographs |
| `ml/eval_layout_generalization.py` | the 37.0% → 96.0% benchmark |
| `notebooks/01_eda.ipynb` | corpus exploration |
| `notebooks/02_train_ocr.ipynb` | **fine-tune PP-OCRv5** — download, prepare, train, evaluate |
| `notebooks/03_forecasting.ipynb` | **SBA vs Chronos-Bolt** — backtest, inventory simulation |

Each notebook is self-contained: it generates or downloads whatever it needs and
defines its own functions. Nothing imports from anything else in this repository.

---

## Quick start

**Run the OCR service** (the model weights are not in git — 114 MB exceeds
GitHub's file limit; use the delivery zip or export from the training notebook):

```bash
cd serving
MOCK=1 docker compose up          # fixture responses, no model, no GPU
docker compose up --build         # the real thing, on :8001
```

```bash
curl -F image=@nota.jpg http://localhost:8001/ocr/nota
```

**Run the notebooks:**

```bash
cp .env.example .env              # set ROBOFLOW_API_KEY
jupyter lab
```

Install PaddlePaddle **first and separately** — the wheel must match your CUDA
version and this is where setup usually fails:

```bash
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -c "import paddle; paddle.utils.run_check()"
```

> If you hit a `paddle.pir` circular import, an unrelated PyPI package named
> `paddle` has shadowed PaddlePaddle:
> `pip uninstall -y paddle && pip install --force-reinstall paddlepaddle==3.0.0`

---

## Two things about the data that will bite you

**1. Never split on `image_id`.** Roboflow generated ~3 brightness-augmented
copies of every source receipt, each with its own `image_id`. Splitting on it
puts copies of the same nota in train, validation *and* test — measured at 96.1%
of validation crops contaminated, leaving a genuinely unseen test set of **three
receipts**. Splits group by the receipt id parsed from the filename, and
preparation asserts the partitions are disjoint before training may start. Full
forensics in [`docs/dataset-audit.html`](docs/dataset-audit.html).

**2. The corrected split still flatters you.** The corpus has only two or three
distinct writers, so grouping by receipt still puts the same handwriting in every
split. Handwriting style — not receipt identity — is what breaks in deployment.
There are no writer labels in the corpus, so every number here carries that
caveat.

---

## Data

`nota-pembelian` v9 from Roboflow (CC BY 4.0): 440 photographed receipts, 1,295
images after augmentation, 18,585 word-level annotations, 739 label classes.
Handwritten ballpoint on the standard pre-printed Indonesian nota booklet.

Nothing under `data/` is committed.

---

## Conventions

Commits follow [Conventional Commits](https://www.conventionalcommits.org):
`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.
