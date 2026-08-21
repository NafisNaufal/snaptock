# SNAPTOCK — AI workstream

Restock recommendation for Indonesian UMKM. Photograph a supplier's purchase
note (*nota pembelian*), extract the line items, update inventory, forecast
when stock runs out.

This repository currently holds the **AI and data pipeline**. Backend, frontend
and the top-level `docker compose` land here as they are built.

---

## What's here

| Path | |
|---|---|
| `ml/download_dataset.sh` | Fetch the nota-pembelian COCO export from Roboflow |
| `ml/prepare_dataset.py` | Build the PaddleOCR recognition dataset (source-grouped split) |
| `ml/finetune_rec.sh` | Fine-tune PP-OCRv5 recognition on handwritten nota |
| `ml/configs/` | Training config, derived from PaddleOCR v3.7.0 |
| `ml/generate_sales.py` | Synthetic daily sales flow for the forecasting engine |
| `notebooks/01_eda.ipynb` | Original exploratory analysis (split superseded — see banner) |
| `docs/dataset-audit.html` | Forensic audit of the corpus. **Read this first.** |
| `docs/run-a-research.html` | Literature review behind the architecture choice |

---

## Quick start

```bash
git clone <this repo> && cd compfest
cp .env.example .env          # set ROBOFLOW_API_KEY
python -m pip install -r requirements.txt

bash ml/download_dataset.sh                                   # ~331 MB
python ml/prepare_dataset.py \
    --coco data/raw/train/_annotations.coco.json \
    --images data/raw/train \
    --out data/rec
```

Expected:

```
train     308 receipts   906 images   12,652 crops
val        66 receipts    66 images      989 crops
test       66 receipts    66 images    1,023 crops
```

### Fine-tuning (GPU)

Install PaddlePaddle **first and separately** — the wheel must match your CUDA
version, and this is where setup usually fails:

```bash
python -m pip install paddlepaddle-gpu==3.0.0    # check paddlepaddle.org.cn for your CUDA
python -c "import paddle; paddle.utils.run_check()"
```

Then:

```bash
bash ml/finetune_rec.sh ~/work
```

It clones PaddleOCR at a pinned tag, downloads pretrained weights, refuses to
start if the split leaks, trains, and exports an inference model to
`~/work/PaddleOCR/output/nota_rec_v5_mobile_infer/`.

---

## Two things that will bite you

**1. Never split on `image_id`.** Roboflow generated ~3 brightness-augmented
copies of every source receipt, each with its own `image_id`. Splitting on it
puts copies of the same nota in train, val *and* test — measured at 96.1% of
val crops contaminated. `prepare_dataset.py` splits on the receipt id parsed
from the filename and `finetune_rec.sh` refuses to train if that guarantee
breaks. Full numbers in `docs/dataset-audit.html`.

**2. The default split still flatters you.** The corpus has only two or three
distinct writers, so grouping by receipt still puts the same handwriting in
every split. For an honest number, label writers into a CSV and pass it:

```bash
# writers.csv
# source_id,writer_id
# nota203,A
# nota046,B
python ml/prepare_dataset.py ... --writer-map writers.csv
```

Handwriting style — not receipt identity — is what breaks in deployment.

---

## Architecture notes

Recognition is a **CTC-based PP-OCRv5** model, not a transformer seq2seq model
and not a document VLM. CTC trains from far less data, runs on CPU after INT8
quantization, and cannot hallucinate values that aren't on the page. The
reasoning and citations are in `docs/run-a-research.html`.

Field assignment (`BANYAKNYA` / `NAMA BARANG` / `HARGA` / `JUMLAH`) is
**geometric**, not learned — the nota booklet is a fixed printed template, so
column position determines field type. This holds for this template and fails
for other suppliers' layouts; that limit is deliberate and declared.

The system exploits the receipt's own arithmetic: `qty x harga = jumlah`, and
the column total. That identity holds on **98.1%** of line items in the corpus,
which makes it usable both to detect recognition errors and to guide repair.
Irreconcilable lines are flagged for user confirmation, never silently rewritten.

---

## Data

`nota-pembelian` v9 from Roboflow (CC BY 4.0): 440 photographed receipts,
1,295 images after augmentation, 18,585 word-level annotations, 739 label
classes. Handwritten ballpoint on the standard pre-printed Indonesian nota
booklet.

Nothing under `data/` is committed. Rebuild it with the two commands above.

---

## Conventions

Commits follow [Conventional Commits](https://www.conventionalcommits.org):
`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.
