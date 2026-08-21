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
| `notebooks/01_eda.ipynb` | Exploratory analysis of the nota corpus |
| `notebooks/02_train_ocr.ipynb` | **Fine-tune PP-OCRv5 on handwritten nota** — download, prepare, train, evaluate |
| `notebooks/03_forecasting.ipynb` | **SBA vs Chronos-Bolt** — generate data, backtest, inventory simulation |
| `docs/dataset-audit.html` | Forensic audit of the corpus. **Read this first.** |
| `docs/run-a-research.html` | Literature review behind the architecture choice |

Each notebook is self-contained: it generates or downloads whatever it needs and defines
its own functions. Nothing imports from anything else in this repository.

---

## Quick start

```bash
git clone <this repo> && cd compfest
cp .env.example .env          # set ROBOFLOW_API_KEY
jupyter lab
```

Open a notebook and run it top to bottom. Each one lists its own `pip install` line in
the first cell; there is no shared requirements file because there are no shared modules.

For the OCR notebook, install PaddlePaddle **first and separately** — the wheel must match
your CUDA version and this is where setup usually fails:

```bash
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -c "import paddle; paddle.utils.run_check()"
```

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

```python
# in notebooks/02_train_ocr.ipynb
WRITER_MAP = {'nota203': 'A', 'nota046': 'B', ...}   # section 2b
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
