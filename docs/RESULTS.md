# Handwritten nota OCR — experimental results

Three training runs on the same held-out test sets, changing one thing at a
time. The question throughout: **what actually makes the recogniser generalise
beyond the 440 receipts it was trained on?**

---

## 1. The corpus, and why it misleads

Roboflow `ocr-fqwqd/nota-pembelian` v9 advertises 1,295 images and 18,585
annotations. It is really **440 photographs**, each emitted three times with a
brightness shift, of **one** pre-printed nota booklet written by **two or three
people**.

Two consequences shaped everything after.

**The original split leaked almost completely.** The preparation code split on
COCO `image_id`, but the three augmented copies of one receipt carry three
different `image_id`s. Re-running that split reproduces its reported
`1036 / 129 / 130` partition exactly, and of the 118 distinct receipts landing
in validation, **115 also appear in training**.

| partition | crops | distinct nota | also in train | contaminated crops |
|---|---:|---:|---:|---:|
| validation | 1,793 | 118 | 115 | **96.1%** |
| test | 1,822 | 116 | 113 | **97.1%** |

The genuinely unseen evaluation set was three receipts. Every accuracy figure
produced before this was corrected is meaningless.

**The labels are a closed vocabulary.** The 739 COCO categories are the literal
text inside each box: 328 numerals and 411 Indonesian words. 274 of the 738 used
classes appear three times or fewer. This forces character-level recognition —
a 739-way classifier cannot emit a product name it has never seen — and it means
the character dictionary must be built from *all* categories, not just those
appearing in train.

---

## 2. Method

Recognition is a **CTC-based PP-OCRv5** model fine-tuned from pretrained weights.
CTC over a transformer sequence-to-sequence model or a document VLM, for three
reasons: it trains from far less data, it runs on CPU after quantisation, and it
cannot hallucinate a value that is not on the page — which matters when a wrong
digit silently corrupts inventory.

All splits group by **source receipt**, and the preparation step asserts the
three partitions are disjoint before training may start.

Two evaluation sets, deliberately measuring different things:

- **real** — 1,023 crops from held-out receipts. Does it read actual nota?
- **unseen layouts** — 1,470 crops from generated pages whose column orders
  (`no nama qty harga jumlah`, `nama harga qty jumlah`) appear in no training
  data. Does it survive a different supplier's form?

---

## 3. Results

| model | training data | crops | real CER | real digits | real exact | unseen CER | unseen digits | unseen exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | real only | 12,652 | 0.60% | 0.21% | 98.14% | 3.87% | 0.75% | 86.87% |
| **A** | + rendered pages | 29,183 | 0.61% | 0.29% | 98.14% | **0.17%** | 0.04% | **99.66%** |
| **B** | + composited on real photos | 35,679 | **0.48%** | **0.13%** | **98.24%** | **0.06%** | **0.00%** | **99.86%** |

### What each run showed

**Arm A: rendered synthetic pages fixed layout generalisation and nothing else.**
16,531 crops of programmatically drawn nota moved unseen-layout error from 3.87%
to 0.17% — and moved real-receipt accuracy by **exactly zero**. The model learned
the renderer's world: its fonts, its flat paper tints, its gentle warp. Not
handwriting.

**Arm B: compositing onto real photographs moved the number A could not.**
Rather than drawing a page, we wrote synthetic line items into the *empty ruled
rows of the real photographs* — inheriting real paper grain, camera noise,
uneven lighting, genuine perspective, and the photographer's hand at the frame
edge. 425 pages yielded 6,496 crops.

A quarter as much data as arm A, and it halved real-receipt digit error
(0.29% → 0.13%). Unseen-layout digit errors reached zero across 1,470 crops.

**The controlled comparison is the finding.** Arm A added 16,531 crops and moved
real accuracy 0.00%. Arm B added 6,496 and moved it. The difference was not
volume, and not augmentation strength — it was whether the pixels beneath the
ink were real.

---

## 4. Layout understanding

Recognition is only half the problem. Assigning each value to the right field is
the other, and it is the harder one: published field tests report handwriting
transcribed in the mid-80s to mid-90s while **field-level** accuracy drops to
around 65%.

The original parser hard-coded one booklet's column positions (0.21 / 0.62 /
0.77 of image width) and its printed header wording. Any other supplier's form
broke all of it at once.

The replacement discovers structure per document: cluster box centres to find
columns, then decide each column's role by **which assignment makes the
receipt's own arithmetic hold**. `qty × harga = jumlah` fails on every row at
once under a wrong assignment, so candidate assignments can be scored and the
best kept — no training, no template knowledge.

| column order | fixed geometry | discovered |
|---|---:|---:|
| `nama qty harga jumlah` | 0% | **100%** |
| `nama harga qty jumlah` | 0% | **100%** |
| `qty nama harga jumlah` *(the corpus layout)* | 97% | **97%** |
| `qty nama jumlah` *(no unit-price column)* | 100% | 86% |
| `no nama qty harga jumlah` | 0% | **94%** |
| **overall (200 documents)** | **37.0%** | **96.0%** |

Regenerate with `python ml/eval_layout_generalization.py`. The benchmark feeds
ground-truth boxes from the generator, so it isolates layout understanding from
recognition error.

**A flaw the benchmark exposed.** Multiplication is commutative, so
`qty × harga == harga × qty`: the arithmetic alone cannot distinguish the
quantity column from the unit-price column, and both assignments reconcile
perfectly. Breaking the tie on magnitude — quantities are small, prices are not
— raised overall accuracy from 46% to 95.5% at the time. The self-consistency signal is
powerful but not sufficient; it needed one weak prior to become decisive.

---

## 4b. Structure that survives a real photograph

The benchmark above uses clean generated geometry. Real phone photographs of a
nota break three further assumptions, each of which returned an **empty document**
to the backend rather than a wrong one. All three were found by running real
photos, and all three are measured.

**The page is rarely upright.** A hand-held nota shot sideways yields a vertical
sliver for every crop and an empty string for every prediction. Recovering the
angle from the detector's own quadrilaterals, then confirming upright-vs-inverted
with the recogniser's confidence:

| | before | after |
|---|---:|---:|
| lines that read | 24 | **34** |
| mean recognition confidence | 0.652 | **0.881** |

A single levelling pass is not enough: on a page lying at 86° the detector is at
its worst and its angle is ~2° out, which across a 3000-pixel page drags one end
of every row a full row-pitch from the other. Iterating drops the residual from
2.05° to 0.58°.

**Detected boxes are taller than the line spacing.** Median box height 74 px
against a 63 px row pitch, so every row overlaps its neighbours and a threshold
drawn from box height never fires — six item rows collapsed into one. Measuring
pitch down each column instead (one entry per row, so consecutive steps *are* the
pitch) recovers the table. Anchoring each row on its running centre rather than on
the gap to the previous box stops single-linkage chaining fusing four rows into
one.

**The detector welds the quantity cell onto the name cell.** `2 kg gula pasir`
arrives as a single text line, and by then no clustering can separate it. Peeling
the leading count off and scoring the document *both ways*, keeping whichever
makes the receipt add up better — measured on 120 generated nota with the
quantity deliberately welded into the name, 423 line items:

| | before | after |
|---|---:|---:|
| item names recovered exactly | 0 (0%) | **333 (78.7%)** |
| quantity also exactly right | 0 (0%) | **309 (73.0%)** |

Two false positives this split had to be taught to avoid: a thousands separator
(`67.500` is not a count of 67 and an item called `.500`) and a one-letter unit
(`g` for gram ate the first letter of every item beginning with one — `gorengan`
became a gram of `orengan`).

**None of this cost the corpus anything.** Across 40 held-out real receipts run
end to end through the service, rows reconciling went 89.6% → 89.3% — one line
item's difference.

## 5. Error analysis

The 19 remaining errors on real receipts are not label noise or illegible crops
(only 1 of 19 is a small crop). They are genuine handwriting confusions:

```
120000 → 12000     dropped zero, a 10x price error
45000  → 4500      dropped zero, a 10x price error
telur  → relur     t / r
maizena→ mai2ena   z / 2
melinjo→ metinjo   l / t
tepung → tepwng    u / w
```

These are *pen* ambiguities — how a stroke is formed. Font-rendered synthetic
text has canonical letterforms and cannot produce them, which explains directly
why arm A's rendered pages moved real accuracy 0.00%.

Errors concentrate on long words: 10 of 19 are 6–7 characters, consistent with
letters (1.08% CER) being far worse than digits (0.13%).

Two of the 19 are dropped-zero errors. Those are the worst possible failure for
an inventory system — and they are exactly what the arithmetic reconciliation
catches, because a dropped zero breaks `qty × harga = jumlah` every time.

---

## 6. Limitations

**The unseen-layout test set is synthetic.** It uses column orders never
trained on, but the same renderer, fonts and degradation. It measures
generalisation *within the generator's world*, not across real suppliers.
Establishing the latter requires photographs of a different shop's nota book.

**Real-receipt exact match moved 98.14% → 98.24%** — one crop in 1,023, inside
the noise. The CER and digit-CER improvements are more trustworthy because they
average over characters rather than counting whole-crop hits.

**No writer-held-out split.** The corpus has no writer labels, so validation
shares handwriting with training and flatters all three models. Labelling
writers is roughly an hour of work and would make these numbers honest.

**The synthetic ink is font-rendered.** Compositing made the *paper* real; the
*strokes* are not. A diffusion handwriting generator was integrated and runs,
but the released IAM-trained model produces label-accurate output only 50% of
the time on Indonesian words and long numerals — `15000` renders as `115000`.
Everything it generates is therefore gated by reading it back with the
recogniser and discarding mismatches, which is why it contributes little volume.

**Recognition on out-of-domain photographs is materially worse than the table
in section 3.** Those figures are measured on held-out crops from the same
corpus. On phone photographs of a different shop's nota book the recogniser
reads `gula pasir` as `ghtsparir` and 16000 as 18000. The structural work in
section 4b is what turns that from an empty response into pre-filled rows
flagged for correction; it does not make the reading correct.

---

## 7. Reproducing

Dataset preparation and training live in `notebooks/02_train_ocr.ipynb`, which is
self-contained — it downloads the corpus, builds the recognition crops with the
receipt-grouped split, fine-tunes and evaluates. The standalone scripts cover the
synthetic data and the layout benchmark:

```bash
python ml/synth_nota.py                 # rendered pages, layout-held-out split
python ml/composite_real.py --raw data/raw/train --out data/composited \
                            --fonts ml/fonts --catalogue ml/sku_catalogue.csv
python ml/eval_layout_generalization.py # the 37.0% -> 96.0% table
```

`SNAPTOK_FONT_DIR` points the generator at a directory of additional handwriting
faces; without it the synthetic pages inherit the corpus's two or three writers.

Training used PP-OCRv5 mobile at PaddleOCR `v3.7.0`, batch 64, Adam with cosine
annealing from 1e-4, 40 epochs. Both runs plateaued well before 40 — arm A best
at epoch 34, arm B at epoch 24 — so longer schedules would not have helped.
