# Design decisions

Every non-obvious choice in the OCR workstream, the alternative it was chosen
over, and the measurement that decided it. This is the document to draw on when
writing or defending the proposal.

---

## 1. CTC recognition, not a sequence-to-sequence transformer or a document VLM

**The alternative.** TrOCR-style encoder–decoder models and document VLMs
(Donut, Qwen-VL) top the handwriting leaderboards and can read a whole receipt
in one shot.

**Why not.** Three reasons, in order of weight:

1. **A generative decoder can hallucinate a value that is not on the page.** For
   an inventory system that is not a cosmetic failure — an invented price is
   written into stock and silently corrupts every downstream forecast. CTC emits
   one symbol per input frame; it has no language-model prior to invent from.
2. **Data.** The corpus is 440 photographs. A seq2seq decoder needs far more
   before it stops memorising.
3. **Deployment.** UMKM-facing means CPU-cheap. CTC over a PP-LCNet backbone
   quantises to INT8 and runs on commodity hardware; a VLM does not.

The citations behind this are in [`run-a-research.html`](run-a-research.html).

## 2. Two stages, not end-to-end

Detection and recognition are separate models. That keeps the detector stock and
untrained, and it means the geometry between them — orientation, rectification,
the table grid — is available as ordinary data rather than being buried inside a
network. Every structural fix in this repository lives in that gap. An end-to-end
model would have had to be retrained for each of them.

## 3. Column roles from the receipt's arithmetic, not from column position

**The alternative, and what shipped first.** The nota booklet is a fixed printed
template, so hard-code the column positions: quantity at 0.02–0.21 of image
width, name at 0.21–0.62, price at 0.62–0.77, total at 0.77–0.97.

**Why it broke.** It works on one supplier's booklet and fails completely on any
other, all at once. This was the first thing a teammate hit in integration.

**The replacement.** Discover columns by clustering, then decide each column's
role by **which assignment makes `qty × harga = jumlah` hold across the rows**. A
wrong assignment fails the identity on every row simultaneously, so candidates
can be scored and the best kept. No training, no template knowledge.

**Evidence** — 200 generated nota, column orders the model never trained on:

| column order | fixed geometry | discovered |
|---|---:|---:|
| `nama qty harga jumlah` | 0% | **100%** |
| `nama harga qty jumlah` | 0% | **100%** |
| `qty nama harga jumlah` *(the corpus layout)* | 97% | **97%** |
| `qty nama jumlah` *(no unit-price column)* | 100% | 86% |
| `no nama qty harga jumlah` | 0% | **94%** |
| **overall** | **37.0%** | **96.0%** |

Reproduce with `python ml/eval_layout_generalization.py`.

**The flaw the benchmark exposed, and the fix.** Multiplication is commutative,
so `qty × harga == harga × qty`: the arithmetic alone cannot tell the quantity
column from the unit-price column, and both assignments reconcile perfectly.
Breaking the tie on magnitude — quantities are small, prices are not — raised
accuracy from 46% to 95.5% at the time. **Self-consistency is powerful but not
sufficient; it needed one weak prior to become decisive.** That is the sharpest
single finding in this workstream.

## 4. Synthetic ink composited onto real photographs, not rendered pages

**The controlled comparison.** Same architecture, same held-out sets, one variable.

| model | training data | crops | real CER | real digits | unseen CER |
|---|---|---:|---:|---:|---:|
| baseline | real only | 12,652 | 0.60% | 0.21% | 3.87% |
| **A** | + rendered pages | 29,183 | 0.61% | 0.29% | **0.17%** |
| **B** | + composited on real photos | 35,679 | **0.48%** | **0.13%** | **0.06%** |

**Arm A added 16,531 crops of programmatically drawn nota and moved
real-receipt accuracy by exactly zero.** It fixed layout generalisation and
nothing else. The model learned the renderer's world: its fonts, its flat paper
tints, its gentle warp.

**Arm B wrote synthetic line items into the empty ruled rows of the real
photographs** — inheriting real paper grain, camera noise, uneven lighting,
genuine perspective and the photographer's hand at the frame edge. A quarter as
much data as arm A, and it halved real-receipt digit error.

**The difference was not volume and not augmentation strength. It was whether
the pixels beneath the ink were real.** The error analysis says why directly: the
19 remaining errors are *pen* ambiguities — `telur → relur`, `melinjo → metinjo`,
`tepung → tepwng` — how a stroke is formed. Font-rendered text has canonical
letterforms and cannot produce them.

## 5. Split on the source receipt, never on `image_id`

**What went wrong.** Roboflow emits ~3 brightness-augmented copies of every
photograph, each with its own COCO `image_id`. Splitting on that puts copies of
the same receipt in train, validation *and* test.

| partition | crops | distinct nota | also in train | contaminated crops |
|---|---:|---:|---:|---:|
| validation | 1,793 | 118 | 115 | **96.1%** |
| test | 1,822 | 116 | 113 | **97.1%** |

The genuinely unseen evaluation set was **three receipts**. Every accuracy figure
produced before this was corrected is meaningless.

Splits now group by the receipt id parsed from the filename, and preparation
asserts the partitions are disjoint before training may start. Full forensics in
[`dataset-audit.html`](dataset-audit.html).

## 6. Orientation recovered from the detector's own geometry, not a second model

**The alternative.** PaddleOCR ships document-orientation and text-line
orientation classifiers. Adding them would have worked.

**Why not.** The detector already returns one quadrilateral per text line, and
text lines run along the writing direction — so the average edge angle of those
quads *is* the page angle. That is two more models' worth of behaviour for no
extra weights, no extra download, and no extra failure mode. The 180° ambiguity
that geometry genuinely cannot resolve is settled by asking the recogniser which
way round it reads with more confidence.

**A related decision made by measurement and then rejected.** Phone photos arrive
around 3000 px on the long side, and the detector shrinks them to 960 before
looking, which closes the gaps between handwritten cells. Raising that limit
visibly helped three sample photos — and cost the 40-receipt corpus:

| detection resolution | rows reconciling |
|---|---:|
| **960 (default)** | **89.3%** |
| 1280 | 81.0% |
| 1600 | 84.8% |

Three photos do not outvote forty receipts. The default stands, and `DET_SIDE_LEN`
is exposed so it can be re-tuned against a larger corpus later.

## 7. Never repair silently

`qty × harga = jumlah` has many solutions. A repair that restores the arithmetic
while inventing a price is worse than no repair, because it is indistinguishable
from a correct reading downstream.

So the service reports rather than fixes. A row that does not reconcile comes back
with `reconciled: false` and the discrepancy spelled out. A value that cannot be
read comes back `null` — **and `null` is not zero**. The grand total is `null`
rather than guessed, and is not looked for at all when no items were found, or the
nota number at the top of the page gets reported as the receipt's total.

The backend branches on one boolean, `needs_review`. Thresholds live inside the
service and will move; the boolean will not.

**This is also what makes the arithmetic worth having.** Two of the 19 residual
recognition errors are dropped zeros — `120000 → 12000`, a 10× price error, the
worst possible failure for an inventory system. A dropped zero breaks
`qty × harga = jumlah` every time, so the reconciliation catches exactly the
errors that matter most.

## 8. Row pitch measured down the columns, not inferred from box height

The detector pads each detected line generously. On a real nota the boxes come
out **taller than the line spacing** — median height 74 px against a 63 px row
pitch — so every row overlaps its neighbours and any threshold drawn from box
height never fires. The whole table collapses into one row and the service returns
an empty document.

A column holds exactly one entry per row, so the step between consecutive entries
down a column *is* the row pitch. Measuring it there is scale-free and survives
the detector's padding.

Clustering is anchored on each group's running centre rather than on the gap to
the previous value, because single-linkage chaining fuses rows: every neighbouring
pair sits under the threshold while the ends of the chain lie two pitches apart.

## 9. What is deliberately *not* learned

Only recognition is a trained model. Detection is stock. Orientation, line
rectification, the table grid, column roles and reconciliation are all geometry
and arithmetic.

That is not a shortcut. It is what lets a different supplier's nota work without
recalibration, it is what makes every failure inspectable rather than opaque, and
it is why the fixes above were possible at all — each one is a few lines in a
readable module rather than a retraining run.
