"""
Build a PaddleOCR recognition dataset from the nota-pembelian COCO export.

Replaces the notebook's cells 19-24. The notebook split on COCO `image_id`,
but Roboflow generated ~3 brightness-augmented copies of every source receipt,
each with its own image_id -- so copies of the same nota landed in train, val
AND test. Measured contamination was 96.1% of val crops and 97.1% of test.
See docs/dataset-audit.html.

This script fixes that and two other things:

  1. Splits on the SOURCE receipt id parsed out of the Roboflow filename
     (`nota203_jpg.rf.<hash>.jpg` -> `nota203`), then asserts disjointness.
  2. Keeps all augmented copies in TRAIN (legitimate augmentation) but uses
     exactly one copy per receipt in VAL/TEST, so evaluation isn't three
     near-identical votes on the same image.
  3. Builds the character dictionary from ALL category names, not just those
     appearing in train -- a character absent from the dict is unlearnable,
     and the long tail means many appear only in val/test.

Usage:
    python ml/prepare_dataset.py --coco data/raw/train/_annotations.coco.json \
                                 --images data/raw/train \
                                 --out data/rec
"""

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image

SOURCE_RE = re.compile(r"^(.*?)_jpg\.rf\.")
PLACEHOLDER_CATEGORIES = {"objects", "object"}


def source_id(file_name):
    """nota203_jpg.rf.93fc05d4...jpg -> nota203"""
    match = SOURCE_RE.match(file_name)
    return match.group(1) if match else file_name


def crop_with_padding(image, bbox, padding_ratio=0.05):
    x, y, w, h = bbox
    pad_x, pad_y = w * padding_ratio, h * padding_ratio
    return image.crop((
        max(0, int(x - pad_x)),
        max(0, int(y - pad_y)),
        min(image.width, int(x + w + pad_x)),
        min(image.height, int(y + h + pad_y)),
    ))


def load_writer_map(path):
    """Optional CSV: source_id,writer_id -- one row per receipt."""
    mapping = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("source"):
            continue
        source, writer = (part.strip() for part in line.split(",")[:2])
        mapping[source] = writer
    return mapping


def split_sources(sources, writer_map, val_frac, test_frac, seed):
    """
    Group-aware split. Groups by writer when a writer map is supplied --
    handwriting style, not receipt identity, is what breaks in deployment
    (Garrido-Munoz & Calvo-Zaragoza, CVPR 2025). Falls back to grouping by
    receipt, which is still correct, just a weaker generalisation estimate.
    """
    rng = random.Random(seed)

    if writer_map:
        groups = collections.defaultdict(list)
        for source in sources:
            groups[writer_map.get(source, f"__unmapped__{source}")].append(source)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_frac))
        n_test = max(1, round(len(keys) * test_frac))
        buckets = {
            "val": keys[:n_val],
            "test": keys[n_val:n_val + n_test],
            "train": keys[n_val + n_test:],
        }
        return {name: {s for k in ks for s in groups[k]} for name, ks in buckets.items()}

    ordered = sorted(sources)
    rng.shuffle(ordered)
    n_val = round(len(ordered) * val_frac)
    n_test = round(len(ordered) * test_frac)
    return {
        "val": set(ordered[:n_val]),
        "test": set(ordered[n_val:n_val + n_test]),
        "train": set(ordered[n_val + n_test:]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--writer-map", type=Path, default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--padding", type=float, default=0.05)
    ap.add_argument("--min-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco = json.loads(args.coco.read_text())
    images = {img["id"]: img for img in coco["images"]}
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    writer_map = load_writer_map(args.writer_map) if args.writer_map else {}
    if not writer_map:
        print("NOTE: no --writer-map given. Splitting by receipt, which is correct\n"
              "      but flatters the model: the same 2-3 hands appear in every split.\n"
              "      Label writers in a CSV (source_id,writer_id) for the honest number.\n")

    # ---- character dictionary from EVERY category, not just train ----
    charset = set()
    for name in categories.values():
        if name not in PLACEHOLDER_CATEGORIES:
            charset.update(ch for ch in str(name) if not ch.isspace())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dict.txt").write_text("\n".join(sorted(charset)) + "\n", encoding="utf-8")

    # ---- split on source receipt ----
    by_source = collections.defaultdict(list)
    for img in coco["images"]:
        by_source[source_id(img["file_name"])].append(img)

    splits = split_sources(list(by_source), writer_map,
                           args.val_frac, args.test_frac, args.seed)

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = splits[a] & splits[b]
        if overlap:
            sys.exit(f"FATAL: {len(overlap)} receipts in both {a} and {b}: "
                     f"{sorted(overlap)[:5]}")

    # Train keeps every augmented copy; val/test keep exactly one per receipt.
    selected = {}
    for split, sources in splits.items():
        picked = []
        for source in sorted(sources):
            copies = sorted(by_source[source], key=lambda i: i["file_name"])
            picked.extend(copies if split == "train" else copies[:1])
        selected[split] = picked

    annotations = collections.defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)

    # ---- crop ----
    counts, skipped = collections.Counter(), collections.Counter()
    for split, imgs in selected.items():
        (args.out / split).mkdir(parents=True, exist_ok=True)
        records = []

        for img in imgs:
            path = args.images / img["file_name"]
            if not path.exists():
                skipped["missing_image"] += 1
                continue
            with Image.open(path) as handle:
                page = handle.convert("RGB")
                for ann in annotations[img["id"]]:
                    text = str(categories.get(ann["category_id"], "")).strip()
                    if not text or text in PLACEHOLDER_CATEGORIES:
                        skipped["placeholder_or_empty"] += 1
                        continue
                    crop = crop_with_padding(page, ann["bbox"], args.padding)
                    if crop.width < args.min_size or crop.height < args.min_size:
                        skipped["degenerate_crop"] += 1
                        continue
                    name = f"{img['id']}_{ann['id']}.jpg"
                    crop.save(args.out / split / name, quality=95)
                    records.append((f"{split}/{name}", text))

        label_file = args.out / f"{split}_rec.txt"
        with label_file.open("w", encoding="utf-8") as fh:
            for rel, text in records:
                fh.write(f"{rel}\t{text}\n")
        counts[split] = len(records)

    manifest = {
        "seed": args.seed,
        "grouped_by": "writer" if writer_map else "source_receipt",
        "receipts": {s: sorted(v) for s, v in splits.items()},
        "crops": dict(counts),
        "charset_size": len(charset),
    }
    (args.out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"grouping        : {manifest['grouped_by']}")
    print(f"character dict  : {len(charset)} chars -> {args.out/'dict.txt'}")
    if any(" " in str(n) for n in categories.values()):
        print("                  transcriptions contain spaces; set "
              "use_space_char: true in the PaddleOCR rec config")
    for split in ("train", "val", "test"):
        print(f"{split:<8} {len(splits[split]):>4} receipts  "
              f"{len(selected[split]):>4} images  {counts[split]:>6} crops")
    if skipped:
        print("skipped         :", dict(skipped))
    print(f"\nmanifest -> {args.out/'split_manifest.json'}")


if __name__ == "__main__":
    main()
