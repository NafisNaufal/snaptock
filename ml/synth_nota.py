"""
Synthetic nota generator.

The corpus is 440 photographs of ONE pre-printed booklet, written by two or
three people. A model trained on it learns that booklet, which is why the
pipeline broke on a different supplier's form.

This renders nota with deliberately varied structure and returns exact ground
truth for free -- box coordinates, field roles, and values -- which is the data
no amount of annotation budget would buy.

Following SAYRE (arXiv:2607.04636), layout conventions are sampled from
observed exemplars rather than enumerated by hand, and the generator accepts
failure cases to expand into hard examples.

Ground truth per document:
    {"image": ndarray,
     "boxes": [{"x","y","w","h","text","role","row"}],
     "items": [{"nama","qty","harga","jumlah"}],
     "total": int,
     "layout": {...}}          <- the sampled layout, for held-out-template splits
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Column orders seen across Indonesian nota booklets. The corpus only ever
# shows the first; the rest are why the model must not memorise positions.
COLUMN_ORDERS = [
    ["qty", "nama", "harga", "jumlah"],
    ["nama", "qty", "harga", "jumlah"],
    ["qty", "nama", "jumlah"],              # no unit-price column
    ["nama", "harga", "qty", "jumlah"],
    ["no", "nama", "qty", "harga", "jumlah"],
]

HEADER_WORDS = {
    "qty":    ["BANYAKNYA", "QTY", "JML", "BANYAK", "JUMLAH BRG"],
    "nama":   ["NAMA BARANG", "URAIAN", "KETERANGAN", "NAMA ITEM"],
    "harga":  ["HARGA", "HARGA SATUAN", "@", "H. SATUAN"],
    "jumlah": ["JUMLAH", "TOTAL", "JML HARGA"],
    "no":     ["NO", "NO.", "#"],
}

TOTAL_LABELS = ["Jumlah Rp.", "TOTAL", "Total Rp.", "Jumlah", "JUMLAH Rp"]
PAPER_TINTS = [(255, 255, 255), (232, 240, 247), (247, 243, 228),
               (245, 232, 235), (236, 246, 236)]


@dataclass
class Layout:
    order: list
    widths: list
    header_text: dict
    total_label: str
    tint: tuple
    ruled: bool
    n_rows: int
    header_y: float
    row_h: float

    def role_at(self, i):
        return self.order[i]


def sample_layout(rng: random.Random) -> Layout:
    order = rng.choice(COLUMN_ORDERS)
    # Column widths: name column always widest, rest jittered.
    weights = []
    for role in order:
        base = {"nama": 3.2, "qty": 0.8, "harga": 1.3, "jumlah": 1.4, "no": 0.5}[role]
        weights.append(base * rng.uniform(0.8, 1.25))
    total = sum(weights)
    widths = [w / total for w in weights]
    return Layout(
        order=order,
        widths=widths,
        header_text={r: rng.choice(HEADER_WORDS[r]) for r in order},
        total_label=rng.choice(TOTAL_LABELS),
        tint=rng.choice(PAPER_TINTS),
        ruled=rng.random() < 0.75,
        n_rows=rng.randint(8, 16),
        header_y=rng.uniform(0.16, 0.28),
        row_h=rng.uniform(0.038, 0.055),
    )


def load_vocabulary(path=None):
    """(product, unit_price) pairs. Defaults to items recovered from the corpus."""
    if path:
        import csv
        with open(path) as fh:
            return [(r["product_name"], int(r["unit_price"])) for r in csv.DictReader(fh)]
    return [("es teh", 5000), ("air es", 2000), ("teh hangat", 5000),
            ("gorengan", 2000), ("kopi", 2000), ("es jeruk", 7000),
            ("risoles", 2500), ("nasi goreng", 15000), ("snack", 15000),
            ("gula", 17000), ("garam", 6000), ("pulpen", 5000),
            ("nasi kotak", 25000), ("buku gambar", 12000), ("hvs a4 sidu", 55000),
            ("indomie", 3000), ("minyak goreng", 35000), ("beras 5kg", 68000),
            ("telur 1kg", 28000), ("sabun cuci", 12000)]


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(layout: Layout, vocabulary, rng: random.Random,
           size=(1000, 1400), realistic=True) -> dict:
    """Draw one nota and return it with exact ground truth."""
    W, H = size
    img = Image.new("RGB", size, layout.tint)
    draw = ImageDraw.Draw(img)
    # One writer per nota, one printer per form. Re-picking a face per cell
    # would teach the recogniser that handwriting changes mid-document.
    hand_path = rng.choice(available(HAND_FONTS)) if realistic and available(HAND_FONTS) else None
    print_path = rng.choice(available(PRINT_FONTS)) if realistic and available(PRINT_FONTS) else None

    def face(path, size):
        if not path:
            return _font(size)
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            return _font(size)

    head_font = face(print_path, int(H * 0.019))
    cell_font = face(hand_path, int(H * 0.024))

    # column x-boundaries
    edges, x = [0.05], 0.05
    for w in layout.widths:
        x += w * 0.90
        edges.append(x)
    boxes = []

    def fit(text, font, max_w, path):
        """Shrink until the text fits its cell, keeping the same face.
        Overlapping text would teach the recogniser to read two words as one."""
        size = getattr(font, "size", 20)
        while size > 8:
            trial = face(path, size)
            if draw.textlength(str(text), font=trial) <= max_w:
                return trial
            size -= 1
        return face(path, 8)

    def put(text, cx, cy, role, row, font, anchor="mm", max_w=None, path=None):
        if max_w:
            font = fit(text, font, max_w, path)
        bbox = draw.textbbox((cx, cy), str(text), font=font, anchor=anchor)
        draw.text((cx, cy), str(text), fill=(20, 20, 30), font=font, anchor=anchor)
        boxes.append({"x": bbox[0], "y": bbox[1], "w": bbox[2] - bbox[0],
                      "h": bbox[3] - bbox[1], "text": str(text),
                      "role": role, "row": row})

    hy = layout.header_y * H
    row_h = layout.row_h * H

    if layout.ruled:
        for i in range(layout.n_rows + 2):
            y = hy + i * row_h
            draw.line([(edges[0] * W, y), (edges[-1] * W, y)], fill=(90, 90, 90), width=1)
        for e in edges:
            draw.line([(e * W, hy), (e * W, hy + (layout.n_rows + 1) * row_h)],
                      fill=(90, 90, 90), width=1)

    for i, role in enumerate(layout.order):
        cx = (edges[i] + edges[i + 1]) / 2 * W
        cell_w = (edges[i + 1] - edges[i]) * W * 0.92
        put(layout.header_text[role], cx, hy + row_h / 2, f"header_{role}", -1,
            head_font, max_w=cell_w, path=print_path)

    n_items = rng.randint(1, min(6, layout.n_rows - 1))
    items, running = [], 0
    for r in range(n_items):
        nama, base = rng.choice(vocabulary)
        harga = int(base * rng.uniform(0.85, 1.2) // 500 * 500) or base
        qty = rng.choice([1, 1, 2, 2, 3, 5, 10, 12, 20])
        jumlah = qty * harga
        running += jumlah
        items.append({"nama": nama, "qty": qty, "harga": harga, "jumlah": jumlah})
        cy = hy + (r + 1) * row_h + row_h / 2
        values = {"no": r + 1, "nama": nama, "qty": qty,
                  "harga": f"{harga:,}".replace(",", "."), "jumlah": f"{jumlah:,}".replace(",", ".")}
        for i, role in enumerate(layout.order):
            cx = (edges[i] + edges[i + 1]) / 2 * W
            cell_w = (edges[i + 1] - edges[i]) * W * 0.88
            put(values[role], cx, cy, role, r, cell_font,
                max_w=cell_w, path=hand_path)

    # Total row: label sits in the second-to-last column, value in the last.
    ty = hy + (layout.n_rows + 1) * row_h + row_h * 0.7
    label_w = (edges[-2] - edges[-3]) * W * 0.9 if len(edges) >= 3 else W * 0.2
    put(layout.total_label, (edges[-2] - 0.012) * W, ty, "total_label", -2,
        head_font, anchor="rm", max_w=label_w,
        path=print_path)
    put(f"{running:,}".replace(",", "."), (edges[-1] + edges[-2]) / 2 * W, ty,
        "total_value", -2, cell_font, anchor="mm",
        max_w=(edges[-1] - edges[-2]) * W * 0.88,
        path=hand_path)

    page = np.array(img)
    if realistic:
        page, M = degrade(page, rng)
        boxes = warp_boxes(boxes, M, W, H)
    return {"image": page, "boxes": boxes, "items": items,
            "total": running, "layout": layout}


# The form is printed; the entries are handwritten. Rendering both with one
# font is the single most unrealistic thing a naive generator does.
HAND_FONTS = [
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    "/System/Library/Fonts/Supplemental/Brush Script.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
]
PRINT_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def available(paths):
    import os
    return [p for p in paths if os.path.exists(p)]


def pick_font(paths, size, rng):
    found = available(paths)
    if not found:
        return _font(size)
    try:
        return ImageFont.truetype(rng.choice(found), size)
    except OSError:
        return _font(size)


def degrade(img: np.ndarray, rng: random.Random):
    """Phone-camera reality: warp, uneven light, blur, sensor noise, JPEG.

    The corpus is flat, sharp and evenly lit, which is exactly what a real
    submission is not. Without this the model learns a studio distribution.
    """
    import cv2

    h, w = img.shape[:2]

    # perspective: photographing a page you are not directly above
    j = 0.02 + rng.random() * 0.05
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[rng.uniform(0, w*j), rng.uniform(0, h*j)],
                      [w - rng.uniform(0, w*j), rng.uniform(0, h*j)],
                      [w - rng.uniform(0, w*j), h - rng.uniform(0, h*j)],
                      [rng.uniform(0, w*j), h - rng.uniform(0, h*j)]])
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (w, h), borderValue=(245, 245, 245))

    # uneven illumination / shadow across the page
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    shade = 1.0 - rng.uniform(0.05, 0.35) * (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6)
    img = np.clip(img.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    if rng.random() < 0.7:
        k = rng.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), rng.uniform(0.4, 1.4))

    img = np.clip(img.astype(np.int16) +
                  rng.uniform(2, 9) * np.random.randn(h, w, 3), 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY,
                                         int(rng.uniform(45, 92))])
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    return out, M


def warp_boxes(boxes, M, W, H, pad=0.10, min_side=8):
    """Move ground-truth boxes through the same perspective the page went
    through. Cropping warped pixels with unwarped coordinates is silent
    mislabelling -- the crop and its label stop matching."""
    import cv2
    out = []
    for b in boxes:
        corners = np.float32([[[b["x"], b["y"]], [b["x"] + b["w"], b["y"]],
                               [b["x"] + b["w"], b["y"] + b["h"]],
                               [b["x"], b["y"] + b["h"]]]])
        moved = cv2.perspectiveTransform(corners, M)[0]
        xs, ys = moved[:, 0], moved[:, 1]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        # ink-tight boxes clip ascenders and can be a pixel tall; pad them out
        px, py = (x1 - x0) * pad, max((y1 - y0) * pad, 3)
        x0, y0 = max(0, x0 - px), max(0, y0 - py)
        x1, y1 = min(W, x1 + px), min(H, y1 + py)
        if x1 - x0 < min_side or y1 - y0 < min_side:
            continue
        out.append({**b, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return out


def build_dataset(out_dir, n=2000, seed=0, vocabulary_csv=None, size=(1000, 1400)):
    """Write a PaddleOCR-format recognition dataset plus detection labels.

    Splits by LAYOUT, not by document: the held-out set uses column orders the
    model never trained on. A random split would let it memorise every layout
    and report a number that means nothing for a new supplier's booklet.
    """
    import json
    from pathlib import Path
    import cv2

    out = Path(out_dir)
    rng = random.Random(seed)
    vocab = load_vocabulary(vocabulary_csv)

    held_out = {tuple(COLUMN_ORDERS[-1]), tuple(COLUMN_ORDERS[-2])}
    counts = {"train": 0, "test": 0}
    rec_lines = {"train": [], "test": []}
    det_lines = {"train": [], "test": []}

    for split in ("train", "test"):
        (out / split / "crops").mkdir(parents=True, exist_ok=True)
        (out / split / "pages").mkdir(parents=True, exist_ok=True)

    made = 0
    while made < n:
        layout = sample_layout(rng)
        split = "test" if tuple(layout.order) in held_out else "train"
        doc = render(layout, vocab, rng, size=size)
        page, idx = doc["image"], counts[split]
        name = f"{idx:06d}.jpg"
        cv2.imwrite(str(out / split / "pages" / name), page[:, :, ::-1])

        polys = []
        for k, b in enumerate(doc["boxes"]):
            x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            if w < 4 or h < 4:
                continue
            crop = page[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            crop_name = f"{idx:06d}_{k}.jpg"
            cv2.imwrite(str(out / split / "crops" / crop_name), crop[:, :, ::-1])
            rec_lines[split].append(f"crops/{crop_name}\t{b['text']}")
            polys.append({"transcription": b["text"],
                          "points": [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]})
        det_lines[split].append(f"pages/{name}\t{json.dumps(polys, ensure_ascii=False)}")
        counts[split] += 1
        made += 1

    for split in ("train", "test"):
        (out / f"{split}_rec.txt").write_text("\n".join(rec_lines[split]) + "\n")
        (out / f"{split}_det.txt").write_text("\n".join(det_lines[split]) + "\n")

    charset = sorted({c for line in rec_lines["train"] + rec_lines["test"]
                      for c in line.split("\t")[1] if not c.isspace()})
    (out / "dict.txt").write_text("\n".join(charset) + "\n")

    meta = {"pages": counts, "held_out_layouts": [list(o) for o in held_out],
            "crops": {k: len(v) for k, v in rec_lines.items()},
            "charset_size": len(charset)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
