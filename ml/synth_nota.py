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
           size=(1000, 1400)) -> dict:
    """Draw one nota and return it with exact ground truth."""
    W, H = size
    img = Image.new("RGB", size, layout.tint)
    draw = ImageDraw.Draw(img)
    head_font, cell_font = _font(int(H * 0.019)), _font(int(H * 0.023))

    # column x-boundaries
    edges, x = [0.05], 0.05
    for w in layout.widths:
        x += w * 0.90
        edges.append(x)
    boxes = []

    def put(text, cx, cy, role, row, font, anchor="mm"):
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
        put(layout.header_text[role], cx, hy + row_h / 2, f"header_{role}", -1, head_font)

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
            put(values[role], cx, cy, role, r, cell_font)

    ty = hy + (layout.n_rows + 1) * row_h + row_h * 0.6
    put(layout.total_label, edges[-2] * W, ty, "total_label", -2, cell_font, anchor="lm")
    put(f"{running:,}".replace(",", "."), (edges[-1] - 0.01) * W, ty,
        "total_value", -2, cell_font, anchor="rm")

    return {"image": np.array(img), "boxes": boxes, "items": items,
            "total": running, "layout": layout}
