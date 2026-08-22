"""
Detected text boxes -> validated line items.

The Indonesian nota booklet is a fixed printed template, so the form tells us
where the table is. We locate two printed landmarks and read only between them:

    BANYAKNYA | NAMA BARANG | HARGA | JUMLAH     <- header, items start below
    ...handwritten rows...
    Jumlah Rp.                     40000        <- footer, grand total

Column boundaries are fractions of image width, measured across the corpus
(see docs/dataset-audit.html). This holds for this booklet and fails for other
suppliers' layouts -- a deliberate, declared limit of the MVP.
"""

from __future__ import annotations

import re

COLUMNS = [("qty", 0.02, 0.21), ("nama", 0.21, 0.62),
           ("harga", 0.62, 0.77), ("jumlah", 0.77, 0.97)]

ROW_BAND = 0.025          # rows cluster into bands of 2.5% of image height
MIN_CONFIDENCE = 0.70     # below this, ask a human to confirm the row

HEADER_WORDS = {"banyaknya", "namabarang", "harga", "jumlah"}
TOTAL_WORDS = {"jumlahrp", "jumlahrp."}


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def to_int(text: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def column_of(cx: float, width: int) -> str | None:
    frac = cx / width
    for name, lo, hi in COLUMNS:
        if lo <= frac < hi:
            return name
    return None


def to_bands(boxes, width: int, height: int) -> dict[int, dict]:
    """Group boxes into {band: {column: [box, ...]}}."""
    bands: dict[int, dict] = {}
    for b in boxes:
        col = column_of(b["x"] + b["w"] / 2, width)
        if col:
            band = int((b["y"] + b["h"] / 2) // (height * ROW_BAND))
            bands.setdefault(band, {}).setdefault(col, []).append(b)
    return bands


def find_landmarks(bands) -> tuple[int, int]:
    """Band index of the printed column header, and of the 'Jumlah Rp.' row."""
    header, total = -1, 10 ** 6
    for band, cols in bands.items():
        words = {norm(b["text"]) for boxes in cols.values() for b in boxes}
        if len(words & HEADER_WORDS) >= 2:
            header = max(header, band)
        if words & TOTAL_WORDS:
            total = min(total, band)
    return header, total


def cell_text(cols, name) -> str:
    boxes = sorted(cols.get(name, []), key=lambda b: b["x"])
    return " ".join(b["text"] for b in boxes)


def reconcile(qty, harga, jumlah) -> tuple[bool, list[str]]:
    """
    The receipt carries a redundant third value, so it checks itself. Holds on
    98.1% of line items in the corpus.

    We never repair silently: qty x harga = jumlah has many solutions, and one
    that restores the arithmetic while inventing a price is worse than none.
    """
    if None in (qty, harga, jumlah):
        return False, ["incomplete_row"]
    if qty * harga == jumlah:
        return True, []
    return False, [f"arithmetic_mismatch: {qty} x {harga} = {qty * harga}, "
                   f"receipt says {jumlah}"]


def assemble(boxes, width: int, height: int) -> dict:
    bands = to_bands(boxes, width, height)
    header, total_band = find_landmarks(bands)

    items, warnings = [], []
    for band in sorted(bands):
        if band <= header or band >= total_band:
            continue                                  # outside the table
        cols = bands[band]
        nama = cell_text(cols, "nama").strip()
        qty = to_int(cell_text(cols, "qty"))
        harga = to_int(cell_text(cols, "harga"))
        jumlah = to_int(cell_text(cols, "jumlah"))

        if not nama or harga is None:
            continue                                  # blank row on the form

        confidence = min(b["conf"] for bs in cols.values() for b in bs)
        reconciled, item_warnings = reconcile(qty, harga, jumlah)
        if confidence < MIN_CONFIDENCE:
            item_warnings.append("low_confidence")

        items.append({"nama": nama, "qty": qty, "harga": harga, "jumlah": jumlah,
                      "confidence": round(confidence, 3),
                      "reconciled": reconciled, "warnings": item_warnings})

    # Grand total: the number in the JUMLAH column of the "Jumlah Rp." row.
    stated = None
    if total_band in bands:
        stated = to_int(cell_text(bands[total_band], "jumlah"))

    computed = sum(i["jumlah"] for i in items if i["jumlah"] is not None)

    if not items:
        warnings.append("no_line_items_found")
    if header < 0:
        warnings.append("table_header_not_found")
    if stated is None:
        # Never invent it. Unknown is not zero.
        warnings.append("grand_total_not_found")
        matches = None
    else:
        matches = computed == stated
        if not matches:
            warnings.append(f"total_mismatch: rows sum to {computed}, "
                            f"receipt says {stated}")

    needs_review = (not items or matches is not True
                    or any(not i["reconciled"] or i["warnings"] for i in items))

    return {"items": items,
            "total": {"computed": computed, "stated": stated, "matches": matches},
            "warnings": warnings,
            "needs_review": needs_review}
