"""
Nota photo -> validated line items.

Five stages. Only stage 3 is a model we trained; the rest is geometry and
arithmetic, which is why the service is mostly not machine learning:

    1. detection        pretrained PP-OCR det finds word boxes
    2. crop             each box, 5% padding (matches training)
    3. recognition      our fine-tuned PP-OCRv5 rec reads each crop
    4. layout           x-position -> field, y-position -> row
    5. reconciliation   qty x harga == jumlah ?

Stage 4 works because the Indonesian nota booklet is a fixed printed template:
the four ruled columns sit at the same fractions of image width on every
receipt. Column boundaries were measured over the corpus (docs/dataset-audit.html).
That assumption holds for this booklet and fails for other suppliers' layouts --
a deliberate, declared limit of the MVP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

# Column boundaries as fractions of image width, measured on the corpus.
COLUMNS = [("qty", 0.02, 0.21), ("nama", 0.21, 0.62),
           ("harga", 0.62, 0.77), ("jumlah", 0.77, 0.97)]

ROW_BAND = 0.025          # rows cluster into bands of 2.5% of image height
MIN_CONFIDENCE = 0.70     # below this a field is flagged for human confirmation


@dataclass
class LineItem:
    nama: str
    qty: int | None
    harga: int | None
    jumlah: int | None
    confidence: float
    reconciled: bool
    warnings: list[str] = field(default_factory=list)


def column_of(cx: float, width: int) -> str | None:
    f = cx / width
    for name, lo, hi in COLUMNS:
        if lo <= f < hi:
            return name
    return None


def to_int(text: str) -> int | None:
    """Receipt numbers: strip thousands separators, reject anything else."""
    cleaned = text.replace(".", "").replace(",", "").replace(" ", "").strip()
    return int(cleaned) if cleaned.isdigit() else None


def group_rows(boxes, width: int, height: int) -> list[dict]:
    """boxes: [{'text','conf','x','y','w','h'}] -> one dict per table row."""
    rows: dict[int, dict] = {}
    for b in boxes:
        col = column_of(b["x"] + b["w"] / 2, width)
        if col is None:
            continue
        band = int((b["y"] + b["h"] / 2) // (height * ROW_BAND))
        rows.setdefault(band, {}).setdefault(col, []).append(b)
    return [rows[k] for k in sorted(rows)]


def reconcile(qty, harga, jumlah) -> tuple[bool, list[str]]:
    """
    The receipt carries a redundant third value, so it checks itself. Holds on
    98.1% of line items in the corpus.

    We deliberately do NOT repair silently: qty x harga = jumlah is
    under-determined, and a repair that restores the arithmetic while inventing
    a wrong price is worse than no repair at all. Flag it, let a human decide.
    """
    if None in (qty, harga, jumlah):
        return False, ["incomplete_row"]
    if qty * harga == jumlah:
        return True, []
    return False, [f"arithmetic_mismatch: {qty} x {harga} = {qty * harga}, "
                   f"receipt says {jumlah}"]


def assemble(boxes, width: int, height: int) -> dict:
    """Turn recognized boxes into the API response body."""
    items, warnings = [], []
    grand_total = None

    for cells in group_rows(boxes, width, height):
        def joined(col):
            parts = sorted(cells.get(col, []), key=lambda b: b["x"])
            return " ".join(p["text"] for p in parts), parts

        nama, nama_parts = joined("nama")
        qty_s, qty_parts = joined("qty")
        harga_s, harga_parts = joined("harga")
        jumlah_s, jumlah_parts = joined("jumlah")

        qty, harga, jumlah = to_int(qty_s), to_int(harga_s), to_int(jumlah_s)

        if not nama and qty is None and harga is None:
            # A jumlah with no item beside it is the "Jumlah Rp." grand total
            # printed under the table. Keep the lowest such row on the page.
            if jumlah is not None:
                grand_total = jumlah
            continue                                   # otherwise a blank form row

        parts = nama_parts + qty_parts + harga_parts + jumlah_parts
        conf = min((p["conf"] for p in parts), default=0.0)
        ok, item_warnings = reconcile(qty, harga, jumlah)
        if conf < MIN_CONFIDENCE:
            item_warnings.append("low_confidence")

        items.append(LineItem(nama=nama, qty=qty, harga=harga, jumlah=jumlah,
                              confidence=round(conf, 3), reconciled=ok,
                              warnings=item_warnings))

    computed = sum(i.jumlah for i in items if i.jumlah is not None)
    if not items:
        warnings.append("no_line_items_found")
    if grand_total is None:
        # Never invent it. An unread total is unknown, not zero, and not the
        # largest line item -- a confidently wrong total is worse than none.
        warnings.append("grand_total_not_found")
        matches = None
    else:
        matches = computed == grand_total
        if not matches:
            warnings.append(f"total_mismatch: rows sum to {computed}, "
                            f"receipt says {grand_total}")

    needs_review = (not items
                    or matches is not True
                    or any(not i.reconciled or i.warnings for i in items))

    return {
        "items": [asdict(i) for i in items],
        "total": {"computed": computed, "stated": grand_total, "matches": matches},
        "warnings": warnings,
        "needs_review": needs_review,
    }
