"""
Detected text boxes -> validated line items.

Structure is discovered per document (see layout.py): columns are found by
clustering box positions, and their roles are decided by which assignment makes
the receipt's own arithmetic add up. Nothing here assumes a particular
supplier's template.

The response shape is fixed - the backend depends on it.
"""

from __future__ import annotations

from layout import (assign_roles, build_grid, is_number, split_count, to_int)

MIN_CONFIDENCE = 0.70     # below this, ask a human to confirm the row


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
    rows, cols = build_grid(boxes, width, height)
    if not rows or not cols:
        return {"items": [], "total": {"computed": 0, "stated": None, "matches": None},
                "warnings": ["no_text_detected"], "needs_review": True}

    roles = assign_roles(boxes, rows, cols)
    cell, row_of, col_of = roles["cell"], roles["row_of"], roles["col_of"]

    def value(r, role):
        col = roles[role]
        return cell.get((r, col)) if col is not None else None

    # worst character confidence in a row, for the rows we keep
    row_conf: dict[int, float] = {}
    for i, b in enumerate(boxes):
        r = row_of[i]
        row_conf[r] = min(row_conf.get(r, 1.0), b["conf"])

    items, warnings, last_item_row = [], [], -1
    for r in range(len(rows)):
        nama = (value(r, "nama") or "").strip()
        harga = to_int(value(r, "harga") or "")
        jumlah = to_int(value(r, "jumlah") or "")
        qty = to_int(value(r, "qty") or "")

        # A row with no quantity, whose name starts with one, is a row where the
        # detector welded the two cells together. layout.py only splits a column
        # when it is that column's rule; this catches the single stray row, and
        # it can only ever fill a hole -- a row that already has a quantity is
        # left alone.
        if qty is None:
            welded = split_count(nama)
            if welded:
                qty, nama = welded

        # A line item needs a word for a name and a price. Printed headers fail
        # this (their "harga" cell reads the word "harga", not a number).
        if not nama or is_number(nama) or harga is None:
            continue

        reconciled, item_warnings = reconcile(qty, harga, jumlah)
        confidence = row_conf.get(r, 0.0)
        if confidence < MIN_CONFIDENCE:
            item_warnings.append("low_confidence")

        items.append({"nama": nama, "qty": qty, "harga": harga, "jumlah": jumlah,
                      "confidence": round(confidence, 3),
                      "reconciled": reconciled, "warnings": item_warnings})
        last_item_row = r

    # Grand total: a jumlah value below the last item, with no item beside it.
    # With no items there is no "below", and the scan would start at the top of
    # the page and read the nota number as the total. Unknown beats invented.
    stated = None
    for r in range(last_item_row + 1, len(rows)) if items else ():
        candidate = to_int(value(r, "jumlah") or "")
        nama = (value(r, "nama") or "").strip()
        if candidate is not None and (not nama or is_number(nama)):
            stated = candidate
            break

    computed = sum(i["jumlah"] for i in items if i["jumlah"] is not None)

    if not items:
        warnings.append("no_line_items_found")
    if stated is None:
        warnings.append("grand_total_not_found")   # unknown is not zero
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
