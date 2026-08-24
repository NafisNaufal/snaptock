"""
Discover a nota's table structure instead of assuming it.

The old parser hard-coded one booklet's geometry (columns at 0.21 / 0.62 / 0.77
of image width) and one booklet's printed wording. Any other supplier's form
broke all of it at once.

This asks two different questions:

    where are the columns?   -> cluster box centres on the x-axis
    what is each column?     -> look at what is *inside* it, not where it sits

Role assignment is the interesting half. A column of short integers is the
quantity. A column of words is the item name. That leaves two numeric columns,
and the receipt itself tells us which is which:

    qty x harga == jumlah

A wrong assignment fails that identity on every row at once, so we score each
candidate assignment by how many rows reconcile and keep the best. No training,
no template knowledge, and it works on any tabular nota.
"""

from __future__ import annotations

import re
from itertools import permutations

MIN_RECONCILE_RATE = 0.5      # below this we don't trust the assignment


def to_int(text: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def is_number(text: str) -> bool:
    stripped = re.sub(r"[^0-9a-z]", "", text.lower())
    return bool(stripped) and stripped.isdigit()


def cluster_1d(values: list[float], gap: float) -> list[list[int]]:
    """Split sorted values wherever consecutive ones differ by more than gap."""
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters, current = [], [order[0]]
    for prev, idx in zip(order, order[1:]):
        if values[idx] - values[prev] > gap:
            clusters.append(current)
            current = []
        current.append(idx)
    clusters.append(current)
    return clusters


def build_grid(boxes, width: int, height: int):
    """Return (rows, columns) as lists of box-index lists, both reading order."""
    if not boxes:
        return [], []
    med_w = sorted(b["w"] for b in boxes)[len(boxes) // 2]
    med_h = sorted(b["h"] for b in boxes)[len(boxes) // 2]

    ys = [b["y"] + b["h"] / 2 for b in boxes]
    xs = [b["x"] + b["w"] / 2 for b in boxes]

    rows = cluster_1d(ys, gap=max(med_h * 0.8, height * 0.008))
    cols = cluster_1d(xs, gap=max(med_w * 0.6, width * 0.02))

    rows.sort(key=lambda idxs: min(ys[i] for i in idxs))
    cols.sort(key=lambda idxs: min(xs[i] for i in idxs))
    return rows, cols


def column_profile(boxes, members) -> dict:
    texts = [boxes[i]["text"] for i in members]
    numeric = [t for t in texts if is_number(t)]
    return {
        "n": len(texts),
        "numeric_frac": len(numeric) / len(texts) if texts else 0.0,
        "mean_digits": (sum(len(re.sub(r"[^0-9]", "", t)) for t in numeric)
                        / len(numeric)) if numeric else 0.0,
    }


def column_magnitude(cell, rows, col) -> float:
    """Median numeric value in a column. Quantities are small, prices are not --
    this is what separates them, because qty x harga == harga x qty and the
    arithmetic alone cannot tell the two columns apart."""
    values = []
    for r in range(len(rows)):
        v = to_int(cell.get((r, col)) or "")
        if v is not None:
            values.append(v)
    return sorted(values)[len(values) // 2] if values else float("inf")


def score_assignment(cell, rows, qty_c, harga_c, jumlah_c) -> tuple[float, int]:
    """(rate, count) of rows where qty x harga == jumlah under this assignment."""
    checked = matched = 0
    for r in range(len(rows)):
        q = cell.get((r, qty_c)) if qty_c is not None else None
        h, j = cell.get((r, harga_c)), cell.get((r, jumlah_c))
        qi = to_int(q) if q else (1 if qty_c is None else None)
        hi, ji = to_int(h) if h else None, to_int(j) if j else None
        if None in (qi, hi, ji):
            continue
        checked += 1
        matched += (qi * hi == ji)
    return (matched / checked if checked else 0.0), matched


def assign_roles(boxes, rows, cols) -> dict:
    """Map role -> column index, decided by the arithmetic rather than by
    position or by thresholds. Printed form labels sit in the same columns as
    the data and would poison any content threshold, so we simply try every
    ordered triple of columns and keep whichever one makes the receipt add up."""
    profiles = [column_profile(boxes, c) for c in cols]
    col_of = {i: c for c, members in enumerate(cols) for i in members}
    row_of = {i: r for r, members in enumerate(rows) for i in members}

    cell: dict[tuple[int, int], str] = {}
    for i, b in enumerate(boxes):
        key = (row_of[i], col_of[i])
        cell[key] = (cell.get(key, "") + " " + b["text"]).strip()

    indices = list(range(len(cols)))
    best = {"score": 0.0, "matched": 0, "plausible": 0.0,
            "qty": None, "harga": None, "jumlah": None}

    def consider(qty_c, harga_c, jumlah_c):
        nonlocal best
        score, matched = score_assignment(cell, rows, qty_c, harga_c, jumlah_c)
        if matched == 0:
            return
        # Multiplication is commutative, so swapping qty and harga reconciles
        # equally well. Break that tie on magnitude: quantities are small.
        plausible = 1.0
        if qty_c is not None:
            q_mag, h_mag = (column_magnitude(cell, rows, qty_c),
                            column_magnitude(cell, rows, harga_c))
            plausible = 1.0 if q_mag <= h_mag else 0.0
        key = (matched, plausible, score)
        if key > (best["matched"], best.get("plausible", 0.0), best["score"]):
            best = {"score": score, "matched": matched, "plausible": plausible,
                    "qty": qty_c, "harga": harga_c, "jumlah": jumlah_c}

    for qty_c, harga_c, jumlah_c in permutations(indices, 3):
        consider(qty_c, harga_c, jumlah_c)
    if best["matched"] == 0:                     # quantity column missing or unread
        for harga_c, jumlah_c in permutations(indices, 2):
            consider(None, harga_c, jumlah_c)
    if best["matched"] == 0:
        # No unit-price column at all: some booklets print only qty and total.
        # The money column is the one with the largest values.
        numeric = [i for i in indices
                   if profiles[i]["numeric_frac"] >= 0.5 and profiles[i]["n"] >= 2]
        if numeric:
            jumlah_c = max(numeric, key=lambda i: column_magnitude(cell, rows, i))
            rest = [i for i in numeric if i != jumlah_c]
            qty_c = min(rest, key=lambda i: column_magnitude(cell, rows, i)) if rest else None
            best = {"score": 0.0, "matched": 0, "plausible": 0.0,
                    "qty": qty_c, "harga": None, "jumlah": jumlah_c}

    used = {best["qty"], best["harga"], best["jumlah"]}
    candidates = [i for i in indices if i not in used]
    # Item name: the wordiest unused column.
    nama = max(candidates,
               key=lambda i: (1 - profiles[i]["numeric_frac"], profiles[i]["n"]),
               default=None)

    return {"nama": nama, **best, "cell": cell,
            "profiles": profiles, "row_of": row_of, "col_of": col_of}
