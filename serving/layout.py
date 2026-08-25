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


UNITS = ("bungkus", "renceng", "sachet", "batang", "lembar", "kaleng",
         "toples", "karung", "piring", "keping", "liter", "botol", "galon",
         "lusin", "butir", "papan", "sisir", "kotak", "meter", "gelas",
         "ikat", "biji", "buah", "pack", "roll", "slop", "cup", "dus",
         "bks", "pcs", "pak", "sct", "box", "rim", "sak", "set", "ons",
         "ltr", "ekor", "kg", "gr", "ml", "lt", "pc")

# Two letters is the shortest unit we will strip. A one-letter unit is not
# worth having: "g" for gram would eat the first letter of every item that
# begins with one, turning "1 gorengan" into a gram of "orengan".
LEADING_COUNT = re.compile(r"^\s*(\d{1,3})\s*(" + "|".join(UNITS) + r")?\.?\s*(?=\D)(.+)$",
                           re.IGNORECASE)


def split_count(text: str):
    """
    Peel a leading count, and its unit, off an item name.

    The detector sometimes swallows the quantity cell and the name cell into a
    single text line, so "2 tempe" and "2 kg gula pasir" reach us as one string
    with the count welded to the front of the name. By then no amount of
    clustering can separate them -- it is one box.

    Splitting is not a guess we have to defend on its own: the caller keeps the
    split only if the receipt adds up better with it.

    Returns (count, name), or None when the text is not a count then a name.
    """
    match = LEADING_COUNT.match(text)
    if not match:
        return None
    count, name = match.group(1), match.group(3).strip()
    letters = re.sub(r"[^a-z]", "", name.lower())
    # An item name begins with a letter and is mostly letters. That is what
    # stops the thousands separator in "67.500" from reading as a count of 67
    # followed by an item called ".500" -- and "2 kg" standing on its own is a
    # quantity, not an item called "kg".
    if not name[:1].isalpha() or len(letters) < 3 or name.lower() in UNITS:
        return None
    return int(count), name


def unmerge_counts(cell, rows, n_cols):
    """Give a column's welded-on counts a column of their own.

    Returns (cell, index of the new column), or the cell unchanged and None.
    """
    for col in range(n_cols):
        filled = [r for r in range(len(rows)) if cell.get((r, col))]
        splits = {r: got for r in filled if (got := split_count(cell[(r, col)]))}
        # It has to be the rule for the column, not one stray row.
        if len(filled) >= 2 and len(splits) >= len(filled) * 0.5:
            split = dict(cell)
            for r, (count, name) in splits.items():
                split[(r, col)] = name
                split[(r, n_cols)] = str(count)
            return split, n_cols
    return cell, None


def cluster_1d(values: list[float], gap: float) -> list[list[int]]:
    """
    Group sorted values, each group anchored on its own running centre.

    Measuring against the previous value instead lets one straddling value
    chain two groups into one, and on a table that is exactly how four item
    rows become a single row: every neighbouring pair sits closer than the
    threshold while the two ends lie a couple of row-pitches apart.
    """
    if not values:
        return []
    clusters, centres = [], []
    for i in sorted(range(len(values)), key=lambda i: values[i]):
        if clusters and values[i] - centres[-1] <= gap:
            clusters[-1].append(i)
            centres[-1] += (values[i] - centres[-1]) / len(clusters[-1])
        else:
            clusters.append([i])
            centres.append(values[i])
    return clusters


def row_pitch(boxes, cols, ys) -> float:
    """
    How far apart the table's rows actually are.

    Measured down each column, because a column holds exactly one entry per row:
    the step between consecutive entries in a column *is* the row pitch. The
    median over every column shrugs off the header cells and the blank tail.

    Box height cannot stand in for this. The detector pads each line generously,
    so on a real nota the boxes come out taller than the line spacing and every
    row overlaps its neighbours -- a threshold drawn from box height then never
    fires, and the whole table collapses into a single row.
    """
    steps = []
    for members in cols:
        column = sorted(ys[i] for i in members)
        steps += [b - a for a, b in zip(column, column[1:])]
    return sorted(steps)[len(steps) // 2] if steps else 0.0


def build_grid(boxes, width: int, height: int):
    """Return (rows, columns) as lists of box-index lists, both reading order."""
    if not boxes:
        return [], []
    med_w = sorted(b["w"] for b in boxes)[len(boxes) // 2]
    med_h = sorted(b["h"] for b in boxes)[len(boxes) // 2]

    ys = [b["y"] + b["h"] / 2 for b in boxes]
    xs = [b["x"] + b["w"] / 2 for b in boxes]

    # Columns first: they are separated by whitespace far wider than any word
    # gap, so they cluster reliably, and rows are then measured against them.
    cols = cluster_1d(xs, gap=max(med_w * 0.6, width * 0.02))
    pitch = row_pitch(boxes, cols, ys)
    rows = cluster_1d(ys, gap=pitch / 2 if pitch else max(med_h * 0.8, height * 0.008))

    rows.sort(key=lambda idxs: min(ys[i] for i in idxs))
    cols.sort(key=lambda idxs: min(xs[i] for i in idxs))
    return rows, cols


def column_profile(boxes, members) -> dict:
    return profile([boxes[i]["text"] for i in members])


def profile(texts) -> dict:
    numeric = [t for t in texts if is_number(t)]
    return {
        "n": len(texts),
        "numeric_frac": len(numeric) / len(texts) if texts else 0.0,
        # "contains a digit" is a far weaker claim than "is a number", and it is
        # the one that survives bad handwriting: a misread "15000" is still
        # visibly a money cell, where is_number has already given up on it.
        "digit_frac": (sum(any(c.isdigit() for c in t) for t in texts)
                       / len(texts)) if texts else 0.0,
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


def search_roles(cell, rows, indices, profiles) -> dict:
    """Map role -> column index, decided by the arithmetic rather than by
    position or by thresholds. Printed form labels sit in the same columns as
    the data and would poison any content threshold, so we simply try every
    ordered triple of columns and keep whichever one makes the receipt add up."""
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
        # The arithmetic could not arbitrate -- on a poor photo too many cells
        # come back unreadable for any assignment to add up. Rather than return
        # an empty document, fall back on the one convention every Indonesian
        # nota shares: the money runs on the right, the running total to the
        # right of the unit price. The rows still leave here flagged, so the
        # shopkeeper corrects six pre-filled rows instead of typing them.
        dense = [i for i in indices if profiles[i]["n"] >= max(2, len(rows) * 0.25)]
        # A column only has to be recognisably numeric, not mostly numeric. On
        # the photos that reach this fallback the recognizer has already turned
        # a couple of prices into letters -- demanding a majority of digits
        # throws away the price column precisely when it is needed.
        money = [i for i in dense if profiles[i]["digit_frac"] >= 0.3]
        if money:
            magnitude = lambda i: column_magnitude(cell, rows, i)
            jumlah_c = max(money, key=magnitude)      # the running total is the biggest
            # A unit price in rupiah runs to at least three digits; a quantity is
            # one or two. That is what tells "this form has no harga column"
            # apart from "the harga column was written too badly to reconcile".
            priced = [i for i in money
                      if i != jumlah_c and profiles[i]["mean_digits"] >= 3]
            harga_c = max(priced, key=magnitude) if priced else None
            # Set the item name aside before looking for the quantity: names
            # like "gula pasir 1 kg" carry a digit, and that stray 1 is the
            # smallest number on the page.
            wordiest = max(dense, key=lambda i: (1 - profiles[i]["numeric_frac"],
                                                 profiles[i]["n"]))
            spare = [i for i in dense if i not in (jumlah_c, harga_c, wordiest)]
            qty_c = min(spare, key=magnitude) if spare else None
            best = {"score": 0.0, "matched": 0, "plausible": 0.0,
                    "qty": qty_c, "harga": harga_c, "jumlah": jumlah_c}

    return best


def assign_roles(boxes, rows, cols) -> dict:
    """Discover which column is which, then hand the parser the filled grid."""
    profiles = [column_profile(boxes, c) for c in cols]
    col_of = {i: c for c, members in enumerate(cols) for i in members}
    row_of = {i: r for r, members in enumerate(rows) for i in members}

    cell: dict[tuple[int, int], str] = {}
    for i, b in enumerate(boxes):
        key = (row_of[i], col_of[i])
        cell[key] = (cell.get(key, "") + " " + b["text"]).strip()

    indices = list(range(len(cols)))
    best = search_roles(cell, rows, indices, profiles)

    # Search again with any welded-on counts split into a column of their own,
    # and let the arithmetic choose between the two readings. A tie goes to the
    # split: if the count really was merged into the name, the unsplit reading
    # reconciles just as well while leaving the receipt with no item names at
    # all, because the only wordy column is the one holding the quantities.
    split, extra = unmerge_counts(cell, rows, len(cols))
    if extra is not None:
        texts = [split[(r, extra)] for r in range(len(rows)) if (r, extra) in split]
        alt_indices = indices + [extra]
        alt_profiles = profiles + [profile(texts)]
        alt = search_roles(split, rows, alt_indices, alt_profiles)
        if alt["matched"] >= best["matched"]:
            cell, indices, profiles, best = split, alt_indices, alt_profiles, alt

    used = {best["qty"], best["harga"], best["jumlah"]}
    candidates = [i for i in indices if i not in used]
    # Item name: the wordiest unused column.
    nama = max(candidates,
               key=lambda i: (1 - profiles[i]["numeric_frac"], profiles[i]["n"]),
               default=None)

    return {"nama": nama, **best, "cell": cell,
            "profiles": profiles, "row_of": row_of, "col_of": col_of}
