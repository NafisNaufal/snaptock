"""Does content-based role discovery survive unseen column orders?

Run:  python ml/eval_layout_generalization.py

Generates nota with column orders the recogniser has never seen, then asks
whether serving/layout.py assigns the right role to the right column. Uses
ground-truth boxes from the generator, so this isolates layout understanding
from OCR error.

Ground-truth boxes from the generator, so this isolates LAYOUT understanding
from OCR error. Compares against a fixed-geometry baseline like the original.
"""
import sys, random, collections
sys.path.insert(0, "ml"); sys.path.insert(0, "serving")
from synth_nota import sample_layout, render, load_vocabulary, COLUMN_ORDERS
import layout as L

FIXED = [("qty", 0.02, 0.21), ("nama", 0.21, 0.62), ("harga", 0.62, 0.77), ("jumlah", 0.77, 0.97)]

def fixed_geometry_roles(boxes, W):
    """The original approach: roles are wherever the corpus put them."""
    out = {}
    for role, lo, hi in FIXED:
        members = [b for b in boxes if lo <= (b["x"] + b["w"]/2)/W < hi]
        out[role] = members
    return out

def evaluate(n=200, seed=7):
    rng = random.Random(seed)
    vocab = load_vocabulary()
    new_ok = old_ok = 0
    per_order = collections.defaultdict(lambda: [0, 0, 0])
    by_rows = collections.defaultdict(lambda: [0, 0])
    for _ in range(n):
        lay = sample_layout(rng)
        doc = render(lay, vocab, rng)
        boxes = [{**b, "conf": 1.0} for b in doc["boxes"]]
        H, W = doc["image"].shape[:2]
        key = " ".join(lay.order)

        # --- new: discovered ---
        rows, cols = L.build_grid(boxes, W, H)
        a = L.assign_roles(boxes, rows, cols)
        truth_rows = {b["row"] for b in boxes if b["row"] >= 0}
        correct = True
        for role in ("qty", "harga", "jumlah"):
            if role not in lay.order:
                continue
            col = a.get(role)
            if col is None:
                correct = False; break
            # majority true-role of boxes in the discovered column
            got = collections.Counter(boxes[i]["role"] for i in cols[col]
                                      if boxes[i]["row"] >= 0)   # items only
            if not got or got.most_common(1)[0][0] != role:
                correct = False; break
        new_ok += correct

        # --- old: fixed geometry ---
        fg = fixed_geometry_roles(boxes, W)
        def maj(bs):
            c = collections.Counter(b["role"] for b in bs if b["row"] >= 0)
            return c.most_common(1)[0][0] if c else None
        old_correct = all(maj(fg[role]) == role
                          for role in ("qty", "harga", "jumlah") if role in lay.order)
        old_ok += old_correct

        per_order[key][0] += correct
        per_order[key][1] += old_correct
        per_order[key][2] += 1
        by_rows[len(doc["items"])][0] += correct
        by_rows[len(doc["items"])][1] += 1

    print(f"documents: {n}")
    print(f"  fixed geometry  (old): {old_ok/n:6.1%}")
    print(f"  discovered      (new): {new_ok/n:6.1%}")
    print()
    print(f"{'column order':<34}{'new':>8}{'old':>8}{'n':>5}")
    for key, (nw, od, tot) in sorted(per_order.items(), key=lambda kv: -kv[1][2]):
        print(f"{key:<34}{nw/tot:>7.0%}{od/tot:>8.0%}{tot:>5}")
    print()
    print(f"{'line items on the nota':<34}{'new':>8}{'n':>5}")
    for k in sorted(by_rows):
        ok, tot = by_rows[k]
        print(f"{k:<34}{ok/tot:>7.0%}{tot:>5}")

evaluate()
