"""
Synthetic daily sales flow for the SNAPTOCK prediction engine.

The MVP cannot accumulate real sales history (the rulebook forbids the automated
logging pipeline that would collect it), and the competition explicitly permits
synthetic data. This generates the sales log the forecaster trains on.

Demand is drawn as a compound Bernoulli-negative binomial process, per SKU:
one draw decides *whether* the SKU sells on a given day, a second decides
*how many*. Weekly and payday seasonality multiply both. This is deliberately
NOT generated from Croston's own assumptions -- fitting Croston to data drawn
from Croston's model would make the evaluation circular.

Product names and unit prices are real, recovered from the nota-pembelian
corpus by column geometry (see docs/dataset-audit.html).

Usage:
    python ml/generate_sales.py --days 120 --seed 42 --out data/sales.csv
"""

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# (name, unit_price, archetype) -- names and prices observed in the real corpus.
CATALOGUE = [
    ("es teh",         5000,  "fast"),
    ("air es",         2000,  "fast"),
    ("teh hangat",     5000,  "fast"),
    ("gorengan",       2000,  "fast"),
    ("air putih",      2000,  "fast"),
    ("kopi",           2000,  "fast"),
    ("es jeruk",       7000,  "steady"),
    ("risoles",        2500,  "steady"),
    ("nasi goreng",   15000,  "steady"),
    ("es kelapa",      7000,  "steady"),
    ("snack",         15000,  "steady"),
    ("gula",          17000,  "intermittent"),
    ("garam",          6000,  "intermittent"),
    ("es campur",     15000,  "intermittent"),
    ("pulpen",         5000,  "intermittent"),
    ("nasi kotak",    25000,  "lumpy"),
    ("semprit 1 kg",  60000,  "lumpy"),
    ("buku gambar",   12000,  "slow"),
    ("hvs a4 sidu",   55000,  "slow"),
    ("charger hp",    60000,  "slow"),
]

# p_sell = chance the SKU moves at all on a given day
# r, p   = negative binomial shape for quantity when it does move
ARCHETYPES = {
    "fast":         dict(p_sell=0.97, r=9.0, p=0.55),
    "steady":       dict(p_sell=0.78, r=4.0, p=0.55),
    "intermittent": dict(p_sell=0.34, r=2.5, p=0.60),
    "lumpy":        dict(p_sell=0.22, r=1.1, p=0.28),
    "slow":         dict(p_sell=0.07, r=1.5, p=0.70),
}

WEEKDAY_MULT = [1.00, 1.00, 1.02, 1.05, 1.15, 1.35, 1.28]  # Mon..Sun


def payday_mult(day_of_month):
    """Indonesian gajian clusters at month end and the first days after."""
    if day_of_month >= 25 or day_of_month <= 2:
        return 1.30
    return 1.0


def generate(days=120, seed=42, start=None):
    rng = np.random.default_rng(seed)
    start = start or date.today() - timedelta(days=days)
    rows = []

    for sku_id, (name, price, archetype) in enumerate(CATALOGUE, start=1):
        cfg = ARCHETYPES[archetype]

        for offset in range(days):
            day = start + timedelta(days=offset)
            season = WEEKDAY_MULT[day.weekday()] * payday_mult(day.day)

            if rng.random() > min(cfg["p_sell"] * season, 0.99):
                continue

            qty = rng.negative_binomial(cfg["r"], cfg["p"])
            qty = int(round(qty * season))
            if qty <= 0:
                continue

            rows.append({
                "sale_date": day.isoformat(),
                "sku_id": sku_id,
                "product_name": name,
                "qty": qty,
                "unit_price": price,
                "revenue": qty * price,
                "archetype": archetype,
            })

    return rows


def classify(qtys):
    """Syntetos-Boylan demand pattern quadrants."""
    nonzero = [q for q in qtys if q > 0]
    if len(nonzero) < 2:
        return float("inf"), 0.0, "slow"

    adi = len(qtys) / len(nonzero)
    mean = float(np.mean(nonzero))
    cv2 = float(np.std(nonzero) / mean) ** 2 if mean else 0.0

    if adi < 1.32:
        label = "smooth" if cv2 < 0.49 else "erratic"
    else:
        label = "intermittent" if cv2 < 0.49 else "lumpy"
    return adi, cv2, label


def sparkline(qtys):
    blocks = "▁▂▃▄▅▆▇█"
    top = max(qtys) or 1
    return "".join("·" if q == 0 else blocks[min(int(q / top * 7), 7)] for q in qtys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = generate(days=args.days, seed=args.seed)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.out}\n")

    print(f"{len(rows)} sale lines over {args.days} days, {len(CATALOGUE)} SKUs\n")
    print("sale_date   sku  product_name          qty  unit_price   revenue")
    print("-" * 66)
    for r in rows[:8]:
        print(f"{r['sale_date']}  {r['sku_id']:>3}  {r['product_name']:<20} "
              f"{r['qty']:>4}  {r['unit_price']:>9,}  {r['revenue']:>8,}")

    print("\nper-SKU demand pattern")
    print("-" * 96)
    print(f"{'product':<16}{'intended':<14}{'ADI':>6}{'CV²':>7}  {'classified':<13}daily demand")
    for sku_id, (name, price, archetype) in enumerate(CATALOGUE, start=1):
        series = [0] * args.days
        base = min(r["sale_date"] for r in rows)
        for r in rows:
            if r["sku_id"] == sku_id:
                idx = (date.fromisoformat(r["sale_date"]) - date.fromisoformat(base)).days
                series[idx] = r["qty"]
        adi, cv2, label = classify(series)
        adi_s = "inf" if adi == float("inf") else f"{adi:.2f}"
        print(f"{name:<16}{archetype:<14}{adi_s:>6}{cv2:>7.2f}  {label:<13}{sparkline(series[:56])}")


if __name__ == "__main__":
    main()
