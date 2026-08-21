"""
Simulate a warung's inventory ledger from generated demand.

generate_sales.py produces DEMAND. That is not what a shop records. A shop
records what it actually sold, which is capped by what was on the shelf, plus
what it bought from distributors -- and those purchases are exactly the nota
line items the OCR pipeline ingests. This script closes that loop:

    demand  ->  [ naive (s,S) reorder policy with supplier lead time ]
            ->  purchases.csv    (what the nota scanner would produce)
            ->  stock_ledger.csv (daily on-hand, per SKU)

Two things this gives you beyond seed data:

1. A BASELINE TO BEAT. The simulated shopkeeper reorders by a fixed rule.
   Its stockout rate and average holding are the numbers SNAPTOCK's
   recommendation has to improve on. "Better than how they do it today" is a
   far stronger claim than a forecast error metric.

2. DEMAND CENSORING, made explicit. When stock hits zero the shop records a
   sale of zero -- not the demand it could not serve. A forecaster trained on
   recorded sales therefore under-predicts, orders less, and stocks out again.
   The ledger records demand and sold separately so this bias is measurable
   rather than invisible.

Usage:
    python ml/generate_sales.py --days 120 --out data/sim/sales.csv
    python ml/simulate_inventory.py --sales data/sim/sales.csv --out data/sim
"""

import argparse
import collections
import csv
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from generate_sales import CATALOGUE, load_catalogue


def simulate(sales_rows, seed=42, review_days=21, catalogue=None):
    rng = np.random.default_rng(seed)
    catalogue = catalogue or CATALOGUE

    demand = collections.defaultdict(dict)
    for r in sales_rows:
        demand[int(r["sku_id"])][r["sale_date"]] = int(r["qty"])

    days = sorted({r["sale_date"] for r in sales_rows})
    start, end = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    span = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    purchases, ledger = [], []

    for sku_id, (name, price, archetype) in enumerate(catalogue, start=1):
        series = demand.get(sku_id, {})
        mean_daily = sum(series.values()) / max(len(span), 1)

        lead = int(rng.integers(1, 5))                      # supplier lead time, days
        reorder = max(1, math.ceil(mean_daily * (lead + 4)))  # s
        order_up_to = max(2, math.ceil(mean_daily * review_days))  # S
        on_hand = order_up_to
        incoming = collections.Counter()                    # arrival_date -> qty

        for day in span:
            iso = day.isoformat()

            received = incoming.pop(day, 0)
            on_hand += received
            if received:
                purchases.append({
                    "purchase_date": iso, "sku_id": sku_id, "product_name": name,
                    "qty": received, "unit_price": price, "total": received * price,
                })

            want = series.get(iso, 0)
            sold = min(want, on_hand)
            lost = want - sold
            opening = on_hand
            on_hand -= sold

            if on_hand <= reorder and not incoming:
                qty = max(1, order_up_to - on_hand)
                incoming[day + timedelta(days=lead)] += qty

            ledger.append({
                "date": iso, "sku_id": sku_id, "product_name": name,
                "archetype": archetype, "opening": opening, "demand": want,
                "sold": sold, "lost": lost, "received": received, "closing": on_hand,
                "reorder_point": reorder, "order_up_to": order_up_to,
            })

    return purchases, ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", type=Path, default=Path("data/sim/sales.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/sim"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--catalogue", type=Path, default=None)
    ap.add_argument("--skus", type=int, default=None)
    args = ap.parse_args()

    with args.sales.open() as fh:
        sales_rows = list(csv.DictReader(fh))

    purchases, ledger = simulate(sales_rows, seed=args.seed,
                                 catalogue=load_catalogue(args.catalogue, args.skus))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("purchases", purchases), ("stock_ledger", ledger)):
        path = args.out / f"{name}.csv"
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows):>6} rows -> {path}")

    # ---- baseline the AI has to beat ----
    by_arch = collections.defaultdict(lambda: collections.Counter())
    for row in ledger:
        a = by_arch[row["archetype"]]
        a["demand"] += row["demand"]
        a["sold"] += row["sold"]
        a["lost"] += row["lost"]
        a["stockout_days"] += (row["closing"] == 0)
        a["days"] += 1
        a["holding"] += row["closing"]

    print("\nnaive (s,S) shopkeeper baseline — what SNAPTOCK has to improve on")
    print(f"{'archetype':<14}{'fill rate':>11}{'stockout days':>15}{'avg on-hand':>13}{'lost units':>12}")
    for arch in ("fast", "steady", "intermittent", "lumpy", "slow"):
        a = by_arch.get(arch)
        if not a:
            continue
        fill = a["sold"] / a["demand"] if a["demand"] else 1.0
        print(f"{arch:<14}{fill:>10.1%}{a['stockout_days']:>15}"
              f"{a['holding']/a['days']:>13.1f}{a['lost']:>12}")

    tot_d = sum(a["demand"] for a in by_arch.values())
    tot_l = sum(a["lost"] for a in by_arch.values())
    print(f"\noverall fill rate  {1 - tot_l/tot_d:.1%}   ({tot_l} of {tot_d} units lost to stockouts)")
    print(f"demand censoring   {tot_l/tot_d:.1%} of demand never appears in recorded sales.")
    print("                   A forecaster fit on `sold` sees less than the truth and")
    print("                   under-orders. Fit on `demand` only if you can observe it.")


if __name__ == "__main__":
    main()
