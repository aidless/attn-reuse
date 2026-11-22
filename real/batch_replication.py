#!/usr/bin/env python3
"""Do the token-level conclusions survive a five-times-larger sample?

Every token-level number in the manuscript comes from four batches of 512
positions. Four batches is enough to see a monotone trend and not enough to
settle a cell: the same grid at twenty batches moves. This probe compares the
two runs cell by cell and asks the only question that matters for the paper:

  ordering   is the ranking we report (k descending, group width ascending)
             the same in both samples?
  zero       which cells were reported as indistinguishable from zero in one
             sample and not in the other?
  deviation  how far do the absolute values move?

The two samples are not nested: `run grid` walks the corpus with stride
seq*7, so four batches reach token ~14k and twenty reach ~72k. A difference
between them is therefore *sampling*, not measurement error, and the honest
reading is "two samples agree on the ordering" -- not "four batches was
wrong". The one claim this probe is designed to kill is any claim about a
cell that is cheap in one sample and not the other, since those are exactly
the cells a reader would quote.

Usage:
  python batch_replication.py                        # data/batch_replication.json
  python batch_replication.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# (label, four-batch artifact, twenty-batch artifact)
PAIRS = [
    ("0.5B", "main.json", "main_20b_05.json"),
    ("1.5B", "main_15b.json", "main_20b_15.json"),
    ("7B", "main_7b.json", "main_20b_7.json"),
    ("0.5B@2048", "long.json", "long_20b.json"),
]

# a cell is "indistinguishable from zero" when |cost| is below this many nats.
# Chosen before looking at the comparison: it is the scale at which a 0.5B
# cell stops being quotable as a cost (1e-3 nats is 0.1% perplexity).
ZERO_BAND = 1e-3


def load(name, data_dir=DATA):
    with open(os.path.join(data_dir, name)) as f:
        return json.load(f)


def cells(doc):
    return {k: v["vs_own"] for k, v in doc["deltas"].items()}


def ordering(delta):
    """The paper's two orderings, as booleans, so their survival is checkable.

    k-descending within a group: bigger k must be cheaper. Group-ascending
    within a k: wider group must be dearer. Both are computed from the
    artifact, not typed in, so a mis-keyed table cannot pass.
    """
    out = {}
    ks = sorted({int(k.split("_")[0][1:]) for k in delta}, reverse=True)
    gs = sorted({int(k.split("_")[1][1:]) for k in delta})
    for g in gs:
        pairs = [(k, delta[f"k{k}_g{g}"]) for k in ks if f"k{k}_g{g}" in delta]
        ok = sum(1 for i in range(len(pairs) - 1) if pairs[i][1] < pairs[i + 1][1])
        out[f"k_desc_g{g}"] = {"holds": ok == len(pairs) - 1,
                              "n_pairs": len(pairs), "agreeing": ok}
    for k in ks:
        seq = [(g, delta[f"k{k}_g{g}"]) for g in gs if f"k{k}_g{g}" in delta]
        ok = sum(1 for i in range(len(seq) - 1) if seq[i][1] < seq[i + 1][1])
        out[f"g_asc_k{k}"] = {"holds": ok == len(seq) - 1,
                              "n_pairs": len(seq), "agreeing": ok}
    return out


def compare(a, b):
    """Cell-by-cell comparison of two grids."""
    ca, cb = cells(a), cells(b)
    shared = sorted(set(ca) & set(cb))
    devs = [abs(ca[k] - cb[k]) for k in shared]
    rels = [abs(ca[k] - cb[k]) / abs(cb[k]) if cb[k] else float("inf") for k in shared]
    # NB: Python chains `<` with `!=`, so `abs(a) < z != abs(b) < z` parses as
    # (abs(a)<z) and (z != abs(b)) and (abs(b)<z) -- never the XOR we want.
    # The parentheses below are load-bearing; the selftest below is what caught
    # this in the first place.
    crossed = [k for k in shared
               if (abs(ca[k]) < ZERO_BAND) != (abs(cb[k]) < ZERO_BAND)]
    signflip = [k for k in shared if (ca[k] > 0) != (cb[k] > 0)]
    return {
        "n_cells": len(shared),
        "cells_only_in_four_batch": sorted(set(ca) - set(cb)),
        "cells_only_in_twenty_batch": sorted(set(cb) - set(ca)),
        "median_abs_deviation": st.median(devs),
        "mean_abs_deviation": st.mean(devs),
        "max_abs_deviation": max(devs),
        "median_relative_deviation": st.median(rels),
        "max_relative_deviation": max(rels),
        "worst_cell": max(shared, key=lambda k: rels[shared.index(k)]),
        "sign_agreement": len(shared) - len(signflip),
        "sign_flips": signflip,
        "cells_crossing_zero_band": crossed,
        "zero_band": ZERO_BAND,
        "four_batch_dense": a["readings"]["full"],
        "twenty_batch_dense": b["readings"]["full"],
        "n_batches_four": a.get("n_batches"),
        "n_batches_twenty": b.get("n_batches"),
        "ordering_four": ordering(ca),
        "ordering_twenty": ordering(cb),
    }


def analyse(data_dir=DATA):
    out = {"zero_band": ZERO_BAND, "models": {}}
    for label, four, twenty in PAIRS:
        try:
            a, b = load(four, data_dir), load(twenty, data_dir)
        except FileNotFoundError:
            continue
        out["models"][label] = compare(a, b)
    return out


def selftest():
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        ok = ok and cond

    def grid(vals):
        return {"n_batches": 1, "readings": {"full": 2.5},
                "deltas": {k: {"vs_own": v} for k, v in vals.items()}}

    # k descending AND group ascending must both hold, or the fixture itself
    # is inconsistent with the ordering the manuscript claims.
    a = grid({"k32_g2": 0.470, "k128_g2": 0.150, "k256_g2": 0.015, "k32_g3": 0.700})
    b = grid({"k32_g2": 0.480, "k128_g2": 0.170, "k256_g2": 0.024, "k32_g3": 0.800})
    c = compare(a, b)
    check("identical ordering is reported as holding in both",
          all(v["holds"] for v in c["ordering_four"].values())
          and all(v["holds"] for v in c["ordering_twenty"].values()))
    check("same-sign cells produce no sign flip", c["sign_flips"] == [])

    # prove-red 1: invert the k ordering in one grid -> must be caught
    d = grid({"k32_g2": 0.05, "k128_g2": 0.15, "k256_g2": 0.24, "k32_g3": 0.47})
    c2 = compare(a, d)
    check("inverted k ordering is caught", not c2["ordering_twenty"]["k_desc_g2"]["holds"])

    # prove-red 2: a cell that crosses the zero band must be named
    e = grid({"k32_g2": 0.470, "k128_g2": 0.150, "k256_g2": -0.0005, "k32_g3": 0.700})
    c3 = compare(a, e)
    check("a cell crossing the zero band is named",
          c3["cells_crossing_zero_band"] == ["k256_g2"])
    check("a sign flip is named", c3["sign_flips"] == ["k256_g2"])

    print("selftest " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "batch_replication.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    doc = analyse()
    json.dump(doc, open(args.out, "w"), indent=1)
    for label, m in doc["models"].items():
        print(f"{label}: {m['n_cells']} cells, median |d| {m['median_abs_deviation']:.5f} "
              f"(rel {100 * m['median_relative_deviation']:.1f}%), "
              f"sign {m['sign_agreement']}/{m['n_cells']}, "
              f"zero-band crossings {len(m['cells_crossing_zero_band'])} "
              f"{m['cells_crossing_zero_band']}")
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
