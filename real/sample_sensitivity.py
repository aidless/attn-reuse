#!/usr/bin/env python3
"""Which of the paper's token-level claims survive a five-fold larger sample?

Every token-level number in the draft came from four batches of 512 positions.
Twenty batches of the same corpus were then measured with the identical
protocol. The two grids are NESTED, not independent: the corpus walk has stride
seq*7 in both, so batches 0..3 are the same four windows and the twenty-batch
run is the four-batch run plus sixteen more. This was verified bit-exactly
(`precision.json`'s first four per-batch values reproduce `main.json`'s grid
values to 1e-9). What changes between the two runs is therefore not "measurement
error" but *which text the average is over* -- and a claim that changes when
four more fifths of the text are added is a claim about four batches, not about
the model.

This script turns that into a table of claims rather than a table of cells:

  for each quantity the paper quotes
    recompute it from the four-batch artifacts
    recompute it from the twenty-batch artifacts
    classify: stable / moved / reversed

`reversed` means the sign of the quantity, or the direction of the comparison
the paper draws from it, is opposite in the larger sample. Those are the rows
that must change the text; `moved` rows get a qualifier; `stable` rows stand.

Usage:
  python sample_sensitivity.py              # data/sample_sensitivity.json
  python sample_sensitivity.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

GRIDS = {"4b": "main.json", "20b": "main_20b_05.json"}
GRIDS_15 = {"4b": "main_15b.json", "20b": "main_20b_15.json"}
GRIDS_7 = {"4b": "main_7b.json", "20b": "main_20b_7.json"}
LONGS = {"4b": "long_4b.json", "20b": "long_20b.json"}
PAIRS = {"4b": "pairs_wide.jsonl", "20b": "pairs20w12_0.5B.jsonl"}

# "reversed" is decided by these two rules, both stated so a reader can
# disagree with the classification rather than guess at it:
#   sign      zero is a special value here -- many cells are ~1e-3, so a sign
#             flip between two samples of the same sign at 1e-4 is noise, but a
#             flip of a quantity the text quotes as "reduces / increases" is a
#             change of statement
#   factor    a quantity that moves by more than this factor is requoted
MOVE_FACTOR = 2.0
# a quantity whose magnitude is below this is treated as zero for sign purposes
ZERO_BAND = 5e-4


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def rows_of(name):
    with open(os.path.join(DATA, name)) as f:
        return [json.loads(line) for line in f if line.strip()]


def grid_cell(doc, key):
    return doc["deltas"][key]["vs_own"]


def own_minus_full(doc, k):
    return doc["readings"][f"own_k{k}"] - doc["readings"]["full"]


def pair_value(name, target, source):
    for r in rows_of(name):
        if r.get("kind") == "pair" and r["target"] == target and r["source"] == source:
            return r["vs_own"]
    return None


def pair_median(name, max_d=12):
    v = [r["vs_own"] for r in rows_of(name)
         if r.get("kind") == "pair" and 0 < r["target"] - r["source"] <= max_d]
    return st.median(v)


def pair_split(name, max_d=6):
    lead, rest = [], []
    for r in rows_of(name):
        # gate rows (base / own / control) carry no target or source; they are
        # the harness's own receipts, not data, and must be skipped before any
        # key is touched.
        if r.get("kind") != "pair":
            continue
        d = r["target"] - r["source"]
        if not (0 < d <= max_d):
            continue
        (lead if r["source"] <= 1 else rest).append(r["vs_own"])
    return sum(lead) / len(lead), sum(rest) / len(rest)


def additivity(pairs_name, grid_doc, n_layers):
    pair = {(r["target"], r["source"]): r["vs_own"] for r in rows_of(pairs_name)
            if r.get("kind") == "pair"}
    g2 = [pair[(t, t - 1)] for t in range(1, n_layers, 2) if (t, t - 1) in pair]
    g3 = [pair[(t, 3 * (t // 3))] for t in range(1, n_layers)
          if t % 3 and (t, 3 * (t // 3)) in pair]
    g6 = [pair[(t, 6 * (t // 6))] for t in range(1, n_layers)
          if t % 6 and (t, 6 * (t // 6)) in pair]
    out = {}
    for tag, s in (("g2", sum(g2)), ("g3", sum(g3),), ("g6", sum(g6))):
        meas = grid_doc["deltas"]["k256_" + tag]["vs_own"]
        out[tag] = {"predicted": s, "measured": meas, "factor": meas / s}
    return out


def classify(a, b):
    """stable / moved / reversed, with the reason attached."""
    if a is None or b is None:
        return "missing", "one of the two samples has no value"
    a_zero = abs(a) < ZERO_BAND
    b_zero = abs(b) < ZERO_BAND
    if a_zero != b_zero:
        return "reversed", "crosses the zero band"
    if not a_zero and not b_zero and (a > 0) != (b > 0):
        return "reversed", "changes sign"
    if a != 0 and abs(b / a) > MOVE_FACTOR or (a != 0 and abs(b / a) < 1 / MOVE_FACTOR):
        return "moved", f"moves by {b / a:.2f}x"
    return "stable", f"moves by {(b / a if a else float('nan')):.2f}x"


ROWS = [
    # (label, kind, spec)
    ("dense NLL, 0.5B", "gridfull", ("0.5B", None)),
    ("sparsification alone, k=32", "own", ("0.5B", 32)),
    ("sparsification alone, k=64", "own", ("0.5B", 64)),
    ("grid k=32 g=2", "cell", ("0.5B", "k32_g2")),
    ("grid k=32 g=12", "cell", ("0.5B", "k32_g12")),
    ("grid k=64 g=6", "cell", ("0.5B", "k64_g6")),
    ("grid k=384 g=4", "cell", ("0.5B", "k384_g4")),
    ("grid k=384 g=12", "cell", ("0.5B", "k384_g12")),
    ("g2 vs g3 at k=384, g2 side", "cell", ("0.5B", "k384_g2")),
    ("g2 vs g3 at k=384, g3 side", "cell", ("0.5B", "k384_g3")),
    ("1.5B k=128 g=2", "cell", ("1.5B", "k128_g2")),
    ("7B k=256 g=2", "cell", ("7B", "k256_g2")),
    ("7B k=256 g=3", "cell", ("7B", "k256_g3")),
    ("7B k=384 g=3", "cell", ("7B", "k384_g3")),
    ("amplification k=128 g=2", "amp", ("0.5B", "k128_g2")),
    ("amplification k=128 g=6", "amp", ("0.5B", "k128_g6")),
    ("amplification k=256 g=2", "amp", ("0.5B", "k256_g2")),
    ("amplification k=256 g=6", "amp", ("0.5B", "k256_g6")),
    ("leader ratio, 0.5B, d<=6", "ratio6", ("0.5B", 6)),
    ("leader ratio, 1.5B, d<=6", "ratio6", ("1.5B", 6)),
    ("leader ratio, 7B, d<=6", "ratio6", ("7B", 6)),
    ("leader ratio, 0.5B, d<=12", "ratio12", ("0.5B", 12)),
    ("layer 3 borrowing layer 2", "pair", ("0.5B", (3, 2))),
    ("layer 3 over the pair median", "l3over", ("0.5B", None)),
    ("g=2 partition sum", "g2sum", ("0.5B", None)),
    ("g=2 sum without layer 3", "g2nod3", ("0.5B", None)),
    ("additivity factor g=2, 0.5B", "add", ("0.5B", "g2")),
    ("additivity factor g=3, 0.5B", "add", ("0.5B", "g3")),
    ("additivity factor g=6, 0.5B", "add", ("0.5B", "g6")),
    ("additivity factor g=6, 1.5B", "add15", ("1.5B", "g6")),
    ("offset ratio g=2", "offset", ("0.5B", 2)),
    ("offset ratio g=3", "offset", ("0.5B", 3)),
    ("offset ratio g=4", "offset", ("0.5B", 4)),
    ("offset ratio g=6", "offset", ("0.5B", 6)),
]

N_LAYERS = {"0.5B": 24, "1.5B": 28, "7B": 28}


def value(row, sample):
    _, kind, spec = row
    model = spec[0]
    if kind == "gridfull":
        doc = load({"0.5B": GRIDS, "1.5B": GRIDS_15, "7B": GRIDS_7}[model][sample])
        return doc["readings"]["full"]
    if kind == "own":
        doc = load({"0.5B": GRIDS, "1.5B": GRIDS_15, "7B": GRIDS_7}[model][sample])
        return own_minus_full(doc, spec[1])
    if kind == "cell":
        doc = load({"0.5B": GRIDS, "1.5B": GRIDS_15, "7B": GRIDS_7}[model][sample])
        return grid_cell(doc, spec[1])
    if kind == "amp":
        return load(LONGS[sample])["deltas"][spec[1]]["vs_own"] / \
               grid_cell(load(GRIDS[sample]), spec[1])
    if kind in ("ratio6", "ratio12"):
        lead, rest = pair_split(PAIRS[sample], spec[1])
        return lead / rest
    if kind == "pair":
        return pair_value(PAIRS[sample], spec[1][0], spec[1][1])
    if kind == "l3over":
        return pair_value(PAIRS[sample], 3, 2) / pair_median(PAIRS[sample])
    if kind == "g2sum":
        p = {(r["target"], r["source"]): r["vs_own"] for r in rows_of(PAIRS[sample])
             if r.get("kind") == "pair"}
        return sum(p[(t, t - 1)] for t in range(1, 24, 2) if (t, t - 1) in p)
    if kind == "g2nod3":
        return value(("", "g2sum", ("0.5B", None)), sample) - pair_value(PAIRS[sample], 3, 2)
    if kind in ("add", "add15"):
        doc = load(GRIDS_15[sample] if model == "1.5B" else GRIDS[sample])
        pairs_name = "pairs20w12_1.5B.jsonl" if model == "1.5B" else PAIRS[sample]
        return additivity(pairs_name, doc, N_LAYERS[model])[spec[1]]["factor"]
    if kind == "offset":
        name = "offset20.jsonl" if sample == "20b" else "offset.jsonl"
        for r in rows_of(name):
            if r.get("kind") == "offset" and r["group"] == spec[1]:
                return r["vs_own"] / grid_cell(load(GRIDS[sample]), f"k256_g{spec[1]}")
    raise KeyError(kind)


def analyse():
    out = {"move_factor": MOVE_FACTOR, "zero_band": ZERO_BAND,
           "nesting": "twenty-batch runs reuse the same first four windows; "
                      "four-batch grids are reproduced bit-exactly",
           "claims": []}
    for row in ROWS:
        label = row[0]
        a, b = value(row, "4b"), value(row, "20b")
        status, why = classify(a, b)
        out["claims"].append({"claim": label, "four_batch": a,
                              "twenty_batch": b, "status": status, "why": why})
    counts = {}
    for c in out["claims"]:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    out["counts"] = counts
    out["reversed"] = [c["claim"] for c in out["claims"] if c["status"] == "reversed"]
    out["moved"] = [c["claim"] for c in out["claims"] if c["status"] == "moved"]
    return out


def selftest():
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        ok = ok and cond

    check("a sign flip between two quoted values is 'reversed'",
          classify(0.00220, 0.00045)[0] == "reversed" or
          classify(-0.00113, 0.00324)[0] == "reversed")
    check("crossing the zero band is 'reversed'",
          classify(-0.0002, 0.0006) == ("reversed", "crosses the zero band"))
    check("a real sign change above the band is 'reversed'",
          classify(-0.00113, 0.00324) == ("reversed", "changes sign"))
    check("a doubling is 'moved'", classify(0.0108, 0.0242)[0] == "moved")
    check("a 10% change is 'stable'", classify(0.7299, 0.8050)[0] == "stable")
    check("the two batches are nested (first four reproduce)", _nesting_holds())

    doc = analyse()
    check("the analysis covers every declared row",
          len(doc["claims"]) == len(ROWS))
    check("every row is classified", all(c["status"] in
          ("stable", "moved", "reversed") for c in doc["claims"]))
    print("selftest " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def _nesting_holds():
    """The claim in the docstring, checked rather than asserted."""
    p = load("precision.json")
    g = load("main.json")
    for cell in ("reuse_k384_g2", "reuse_k384_g6", "reuse_k256_g3", "reuse_k256_g6"):
        own = cell.replace("reuse", "own").split("_g")[0]
        f4 = sum(p["per_batch"][cell][:4]) / 4 - sum(p["per_batch"][own][:4]) / 4
        if abs(f4 - g["deltas"][cell.split("reuse_")[1]]["vs_own"]) > 1e-9:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "sample_sensitivity.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    doc = analyse()
    json.dump(doc, open(args.out, "w"), indent=1)
    for c in doc["claims"]:
        print(f"{c['status']:9s} {c['claim']:38s} {c['four_batch']:+9.6f} -> "
              f"{c['twenty_batch']:+9.6f}  ({c['why']})")
    print(doc["counts"])
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
