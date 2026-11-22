#!/usr/bin/env python3
"""The binary early-leader control rev-da asked for (finding I2).

Round 5 adversarial review, rev-da I2: the manuscript controlled the
*continuous* source-layer number, but the theory's real confound is the
*binary* leader identity (source <= 1).  Deliver-mass was designed to encode
"the leader's positions are bad for the follower"; if it only restates "this
pair's source is an early leader" -- a variable we already know is dominant
(the leader anomaly itself) -- then the correlation is a repackaging of a
known effect, not a new quantity.

This script joins the same artifacts as leader_deliver_analysis.py and adds:

  R   corr(deliver, early_dummy)         how much of deliver is the dummy
  P   partial(deliver, vs_own | dummy)    does the correlation survive the
                                          binary control?
  DR2 delta-R2 of deliver over the dummy  does deliver add variance beyond
                                          "is the source an early leader"?
  PL  partial(deliver, vs_own | dummy AND source AND distance)
                                          the full-control stack

Selftest fixtures
-----------------
  dummy_only : deliver is a pure function of the dummy plus noise that is
               uncorrelated with cost -> P collapses, dR2 ~ 0 (the gate can
               go red: a restated dummy must NOT pass).
  independent: the dummy is silent, deliver tracks cost -> P stays strong.
  layered    : deliver = dummy component + independent component -> the
               partial keeps the independent part only, and dR2 > 0.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

from leader_deliver_analysis import load_mech, load_pairs, join, pearson, \
    partial_pearson, _residual


def load_mech_any(json_path):
    """Accept both probe formats: leader_mechanism writes `deliver_mass`,
    overlap_head2head writes `deliver`. Same hook, same dense pass; the two
    keys are the same quantity measured by the same code path (verified
    bit-identical on the 0.5B overlap: max |diff| = 0.0)."""
    d = json.load(open(json_path))
    pairs = {}
    for p in d["pairs"]:
        v = p.get("deliver_mass", p.get("deliver"))
        assert v is not None, ("pair without a deliver reading", p)
        pairs[(int(p["source"]), int(p["target"]))] = float(v)
    return d, pairs


def r2(y, Z):
    """R^2 of regressing y on design Z (with intercept)."""
    y = np.asarray(y, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    Z = np.column_stack([np.ones(len(y)), Z]) if Z.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def battery(rows):
    n = len(rows)
    deliver = np.array([r["deliver"] for r in rows])
    cost = np.array([r["vs_own"] for r in rows])
    source = np.array([r["source"] for r in rows], float)
    dist = np.array([r["distance"] for r in rows], float)
    dummy = (source <= 1).astype(float)

    R = pearson(deliver, dummy)
    P = partial_pearson(deliver, cost, dummy)
    P_full = partial_pearson(deliver, cost, np.column_stack(
        [dummy, source, dist]))
    r2_d = r2(cost, dummy)
    r2_dd = r2(cost, np.column_stack([dummy, deliver]))
    return {
        "n": n,
        "R_deliver_vs_dummy": R,
        "P_partial_by_dummy": P,
        "P_partial_full_stack": P_full,
        "R2_dummy": r2_d,
        "R2_dummy_plus_deliver": r2_dd,
        "dR2_deliver_over_dummy": r2_dd - r2_d,
    }


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _mk_rows(deliver, cost, source):
    dist = np.abs(np.arange(len(source)) - source)
    return [{"source": int(s), "target": int(s) + max(1, int(d)),
             "distance": int(d), "deliver": float(de), "vs_own": float(c)}
            for s, d, de, c in zip(source, dist, deliver, cost)]


def selftest():
    rng = np.random.default_rng(20261120)
    n = 200
    source = np.sort(rng.integers(0, 20, n))

    # fixture 1: deliver restates the dummy (+ noise uncorrelated with cost)
    cost = -0.5 * (source <= 1) + 0.02 * rng.standard_normal(n)
    deliver = (source <= 1).astype(float) * (-0.6) + \
        0.15 * rng.standard_normal(n)
    b1 = battery(_mk_rows(deliver, cost, source))
    assert b1["P_partial_by_dummy"]["r"] > -0.2, \
        ("dummy_only fixture must collapse the partial", b1)
    assert abs(b1["dR2_deliver_over_dummy"]) < 0.05, \
        ("dummy_only fixture must not add variance", b1)

    # fixture 2: dummy silent, deliver tracks cost
    cost2 = rng.standard_normal(n)
    deliver2 = -0.8 * cost2 + 0.05 * rng.standard_normal(n)
    b2 = battery(_mk_rows(deliver2, cost2, source))
    assert b2["P_partial_by_dummy"]["r"] < -0.5, \
        ("independent fixture must keep the partial", b2)

    # fixture 3: half dummy, half independent
    indep = rng.standard_normal(n)
    cost3 = -0.5 * (source <= 1) + 0.3 * indep + 0.05 * rng.standard_normal(n)
    deliver3 = -0.5 * (source <= 1) - 0.5 * indep + \
        0.05 * rng.standard_normal(n)
    b3 = battery(_mk_rows(deliver3, cost3, source))
    assert b3["P_partial_by_dummy"]["r"] < -0.4, \
        ("layered fixture must keep an independent partial", b3)
    assert b3["dR2_deliver_over_dummy"] > 0.05, \
        ("layered fixture must add variance over the dummy", b3)
    print("SELFTEST OK")
    print("  dummy_only  P=%.3f dR2=%.3f" %
          (b1["P_partial_by_dummy"]["r"], b1["dR2_deliver_over_dummy"]))
    print("  independent P=%.3f dR2=%.3f" %
          (b2["P_partial_by_dummy"]["r"], b2["dR2_deliver_over_dummy"]))
    print("  layered     P=%.3f dR2=%.3f" %
          (b3["P_partial_by_dummy"]["r"], b3["dR2_deliver_over_dummy"]))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mech", action="append", nargs=2,
                    metavar=("LABEL", "JSON"), default=[])
    ap.add_argument("--pairs", action="append", nargs=2,
                    metavar=("LABEL", "JSONL"), default=[])
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.selftest or not args.mech:
        selftest()
        if not args.mech:
            return
    labels = [m[0] for m in args.mech]
    out = {"models": {}}
    for (lab, mech_f), (_, pairs_f) in zip(args.mech, args.pairs):
        _, mech_pairs = load_mech_any(mech_f)
        cost_pairs = load_pairs(pairs_f)
        rows = join(mech_pairs, cost_pairs)
        b = battery(rows)
        out["models"][lab] = b
        print("%-10s n=%d R(dummy)=%.4f P(dummy)=%.4f P(full)=%.4f "
              "R2(dummy)=%.4f R2(+deliver)=%.4f dR2=%.4f" %
              (lab, b["n"], b["R_deliver_vs_dummy"]["r"],
               b["P_partial_by_dummy"]["r"], b["P_partial_full_stack"]["r"],
               b["R2_dummy"], b["R2_dummy_plus_deliver"],
               b["dR2_deliver_over_dummy"]))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
