#!/usr/bin/env python3
"""Power of the paired MCQ design, so a null is reported as a bound, not a claim.

The MCQ table has 9 (0.5B) cells and none reaches p<0.05. That is only
informative if the experiment could have detected the cost that the NLL grid
says is there. So this script answers one question per cell:

    Given this many items and this many discordant pairs, what is the smallest
    accuracy difference the exact McNemar test would have called significant?

Two quantities, both from the observed discordance rate rather than from an
assumed one:

* `mde_net_loss(alpha=0.05)`: the smallest (b - c) with b + c fixed at the
  observed count that yields p < alpha. Reported as a fraction of n, i.e. the
  minimum detectable accuracy drop.
* `ci95_on_acc_diff`: a Wilson-style interval on the paired difference. Harder
  than it looks — the paired difference is not a proportion of n but a function
  of the two discordant counts — so this reports the interval obtained by
  inverting the exact binomial on the discordant split, which is the honest
  interval for this design.

Run: python3 power_mcq.py --in data/mcq_05b.json
     python3 power_mcq.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if n <= 900:
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
        return min(1.0, 2.0 * tail)
    lg = math.lgamma
    tail = sum(math.exp(lg(n + 1) - lg(i + 1) - lg(n - i + 1) - n * math.log(2.0))
               for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def mde_net_loss(discordant, n_items, alpha=0.05):
    """Smallest b-c that reaches p<alpha with b+c = discordant, as a share of n."""
    if discordant == 0:
        return None
    for delta in range(0, discordant + 1):
        # b + c = discordant and b - c = delta; both must be non-negative
        # integers, so delta must have the same parity as discordant.
        if (discordant + delta) % 2:
            continue
        b = (discordant + delta) // 2
        c = (discordant - delta) // 2
        if mcnemar_exact(b, c) < alpha:
            return {"net_loss": b - c, "b": b, "c": c,
                    "share_of_items": (b - c) / n_items}
    return None


def _binom_cdf(k, n, q):
    """P(X <= k) for X ~ Binomial(n, q)."""
    if q <= 0:
        return 1.0
    if q >= 1:
        return 1.0 if k >= n else 0.0
    return sum(math.comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k + 1))


def clopper_pearson(x, n, level=0.95):
    """Exact interval on the binomial share x/n, both bounds by bisection.

    Monotonicity, worked out once so it is not re-guessed:
      P(X >= x | q) is INCREASING in q  ->  lower bound is where it equals alpha/2
      P(X <= x | q) is DECREASING in q  ->  upper bound is where it equals alpha/2
    Getting either direction backwards collapses a bound to 0 or 1 and yields an
    interval that does not contain the observation; the selftest pins both.
    """
    if n == 0:
        return None
    alpha = 1.0 - level

    lower = 0.0
    if x > 0:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if 1.0 - _binom_cdf(x - 1, n, mid) < alpha / 2:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2

    upper = 1.0
    if x < n:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if _binom_cdf(x, n, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return {"share_low": lower, "share_high": upper}


def exact_ci_on_net_loss(b, c, level=0.95):
    """Exact CI on (b - c), from the Clopper-Pearson interval on b/(b+c).

    net_loss = b - c = n * (2p - 1) where p is the share of discordant pairs that
    go right->wrong, so the interval is a monotone transform of the exact interval
    on p. Built this way the interval always contains the observation; the earlier
    version inverted the McNemar test itself and produced ranges that excluded a
    significant observation, which is not a confidence interval at all.
    """
    n = b + c
    if n == 0:
        return None
    cp = clopper_pearson(b, n, level)
    if cp is None:
        return None
    return {"net_loss_low": n * (2 * cp["share_low"] - 1),
            "net_loss_high": n * (2 * cp["share_high"] - 1),
            "share_low": cp["share_low"], "share_high": cp["share_high"]}


def has_rejection_region(discordant, alpha=0.05):
    """Can this many discordant pairs reject at all, at ANY observed asymmetry?

    With n discordant pairs the smallest attainable two-sided exact p is
    2 * 0.5^n (everything on one side), so below about n=5 no result can reach
    alpha. Reporting MDE for such a cell would be reporting a threshold that does
    not exist.
    """
    return mcnemar_exact(discordant, 0) < alpha


def rows_for(doc):
    out = []
    for key, fl in (doc.get("flips_vs_own") or {}).items():
        k_str, g_str = key.split("_")
        b, c = fl["right_to_wrong"], fl["wrong_to_right"]
        out.append({
            "cell": f"{key}", "k": int(k_str[1:]), "group": int(g_str[1:]),
            "n_items": doc["n_items"], "discordant": b + c,
            "right_to_wrong": b, "wrong_to_right": c,
            "net_loss": b - c, "p_exact": mcnemar_exact(b, c),
            "mde": (mde_net_loss(b + c, doc["n_items"])
                    if has_rejection_region(b + c) else None),
            "mde_pt": ((mde_net_loss(b + c, doc["n_items"])["share_of_items"] * 100.0)
                       if has_rejection_region(b + c) else None),
            "rejection_region_exists": has_rejection_region(b + c),
            "ci95": exact_ci_on_net_loss(b, c),
        })
    return out


def selftest():
    # 1. As discordance grows the *absolute* detectable asymmetry grows too
    #    (more noise), but its share of the discordant pairs falls. That share
    #    is the quantity a reader cares about, so monotonicity is asserted there.
    prev = None
    for d in (8, 12, 30, 60, 120):
        m = mde_net_loss(d, 200)
        share = m["net_loss"] / d
        if prev is not None:
            assert share <= prev, (d, share, prev)
        prev = share

    # 2. the MDE must actually be significant at alpha, and no smaller asymmetry
    #    at the same discordance may be — otherwise it is a lower bound dressed
    #    up as a threshold. (b+c stays fixed, so "one tighter" = b-2/c+2.)
    m = mde_net_loss(38, 200)
    assert mcnemar_exact(m["b"], m["c"]) < 0.05
    tighter = {"b": m["b"] - 2, "c": m["c"] + 2}
    assert mcnemar_exact(tighter["b"], tighter["c"]) >= 0.05, tighter

    # 3. no discordant pairs -> no MDE, not a zero-width interval
    assert mde_net_loss(0, 200) is None

    # 4. the CI must contain the observation — including when the observation is
    #    significant. (An earlier version inverted the test instead of the share
    #    and returned [-5,5] for an observed +7 with p=0.039, i.e. an "interval"
    #    that excludes the estimate. This assertion is what catches that.)
    for b, c in ((25, 13), (8, 1), (34, 8), (5, 0), (0, 5)):
        ci = exact_ci_on_net_loss(b, c)
        assert ci is not None, (b, c)
        assert ci["net_loss_low"] <= b - c <= ci["net_loss_high"], (b, c, ci)
    # symmetry in the counts
    a1, a2 = exact_ci_on_net_loss(13, 25), exact_ci_on_net_loss(25, 13)
    assert abs(a1["net_loss_low"] + a2["net_loss_high"]) < 1e-9, (a1, a2)

    # 4b. cells without a rejection region must be flagged, not given an MDE
    assert not has_rejection_region(4) and has_rejection_region(6)

    # 5. a perfectly balanced split must give a CI that contains 0
    ci0 = exact_ci_on_net_loss(10, 10)
    assert ci0["net_loss_low"] <= 0 <= ci0["net_loss_high"], ci0

    print("selftest OK (MDE monotone in discordance, MDE is tight, no-discordant "
          "has no MDE, CI contains observation and is symmetric, balanced CI covers 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs",
                    default=",".join(os.path.join(DATA, f) for f in
                                     ("mcq_05b.json", "mcq_15b.json", "mcq_7b.json"))),
    ap.add_argument("--out", default=os.path.join(DATA, "mcq_power.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    docs, rows = [], []
    for p in [x for x in a.inputs.split(",") if x]:
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        docs.append(os.path.basename(p))
        rows.extend(dict(r, model=os.path.basename(d["model"]), bench=d["bench"],
                         shots=d["shots"]) for r in rows_for(d))

    def keyed(model, k, group):
        return re.sub(r"[^0-9A-Za-z]+", "_", f"{model}_k{k}_g{group}").strip("_")

    out = {"schema": "mcq-power/1", "sources": docs,
           "by_cell": {keyed(r["model"], r["k"], r["group"]): r for r in rows},
           "note": ("mde is the smallest (right_to_wrong - wrong_to_right) the exact "
                    "test would call significant at alpha=0.05 given the observed "
                    "discordance; ci95 inverts the exact binomial on the discordant "
                    "split. Both are properties of THIS n, not of reuse."),
           "cells": rows}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"WROTE {a.out} cells={len(rows)}")
    for r in rows:
        m = r["mde"]
        ci = r["ci95"]
        print(f"  {r['model']:<22} {r['bench']:<9} k={r['k']:<4} g={r['group']:<3} "
              f"disc={r['discordant']:<3} net={r['net_loss']:<4} p={r['p_exact']:.3f} "
              f"MDE={m['net_loss'] if m else '-'}({(m['share_of_items']*100 if m else 0):.1f}pt) "
              + (f"CI=[{ci['net_loss_low']:+.1f},{ci['net_loss_high']:+.1f}]" if ci else "CI=-")
              + ("" if r["rejection_region_exists"] else "  [no rejection region]"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
