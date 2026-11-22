#!/usr/bin/env python3
"""14B leader-anomaly check: does the lead-layer effect survive at 48 layers?

Reads the per-pair sweep produced on the 48GB instance (`pairs14b.jsonl`) and
applies the *same* protocol as `cross_family.py` (window<=6, lead = source<=1,
rest = source>=2, 10,000-permutation null, seed 20261119) so the 14B number is
directly comparable to the four models already in the paper.

This is a robustness certificate, not a manuscript number: the 14B sweep ran
with CPU offload (device_map=auto), so it never enters the audit pins.

No GPU needed.  Run --selftest to prove both the positive gate and the
null-flatness gate can fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MAX_DISTANCE = 6
PERMUTATIONS = 10000
SEED = 20261119


def load(path):
    recs = [json.loads(l) for l in open(path)]
    pairs = [r for r in recs if r.get("kind") == "pair"
             and 0 < r["target"] - r["source"] <= MAX_DISTANCE]
    base = next((r for r in recs if r.get("kind") == "base"), None)
    return recs, pairs, base


def ratio_of(lead, rest):
    ml, mr = st.mean(lead), st.mean(rest)
    return (ml / mr if mr else math.inf), ml, mr


def permutation(lead, rest, observed, n=PERMUTATIONS, seed=SEED):
    vals = lead + rest
    nl = len(lead)
    rng = random.Random(seed)
    null = []
    for _ in range(n):
        pp = rng.sample(vals, len(vals))
        d = st.mean(pp[nl:])
        null.append(st.mean(pp[:nl]) / d if d else 0.0)
    null.sort()
    exceed = sum(1 for x in null if x >= observed)
    return {"n": n, "seed": seed, "median": null[n // 2],
            "p99": null[int(0.99 * n)], "max": null[-1],
            "p_value": (exceed + 1) / (n + 1)}


def analyse(path):
    recs, pairs, base = load(path)
    lead = [p["vs_own"] for p in pairs if p["source"] <= 1]
    rest = [p["vs_own"] for p in pairs if p["source"] >= 2]
    ratio, ml, mr = ratio_of(lead, rest)
    by_source = {}
    for p in pairs:
        by_source.setdefault(p["source"], []).append(p["vs_own"])
    return {
        "source_file": os.path.basename(path),
        "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        "n_pairs": len(pairs), "n_lead": len(lead), "n_rest": len(rest),
        "lead_mean": ml, "rest_mean": mr, "ratio": ratio,
        "n_layers": base["n_layers"] if base else None,
        "n_batches": base["n_batches"] if base else None,
        "dtype": base["dtype"] if base else None,
        "own_nll": base["own_nll"] if base else None,
        "permutation": permutation(lead, rest, ratio),
        "bootstrap": bootstrap_ratio(lead, rest),
        "lead_boundary_sensitivity": definition_sensitivity(pairs),
        "top_two_order": top_two_order(pairs),
        "source_means": {str(s): st.mean(v) for s, v in sorted(by_source.items())},
        "distance_medians": {str(d): st.median([p["vs_own"] for p in pairs
                                                if p["distance"] == d])
                            for d in sorted({p["distance"] for p in pairs})},
    }


def bootstrap_ratio(lead, rest, n=10000, seed=SEED):
    """Percentile bootstrap CI for the lead/rest ratio.

    Resamples *within* each group, which is the question a reader actually asks
    ("how uncertain is the 8.5x?"), as opposed to the permutation null below
    ("could a relabelling produce it?").
    """
    rng = random.Random(seed + 1)
    vals = []
    for _ in range(n):
        l = [lead[rng.randrange(len(lead))] for _ in range(len(lead))]
        r = [rest[rng.randrange(len(rest))] for _ in range(len(rest))]
        d = st.mean(r)
        vals.append(st.mean(l) / d if d else math.inf)
    vals.sort()
    return {"n": n, "seed": seed + 1,
            "lo": vals[int(0.025 * n)], "hi": vals[int(0.975 * n)],
            "median": vals[n // 2]}


def definition_sensitivity(pairs):
    """Does the verdict depend on where the lead boundary is drawn?

    The paper's boundary (source <= 1) is a choice, not a law.  We report the
    ratio under the neighbouring choices so a reader can see whether the
    finding survives moving it by one layer in either direction.
    """
    out = {}
    for boundary in (0, 1, 2):
        lead = [p["vs_own"] for p in pairs if p["source"] <= boundary]
        rest = [p["vs_own"] for p in pairs if p["source"] > boundary]
        if not lead or not rest:
            continue
        r, ml, mr = ratio_of(lead, rest)
        out[str(boundary)] = {"ratio": r, "n_lead": len(lead), "n_rest": len(rest),
                              "lead_mean": ml, "rest_mean": mr}
    return out


def top_two_order(pairs, n=10000, seed=SEED + 7):
    """Are sources 0 and 1 distinguishable from each other?

    Report 21 originally wrote that on 14B "source 1 is more expensive than
    source 0", the reverse of 0.5B.  That was a reading of two means, not a
    test.  With six pairs per source we must check whether the difference is
    distinguishable at all before writing it down as a finding.
    """
    a = [p["vs_own"] for p in pairs if p["source"] == 0]
    b = [p["vs_own"] for p in pairs if p["source"] == 1]
    if not a or not b:
        return None
    obs = st.mean(b) - st.mean(a)
    vals = a + b
    rng = random.Random(seed)
    null = []
    for _ in range(n):
        pp = rng.sample(vals, len(vals))
        null.append(st.mean(pp[len(a):]) - st.mean(pp[:len(a)])
                    if len(vals) > len(a) else 0.0)
    exceed = sum(1 for d in null if d >= obs)
    rng2 = random.Random(seed + 1)
    bs = []
    for _ in range(n):
        x = [a[rng2.randrange(len(a))] for _ in range(len(a))]
        y = [b[rng2.randrange(len(b))] for _ in range(len(b))]
        bs.append(st.mean(y) - st.mean(x))
    bs.sort()
    return {"src0_mean": st.mean(a), "src1_mean": st.mean(b), "diff": obs,
            "n_src0": len(a), "n_src1": len(b),
            "perm_p_src1_gt_src0": (exceed + 1) / (n + 1),
            "boot_ci": {"lo": bs[int(0.025 * n)], "hi": bs[int(0.975 * n)]},
            "distinguishable": (bs[int(0.025 * n)] > 0) or (bs[int(0.975 * n)] < 0)}


def selftest():
    # positive: a strong lead effect must be flagged and permutation extreme
    lead = [1.0] * 6 + [1.2] * 6
    rest = [0.01] * 20
    r, ml, mr = ratio_of(lead, rest)
    assert r > 50, (r, ml, mr)
    p = permutation(lead, rest, r)
    assert p["p_value"] <= 0.001, p
    # negative: flat labels must give ratio ~1 and a non-extreme p
    flat = [0.5] * 30
    r2, _, _ = ratio_of(flat[:12], flat[12:])
    assert abs(r2 - 1.0) < 1e-9, r2
    p2 = permutation(flat[:12], flat[12:], r2)
    assert p2["p_value"] > 0.05, p2
    # prove-red: invert the lead labels -> ratio must drop below 1
    r3, _, _ = ratio_of([0.01] * 12, [1.0] * 20)
    assert r3 < 0.1, r3
    # bootstrap CI must bracket the point estimate on a clean effect
    ci = bootstrap_ratio(lead, rest, n=2000)
    assert ci["lo"] <= r <= ci["hi"], (ci, r)
    # prove-red: on identical groups the CI must contain 1.0 and be wide
    ci2 = bootstrap_ratio([0.3, 0.5, 0.9, 0.2], [0.3, 0.5, 0.9, 0.2], n=2000)
    assert ci2["lo"] < 1.0 < ci2["hi"], ci2
    # prove-red: definition sensitivity must collapse when the boundary is
    # pushed past every lead-like layer (all sources > 2 -> skip / ratio ~1)
    mk = lambda src, v: {"source": src, "target": src + 1, "distance": 1, "vs_own": v}
    sens = definition_sensitivity([mk(0, 1.0), mk(1, 1.0)] + [mk(s, 0.01) for s in range(2, 8)])
    assert sens["1"]["ratio"] > 50, sens
    # widening the lead group to include a cheap layer must dilute the ratio
    assert sens["2"]["ratio"] < sens["1"]["ratio"] * 0.9, sens
    # prove-red: a genuinely separated top pair must be called distinguishable,
    # and an overlapping one must not.
    sep = [{"source": 0, "target": 1, "distance": 1, "vs_own": 0.1},
           {"source": 0, "target": 2, "distance": 2, "vs_own": 0.11},
           {"source": 1, "target": 2, "distance": 1, "vs_own": 0.9},
           {"source": 1, "target": 3, "distance": 2, "vs_own": 0.92}]
    t = top_two_order(sep, n=2000)
    assert t["distinguishable"] and t["boot_ci"]["lo"] > 0, t
    overlap = [{"source": 0, "target": 1, "distance": 1, "vs_own": 0.10},
               {"source": 0, "target": 2, "distance": 2, "vs_own": 0.90},
               {"source": 1, "target": 2, "distance": 1, "vs_own": 0.11},
               {"source": 1, "target": 3, "distance": 2, "vs_own": 0.89}]
    t2 = top_two_order(overlap, n=2000)
    assert not t2["distinguishable"], t2
    print("SELFTEST OK: positive lead, null flat, inverted-lead, bootstrap CI, "
          "definition sweep, top-two separability all behave")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pairs", default=os.path.join(DATA, "pairs14b.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA, "scale_leader_14b.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res = analyse(a.pairs)
    print("14B lead anomaly (k=256, window<=6)")
    print("  layers %s  batches %s  dtype %s  pairs %d (lead %d / rest %d)" % (
        res["n_layers"], res["n_batches"], res["dtype"],
        res["n_pairs"], res["n_lead"], res["n_rest"]))
    print("  lead mean %+.6f   rest mean %+.6f   ratio %.2fx" % (
        res["lead_mean"], res["rest_mean"], res["ratio"]))
    print("  bootstrap 95%% CI [%.2f, %.2f]" % (
        res["bootstrap"]["lo"], res["bootstrap"]["hi"]))
    print("  boundary sensitivity (source<=b):",
          ", ".join("b%d:%.2fx(n=%d)" % (int(b), v["ratio"], v["n_lead"])
                    for b, v in res["lead_boundary_sensitivity"].items()))
    print("  permutation p = %.5f (n=%d, seed=%d)" % (
        res["permutation"]["p_value"], res["permutation"]["n"], res["permutation"]["seed"]))
    tto = res.get("top_two_order") or {}
    print("  src0 vs src1: diff %+.6f  p=%.4f  CI [%+.6f, %+.6f]  distinguishable=%s" % (
        tto.get("diff", float("nan")), tto.get("perm_p_src1_gt_src0", float("nan")),
        tto["boot_ci"]["lo"], tto["boot_ci"]["hi"], tto.get("distinguishable")))
    print("  by source:", ", ".join("%s:%+.5f" % (k, v)
                                   for k, v in list(res["source_means"].items())[:6]))
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("WROTE", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
