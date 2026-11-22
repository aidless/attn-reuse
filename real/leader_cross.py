#!/usr/bin/env python3
"""Is the leader anomaly a property of scale, or of the 0.5B model?

The token-level grid on 0.5B says borrowing a group leader's top-k costs
~14x more when the leader is layer 0 or 1 than when it is deeper. That is
one model. This probe reruns the single-pair sweep on all three sizes at
the same protocol (k=256, window<=6, 512 positions, four batches) and asks
three things:

  ratio    mean(leaders 0-1) / mean(leaders 2+), per model
  null     the same ratio under 10,000 permutations of the source labels
           (the pairs' costs are reshuffled, so any ratio this large could
           come from a cost distribution with no source structure at all)
  distance does the median cost still rise with the borrow span

The distinction the ratio hides: the *absolute* cost of borrowing from
layer 0/1 can fall with scale while the ratio rises, because the cost of
borrowing from any deeper layer falls faster and approaches zero. Both
numbers are reported so the ratio is never quoted alone.

Usage:
  python leader_cross.py                       # writes data/leader_cross.json
  python leader_cross.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SOURCES = {
    "0.5B": "pairs_wide.jsonl",
    "1.5B": "pairs_15b.jsonl",
    "7B": "pairs_7b.jsonl",
}
# the same sweep at twenty batches (a different, five-times-larger sample)
SOURCES_20B = {
    "0.5B": "pairs20_0.5B.jsonl",
    "1.5B": "pairs20_1.5B.jsonl",
    "7B": "pairs20_7B.jsonl",
}
# ... and the wide grid at twenty batches (every pair up to distance 12)
SOURCES_20B_W12 = {
    "0.5B": "pairs20w12_0.5B.jsonl",
    "1.5B": "pairs20w12_1.5B.jsonl",
    "7B": "pairs20w12_7B.jsonl",
}
MAX_DISTANCE = 6
PERMUTATIONS = 10000
SEED = 20261119


def load(name, data_dir=DATA, max_distance=MAX_DISTANCE, sources=None):
    """Only genuine pair records; base/own/control lines are gates, not data."""
    sources = sources or SOURCES
    out = []
    for line in open(os.path.join(data_dir, sources[name])):
        rec = json.loads(line)
        if rec.get("kind") != "pair":
            continue
        d = rec["target"] - rec["source"]
        if 0 < d <= max_distance:
            out.append(rec)
    return out


def split(pairs):
    lead = [p["vs_own"] for p in pairs if p["source"] <= 1]
    rest = [p["vs_own"] for p in pairs if p["source"] >= 2]
    return lead, rest


def ratio_of(lead, rest):
    ml = sum(lead) / len(lead)
    mr = sum(rest) / len(rest)
    if mr == 0:
        return float("inf"), ml, mr
    return ml / mr, ml, mr


def permutation(lead, rest, observed, n=PERMUTATIONS, seed=SEED):
    """Reshuffle the costs across the lead/rest split, keeping the split sizes.

    This is the null the ratio needs: with the observed costs but no
    association between source layer and cost, how large a ratio appears?
    The ratio is not a difference, so a permutation null of it is not
    centred at one -- it is centred at mean(lead_all)/mean(rest_all), which
    for a cost distribution with a positive mean is near but not equal to
    one. We report the observed ratio's percentile, not its z-score.
    """
    vals = lead + rest
    n_lead = len(lead)
    rng = random.Random(seed)
    null = []
    for _ in range(n):
        perm = rng.sample(vals, len(vals))
        null.append(ratio_of(perm[:n_lead], perm[n_lead:])[0])
    null.sort()
    exceed = sum(1 for x in null if x >= observed)
    return {
        "n_permutations": n,
        "seed": seed,
        "median": null[n // 2],
        "p90": null[int(0.9 * n)],
        "p99": null[int(0.99 * n)],
        "max": null[-1],
        "p_value": (exceed + 1) / (n + 1),
    }


def distance_medians(pairs):
    by_d = {}
    for p in pairs:
        by_d.setdefault(p["target"] - p["source"], []).append(p["vs_own"])
    return {str(d): {"median": st.median(v), "mean": sum(v) / len(v), "n": len(v)}
            for d, v in sorted(by_d.items())}


def per_source(pairs):
    by_s = {}
    for p in pairs:
        by_s.setdefault(p["source"], []).append(p["vs_own"])
    return {str(s): {"mean": sum(v) / len(v), "max": max(v), "n": len(v)}
            for s, v in sorted(by_s.items())}


def analyse(data_dir=DATA, max_distance=MAX_DISTANCE, models=None,
            sources=None, n_batches=4):
    out = {"max_distance": max_distance, "k": 256, "seq": 512,
           "n_batches": n_batches, "models": {}}
    sources = sources or SOURCES
    for name in (models or sources):
        pairs = load(name, data_dir, max_distance, sources)
        if not pairs:
            continue
        lead, rest = split(pairs)
        r, ml, mr = ratio_of(lead, rest)
        out["models"][name] = {
            "n_pairs": len(pairs),
            "n_leaders_0_1": len(lead),
            "n_leaders_2_plus": len(rest),
            "mean_leaders_0_1": ml,
            "mean_leaders_2_plus": mr,
            "ratio": r,
            "abs_cost_leaders_0_1": ml,
            "permutation": permutation(lead, rest, r),
            "distance_medians": distance_medians(pairs),
            "per_source": per_source(pairs),
        }
    return out


def selftest():
    """Two-directional: a synthetic anomaly must be found, a flat one must not."""
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        ok = ok and cond

    tmp = os.path.join(DATA, "_selftest")
    os.makedirs(tmp, exist_ok=True)
    # 12 expensive leaders, 135 cheap followers -> ratio must be large and
    # the permutation must almost never reach it.
    rng = random.Random(7)
    with open(os.path.join(tmp, SOURCES["0.5B"]), "w") as f:
        f.write(json.dumps({"kind": "base", "full": 2.0}) + "\n")
        f.write(json.dumps({"kind": "control", "vs_full": 0.0}) + "\n")
        idx = 0
        for src in range(0, 8):
            for tgt in range(src + 1, min(src + MAX_DISTANCE + 1, 12)):
                val = (0.010 + rng.gauss(0, 0.001) if src <= 1
                       else 0.0001 + rng.gauss(0, 0.0001))
                f.write(json.dumps({"kind": "pair", "k": 256, "target": tgt,
                                    "source": src, "vs_own": val}) + "\n")
                idx += 1
    sep = analyse(tmp, models=["0.5B"])
    check("separated fixture gives a ratio above 20",
          sep["models"]["0.5B"]["ratio"] > 20)
    check("separated fixture's permutation rejects at p<0.01",
          sep["models"]["0.5B"]["permutation"]["p_value"] < 0.01)

    # flat fixture: every pair costs the same -> ratio 1, permutation all 1.
    with open(os.path.join(tmp, SOURCES["0.5B"]), "w") as f:
        for src in range(0, 8):
            for tgt in range(src + 1, min(src + MAX_DISTANCE + 1, 12)):
                f.write(json.dumps({"kind": "pair", "k": 256, "target": tgt,
                                    "source": src, "vs_own": 0.001}) + "\n")
    flat = analyse(tmp, models=["0.5B"])
    check("flat fixture gives ratio 1.0",
          abs(flat["models"]["0.5B"]["ratio"] - 1.0) < 1e-9)
    check("flat fixture's permutation cannot reject",
          flat["models"]["0.5B"]["permutation"]["p_value"] > 0.5)

    # gate records must not be counted as pairs
    with open(os.path.join(tmp, SOURCES["0.5B"]), "a") as f:
        f.write(json.dumps({"kind": "pair", "k": 256, "target": 11,
                            "source": 0, "vs_own": 0.99}) + "\n")
        f.write(json.dumps({"kind": "control", "k": 512, "vs_full": 0.0}) + "\n")
    kept = load("0.5B", tmp)
    check("control lines stay out of the pair set",
          all(p["kind"] == "pair" for p in kept))

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("selftest " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "leader_cross.json"))
    ap.add_argument("--batches", type=int, default=4, choices=[4, 20])
    ap.add_argument("--window", type=int, default=MAX_DISTANCE,
                    help="max borrow distance; 6 is the narrow grid, 12 the wide one")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    want20 = args.batches == 20
    # A window wider than the sweep's own cannot invent rows: pick the file set
    # that contains them. Asking for window=12 on the window-6 files silently
    # returns the window-6 answer, which is exactly the kind of quiet
    # substitution this probe exists to prevent.
    if want20 and args.window > 6:
        sources = SOURCES_20B_W12
    elif want20:
        sources = SOURCES_20B
    else:
        sources = SOURCES
    doc = analyse(sources=sources, n_batches=20 if want20 else 4,
                  max_distance=args.window)
    json.dump(doc, open(args.out, "w"), indent=1)
    for name, m in doc["models"].items():
        if not m["n_pairs"]:
            continue
        print(f"{name}: n={m['n_pairs']} leaders0-1 {m['mean_leaders_0_1']:+.6f} "
              f"({m['n_leaders_0_1']}) rest {m['mean_leaders_2_plus']:+.6f} "
              f"({m['n_leaders_2_plus']}) ratio {m['ratio']:.2f} "
              f"p={m['permutation']['p_value']:.4f}")
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
