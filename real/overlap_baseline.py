#!/usr/bin/env python3
"""Correct chance baseline for per-query adjacent-layer top-k overlap.

The naive baseline k/(T/2) is wrong: for queries whose prefix is shorter than k,
the top-k *is* the whole prefix, so both layers necessarily agree. The exact
expectation for two independent draws of size m=min(k,u) from u candidates is
m/u, averaged over u=1..T:

    E[overlap] = (1/T) * [ sum_{u<=k} 1 + sum_{u>k} k/u ]

Run --selftest to confirm the closed forms against brute-force draws.
"""
from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
HARMONY = {}


def H(n):
    """Harmonic number, exact enough via summation (n <= 4096 here)."""
    if n not in HARMONY:
        s = 0.0
        for i in range(1, n + 1):
            s += 1.0 / i
        HARMONY[n] = s
    return HARMONY[n]


def expected_overlap(T, k):
    k = min(k, T)
    inner = k * 1.0 + k * (H(T) - H(k))
    return inner / T


def brute(T, k, trials=4000, seed=0):
    rng = random.Random(seed)
    tot = 0.0
    for _ in range(trials):
        u = rng.randint(1, T)
        m = min(k, u)
        a = set(rng.sample(range(u), m)) if m < u else set(range(u))
        b = set(rng.sample(range(u), m)) if m < u else set(range(u))
        tot += len(a & b) / m
    return tot / trials


def main():
    obs = json.load(open(os.path.join(DATA, "overlap2.json")))
    rows = []
    for key, v in obs.items():
        T = int(key.split("_")[0][3:])
        k = int(key.split("_k")[1])
        base = expected_overlap(T, k)
        rows.append({"cell": key, "seq": T, "k": k, "k_over_T": k / T,
                     "measured": v["adjacent_mean"], "chance": base,
                     "enrichment": v["adjacent_mean"] / base,
                     "naive_baseline": k / (T / 2),
                     "per_pair_min": min(v["per_pair"]),
                     "per_pair_max": max(v["per_pair"])})
    json.dump(rows, open(os.path.join(DATA, "overlap_baseline.json"), "w"), indent=2)
    print(f"{'cell':>15} {'k/T':>6} {'measured':>9} {'chance':>8} {'x':>5} {'naive':>7}")
    for r in rows:
        print(f"{r['cell']:>15} {r['k_over_T']:>6.3f} {r['measured']:>9.4f} "
              f"{r['chance']:>8.4f} {r['enrichment']:>5.2f} {r['naive_baseline']:>7.4f}")


def selftest():
    # closed form vs brute force
    for T, k in ((64, 8), (64, 32), (128, 16)):
        e, b = expected_overlap(T, k), brute(T, k, trials=3000, seed=T + k)
        assert abs(e - b) < 0.02, f"T={T} k={k}: closed {e:.4f} vs brute {b:.4f}"
    # edge cases
    assert abs(expected_overlap(10, 10) - 1.0) < 1e-12
    assert expected_overlap(100, 1) < 0.06
    # prove the naive baseline is wrong, i.e. this script's premise is real
    assert expected_overlap(512, 256) > 0.8, "short-prefix saturation must show up"
    assert abs(256 / (512 / 2) - 1.0) < 1e-12, "naive baseline saturates at 1.0"
    print("selftest OK (closed form == brute force; naive baseline provably saturates)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
