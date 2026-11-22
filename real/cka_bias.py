"""Measure the finite-sample upward bias of linear-kernel CKA.

Linear CKA is not zero for independent random matrices: with n samples and d
dimensions it settles near 1/sqrt(n), so a value is only interpretable next to
(n, d). The manuscript quotes two values; this script derives both from a
seeded measurement instead of from memory, and writes them to an artifact the
audit can read.

Usage: python3 cka_bias.py [--out PATH] [--selftest]
"""
import argparse
import json
import os
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "cka_bias.json")
SEED = 20261120
REPEATS = 20
DIMS = 64


def linear_cka(X, Y):
    # float64: at 2000 samples the Gram matrices are 2000x2000, and float32
    # accumulation puts the self-similarity at 1.00015 rather than 1.00000 --
    # a "error" of the same size as the bias we are trying to measure.
    X = X.double()
    Y = Y.double()
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K = X @ X.t()
    L = Y @ Y.t()
    return float((K * L).sum() / (K.norm() * L.norm()))


def measure(n, d=DIMS, repeats=REPEATS, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    vals = [linear_cka(torch.randn(n, d, generator=g),
                       torch.randn(n, d, generator=g)) for _ in range(repeats)]
    return {"n": n, "d": d, "repeats": repeats,
            "mean": st.mean(vals), "min": min(vals), "max": max(vals),
            "values": vals}


def selftest():
    x = torch.randn(2000, 64, generator=torch.Generator().manual_seed(0))
    assert abs(linear_cka(x, x) - 1.0) < 1e-9, "self-similarity must be 1"
    assert abs(linear_cka(x, 3 * x) - 1.0) < 1e-9, "CKA is scale invariant"
    # the bias must shrink with n, in the right direction and the right shape
    small = measure(200, repeats=8)
    large = measure(2000, repeats=8)
    assert small["mean"] > 4 * large["mean"], (small["mean"], large["mean"])
    assert large["mean"] < 0.10, large["mean"]
    print("selftest OK (self-similarity 1, scale invariance, bias falls with n "
          "from %.4f at n=200 to %.4f at n=2000)" % (small["mean"], large["mean"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    frozen = {"schema": "cka-bias/1", "seed": SEED, "dims": DIMS,
              "basis": "two independent standard-normal matrices per repeat",
              "points": [measure(n) for n in (200, 2000)]}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=1, sort_keys=True)
    for p in frozen["points"]:
        print("n=%-5d d=%d  mean=%.4f  [%.4f, %.4f]"
              % (p["n"], p["d"], p["mean"], p["min"], p["max"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
