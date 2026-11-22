#!/usr/bin/env python3
"""Cross-family check: does the leader-layer cost survive a model-family change?

The main paper measures one family (Qwen2.5, three sizes). A reviewer's first
question is whether the leader anomaly is a Qwen2.5 artifact. This script adds
a fourth model from a *different* family (TinyLlama-1.1B, Llama architecture,
non-gated) measured with the identical protocol: same corpus, k=256, 512
positions, 20 batches, single-pair borrowing, window<=6. It recomputes the
same three readings for all four models from stored pair records:

  ratio   mean cost when leader is layer 0 or 1 / mean cost for deeper leaders
  null    the same ratio under 10,000 permutations of the cost labels
  sign    mean absolute cost, so the ratio is never quoted alone

No GPU is needed here: this reads the pair jsonl files produced on the remote
host. Run --selftest to prove the gate can fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SOURCES = {
    "Qwen2.5-0.5B": "pairs20_0.5B.jsonl",
    "Qwen2.5-1.5B": "pairs20_1.5B.jsonl",
    "Qwen2.5-7B": "pairs20_7B.jsonl",
    "TinyLlama-1.1B": "pairs20_TinyLlama.jsonl",
}
MAX_DISTANCE = 6
PERMUTATIONS = 10000
SEED = 20261119


def load(name, data_dir=DATA):
    recs = [json.loads(l) for l in open(os.path.join(data_dir, SOURCES[name]))]
    pairs = [r for r in recs if r.get("kind") == "pair"
             and 0 < r["target"] - r["source"] <= MAX_DISTANCE]
    return recs, pairs


def split(pairs):
    lead = [p["vs_own"] for p in pairs if p["source"] <= 1]
    rest = [p["vs_own"] for p in pairs if p["source"] >= 2]
    return lead, rest


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
        r = st.mean(pp[:nl]) / st.mean(pp[nl:]) if st.mean(pp[nl:]) else 0.0
        null.append(r)
    null.sort()
    exceed = sum(1 for x in null if x >= observed)
    return {
        "n": n, "seed": seed,
        "median": null[n // 2], "p99": null[int(0.99 * n)], "max": null[-1],
        "p_value": (exceed + 1) / (n + 1),
    }


def analyse(name, data_dir=DATA):
    recs, pairs = load(name, data_dir)
    lead, rest = split(pairs)
    ratio, ml, mr = ratio_of(lead, rest)
    perm = permutation(lead, rest, ratio)
    by = {}
    for p in pairs:
        by.setdefault(p["distance"], []).append(p["vs_own"])
    return {
        "model": name,
        "source_file": SOURCES[name],
        "sha256": hashlib.sha256(open(os.path.join(data_dir, SOURCES[name]), "rb").read()).hexdigest(),
        "n_pairs": len(pairs), "n_lead": len(lead), "n_rest": len(rest),
        "lead_mean": ml, "rest_mean": mr, "ratio": ratio,
        "permutation": perm,
        "distance_medians": {str(d): st.median(v) for d, v in sorted(by.items())},
        "control_vs_full": next((r["vs_full"] for r in recs if r.get("kind") == "control"), None),
    }


def selftest():
    # positive control: a synthetic set with a real lead effect must be flagged
    lead = [1.0] * 5 + [1.2] * 5
    rest = [0.01] * 20
    r, ml, mr = ratio_of(lead, rest)
    assert r > 50 and abs(ml - 1.1) < 1e-9, (r, ml, mr)
    p = permutation(lead, rest, r)
    assert p["p_value"] <= 0.001, p
    # negative control: no lead effect -> ratio near 1 and permutation not extreme
    flat = [0.5] * 25
    r2, _, _ = ratio_of(flat[:10], flat[10:])
    assert abs(r2 - 1.0) < 1e-9, r2
    p2 = permutation(flat[:10], flat[10:], r2)
    assert p2["p_value"] > 0.05, p2
    # real data: control must be bit-exact, both splits non-empty
    for name in SOURCES:
        a = analyse(name)
        assert a["control_vs_full"] == 0.0, (name, a["control_vs_full"])
        assert a["n_lead"] > 0 and a["n_rest"] > 0, name
        assert a["permutation"]["p_value"] <= 0.001, (name, a["permutation"])
    print("selftest OK (positive lead, null flat, 4 real controls + permutation gates)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "cross_family.json"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    result = {name: analyse(name) for name in SOURCES}
    fam = [result["Qwen2.5-0.5B"]["ratio"], result["Qwen2.5-1.5B"]["ratio"],
           result["Qwen2.5-7B"]["ratio"]]
    out = {
        "protocol": {"k": 256, "seq": 512, "n_batches": 20,
                     "window": MAX_DISTANCE, "corpus": "wt2_test.txt",
                     "attn": "eager", "routing": "single-pair borrowed"},
        "models": result,
        "qwen_ratios": fam,
        "cross_family_ratio": result["TinyLlama-1.1B"]["ratio"],
        "verdict": ("REPLICATED across families" if
                    all(v >= 2.0 for v in fam) and result["TinyLlama-1.1B"]["ratio"] >= 2.0
                    and result["TinyLlama-1.1B"]["permutation"]["p_value"] <= 0.001
                    else "NOT replicated"),
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if args.json:
        print(text)
    print(f"verdict={out['verdict']} qwen={[round(v,2) for v in fam]} "
          f"tinyllama={round(out['cross_family_ratio'],2)} out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
