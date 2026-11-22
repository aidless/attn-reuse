#!/usr/bin/env python3
"""Formal analysis of deliver_mass as a mechanism for the leader anomaly.

Question
--------
Why is borrowing the selected token set from layers 0-1 (the "leaders")
disproportionately costly, when the same operation on layers 2+ is nearly free?

Hypothesis
----------
The positions selected by an early leader carry little mass under a *later*
layer's own dense attention distribution ("deliver_mass" low), so forcing that
later layer to attend only there destroys information.  This is distinct from
"the leader's selection is dissimilar" -- self_mass shows the leader's own
selection is high-mass under its own distribution.

Inputs (both GPU-free, already on disk)
---------------------------------------
  mech_*.json       dense-attention probe: (source,target) -> deliver_mass,
                    plus self_mass and the early/late summary
  pairs20_*.jsonl   masked-NLL probe: (source,target) -> vs_own, distance

Battery
-------
  A  corr(deliver, vs_own)                   primary predictive claim
  B  corr(deliver, source_layer)             layer-number confound
  C  partial(deliver, vs_own | source)       is it only depth?
  D  partial(deliver, vs_own | distance)     is it only distance?
  E  corr on late-only pairs (source >= 2)   does it hold inside the plateau?
  F  corr(deliver_shuffled, vs_own)          negative control (should be ~0)
  G  mean within-source corr                 does it hold at fixed leader?
  H  per-distance corr                       does it hold at fixed distance?

A note on tautology
-------------------
deliver_mass is measured on *dense* attention (no reuse mask), whereas vs_own
is measured after restricting the target to the leader's k positions.  They
are related by construction but not identical: a layer can give a position
dense mass yet decode differently once value vectors and downstream layers are
involved.  The informative result is not that A is negative -- it is that the
three "similarity" measures (index overlap, CKA, contribution magnitude) all
failed to predict vs_own while this *directional* mass measure does, and that
A survives the layer-number confound (C) that made the contribution magnitude
collapse.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# correlation primitives
# ---------------------------------------------------------------------------

def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": 0.0, "p": 1.0, "n": len(x)}
    r, p = sps.pearsonr(x, y)
    return {"r": float(r), "p": float(p), "n": int(len(x))}


def _residual(y, Z):
    y = np.asarray(y, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.size == 0:
        return y
    Z = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ beta


def partial_pearson(x, y, Z):
    rx = _residual(x, Z)
    ry = _residual(y, Z)
    return pearson(rx, ry)


# ---------------------------------------------------------------------------
# data loading / join
# ---------------------------------------------------------------------------

def load_pairs(jsonl_path):
    out = {}
    with open(jsonl_path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") != "pair":
                continue
            out[(int(r["source"]), int(r["target"]))] = {
                "vs_own": float(r["vs_own"]),
                "distance": int(r["distance"]),
                "k": int(r.get("k", 0)),
            }
    return out


def load_mech(json_path):
    d = json.load(open(json_path))
    pairs = {(int(p["source"]), int(p["target"])): float(p["deliver_mass"])
             for p in d["pairs"]}
    return d, pairs


def join(mech_pairs, cost_pairs):
    keys = sorted(set(mech_pairs) & set(cost_pairs))
    rows = []
    for key in keys:
        s, t = key
        rows.append({
            "source": s, "target": t, "distance": cost_pairs[key]["distance"],
            "deliver": mech_pairs[key], "vs_own": cost_pairs[key]["vs_own"],
        })
    return rows


# ---------------------------------------------------------------------------
# the battery
# ---------------------------------------------------------------------------

def analyze(rows, seed=20261119, n_perm=10000):
    deliver = np.array([r["deliver"] for r in rows])
    cost = np.array([r["vs_own"] for r in rows])
    source = np.array([r["source"] for r in rows], float)
    distance = np.array([r["distance"] for r in rows], float)

    out = {"n": len(rows)}
    out["A_deliver_vs_cost"] = pearson(deliver, cost)
    out["B_deliver_vs_layer"] = pearson(deliver, source)
    out["C_partial_by_layer"] = partial_pearson(deliver, cost, source)
    out["D_partial_by_distance"] = partial_pearson(deliver, cost, distance)

    late = [r for r in rows if r["source"] >= 2]
    out["E_late_only"] = pearson([r["deliver"] for r in late],
                                 [r["vs_own"] for r in late])

    # negative control: shuffle deliver, keep the layer confound intact in B
    rnd = random.Random(seed)
    shuffled = list(deliver)
    rnd.shuffle(shuffled)
    out["F_shuffled"] = pearson(shuffled, cost)

    # within-source: correlate at fixed leader, then average (n-weighted)
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    within = []
    for s, rs in sorted(by_source.items()):
        c = pearson([r["deliver"] for r in rs], [r["vs_own"] for r in rs])
        if c["n"] >= 4:
            within.append((s, c))
    out["G_within_source"] = {
        "per_source": {str(s): c for s, c in within},
        "mean_r": float(np.mean([c["r"] for _, c in within])) if within else 0.0,
        "n_sources": len(within),
    }

    by_dist = {}
    for r in rows:
        by_dist.setdefault(r["distance"], []).append(r)
    per_dist = {}
    for d, rs in sorted(by_dist.items()):
        per_dist[str(d)] = pearson([r["deliver"] for r in rs],
                                   [r["vs_own"] for r in rs])
    out["H_per_distance"] = per_dist

    # permutation p for the primary A: shuffle cost labels
    obs = out["A_deliver_vs_cost"]["r"]
    if not math.isfinite(obs):
        obs = 0.0
    ge = 0
    rnd2 = random.Random(seed + 1)
    perm = list(cost)
    for _ in range(n_perm):
        rnd2.shuffle(perm)
        rp = float(np.corrcoef(deliver, perm)[0, 1])
        if abs(rp) >= abs(obs) - 1e-15:
            ge += 1
    out["A_permutation_p"] = (ge + 1) / (n_perm + 1)
    out["A_permutation_n"] = n_perm
    out["seed"] = seed
    return out


# ---------------------------------------------------------------------------
# self-test: positive / negative / confound / tautology-guard fixtures
# ---------------------------------------------------------------------------

def make_rows(deliver, cost, source, distance):
    return [{"deliver": float(a), "vs_own": float(b),
             "source": int(c), "distance": int(d)}
            for a, b, c, d in zip(deliver, cost, source, distance)]


def selftest():
    rnd = random.Random(7)
    n = 60
    src = [0, 1] * 10 + [2] * 40
    dist = [1 if s < 2 else 2 for s in src]

    # fixture 1 (positive): low deliver -> high cost, real signal
    deliv = [0.85 if s < 2 else 0.98 for s in src]
    cost = [0.05 - 0.9 * (v - 0.85) + 0.002 * rnd.random() for v in deliv]
    a = analyze(make_rows(deliv, cost, src, dist), n_perm=500)
    assert a["A_deliver_vs_cost"]["r"] < -0.8, a["A_deliver_vs_cost"]
    assert a["A_permutation_p"] < 0.01, a["A_permutation_p"]
    assert a["F_shuffled"]["r"] > -0.3, a["F_shuffled"]

    # fixture 2 (negative): deliver and cost independent
    deliv = [0.9 + 0.02 * rnd.random() for _ in range(n)]
    cost = [0.01 + 0.01 * rnd.random() for _ in range(n)]
    a = analyze(make_rows(deliv, cost, src, dist), n_perm=500)
    assert abs(a["A_deliver_vs_cost"]["r"]) < 0.3, a["A_deliver_vs_cost"]
    assert a["A_permutation_p"] > 0.05, a["A_permutation_p"]

    # fixture 3 (confound guard): both deliver and cost scale with layer only.
    # Raw A looks large, but partial by layer and within-source both collapse.
    deliv = [0.8 + 0.02 * s + 0.001 * rnd.random() for s in src]
    cost = [0.001 * s + 0.0001 * rnd.random() for s in src]
    a = analyze(make_rows(deliv, cost, src, dist), n_perm=500)
    assert a["B_deliver_vs_layer"]["r"] > 0.8, a["B_deliver_vs_layer"]
    assert abs(a["C_partial_by_layer"]["r"]) < 0.3, a["C_partial_by_layer"]
    assert abs(a["G_within_source"]["mean_r"]) < 0.3, a["G_within_source"]

    print("SELFTEST OK")
    print(f"  positive A={a['A_deliver_vs_cost']['r']:.3f} (fixture 1, from last run)")
    print(f"  confound-guard partial={a['C_partial_by_layer']['r']:.3f} (fixture 3)")
    return True


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mech", nargs="+", help="mech_*.json path:TAG")
    ap.add_argument("--pairs", nargs="+", help="pairs20_*.jsonl path:TAG")
    ap.add_argument("--out", default="attn-reuse/real/data/deliver_corr.json")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    def untag(items):
        d = {}
        for item in items:
            path, tag = item.rsplit(":", 1)
            d[tag] = (path, None)
        return d

    mech = {}
    for item in args.mech:
        path, tag = item.rsplit(":", 1)
        mech[tag] = path
    pairs = {}
    for item in args.pairs:
        path, tag = item.rsplit(":", 1)
        pairs[tag] = path

    result = {"models": {}}
    for tag in sorted(set(mech) & set(pairs)):
        meta, mp = load_mech(mech[tag])
        cp = load_pairs(pairs[tag])
        rows = join(mp, cp)
        res = analyze(rows)
        res["mech_summary"] = meta["summary"]
        res["self_mass"] = meta["self_mass"]
        result["models"][tag] = res
        print(f"{tag}: n={res['n']} A={res['A_deliver_vs_cost']['r']:+.4f} "
              f"(perm p={res['A_permutation_p']:.5f}) "
              f"B={res['B_deliver_vs_layer']['r']:+.4f} "
              f"C={res['C_partial_by_layer']['r']:+.4f} "
              f"E={res['E_late_only']['r']:+.4f} "
              f"F={res['F_shuffled']['r']:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
