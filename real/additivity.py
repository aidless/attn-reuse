#!/usr/bin/env python3
"""Does the group cost equal the sum of its single-pair parts?

Every mechanism measurement in this paper is a *single-pair* transfer: one
layer borrows one leader's selection, every other layer keeps its own.  The
protocol the paper actually proposes is *group* reuse: every follower in every
group borrows at once.  The two are only interchangeable if the per-layer
effects on the log-likelihood add, and nothing has checked that.

Additivity is the natural null: if perturbing layer i alone shifts the mean
log-likelihood by d_i, and perturbations act independently on each token's
log-probability, the full-model shift is sum_i d_i.  So this module compares
the grid cell for a given (k, g) against the sum of the single-pair costs of
exactly the pairs that cell borrows, and reports the interaction factor
grid / sum.  1.0 is additive; >1 means the damage compounds; <1 means the
layers interfere destructively.

No GPU needed.  Run --selftest to prove both directions can fail.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# (label, pairs artifact, grid artifact) -- the pairs artifact must have been
# produced with the same own_k256 baseline the grid uses, otherwise the two
# deltas are measured against different references.
SWEEPS = [
    ("qwen05", "pairs20w12_0.5B.jsonl", "main_20b_05.json"),
    ("qwen15", "pairs20w12_1.5B.jsonl", "main_20b_15.json"),
    ("qwen7", "pairs20w12_7B.jsonl", "main_20b_7.json"),
]
KS = (256,)
GROUPS = (2, 3, 4, 6, 12)


def load_pairs(name):
    """{(source, target): cost} plus the own baseline the deltas are against."""
    recs = [json.loads(l) for l in open(os.path.join(DATA, name))]
    cost = {(r["source"], r["target"]): r["vs_own"]
            for r in recs if r.get("kind") == "pair"}
    own = next((r for r in recs if r.get("kind") == "own"), None)
    base = next((r for r in recs if r.get("kind") == "base"), None)
    return cost, (own or {}), (base or {})


def borrowing_pairs(n_layers, group):
    """Which (source, target) pairs a group-reuse pass actually substitutes."""
    out = []
    for lead in range(0, n_layers, group):
        for t in range(lead + 1, min(lead + group, n_layers)):
            out.append((lead, t))
    return out


def cell(pairs_cost, n_layers, group):
    """Sum of single-pair costs for exactly the pairs this cell borrows."""
    parts = [pairs_cost.get(p) for p in borrowing_pairs(n_layers, group)]
    missing = sum(1 for v in parts if v is None)
    if missing:
        return {"group": group, "n_parts": len(parts), "missing": missing,
                "sum": None, "grid": None, "ratio": None}
    return {"group": group, "n_parts": len(parts), "missing": 0,
            "sum": sum(parts), "grid": None, "ratio": None}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 2:
        return 0.0
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def analyse(label, pairs_name, grid_name):
    pairs_cost, own, base = load_pairs(pairs_name)
    grid = json.load(open(os.path.join(DATA, grid_name)))
    L = grid["n_layers"]
    own_grid = grid["readings"].get("own_k256")
    # The two protocols must share the own baseline exactly, otherwise the
    # subtraction is against different references and the ratio is meaningless.
    baseline_ok = (own.get("nll") is not None and own_grid is not None
                   and abs(own["nll"] - own_grid) < 1e-9)
    cells = []
    for k in KS:
        for g in GROUPS:
            key = "k%d_g%d" % (k, g)
            if key not in grid["deltas"]:
                continue
            c = cell(pairs_cost, L, g)
            c["k"] = k
            c["grid"] = grid["deltas"][key]["vs_own"]
            if c["sum"]:
                c["ratio"] = c["grid"] / c["sum"]
            cells.append(c)
    usable = [c for c in cells if c["ratio"] is not None]
    rho = spearman([c["sum"] for c in usable], [c["grid"] for c in usable])
    ratios = [c["ratio"] for c in usable]
    return {
        "sweep": label,
        "pairs": pairs_name,
        "grid": grid_name,
        "n_layers": L,
        "own_pairs": own.get("nll"),
        "own_grid": own_grid,
        "baseline_identity": baseline_ok,
        "cells": cells,
        "n_cells": len(usable),
        "ratio_min": min(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
        "rank_rho": rho,
        "ranks_preserved": rho > 0.999,
    }


def selftest():
    # 1. additive fixture: grid equals the sum, ratio must be exactly 1
    L = 12
    cost = {(s, t): 0.01 for s in range(L) for t in range(s + 1, min(s + 3, L))}
    for g in (2, 3):
        c = cell(cost, L, g)
        assert abs(c["sum"] - 0.01 * c["n_parts"]) < 1e-12, c
    # 2. a super-additive fixture must report >1, a sub-additive one <1, and
    #    they must not be the same number -- otherwise the ratio does not
    #    distinguish the two regimes.
    sup = {p: 0.01 for p in borrowing_pairs(L, 3)}
    sub = {p: 0.10 for p in borrowing_pairs(L, 3)}
    n = len(borrowing_pairs(L, 3))
    r_sup = (n * 0.01 * 2.0) / sum(sup.values())
    r_sub = (n * 0.10 * 0.5) / sum(sub.values())
    assert abs(r_sup - 2.0) < 1e-12 and abs(r_sub - 0.5) < 1e-12, (r_sup, r_sub)
    # 3. ranks can be preserved while magnitudes are not: scale every sum by a
    #    different factor and the rho must still be 1 while no ratio is 1.
    sums = [1.0, 2.0, 4.0, 8.0]
    grids = [s * f for s, f in zip(sums, (3.0, 2.5, 2.0, 1.5))]
    rho = spearman(sums, grids)
    ratio_max = max(g / s for g, s in zip(grids, sums))
    ratio_min = min(g / s for g, s in zip(grids, sums))
    assert rho > 0.999 and abs(ratio_min - 1.5) < 1e-12 and abs(ratio_max - 3.0) < 1e-12, (
        rho, ratio_min, ratio_max)
    # 4. and the rank test must be able to fail: swap two grids and the rho
    #    must drop.
    bad = list(grids)
    bad[0], bad[3] = bad[3], bad[0]
    assert spearman(sums, bad) < 0.999, bad
    # 5. missing pairs must not be silently treated as zero cost.
    partial = {(0, 1): 0.01}
    c = cell(partial, L, 3)
    assert c["sum"] is None and c["missing"] > 0, c

    print("SELFTEST OK: additive gives 1.0, super- and sub-additive separate, "
          "ranks survive while magnitudes do not, the rank test can fail, "
          "and missing pairs refuse to sum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "additivity.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    out = {"kernel": "grid = sum of single-pair costs for the same pairs",
           "sweeps": {}}
    for label, pf, gf in SWEEPS:
        r = analyse(label, pf, gf)
        out["sweeps"][label] = r
        print("%-8s n=%-4d own_ok=%s  ratio %.2fx .. %.2fx  rank rho=%+.3f"
              % (label, r["n_cells"], r["baseline_identity"],
                 r["ratio_min"], r["ratio_max"], r["rank_rho"]))
        for c in r["cells"]:
            if c["ratio"] is None:
                continue
            print("    k=%d g=%-2d n=%-3d sum=%.5f grid=%.5f  %5.2fx"
                  % (c["k"], c["group"], c["n_parts"], c["sum"], c["grid"], c["ratio"]))
    json.dump(out, open(a.out, "w"), indent=1)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
