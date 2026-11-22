#!/usr/bin/env python3
"""Is distance an axis of its own, or an artifact of which sources it can reach?

The manuscript's `Distance is not the axis; the leader is` section reports a
median cost per *distance*: +0.00067 at 1, +0.00076 at 5, +0.00116 at 10, and
concludes that twelve times the distance buys less than twice the cost. The
profile is real, but it pools every source that can reach a given distance,
and in an L-layer stack a pair at distance d can only have source 0..L-1-d.
So the deep columns are populated by systematically shallower sources -- the
mean source at distance d is exactly (L-1-d)/2, which is arithmetic, not a
property of the model.

This script does what the paper's profile does not: it removes the source
composition before asking whether distance carries anything. Two estimators,
both stratified by source:

  * `within_source_wins` -- for every source observed at >= 2 distances, does
    the cost rise from its nearest to its farthest target? A trivial sign
    count, no assumptions.
  * `stratified_permutation` -- the statistic is the mean within-source
    Spearman correlation between distance and cost; the null permutes the
    cost values *inside* each source, which preserves each source's cost
    distribution exactly and destroys only the distance ordering.

Everything is computed from the published sweeps; no GPU and no new runs.
Run --selftest to prove the permutation can return a null and that a genuine
distance trend is detected.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED = 20261121
PERMUTATIONS = 20000

SWEEPS = {
    # name: (artifact, n_layers)
    "qwen05_w12": ("pairs20w12_0.5B.jsonl", 24),
    "qwen7_w12": ("pairs20w12_7B.jsonl", 28),
    "qwen14_w6": ("pairs14b.jsonl", 48),
    "tiny_w6": ("pairs20_TinyLlama.jsonl", 22),
}


def load_pairs(name: str):
    f, n_layers = SWEEPS[name]
    recs = [json.loads(l) for l in open(os.path.join(DATA, f))]
    pairs = [r for r in recs if r.get("kind") == "pair"]
    return pairs, n_layers


def by_source(pairs):
    """{source: [(distance, cost), ...]} -- the unit the stratification uses."""
    out = {}
    for r in pairs:
        out.setdefault(r["source"], []).append((r["target"] - r["source"], r["vs_own"]))
    return {s: sorted(v) for s, v in out.items()}


def _by_target(pairs):
    out = {}
    for r in pairs:
        out.setdefault(r["target"], []).append((r["target"] - r["source"], r["vs_own"]))
    return {t: sorted(v) for t, v in out.items()}


def composition_table(pairs, n_layers):
    """Which sources can appear at each distance, and what the data actually has.

    The point of the table is that `mean_source` is pinned by the geometry of
    the sweep: at distance d the sources are 0..L-1-d, so their mean is
    (L-1-d)/2 unless pairs are missing from that column. Comparing the two
    columns is what tells a reader the profile is confounded by construction.
    """
    rows = []
    ds = sorted({r["target"] - r["source"] for r in pairs})
    for d in ds:
        sel = [r for r in pairs if r["target"] - r["source"] == d]
        rows.append({
            "distance": d,
            "n": len(sel),
            "mean_source_observed": st.mean(r["source"] for r in sel),
            "mean_source_arithmetic": (n_layers - 1 - d) / 2.0,
            "median_cost": st.median(r["vs_own"] for r in sel),
            "mean_cost": st.mean(r["vs_own"] for r in sel),
        })
    return rows


def within_source_wins(grouped):
    """For each source with >= 2 distances, is the farthest cost above the nearest?"""
    rises = falls = 0
    used = 0
    for s, v in grouped.items():
        if len(v) < 2:
            continue
        used += 1
        first, last = v[0][1], v[-1][1]
        if last > first:
            rises += 1
        elif last < first:
            falls += 1
    return {"sources_used": used, "rises": rises, "falls": falls,
            "tie_or_flat": used - rises - falls}


def spearman(xs, ys):
    """Rank correlation with average ranks, so ties do not bias one way."""
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
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def stratified_statistic(grouped):
    """Mean within-source Spearman(distance, cost) over sources with >= 2 points."""
    vals = []
    for s, v in grouped.items():
        if len(v) < 2:
            continue
        ds = [d for d, _ in v]
        cs = [c for _, c in v]
        vals.append(spearman(ds, cs))
    return (st.mean(vals) if vals else 0.0), len(vals)


def stratified_permutation(grouped, observed, n=PERMUTATIONS, seed=SEED):
    """Null: shuffle the costs inside each source, so each source keeps its own
    distribution and only the distance ordering is destroyed."""
    keys = [s for s, v in grouped.items() if len(v) >= 2]
    rng = random.Random(seed)
    null = []
    for _ in range(n):
        vals = []
        for s in keys:
            v = grouped[s]
            ds = [d for d, _ in v]
            cs = [c for _, c in v]
            rng.shuffle(cs)
            vals.append(spearman(ds, cs))
        null.append(st.mean(vals))
    extreme = sum(1 for x in null if abs(x) >= abs(observed) - 1e-15)
    # two-sided: the profile does not predict a sign in advance
    p = (extreme + 1) / (n + 1)
    null_sorted = sorted(null)
    return {
        "observed": observed,
        "permutations": n,
        "seed": seed,
        "p_value": p,
        "null_mean": st.mean(null),
        "null_sd": st.pstdev(null),
        "_null_q": [null_sorted[int(0.025 * n)], null_sorted[int(0.975 * n)]],
    }


def ols_r2(rows, y):
    """R^2 of a least-squares fit with an intercept. Normal equations, solved by
    Gaussian elimination -- small enough that a dependency is not worth adding."""
    n = len(y)
    p = len(rows[0]) + 1
    A = [[1.0] + list(r) for r in rows]
    M = [[sum(A[k][i] * A[k][j] for k in range(n)) for j in range(p)]
         + [sum(A[k][i] * y[k] for k in range(n))] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return 0.0
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(p):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    b = [M[i][p] for i in range(p)]
    pred = [sum(b[i] * A[k][i] for i in range(p)) for k in range(n)]
    my = sum(y) / n
    sst = sum((v - my) ** 2 for v in y)
    sse = sum((y[k] - pred[k]) ** 2 for k in range(n))
    return (1.0 - sse / sst) if sst else 0.0


def additive_decomposition(pairs):
    """How much of the pair-cost variance does distance carry *once source is known*?

    The published profile is a marginal one: it varies distance and lets source
    drift. This asks the marginal question properly -- R^2(source), R^2(distance)
    alone, and R^2(both) -- so the distance share that is not already source is
    read off as the increment.
    """
    y = [r["vs_own"] for r in pairs]
    src = [float(r["source"]) for r in pairs]
    dist = [float(r["target"] - r["source"]) for r in pairs]
    r2s = ols_r2([[v] for v in src], y)
    r2d = ols_r2([[v] for v in dist], y)
    r2b = ols_r2([[a, b] for a, b in zip(src, dist)], y)
    return {"r2_source": r2s, "r2_distance": r2d, "r2_both": r2b,
            "r2_distance_increment": r2b - r2s}


def near_far(pairs):
    """Magnitude of the within-source distance effect: nearest vs farthest target.

    Both the mean and the median are reported because the two disagree in sign
    on one sweep -- a mean-only number would overstate the effect there.
    """
    by = {}
    for r in pairs:
        by.setdefault(r["source"], []).append((r["target"] - r["source"], r["vs_own"]))
    near, far = [], []
    for s, v in by.items():
        if len(v) < 2:
            continue
        v = sorted(v)
        near.append(v[0][1])
        far.append(v[-1][1])
    if not near:
        return {}
    return {"n_sources": len(near),
            "near_mean": st.mean(near), "far_mean": st.mean(far),
            "mean_delta": st.mean(far) - st.mean(near),
            "near_median": st.median(near), "far_median": st.median(far),
            "median_delta": st.median(far) - st.median(near)}


def within_target(grouped_by_target):
    """The same within-source test with the *target* held fixed instead.

    Holding the source fixed leaves target depth free to move with distance;
    holding the target fixed leaves source depth free to move against it. The
    two rho's answer different questions and both are reported, because a
    single within-source number cannot separate "later targets cost more" from
    "distant sources are shallower".
    """
    vals = []
    for t, v in grouped_by_target.items():
        if len(v) < 3:
            continue
        vals.append(spearman([d for d, _ in v], [c for _, c in v]))
    return {"mean_rho": st.mean(vals) if vals else 0.0, "targets_used": len(vals)}


def basis_check(pairs):
    """Is the source-vs-distance comparison unique, or an artefact of which
    variable was put first?

    ``target = source + distance`` holds for every pair, so the three columns
    span only two dimensions and any two of them generate the same space.  The
    decomposition in ``additive_decomposition`` orders the two predictors; this
    reports the same quantities under all three orderings, plus the near
    collinearity that makes the split unidentifiable, plus the two stratification
    identities that show a "within-target" rho is the leader effect read
    backwards rather than an independent regularity.
    """
    y = [r["vs_own"] for r in pairs]
    src = [float(r["source"]) for r in pairs]
    tgt = [float(r["target"]) for r in pairs]
    dist = [float(r["target"] - r["source"]) for r in pairs]

    r2s = ols_r2([[v] for v in src], y)
    r2t = ols_r2([[v] for v in tgt], y)
    r2d = ols_r2([[v] for v in dist], y)
    r2sd = ols_r2([[a, b] for a, b in zip(src, dist)], y)
    r2td = ols_r2([[a, b] for a, b in zip(tgt, dist)], y)
    r2st = ols_r2([[a, b] for a, b in zip(src, tgt)], y)

    ms, mt = st.mean(src), st.mean(tgt)
    cov = sum((a - ms) * (b - mt) for a, b in zip(src, tgt)) / len(src)
    ss = (sum((a - ms) ** 2 for a in src) / len(src)) ** 0.5
    tt = (sum((b - mt) ** 2 for b in tgt) / len(tgt)) ** 0.5
    corr = cov / (ss * tt) if ss and tt else 0.0

    # within-source: distance and target differ by a constant, so the two rho
    # are the same number.  within-target: distance and -source differ by a
    # constant, so again the same number -- and that number is the leader
    # effect, not a statement about targets.
    # by_source yields {source: [(distance, cost), ...]}; by _by_target yields
    # {target: [(distance, cost), ...]}.  Rebuild both depth columns from the
    # key plus the distance so the identity is tested on the real grouping
    # rather than on a re-derived one.
    bys, byt = by_source(pairs), _by_target(pairs)
    ws_d, ws_t, wt_d, wt_s = [], [], [], []
    for s, v in bys.items():
        if len(v) < 3:
            continue
        ws_d.append(spearman([d for d, _ in v], [c for _, c in v]))
        ws_t.append(spearman([s + d for d, _ in v], [c for _, c in v]))
    for t, v in byt.items():
        if len(v) < 3:
            continue
        wt_d.append(spearman([d for d, _ in v], [c for _, c in v]))
        wt_s.append(spearman([-(t - d) for d, _ in v], [c for _, c in v]))
    ws_d = st.mean(ws_d) if ws_d else 0.0
    ws_t = st.mean(ws_t) if ws_t else 0.0
    wt_d = st.mean(wt_d) if wt_d else 0.0
    wt_s = st.mean(wt_s) if wt_s else 0.0

    return {
        "r2_source": r2s, "r2_target": r2t, "r2_distance": r2d,
        "inc_distance_given_source": r2sd - r2s,
        "inc_distance_given_target": r2td - r2t,
        "inc_position_given_distance": r2sd - r2d,
        "inc_position_given_distance_target_side": r2td - r2d,
        "r2_two_var": max(r2sd, r2td, r2st),
        "r2_unexplained": 1.0 - max(r2sd, r2td, r2st),
        "corr_source_target": corr,
        "within_source_rho_distance": ws_d,
        "within_source_rho_target": ws_t,
        "within_source_identity": abs(ws_d - ws_t) < 1e-12,
        "within_target_rho_distance": wt_d,
        "within_target_rho_minus_source": wt_s,
        "within_target_identity": abs(wt_d - wt_s) < 1e-12,
    }

def analyse(name):
    pairs, n_layers = load_pairs(name)
    grouped = by_source(pairs)
    obs, n_used = stratified_statistic(grouped)
    return {
        "sweep": name,
        "n_pairs": len(pairs),
        "n_sources": len(grouped),
        "n_layers": n_layers,
        "composition": composition_table(pairs, n_layers),
        "within_source": within_source_wins(grouped),
        "stratified": dict(stratified_permutation(grouped, obs), sources=n_used),
        "unstratified_distance_spearman": spearman(
            [r["target"] - r["source"] for r in pairs], [r["vs_own"] for r in pairs]),
        "additive": additive_decomposition(pairs),
        "near_far": near_far(pairs),
        "within_target": within_target(_by_target(pairs)),
        "basis": basis_check(pairs),
    }


def selftest():
    # 1. composite check: the arithmetic column must equal the observed one only
    #    when the column is complete. A sweep with holes must disagree.
    # A complete sweep is one where every legal (source, target) pair is present.
    # The fixture has to respect the layer bound: target <= L-1, so a source at
    # L-1-d is the deepest one that can appear at distance d.  My first version
    # let source 9 emit target 10 in a 10-layer stack, which made the observed
    # mean 4.5 against the arithmetic 4.0 and the assertion fired.
    L = 10
    full = [{"source": s, "target": s + d, "vs_own": 0.0, "kind": "pair"}
            for s in range(L) for d in (1, 2) if s + d <= L - 1]
    rows = composition_table(full, L)
    for r in rows:
        assert abs(r["mean_source_observed"] - r["mean_source_arithmetic"]) < 1e-9, r
    holed = [r for r in full if not (r["source"] == 0 and r["target"] == 2)]
    rows_h = composition_table(holed, L)
    assert abs(rows_h[1]["mean_source_observed"] - rows_h[1]["mean_source_arithmetic"]) > 0.01, rows_h[1]

    # 2. prove-red on the sign count: a planted rising trend must be all-rises.
    rising = {s: [(d, float(d)) for d in (1, 2, 3)] for s in range(5)}
    w = within_source_wins(rising)
    assert w["rises"] == 5 and w["falls"] == 0, w

    # 3. prove-red on the permutation: a planted trend must beat its own null.
    rng = random.Random(7)
    planted = {s: [(d, d * 0.5 + rng.gauss(0, 0.01)) for d in (1, 2, 3, 4)]
               for s in range(30)}
    obs, k = stratified_statistic(planted)
    perm = stratified_permutation(planted, obs, n=2000)
    assert obs > 0.9, obs
    assert perm["p_value"] < 0.01, perm["p_value"]

    # 4. prove-red the other way: no trend must stay inside its null.
    flat = {s: [(d, rng.gauss(0, 1)) for d in (1, 2, 3, 4)] for s in range(30)}
    obs_f, _ = stratified_statistic(flat)
    perm_f = stratified_permutation(flat, obs_f, n=2000)
    assert abs(obs_f) < 0.3, obs_f
    assert perm_f["p_value"] > 0.01, perm_f["p_value"]

    # 5. prove-red on the decomposition: when distance is the only real driver,
    #    the increment must be the whole story; when source is, it must vanish.
    lines = [{"source": s, "target": s + d, "vs_own": float(d), "kind": "pair"}
             for s in range(20) for d in (1, 2, 3, 4)]
    dec = additive_decomposition(lines)
    assert dec["r2_distance_increment"] > 0.9, dec
    src_only = [{"source": s, "target": s + d, "vs_own": float(s), "kind": "pair"}
                for s in range(20) for d in (1, 2, 3, 4)]
    dec2 = additive_decomposition(src_only)
    assert dec2["r2_distance_increment"] < 0.01, dec2

    # 6. near_far must expose a mean/median disagreement rather than hide it:
    #    one large rising outlier with many small falling ones.
    mixed = [{"source": 0, "target": d, "vs_own": (100.0 if d == 4 else -0.1),
              "kind": "pair"} for d in (1, 2, 3, 4)]
    mixed += [{"source": s, "target": s + 4, "vs_own": -0.2, "kind": "pair"}
              for s in range(1, 6)]
    mixed += [{"source": s, "target": s + 1, "vs_own": 0.0, "kind": "pair"}
              for s in range(1, 6)]
    nf = near_far(mixed)
    # The property that matters is that the two statistics disagree *in sign*,
    # so the assertion is on sign disagreement, not on a magnitude threshold:
    # my first version asserted abs(median_delta) < 0.2 and the fixture landed
    # exactly on -0.2, which is a threshold artefact, not a failing gate.
    assert nf["mean_delta"] > 0 and nf["median_delta"] < 0, nf

    # 7. basis check: the three single-variable fits must satisfy
    #    R2(source) + inc(d|s) == R2(target) + inc(d|t) == R2(d) + inc(pos|d),
    #    because all three are the same two-variable fit written three ways.
    #    A fixture where the split is NOT order-invariant must be caught.
    grid = [{"source": s, "target": s + d, "vs_own": float((s - 5) ** 2) + 0.1 * d,
             "kind": "pair"} for s in range(11) for d in (1, 2) if s + d <= 11]
    b = basis_check(grid)
    lhs_a = b["r2_source"] + b["inc_distance_given_source"]
    lhs_b = b["r2_target"] + b["inc_distance_given_target"]
    lhs_c = b["r2_distance"] + b["inc_position_given_distance"]
    assert abs(lhs_a - lhs_b) < 1e-9 and abs(lhs_b - lhs_c) < 1e-9, (lhs_a, lhs_b, lhs_c)
    assert b["within_source_identity"] and b["within_target_identity"], b
    # the two orderings must give different *numbers* on a fixture where
    # distance and position carry different amounts -- otherwise the check
    # above would be satisfied by everything and prove nothing.
    assert abs(b["inc_distance_given_source"] - b["inc_position_given_distance"]) > 1e-3, b

    print("SELFTEST OK: composition is arithmetic, holed sweeps disagree, "
          "planted trend detected, flat data stays inside its null, "
          "sign count reads both directions, decomposition splits by driver, "
          "near/far exposes mean-median disagreement, basis orderings agree "
          "and the two stratification identities hold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "distance_axis.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    out = {"permutations": PERMUTATIONS, "seed": SEED,
           "sweeps": {n: analyse(n) for n in SWEEPS}}
    json.dump(out, open(a.out, "w"), indent=1)
    for n, r in out["sweeps"].items():
        w = r["within_source"]
        s = r["stratified"]
        print("%-11s n=%-4d  unstrat rho=%+.3f | stratified rho=%+.3f p=%.4f "
              "(rises %d / falls %d of %d sources)" % (
                  n, r["n_pairs"], r["unstratified_distance_spearman"],
                  s["observed"], s["p_value"], w["rises"], w["falls"], w["sources_used"]))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
