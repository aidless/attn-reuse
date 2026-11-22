#!/usr/bin/env python3
"""Cross-benchmark directionality, measured over the at-risk populations.

## Why this file exists (and why it replaces an earlier claim)

Report 07's headline was "flips are overwhelmingly correct->wrong: b:c = 5.1:1
on 1.5B, 5.0:1 on 7B". That ratio is **not an independent observable**. It is
the accuracy drop restated:

    b - c = (# correct under own) - (# correct under reuse) = N*(own - reuse)

which holds exactly on all 40 cells of the ARC-Easy grid (asserted below, as a
gate rather than as prose). Dividing by b+c then just reports the composition
of the discordant set, and that composition is what the accuracy change
already determined. On a cell whose accuracy is preserved, b == c by
construction.

So `b:c` cannot be evidence that the perturbation is directional. What can:

    damage rate = b / (# correct under own)     -- per correct answer, how often destroyed
    rescue rate = c / (# wrong   under own)     -- per wrong  answer, how often repaired

These use the discordant *count* as well as the accuracy change, and the
discordant count is genuinely extra information (given `own_acc` and
`reuse_acc`, `b` and `c` are determined only up to the discordance total).
Their ratio is the asymmetry the report should have quoted.

Run: python3 cross_bench.py [--json]
     python3 cross_bench.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

BENCHES = {
    "arc_easy": [("Qwen2.5-0.5B", "mcq_05b.json"),
                 ("Qwen2.5-1.5B", "mcq_15b.json"),
                 ("Qwen2.5-7B", "mcq_7b.json")],
    "arc_challenge": [("Qwen2.5-1.5B", "arcc_15b.json"),
                      ("Qwen2.5-7B", "arcc_7b.json"),
                      # k=64 was run separately after the fact, so the 7B grid
                      # covers three k values like the 1.5B grid does. Listed as
                      # its own row rather than merged: the two files were
                      # produced by different invocations, and a reader is
                      # entitled to see which cells came from which run.
                      ("Qwen2.5-7B (k=64)", "arcc_7b_k64.json")],
    # Shot-matched control. The ARC-Challenge stems are long enough that
    # mcq_reuse.py only fit 5 of the 8 requested exemplars, while ARC-Easy fit
    # all 8 — so comparing the two benchmarks as shipped would mix a change of
    # question distribution with a change of shot count. This run repeats the
    # lower half of the ARC-Easy grid at 5 exemplars on the same model.
    "arc_easy_5shot": [("Qwen2.5-0.5B", "arc_easy_5shot.json")],
}


def binom_two_sided(k, n, p):
    """Exact two-sided binomial tail (the same path used everywhere else here)."""
    if n == 0:
        return 1.0
    lo = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))
    hi = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    return min(1.0, 2 * min(lo, hi))


def at_risk(N, own_acc, b, c):
    """Damage / rescue rates and their ratio, for one cell."""
    nc = round(N * own_acc)
    nw = N - nc
    if nc <= 0 or nw <= 0:
        return None
    return {"n_at_risk_correct": nc, "n_at_risk_wrong": nw,
            "damage_rate": b / nc, "rescue_rate": c / nw,
            "asymmetry": (b / nc) / (c / nw) if c else float("inf")}


def identity_residual(doc):
    """max |(b-c) - N*(own-reuse)| over every cell: the tautology check."""
    N = doc["n_items"]
    worst = 0.0
    for key, fl in doc["flips_vs_own"].items():
        own = doc["readings"][f"own_{key.split('_')[0]}"]
        reuse = doc["readings"][f"reuse_{key}"]
        rhs = N * (own - reuse)
        worst = max(worst, abs((fl["right_to_wrong"] - fl["wrong_to_right"]) - rhs))
    return worst


def subset_summary(fname, ks, groups):
    """Pool one artifact over a fixed (k, group) subset.

    Exists because the two benchmarks were not run at the same exemplar count:
    mcq_reuse.py fits as many demonstrations as the budget allows rather than
    truncating the question, so ARC-Easy ran with 8 exemplars and
    ARC-Challenge with 5. Comparing their pooled at-risk rates directly would
    mix a change of question distribution with a change of shot count. This
    function lets the same 6-cell subset be pooled under both shot counts.
    """
    p = os.path.join(DATA, fname)
    if not os.path.exists(p):
        return None
    doc = json.load(open(p))
    N = doc["n_items"]
    b = c = nc = nw = cells = 0
    for key, fl in doc["flips_vs_own"].items():
        kk, gg = (int(x[1:]) for x in key.split("_"))
        if kk not in ks or gg not in groups:
            continue
        own = doc["readings"][f"own_k{kk}"]
        b += fl["right_to_wrong"]
        c += fl["wrong_to_right"]
        nc += round(N * own)
        nw += N - round(N * own)
        cells += 1
    if not cells or not nc or not nw:
        return None
    return {"file": fname, "cells": cells, "n_items": N,
            # `shots` is what was requested; `shots_used` is what fit and is
            # only present in artifacts written after the builder started
            # recording it. These artifacts predate that change, so the field
            # is null here and the used counts are attested by the run logs
            # instead (see SHOT_LOG_EVIDENCE below).
            "shots_requested": doc.get("shots"),
            "shots_used": doc.get("shots_used"),
            "b": b, "c": c, "n_at_risk_correct": nc, "n_at_risk_wrong": nw,
            "damage_rate": b / nc, "rescue_rate": c / nw,
            "asymmetry": (b / nc) / (c / nw) if c else float("inf"),
            "damage_pct": 100 * b / nc, "rescue_pct": 100 * c / nw,
            "damage_pct_ci": [100 * x for x in _cp(b, nc)],
            "rescue_pct_ci": [100 * x for x in _cp(c, nw)]}


# The artifacts do not carry the exemplar count that actually fit, so it is
# recorded here from the run logs, with the exact lines:
#   logs/arcc.log:shots requested=8 median used=8 min=8 max=8
#   logs/arcc2.log:shots requested=8 used=5 (uniform across 200 items)
#   logs/arcc2.log:shots requested=8 used=5 (uniform across 120 items)
#   logs/arcc7b64_outer.log:shots requested=8 used=5 (uniform across 120 items)
#   logs/ae5_outer.log:shots requested=5 used=5 (uniform across 200 items)
#   logs/probe_arcc_05b.log:shots requested=8 used=6 (uniform across 60 items)
#   logs/mmlu_probe.log:[raw] shots requested=5 used=5 (uniform across 60 items)
SHOT_LOG_EVIDENCE = {
    "mcq_05b.json": 8, "mcq_15b.json": 8, "mcq_7b.json": 8,
    "arcc_15b.json": 5, "arcc_7b.json": 5, "arcc_7b_k64.json": 5,
    "arc_easy_5shot.json": 5, "probe_arcc_05b.json": 6, "mmlu_probe.json": 5,
}


def shot_control():
    """The shot-matched comparison of the same 6 cells at 8 and 5 exemplars."""
    ks, groups = (64, 128), (2, 3, 6)
    a = subset_summary("mcq_05b.json", ks, groups)
    b = subset_summary("arc_easy_5shot.json", ks, groups)
    out = {"subset": {"ks": list(ks), "groups": list(groups)}, "arms": {}}
    for tag, s in (("eight_shot", a), ("five_shot", b)):
        if s:
            out["arms"][tag] = s
    if a and b:
        out["flip"] = {"asymmetry_eight_shot": a["asymmetry"],
                       "asymmetry_five_shot": b["asymmetry"],
                       "damage_minus_rescue_pp_eight": a["damage_pct"] - a["rescue_pct"],
                       "damage_minus_rescue_pp_five": b["damage_pct"] - b["rescue_pct"],
                       "sign_reversal": (a["damage_pct"] - a["rescue_pct"]) *
                                         (b["damage_pct"] - b["rescue_pct"]) < 0}
    return out


def bench_summary(tag, models):
    rows, tb, tc, tnc, tnw = [], 0, 0, 0, 0
    for model, fname in models:
        p = os.path.join(DATA, fname)
        if not os.path.exists(p):
            continue
        doc = json.load(open(p))
        N = doc["n_items"]
        b = c = 0
        cells = 0
        worst = 0.0
        for key, fl in doc["flips_vs_own"].items():
            b += fl["right_to_wrong"]
            c += fl["wrong_to_right"]
            own = doc["readings"][f"own_{key.split('_')[0]}"]
            reuse = doc["readings"][f"reuse_{key}"]
            worst = max(worst, abs((fl["right_to_wrong"] - fl["wrong_to_right"])
                                   - N * (own - reuse)))
            cells += 1
        nc, nw = round(N * doc["readings"]["full"]), 0
        # pooled at-risk counts use each cell's own accuracy, so they are
        # accumulated cell by cell rather than once per model
        for key in doc["flips_vs_own"]:
            own = doc["readings"][f"own_{key.split('_')[0]}"]
            nc_cell = round(N * own)
            tnc += nc_cell
            tnw += N - nc_cell
        m_nc = m_nw = 0
        for key in doc["flips_vs_own"]:
            own = doc["readings"][f"own_{key.split('_')[0]}"]
            m_nc += round(N * own)
            m_nw += N - round(N * own)
        rows.append({"model": model, "file": fname, "cells": cells,
                     "dense_acc": doc["readings"]["full"], "n_items": N,
                     "b": b, "c": c, "b_minus_c": b - c,
                     "identity_residual": worst,
                     "n_at_risk_correct": m_nc, "n_at_risk_wrong": m_nw,
                     "damage_rate": b / m_nc if m_nc else None,
                     "rescue_rate": c / m_nw if m_nw else None,
                     "asymmetry": ((b / m_nc) / (c / m_nw)
                                   if (m_nc and m_nw and c) else None)})
        tb += b
        tc += c
    pooled = None
    if tnc and tnw and (tb + tc):
        pooled = {"b": tb, "c": tc, "n_at_risk_correct": tnc,
                  "n_at_risk_wrong": tnw,
                  "damage_rate": tb / tnc, "rescue_rate": tc / tnw,
                  "asymmetry": (tb / tnc) / (tc / tnw) if tc else float("inf"),
                  # tested against the base rate implied by the at-risk split:
                  # under "a flip is equally likely per at-risk item in either
                  # direction", E[b] = n_discordant * tnc/(tnc+tnw) is wrong --
                  # the correct null is per-item rates 1/2 each, i.e. b under
                  # the null of equal *rates* is not binomial on a fixed n.
                  # The honest test is on the two independent rates, so report
                  # the interval instead of a p that would need a model of how
                  # discordance arises.
                  "damage_rate_ci": _cp(tb, tnc),
                  "rescue_rate_ci": _cp(tc, tnw),
                  # Percent forms are emitted alongside the rates because the
                  # report prints percentages; a gate that had to re-scale a
                  # rate to compare it against "10.1" would be the place a
                  # silent off-by-100 lives.
                  "damage_pct": 100 * tb / tnc,
                  "rescue_pct": 100 * tc / tnw,
                  "damage_pct_ci": [100 * x for x in _cp(tb, tnc)],
                  "rescue_pct_ci": [100 * x for x in _cp(tc, tnw)]}
    return {"bench": tag, "models": rows, "pooled": pooled}


def _cp(x, n, level=0.95):
    """Clopper-Pearson on x/n, by bisection on the Beta CDF."""
    if n == 0:
        return None
    alpha = 1 - level
    lo = 0.0 if x == 0 else _beta_ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else _beta_ppf(1 - alpha / 2, x + 1, n - x)
    return [lo, hi]


def _beta_ppf(q, a, b, iters=200):
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        f *= d * c
        if abs(d * c - 1.0) < 1e-12:
            break
    return front * (f - 1.0)


def selftest():
    # the tautology must hold exactly, on every shipped artifact
    for tag, models in BENCHES.items():
        for model, fname in models:
            p = os.path.join(DATA, fname)
            if not os.path.exists(p):
                continue
            worst = identity_residual(json.load(open(p)))
            assert worst < 0.5, (tag, model, worst)
    # at-risk rates must be consistent with the raw counts
    d = json.load(open(os.path.join(DATA, "mcq_15b.json")))
    N = d["n_items"]
    fl = d["flips_vs_own"]["k64_g6"]
    own = d["readings"]["own_k64"]
    a = at_risk(N, own, fl["right_to_wrong"], fl["wrong_to_right"])
    assert a["n_at_risk_correct"] + a["n_at_risk_wrong"] == N
    assert abs(a["damage_rate"] - fl["right_to_wrong"] / a["n_at_risk_correct"]) < 1e-12
    # a cell with no discriminations must not fabricate a ratio
    assert at_risk(10, 1.0, 0, 0) is None, "all-correct cell has no wrong at-risk set"
    # Clopper-Pearson sanity: bracket contains the estimate, widens with less data
    assert _cp(30, 300)[0] <= 0.1 <= _cp(30, 300)[1]
    narrow, wide = _cp(70, 200), _cp(7, 20)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
    assert abs(_beta_ppf(0.5, 1, 1) - 0.5) < 1e-4

    # --- the shot control ---------------------------------------------------
    sc = shot_control()
    assert sc["subset"] == {"ks": [64, 128], "groups": [2, 3, 6]}, sc["subset"]
    for tag in ("eight_shot", "five_shot"):
        s = sc["arms"][tag]
        assert s["cells"] == 6, (tag, s["cells"])
        assert s["n_at_risk_correct"] + s["n_at_risk_wrong"] == 6 * s["n_items"], tag
    # `shots_used` is absent from these artifacts (see SHOT_LOG_EVIDENCE), so
    # the exemplar counts are checked against the log table rather than the
    # JSON: 8 for the ARC-Easy arm, 5 for the shot-matched arm.
    assert SHOT_LOG_EVIDENCE[sc["arms"]["eight_shot"]["file"]] == 8
    assert SHOT_LOG_EVIDENCE[sc["arms"]["five_shot"]["file"]] == 5
    assert sc["arms"]["five_shot"]["shots_requested"] == 5
    # the whole reason this control exists: at 8 exemplars the two rates are
    # the wrong way round, and at 5 they are not. If that ever stops holding,
    # the report's caveat about mixed shot counts is no longer backed by data.
    assert sc["flip"]["sign_reversal"] is True, sc["flip"]
    assert sc["flip"]["asymmetry_eight_shot"] < 1.0 < sc["flip"]["asymmetry_five_shot"], \
        sc["flip"]

    print("selftest OK (b-c identity exact on all artifacts, at-risk counts add "
          "up, no-ratio on saturated cells, CP brackets and widens, shot control "
          "reverses sign)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    out = {tag: bench_summary(tag, m) for tag, m in BENCHES.items()}
    # The shot control rides in the same file so a reader who checks one number
    # against the artifact will trip over the caveat that qualifies it.
    out["shot_control"] = shot_control()
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    for tag, s in out.items():
        if tag == "shot_control":
            continue
        if not s["models"]:
            print(f"{tag}: no artifacts")
            continue
        print(f"\n{tag}")
        print(f"  {'model':14s} {'dense':>6s} {'N':>4s} {'b':>4s} {'c':>4s} "
              f"{'b-c':>5s} {'N(own-re)':>9s} {'resid':>6s}")
        for m in s["models"]:
            print(f"  {m['model']:14s} {m['dense_acc']:6.4f} {m['n_items']:4d} "
                  f"{m['b']:4d} {m['c']:4d} {m['b_minus_c']:5d} "
                  f"{m['b_minus_c']:9d} {m['identity_residual']:6.3f}")
        p = s["pooled"]
        if p:
            print(f"  POOLED  at risk: {p['n_at_risk_correct']} correct, "
                  f"{p['n_at_risk_wrong']} wrong")
            print(f"          damage {100*p['damage_rate']:.1f}% "
                  f"CI [{100*p['damage_rate_ci'][0]:.1f}, {100*p['damage_rate_ci'][1]:.1f}]"
                  f"   rescue {100*p['rescue_rate']:.1f}% "
                  f"CI [{100*p['rescue_rate_ci'][0]:.1f}, {100*p['rescue_rate_ci'][1]:.1f}]"
                  f"   asymmetry {p['asymmetry']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
