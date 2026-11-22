#!/usr/bin/env python3
"""GPU-free incremental-value analysis: deliver_mass vs plain jaccard overlap.

Answers rev-da I1's sharpest question: is deliver_mass just a decorative
restatement of selection overlap? If so, after controlling for jaccard the
partial correlation of deliver with reuse cost should collapse to ~0 and the
incremental R^2 of adding deliver should be negligible.

Inputs:
  mech_h2h_*.json   (source,target) -> {deliver, jaccard}, same dense pass
  pairs20*.jsonl    (source,target) -> vs_own

Battery:
  A  corr(deliver, vs_own)
  B  corr(jaccard, vs_own)
  C  partial(deliver, vs_own | jaccard)     <-- does deliver add anything?
  D  partial(jaccard, vs_own | deliver)     <-- does overlap add anything?
  E  dR2 = R2(deliver+jaccard) - R2(jaccard) <-- incremental variance explained
  F  dR2 = R2(deliver+jaccard) - R2(deliver) <-- symmetric check

selftest has a POSITIVE fixture (deliver carries signal beyond overlap) and a
NEGATIVE fixture (deliver is a pure monotone function of overlap -> must show
partial ~0 and dR2 ~0, i.e. the gate can go red).
"""
import argparse
import json
import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# primitives (same as leader_deliver_analysis.py, kept self-contained)
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
    return pearson(_residual(x, Z), _residual(y, Z))


def r2(y, cols):
    """R^2 of OLS y ~ [1, cols...], via projection residual sum of squares."""
    y = np.asarray(y, float)
    Z = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    return 1.0 - float((resid ** 2).sum()) / ss_tot


# ---------------------------------------------------------------------------
# loading / join
# ---------------------------------------------------------------------------
def load_h2h(path):
    d = json.load(open(path))
    return d, {(int(p["source"]), int(p["target"])):
               (float(p["deliver"]), float(p["jaccard"])) for p in d["pairs"]}


def load_pairs(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") != "pair":
            continue
        out[(int(r["source"]), int(r["target"]))] = float(r["vs_own"])
    return out


def analyze(h2h, pairs):
    keys = sorted(set(h2h) & set(pairs))
    deliver = np.array([h2h[k][0] for k in keys])
    jacc = np.array([h2h[k][1] for k in keys])
    cost = np.array([pairs[k] for k in keys])
    A = pearson(deliver, cost)
    B = pearson(jacc, cost)
    C = partial_pearson(deliver, cost, jacc[:, None])
    D = partial_pearson(jacc, cost, deliver[:, None])
    r_j = r2(cost, [jacc])
    r_d = r2(cost, [deliver])
    r_both = r2(cost, [jacc, deliver])
    return {
        "n": len(keys),
        "A_deliver_vs_cost": A,
        "B_jaccard_vs_cost": B,
        "C_partial_deliver_given_jaccard": C,
        "D_partial_jaccard_given_deliver": D,
        "E_dR2_adding_deliver": r_both - r_j,
        "F_dR2_adding_jaccard": r_both - r_d,
        "r2_jaccard": r_j,
        "r2_deliver": r_d,
        "r2_both": r_both,
    }


# ---------------------------------------------------------------------------
# selftest (POSITIVE signal + NEGATIVE decorative-control)
# ---------------------------------------------------------------------------
def make_rows(n, rng, kind):
    jacc = np.clip(rng.normal(0.85, 0.03, n), 0, 1)
    if kind == "positive":
        # deliver has an independent component that drives cost
        deliver = np.clip(jacc + rng.normal(0, 0.02, n), 0, 1)
        cost = 0.10 - 0.6 * jacc - 0.8 * (deliver - jacc) + rng.normal(0, 0.002, n)
    else:  # decorative: deliver is a pure monotone function of jaccard
        deliver = 0.90 + 0.05 * (jacc - jacc.mean()) / (jacc.std() + 1e-12)
        cost = 0.10 - 0.6 * jacc + rng.normal(0, 0.002, n)
    h2h = {(0, i): (float(deliver[i]), float(jacc[i])) for i in range(n)}
    pairs = {(0, i): float(cost[i]) for i in range(n)}
    return h2h, pairs


def selftest():
    rng = np.random.default_rng(11)
    pos = analyze(*make_rows(120, rng, "positive"))
    neg = analyze(*make_rows(120, rng, "decorative"))
    # positive: deliver keeps a strong partial and adds variance
    assert pos["C_partial_deliver_given_jaccard"]["r"] < -0.5, pos["C_partial_deliver_given_jaccard"]
    assert pos["E_dR2_adding_deliver"] > 0.10, pos["E_dR2_adding_deliver"]
    # negative (the gate can go red): decorative deliver collapses
    assert abs(neg["C_partial_deliver_given_jaccard"]["r"]) < 0.25, neg["C_partial_deliver_given_jaccard"]
    assert neg["E_dR2_adding_deliver"] < 0.05, neg["E_dR2_adding_deliver"]
    print("SELFTEST OK")
    print("  positive: C=%.3f dR2=%.3f" %
          (pos["C_partial_deliver_given_jaccard"]["r"], pos["E_dR2_adding_deliver"]))
    print("  decorative: C=%.3f dR2=%.3f" %
          (neg["C_partial_deliver_given_jaccard"]["r"], neg["E_dR2_adding_deliver"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--h2h", nargs="+", required=False)
    ap.add_argument("--pairs", nargs="+", required=False)
    ap.add_argument("--labels", nargs="+", required=False)
    ap.add_argument("--out", default="attn-reuse/real/data/h2h_corr.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    out = {}
    for h2hp, pairsp, label in zip(a.h2h, a.pairs, a.labels):
        _, h2h = load_h2h(h2hp)
        pairs = load_pairs(pairsp)
        out[label] = analyze(h2h, pairs)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)
    for label, r in out.items():
        print("%-10s n=%3d A(deliv)=%+.3f B(jacc)=%+.3f C(part|jacc)=%+.3f "
              "dR2(+deliv)=%+.4f" %
              (label, r["n"], r["A_deliver_vs_cost"]["r"],
               r["B_jaccard_vs_cost"]["r"],
               r["C_partial_deliver_given_jaccard"]["r"],
               r["E_dR2_adding_deliver"]))


if __name__ == "__main__":
    main()
