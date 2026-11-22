#!/usr/bin/env python3
"""How much of the leader anomaly does deliver-mass actually explain? (rev-da I6)

Round 5 adversarial review, rev-da I6: the selection *shape* is nearly identical
across layers (top-k mass 0.9928-0.9995, local share ~0.279, position-0 share
~0.003), yet early leaders' deliver-mass is only ~6 points below late leaders'
(0.9087 vs 0.9722 on 0.5B) while their reuse cost is 11.9-12.4x higher. A 6-point
mass gap driving a 12x cost gap is an extreme nonlinearity that the paper never
explains. If deliver-mass were the mechanism, moving it by the observed amount
should move the cost by the observed amount.

This script tests that quantitatively, GPU-free, from the same artifacts as
leader_deliver_analysis.py / dummy_controls.py:

  support   fraction of early-leader pairs whose deliver lies inside the late
            group's [min,max] range. If ~0, the two groups do not overlap in
            deliver at all and no matched comparison is possible -- deliver is
            then almost collinear with the binary leader flag.
  slope     OLS slope of cost ~ deliver fitted on LATE leaders only (the regime
            where reuse is cheap and deliver varies).
  explained extrapolate that late slope to the early group's mean deliver and
            ask what fraction of the actual early-minus-late cost gap it
            predicts. This is deliberately generous to deliver: it assumes the
            late-group linear response holds unchanged in the early regime.
  residual  1 - explained: the part of the leader anomaly that deliver-mass,
            even under this generous linear extrapolation, cannot account for.

The extrapolation is out-of-support whenever `support` is low, so the number is
an upper bound on deliver's explanatory reach, not a causal decomposition. We
report both, and we say plainly when there is no common support.

Selftest fixtures (the gate must be able to go red):
  full      : cost is exactly linear in deliver across BOTH groups -> explained
              ~ 1.0 (deliver fully accounts for the gap).
  none      : cost is independent of deliver but differs by group -> explained
              ~ 0 (deliver explains nothing; the gap is pure leader identity).
  partial   : cost = linear(deliver) + group offset -> explained strictly
              between 0 and 1, matching the analytic fraction.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dummy_controls import load_mech_any
from leader_deliver_analysis import load_pairs, join


def magnitude(rows):
    """rows: joined dicts with source/deliver/vs_own. Returns the battery."""
    E = [r for r in rows if r["source"] <= 1]
    L = [r for r in rows if r["source"] >= 2]
    assert E and L, ("need both early and late leaders", len(E), len(L))
    dE = np.array([r["deliver"] for r in E])
    dL = np.array([r["deliver"] for r in L])
    cE = np.array([r["vs_own"] for r in E])
    cL = np.array([r["vs_own"] for r in L])

    lo, hi = float(dL.min()), float(dL.max())
    support = float(np.mean((dE >= lo) & (dE <= hi)))
    # OLS cost ~ deliver on the late group only.
    b, a = np.polyfit(dL, cL, 1)
    pred_early = a + b * float(dE.mean())
    late_at_mean = a + b * float(dL.mean())
    gap_actual = float(cE.mean() - cL.mean())
    gap_pred = float(pred_early - late_at_mean)
    explained = gap_pred / gap_actual if abs(gap_actual) > 1e-12 else float("nan")
    return {
        "n_early": len(E), "n_late": len(L),
        "deliver_early_mean": float(dE.mean()),
        "deliver_late_mean": float(dL.mean()),
        "deliver_gap": float(dL.mean() - dE.mean()),
        "cost_early_mean": float(cE.mean()),
        "cost_late_mean": float(cL.mean()),
        "cost_ratio": float(cE.mean() / cL.mean()) if cL.mean() > 0 else float("inf"),
        "late_support_lo": lo, "late_support_hi": hi,
        "early_deliver_max": float(dE.max()), "early_deliver_min": float(dE.min()),
        "early_in_late_support": support,
        "late_slope": float(b), "late_intercept": float(a),
        "gap_actual": gap_actual, "gap_predicted_by_deliver": gap_pred,
        "explained_fraction": float(explained),
        "residual_fraction": float(1.0 - explained) if explained == explained else float("nan"),
        "out_of_support": bool(support < 0.5),
    }


def selftest():
    rng = np.random.default_rng(7)

    def mk(deliver_e, deliver_l, cost_fn):
        rows = []
        for s, ds in ((0, deliver_e), (5, deliver_l)):
            for d in ds:
                rows.append({"source": s, "target": s + 1,
                             "deliver": float(d), "vs_own": float(cost_fn(s, d))})
        return rows

    # full: cost = -1*(deliver-0.95), identical law in both groups.
    de = np.linspace(0.86, 0.94, 12)
    dl = np.linspace(0.92, 0.99, 40)
    full = magnitude(mk(de, dl, lambda s, d: -(d - 0.95)))
    assert abs(full["explained_fraction"] - 1.0) < 1e-6, full["explained_fraction"]

    # none: cost depends only on group, not on deliver.
    none = magnitude(mk(de, dl, lambda s, d: 0.05 if s <= 1 else 0.001))
    assert abs(none["explained_fraction"]) < 1e-6, none["explained_fraction"]

    # partial: cost = -0.5*(deliver-0.95) + (0.02 if early else 0).
    part = magnitude(mk(de, dl, lambda s, d: -0.5 * (d - 0.95) + (0.02 if s <= 1 else 0.0)))
    dgap = float(dl.mean() - de.mean())
    lin = 0.5 * dgap
    total = lin + 0.02
    want = lin / total
    assert abs(part["explained_fraction"] - want) < 1e-6, (part["explained_fraction"], want)
    assert 0.0 < part["explained_fraction"] < 1.0

    print("SELFTEST OK")
    print("  full    explained=%.4f" % full["explained_fraction"])
    print("  none    explained=%.4f" % none["explained_fraction"])
    print("  partial explained=%.4f (analytic %.4f)" % (part["explained_fraction"], want))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mech", nargs="+", required=False)
    ap.add_argument("--pairs", nargs="+", required=False)
    ap.add_argument("--labels", nargs="+", required=False)
    ap.add_argument("--out", default="attn-reuse/real/data/leader_magnitude.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    assert a.mech and a.pairs and a.labels and \
        len(a.mech) == len(a.pairs) == len(a.labels), "need matched --mech/--pairs/--labels"
    out = {}
    for mf, pf, label in zip(a.mech, a.pairs, a.labels):
        _, mp = load_mech_any(mf)
        cp = load_pairs(pf)
        out[label] = magnitude(join(mp, cp))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)
    for label, r in out.items():
        print("%-7s n=%3d/%3d deliver_gap=%.4f cost_ratio=%.1fx "
              "explained=%.3f residual=%.3f support=%.0f%%%s" %
              (label, r["n_early"], r["n_late"], r["deliver_gap"], r["cost_ratio"],
               r["explained_fraction"], r["residual_fraction"],
               r["early_in_late_support"] * 100,
               " OUT-OF-SUPPORT" if r["out_of_support"] else ""))


if __name__ == "__main__":
    main()