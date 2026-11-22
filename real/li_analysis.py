"""Read the LatentIndex four-cell run and answer one question.

IndexCache followers all read the leader's single projection.  LatentIndex
followers each read their own offset's projection out of the leader's route
list -- that is what "layer-specific scoring" means in the paper it comes
from.  So the two checkpoints differ in exactly one thing at the follower:

    offset0  : follower scores with the leader's projection
    own      : follower scores with its own projection

If layer-specific scoring helps, the cost of substituting the set should be
smaller when the follower uses its own score.  `null_own / null_off0` is
that comparison, measured on the same batches in the same run.
"""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def t_ci(vals, level=0.95):
    n = len(vals)
    m = sum(vals) / n
    sd = st.stdev(vals) if n > 1 else 0.0
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 2.0)
    half = t * sd / (n ** 0.5)
    return {"n": n, "mean": m, "sd": sd, "ci": [m - half, m + half],
            "all_positive": all(x > 0 for x in vals)}


def analyse(d):
    cells = d["deltas"]
    out = {"model": d["model"], "method": d["method"], "seq": d["seq"],
           "topk": d["topk"], "chunk": d["chunk"], "n_batches": d["n_batches"],
           "n_followers": d["n_followers"], "n_leaders": len(d["leaders"]),
           "base_mean": sum(d["base_per_batch"]) / len(d["base_per_batch"]),
           "repeat_delta": d["repeat_delta"], "cells": {}}
    for name in ("own", "offset0", "null_own", "null_off0"):
        out["cells"][name] = t_ci(cells[name])

    own = out["cells"]["own"]["mean"]
    off0 = out["cells"]["offset0"]["mean"]
    no = out["cells"]["null_own"]["mean"]
    n0 = out["cells"]["null_off0"]["mean"]

    out["control_own_is_zero"] = abs(own) < 1e-9
    out["cost_of_losing_own_score"] = off0 - own
    # ratio per batch, then averaged -- averaging two means of ratios is a
    # trap this project already fell into once
    per_batch_ratio = [a / b if b > 0 else float("inf") for a, b in zip(cells["null_own"], cells["null_off0"])]
    per_batch_offset = [a - b for a, b in zip(cells["offset0"], cells["own"])]
    out["null_own_over_null_off0_mean_of_ratios"] = sum(per_batch_ratio) / len(per_batch_ratio)
    out["null_own_over_null_off0_ratio_of_means"] = no / n0 if n0 > 0 else float("inf")
    out["offsets"] = {"per_batch": per_batch_offset,
                      "mean": sum(per_batch_offset) / len(per_batch_offset),
                      "all_positive": all(x > 0 for x in per_batch_offset),
                      "ci": t_ci(per_batch_offset)["ci"]}
    return out


def selftest():
    fake = {"model": "x", "method": "m", "seq": 4096, "topk": 2048, "chunk": 128,
            "n_batches": 3, "n_followers": 18, "leaders": [0, 3, 6],
            "base_per_batch": [1.0, 1.0, 1.0], "repeat_delta": 0.0,
            "deltas": {"own": [0.0, 0.0, 0.0], "offset0": [0.02, 0.02, 0.02],
                       "null_own": [0.03, 0.03, 0.03], "null_off0": [0.06, 0.06, 0.06]}}
    a = analyse(fake)
    assert a["control_own_is_zero"] is True, a
    assert abs(a["cost_of_losing_own_score"] - 0.02) < 1e-9, a
    # own score halves the substitution cost -> ratio must be exactly 0.5
    assert abs(a["null_own_over_null_off0_mean_of_ratios"] - 0.5) < 1e-12, a
    assert abs(a["null_own_over_null_off0_ratio_of_means"] - 0.5) < 1e-12, a

    # the trap: mean-of-ratios is NOT ratio-of-means.  Build cells where they
    # differ by a lot and assert the two fields really are different numbers.
    t = dict(fake)
    t["deltas"] = {"own": [0.0, 0.0, 0.0], "offset0": [0.02, 0.0, 0.0],
                   "null_own": [9.0, 0.1, 0.1], "null_off0": [0.1, 0.9, 0.9]}
    b = analyse(t)
    mor = b["null_own_over_null_off0_mean_of_ratios"]
    rom = b["null_own_over_null_off0_ratio_of_means"]
    assert abs(mor - rom) > 10.0, (mor, rom)

    # if the own-score cell were WORSE than the leader-score cell the ratio
    # would exceed 1 and the "layer-specific scoring helps" reading must fail
    u = dict(fake)
    u["deltas"] = {"own": [0.0, 0.0, 0.0], "offset0": [0.02, 0.02, 0.02],
                   "null_own": [0.09, 0.09, 0.09], "null_off0": [0.06, 0.06, 0.06]}
    c = analyse(u)
    assert c["null_own_over_null_off0_mean_of_ratios"] > 1.4, c
    print("SELFTEST OK: own-score control zero, cost visible, ratio 0.5 exact, "
          "mean-of-ratios differs from ratio-of-means, worse-cell flagged")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); raise SystemExit(0)
    runs = []
    for name in ("li_run.json",):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            runs.append(analyse(json.load(open(p))))
    ic = os.path.join(DATA, "ic_analysis.json")
    if os.path.exists(ic):
        d = json.load(open(ic))
        runs.append({"reference": "indexcache", "path": "real/data/ic_analysis.json",
                     "null_mean": d["cells"]["null"]["mean"],
                     "othergroup_mean": d["cells"]["othergroup"]["mean"],
                     "ratio": d["null_over_othergroup"]})
    out = os.path.join(DATA, "li_analysis.json")
    json.dump(runs if len(runs) > 1 else runs[0], open(out, "w"), indent=1)
    print("WROTE", out)
