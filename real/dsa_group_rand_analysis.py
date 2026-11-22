"""Numeric gate for the trained-sparse null: reuse cost against an arbitrary set.

Three states are compared per cell, all on the same batches in one run:
  reuse = follower borrows the group leader's trained selection
  rand  = the same followers get an arbitrary k-set of the same size
  base  = dense (no sharing)
The quantity the paper reports is reuse - own, and the null asks how much of it
is simply "a set this layer did not choose". The ratio must be computed per
batch and then averaged, not as a ratio of two separately-averaged numbers --
otherwise a batch whose null is small can dominate the answer.
"""
import json, math, statistics as st, sys

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
DENSE = {"2": 0.0242, "3": 0.0114, "6": 0.0736}      # tab:grid, 0.5B, same keep fraction


def t_ci(vals):
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    m, sd = st.mean(vals), st.stdev(vals)
    return (m - T975.get(n - 1, 1.96) * sd / math.sqrt(n),
            m + T975.get(n - 1, 1.96) * sd / math.sqrt(n))


def analyse(d):
    if "cells" not in d:
        raise SystemExit("this artifact has no null cells (%s); the null run is "
                         "dsa_group_rand.json, the plain sharing run is dsa_group.json"
                         % ",".join(sorted(d)))
    out = {"model": d["model"], "seq": d["seq"], "n_batches": d["n_batches"],
           "topk": d.get("topk"), "keep_fraction": d["keep_fraction"],
           "n_layers": d["n_layers"], "seed": d.get("seed"),
           "same_batch_repeat_range": d["same_batch_repeat_range"],
           "cells": {}}
    for g, v in d["cells"].items():
        # paired per batch: the ratio is a property of the batch, so average the
        # per-batch ratios rather than a ratio of means
        per = [r / n for r, n in zip(v["reuse_delta"], v["rand_delta"]) if n > 0]
        lo, hi = t_ci(per)
        rlo, rhi = t_ci(v["reuse_delta"])
        nlo, nhi = t_ci(v["rand_delta"])
        out["cells"][g] = {
            "reuse_mean": v["reuse_mean"], "rand_mean": v["rand_mean"],
            "reuse_ci": [rlo, rhi], "rand_ci": [nlo, nhi],
            "reuse_over_rand_per_batch": per, "n": len(per),
            "ratio_mean": st.mean(per) if per else float("nan"),
            "ratio_ci": [lo, hi],
            "ratio_of_means": v["ratio_mean"],
            "rand_over_reuse": (v["rand_mean"] / v["reuse_mean"]) if v["reuse_mean"] > 0 else float("nan"),
            "dense_same_keep": DENSE.get(g),
            "dense_over_trained": (DENSE[g] / v["reuse_mean"]) if g in DENSE and v["reuse_mean"] > 0 else float("nan"),
            "rand_exceeds_reuse_every_batch": all(r < n for r, n in zip(v["reuse_delta"], v["rand_delta"])),
        }
    out["sum_reuse"] = sum(c["reuse_mean"] for c in out["cells"].values())
    out["sum_rand"] = sum(c["rand_mean"] for c in out["cells"].values())
    out["sum_ratio"] = (out["sum_reuse"] / out["sum_rand"]) if out["sum_rand"] > 0 else float("nan")
    return out


def selftest():
    # (a) reproduce a known t interval
    v = [1.0, 2.0, 3.0]
    lo, hi = t_ci(v)
    assert abs((hi - lo) - 2 * 4.303 * 1.0 / math.sqrt(3)) < 1e-9
    # (b) prove-red 1: a ratio-of-means and a mean-of-ratios must disagree when
    #     one batch has a tiny null -- if these always agreed the choice would
    #     be cosmetic and we would not have to defend it
    fake = {"model": "synthetic", "seq": 8, "n_batches": 2, "topk": 4,
            "keep_fraction": 0.5, "n_layers": 4, "seed": 7,
            "same_batch_repeat_range": 0.0,
            "cells": {"2": {"reuse_mean": 0.5, "rand_mean": 0.5, "ratio_mean": 1.0,
                            "reuse_delta": [0.9, 0.1], "rand_delta": [0.1, 0.9]}}}
    a = analyse(fake)
    assert abs(a["cells"]["2"]["ratio_of_means"] - 1.0) < 1e-9
    # 0.9/0.1 = 9.0 and 0.1/0.9 = 0.111..., so the mean of ratios is 4.5556,
    # NOT 1.0 (the ratio of means). Hand-computing this as "5" is the trap.
    assert abs(a["cells"]["2"]["ratio_mean"] - 4.555555555555555) < 1e-9, a["cells"]["2"]["ratio_mean"]
    # (c) prove-red 2: a cell where the null is NOT bigger than reuse must be
    #     reported as such rather than silently coerced to a nice ratio
    fake2 = dict(fake); fake2 = json.loads(json.dumps(fake))
    fake2["cells"]["2"]["reuse_delta"] = [0.5, 0.5]
    fake2["cells"]["2"]["rand_delta"] = [0.4, 0.4]
    b = analyse(fake2)
    assert b["cells"]["2"]["rand_exceeds_reuse_every_batch"] is False
    print("SELFTEST OK: t interval reproduces; ratio-of-means vs mean-of-ratios "
          "disagree as they must; a null smaller than reuse is reported as such")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); raise SystemExit(0)
    src = sys.argv[1] if len(sys.argv) > 1 else "/root/attn_newstack/dsa_group_rand.json"
    res = analyse(json.load(open(src)))
    dest = sys.argv[2] if len(sys.argv) > 2 else "attn-reuse/real/data/dsa_group_rand_analysis.json"
    json.dump(res, open(dest, "w"), indent=1)
    print("%-4s %12s %12s %8s %18s %10s %8s" %
          ("g", "reuse", "rand", "ratio", "95% CI (t)", "dense", "dense/x"))
    for g, c in sorted(res["cells"].items()):
        print("%-4s %+12.6f %+12.6f %8.4f  [%+.5f, %+.5f] %10.4f %8.1f" %
              (g, c["reuse_mean"], c["rand_mean"], c["ratio_mean"],
               c["ratio_ci"][0], c["ratio_ci"][1],
               c["dense_same_keep"] or 0, c["dense_over_trained"]))
    print("SUM  %+12.6f %+12.6f %8.4f" % (res["sum_reuse"], res["sum_rand"], res["sum_ratio"]))
