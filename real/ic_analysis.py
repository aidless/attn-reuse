"""What does it cost to break a sharing that was trained in?
"""
import json, statistics as st, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def t_ci(vals, level=0.95):
    """Two-sided t interval; n=3 -> 4.303."""
    n = len(vals)
    m = sum(vals) / n
    sd = st.stdev(vals) if n > 1 else 0.0
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.0)
    half = t * sd / (n ** 0.5)
    return m, m - half, m + half


def analyse(d, row):
    base = d["base_per_batch"]
    out = {"model": d["model"], "method": d["method"], "seq": d["seq"],
           "topk": d["topk"], "chunk": d["chunk"], "n_batches": d["n_batches"],
           "n_leaders": len(d["leaders"]), "n_followers": d["n_followers"],
           "base_mean": sum(base) / len(base), "cells": {}}
    for name in ("null", "othergroup"):
        per = d["deltas"][name]
        m, lo, hi = t_ci(per)
        out["cells"][name] = {"per_batch": per, "mean": m, "ci": [lo, hi],
                              "all_positive": all(x > 0 for x in per)}
    out["order_only_delta"] = d["shifted_delta"]
    out["null_over_othergroup"] = (out["cells"]["null"]["mean"] /
                                   out["cells"]["othergroup"]["mean"])
    if row is not None:
        r = row["deltas_all_rows"]
        lv = row["deltas_live_rows"]
        out["row_dilution"] = {"all_rows": r, "live_rows": lv,
                               "multipliers": row["measured_multipliers"],
                               "expected": row["expected_multiplier"],
                               "worst_dev": max(abs(x - row["expected_multiplier"])
                                                for x in row["measured_multipliers"])}
        n, k = row["seq"] - 1, row["topk"]
        inert = [((a * n) - (l * (n - k))) / k for a, l in zip(r, lv)]
        out["inert_row_delta_mean"] = sum(inert) / len(inert)
        out["inert_share_of_live"] = out["inert_row_delta_mean"] / (sum(lv) / len(lv))
    return out


def selftest():
    # a cell with a genuinely larger null must report it, and one where the two
    # markets are identical must report a ratio of exactly 1
    fake = {"model": "x", "method": "m", "seq": 4096, "topk": 2048, "chunk": 128,
            "n_batches": 3, "leaders": [0, 3], "n_followers": 4,
            "base_per_batch": [1.0, 1.0, 1.0], "shifted_delta": 0.0,
            "deltas": {"null": [0.07, 0.07, 0.07], "othergroup": [0.005, 0.005, 0.005]}}
    a = analyse(fake, None)
    assert abs(a["null_over_othergroup"] - 14.0) < 1e-9, a["null_over_othergroup"]
    assert a["cells"]["null"]["all_positive"]
    fake2 = dict(fake, deltas={"null": [0.01, 0.01, 0.01], "othergroup": [0.01, 0.01, 0.01]})
    assert abs(analyse(fake2, None)["null_over_othergroup"] - 1.0) < 1e-9
    # a dud cell where the null is zero must not be reported as a win
    fake3 = dict(fake, deltas={"null": [0.0, 0.0, 0.0], "othergroup": [0.005, 0.005, 0.005]})
    assert analyse(fake3, None)["null_over_othergroup"] == 0.0
    # the row-dilution identity must reproduce on a synthetic pair
    row = {"deltas_all_rows": [0.01], "deltas_live_rows": [0.01 * 4095 / 2047],
           "measured_multipliers": [4095 / 2047], "expected_multiplier": 4095 / 2047,
           "seq": 4096, "topk": 2048}
    b = analyse(fake, row)
    assert abs(b["row_dilution"]["worst_dev"]) < 1e-12
    assert abs(b["inert_row_delta_mean"]) < 1e-15
    print("SELFTEST OK: ratio reproduces, equal cells give 1.0, a zero null is not a win, "
          "and the row identity is exact on a synthetic pair")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); raise SystemExit(0)
    d = load("ic_null.json")
    row = load("ic_row_dilution.json")
    out = analyse(d, row)
    print("%-12s %-8s %-12s %-22s %-6s" % ("reading", "mean", "ci_lo", "ci_hi", "pos"))
    for name, c in out["cells"].items():
        print("%-12s %+0.6f %+0.6f %+0.6f   %s" % (name, c["mean"], c["ci"][0], c["ci"][1],
              c["all_positive"]))
    print()
    print("null / other-group  = %.2f x" % out["null_over_othergroup"])
    print("order-only (roll the index list) = %+0.3e  (must be 0: scatter_ is order-invariant)"
          % out["order_only_delta"])
    print("row dilution: all_rows %s -> live_rows %s" %
          (["%+.6f" % x for x in out["row_dilution"]["all_rows"]],
           ["%+.6f" % x for x in out["row_dilution"]["live_rows"]]))
    print("  multipliers %s vs expected %.4f (worst dev %.4f)" %
          (["%.4f" % x for x in out["row_dilution"]["multipliers"]],
           out["row_dilution"]["expected"], out["row_dilution"]["worst_dev"]))
    print("  inert rows contribute %+.6f = %.2f%% of the live-row delta"
          % (out["inert_row_delta_mean"], 100 * out["inert_share_of_live"]))
    json.dump(out, open(os.path.join(DATA, "ic_analysis.json"), "w"), indent=1)
    print("WROTE data/ic_analysis.json")
