"""How much of the headline nats-per-token is dilution?

At sequence length S and top-k k, a query at position q can see q+1 keys. While
q+1 <= k the top-k set IS every visible key, so a borrowing layer's set and its
own set coincide, and the row contributes exactly zero to reuse - own. Those
rows still sit in the denominator.

So if the inert rows really contribute zero, then

    delta(active rows only) = delta(all rows) * (S-1) / (S-1-k)

exactly, with no free parameter. This script checks that identity against the
measured deltas rather than assuming it: agreement in every cell is evidence the
inert rows are inert; disagreement would mean something else is diluting.
"""
import json, math, statistics as st, sys

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def predicted_multiplier(seq, k):
    rows = seq - 1                      # loss is defined on targets 1..seq-1
    active = rows - min(k, rows)
    return rows / active if active > 0 else float("inf")


def t_ci(vals):
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    m, sd = st.mean(vals), st.stdev(vals)
    return (m - T975.get(n - 1, 1.96) * sd / math.sqrt(n),
            m + T975.get(n - 1, 1.96) * sd / math.sqrt(n))


def analyse(d, tol=5e-4):
    seq = d["seq"]
    out = {"model": d["model"], "seq": seq, "n_batches": d["n_batches"],
           "corpus": d.get("corpus"), "dtype": d.get("dtype"),
           "scoring_rows": seq - 1, "cells": {}, "identity_holds_everywhere": True}
    worst = 0.0
    for k in d["k"]:
        pred = predicted_multiplier(seq, k)
        for g in d["groups"]:
            key = "k%d_g%d" % (k, g)
            allrows = d["deltas_all_rows"][key]
            active = d["deltas_active_rows"][key]
            obs = (active / allrows) if allrows else float("nan")
            dev = abs(obs - pred)
            worst = max(worst, dev)
            if dev > tol * max(1.0, pred):
                out["identity_holds_everywhere"] = False
            out["cells"][key] = {
                "all_rows": allrows, "active_rows": active,
                "observed_multiplier": obs, "predicted_multiplier": pred,
                "abs_deviation": dev,
                "inert_row_share": min(k, seq - 1) / (seq - 1),
                "active_row_share": (seq - 1 - min(k, seq - 1)) / (seq - 1),
            }
    out["worst_abs_deviation"] = worst
    return out


def selftest():
    # (a) the derivation itself: 512/256 has exactly half the scoring rows inert
    assert abs(predicted_multiplier(512, 256) - 511 / 255) < 1e-12
    assert predicted_multiplier(512, 256) > 2.0
    # (b) k >= seq is degenerate: no row can discriminate, the multiplier blows up
    assert predicted_multiplier(64, 64) == float("inf")
    # (c) prove-red: a fixture where the active-row delta is NOT the predicted
    #     multiple must be flagged. If this never fired the identity would be
    #     decoration rather than a check.
    fake = {"model": "synthetic", "seq": 8, "n_batches": 2, "dtype": "float16",
            "corpus": "x", "k": [2], "groups": [2],
            "deltas_all_rows": {"k2_g2": 1.0},
            "deltas_active_rows": {"k2_g2": 7.0 * 6 / 4}}   # wrong on purpose
    bad = analyse(fake)
    assert bad["identity_holds_everywhere"] is False, "the gate did not go red on a violated identity"
    # (d) and a fixture that DOES obey it must pass, so the gate is not just
    #     always-red
    good = dict(fake)
    good["deltas_active_rows"] = {"k2_g2": 1.0 * predicted_multiplier(8, 2)}
    ok = analyse(good)
    assert ok["identity_holds_everywhere"] is True
    print("SELFTEST OK: multiplier derivation exact; degenerate k==seq handled; "
          "violated identity flagged; obeyed identity accepted")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); raise SystemExit(0)
    src = sys.argv[1] if len(sys.argv) > 1 else "attn-reuse/real/data/position_matched_05_20b.json"
    dest = sys.argv[2] if len(sys.argv) > 2 else "attn-reuse/real/data/dilution_analysis.json"
    res = analyse(json.load(open(src)))
    json.dump(res, open(dest, "w"), indent=1)
    print("%-4s %-4s %11s %11s %10s %10s" % ("k", "g", "all_rows", "active", "predicted", "observed"))
    for key, c in sorted(res["cells"].items(), key=lambda kv: (int(kv[0].split("_")[0][1:]), int(kv[0].split("_g")[1]))):
        print("%-4s %-4s %+11.5f %+11.5f %10.4f %10.4f"
              % (c and key.split("_")[0][1:], key.split("_g")[1], c["all_rows"], c["active_rows"],
                 c["predicted_multiplier"], c["observed_multiplier"]))
    print("identity holds in every cell:", res["identity_holds_everywhere"],
          "| worst deviation %.2e" % res["worst_abs_deviation"])
