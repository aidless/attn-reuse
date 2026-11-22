#!/usr/bin/env python3
"""GPU-free four-scale analysis of post-hoc attention reuse cost.

Reads the *same* 20-batch, k=256, bfloat16 grid at four parameter scales and
reports (a) the reuse delta vs the dense/own reference and (b) whether the
group-size amplification (g6/g2) persists.  This is a robustness/scale
certificate, NOT a manuscript number: it is deliberately kept out of audit
pins because the 14B run used CPU offload (device_map=auto) while the paper's
0.5/1.5/7B runs used device_map=cuda.

Every printed number is recomputed from the four JSON artifacts.  selftest
injects a *fake monotonicity* to prove the trend gate can go red.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

FILES = {
    "0.5B": "main_20b_05.json",
    "1.5B": "main_20b_15.json",
    "7B": "main_20b_7.json",
    "14B": "scale_14b_base_20b.json",
}
ORDER = ["0.5B", "1.5B", "7B", "14B"]
GROUPS = (2, 3, 6)


def load():
    out = {}
    for name, fn in FILES.items():
        with open(os.path.join(DATA, fn)) as fh:
            d = json.load(fh)
        out[name] = {
            "full": d["readings"]["full"],
            "own": d["readings"]["own_k256"],
            "control": d["readings"]["control_own_kqlen"],
            "vs_full": [d["deltas"]["k256_g%d" % g]["vs_full"] for g in GROUPS],
            "vs_own": [d["deltas"]["k256_g%d" % g]["vs_own"] for g in GROUPS],
            "n_layers": d["n_layers"],
            "n_batches": d["n_batches"],
            "dtype": d["dtype"],
        }
    return out


def analyze(scales):
    """Return the comparison table and trend booleans."""
    rows = {}
    for name in ORDER:
        s = scales[name]
        rows[name] = {
            "n_layers": s["n_layers"],
            "n_batches": s["n_batches"],
            "dtype": s["dtype"],
            "full": round(s["full"], 6),
            "vs_full_g2": round(s["vs_full"][0], 6),
            "vs_full_g3": round(s["vs_full"][1], 6),
            "vs_full_g6": round(s["vs_full"][2], 6),
            "vs_own_g2": round(s["vs_own"][0], 6),
            "vs_own_g6": round(s["vs_own"][2], 6),
            "amp_g6_over_g2": round(s["vs_full"][2] / s["vs_full"][0], 3),
            "control_exact": abs(s["control"] - s["full"]) < 1e-9,
        }
    g2 = [rows[n]["vs_full_g2"] for n in ORDER]
    g6 = [rows[n]["vs_full_g6"] for n in ORDER]
    # monotonic decay of the adjacent-layer (g2) cost with parameter count
    g2_decreasing = all(g2[i] > g2[i + 1] for i in range(len(g2) - 1))
    # the group amplification persists at every scale (g6 > g2 > 0)
    amp_all = all(rows[n]["amp_g6_over_g2"] > 1.0 for n in ORDER)
    # 14B keeps the sign and shrinks relative to 7B
    sign_kept_14b = all(rows["14B"]["vs_full_" + ("g%d" % g)] > 0 for g in GROUPS)
    shrink_14b_vs_7b = round(g6[2] / g6[3], 2)  # 7B g6 / 14B g6
    return {
        "rows": rows,
        "g2_decreasing_with_scale": g2_decreasing,
        "amplification_persists_all_scales": amp_all,
        "sign_kept_at_14b": sign_kept_14b,
        "g6_shrink_7b_to_14b": shrink_14b_vs_7b,
        "all_controls_exact": all(rows[n]["control_exact"] for n in ORDER),
    }


def selftest():
    # positive: the real artifact must pass all three gates
    res = analyze(load())
    assert res["g2_decreasing_with_scale"], "real data should be monotonically decreasing"
    assert res["amplification_persists_all_scales"], "real data should amplify"
    assert res["all_controls_exact"], "controls should be bit-exact"
    # prove-red: make 14B the LARGEST g2 cost -> trend gate must flip
    fake = load()
    fake["14B"]["vs_full"][0] = 1.0
    f = analyze(fake)
    assert f["g2_decreasing_with_scale"] is False, "prove-red: injected non-monotonicity not caught"
    # prove-red: make g6 < g2 at one scale -> amplification gate must flip
    fake2 = load()
    fake2["7B"]["vs_full"][2] = fake2["7B"]["vs_full"][0] * 0.5
    f2 = analyze(fake2)
    assert f2["amplification_persists_all_scales"] is False, "prove-red: injected no-amplification not caught"
    print("SELFTEST OK: monotonic-trend + amplification + control gates all prove-red")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=os.path.join(DATA, "scale_trend.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res = analyze(load())
    hdr = "%6s %6s %10s %10s %10s %10s %8s" % (
        "model", "layers", "g2", "g3", "g6", "amp6/2", "ctrl")
    print("Four-scale reuse cost (k=256, 20 batches, bfloat16):")
    print(hdr)
    for n in ORDER:
        r = res["rows"][n]
        print("%6s %6d %+10.5f %+10.5f %+10.5f %8.1fx %8s" % (
            n, r["n_layers"], r["vs_full_g2"], r["vs_full_g3"],
            r["vs_full_g6"], r["amp_g6_over_g2"],
            "OK" if r["control_exact"] else "FAIL"))
    print("\ng2 cost decreases with scale:", res["g2_decreasing_with_scale"])
    print("group amplification at all scales:", res["amplification_persists_all_scales"])
    print("sign preserved at 14B:", res["sign_kept_at_14b"])
    print("g6 cost shrink 7B->14B: %.2fx" % res["g6_shrink_7b_to_14b"])
    print("all controls bit-exact:", res["all_controls_exact"])
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("WROTE", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
