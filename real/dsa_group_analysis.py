"""Group-wise sharing cost: trained-sparse vs dense, with honest small-n intervals."""
import json, math, statistics as st, sys

DENSE = {"2": 0.0242, "3": 0.0114, "4": 0.0560, "6": 0.0736}   # tab:grid, 0.5B, k=256/512
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

def t_ci(vals):
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    m = st.mean(vals); sd = st.stdev(vals); sem = sd / math.sqrt(n)
    h = T975.get(n - 1, 1.96) * sem
    return (m - h, m + h)

def analyse(path):
    d = json.load(open(path))
    out = {"model": d["model"], "seq": d["seq"], "n_batches": d["n_batches"],
           "keep_fraction": d["keep_fraction"], "repeat_range": d["same_batch_repeat_range"],
           "base_nll": d["base_nll"], "groups": {}}
    for g, v in d["groups"].items():
        deltas = v["deltas_per_batch"]
        lo, hi = t_ci(deltas)
        ref = DENSE.get(g)
        ratio = (st.mean(deltas) / ref) if ref else float("nan")
        out["groups"][g] = {
            "mean_delta": st.mean(deltas), "sd": st.stdev(deltas),
            "ci_lo": lo, "ci_hi": hi, "n": len(deltas),
            "deltas_per_batch": deltas,
            "dense_same_keep": ref, "ratio_vs_dense": ratio,
            "straddles_zero": lo < 0 < hi,
        }
    return out

def selftest():
    # a known set must reproduce its own t interval
    v = [1.0, 2.0, 3.0]
    lo, hi = t_ci(v)
    assert abs(st.mean(v) - 2.0) < 1e-12
    assert lo < 2.0 < hi and abs((hi - lo) - 2 * 4.303 * 1.0 / math.sqrt(3)) < 1e-9
    # prove-red: a set that straddles zero must be flagged
    assert t_ci([-1.0, 0.5, 0.2])[0] < 0 < t_ci([-1.0, 0.5, 0.2])[1]
    # prove-red: an all-positive set with no spread must NOT straddle
    assert not (t_ci([1.0, 1.0, 1.0])[0] < 0)
    print("SELFTEST OK: t intervals reproduce, straddle detection both directions")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); raise SystemExit(0)
    src = sys.argv[1] if len(sys.argv) > 1 else "attn-reuse/real/data/dsa_group.json"
    res = analyse(src)
    json.dump(res, open("attn-reuse/real/data/dsa_group_analysis.json", "w"), indent=1)
    print("%-5s %10s %10s %22s %10s %8s" % ("g", "mean", "sd", "95% CI (t)", "dense", "ratio"))
    for g, r in sorted(res["groups"].items()):
        print("%-5s %+10.6f %10.6f  [%+.6f, %+.6f] %10.4f  %7.1f%%  %s"
              % (g, r["mean_delta"], r["sd"], r["ci_lo"], r["ci_hi"],
                 r["dense_same_keep"], 100 * r["ratio_vs_dense"],
                 "<- CI straddles 0" if r["straddles_zero"] else "<- CI excludes 0"))
