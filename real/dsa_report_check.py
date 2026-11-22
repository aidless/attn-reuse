
#!/usr/bin/env python3
"""GPU-free gate for the trained-sparse real-model result (report 23).

Two jobs:
  1. Reproject the five published dense ratios under the identical distance<=6
     cut, so the trained-sparse number is compared against a re-derived baseline
     rather than a quoted one.  A prove-red cuts at distance<=12 instead and
     must fail to reproduce, proving the gate can go red.
  2. Recompute the trained-sparse verdict (ratio, permutation, difference CI,
     mean-delta CI) from the raw 141 pairs.

No GPU, no model weights: reads attn-reuse/real/data/*.json only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED = 20261119


def load_pairs(path, maxd=6):
    """Read either a single JSON artifact or a JSONL pair sweep.

    The dense sweeps are JSONL (one record per line: a `base` row plus `pair`
    rows); the trained-sparse sweep is a single JSON object with a `pairs` list.
    Handling only one shape silently breaks the reprojection gate.
    """
    text = open(path).read()
    try:
        d = json.loads(text)
        pretty = True
    except json.JSONDecodeError:
        pretty = False
    if pretty:
        if "pairs" not in d:
            raise SystemExit("%s is JSON but has no 'pairs' key" % path)
        pr = [p for p in d["pairs"]
              if 0 < p["target"] - p["source"] <= maxd]
        return d, pr
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    if not rows or rows[0].get("kind") not in ("base", "pair"):
        raise SystemExit("%s is neither a pairs artifact nor a sweep" % path)
    pairs = [r for r in rows if r.get("kind") == "pair"
             and 0 < r["target"] - r["source"] <= maxd]
    return {"records": rows}, pairs


def ratio(a, b):
    mb = st.mean(b)
    return st.mean(a) / mb if mb else float("inf")


def perm_two_sided(lead, rest, n=10000, seed=SEED):
    """Two-sided permutation on the ratio: could a relabelling be this far from 1?"""
    vals, nl = lead + rest, len(lead)
    rng = random.Random(seed)
    obs = ratio(lead, rest)
    dev = abs(obs - 1.0)
    cnt = 0
    for _ in range(n):
        p = rng.sample(vals, len(vals))
        dd = st.mean(p[nl:])
        r = st.mean(p[:nl]) / dd if dd else 0.0
        if abs(r - 1.0) >= dev:
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


def boot_diff(lead, rest, n=10000, seed=SEED + 1):
    """Percentile bootstrap of the *difference*; a ratio CI is unreadable when
    the denominator can approach zero."""
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        l = [lead[rng.randrange(len(lead))] for _ in range(len(lead))]
        r = [rest[rng.randrange(len(rest))] for _ in range(len(rest))]
        ds.append(st.mean(l) - st.mean(r))
    ds.sort()
    return ds[int(0.025 * n)], ds[int(0.975 * n)], ds[n // 2], sum(1 for x in ds if x > 0) / n


def reproject(path, expect, n_lead, n_rest, tol=0.02, maxd=6):
    _, pr = load_pairs(path, maxd)
    lead = [p["vs_own"] for p in pr if p["source"] <= 1]
    rest = [p["vs_own"] for p in pr if p["source"] >= 2]
    got = ratio(lead, rest)
    return {"path": path, "ratio": got, "n_lead": len(lead),
            "n_rest": len(rest), "ok": abs(got - expect) < tol
            and len(lead) == n_lead and len(rest) == n_rest}


DENSE = [(os.path.join(DATA, "pairs20w12_0.5B.jsonl"), 7.54, 12, 111),
         (os.path.join(DATA, "pairs20w12_1.5B.jsonl"), 34.99, 12, 135),
         (os.path.join(DATA, "pairs20w12_7B.jsonl"), 11.82, 12, 135),
         (os.path.join(DATA, "pairs20_TinyLlama.jsonl"), 28.71, 12, 99),
         (os.path.join(DATA, "pairs14b.jsonl"), 12.67, 12, 255)]


def selftest():
    # 1. the five published ratios must reproduce under the same cut
    for path, want, nl, nr in DENSE:
        r = reproject(path, want, nl, nr)
        assert r["ok"], ("reprojection failed", r, want, nl, nr)
    # 2. prove-red: cut at distance<=12 instead and 0.5B must NOT reproduce
    _, pr = load_pairs(DENSE[0][0], maxd=12)
    lead = [p["vs_own"] for p in pr if p["source"] <= 1]
    rest = [p["vs_own"] for p in pr if p["source"] >= 2]
    got12 = ratio(lead, rest)
    assert abs(got12 - 7.54) >= 0.02, ("cut-12 should not match the published ratio", got12)
    # 3. trained-sparse must be indistinguishable from 1 on both scales
    _, sp = load_pairs(os.path.join(DATA, "dsa_multi.json"))
    lead = [p["vs_own"] for p in sp if p["source"] <= 1]
    rest = [p["vs_own"] for p in sp if p["source"] >= 2]
    obs, p = perm_two_sided(lead, rest, n=2000)
    assert p > 0.05, ("trained-sparse should not reject", obs, p)
    lo, hi, _, _ = boot_diff(lead, rest, n=2000)
    assert lo < 0 < hi, ("sparse difference CI must straddle zero", lo, hi)
    # 4. prove-red: a synthetic strong lead effect must be flagged
    obs2, p2 = perm_two_sided([1.0] * 12, [0.01] * 20, n=2000)
    assert obs2 > 50 and p2 <= 0.001, (obs2, p2)
    print("SELFTEST OK: 5 dense reprojections reproduce, cut-12 prove-red fails, "
          "trained-sparse non-extreme, synthetic lead flagged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=os.path.join(DATA, "dsa_report.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    _, pr = load_pairs(os.path.join(DATA, "dsa_multi.json"))
    lead = [p["vs_own"] for p in pr if p["source"] <= 1]
    rest = [p["vs_own"] for p in pr if p["source"] >= 2]
    obs, p_two = perm_two_sided(lead, rest)
    lo, hi, med, share = boot_diff(lead, rest)
    sens = {}
    for b in (0, 1, 2, 3):
        l2 = [p["vs_own"] for p in pr if p["source"] <= b]
        r2 = [p["vs_own"] for p in pr if p["source"] > b]
        sens[str(b)] = {"ratio": ratio(l2, r2), "n_lead": len(l2)}
    dmean = [p["vs_own"] for p in pr]
    rng = random.Random(SEED + 21)
    bs = []
    for _ in range(10000):
        s = [dmean[rng.randrange(len(dmean))] for _ in range(len(dmean))]
        bs.append(st.mean(s))
    bs.sort()
    out = {"n_pairs": len(pr), "n_lead": len(lead), "n_rest": len(rest),
           "lead_mean": st.mean(lead), "rest_mean": st.mean(rest), "ratio": obs,
           "permutation_two_sided": {"n": 10000, "seed": SEED, "p": p_two},
           "diff": {"value": st.mean(lead) - st.mean(rest),
                    "ci_lo": lo, "ci_hi": hi, "median": med, "share_positive": share},
           "boundary_sensitivity": sens,
           "mean_delta": {"value": st.mean(dmean),
                          "ci_lo": bs[int(0.025 * len(bs))],
                          "ci_hi": bs[int(0.975 * len(bs))],
                          "n_positive": sum(1 for x in dmean if x > 0),
                          "n_negative": sum(1 for x in dmean if x < 0)},
           "dense_reprojection": [reproject(*d) for d in DENSE],
           "inputs": ["real/data/dsa_multi.json", "real/data/dsa_overlap.json"]}
    json.dump(out, open(a.out, "w"), indent=1)
    print("ratio %.3f  perm p=%.4f  diff %+.2e CI [%+.2e, %+.2e]"
          % (obs, p_two, out["diff"]["value"], lo, hi))
    print("mean delta %+.3e CI [%+.3e, %+.3e]  positive %d/%d"
          % (out["mean_delta"]["value"], bs[int(0.025 * len(bs))],
             bs[int(0.975 * len(bs))], out["mean_delta"]["n_positive"], len(dmean)))
    print("dense reprojection all ok:", all(r["ok"] for r in out["dense_reprojection"]))
    print("WROTE", a.out)


if __name__ == "__main__":
    raise SystemExit(main())
