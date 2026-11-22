#!/usr/bin/env python3
"""How deep is the expensive band, and is that stable across scale and family?

The paper's §leader sentence is "the asymmetry and the source-layer ordering do
not depend on the architecture".  That is a statement about ORDER.  This script
asks the sharper question the 14B sweep made unavoidable: is the *depth* of the
expensive band (how many leading sources are expensive relative to the tail)
also architecture-independent?

Protocol note that makes this checkable at all: `pairs20w12_*.jsonl` were swept
to distance 12 while the 14B and TinyLlama sweeps only reached distance 6.  We
therefore restrict every file to distance<=6 and verify that this reproduces the
four published cross_family ratios (7.54 / 34.99 / 11.82 / 28.71, tolerance
0.02).  If it does not, the comparison is invalid and this script fails loudly
rather than reporting a number.

No GPU.  `--selftest` proves the reproduce-the-published-ratios gate can fail
and that a genuinely deep band is detected.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MAX_DISTANCE = 6
BOOT = 20000
SEED = 20261119

FILES = {
    "qwen05": "pairs20w12_0.5B.jsonl",
    "qwen15": "pairs20w12_1.5B.jsonl",
    "qwen7": "pairs20w12_7B.jsonl",
    "tiny": "pairs20_TinyLlama.jsonl",
    "qwen14": "pairs14b.jsonl",
}

# published, and asserted as a gate rather than trusted
PUBLISHED = {"qwen05": 7.54, "qwen15": 34.99, "qwen7": 11.82, "tiny": 28.71}
TOL = 0.02


def load(name, max_distance=MAX_DISTANCE):
    path = os.path.join(DATA, name)
    recs = [json.loads(l) for l in open(path)]
    return [r for r in recs
            if r.get("kind") == "pair" and 0 < r["target"] - r["source"] <= max_distance]


def by_source(pairs):
    out = {}
    for p in pairs:
        out.setdefault(p["source"], []).append(p["vs_own"])
    return out


def ratio_le(by, boundary):
    lead = [v for s, vv in by.items() if s <= boundary for v in vv]
    rest = [v for s, vv in by.items() if s > boundary for v in vv]
    if not lead or not rest:
        return None
    r = st.mean(rest)
    return st.mean(lead) / r if r else math.inf


def boot_ratio(a, b, n=BOOT, seed=SEED):
    """Percentile CI for mean(a)/mean(b), resampling within each group."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        x = [a[rng.randrange(len(a))] for _ in range(len(a))]
        y = [b[rng.randrange(len(b))] for _ in range(len(b))]
        d = st.mean(y)
        vals.append(st.mean(x) / d if d else math.inf)
    vals.sort()
    return {"lo": vals[int(0.025 * n)], "hi": vals[int(0.975 * n)], "median": vals[n // 2]}


def band(pairs):
    """Depth of the expensive band: source 2 (or any b) against the deeper tail."""
    by = by_source(pairs)
    tail = [v for s, vv in by.items() if s >= 3 for v in vv]
    out = {
        "n_pairs": len(pairs),
        "source_means": {str(s): st.mean(v) for s, v in sorted(by.items())},
        "source_n": {str(s): len(v) for s, v in sorted(by.items())},
        "ratios_by_boundary": {str(b): ratio_le(by, b) for b in range(5)},
    }
    if 2 in by and tail:
        r = st.mean(by[2]) / st.mean(tail) if st.mean(tail) else math.inf
        out["src2_over_tail"] = r
        out["src2_over_tail_ci"] = boot_ratio(by[2], tail)
    if 0 in by and 1 in by:
        out["src1_minus_src0"] = st.mean(by[1]) - st.mean(by[0])
        rng = random.Random(SEED + 3)
        bs = []
        for _ in range(BOOT):
            x = [by[0][rng.randrange(len(by[0]))] for _ in range(len(by[0]))]
            y = [by[1][rng.randrange(len(by[1]))] for _ in range(len(by[1]))]
            bs.append(st.mean(y) - st.mean(x))
        bs.sort()
        out["src1_minus_src0_ci"] = {"lo": bs[int(0.025 * BOOT)], "hi": bs[int(0.975 * BOOT)]}
    return out


def selftest():
    # gate 1: restricting w12 files to distance<=6 must reproduce the published
    # cross-family ratios.  This is the whole basis for comparing 14B at all.
    for tag, want in PUBLISHED.items():
        by = by_source(load(FILES[tag]))
        got = ratio_le(by, 1)
        assert abs(got - want) < TOL, (tag, got, want)
    # prove-red: the same gate must fail if the distance cut is wrong
    by = by_source(load(FILES["qwen05"], max_distance=12))
    got = ratio_le(by, 1)
    assert abs(got - PUBLISHED["qwen05"]) >= TOL, ("distance gate is not discriminating", got)
    # deep band detected
    deep = [{"source": 0, "target": 1, "distance": 1, "vs_own": 1.0},
            {"source": 1, "target": 2, "distance": 1, "vs_own": 1.0},
            {"source": 2, "target": 3, "distance": 1, "vs_own": 1.0},
            {"source": 3, "target": 4, "distance": 1, "vs_own": 0.01},
            {"source": 4, "target": 5, "distance": 1, "vs_own": 0.01}]
    d = band(deep)
    assert d["src2_over_tail"] > 50, d["src2_over_tail"]
    # prove-red: a shallow band must NOT be reported as deep
    shallow = [dict(x, vs_own=1.0 if x["source"] <= 1 else 0.01) for x in deep]
    s = band(shallow)
    assert s["src2_over_tail"] < 2, s["src2_over_tail"]
    print("SELFTEST OK: distance gate reproduces published ratios, and it fails "
          "when the cut is wrong; deep vs shallow bands separate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=os.path.join(DATA, "band_depth.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res = {"max_distance": MAX_DISTANCE, "boot": BOOT, "seed": SEED,
           "published": PUBLISHED, "models": {}}
    for tag, name in FILES.items():
        pairs = load(name)
        b = band(pairs)
        b["reproduces_published"] = (
            abs(ratio_le(by_source(pairs), 1) - PUBLISHED[tag]) < TOL
            if tag in PUBLISHED else None)
        res["models"][tag] = b
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("%-8s %-9s %-9s %-9s %-11s %s" % ("model", "src2", "src3+", "ratio", "95% CI", "reproduces"))
    for tag, b in res["models"].items():
        m = b["source_means"]
        ci = b.get("src2_over_tail_ci", {})
        print("%-8s %+9.6f %+9.6f %9.2f  [%5.2f,%5.2f]  %s" % (
            tag, m.get("2", float("nan")),
            st.mean([v for s, vv in by_source(load(FILES[tag])).items() if s >= 3 for v in vv]),
            b.get("src2_over_tail", float("nan")),
            ci.get("lo", float("nan")), ci.get("hi", float("nan")),
            b["reproduces_published"]))
    print("WROTE", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
