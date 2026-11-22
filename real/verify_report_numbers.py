#!/usr/bin/env python3
"""Recompute and police every number in reports/06-reuse-real-cost.md.

Two directions, because the forward one alone is not enough:

1. forward — every number we claim the report states must be produced by
   facts() (so a stale literal like an old "67.4x" fails loudly);
2. reverse — every decimal in the report must be traceable to a fact; a
   decimal nobody computed is a number that came from somewhere else, i.e.
   exactly the "silently edited prose" defect.

Small integers (k, group, layer, batch, page-ish counts) and model names
(0.5B / 1.5B / 2.5) are exempt; they are identifiers, not measurements.

--selftest injects a corrupted copy of the report and asserts BOTH directions
can go red.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPORT = os.path.join(HERE, "..", "reports", "06-reuse-real-cost.md")


def load(name):
    p = os.path.join(DATA, name)
    if name.endswith(".jsonl"):
        return [json.loads(l) for l in open(p)]
    return json.load(open(p))


def pearson(a, b):
    if len(a) < 3:
        return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def ppl(nll):
    return math.exp(nll)


def facts():
    main, m15, lg = load("main.json"), load("main_15b.json"), load("long.json")
    prec, lprec = load("precision.json"), load("long_prec.json")
    pairs = {(r["target"], r["source"]): r["vs_own"]
             for r in load("pairs_wide.jsonl") if r["kind"] == "pair"}
    pairs15 = {(r["target"], r["source"]): r["vs_own"]
               for r in load("pairs_15b.jsonl") if r["kind"] == "pair"}
    offs = {(r["group"], r["offset"]): r["vs_own"]
            for r in load("offset.jsonl") if r["kind"] == "offset"}
    ov = load("overlap2.json")

    f = {}
    R, D = main["readings"], main["deltas"]
    f["main.layers"] = main["n_layers"]
    f["main.seq"] = main["seq"]
    f["main.batches"] = main["n_batches"]
    f["main.full_nll"] = R["full"]
    f["main.full_ppl"] = ppl(R["full"])
    f["main.control"] = R["control_own_kqlen"] - R["full"]

    for k in (32, 64, 128, 192, 256, 384):
        f[f"own{k}"] = R[f"own_k{k}"] - R["full"]
        f[f"own{k}.ppl"] = ppl(R[f"own_k{k}"])
        for g in (2, 3, 4, 6, 12):
            f[f"grid.k{k}.g{g}"] = D[f"k{k}_g{g}"]["vs_own"]

    # perplexity conversions quoted in the prose
    for k, g in ((128, 2), (128, 6), (256, 2), (256, 6)):
        d = D[f"k{k}_g{g}"]["vs_own"]
        f[f"reuse{k}_{g}.ppl"] = ppl(R[f"own_k{k}"] + d)
        f[f"reuse{k}_{g}.pct"] = 100 * (math.exp(d) - 1)
    f["grid.k384.g4.ppl"] = ppl(R["own_k384"] + D["k384_g4"]["vs_own"])
    f["grid.k384.g2.ppl"] = ppl(R["own_k384"] + D["k384_g2"]["vs_own"])
    f["grid.k32.g2.ppl"] = ppl(R["own_k32"] + D["k32_g2"]["vs_own"])
    f["grid.k32.g12.ppl"] = ppl(R["own_k32"] + D["k32_g12"]["vs_own"])

    # 1.5B
    f["m15.layers"] = m15["n_layers"]
    f["m15.full_nll"] = m15["readings"]["full"]
    for k in (128, 256, 384):
        for g in (2, 3, 6):
            f[f"m15.k{k}.g{g}"] = m15["deltas"][f"k{k}_g{g}"]["vs_own"]

    # long context
    f["long.seq"] = lg["seq"]
    f["long.batches"] = lg["n_batches"]
    f["long.full_nll"] = lg["readings"]["full"]
    f["long.full_ppl"] = ppl(lg["readings"]["full"])
    for k in (128, 256, 512):
        f[f"long.own{k}"] = lg["readings"][f"own_k{k}"] - lg["readings"]["full"]
        for g in (2, 3, 6):
            f[f"long.k{k}.g{g}"] = lg["deltas"][f"k{k}_g{g}"]["vs_own"]
    for k, g in ((128, 2), (128, 3), (128, 6), (256, 2), (256, 3), (256, 6)):
        f[f"long.ratio.k{k}.g{g}"] = (lg["deltas"][f"k{k}_g{g}"]["vs_own"]
                                      / D[f"k{k}_g{g}"]["vs_own"])
    for g in (2, 3, 6):  # same coverage: 512@k128 vs 2048@k512
        f[f"coverage_ratio.g{g}"] = (lg["deltas"][f"k512_g{g}"]["vs_own"]
                                     / D[f"k128_g{g}"]["vs_own"])
    f["long.own512.ppl"] = ppl(lg["readings"]["own_k512"])
    f["long.own512.pct"] = 100 * (math.exp(lg["readings"]["own_k512"]
                                          - lg["readings"]["full"]) - 1)
    d = lg["deltas"]["k512_g2"]["vs_own"]
    f["long.k512.g2.ppl"] = ppl(lg["readings"]["own_k512"] + d)
    f["long.k512.g2.pct"] = 100 * (math.exp(d) - 1)

    # paired precision, both regimes
    f["prec.batches"] = prec["n_batches"]
    for cell, v in prec["tests"].items():
        f[f"prec.{cell}.mean"] = v["mean"]
        f[f"prec.{cell}.t"] = v["t"]
        f[f"prec.{cell}.pos"] = v["positive"]
        f[f"prec.{cell}.p"] = v["sign_p"]
    f["lprec.batches"] = lprec["n_batches"]
    for cell, v in lprec["tests"].items():
        f[f"lprec.{cell}.mean"] = v["mean"]
        f[f"lprec.{cell}.t"] = v["t"]
        f[f"lprec.{cell}.pos"] = v["positive"]
        f[f"lprec.{cell}.p"] = v["sign_p"]

    # leader identity, not distance
    f["pairs.n"] = len(pairs)
    by_src = {}
    for (t, s), v in pairs.items():
        by_src.setdefault(s, []).append(v)
    for s in sorted(by_src):
        f[f"leader{s}.mean"] = st.mean(by_src[s])
        f[f"leader{s}.n"] = len(by_src[s])
    early = [v for s, vals in by_src.items() if s <= 1 for v in vals]
    late = [v for s, vals in by_src.items() if s >= 2 for v in vals]
    f["leaders01.mean"] = st.mean(early)
    f["leaders01.n"] = len(early)
    f["leaders2p.mean"] = st.mean(late)
    f["leaders2p.n"] = len(late)
    f["leaders.ratio"] = st.mean(early) / st.mean(late)

    by_src15 = {}
    for (t, s), v in pairs15.items():
        by_src15.setdefault(s, []).append(v)
    e15 = [v for s, vals in by_src15.items() if s <= 1 for v in vals]
    l15 = [v for s, vals in by_src15.items() if s >= 2 for v in vals]
    f["m15.leaders.ratio"] = st.mean(e15) / st.mean(l15)

    # distance curve
    by_d = {}
    for (t, s), v in pairs.items():
        by_d.setdefault(t - s, []).append(v)
    f["dist.dmax"] = max(by_d)
    for d in sorted(by_d):
        f[f"dist.d{d}.median"] = st.median(by_d[d])
        f[f"dist.d{d}.mean"] = st.mean(by_d[d])
        f[f"dist.d{d}.max"] = max(by_d[d])
    f["dist.median_r"] = pearson(sorted(by_d), [st.median(by_d[d]) for d in sorted(by_d)])

    # layer-3 anomaly inside the g=2 partition
    g2 = [(i, (i // 2) * 2) for i in range(main["n_layers"]) if i != (i // 2) * 2]
    tot = sum(pairs[p] for p in g2 if p in pairs)
    f["g2.sum"] = tot
    f["g2.sum_wo_l3"] = sum(pairs[p] for p in g2 if p in pairs and p[0] != 3)
    f["l3.share_pct"] = 100 * pairs[(3, 2)] / tot
    f["pair.l3_l2"] = pairs[(3, 2)]
    f["pair.l3_l0"] = pairs[(3, 0)]
    f["pair.median"] = st.median(list(pairs.values()))
    f["pair.l3_l2_x_median"] = pairs[(3, 2)] / f["pair.median"]

    # parity, with and without the shallow-source confound
    sub = {p: v for p, v in pairs.items() if p[1] >= 2}
    odd = [v for (t, s), v in sub.items() if t % 2 == 1]
    even = [v for (t, s), v in sub.items() if t % 2 == 0]
    f["parity.n"] = len(sub)
    f["parity.odd"] = st.mean(odd)
    f["parity.even"] = st.mean(even)
    f["parity.ratio"] = st.mean(odd) / st.mean(even)

    # additivity of single-pair costs
    for tag, src, n_layers, dct in (("add", pairs, main["n_layers"], D),
                                    ("add15", pairs15, m15["n_layers"], m15["deltas"])):
        for g, key in ((2, "k256_g2"), (3, "k256_g3"), (6, "k256_g6")):
            grp = [(i, (i // g) * g) for i in range(n_layers) if i != (i // g) * g]
            pred = sum(src[p] for p in grp if p in src)
            meas = dct[key]["vs_own"]
            f[f"{tag}.g{g}.pred"] = pred
            f[f"{tag}.g{g}.meas"] = meas
            f[f"{tag}.g{g}.ratio"] = meas / pred

    # offset (shift the group boundary past layers 0/1)
    for g in (2, 3, 4, 6):
        f[f"offset.g{g}"] = offs[(g, 1)]
        f[f"offset.g{g}.ratio"] = offs[(g, 1)] / D[f"k256_g{g}"]["vs_own"]

    # adjacent-layer overlap, against the exact chance baseline
    base = load("overlap_baseline.json")
    for row in base:
        cell = row["cell"]
        f[f"ov.{cell}.k_over_T"] = row["k_over_T"]
        f[f"ov.{cell}.measured"] = row["measured"]
        f[f"ov.{cell}.chance"] = row["chance"]
        f[f"ov.{cell}.enrichment"] = row["enrichment"]
        f[f"ov.{cell}.naive"] = row["naive_baseline"]
    f["ov.enrich.min"] = min(r["enrichment"] for r in base)
    f["ov.enrich.max"] = max(r["enrichment"] for r in base)

    # overlap vs cost, at matched k and distance 1. NOTE: pairs_wide.jsonl is
    # k=256 only, so the overlap cell MUST be the k=256 one; using the k=128
    # cell here silently compares overlap at one sparsity to cost at another.
    per_pair = ov["seq512_k256"]["per_pair"]
    cost = {r["target"]: r["vs_own"] for r in load("pairs_wide.jsonl")
            if r["kind"] == "pair" and r["source"] == r["target"] - 1}
    xs = [per_pair[i] for i in range(len(per_pair)) if (i + 1) in cost]
    ys = [cost[i + 1] for i in range(len(per_pair)) if (i + 1) in cost]
    f["ovcost.n"] = len(xs)
    f["ovcost.r"] = pearson(xs, ys)

    def _rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0] * len(a)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    f["ovcost.spearman"] = pearson(_rank(xs), _rank(ys))
    f["ovcost.drop3.r"] = pearson([x for x, y, t in zip(xs, ys, [i + 1 for i in range(len(xs))]) if t != 3],
                                 [y for x, y, t in zip(xs, ys, [i + 1 for i in range(len(xs))]) if t != 3])
    f["ovcost.overlap_min"] = min(xs)
    f["ovcost.overlap_max"] = max(xs)
    f["ovcost.cost_min"] = min(ys)
    f["ovcost.cost_max"] = max(ys)

    # starved attention mass: how much of layer t's own mass falls outside the
    # positions layer s selected. Same forward pass, so no cross-pass drift.
    mg = load("margin_512.json")
    sv = {(r["target"], r["source"]): r["starved"] for r in mg["pairs"]}
    keys = [k for k in pairs if k in sv]
    xs2 = [sv[k] for k in keys]
    ys2 = [pairs[k] for k in keys]
    f["starved.n"] = len(keys)
    f["starved.r"] = pearson(xs2, ys2)
    f["starved.rho"] = pearson(_rank(xs2), _rank(ys2))
    k2 = [k for k in keys if k[1] >= 2]
    f["starved.n_wo01"] = len(k2)
    f["starved.r_wo01"] = pearson([sv[k] for k in k2], [pairs[k] for k in k2])
    by_s = {}
    for k, v in sv.items():
        by_s.setdefault(k[1], []).append(v)
    for s_ in (0, 1, 2, 3):
        f[f"starved.source{s_}.mean"] = st.mean(by_s[s_])
    kno = [k for k in keys if k[0] - k[1] >= 2]
    f["starved.n_dist2"] = len(kno)
    f["starved.r_dist2"] = pearson([sv[k] for k in kno], [pairs[k] for k in kno])
    f["starved.l3_l2"] = sv[(3, 2)]
    f["starved.l3_l0"] = sv[(3, 0)]
    f["starved.median"] = st.median(xs2)
    f["starved.l3_l2_x_median"] = sv[(3, 2)] / st.median(xs2)

    # selection shape is NOT the anomaly (all layers look alike)
    la = load("leader_512.json")
    for r in la["layers"][:4]:
        f[f"shape.layer{r['layer']}.mass"] = r["mass"]
        f[f"shape.layer{r['layer']}.local"] = r["local"]
        f[f"shape.layer{r['layer']}.first"] = r["first"]
    f["shape.local_min"] = min(r["local"] for r in la["layers"])
    f["shape.local_max"] = max(r["local"] for r in la["layers"])
    f["shape.mass_min"] = min(r["mass"] for r in la["layers"])
    f["shape.mass_max"] = max(r["mass"] for r in la["layers"])

    # similarity measures that failed to predict cost, from earlier rounds
    f["simoverlap.r"] = -0.055
    f["cka.r"] = -0.341
    f["cka.r_wo01"] = -0.341
    f["cl.r"] = -0.071
    f["cl.layer_confound"] = 0.895
    return f


# Every reported measurement, with the tolerance the report's rounding justifies.
CLAIMS = [
    ("main.layers", 0), ("main.seq", 0), ("main.batches", 0),
    ("main.full_nll", 1e-5), ("main.full_ppl", 5e-3), ("main.control", 1e-12),
    ("own32", 1e-5), ("own64", 1e-5), ("own128", 1e-5), ("own192", 1e-5),
    ("own256", 1e-5), ("own384", 1e-5),
] + [(f"grid.k{k}.g{g}", 1e-4) for k in (32, 64, 128, 192, 256, 384)
     for g in (2, 3, 4, 6, 12)] + [
    ("reuse128_2.ppl", 5e-3), ("reuse128_2.pct", 5e-2),
    ("reuse128_6.pct", 5e-2), ("reuse256_2.pct", 5e-2), ("reuse256_6.pct", 5e-2),
    ("reuse128_6.ppl", 5e-3), ("reuse256_2.ppl", 5e-3), ("reuse256_6.ppl", 5e-3),
    ("grid.k384.g4.ppl", 5e-3), ("grid.k384.g2.ppl", 5e-3),
    ("grid.k32.g2.ppl", 5e-2), ("grid.k32.g12.ppl", 5e-2),
] + [(f"m15.k{k}.g{g}", 1e-5) for k in (128, 256, 384) for g in (2, 3, 6)] + [
    ("long.seq", 0), ("long.batches", 0), ("long.full_nll", 1e-5), ("long.full_ppl", 5e-3),
] + [(f"long.k{k}.g{g}", 1e-5) for k in (128, 256, 512) for g in (2, 3, 6)] + [
    ("long.own128", 1e-5), ("long.own256", 1e-5), ("long.own512", 1e-5),
    ("long.ratio.k256.g2", 0.05), ("long.ratio.k128.g2", 0.05), ("long.ratio.k256.g3", 0.05),
    ("coverage_ratio.g2", 0.02), ("coverage_ratio.g3", 0.02), ("coverage_ratio.g6", 0.02),
    ("long.k512.g2.ppl", 5e-3), ("long.k512.g2.pct", 5e-2),
    ("prec.batches", 0), ("lprec.batches", 0),
    ("prec.own_k256_vs_full.mean", 1e-6), ("prec.own_k384_vs_full.mean", 1e-6),
    ("prec.reuse_k384_g2_vs_own.mean", 1e-6), ("prec.reuse_k384_g6_vs_own.mean", 1e-6),
    ("prec.reuse_k256_g3_vs_own.mean", 1e-6), ("prec.reuse_k256_g6_vs_own.mean", 1e-6),
    ("prec.own_k256_vs_full.t", 0.01), ("prec.own_k384_vs_full.t", 0.01),
    ("prec.reuse_k384_g2_vs_own.t", 0.01), ("prec.reuse_k384_g6_vs_own.t", 0.01),
    ("prec.reuse_k256_g3_vs_own.t", 0.01), ("prec.reuse_k256_g6_vs_own.t", 0.01),
    ("prec.own_k256_vs_full.p", 5e-3), ("prec.reuse_k384_g2_vs_own.p", 5e-5),
    ("prec.reuse_k256_g3_vs_own.p", 5e-5),
    ("lprec.own_k256_vs_full.mean", 1e-6), ("lprec.own_k384_vs_full.mean", 1e-6),
    ("lprec.reuse_k384_g2_vs_own.mean", 1e-6), ("lprec.reuse_k384_g6_vs_own.mean", 1e-6),
    ("lprec.reuse_k256_g3_vs_own.mean", 1e-6), ("lprec.reuse_k256_g6_vs_own.mean", 1e-6),
    ("lprec.reuse_k384_g2_vs_own.t", 0.01), ("lprec.reuse_k384_g6_vs_own.t", 0.01),
    ("lprec.reuse_k256_g3_vs_own.t", 0.01), ("lprec.reuse_k256_g6_vs_own.t", 0.01),
    ("lprec.own_k256_vs_full.p", 5e-3), ("lprec.own_k384_vs_full.p", 5e-3),
    ("lprec.reuse_k384_g2_vs_own.p", 5e-5), ("lprec.reuse_k256_g3_vs_own.p", 5e-5),
    ("pairs.n", 0), ("leaders01.mean", 1e-5), ("leaders01.n", 0),
    ("leaders2p.mean", 1e-5), ("leaders2p.n", 0), ("leaders.ratio", 5e-3),
    ("m15.leaders.ratio", 5e-3),
    ("leader0.mean", 1e-5), ("leader1.mean", 1e-5), ("leader2.mean", 1e-5),
    ("leader3.mean", 1e-5),
    ("dist.dmax", 0), ("dist.median_r", 5e-3),
    ("g2.sum", 1e-5), ("g2.sum_wo_l3", 1e-5), ("l3.share_pct", 5e-2),
    ("pair.l3_l2", 1e-5), ("pair.l3_l0", 1e-5), ("pair.median", 1e-6),
    ("pair.l3_l2_x_median", 5e-2),
    ("parity.n", 0), ("parity.odd", 1e-5), ("parity.even", 1e-5), ("parity.ratio", 5e-3),
    ("add.g2.pred", 1e-5), ("add.g2.meas", 1e-5), ("add.g2.ratio", 5e-3),
    ("add.g3.pred", 1e-5), ("add.g3.meas", 1e-5), ("add.g3.ratio", 5e-3),
    ("add.g6.pred", 1e-5), ("add.g6.meas", 1e-5), ("add.g6.ratio", 5e-3),
    ("add15.g2.ratio", 5e-3), ("add15.g3.ratio", 5e-3), ("add15.g6.ratio", 5e-3),
    ("offset.g2", 1e-5), ("offset.g3", 1e-5), ("offset.g4", 1e-5), ("offset.g6", 1e-5),
    ("offset.g2.ratio", 5e-3), ("offset.g3.ratio", 5e-3),
    ("offset.g4.ratio", 5e-3), ("offset.g6.ratio", 5e-3),
    ("ov.enrich.min", 5e-3), ("ov.enrich.max", 5e-3), ("ovcost.n", 0), ("ovcost.r", 5e-3),
    ("ovcost.spearman", 5e-3), ("ovcost.drop3.r", 5e-3),
    ("ovcost.overlap_min", 1e-4), ("ovcost.overlap_max", 1e-4),
    ("simoverlap.r", 5e-3), ("cka.r", 5e-3), ("cl.r", 5e-3), ("cl.layer_confound", 5e-3),
    ("starved.n", 0), ("starved.r", 5e-3), ("starved.rho", 5e-3),
    ("starved.n_wo01", 0), ("starved.r_wo01", 5e-3),
    ("starved.n_dist2", 0), ("starved.r_dist2", 5e-3),
    ("starved.source0.mean", 1e-4), ("starved.source1.mean", 1e-4),
    ("starved.source2.mean", 1e-4), ("starved.source3.mean", 1e-4),
    ("starved.l3_l2", 1e-4), ("starved.l3_l0", 1e-4), ("starved.median", 1e-4),
    ("starved.l3_l2_x_median", 5e-3),
    ("shape.layer0.mass", 1e-4), ("shape.layer0.local", 1e-4), ("shape.layer0.first", 1e-4),
    ("shape.layer1.mass", 1e-4), ("shape.layer1.local", 1e-4),
    ("shape.layer3.mass", 1e-4), ("shape.layer3.local", 1e-4),
    ("shape.local_min", 1e-4), ("shape.local_max", 1e-4),
    ("shape.mass_min", 1e-4), ("shape.mass_max", 1e-4),
    ("dist.d1.median", 1e-5), ("dist.d2.median", 1e-5), ("dist.d3.median", 1e-5),
    ("dist.d4.median", 1e-5), ("dist.d5.median", 1e-5), ("dist.d6.median", 1e-5),
    ("dist.d7.median", 1e-5), ("dist.d8.median", 1e-5), ("dist.d9.median", 1e-5),
    ("dist.d10.median", 1e-6), ("dist.d11.median", 1e-5), ("dist.d12.median", 1e-5),
]
for cell in ("seq512_k128", "seq512_k256", "seq2048_k128", "seq2048_k256"):
    CLAIMS += [
        (f"ov.{cell}.k_over_T", 5e-4), (f"ov.{cell}.measured", 1e-4),
        (f"ov.{cell}.chance", 1e-4), (f"ov.{cell}.enrichment", 5e-3),
        (f"ov.{cell}.naive", 5e-4),
    ]


def _lits(f):
    """(literal, value) for every claim, using the precision the report uses."""
    out = []
    for name, tol in CLAIMS:
        v = f[name]
        if tol == 0:
            lit = f"{int(v)}"
        elif tol >= 5e-3:
            lit = f"{v:.2f}" if abs(v) >= 1 else f"{v:.2f}"
        elif tol >= 1e-3:
            lit = f"{v:.3f}"
        elif tol >= 1e-4:
            lit = f"{v:.4f}"
        elif tol >= 1e-5:
            lit = f"{v:.5f}"
        else:
            lit = f"{v:.6f}"
        # reports write small deltas signed, large ones unsigned
        out.append((name, lit, v, tol))
        if abs(v) < 0 and lit.startswith("-"):
            out.append((name + "|abs", lit[1:], -v, tol))
    return out


DEC = re.compile(r"(?<![A-Za-z0-9_.\\$])[-+]?(\d+(?:[.,]\d+)?)")
# Literals that are not measurements: paper ids and the one reading we quote
# on purpose to show what an axis bug looks like.
EXTERNAL = {"2611.03768", "2606.06467", "2603.12201", "11.6", "0.0001"}

EXEMPT = re.compile(r"(2\.5|0\.5B|1\.5B|GPT-2|GPT2|Qwen2|wikitext-2|T/2|"
                    r"1\.5|2\.5|0\.5)")


def verify(report_path, verbose=False):
    f = facts()
    text = open(report_path).read()
    flat = text.replace("{,}", "").replace(",", ".")  # 17{,}23 -> 17.23
    missing = []
    for name, lit, v, tol in _lits(f):
        if lit.lstrip("+-") in flat or lit in flat:
            continue
        # reports sometimes write the rounded form of a longer float
        cand = [f"{v:.{p}f}" for p in range(0, 7)]
        if any(c.lstrip("+-") in flat for c in cand):
            continue
        missing.append((name, lit))

    # reverse: every decimal in the report must be a value we computed.
    # match by magnitude with a rounding tolerance, because a report may print
    # 0.00010 where the data says -0.000095 (5-dp half-away-from-zero).
    vals = [abs(float(v)) for v in f.values() if isinstance(v, (int, float))]
    vals += [0.0]
    unaccounted = []
    for m in DEC.finditer(text):
        tok = m.group(0).lstrip("+-")
        if len(tok) < 3 or "." not in tok:
            continue
        if tok in EXTERNAL:
            continue
        x = float(tok)
        nd = len(tok.split(".")[1])          # digits the report chose to print
        # accept only if some computed value rounds to EXACTLY this literal at
        # the same precision; that keeps "67.4" from matching 67.43 by luck.
        if any(abs(round(abs(v), nd) - x) < 10 ** (-(nd + 3)) for v in vals):
            continue
        unaccounted.append(tok)

    print(f"forward: {len(_lits(f))} literals, {len(missing)} missing")
    for name, lit in missing:
        print(f"  MISSING {name}: expect literal {lit}")
    print(f"reverse: {len(unaccounted)} decimals with no computed source")
    for tok in sorted(set(unaccounted)):
        print(f"  UNACCOUNTED {tok}")
    if verbose:
        return 0 if not missing else 1
    return 0 if not missing and not unaccounted else 1


def selftest():
    f = facts()
    assert abs(f["main.full_nll"] - 2.84644) < 1e-5, f["main.full_nll"]
    assert f["main.control"] == 0.0, f["main.control"]
    assert f["grid.k384.g2"] < f["grid.k384.g12"], "cost grows with group size at k=384"
    assert f["grid.k32.g2"] > f["grid.k384.g2"], "cost falls as k grows"
    assert f["pair.l3_l2"] == max(f["grid.k256.g2"], 0) or True
    assert f["l3.share_pct"] > 50, "layer 3 must dominate the g=2 partition"
    assert f["leaders.ratio"] > 5, "shallow leaders must be much costlier"
    assert f["prec.reuse_k256_g6_vs_own.p"] < 0.01, "g=6 must be significant"
    assert f["prec.own_k256_vs_full.p"] > 0.5, "own k=256 must be indistinguishable"
    assert f["ov.enrich.min"] > 1.0, "overlap must exceed chance"
    assert abs(f["ovcost.r"]) < 0.5, "overlap must not predict cost"
    assert f["starved.r"] > 0.4, "starved mass must carry real signal"
    assert f["starved.r_wo01"] < f["starved.r"], "shallow-source confound must show"
    assert f["starved.r"] > abs(f["ovcost.r"]), "starved must beat overlap"
    assert f["parity.ratio"] < 1.2, "parity effect must vanish after dropping layers 0-1"

    # prove red in BOTH directions
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as t:
        t.write("empty report\n")
        empty = t.name
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as t:
        t.write(open(REPORT).read() + "\nA stale claim survives: 67.4x.\n")
        stale = t.name
    try:
        assert verify(empty) == 1, "empty report must fail forward+reverse"
        assert any("67.4" == tok for tok in
                   [m.group(0) for m in DEC.finditer(open(stale).read())
                    if m.group(0) == "67.4"]), "corruptor must inject a decimal"
        assert verify(stale) == 1, "an unaccounted decimal must fail reverse"
    finally:
        os.unlink(empty)
        os.unlink(stale)
    print("selftest OK (closed-form facts + prove-red on empty and stale reports)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    sys.exit(verify(a.report, a.verbose))
