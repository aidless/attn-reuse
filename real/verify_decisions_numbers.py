#!/usr/bin/env python3
"""Two-way number check for reports/07-decisions.md.

Forward: every number the report prints must be derivable from the artifacts.
Reverse: every decimal in the report must match some computed value at the
precision the report prints it — so a number that quietly drifted out of the
data cannot survive a later edit of the data.

This is the same instrument as `verify_report_numbers.py`, kept separate because
report 07 reads a different artifact set (mcq_*.json) and because a single
script growing to cover both would hide which report broke.

Run: python3 verify_decisions_numbers.py [--report PATH] [--verbose]
     python3 verify_decisions_numbers.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPORT = os.path.join(HERE, "..", "reports", "07-decisions.md")

DEC = re.compile(r"[-+\u2212]?\d+(?:\.\d+)?(?:e[-+\u2212]?\d+)?", re.I)
HEADING = re.compile(r"^#{1,6}\s+[^\n]*", re.M)
# A cross-reference such as "see §8.2" is a pointer, not a measurement. The token
# is exempt only when it directly follows a section sign.
SECTREF = re.compile(r"§(\d+(?:\.\d+)*)")


def load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / (sxx * syy) ** 0.5


def two_sided_exact(b, c):
    """Exact two-sided McNemar p from the two discordant counts.

    Same binomial tail as mcnemar_exact in analyze_mcq.py, re-implemented here
    on purpose: this file must be able to check the artifacts without
    importing the module that produced them.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def facts():
    f = {}
    s = load("mcq_summary.json")
    pw = load("mcq_power.json")
    main, m15 = load("main.json"), load("main_15b.json")
    per = {}
    for r in s["cells"]:
        per.setdefault(r["model"], []).append(r)

    # dense accuracy per model, from the readings of the raw artifacts
    for tag, name in (("Qwen2.5-0.5B", "mcq_05b.json"),
                      ("Qwen2.5-1.5B", "mcq_15b.json"),
                      ("Qwen2.5-7B", "mcq_7b.json")):
        doc = load(name)
        f[f"{tag}.dense_acc"] = doc["readings"]["full"]
        f[f"{tag}.n_items"] = doc["n_items"]
        f[f"{tag}.shots"] = doc["shots"]
        f[f"{tag}.seq"] = doc["seq"]
        f[f"{tag}.prompt_median"] = doc["prompt_len_median"]
        f[f"{tag}.prompt_max"] = doc["prompt_len_max"]
        f[f"{tag}.control_dev"] = doc["control"]["max_abs_score_deviation"]
        f[f"{tag}.fallbacks_max"] = max((doc["fallbacks"] or {"x": 0}).values())

    # per-cell accuracy deltas in percentage points, keyed by model/k/group
    for tag, rows in per.items():
        f[f"{tag}.cells"] = len(rows)
        f[f"{tag}.n_significant"] = sum(1 for r in rows if r["p_exact"] < 0.05)
        for r in rows:
            key = f"{tag}.k{r['k']}.g{r['group']}"
            f[key + ".acc_pt"] = 100 * r["acc_diff"]
            f[key + ".p"] = r["p_exact"]
            f[key + ".b"] = r["right_to_wrong"]
            f[key + ".c"] = r["wrong_to_right"]

    # per-model discordance totals and their ratio
    for tag, agg in s["per_model"].items():
        f[f"{tag}.sum_b"] = agg["b"]
        f[f"{tag}.sum_c"] = agg["c"]
        f[f"{tag}.b_over_c"] = agg["b_over_c"]
        f[f"{tag}.min_cell_p"] = agg["min_cell_p"]

    # NLL band table: band -> n, mean acc pt, max |acc pt|, min p
    m0 = main["deltas"]
    rows0 = [r for r in per["Qwen2.5-0.5B"]
             if f"k{r['k']}_g{r['group']}" in m0]
    bands = [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
    for lo, hi in bands:
        sel = [r for r in rows0
               if lo <= m0[f"k{r['k']}_g{r['group']}"]["vs_own"] < hi]
        key = f"band.{lo}.{hi}"
        f[key + ".n"] = len(sel)
        f[key + ".mean_pt"] = (sum(100 * r["acc_diff"] for r in sel) / len(sel)
                               if sel else None)
        f[key + ".max_abs_pt"] = (max(abs(100 * r["acc_diff"]) for r in sel)
                                  if sel else None)
        f[key + ".min_p"] = min((r["p_exact"] for r in sel), default=None)

    # correlation between NLL cost and accuracy damage on the 0.5B grid
    xs = [m0[f"k{r['k']}_g{r['group']}"]["vs_own"] for r in rows0]
    ys = [100 * r["acc_diff"] for r in rows0]
    f["corr.nll_vs_acc_pt"] = pearson(xs, ys)
    f["corr.n"] = len(xs)

    # scale comparison at matched k/g: 1.5B cost as a multiple of 0.5B
    for k, g in ((128, 2), (256, 2)):
        a = m0[f"k{k}_g{g}"]["vs_own"]
        b = m15["deltas"][f"k{k}_g{g}"]["vs_own"]
        f[f"scale_ratio.k{k}_g{g}"] = b / a
        f[f"nll05.k{k}_g{g}"] = a
        f[f"nll15.k{k}_g{g}"] = b

    # offset decomposition
    for tag in ("k64g6", "k128g6"):
        d = load(f"mcq_offset_05b_{tag}.json")
        a = d["aggregate"]
        p = f"off.{tag}"
        f[p + ".n"] = d["n_items"]
        f[p + ".n_flipped"] = a["n_flipped"]
        f[p + ".mean_shift"] = a["mean_shift_mean"]
        f[p + ".spread"] = a["mean_spread_mean"]
        f[p + ".margin"] = a["mean_margin"]
        f[p + ".auc_margin"] = a["auc_margin_predicts_flip"]
        f[p + ".auc_spread"] = a["auc_spread_predicts_flip"]
        f[p + ".auc_shift"] = a["auc_mean_shift_predicts_flip"]
        f[p + ".flip_margin"] = a["flip_margin_mean"]
        f[p + ".nofip_margin"] = a["nofip_margin_mean"]
        f[p + ".gold_penalty_flip"] = a["gold_penalty_flip_mean"]
        f[p + ".gold_penalty_nofip"] = a["gold_penalty_nofip_mean"]

    # power table
    for r in pw["cells"]:
        key = f"pw.{r['model']}.k{r['k']}.g{r['group']}"
        f[key + ".mde"] = r["mde"]["net_loss"] if r["mde"] else None
        f[key + ".mde_pt"] = r["mde"]["share_of_items"] * 100 if r["mde"] else None
        f[key + ".discordant"] = r["discordant"]
        f[key + ".net"] = r["net_loss"]
        f[key + ".p"] = r["p_exact"]
        f[key + ".ci_lo"] = r["ci95"]["net_loss_low"] if r["ci95"] else None
        f[key + ".ci_hi"] = r["ci95"]["net_loss_high"] if r["ci95"] else None
        f[key + ".rejection_region"] = r["rejection_region_exists"]
        f[key + ".n_items"] = r["n_items"]

    # NLL cost of every 0.5B cell that also appears in the MCQ grid, so the
    # report can quote a NLL figure next to an accuracy figure and have it stay
    # checkable (report §9 does exactly that for k=64/g=3).
    for r in rows0:
        key = f"k{r['k']}_g{r['group']}"
        f[f"nll05.{key}"] = m0[key]["vs_own"]

    # the control failure that forced the mask rewrite, quoted verbatim in §8.1
    cf = os.path.join(DATA, "control_failure_7b.txt")
    if os.path.exists(cf):
        for m in re.finditer(r"max \|score deviation\| = (\d+\.\d+e[-+]\d+)",
                             open(cf).read()):
            f["ctl7b.fail_dev"] = float(m.group(1))

    # MDE range per model (only cells that have a rejection region)
    for tag in ("Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-7B"):
        vals = [r["mde"]["share_of_items"] * 100 for r in pw["cells"]
                if r["model"] == tag and r["mde"]]
        f[f"{tag}.mde_min_pt"] = min(vals) if vals else None
        f[f"{tag}.mde_max_pt"] = max(vals) if vals else None

    # --- second substrate (ARC-Challenge) -----------------------------------
    # Its numbers live in their own artifacts rather than mcq_summary.json, so
    # they are loaded here; without this the reverse check would call every
    # ARC-Challenge figure unsourced, which is exactly what happened when §3.1
    # was first written.
    for tag, name in (("arcc.Qwen2.5-1.5B", "arcc_15b.json"),
                      ("arcc.Qwen2.5-7B", "arcc_7b.json")):
        doc = load(name)
        N = doc["n_items"]
        f[f"{tag}.dense_acc"] = doc["readings"]["full"]
        f[f"{tag}.n_items"] = N
        f[f"{tag}.prompt_median"] = doc["prompt_len_median"]
        f[f"{tag}.control_dev"] = doc["control"]["max_abs_score_deviation"]
        cells = 0
        sig = 0
        for key, fl in doc["flips_vs_own"].items():
            own = doc["readings"][f"own_{key.split('_')[0]}"]
            reuse = doc["readings"][f"reuse_{key}"]
            b, c = fl["right_to_wrong"], fl["wrong_to_right"]
            p = two_sided_exact(b, c)
            f[f"{tag}.{key}.acc_pt"] = 100 * (reuse - own)
            f[f"{tag}.{key}.b"] = b
            f[f"{tag}.{key}.c"] = c
            f[f"{tag}.{key}.p"] = p
            cells += 1
            sig += 1 if p < 0.05 else 0
        f[f"{tag}.cells"] = cells
        f[f"{tag}.n_significant"] = sig

    # --- the at-risk normalisation that replaced the b:c claim --------------
    cb = load("cross_bench.json")
    for bench, s in cb.items():
        if bench == "shot_control":
            continue      # different shape; handled just below
        p_ = s["pooled"]
        if not p_:
            continue
        f[f"cb.{bench}.damage_pct"] = 100 * p_["damage_rate"]
        f[f"cb.{bench}.rescue_pct"] = 100 * p_["rescue_rate"]
        f[f"cb.{bench}.asymmetry"] = p_["asymmetry"]
        f[f"cb.{bench}.damage_lo_pct"] = 100 * p_["damage_rate_ci"][0]
        f[f"cb.{bench}.damage_hi_pct"] = 100 * p_["damage_rate_ci"][1]
        f[f"cb.{bench}.rescue_lo_pct"] = 100 * p_["rescue_rate_ci"][0]
        f[f"cb.{bench}.rescue_hi_pct"] = 100 * p_["rescue_rate_ci"][1]
        f[f"cb.{bench}.identity_residual"] = max(
            m["identity_residual"] for m in s["models"]
            if m["identity_residual"] is not None)

    # --- the shot control: the same 6 cells at 8 and at 5 exemplars --------
    sc = cb["shot_control"]
    for arm, s in sc["arms"].items():
        f[f"sc.{arm}.b"] = s["b"]
        f[f"sc.{arm}.c"] = s["c"]
        f[f"sc.{arm}.damage_pct"] = s["damage_pct"]
        f[f"sc.{arm}.rescue_pct"] = s["rescue_pct"]
        f[f"sc.{arm}.asymmetry"] = s["asymmetry"]
        f[f"sc.{arm}.cells"] = s["cells"]
        f[f"sc.{arm}.damage_lo_pct"] = s["damage_pct_ci"][0]
        f[f"sc.{arm}.damage_hi_pct"] = s["damage_pct_ci"][1]
        f[f"sc.{arm}.rescue_lo_pct"] = s["rescue_pct_ci"][0]
        f[f"sc.{arm}.rescue_hi_pct"] = s["rescue_pct_ci"][1]
    f["sc.flip.sign_reversal"] = sc["flip"]["sign_reversal"]
    f["sc.flip.damage_minus_rescue_pp_eight"] = sc["flip"]["damage_minus_rescue_pp_eight"]
    f["sc.flip.damage_minus_rescue_pp_five"] = sc["flip"]["damage_minus_rescue_pp_five"]

    # --- and the benchmark that was dropped, with its measured reason -------
    mp = load("mmlu_probe.json")
    for model, r in mp["metrics"].items():
        f[f"mmlu.{model}.raw"] = r["raw"]
        f[f"mmlu.{model}.lengthnorm"] = r["lengthnorm"]
    f["mmlu.chance"] = mp["chance"]
    f["mmlu.n_items"] = mp["n_items"]

    # the 0.5B ARC-Challenge probe, re-run after the prompt builder was fixed
    # (the earlier artifact had prompt_len_median 68 — a truncated prompt — and
    # its readings are not comparable to the two models that were kept)
    pp = load("probe_arcc_05b.json")
    f["arcc.Qwen2.5-0.5B.dense_acc"] = pp["readings"]["full"]
    f["arcc.Qwen2.5-0.5B.n_items"] = pp["n_items"]
    f["arcc.Qwen2.5-0.5B.prompt_median"] = pp["prompt_len_median"]
    f["arcc.chance"] = 0.25

    return f


def verify(path, verbose=False):
    f = facts()
    text = open(path).read()
    flat = json.dumps(f, default=str)

    # forward: the values that carry the argument must be present verbatim-ish
    must = ["corr.nll_vs_acc_pt", "off.k64g6.auc_margin", "off.k128g6.auc_margin",
            "off.k64g6.mean_shift", "off.k64g6.spread", "off.k64g6.margin",
            "band.0.2.mean_pt", "band.2.0.4.0.max_abs_pt",
            "Qwen2.5-0.5B.dense_acc", "Qwen2.5-1.5B.dense_acc", "Qwen2.5-7B.dense_acc",
            # the at-risk normalisation is now the report's directionality claim,
            # so its four values have to be pinned forward, not only reverse-
            # checked. `b_over_c` is deliberately absent: §6 retired it.
            "cb.arc_easy.damage_pct", "cb.arc_easy.rescue_pct",
            "cb.arc_easy.asymmetry", "cb.arc_challenge.damage_pct",
            "cb.arc_challenge.rescue_pct", "cb.arc_challenge.asymmetry",
            "arcc.Qwen2.5-1.5B.dense_acc", "arcc.Qwen2.5-7B.dense_acc"]
    missing = []
    for key in must:
        v = f.get(key)
        if v is None:
            continue
        cands = {f"{v:.4f}", f"{v:.3f}", f"{v:.2f}", f"{v:.1f}", f"{v:.0f}",
                 f"{abs(v):.4f}", f"{abs(v):.3f}", f"{abs(v):.2f}", f"{abs(v):.1f}"}
        if not any(c in text for c in cands):
            missing.append(f"{key}={v}")

    # reverse: every number printed with a decimal point must equal some computed
    # value at the precision it is printed with. Compared numerically with a
    # half-ulp tolerance, not by string equality — the report rounds half up
    # ("0.617" from 0.61649) which string formatting would call a mismatch.
    #
    # LIMITATION, stated rather than hidden: bare integers (k, group size, layer
    # counts, item counts, section numbers) are treated as structural and not
    # reverse-checked. Re-checking them would flag every "k=128" in the tables.
    # The forward list above is what pins the load-bearing values.
    unaccounted = []
    # Section numbers in headings are structural, like k and group sizes; they are
    # exempt here and named in the docstring rather than silently ignored.
    heading_spans = [(m.start(), m.end()) for m in HEADING.finditer(text)]

    sectref_spans = [(m.start(1), m.end(1)) for m in SECTREF.finditer(text)]

    def in_heading(pos):
        return any(a <= pos < b for a, b in heading_spans + sectref_spans)

    numeric = [v for v in f.values() if isinstance(v, (int, float)) and v is not None]
    for m in DEC.finditer(text):
        tok = m.group(0).replace("\u2212", "-")
        if in_heading(m.start()):
            continue
        if "." not in tok and "e" not in tok.lower():
            continue
        try:
            x = float(tok)
        except ValueError:
            continue
        prec = len(tok.split(".")[1].split("e")[0]) if "." in tok else 0
        tol = 0.5 * (10 ** -prec) + 1e-9
        if any(abs(v - x) <= tol or abs(abs(v) - abs(x)) <= tol for v in numeric):
            continue
        unaccounted.append(tok)

    if verbose:
        print(f"forward: {len(must)} key values; missing {len(missing)}")
        for x in missing:
            print("  MISSING", x)
        print(f"reverse: {len(unaccounted)} decimals with no computed source")
        for tok in unaccounted[:40]:
            print("  UNACCOUNTED", tok)
    return 0 if not missing and not unaccounted else 1


def selftest():
    f = facts()
    assert abs(f["Qwen2.5-0.5B.dense_acc"] - 0.61) < 1e-9, f["Qwen2.5-0.5B.dense_acc"]
    assert abs(f["Qwen2.5-7B.dense_acc"] - 0.7933) < 0.005, f["Qwen2.5-7B.dense_acc"]
    assert f["Qwen2.5-0.5B.cells"] == 25, f["Qwen2.5-0.5B.cells"]
    assert f["Qwen2.5-0.5B.n_significant"] == 5, f["Qwen2.5-0.5B.n_significant"]
    assert f["corr.nll_vs_acc_pt"] < -0.9, "cost and damage must track at band scale"
    # These two assertions used to pin b_over_c > 4 ("flips must be
    # directional"). That was the retired claim: b:c restates the accuracy drop
    # (b-c = N*(own-reuse) is an identity), so a large ratio is not evidence of
    # asymmetry. The directionality assertion now lives on the at-risk rates
    # further down, where the two ratios have different denominators.
    assert f["cb.arc_easy.asymmetry"] < 3.5, "b:c must not be re-quoted as 5:1"
    assert f["cb.arc_challenge.asymmetry"] < 3.5
    assert f["off.k64g6.auc_margin"] > 0.7, "margin must predict flips"
    assert f["off.k64g6.auc_spread"] < f["off.k64g6.auc_margin"], \
        "spread must not beat margin"
    assert f["off.k64g6.mean_shift"] < -2.0, "the common offset must be large"
    assert abs(f["off.k64g6.margin"] - 2.83) < 0.05

    # every CI must contain its own point estimate — the failure mode this
    # script was written after (see report §8.2)
    for key, v in f.items():
        if key.endswith(".ci_lo") and v is not None:
            base = key[:-6]
            lo, hi, net = v, f[base + ".ci_hi"], f[base + ".net"]
            assert lo <= net <= hi, (base, lo, net, hi)
    # no cell without a rejection region may carry an MDE
    for key, v in f.items():
        if key.endswith(".mde") and v is not None:
            base = key[:-4]
            assert f[base + ".rejection_region"], base

    # --- the second substrate and the at-risk rewrite -----------------------
    # The b-c identity is what made the old headline non-evidence; if it ever
    # stops holding exactly, the reason to distrust b:c disappears too.
    for bench in ("arc_easy", "arc_challenge"):
        resid = f.get(f"cb.{bench}.identity_residual")
        assert resid is not None and resid < 0.5, (bench, resid)
    # damage must exceed rescue on both benchmarks, and the asymmetry must be
    # in the modest 1.3-2x range rather than the 5x the old claim implied
    for bench in ("arc_easy", "arc_challenge"):
        asym = f[f"cb.{bench}.asymmetry"]
        assert 1.2 < asym < 2.2, (bench, asym)
    for bench in ("arc_easy", "arc_challenge"):
        lo, hi = f[f"cb.{bench}.damage_lo_pct"], f[f"cb.{bench}.damage_hi_pct"]
        pt = f[f"cb.{bench}.damage_pct"]
        assert lo <= pt <= hi, (bench, lo, pt, hi)
        assert f[f"cb.{bench}.damage_pct"] > f[f"cb.{bench}.rescue_pct"], bench
    # ARC-Challenge was measured on more wrong than correct items, so it is the
    # benchmark that rules out "drift toward whichever side is larger"
    assert f["arcc.Qwen2.5-1.5B.n_items"] == 200
    assert f["arcc.Qwen2.5-7B.n_items"] == 120
    # the dropped benchmarks must carry their measured floor, not a verdict
    assert f["mmlu.chance"] == 0.25
    for model in ("Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-7B"):
        assert f[f"mmlu.{model}.raw"] <= f["mmlu.chance"] + 0.02, model

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as t:
        t.write("empty report\n")
        empty = t.name
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as t:
        t.write(open(REPORT).read() + "\nA stale decimal survives: 88.8815.\n")
        stale = t.name
    # the shot control has to be able to fail in the direction that matters:
    # if the two arms ever stop disagreeing in sign, the report's caveat about
    # mixed exemplar counts loses its evidence and the text has to change.
    try:
        assert verify(empty) == 1, "empty report must fail"
        assert verify(stale) == 1, "an unaccounted decimal must fail reverse"
        assert verify(REPORT) == 0, "the shipped report must pass both directions"
    finally:
        os.unlink(empty)
        os.unlink(stale)
    # and the assertion that the control is not vacuous, on the real artifact
    real = load("cross_bench.json")["shot_control"]["flip"]
    assert real["sign_reversal"] is True, real
    assert real["asymmetry_eight_shot"] < 1.0 < real["asymmetry_five_shot"], real
    print("selftest OK (facts pinned, CI contains its estimate, MDE only where "
          "rejection region exists, shot control reverses sign, prove-red on "
          "empty and stale)")


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
