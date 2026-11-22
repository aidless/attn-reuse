#!/usr/bin/env python3
"""Why can a 2.5-nat/token NLL cost leave MCQ decisions nearly intact?

The MCQ sweep says the decision-level cost is small and not significant at
n=200 while the token-level log-prob shift is huge (mean score shift -2.55 per
token at k=64/g=6 on 0.5B). Those two are not contradictory — they measure
different things — but the explanation has to be measured, not asserted.

The candidate explanation is a **common offset**: reuse shifts every option's
score by roughly the same amount within an item, and argmax ignores common
offsets. So this script decomposes the per-option score change into

    shift_mean(item)  = mean over options of (reuse_score - own_score)
    shift_spread(item) = max - min over options of that same difference

and then asks which of the two carries the decision changes:

* if |spread| is small compared with the margin between the top two options,
  the decision is invariant no matter how large the offset is;
* the items that flip should be the ones where the spread is large relative to
  their margin — and that is checkable.

Outputs per item: own scores, reuse scores, margin, mean shift, spread, and
whether the reading flipped. Aggregate: correlation between |spread| and flip,
and the same after dividing spread by margin.

Run: python3 mcq_offset.py --model models/Qwen2.5-0.5B --n 120 \
        --k 64 --group 6 --shots 8 --out out/mcq_offset.json
     python3 mcq_offset.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from mcq_reuse import STATE, build_items, patch_model, score_item


def rank_auc(pos, neg):
    """Probability that a random flipped item has a larger value than a non-flipped.

    Plain Pearson on a 0/1 outcome is a poor summary: it is bounded by the class
    balance and reads as "no relation" whenever flips are a minority. The
    rank-based statistic does not have that defect, so both are reported.
    """
    if not pos or not neg:
        return None
    wins = 0.0
    for x in pos:
        for y in neg:
            wins += 1.0 if x > y else (0.5 if x == y else 0.0)
    return wins / (len(pos) * len(neg))


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def decompose(own_scores, reuse_scores):
    """Per-item decomposition of the score change into offset and spread."""
    diffs = [r - o for o, r in zip(own_scores, reuse_scores)]
    srt = sorted(own_scores, reverse=True)
    margin = srt[0] - srt[1]
    return {"diffs": diffs,
            "mean_shift": sum(diffs) / len(diffs),
            "spread": max(diffs) - min(diffs),
            "own_margin": margin,
            "reuse_argmax": max(range(len(reuse_scores)), key=lambda i: reuse_scores[i]),
            "own_argmax": max(range(len(own_scores)), key=lambda i: own_scores[i])}


def selftest():
    # 1. a pure common offset must not change argmax
    own = [-1.0, -2.0, -3.0]
    off = [x - 5.0 for x in own]
    d = decompose(own, off)
    assert d["mean_shift"] == -5.0 and d["spread"] == 0.0
    assert d["own_argmax"] == d["reuse_argmax"] == 0

    # 2. a differential shift CAN change argmax even when the mean shift is 0
    own = [-1.0, -1.5, -2.0]
    dif = [-0.1, +0.8, 0.0]
    re_ = [o + x for o, x in zip(own, dif)]
    d = decompose(own, re_)
    assert d["reuse_argmax"] == 1 and d["own_argmax"] == 0
    assert abs(d["mean_shift"] - sum(dif) / 3) < 1e-12
    assert abs(d["spread"] - (0.8 - (-0.1))) < 1e-12

    # 3. spread must be measured across options, not across items
    assert decompose([0, 0], [1, 3])["spread"] == 2.0

    # 4. pearson sanity
    assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12
    assert pearson([1, 2], [1, 2]) is None   # too short to correlate

    # 5. rank AUC must separate the two classes in the obvious directions and must
    #    return 0.5 for a value that carries no information
    assert rank_auc([3, 4], [1, 2]) == 1.0
    assert rank_auc([1, 2], [3, 4]) == 0.0
    assert rank_auc([2, 2], [2, 2]) == 0.5
    assert rank_auc([], [1]) is None

    print("selftest OK (common offset is decision-invariant, differential shift "
          "flips argmax, spread is across options, pearson guards)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--seq", type=int, default=384)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--out", default="out/mcq_offset.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), device_map="cuda",
        attn_implementation="eager")
    model.eval()
    patch_model(model)

    items = build_items("arc_easy", a.n + a.shots, tok, a.seq, a.shots)[a.shots:]
    rows = []
    for mode, k in (("own", a.k), ("reuse", a.k)):
        STATE.update({"mode": mode, "k": k, "group": a.group,
                      "own_sel": {}, "reuse_sel": {}, "fallbacks": 0, "borrows": 0})
        per_mode = []
        for head_ids, opt_ids, gold in items:
            STATE["reuse_sel"] = {}
            best, sc = score_item(model, head_ids, opt_ids, "cuda")
            per_mode.append({"best": best, "scores": sc, "gold": gold})
        if mode == "own":
            own_rows = per_mode
        else:
            re_rows = per_mode
    if STATE.get("borrows", 0) == 0:
        print("NO-OP REUSE: nothing was borrowed", flush=True)
        return 3

    for o, r in zip(own_rows, re_rows):
        d = decompose(o["scores"], r["scores"])
        rows.append(dict(d, gold=o["gold"], own_best=o["best"], reuse_best=r["best"],
                         flipped=int(o["best"] != r["best"]),
                         assert_own_matches_gold=None))

    # aggregate: does |spread| explain flipping, before and after scaling by margin
    sp = [abs(x["spread"]) for x in rows]
    sp_rel = [abs(x["spread"]) / x["own_margin"] for x in rows if x["own_margin"] > 1e-9]
    fl = [x["flipped"] for x in rows]
    fl_rel = [x["flipped"] for x in rows if x["own_margin"] > 1e-9]
    out = {
        "schema": "mcq-offset/1", "model": os.path.basename(a.model), "n_items": len(rows),
        "k": a.k, "group": a.group, "shots": a.shots,
        "aggregate": {
            "n_flipped": sum(fl),
            "mean_shift_mean": st.mean(x["mean_shift"] for x in rows),
            "mean_spread_mean": st.mean(x["spread"] for x in rows),
            "median_spread": st.median(x["spread"] for x in rows),
            "mean_margin": st.mean(x["own_margin"] for x in rows),
            "r_spread_vs_flip": pearson(sp, fl),
            "r_relspread_vs_flip": pearson(sp_rel, fl_rel),
            "flip_spread_median": (st.median([abs(x["spread"]) for x in rows if x["flipped"]])
                                   if sum(fl) else None),
            "nofip_spread_median": (st.median([abs(x["spread"]) for x in rows if not x["flipped"]])
                                    if sum(fl) < len(rows) else None),
            # The decision is a comparison against the margin, so the margin is the
            # candidate explanation to beat: a perturbed item flips when the
            # differential shift is large *relative to* how close the top two
            # options were. Reported per flip / no-flip and as rank AUC.
            "flip_margin_mean": (st.mean(x["own_margin"] for x in rows if x["flipped"])
                                 if sum(fl) else None),
            "nofip_margin_mean": (st.mean(x["own_margin"] for x in rows if not x["flipped"])
                                  if sum(fl) < len(rows) else None),
            "flip_margin_median": (st.median([x["own_margin"] for x in rows if x["flipped"]])
                                   if sum(fl) else None),
            "nofip_margin_median": (st.median([x["own_margin"] for x in rows if not x["flipped"]])
                                    if sum(fl) < len(rows) else None),
            "auc_margin_predicts_flip": rank_auc(
                [-x["own_margin"] for x in rows if x["flipped"]],
                [-x["own_margin"] for x in rows if not x["flipped"]]),
            "auc_spread_predicts_flip": rank_auc(
                [abs(x["spread"]) for x in rows if x["flipped"]],
                [abs(x["spread"]) for x in rows if not x["flipped"]]),
            "auc_mean_shift_predicts_flip": rank_auc(
                [abs(x["mean_shift"]) for x in rows if x["flipped"]],
                [abs(x["mean_shift"]) for x in rows if not x["flipped"]]),
            # how the gold option is treated relative to the average other option:
            # a directionless perturbation penalises them equally, a directional
            # one penalises the correct answer more.
            "gold_penalty_flip_mean": (st.mean(x["diffs"][x["gold"]]
                                               - st.mean([v for i, v in enumerate(x["diffs"])
                                                          if i != x["gold"]])
                                               for x in rows if x["flipped"])
                                       if sum(fl) else None),
            "gold_penalty_nofip_mean": (st.mean(x["diffs"][x["gold"]]
                                                - st.mean([v for i, v in enumerate(x["diffs"])
                                                           if i != x["gold"]])
                                                for x in rows if not x["flipped"])
                                        if sum(fl) < len(rows) else None),
        },
        "items": rows,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)
    for k2, v in out["aggregate"].items():
        print(f"  {k2}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
