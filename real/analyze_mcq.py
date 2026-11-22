#!/usr/bin/env python3
"""Turn per-item MCQ outcomes into decision-level statements.

The NLL grid answers "how many nats does reuse cost". This answers a different
question: "does it change which option the model picks". Two accuracies are not
enough to answer that — 0.75 vs 0.75 can hide six items flipping each way. So
everything here is built from the per-item outcome vectors that
`mcq_reuse.py` records, not from the aggregate accuracies.

Three quantities, in increasing strength:

1. **flip counts** — right→wrong and wrong→right separately. Their sum is not
   the cost; their *asymmetry* is.
2. **McNemar exact two-sided test on the discordant pairs** — the paired test
   that is the right one here, because the same items are scored under both
   readings. With small discordant counts the asymptotic chi-square is the wrong
   instrument (that is the whole subject of `exactness-audit/`), so this uses the
   exact binomial.
3. **per-model, per-k, per-group sign consistency** — if reuse were noise, the
   right→wrong count would not exceed wrong→right in every cell.

Run: python3 analyze_mcq.py --in ../real/data/mcq_7b.json,...
     python3 analyze_mcq.py --selftest
"""
from __future__ import annotations

import argparse
import re
import json
import math
import os
import sys
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p over discordant counts b, c.

    Sum of binomial tails at p=1/2, clipped at 1. Small-n correct by
    construction; at n>900 the log-gamma form avoids 2**n overflow.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if n <= 900:
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
        return min(1.0, 2.0 * tail)
    # log domain
    lg = math.lgamma
    tail = 0.0
    for i in range(k + 1):
        tail += math.exp(lg(n + 1) - lg(i + 1) - lg(n - i + 1) - n * math.log(2.0))
    return min(1.0, 2.0 * tail)


def cells(doc):
    """One row per reuse cell: accuracy pair, flip counts, exact McNemar p.

    `acc_diff` is written as own -> reuse so the sign convention is unambiguous
    (negative = reuse is worse). `n_items` comes from the artifact, never from
    counting a list that might be truncated.
    """
    out = []
    name = os.path.basename(doc["model"])
    for key, acc in doc["readings"].items():
        if not key.startswith("reuse_k"):
            continue
        # key is "reuse_k128_g3"; strip "reuse_" (not "reuse_k", which would
        # eat the "k" and leave an empty number).
        k_str, g_str = key[len("reuse_"):].split("_")
        k, g = int(k_str[1:]), int(g_str[1:])
        fl = doc["flips_vs_own"][f"k{k}_g{g}"]
        b, c = fl["right_to_wrong"], fl["wrong_to_right"]
        own_acc = doc["readings"][f"own_k{k}"]
        full_acc = doc["readings"]["full"]
        out.append({"model": name, "bench": doc.get("bench"), "shots": doc.get("shots"),
                    "k": k, "group": g, "n_items": doc["n_items"],
                    "own_acc": own_acc, "reuse_acc": acc, "full_acc": full_acc,
                    # percentage points, so a report that quotes "−25.5 pt" cites a
                    # field holding exactly that number instead of a fraction the
                    # reader has to re-scale (a scale mismatch is a silent defect).
                    "acc_diff": acc - own_acc,
                    "acc_pt": 100.0 * (acc - own_acc),
                    # dense accuracy of the same item set, repeated on every cell
                    # so a claim can cite one cell instead of a global reading
                    "dense_acc": full_acc,
                    "own_minus_full": own_acc - full_acc,
                    "right_to_wrong": b, "wrong_to_right": c,
                    "discordant": b + c, "net_loss": b - c,
                    "p_exact": mcnemar_exact(b, c),
                    "borrows": (doc.get("borrows") or {}).get(f"k{k}_g{g}"),
                    "fallbacks": (doc.get("fallbacks") or {}).get(f"k{k}_g{g}"),
                    "mean_score_shift_vs_own":
                        (doc.get("score_delta_vs_own") or {}).get(f"k{k}_g{g}")})
    return out


def summarize(docs):
    rows = []
    for d in docs:
        rows.extend(cells(d))
    return rows


def selftest():
    # 1. exact McNemar sanity: the classic b=0,c=4 -> 0.125
    assert abs(mcnemar_exact(0, 4) - 0.125) < 1e-12, mcnemar_exact(0, 4)
    assert abs(mcnemar_exact(4, 4) - 1.0) < 1e-12
    # 2. symmetric in b,c and never above 1
    for b in range(6):
        for c in range(6):
            p = mcnemar_exact(b, c)
            assert 0 <= p <= 1
            assert abs(p - mcnemar_exact(c, b)) < 1e-12
    # 3. large-n path must stay finite and small for a lopsided split
    assert mcnemar_exact(10, 300) < 1e-10
    # 4. no discordant pairs -> p = 1, not 0
    assert mcnemar_exact(0, 0) == 1.0

    # 5. flip accounting is order-insensitive in aggregate but not in sign
    own = [1, 1, 1, 0, 0]
    re_ = [1, 1, 0, 1, 0]
    b = sum(1 for x, y in zip(own, re_) if x and not y)
    c = sum(1 for x, y in zip(own, re_) if not x and y)
    assert (b, c) == (1, 1), (b, c)

    # 6. cells() must not silently read a missing flips record
    doc = {"model": "/x/0.5B", "n_items": 4, "readings": {"full": 0.5, "own_k8": 0.5,
            "reuse_k8_g2": 0.25},
           "flips_vs_own": {"k8_g2": {"right_to_wrong": 1, "wrong_to_right": 0}}}
    cs = cells(doc)
    assert len(cs) == 1 and cs[0]["k"] == 8 and cs[0]["group"] == 2
    print("selftest OK (b=0,c=4 -> 0.125; symmetry; large-n finite; no-discordant p=1; "
          "flip sign; cell parsing)")


def corr_with_nll(rows, main_path):
    """Pearson(cost, damage) on the cells that exist in the token-NLL grid.

    This is the report's §4 number. It lives in the artifact rather than in
    prose so a claim gate can cite it by name: a correlation quoted from memory
    is the kind of number that quietly survives a data regeneration.
    """
    if not os.path.exists(main_path):
        return None, 0
    main = json.load(open(main_path))
    pairs = []
    for r in rows:
        key = f"k{r['k']}_g{r['group']}"
        if key in main["deltas"]:
            pairs.append((main["deltas"][key]["vs_own"], 100.0 * r["acc_diff"]))
    if len(pairs) < 3:
        return None, len(pairs)
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((a - mx) * (b - my) for a, b in pairs)
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None, len(pairs)
    return sxy / (sxx * syy) ** 0.5, len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", default="")
    ap.add_argument("--out", default=os.path.join(DATA, "mcq_summary.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    paths = [p for p in a.inputs.split(",") if p] if a.inputs else sorted(
        os.path.join(DATA, f) for f in os.listdir(DATA) if f.startswith("mcq_") and
        f.endswith(".json") and f != "mcq_summary.json")
    docs = [json.load(open(p)) for p in paths if os.path.exists(p)]
    if not docs:
        print("no input files")
        return 1

    rows = summarize(docs)
    out = {"schema": "mcq-summary/1", "sources": [os.path.basename(p) for p in paths],
           "cells": rows,
           "sign_consistency": {
               "cells": len(rows),
               "cells_net_loss_positive": sum(1 for r in rows if r["net_loss"] > 0),
               "cells_net_loss_zero": sum(1 for r in rows if r["net_loss"] == 0),
               "cells_net_loss_negative": sum(1 for r in rows if r["net_loss"] < 0),
               "cells_p_below_0.05": sum(1 for r in rows if r["p_exact"] < 0.05),
           }}
    # A reuse cell that borrowed nothing is a copy of its own-k reading; such a
    # cell must be excluded from every aggregate rather than averaged in.
    noop = [r for r in rows if r.get("borrows") in (0, None)]
    out["noop_cells"] = [f"{r['model']}/k{r['k']}/g{r['group']}" for r in noop]
    rows = [r for r in rows if r.get("borrows") not in (0, None)]
    out["cells"] = rows
    # Per-model discordance shape. The counts are descriptive: cells share their
    # items across k, so pooling b and c over cells is NOT an independent test and
    # no p-value is derived from these sums. The per-cell exact p is the evidence.
    by_model = {}
    for r in rows:
        m = by_model.setdefault(r["model"], {"cells": 0, "b": 0, "c": 0})
        m["cells"] += 1
        m["b"] += r["right_to_wrong"]
        m["c"] += r["wrong_to_right"]
    for m in by_model.values():
        tot = m["b"] + m["c"]
        m["discordant"] = tot
        m["share_right_to_wrong"] = (m["b"] / tot) if tot else None
        m["b_over_c"] = (m["b"] / m["c"]) if m["c"] else None
        m["min_cell_p"] = None
    for r in rows:
        cur = by_model[r["model"]]["min_cell_p"]
        if cur is None or r["p_exact"] < cur:
            by_model[r["model"]]["min_cell_p"] = r["p_exact"]

    # Keyed index so a claim gate can cite one cell by name instead of by list
    # position. Non-alphanumerics are normalised because the gate's resolver
    # walks dotted paths.
    def keyed(model, k, group):
        return re.sub(r"[^0-9A-Za-z]+", "_", f"{model}_k{k}_g{group}").strip("_")

    out["by_cell"] = {keyed(r["model"], r["k"], r["group"]): r for r in rows}
    out["per_model"] = by_model
    out["per_model_keyed"] = {re.sub(r"[^0-9A-Za-z]+", "_", k): v
                              for k, v in by_model.items()}
    out["per_model_note"] = ("b/c pooled over cells is descriptive only; cells share "
                             "items across k so the pooled counts are not independent")
    out["aggregate"] = {
        "cells": len(rows),
        "net_loss_positive": sum(1 for r in rows if r["net_loss"] > 0),
        "net_loss_zero": sum(1 for r in rows if r["net_loss"] == 0),
        "net_loss_negative": sum(1 for r in rows if r["net_loss"] < 0),
        "p_below_0.05": sum(1 for r in rows if r["p_exact"] < 0.05),
        "total_right_to_wrong": sum(r["right_to_wrong"] for r in rows),
        "total_wrong_to_right": sum(r["wrong_to_right"] for r in rows),
        "max_abs_acc_diff": max((abs(r["acc_diff"]) for r in rows), default=0.0),
    }
    # per-model correlation with the token-NLL grid, when that grid is present
    for model, mrows in ((m, [r for r in rows if r["model"] == m])
                         for m in {r["model"] for r in rows}):
        c, n = corr_with_nll(mrows, os.path.join(DATA, "main.json"))
        if c is not None:
            out.setdefault("corr_nll_vs_acc_pt_keyed", {})[
                re.sub(r"[^0-9A-Za-z]+", "_", model).strip("_")] = {"r": c, "n": n}

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"WROTE {a.out} cells={len(rows)} noop_excluded={len(noop)}")
    print(f"  {out['aggregate']}")
    for r in rows:
        print(f"  {r['model']:<22} {r['bench'] or '':<9} k={r['k']:<4} g={r['group']:<3} "
              f"acc {r['own_acc']:.3f}->{r['reuse_acc']:.3f} "
              f"b={r['right_to_wrong']:<3} c={r['wrong_to_right']:<3} "
              f"p={r['p_exact']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
