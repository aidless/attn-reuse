#!/usr/bin/env python3
"""Does the cross-layer attention-reuse premise hold in a real model?

Two 2026 lines build on the same observation:

  * Multi-layer Indexer (arXiv 2611.03768) reuses one layer's selected tokens in
    the next layer to save decode time.
  * CLSA (arXiv 2606.06467) shares routing across layers to save prefill/storage.

Both are efficiency work and both assume that neighbouring layers attend to
substantially the same tokens. Neither reports how much of that assumption
survives in the raw attention of a shipped model. This script measures it.

The measurement has to clear three baselines or it proves nothing:

  1. adjacent-layer, same head      — the quantity the papers assume is high
  2. same-layer, different heads    — control: if heads already disagree wildly,
                                      "layers agree" is not a layer property
  3. random k-subsets               — analytic expectation k/(2S-k)

If adjacent-layer overlap is no higher than the same-layer cross-head overlap,
the premise is about position in the stack, not about layers specifically.

Usage:
    python3 layer_reuse.py --model openai-community/gpt2 --k 8
    python3 layer_reuse.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Short texts are a trap here: with seq~15 and k=8 the top-k is half the
# sequence, so even two unrelated attention maps overlap ~44% by chance and
# every comparison looks high. LONG_TEXTS keeps k/seq near 1/40 so the random
# baseline is ~2.5% and the contrast is real.
LONG_TEXTS = [
    "Sparse attention mechanisms select a small subset of keys for each query position, "
    "which reduces the quadratic cost of the attention operation. The selection is usually "
    "performed by a lightweight indexer that scores every key and keeps the top few. Because "
    "the indexer runs at every layer, its cost can dominate once the sequence is long. A "
    "natural optimisation is therefore to reuse the selection computed at one layer for the "
    "layers that follow, on the assumption that attention patterns change slowly with depth. "
    "Whether that assumption holds has mostly been argued from aggregate statistics rather "
    "than measured against a matched control. In this note we measure it directly, comparing "
    "the overlap of selections across depth with the overlap of selections across heads at "
    "the same depth, and with the overlap expected from independent sampling. The control "
    "matters because attention concentrates on a few positions regardless of layer, so any "
    "two attention maps taken from natural text will agree far above chance. Reporting only "
    "the across-depth overlap therefore cannot distinguish a property of the stack from a "
    "property of attention in general, and the distinction is what a reuse strategy needs.",
    "The committee met on a Tuesday morning in a room that had not been cleaned since the "
    "previous spring. Chairs were arranged in a rough semicircle facing a whiteboard on which "
    "someone had written the agenda in three colours, though the third colour was never "
    "explained. The chair opened by asking whether the budget shortfall was structural or "
    "temporary, and the finance officer replied that it depended on which definition of "
    "structural one preferred, a remark that was received in silence. For the next hour the "
    "discussion moved between the cost of replacing the roof, the cost of not replacing the "
    "roof, and a proposal to defer both decisions until the following year. Two members "
    "argued that deferral was itself a decision with consequences, and produced a table "
    "showing the interest that would accrue. The table was passed around, examined, and "
    "returned without comment. At noon a vote was taken on whether to continue, and the "
    "motion failed by one, so the meeting ended without resolving the question it had been "
    "convened to answer. The minutes record the outcome as postponed pending clarification.",
]


def load(model_name: str):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    return tok, model


def blocks(model):
    """Return the list of transformer blocks, across the architecture variants."""
    for path in ("h", "transformer.h", "model.layers", "layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.ModuleList):
            return list(obj)
    raise RuntimeError("cannot locate transformer blocks")


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jaccard of two 1-D index sets."""
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def random_expectation(k: int, seq: int) -> float:
    """E[Jaccard] for two independent uniform k-subsets of a seq-element set."""
    if seq <= 0:
        return 0.0
    e_inter = k * k / seq
    e_union = 2 * k - e_inter
    return e_inter / e_union if e_union else 0.0


def measure(tok, model, texts, k: int):
    blks = blocks(model)
    rows = []
    for text in texts:
        ids = tok(text, return_tensors="pt")
        seq = ids["input_ids"].shape[1]
        with torch.no_grad():
            out = model(**ids, output_attentions=True, output_hidden_states=False)
        if out.attentions is None or out.attentions[0] is None:
            raise RuntimeError("model did not return attentions; attn_implementation must be eager")
        # (heads, seq, seq) per layer
        attn = [a[0] for a in out.attentions]
        n_layers, n_heads = len(attn), attn[0].shape[0]
        top = [a.topk(min(k, seq), dim=-1).indices for a in attn]   # (heads, seq, k)

        # 1. adjacent-layer, same head
        adj = []
        for l in range(n_layers - 1):
            vals = [jaccard(top[l][h, q], top[l + 1][h, q])
                    for h in range(n_heads) for q in range(seq)]
            adj.append(sum(vals) / len(vals))

        # 2. same-layer, different heads (control)
        cross = []
        for l in range(n_layers):
            if n_heads < 2:
                cross.append(float("nan"))
                continue
            vals = []
            for h in range(n_heads):
                for h2 in range(h + 1, n_heads):
                    for q in range(seq):
                        vals.append(jaccard(top[l][h, q], top[l][h2, q]))
            cross.append(sum(vals) / len(vals) if vals else float("nan"))

        # 3. same head, same layer, DIFFERENT query position. This is the
        #    decisive control: if a query's selection is mostly determined by
        #    where the mass sits globally (BOS, punctuation, frequent tokens),
        #    then two different queries in the same head agree as much as two
        #    adjacent layers do, and "layers agree" says nothing about depth.
        shift = []
        for l in range(n_layers):
            vals = [jaccard(top[l][h, q], top[l][h, (q + 1) % seq])
                    for h in range(n_heads) for q in range(seq) if seq > 1]
            shift.append(sum(vals) / len(vals) if vals else float("nan"))

        # 4. The same three quantities after removing the globally most-selected
        #    keys. If attention is dominated by a handful of positions (beginning
        #    of sequence, punctuation, frequent words), every comparison above is
        #    measuring that domination. Stripping the top COMMON keys per layer
        #    leaves only the part of the selection that varies.
        COMMON = min(5, max(seq // 20, 1))
        filt = []
        for l in range(n_layers):
            freq = torch.zeros(seq)
            for h in range(n_heads):
                for q in range(seq):
                    freq[top[l][h, q]] += 1
            common = set(freq.topk(COMMON).indices.tolist())
            filt.append([[int(kk) for kk in top[l][h, q].tolist() if kk not in common]
                         for h in range(n_heads) for q in range(seq)])

        adj_f, cross_f, shift_f = [], [], []
        for l in range(n_layers - 1):
            a = filt[l]; b = filt[l + 1]
            vals = [jaccard(torch.tensor(a[i]), torch.tensor(b[i]))
                    for i in range(min(len(a), len(b)))]
            adj_f.append(sum(vals) / len(vals) if vals else float("nan"))
        for l in range(n_layers):
            flat = filt[l]
            if n_heads < 2 or len(flat) < n_heads * seq:
                cross_f.append(float("nan"))
                continue
            vals = [jaccard(torch.tensor(flat[h * seq + q]), torch.tensor(flat[h2 * seq + q]))
                    for h in range(n_heads) for h2 in range(h + 1, n_heads)
                    for q in range(seq)]
            cross_f.append(sum(vals) / len(vals) if vals else float("nan"))
        for l in range(n_layers):
            flat = filt[l]
            vals = [jaccard(torch.tensor(flat[h * seq + q]), torch.tensor(flat[h * seq + (q + 1) % seq]))
                    for h in range(n_heads) for q in range(seq) if seq > 1]
            shift_f.append(sum(vals) / len(vals) if vals else float("nan"))

        rows.append({
            "common_keys_removed": COMMON,
            "adjacent_filt": adj_f, "cross_head_filt": cross_f, "shifted_filt": shift_f,
            "text": text[:40], "seq": seq, "n_layers": n_layers, "n_heads": n_heads,
            "adjacent": adj, "cross_head": cross, "shifted_query": shift,
            "random_expectation": random_expectation(min(k, seq), seq),
        })
    return rows


def summarise(rows):
    n_layers = rows[0]["n_layers"]
    adj_mean = [sum(r["adjacent"][l] for r in rows) / len(rows) for l in range(n_layers - 1)]
    cross_mean = [sum(r["cross_head"][l] for r in rows) / len(rows) for l in range(n_layers)]
    shift_mean = [sum(r["shifted_query"][l] for r in rows) / len(rows) for l in range(n_layers)]
    rand = sum(r["random_expectation"] for r in rows) / len(rows)
    return {
        "adjacent_mean_by_layer": adj_mean,
        "cross_head_mean_by_layer": cross_mean,
        "shifted_query_mean_by_layer": shift_mean,
        "shifted_query_overall": sum(shift_mean) / len(shift_mean),
        "adjacent_filt": sum(sum(r["adjacent_filt"][l] for r in rows) / len(rows)
                             for l in range(n_layers - 1)) / (n_layers - 1),
        "cross_head_filt": sum(sum(r["cross_head_filt"][l] for r in rows) / len(rows)
                               for l in range(n_layers)) / n_layers,
        "shifted_filt": sum(sum(r["shifted_filt"][l] for r in rows) / len(rows)
                            for l in range(n_layers)) / n_layers,
        "common_keys_removed": rows[0]["common_keys_removed"],
        "random_expectation": rand,
        "adjacent_overall": sum(adj_mean) / len(adj_mean),
        "cross_head_overall": sum(cross_mean) / len(cross_mean),
        "n_texts": len(rows),
    }


def selftest() -> int:
    """Controls that make the measurement meaningful rather than decorative."""
    checks = []

    # 1. Jaccard arithmetic.
    a = torch.tensor([1, 2, 3, 4]); b = torch.tensor([3, 4, 5, 6])
    checks.append((f"jaccard identity (got {jaccard(a,a):.3f})", abs(jaccard(a, a) - 1.0) < 1e-9))
    checks.append((f"jaccard half-overlap (got {jaccard(a,b):.3f})", abs(jaccard(a, b) - 2/6) < 1e-9))

    # 2. Random baseline must be positive but small, and must fall as k/S rises.
    r1 = random_expectation(8, 16)
    r2 = random_expectation(8, 64)
    checks.append((f"random baseline shrinks with seq ({r1:.3f} > {r2:.3f})", r1 > r2 > 0))

    # 3. The real model must actually return attentions, and adjacent-layer
    #    overlap must exceed the random baseline. If it did not, the premise
    #    under test would be false — a publishable result, but not this one.
    tok, model = load("openai-community/gpt2")
    rows = measure(tok, model, LONG_TEXTS[:1], k=8)
    s = summarise(rows)
    checks.append((f"gpt2 returns {rows[0]['n_layers']} layers of attention",
                   rows[0]["n_layers"] == 12))
    checks.append((f"adjacent overlap {s['adjacent_overall']:.3f} exceeds random {s['random_expectation']:.3f}",
                   s["adjacent_overall"] > s["random_expectation"]))

    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    bad = [n for n, o in checks if not o]
    if bad:
        print(f"SELFTEST FAILED: {bad}")
        return 1
    print("SELFTEST PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    tok, model = load(args.model)
    rows = measure(tok, model, LONG_TEXTS, args.k)
    s = summarise(rows)

    print(f"model {args.model}, top-k={args.k}, {s['n_texts']} texts")
    print(f"  random k-subset baseline: {s['random_expectation']:.4f}")
    print(f"  adjacent-layer same-head : {s['adjacent_overall']:.4f}")
    print(f"  same-layer cross-head    : {s['cross_head_overall']:.4f}")
    print(f"  same-head shifted query  : {s['shifted_query_overall']:.4f}   <- decisive control")
    print(f"\n  after removing {s['common_keys_removed']} globally most-selected keys per layer:")
    print(f"    adjacent-layer : {s['adjacent_filt']:.4f}")
    print(f"    cross-head     : {s['cross_head_filt']:.4f}")
    print(f"    shifted query  : {s['shifted_filt']:.4f}")
    ratio = (s["adjacent_overall"] - s["random_expectation"]) / \
            max(s["cross_head_overall"] - s["random_expectation"], 1e-9)
    print(f"  (adjacent - random) / (cross-head - random) = {ratio:.3f}")
    print("\n  per layer (adjacent to next | cross-head within):")
    for l, (a, c) in enumerate(zip(s["adjacent_mean_by_layer"],
                                   s["cross_head_mean_by_layer"])):
        print(f"    layer {l:>2} -> {l+1:>2}: {a:.3f} | {c:.3f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "k": args.k, "summary": s,
                       "rows": [{kk: v for kk, v in r.items()
                                 if kk not in ("adjacent", "cross_head", "shifted_query")}
                                for r in rows]}, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
