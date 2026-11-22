#!/usr/bin/env python3
"""Paired precision run: is the recomputed cost distinguishable from zero?

The sweep numbers are means over 4 batches (2,044 scored tokens). A cost of
+0.0017 nats could be noise at that size, so the headline cells are re-run over
20 disjoint batches and tested per batch (paired t on the per-batch NLL
differences plus an exact sign test), which treats the batch as the unit.

Usage: python precision.py --model DIR --dtype float16 --out out/precision.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reuse_real as R  # noqa: E402

CORPUS = "/root/attn-reuse/wt2_test.txt"


def batches(seq, n, stride_mult=7):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok(open(CORPUS).read(), return_tensors="pt").input_ids[0]
    out = []
    for i in range(n):
        c = ids[i * seq * stride_mult: i * seq * stride_mult + seq]
        if c.numel() < seq:
            break
        out.append(c.unsqueeze(0).cuda())
    return out


def pass_per_batch(model, bs, mode, k=None, group=None, force=None):
    """Mean NLL over batches, plus the per-batch NLL list."""
    model.eval()
    hs = R._hooks(model, group or 1, force=force)
    R.STATE.update({"mode": mode, "k": k, "group": group,
                    "leader_sel": {}, "collect": None})
    per = []
    with torch.no_grad():
        for ids in bs:
            R.STATE["leader_sel"] = {}
            out = model(input_ids=ids, labels=ids)
            per.append(float(out.loss))
    for h in hs:
        h.remove()
    return per


def paired_stats(a, b):
    """a - b per batch: mean, sd, paired t, exact sign test."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    pos = sum(1 for x in d if x > 0)
    # two-sided exact sign test, binomial(n, 0.5)
    k = min(pos, n - pos)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"mean": mean, "sd": sd, "t": t, "n": n,
            "positive": pos, "sign_p": min(1.0, 2 * tail)}


def main():
    global MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/attn-reuse/models/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--out", default="/root/attn-reuse/out/precision.json")
    args = ap.parse_args()
    MODEL_DIR = args.model

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map="cuda",
        attn_implementation="eager")
    R.patch_model(model)
    bs = batches(args.seq, args.n_batches)
    print(f"layers={model.config.num_hidden_layers} batches={len(bs)}", flush=True)

    cells = [("full", None, None), ("own", 256, None), ("own", 384, None),
             ("reuse", 384, 2), ("reuse", 384, 6),
             ("reuse", 256, 3), ("reuse", 256, 6),
             ("reuse", 64, 2), ("reuse", 64, 6),
             ("reuse", 32, 12)]

    out = {"model": args.model, "dtype": args.dtype, "seq": args.seq,
           "n_batches": len(bs), "per_batch": {}, "tests": {}}
    ref = None
    ownk = {}
    for mode, k, g in cells:
        name = f"{mode}_k{k}_g{g}" if g else f"{mode}_k{k}" if k else "full"
        per = pass_per_batch(model, bs, mode, k=k, group=g)
        out["per_batch"][name] = per
        m = sum(per) / len(per)
        print(f"{name:24s} nll={m:.6f} ppl={math.exp(m):.3f}", flush=True)
        if mode == "full":
            ref = per
        elif mode == "own":
            ownk[k] = per
            out["tests"][f"{name}_vs_full"] = paired_stats(per, ref)
        else:
            base = ownk.get(k)
            if base:
                out["tests"][f"{name}_vs_own"] = paired_stats(per, base)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
