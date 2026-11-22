#!/usr/bin/env python3
"""Second-stage sweep: does the price of borrowed routing depend on distance?

Stage 1 (reuse_real.py) borrows a whole group's routing from its leader and
shows the cost rising with group size at every k -- except that group=2 is
*dearer* than group=3, which is backwards. Two candidate explanations:

  H1 distance   cost grows with how far the borrow reaches (leader -> member)
  H2 identity   cost is driven by which layer is the leader, not by the span;
                with group=2 layer 1 always borrows layer 0's routing, and the
                first decoder layer's attention is not interchangeable with the
                rest (the earlier output-reuse run already saw 0->1 as the one
                expensive single pair).

This script separates them:

  --pairs    every single (source j, target l) pair within a window, all other
             layers left at their own routing. Gives the distance curve AND the
             per-pair map, so H1 and H2 can be told apart.
  --offset   group-2 partitioning shifted by one, so layer 0 stops serving as
             everybody's leader.

Writes one JSON record per reading to --out (jsonl) so a partial run is still
usable, and a summary object at the end.

Usage:
  python sweep2.py --mode pairs  --window 4 --out out/pairs.jsonl
  python sweep2.py --mode offset --out out/offset.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reuse_llama as R  # noqa: E402

MODEL = "/root/attn-reuse/models/Qwen2.5-0.5B"
CORPUS = "/root/attn-reuse/wt2_test.txt"


def load(seq=512, n_batches=4, model_dir=None, corpus=None, dtype="float16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdl = model_dir or MODEL
    tok = AutoTokenizer.from_pretrained(mdl)
    model = AutoModelForCausalLM.from_pretrained(
        mdl, dtype=getattr(torch, dtype), device_map="cuda", attn_implementation="eager")
    R.patch_model(model)
    ids = tok(open(corpus or CORPUS).read(), return_tensors="pt").input_ids[0]
    stride = seq * 7
    batches = [ids[i * stride:i * stride + seq].unsqueeze(0).cuda()
               for i in range(n_batches)]
    batches = [b for b in batches if b.shape[1] == seq]
    return model, batches


def read_pair(model, batches, k, target, source, base_own):
    """Layer `target` borrows `source`'s routing; every other layer is its own."""
    force = {target: source}
    hs = R._hooks(model, group=1, force=force)
    R.STATE.update({"mode": "reuse", "k": k, "group": 1,
                    "leader_sel": {}, "collect": None})
    nll, _ = R.run_pass(model, batches, "reuse", k=k, group=1, force=force)
    for h in hs:
        h.remove()
    return nll, nll - base_own


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pairs", "offset"], default="pairs")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--out", default="/root/attn-reuse/out/sweep2.jsonl")
    ap.add_argument("--offset", type=int, default=1)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    model, batches = load(args.seq, args.n_batches, args.model, args.corpus, args.dtype)
    n_layers = model.config.num_hidden_layers
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fh = open(args.out, "w")

    def emit(rec):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    base, _ = R.run_pass(model, batches, "full")
    emit({"kind": "base", "full": base, "ppl": math.exp(base)})
    print(f"full nll={base:.6f}", flush=True)

    k = args.k
    own, _ = R.run_pass(model, batches, "own", k=k)
    emit({"kind": "own", "k": k, "nll": own, "vs_full": own - base})
    print(f"own k={k} nll={own:.6f} vs_full={own - base:+.6f}", flush=True)

    ctrl, _ = R.run_pass(model, batches, "own", k=args.seq)
    emit({"kind": "control", "k": args.seq, "nll": ctrl, "vs_full": ctrl - base})
    print(f"control own(k=qlen) vs_full={ctrl - base:+.3e}", flush=True)

    if args.mode == "pairs":
        for tgt in range(1, n_layers):
            for d in range(1, args.window + 1):
                src = tgt - d
                if src < 0:
                    continue
                nll, delta = read_pair(model, batches, k, tgt, src, own)
                emit({"kind": "pair", "k": k, "target": tgt, "source": src,
                      "distance": d, "nll": nll, "vs_own": delta})
                print(f"pair l={tgt:2d}<-{src:2d} d={d} vs_own={delta:+.6f}", flush=True)
    else:
        for g in (2, 3, 4, 6):
            force = {}
            for i in range(n_layers):
                lead = ((i - args.offset) // g) * g + args.offset
                if lead < 0:
                    lead = 0
                force[i] = min(lead, i)
            hs = R._hooks(model, 1, force=force)
            R.STATE.update({"mode": "reuse", "k": k, "group": g,
                            "leader_sel": {}, "collect": None})
            nll, _ = R.run_pass(model, batches, "reuse", k=k, group=g, force=force)
            for h in hs:
                h.remove()
            emit({"kind": "offset", "k": k, "group": g, "offset": args.offset,
                  "nll": nll, "vs_own": nll - own})
            print(f"offset g={g} off={args.offset} vs_own={nll - own:+.6f}", flush=True)

    fh.close()
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
