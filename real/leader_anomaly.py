#!/usr/bin/env python3
"""Why may the first two layers not serve as a reuse group's leader?

The cost grid says borrowing the top-k *positions* selected by layer 0 or 1 is
~14x (0.5B) / ~24x (1.5B) more expensive than borrowing any deeper layer's.
That is a description, not a mechanism. This probe measures, per layer, the
properties of the selection itself so the anomaly can be attributed:

  mass    share of the attention distribution that the top-k positions carry.
          A layer whose top-k already covers ~all the mass loses nothing when
          someone else picks badly; a layer with a flat distribution loses a lot.
  local   share of selected positions inside the last LOCAL_W tokens.
  first   share of selected positions equal to position 0 (attention sink).
  ent     normalised entropy of the selection over the visible prefix.

If layers 0-1 are the only layers with high `local` (or low `mass`), then the
anomaly is a *selection shape* property, and the practical rule ("do not put
layer 0/1 in the leader seat") follows from it rather than being fitted to the
cost numbers.

Run: python leader_anomaly.py --model <path> --seq 512 --n-batches 4 --k 256
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOCAL_W = 8


def patch_collect(model):
    """Wrap every layer's attention so it records its own top-k selection and
    the mass that selection carries. Mirrors reuse_real.py's instrumentation but
    is read-only: it computes the DENSE output, then measures the selection."""
    from transformers.models.qwen2 import modeling_qwen2 as Q

    orig = {}
    stats = {}

    def make(layer_idx, mod):
        def forward(hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, cache_position=None, **kw):
            from transformers.models.qwen2.modeling_qwen2 import (
                apply_rotary_pos_emb, repeat_kv)
            bsz, qlen, _ = hidden_states.shape
            q = mod.q_proj(hidden_states).view(bsz, qlen, mod.config.num_attention_heads, mod.head_dim).transpose(1, 2)
            k = mod.k_proj(hidden_states).view(bsz, qlen, mod.config.num_key_value_heads, mod.head_dim).transpose(1, 2)
            v = mod.v_proj(hidden_states).view(bsz, qlen, mod.config.num_key_value_heads, mod.head_dim).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, mod.num_key_value_groups)
            v = repeat_kv(v, mod.num_key_value_groups)
            scores = torch.matmul(q, k.transpose(2, 3)) * mod.scaling   # (B,H,T,T)
            neg = torch.finfo(scores.dtype).min
            causal = torch.full((qlen, qlen), neg, dtype=scores.dtype, device=scores.device)
            causal = torch.triu(causal, diagonal=1)
            scores = scores + causal.view(1, 1, qlen, qlen)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, qlen, -1)
            out = mod.o_proj(out)

            if mod.head_dim and COLLECT["k"] is not None:
                kk = min(COLLECT["k"], qlen)
                sel = scores.topk(kk, dim=-1).indices            # (B,H,T,k)
                kept = torch.gather(attn, 3, sel)
                mass = kept.sum(-1)                              # (B,H,T)
                idx = torch.arange(qlen, device=scores.device)
                visible = (idx.view(1, 1, 1, qlen) <= idx.view(1, 1, qlen, 1))
                ent = -(attn.clamp_min(1e-12).log() * attn * visible).sum(-1)
                denom = visible.sum(-1).clamp_min(2).float().log()
                ent = ent / denom                                # normalised
                local = (sel >= (idx.view(1, 1, qlen, 1) - LOCAL_W + 1)).float().mean(-1)
                first = (sel == 0).float().mean(-1)
                COLLECT.setdefault(layer_idx, []).append({
                    "mass": float(mass.mean()),
                    "local": float(local.mean()),
                    "first": float(first.mean()),
                    "ent": float(ent.mean()),
                })
            return out, None
        return forward

    for i, layer in enumerate(model.model.layers):
        m = layer.self_attn
        orig[i] = m.forward
        m.forward = make(i, m)
    return orig, stats


COLLECT = {"k": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/attn-reuse/models/Qwen2.5-0.5B")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--corpus", default="/root/attn-reuse/wt2_test.txt")
    ap.add_argument("--out", default="/root/attn-reuse/out/leader_anomaly.json")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dt, device_map="cuda", attn_implementation="eager").eval()
    ids = tok(open(a.corpus).read(), return_tensors="pt").input_ids[0]
    bs = [ids[i * a.seq * 5:i * a.seq * 5 + a.seq].unsqueeze(0).cuda()
          for i in range(a.n_batches)]
    bs = [b for b in bs if b.shape[1] == a.seq]
    assert bs, "corpus too short"

    COLLECT["k"] = a.k
    patch_collect(model)
    with torch.no_grad():
        for b in bs:
            model(input_ids=b, labels=b)

    rows = []
    for li in range(len(COLLECT) - 1):
        recs = COLLECT[li]
        rows.append({"layer": li,
                     "mass": sum(r["mass"] for r in recs) / len(recs),
                     "local": sum(r["local"] for r in recs) / len(recs),
                     "first": sum(r["first"] for r in recs) / len(recs),
                     "ent": sum(r["ent"] for r in recs) / len(recs)})
    with open(a.out, "w") as fh:
        json.dump({"model": a.model, "seq": a.seq, "k": a.k,
                   "n_batches": len(bs), "layers": rows}, fh, indent=2)
    print(f"{'layer':>5} {'mass':>8} {'local':>8} {'first':>8} {'ent':>8}")
    for r in rows:
        print(f"{r['layer']:>5} {r['mass']:>8.4f} {r['local']:>8.4f} "
              f"{r['first']:>8.4f} {r['ent']:>8.4f}")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
