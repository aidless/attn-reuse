#!/usr/bin/env python3
"""Does the *starved attention mass* explain the reuse cost? -- a mechanistic probe.

For a target layer t that borrows the top-k positions selected by layer s, the
damage should be the attention mass that t puts on positions s did not select:

    starved(t,s) = 1 - sum_{i in sel_s} attn_t[i]

averaged over (batch, head, query). If reuse cost is caused by this, then
cost should be an increasing function of starved, and the layer-3 anomaly should
be visible as a large starved value at (3,2).

Read-only instrumentation: one dense forward pass per batch records, per layer,
its own top-k selection AND its full attention row, so every (t,s) pair is
computed from the SAME pass (no re-running, no cross-pass drift).

Run: python margin_probe.py --seq 512 --k 256 --n-batches 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REC = {"k": None, "sel": {}, "w": {}}


def patch_record(model):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    for i, layer in enumerate(model.model.layers):
        mod = layer.self_attn

        def make(idx, m):
            def forward(hidden_states, position_embeddings, attention_mask,
                        past_key_values=None, cache_position=None, **kw):
                bsz, qlen, _ = hidden_states.shape
                cfg = m.config
                q = m.q_proj(hidden_states).view(bsz, qlen, cfg.num_attention_heads, m.head_dim).transpose(1, 2)
                k = m.k_proj(hidden_states).view(bsz, qlen, cfg.num_key_value_heads, m.head_dim).transpose(1, 2)
                v = m.v_proj(hidden_states).view(bsz, qlen, cfg.num_key_value_heads, m.head_dim).transpose(1, 2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = repeat_kv(k, m.num_key_value_groups)
                v = repeat_kv(v, m.num_key_value_groups)
                scores = torch.matmul(q, k.transpose(2, 3)) * m.scaling
                neg = torch.finfo(scores.dtype).min
                causal = torch.triu(torch.full((qlen, qlen), neg,
                                               dtype=scores.dtype, device=scores.device), 1)
                scores = scores + causal.view(1, 1, qlen, qlen)
                attn = torch.softmax(scores, dim=-1)
                out = m.o_proj(torch.matmul(attn, v).transpose(1, 2).reshape(bsz, qlen, -1))

                if REC["k"] is not None:
                    kk = min(REC["k"], qlen)
                    sel = scores.topk(kk, dim=-1).indices               # (B,H,T,kk)
                    REC["sel"][idx] = sel.detach()
                    REC["w"][idx] = attn.detach()
                return out, None
            return forward
        mod.forward = make(i, mod)


def starved_pairs(sel, w, layers, max_distance=12):
    """(t,s) -> mean over (b,h,q) of 1 - mass(attn_t at sel_s).

    Bounded to |t-s| <= max_distance because each pair costs one (B,H,T,T)
    masked sum; the full 24x23 grid is 552 of them per batch."""
    out = {}
    for t in range(layers):
        W = w[t]                            # (B,H,T,T)
        for s in range(max(0, t - max_distance), min(layers, t + max_distance + 1)):
            if t == s:
                continue
            A = sel[s]                      # (B,H,T,kk)
            mk = torch.zeros_like(W, dtype=torch.bool).scatter_(3, A, True)
            out[(t, s)] = float(1.0 - (W * mk).sum(-1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/attn-reuse/models/Qwen2.5-0.5B")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--corpus", default="/root/attn-reuse/wt2_test.txt")
    ap.add_argument("--out", default="/root/attn-reuse/out/margin.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        # starved() must be exact on a hand-built case: uniform attention over 4
        # keys, sel_s = {0,1} -> starved = 0.5
        W = torch.full((1, 1, 1, 4), 0.25)
        S = torch.tensor([[[[0, 1]]]])
        mk = torch.zeros_like(W, dtype=torch.bool).scatter_(3, S, True)
        assert abs(float(1 - (W * mk).sum(-1).mean()) - 0.5) < 1e-6
        # kk == T must give starved == 0 for every pair
        W = torch.rand(1, 1, 3, 3); W = W / W.sum(-1, keepdim=True)
        S = torch.arange(3).view(1, 1, 1, 3).expand(1, 1, 3, 3)
        mk = torch.zeros_like(W, dtype=torch.bool).scatter_(3, S, True)
        assert float((W * mk).sum(-1).mean()) > 0.999999
        print("selftest OK (starved is exact at k=2-of-4 and zero at k=T)")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dt, device_map="cuda", attn_implementation="eager").eval()
    ids = tok(open(a.corpus).read(), return_tensors="pt").input_ids[0]
    bs = [ids[i * a.seq * 5:i * a.seq * 5 + a.seq].unsqueeze(0).cuda()
          for i in range(a.n_batches)]
    bs = [b for b in bs if b.shape[1] == a.seq]
    assert bs, "corpus too short for the requested seq"

    REC["k"] = a.k
    patch_record(model)
    acc = {}
    with torch.no_grad():
        for b in bs:
            model(input_ids=b, labels=b)
            for pair, v in starved_pairs(REC["sel"], REC["w"],
                                         len(REC["sel"])).items():
                acc.setdefault(pair, []).append(v)

    rows = [{"target": t, "source": s, "distance": t - s,
             "starved": sum(v) / len(v)}
            for (t, s), v in sorted(acc.items())]
    with open(a.out, "w") as fh:
        json.dump({"model": a.model, "seq": a.seq, "k": a.k,
                   "n_batches": len(bs), "pairs": rows}, fh, indent=2)
    adj = [r for r in rows if r["distance"] == 1]
    print(f"{'t<-s':>6} {'starved':>8}")
    for r in adj[:12]:
        print(f"{r['target']:>3}<-{r['source']:<3} {r['starved']:>8.4f}")
    print(f"WROTE {a.out}  ({len(rows)} pairs, {len(bs)} batches, k={a.k}, seq={a.seq})")


if __name__ == "__main__":
    main()
