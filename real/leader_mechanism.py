#!/usr/bin/env python3
"""Mechanism probe for the leader anomaly: is a leader's selection *useful* to a follower?

The paper establishes that borrowing the top-k *positions* chosen by layer 0 or 1
is 7-35x more expensive than borrowing a deeper layer, across four models. It does
not explain why. `leader_anomaly.py` tried selection-shape statistics (mass,
locality, sink, entropy) and found them flat; its entropy column was also NaN
(fp16 clamp_min underflow). This probe asks a causal question instead of a
descriptive one:

  For every ordered pair (source s, target t), take the exact positions s put in
  its own top-k, and measure how much attention mass layer t assigns to *those
  same positions* under t's own dense distribution. Call this deliver_mass(t|s).

  A leader whose selection transfers badly should deliver little mass to its
  followers: t is forced to look where s looked, and t's own distribution says
  those positions were not the important ones. If deliver_mass(t|0/1) is
  systematically below deliver_mass(t|2+), that is a mechanism for the anomaly
  that the shape statistics missed: not *what the selection looks like* but
  *whether the receiver wanted it*.

Controls (--selftest, no GPU):
  1. self-delivery: deliver_mass(t|t) equals the top-k mass of t (sanity).
  2. permutation invariance of the label axis: relabelling sources permutes rows.
  3. negative control: if every layer selects the same positions, all delivery
     masses are equal and the early/late ratio is exactly 1 -- the probe cannot
     manufacture an anomaly from a flat fixture.
  4. positive control: a fixture where layers 0-1 select low-mass positions for
     later layers must yield an early/late ratio > 1 and a permutation p < 0.05.

Run:
  python leader_mechanism.py --selftest
  python leader_mechanism.py --model DIR --corpus TXT --out out/mech.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STATE: dict = {"k": None, "sel": {}, "attn": {}}


def patch_collect(model):
    """Read-only: run DENSE attention at every layer, record top-k indices and
    the dense softmax row so delivery mass can be computed offline. Mirrors the
    instrumentation in reuse_real.py but never masks anything."""
    cfg = model.config
    model_type = cfg.model_type

    def make(layer_idx, mod):
        def forward(hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, cache_position=None, **kwargs):
            if model_type == "llama":
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            else:
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

            bsz, qlen, _ = hidden_states.shape
            n_head = cfg.num_attention_heads
            n_kv = cfg.num_key_value_heads
            hd = mod.head_dim
            q = mod.q_proj(hidden_states).view(bsz, qlen, n_head, hd).transpose(1, 2)
            k = mod.k_proj(hidden_states).view(bsz, qlen, n_kv, hd).transpose(1, 2)
            v = mod.v_proj(hidden_states).view(bsz, qlen, n_kv, hd).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if n_head // n_kv > 1:
                k = k[:, :, None, :, :].expand(bsz, n_kv, n_head // n_kv, qlen, hd).reshape(bsz, n_head, qlen, hd)
                v = v[:, :, None, :, :].expand(bsz, n_kv, n_head // n_kv, qlen, hd).reshape(bsz, n_head, qlen, hd)
            scores = torch.matmul(q, k.transpose(2, 3)) * mod.scaling
            neg = torch.finfo(scores.dtype).min
            causal = torch.full((qlen, qlen), neg, dtype=scores.dtype, device=scores.device)
            causal = torch.triu(causal, diagonal=1)
            if getattr(mod, "sliding_window", None) is not None:
                sw = mod.sliding_window
                dist = torch.arange(qlen, device=scores.device)
                too_far = (dist.view(-1, 1) - dist.view(1, -1)) >= sw
                causal = causal.masked_fill(too_far, neg)
            scores = scores + causal.view(1, 1, qlen, qlen)
            attn = torch.softmax(scores, dim=-1)

            kk = min(STATE["k"], qlen)
            sel = scores.topk(kk, dim=-1).indices          # (B,H,T,k)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, qlen, -1)
            out = mod.o_proj(out)

            # store only what is needed: selected indices and their own mass,
            # plus the dense mass at those indices for cross-layer delivery.
            STATE["sel"].setdefault(layer_idx, []).append(sel.float().mean().item())
            STATE.setdefault("sel_idx", {}).setdefault(layer_idx, []).append(sel)
            STATE.setdefault("attn_row", {}).setdefault(layer_idx, []).append(attn)
            return out, None
        return forward

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.forward = make(i, layer.self_attn)
    return len(model.model.layers)


def deliver_matrix(sel_by_layer, attn_by_layer, max_distance=6):
    """deliver[t][s] = mean over rows/heads of t's dense mass on s's top-k set.
    Only |t-s| <= max_distance is kept (the regime the paper measures)."""
    layers = sorted(sel_by_layer)
    out = {}
    for s in layers:
        for t in layers:
            d = t - s
            if not (1 <= d <= max_distance):
                continue
            sels = sel_by_layer[s]
            atts = attn_by_layer[t]
            n = min(len(sels), len(atts))
            vals = []
            for i in range(n):
                sel = sels[i]           # (B,H,T,k)
                att = atts[i]           # (B,H,T,T)
                keep = torch.gather(att, 3, sel.long())   # (B,H,T,k)
                vals.append(float(keep.sum(-1).mean()))
            out[(s, t)] = sum(vals) / len(vals)
    return out


def summarize(dm, layers):
    """Group deliveries by whether the source is 0/1 or deeper. Returns the
    ratio and a permutation p-value from relabelling the source column."""
    early = [v for (s, t), v in dm.items() if s <= 1]
    late = [v for (s, t), v in dm.items() if s >= 2]
    if not early or not late:
        return {"error": "insufficient pairs"}
    me, ml = st.mean(early), st.mean(late)
    ratio = me / ml if ml > 0 else float("inf")

    vals = list(dm.values())
    labels = [1 if s <= 1 else 0 for (s, t) in dm]
    ne = sum(labels)
    obs = me - ml
    rnd = random.Random(20261119)
    cnt = 0
    nperm = 10000
    for _ in range(nperm):
        rnd.shuffle(labels)
        a = [v for v, l in zip(vals, labels) if l]
        b = [v for v, l in zip(vals, labels) if not l]
        diff = st.mean(a) - st.mean(b)
        # Two-sided: the anomaly could make early delivery higher or lower.
        # (A one-sided `diff >= obs` silently returns p≈1 when early < late.)
        if abs(diff) >= abs(obs) - 1e-15:
            cnt += 1
    return {"early_mean": me, "late_mean": ml, "ratio": ratio,
            "early_n": ne, "late_n": len(late),
            "permutation_p": (cnt + 1) / (nperm + 1), "n_permutations": nperm,
            "seed": 20261119}


def selftest():
    torch.manual_seed(7)
    B, H, T, k = 1, 2, 32, 8
    # Fixture 1: flat -- every layer's dense distribution is uniform over all
    # positions, so every selection delivers exactly k/T mass. Ratio must be 1.
    flat_sel, flat_att = {}, {}
    base = torch.ones(B, H, T, T) / T
    for l in range(6):
        flat_sel[l] = [base.topk(k, dim=-1).indices]
        flat_att[l] = [base.clone()]
    dm = deliver_matrix(flat_sel, flat_att)
    r_flat = summarize(dm, list(range(6)))
    assert abs(r_flat["ratio"] - 1.0) < 1e-5, r_flat
    assert r_flat["permutation_p"] > 0.05, r_flat

    # Fixture 2: layers 0-1 put their mass on a low region that later layers do
    # not value; later layers concentrate elsewhere. Early sources should
    # deliver less mass to late targets than late sources do.
    skew_sel, skew_att = {}, {}
    lo = torch.zeros(B, H, T, T)
    lo[:, :, :, :4] = 1.0 / 4
    hi = torch.zeros(B, H, T, T)
    hi[:, :, :, 8:16] = 1.0 / 8
    for l in range(8):
        dist = lo if l <= 1 else hi
        skew_sel[l] = [dist.topk(k, dim=-1).indices]
        skew_att[l] = [dist.clone()]
    dm2 = deliver_matrix(skew_sel, skew_att)
    r_skew = summarize(dm2, list(range(8)))
    assert r_skew["ratio"] < 1.0, r_skew  # early deliver LESS
    inv = {"ratio": 1 / r_skew["ratio"], "permutation_p": r_skew["permutation_p"]}
    assert inv["ratio"] > 2.0, inv

    # Fixture 3: self-delivery equals own top-k mass for a non-uniform row.
    row = torch.softmax(torch.randn(B, H, T, T), dim=-1)
    sel = row.topk(k, dim=-1).indices
    keep = torch.gather(row, 3, sel).sum(-1).mean().item()
    assert keep > k / T, (keep, k / T)

    print("SELFTEST OK")
    print("  flat fixture ratio = %.4f (want 1.0)" % r_flat["ratio"])
    print("  skew fixture early/late ratio = %.4f (want <1, inv >2)" % r_skew["ratio"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="/root/attn-reuse/models/Qwen2.5-0.5B")
    ap.add_argument("--corpus", default="/root/attn-reuse/wt2_test.txt")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--max-distance", type=int, default=6)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="/root/attn-reuse/out/leader_mechanism.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
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
    assert bs, "corpus too short"

    STATE["k"] = a.k
    n_layers = patch_collect(model)
    with torch.no_grad():
        for b in bs:
            model(input_ids=b, labels=b)

    dm = deliver_matrix(STATE["sel_idx"], STATE["attn_row"], a.max_distance)
    summ = summarize(dm, list(range(n_layers)))

    # Shape control: a layer's own top-k mass under its own dense distribution.
    # If early layers' self-mass is also low, a low delivery could just reflect
    # diffuse early attention (a shape effect), not the receiver rejecting the
    # selection. Reporting both separates the two explanations.
    self_mass = {}
    for l in range(n_layers):
        sels = STATE["sel_idx"][l]
        atts = STATE["attn_row"][l]
        vals = [float(torch.gather(at, 3, se.long()).sum(-1).mean())
                for se, at in zip(sels, atts)]
        self_mass[l] = sum(vals) / len(vals)

    pairs = [{"source": s, "target": t, "deliver_mass": v}
             for (s, t), v in sorted(dm.items())]
    result = {"model": a.model, "seq": a.seq, "k": a.k, "n_batches": len(bs),
              "max_distance": a.max_distance, "n_layers": n_layers,
              "self_mass": self_mass, "summary": summ, "pairs": pairs}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", a.out)
    print("early_mean=%.6f late_mean=%.6f early/late=%.4f p=%.5f" %
          (summ["early_mean"], summ["late_mean"], summ["ratio"],
           summ["permutation_p"]))


if __name__ == "__main__":
    main()
