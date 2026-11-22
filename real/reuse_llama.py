#!/usr/bin/env python3
"""Cross-family variant: same measurement, Llama rotary support.

The efficiency papers (MLI, arXiv 2611.03768; CLSA, arXiv 2606.06467) assume a
layer can borrow the top-k key *routing* computed by a neighbouring layer and
save the indexer pass. They report throughput and downstream scores, not the
cost of borrowing on the same weights.

Every reading here runs the same weights over the same tokens; only routing
changes:

  full   -- dense attention (what the model was trained with)
  own    -- layer l keeps its own top-k keys (sparse but self-consistent)
  reuse  -- layer l borrows the top-k *positions* chosen by its group leader;
             q, k, v are still layer l's own, so only routing is borrowed.

`reuse - own` is therefore the price of the borrowed routing alone.

Usage:
  python reuse_real.py --model DIR --corpus TXT --out out/results.json
  python reuse_real.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

STATE: dict = {
    "mode": "full",      # 'full' | 'own' | 'reuse' | 'rand'
    "k": None,           # keys kept per query
    "group": None,       # leader of layer l is (l // group) * group
    "layer_idx": None,
    "lead_idx": None,
    "leader_sel": {},    # layer_idx -> selection that later layers may borrow
    "collect": None,     # layer_idx -> own selection, for overlap accounting
    "rand": None,        # (seed,k,layer,shape) -> random selections, built once
    "rand_seed": 7,
}


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, kvh, t, d = x.shape
    return x[:, :, None, :, :].expand(b, kvh, n_rep, t, d).reshape(b, kvh * n_rep, t, d)


def random_selection(scores, kk, seed, layer):
    """A random k-subset per (head, query), honouring causality.

    This is the null a reader needs in order to read `reuse - own` at all: if
    an arbitrary set of k keys costs what borrowing costs, then the leader's
    selection carries no information and `reuse - own` is not measuring
    "borrowing" but "using a set the layer did not choose". Positions above
    the query are excluded exactly as the causal mask excludes them, so the
    count of visible positions per row is q+1 and k <= q is required.
    """
    key = (seed, kk, layer, tuple(scores.shape))
    cache = STATE.get("rand")
    if cache is not None and key in cache:
        return cache[key]
    qlen = scores.shape[-1]
    t = torch.arange(qlen, device=scores.device)
    visible = t[None, :] <= t[:, None]                      # (q, key)
    g = torch.Generator(device=scores.device).manual_seed(seed * 100003 + layer)
    noise = torch.rand(scores.shape, generator=g, device=scores.device,
                       dtype=torch.float32).masked_fill_(~visible, -1.0)
    sel = noise.topk(kk, dim=-1).indices
    # Early queries see fewer than kk positions, so topk fills the remainder
    # with future positions (masked to -1.0, hence never in the top kk). Those
    # fillers carry ~0 weight because the causal mask already sent their score
    # to finfo.min, so they are harmless -- the same fillers the 'own' path
    # produces for early rows. The clamp below is therefore belt-and-braces:
    # it makes "no future key is selected" a literal invariant so the selftest
    # can check it directly, and it is the guard that fires if the visibility
    # mask is ever dropped from the noise.
    sel = torch.minimum(sel, t[None, None, :, None].expand_as(sel))
    if cache is not None:
        cache[key] = sel
    return sel


def attn_forward(self, hidden_states, position_embeddings, attention_mask=None,
                 past_key_values=None, cache_position=None, **kwargs):
    if self.config.model_type == "llama":
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    else:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    bsz, qlen, _ = hidden_states.shape
    n_head = self.config.num_attention_heads
    n_kv = self.config.num_key_value_heads
    head_dim = self.head_dim
    scaling = self.scaling

    q = self.q_proj(hidden_states).view(bsz, qlen, n_head, head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, qlen, n_kv, head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, qlen, n_kv, head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    n_rep = n_head // n_kv
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    scores = torch.matmul(q, k.transpose(2, 3)) * scaling          # (B,H,T,T)
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    else:
        # transformers hands us None when the backend applies causality itself
        # (sdpa/is_causal). We compute scores explicitly, so build the mask.
        causal = torch.full((qlen, qlen), torch.finfo(scores.dtype).min,
                            dtype=scores.dtype, device=scores.device)
        causal = torch.triu(causal, diagonal=1)
        if getattr(self, "sliding_window", None) is not None:
            far = torch.full_like(causal, torch.finfo(scores.dtype).min)
            far = torch.triu(far, diagonal=self.sliding_window + 1)
            causal = torch.maximum(causal, far)
        scores = scores + causal

    li = STATE["layer_idx"]
    mode, kk = STATE["mode"], STATE["k"]

    if mode != "full" and kk is not None and kk <= qlen:
        own = scores.topk(kk, dim=-1).indices                     # (B,H,T,k)
        if STATE["collect"] is not None:
            STATE["collect"][li] = own.detach()

        if mode == "reuse":
            borrowed = STATE["leader_sel"].get(STATE["lead_idx"])
            if borrowed is not None and borrowed.shape[2] == qlen:
                sel = borrowed.to(scores.device)                  # routing only
            else:                                                 # leader itself
                sel = own
                STATE["leader_sel"][li] = own.detach()
        elif mode == "rand":                                      # null: arbitrary set
            # With a group, this is the fair null for a reuse cell: the leader
            # keeps the selection it would have made, and only the borrowing
            # layers get an arbitrary set of the same size. Comparing a
            # whole-model random pass against a grouped reuse cell would
            # overstate the null -- the cell only perturbs the followers.
            if STATE.get("group") and STATE["group"] > 1 and li == STATE.get("lead_idx"):
                sel = own
                STATE["leader_sel"][li] = own.detach()
            else:
                sel = random_selection(scores, kk, STATE["rand_seed"], li)
        else:                                                     # 'own'
            sel = own
            STATE["leader_sel"][li] = own.detach()

        # scores are (B, H, query, key): the selected weights belong on the KEY
        # axis, so the mask uses dim 3. Two earlier versions used dim 2 and were
        # silently wrong -- indexing along the query axis returns a tensor of
        # the same shape holding different numbers, so nothing crashed; k ==
        # qlen then looked like it cost ~10 nats, which would have been
        # reported as a model result. The k == qlen control caught it.
        #
        # masked_fill (not add -large) is what makes the control EXACT: with
        # k == qlen every position is kept, so the mask is a no-op and this
        # branch is bit-identical to the dense one. Adding a large negative to
        # positions that already hold fp16's most-negative value overflowed to
        # -inf, and -inf + -inf in the causal sum produced NaN. Here an already
        # masked position merely stays at its own value and exp() sends it to 0.
        keep = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
        keep.scatter_(3, sel, True)
        neg = torch.finfo(scores.dtype).min
        attn = torch.softmax(scores.masked_fill(~keep, neg), dim=-1)
        out = torch.matmul(attn, v)
    else:
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

    out = out.transpose(1, 2).contiguous().view(bsz, qlen, n_head * head_dim)
    return self.o_proj(out), None


def patch_model(model):
    for layer in model.model.layers:
        a = layer.self_attn
        a.forward = attn_forward.__get__(a, type(a))
    return model


def _hooks(model, group, force=None):
    """Pre-hooks keep STATE's layer/leader counters correct for each layer."""
    hs = []
    for i, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, _i=i):
            STATE["layer_idx"] = _i
            if force and _i in force:
                STATE["lead_idx"] = force[_i]
            else:
                STATE["lead_idx"] = (_i // group) * group if group else _i
        hs.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    return hs


def run_pass(model, batches, mode, k=None, group=None, force=None):
    """Mean token NLL plus per-layer own selections (for overlap accounting)."""
    model.eval()
    hs = _hooks(model, group or 1, force=force)
    STATE.update({"mode": mode, "k": k, "group": group or 1,
                  "leader_sel": {}, "collect": {}, "rand": {}})
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for ids in batches:
            STATE["leader_sel"] = {}
            STATE["collect"] = {}
            out = model(input_ids=ids, labels=ids)
            total_nll += float(out.loss) * (ids.shape[1] - 1)
            total_tok += ids.shape[1] - 1
    for h in hs:
        h.remove()
    return total_nll / max(total_tok, 1), STATE["collect"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/attn-reuse/models/Qwen2.5-0.5B")
    ap.add_argument("--corpus", default="/root/attn-reuse/wt2_test.txt")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--ks", default="64,128,256")
    ap.add_argument("--groups", default="2,3,6")
    ap.add_argument("--out", default="/root/attn-reuse/out/results.json")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--rand-control", action="store_true",
                    help="also run the random-selection null at each k")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map="cuda",
        attn_implementation="eager")
    patch_model(model)
    n_layers = model.config.num_hidden_layers

    ids_all = tok(open(args.corpus).read(), return_tensors="pt").input_ids[0]
    stride = args.seq * 7
    batches = []
    for i in range(args.n_batches):
        chunk = ids_all[i * stride: i * stride + args.seq]
        if chunk.numel() < args.seq:
            break
        batches.append(chunk.unsqueeze(0).cuda())
    print(f"model_layers={n_layers} batches={len(batches)} seq={args.seq}", flush=True)

    res = {"model": args.model, "dtype": args.dtype,
           "n_layers": n_layers, "seq": args.seq,
           "n_batches": len(batches), "readings": {}, "deltas": {}}

    base, _ = run_pass(model, batches, "full")
    res["readings"]["full"] = base
    print(f"full           nll={base:.5f} ppl={math.exp(base):.2f}", flush=True)

    # Control: with k == qlen the sparse path must reproduce the dense loss
    # exactly. If it does not, the harness (not the model) is producing the
    # deltas and nothing below is reportable.
    ctrl, _ = run_pass(model, batches, "own", k=args.seq)
    if abs(ctrl - base) > 1e-7:
        print(f"CONTROL FAILED: own(k=qlen)={ctrl:.8f} vs full={base:.8f} "
              f"delta={ctrl - base:+.3e}; the harness is adding error", flush=True)
        return 2
    res["readings"]["control_own_kqlen"] = ctrl
    print(f"control own(k=qlen)={ctrl:.5f} == full (|d|<1e-6)  OK", flush=True)

    for k in [int(x) for x in args.ks.split(",")]:
        own, _ = run_pass(model, batches, "own", k=k)
        res["readings"][f"own_k{k}"] = own
        print(f"own   k={k:3d}    nll={own:.5f} d_vs_full={own - base:+.5f}", flush=True)

        for g in [int(x) for x in args.groups.split(",")]:
            rr, _ = run_pass(model, batches, "reuse", k=k, group=g)
            res["readings"][f"reuse_k{k}_g{g}"] = rr
            d = {"vs_own": rr - own, "vs_full": rr - base}
            res["deltas"][f"k{k}_g{g}"] = d
            print(f"reuse k={k:3d} g={g} nll={rr:.5f} d_vs_own={d['vs_own']:+.5f} "
                  f"d_vs_full={d['vs_full']:+.5f}", flush=True)

            # The null for THIS cell: same group, same k, but every borrowing
            # layer gets an arbitrary causal-valid set of the same size instead
            # of the leader's set. Leaders keep their own selection, so the
            # null perturbs exactly the layers the cell perturbs. Reading
            # `reuse - own` without it cannot separate "the price of borrowing
            # a neighbour" from "the price of not choosing your own set".
            if args.rand_control:
                rn, _ = run_pass(model, batches, "rand", k=k, group=g)
                d["rand_vs_own"] = rn - own
                d["rand_vs_full"] = rn - base
                res["readings"][f"rand_k{k}_g{g}"] = rn
                print(f"null  k={k:3d} g={g} nll={rn:.5f} "
                      f"d_vs_own={rn - own:+.5f} ratio={d['vs_own'] / (rn - own):.4f}",
                      flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    return 0


def selftest():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).eval()
    ids = torch.randint(0, 128, (1, 32))

    with torch.no_grad():
        ref = m(input_ids=ids).logits

    patch_model(m)

    hs = _hooks(m, 1)
    STATE.update({"mode": "full", "k": None, "leader_sel": {}, "collect": None})
    with torch.no_grad():
        got = m(input_ids=ids).logits
    for h in hs:
        h.remove()
    d = (got - ref).abs().max().item()
    print(f"  full  max|diff|={d:.3e}")
    assert d < 1e-3, f"patched dense path must reproduce the original ({d})"

    hs = _hooks(m, 1)
    STATE.update({"mode": "own", "k": 8, "leader_sel": {}, "collect": None})
    with torch.no_grad():
        sp = m(input_ids=ids).logits
    for h in hs:
        h.remove()
    d2 = (sp - ref).abs().max().item()
    print(f"  own   max|diff|={d2:.3e}")
    assert d2 > 1e-6, "top-k selection must change the output"

    hs = _hooks(m, 2)
    STATE.update({"mode": "reuse", "k": 8, "leader_sel": {}, "collect": None})
    with torch.no_grad():
        rr = m(input_ids=ids).logits
    for h in hs:
        h.remove()
    d3 = (rr - sp).abs().max().item()
    print(f"  reuse max|diff vs own|={d3:.3e}")
    assert d3 > 1e-6, "borrowing a leader's routing must change the output"

    # The random-selection null must (a) change the output, (b) never select a
    # position beyond the query, and (c) be deterministic for a fixed seed.
    # (b) is the one that matters: a random set that reaches into the future
    # would be a *better* set than anything a layer can choose, and any cost we
    # attribute to "an arbitrary set" would be systematically understated.
    hs = _hooks(m, 1)
    STATE.update({"mode": "rand", "k": 8, "leader_sel": {}, "collect": None,
                  "rand": {}, "rand_seed": 7})
    with torch.no_grad():
        rn = m(input_ids=ids).logits
    for h in hs:
        h.remove()
    d4 = (rn - sp).abs().max().item()
    print(f"  rand  max|diff vs own|={d4:.3e}")
    assert d4 > 1e-6, "a random set must change the output"

    with torch.no_grad():
        probe = torch.randn(2, 3, 6, 6)
    sel = random_selection(probe, 4, seed=7, layer=0)
    q = torch.arange(6)[:, None]
    bad = int((sel > q[None, None]).sum().item())
    print(f"  rand  causal violations={bad}")
    assert bad == 0, f"a random selection reached past the query ({bad} rows)"

    STATE["rand"] = {}
    a1 = random_selection(probe, 4, seed=7, layer=0)
    a2 = random_selection(probe, 4, seed=7, layer=0)
    assert torch.equal(a1, a2), "the random null must be reproducible for a fixed seed"
    a3 = random_selection(probe, 4, seed=8, layer=0)
    assert not torch.equal(a1, a3), "a different seed must give a different null"
    print("  rand  seed-determinism and seed-sensitivity OK")
    print("selftest OK")


if __name__ == "__main__":
    sys.exit(main())
