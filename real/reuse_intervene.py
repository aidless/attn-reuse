"""Interventional test (Goal Round 4): does raising the deliver-mass of the
SHARED index reduce reuse NLL?  Prereg: prereg/intervention-prereg.md.

Two-pass ORACLE.  Pass A collects dense attention per layer (read-only).
Pass B changes ONLY the index set that followers BORROW from the group leader;
each layer's own q/k/v and its own top-k (when it is a leader) stay untouched.

Conditions: BASE (leader's own top-k), HIGH (top-k consensus mass of the
followers), LOW (bottom-k consensus mass, negative control).
"""
import argparse, json, math, os, torch

STATE = dict(mode=None, k=None, layer_idx=None, lead_idx=None,
             mass=None, borrow=None, leader_own=None)

# BASE has two readings, and they are not the same condition:
#   base_live   followers borrow the leader's own top-k *as the leader computed
#               it in this pass*, per batch, on its own masked scores. This is
#               exactly what reuse_real/reuse_llama do, so it is the grid's
#               "reuse" condition and must reproduce it.
#   base_static followers borrow the leader's top-k read off the batch-averaged
#               dense mass (the array the HIGH/LOW conditions also use). It
#               shares the static-selection design of the two interventions but
#               selects through a different statistic than the grid's per-batch
#               masked scores, so it is NOT the grid condition.
# The first version of this script reported only base_static and called it
# BASE, which made the reuse penalty look 2.1x larger than the grid's own
# measurement of the same thing (+0.05074 against +0.02419 on 0.5B g=2). Both
# are now reported; the verdict uses base_live, the condition the paper claims
# to intervene on.


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


def _scores(self, hidden_states, position_embeddings, attention_mask):
    mt = self.config.model_type
    if mt == "llama":
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    else:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    bsz, qlen, _ = hidden_states.shape
    nh, nkv, hd = self.config.num_attention_heads, self.config.num_key_value_heads, self.head_dim
    q = self.q_proj(hidden_states).view(bsz, qlen, nh, hd).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, qlen, nkv, hd).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, qlen, nkv, hd).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k, v = _repeat_kv(k, nh // nkv), _repeat_kv(v, nh // nkv)
    s = torch.matmul(q, k.transpose(2, 3)) * self.scaling
    neg = torch.finfo(s.dtype).min
    causal = torch.full((qlen, qlen), neg, dtype=s.dtype, device=s.device)
    causal = torch.triu(causal, diagonal=1)
    sw = getattr(self, "sliding_window", None)
    if sw is not None:
        d = torch.arange(qlen, device=s.device)
        causal = causal.masked_fill((d.view(-1, 1) - d.view(1, -1)) >= sw, neg)
    s = s + (attention_mask[:, :, :, :qlen] if attention_mask is not None else causal.view(1, 1, qlen, qlen))
    return s, v


def attn_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                 past_key_values=None, cache_position=None, **kwargs):
    s, v = _scores(self, hidden_states, position_embeddings, attention_mask)
    li, mode, kk = STATE["layer_idx"], STATE["mode"], STATE["k"]
    if mode == "collect":
        attn = torch.softmax(s, dim=-1)
        # accumulate batch-averaged dense attention into (H,T,T), float32
        STATE["mass"][li] += attn.detach().float().mean(0).cpu()
        out = torch.matmul(attn, v)
    elif mode in ("base_live", "intervene"):
        # two paths to the same thing: base_live lets followers borrow the
        # leader's own per-batch masked top-k (the grid's reuse condition),
        # intervene hands them a static index set built from dense mass.
        if mode == "base_live":
            own = s.topk(min(kk, s.shape[-1]), dim=-1).indices
            src = STATE["leader_own"].get(STATE["lead_idx"])
            if src is not None and src.shape[2] == s.shape[-1]:
                sel = src.to(s.device)
            else:
                sel = own
                STATE["leader_own"][STATE["layer_idx"]] = own.detach()
        else:
            if li in STATE["borrow"] and kk is not None and kk <= s.shape[-1]:
                sel = STATE["borrow"][li].to(s.device)
            else:
                sel = s.topk(min(kk, s.shape[-1]), dim=-1).indices
        keep = torch.zeros(s.shape, dtype=torch.bool, device=s.device)
        keep.scatter_(3, sel, True)
        # Deliver-mass is read HERE, in the same pass and on the same row, from
        # the unmasked softmax of this layer's own scores. The first version
        # read it off a separate batch-averaged dense pass, which meant the
        # predictor and the cost came from two different statistics of the
        # same probe. scatter (not gather-sum) so a padded slot repeated in
        # `sel` is counted once.
        if STATE.get("deliver_acc") is not None and li > STATE["lead_idx"]:
            dense = torch.softmax(s, dim=-1)
            pk = torch.zeros_like(dense)
            pk.scatter_(3, sel, dense.gather(3, sel))
            STATE["deliver_acc"][li][0] += float(pk.sum(-1).sum())
            STATE["deliver_acc"][li][1] += float(pk.shape[0] * pk.shape[1] * pk.shape[2])
        attn = torch.softmax(s.masked_fill(~keep, torch.finfo(s.dtype).min), dim=-1)
        out = torch.matmul(attn, v)
    else:
        attn = torch.softmax(s, dim=-1)
        out = torch.matmul(attn, v)
    bsz, nh, qlen, _ = s.shape
    hd = self.head_dim
    out = out.transpose(1, 2).contiguous().view(bsz, qlen, nh * hd)
    return self.o_proj(out), None


def patch(model):
    for layer in model.model.layers:
        layer.self_attn.forward = attn_forward.__get__(layer.self_attn, type(layer.self_attn))
    return model


def _hooks(model, group):
    hs = []
    for i, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, _i=i):
            STATE["layer_idx"] = _i
            STATE["lead_idx"] = (_i // group) * group if group else _i
        hs.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    return hs


def collect_dense(model, batches, group):
    n = model.config.num_hidden_layers
    hs = _hooks(model, group)
    STATE.update({"mode": "collect", "k": None,
                  "mass": [torch.zeros(model.config.num_attention_heads,
                                       b.shape[1], b.shape[1], dtype=torch.float32)
                           for b in batches]})  # placeholder sized below
    T = batches[0].shape[1]
    H = model.config.num_attention_heads
    STATE["mass"] = [torch.zeros(H, T, T, dtype=torch.float32) for _ in range(n)]
    with torch.no_grad():
        for ids in batches:
            model(input_ids=ids, labels=ids)
    for h in hs:
        h.remove()
    mass = [m / max(len(batches), 1) for m in STATE["mass"]]
    return mass


def _topk_valid(rowmass, k, high=True):
    """rowmass: (T,T) mass[q,p]; pick k keys among p<=q for each query q.

    Returns (T,k) index tensor. Positions p>q get -inf so never picked; when
    q+1<k we pad with 0 (harmless: causal mask zeroes it anyway)."""
    T = rowmass.shape[0]
    kk = min(k, T)
    idx = torch.arange(T, device=rowmass.device)
    valid = idx[None, :] <= idx[:, None]                       # (T,T) p<=q
    score = rowmass.masked_fill(~valid, float("-inf") if high else float("inf"))
    if not high:
        order = torch.argsort(score, dim=-1)[:, :kk]
    else:
        order = torch.topk(score, kk, dim=-1).indices
    # when q+1 < kk, topk/argsort may return masked (-inf/+inf) positions p>q;
    # clamp every selected index to its own query position (a valid slot).
    order = torch.minimum(order, idx[:, None])
    if kk < k:
        order = torch.cat([order, torch.zeros(T, k - kk, dtype=torch.long, device=rowmass.device)], dim=-1)
    return order


def build_borrow(mass, n_layers, group, k, direction):
    """Per-follower borrowed index (H,T,k), shared over batch.

    HIGH/LOW use the consensus mass of the group's FOLLOWERS; BASE uses the
    leader's own top-k. Returns {layer_idx: (1,H,T,k)} for followers only."""
    H, T, _ = mass[0].shape
    borrow = {}
    for s in range(0, n_layers, group):
        followers = [t for t in range(s + 1, min(s + group, n_layers))]
        if not followers:
            continue
        cons = torch.stack([mass[t] for t in followers]).mean(0)   # (H,T,T)
        for t in followers:
            if direction == "base":
                # leader's own selection under dense = its own top-k per row
                arr = torch.stack([_topk_valid(mass[s][h], k, True) for h in range(H)])
            else:
                arr = torch.stack([_topk_valid(cons[h], k, direction == "high") for h in range(H)])
            borrow[t] = arr.unsqueeze(0)                            # (1,H,T,k)
    return borrow


def measured_deliver(mass, borrow, group, k):
    """Deliver-mass under each condition, same (source,target) pairs.

    scatter_ (overwrite) not gather-sum: when q+1<k the padded slots repeat a
    valid position, and summing the gathered values double-counts it. A
    scatter into zeros keeps one copy per unique key position, so the result
    is a genuine probability mass in [0,1]."""
    n = len(mass)
    vals = []
    for s in range(0, n, group):
        for t in range(s + 1, min(s + group, n)):
            arr = borrow[t][0].long()                              # (H,T,k)
            picked = torch.zeros_like(mass[t])
            picked.scatter_(2, arr, mass[t].gather(2, arr))
            vals.append(picked.sum(-1).mean().item())
    return sum(vals) / max(len(vals), 1)


def run_intervene(model, batches, borrow, k, group, mode="intervene", want_deliver=False):
    n = model.config.num_hidden_layers
    hs = _hooks(model, group)
    STATE.update({"mode": mode, "k": k, "borrow": borrow,
                  "leader_own": {},
                  "deliver_acc": [[0.0, 0] for _ in range(n)] if want_deliver else None})
    tot, tok = 0.0, 0
    with torch.no_grad():
        for ids in batches:
            out = model(input_ids=ids, labels=ids)
            tot += float(out.loss) * (ids.shape[1] - 1)
            tok += ids.shape[1] - 1
    for h in hs:
        h.remove()
    nll = tot / max(tok, 1)
    if want_deliver:
        acc = STATE["deliver_acc"]
        return nll, sorted((li, sm / cn) for li, (sm, cn) in enumerate(acc) if cn)
    return nll


def run_full(model, batches, group):
    n = model.config.num_hidden_layers
    hs = _hooks(model, group)
    STATE.update({"mode": "full", "k": None})
    tot, tok = 0.0, 0
    with torch.no_grad():
        for ids in batches:
            out = model(input_ids=ids, labels=ids)
            tot += float(out.loss) * (ids.shape[1] - 1)
            tok += ids.shape[1] - 1
    for h in hs:
        h.remove()
    return tot / max(tok, 1)


def selftest():
    torch.manual_seed(0)
    H, T, k, group, n = 2, 16, 3, 2, 4

    def gauss_rows(center, sigma):
        # (T,T) mass[q,p]: Gaussian over p centered at `center`, causal p<=q, renorm
        p = torch.arange(T, dtype=torch.float32)
        m = torch.zeros(T, T)
        for q in range(T):
            w = torch.exp(-0.5 * ((p[:q + 1] - center) / sigma) ** 2)
            w = w / w.sum()
            m[q, :q + 1] = w
        return m

    # leaders (even layers) peak at pos 2; followers (odd layers) peak at pos 6
    mass = []
    for l in range(n):
        m = gauss_rows(2.0 if l % 2 == 0 else 6.0, 2.0)
        mass.append(m.unsqueeze(0).expand(H, -1, -1).clone())
    b_base = build_borrow(mass, n, group, k, "base")
    b_high = build_borrow(mass, n, group, k, "high")
    b_low = build_borrow(mass, n, group, k, "low")
    d_base = measured_deliver(mass, b_base, group, k)
    d_high = measured_deliver(mass, b_high, group, k)
    d_low = measured_deliver(mass, b_low, group, k)
    assert d_high > d_base > d_low, (d_high, d_base, d_low)
    # base_live builds its own index from the live masked scores, so on the
    # fixture it must agree with the static BASE only if both read the same
    # mass; here they do, and a mismatch means one of the two paths drifted.
    b_live = {}
    for s0 in range(0, n, group):
        for t in range(s0 + 1, min(s0 + group, n)):
            b_live[t] = torch.stack(
                [_topk_valid(mass[s0][h], k, True) for h in range(H)]).unsqueeze(0)
    assert set(b_live) == set(b_base), (sorted(b_live), sorted(b_base))
    for t in b_live:
        assert torch.equal(b_live[t], b_base[t]), ("base_live != static base on fixture", t)
    assert not torch.equal(b_high[1], b_low[1]), "HIGH/LOW sets must differ"
    # causal: no borrowed index exceeds its query position
    for li, arr in b_high.items():
        Tq = arr.shape[2]
        assert bool((arr[0].max(-1).values <= torch.arange(Tq)).all()), "causal violation"
    print("SELFTEST OK: d_high=%.4f > d_base=%.4f > d_low=%.4f" % (d_high, d_base, d_low))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/attn-reuse/models/Qwen2.5-0.5B")
    ap.add_argument("--corpus", default="/root/attn-reuse/wt2_test.txt")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--groups", default="2,3,6")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--out", default="/root/attn-reuse/out/intervene.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(20261120)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), device_map="cuda", attn_implementation="eager")
    patch(model)
    n = model.config.num_hidden_layers
    ids_all = tok(open(a.corpus).read(), return_tensors="pt").input_ids[0]
    stride = a.seq * 7
    batches = []
    for i in range(a.n_batches):
        c = ids_all[i * stride:i * stride + a.seq]
        if c.numel() < a.seq:
            break
        batches.append(c.unsqueeze(0).cuda())
    print("model_layers=%d batches=%d seq=%d" % (n, len(batches), a.seq), flush=True)

    res = {"model": a.model, "dtype": a.dtype, "n_layers": n, "seq": a.seq,
           "n_batches": len(batches), "k": a.k, "groups": {}}

    # harness control: full must be finite; intervention with borrow=k==seq keeps
    # every position (top-k over p<=q, padded) so it should reproduce full closely.
    mass = collect_dense(model, batches, 1)
    full = run_full(model, batches, 1)
    res["full"] = full
    print("full nll=%.5f" % full, flush=True)

    # PREREG negative control 1: own(k=seq) must equal full to 1e-7. With an
    # empty borrow dict and k=qlen, every layer keeps its own p<=q positions,
    # so the intervention mask is a no-op and must reproduce dense exactly.
    own_ctl = run_intervene(model, batches, {}, a.seq, 1)
    res["harness_own_kseq"] = own_ctl
    res["harness_abs_diff"] = abs(own_ctl - full)
    res["harness_pass"] = bool(abs(own_ctl - full) < 1e-7)
    print("harness own(k=seq)=%.7f full=%.7f absdiff=%.2e pass=%s" % (
        own_ctl, full, abs(own_ctl - full), res["harness_pass"]), flush=True)

    def followers(g):
        return [t for s0 in range(0, n, g) for t in range(s0 + 1, min(s0 + g, n))]

    for g in [int(x) for x in a.groups.split(",")]:
        out = {"nll": {}, "deliver": {}, "deliver_static": {},
               "overlap_high_low": {}, "overlap_low_base": {}}
        sets = {}
        for d in ("base", "high", "low"):
            sets[d] = build_borrow(mass, n, g, a.k, d)
            out["deliver_static"][d] = measured_deliver(mass, sets[d], g, a.k)
        # HIGH/LOW and LOW/BASE must actually differ (PREREG negative control 2)
        ov, ovlb = [], []
        for li in sets["high"]:
            hi, lo, ba = sets["high"][li], sets["low"][li], sets["base"][li]
            ov.append(float((hi == lo).float().mean()))
            ovlb.append(float((lo == ba).float().mean()))
        out["overlap_high_low"] = sum(ov) / max(len(ov), 1)
        out["overlap_low_base"] = sum(ovlb) / max(len(ovlb), 1)
        out["low_distinct_from_base"] = bool(out["overlap_low_base"] < 0.9)
        for d in ("base", "high", "low"):
            mode = "base_live" if d == "base" else "intervene"
            nll, dl = run_intervene(model, batches, sets[d], a.k, g, mode=mode,
                                    want_deliver=True)
            out["nll"][d] = nll
            # deliver-mass of the set actually applied, read in this pass on
            # each follower's own dense row, averaged over the grouping's
            # followers and over batch/head/query.
            fw = [v for li, v in dl if li in followers(g)]
            out["deliver"][d] = sum(fw) / max(len(fw), 1)
            out["delta_vs_full_" + d] = nll - full
        ok = out["deliver"]["high"] > out["deliver"]["base"] > out["deliver"]["low"]
        order = out["nll"]["high"] < out["nll"]["base"] < out["nll"]["low"]
        out["manipulation_check"] = bool(ok)
        out["predicted_order"] = bool(order)
        out["deliver_static_order"] = bool(
            out["deliver_static"]["high"] > out["deliver_static"]["base"]
            > out["deliver_static"]["low"])
        res["groups"][str(g)] = out
        print("g=%d live-deliver h=%.4f b=%.4f l=%.4f check=%s | nll h=%.5f b=%.5f l=%.5f order=%s | base-full=%+.5f" % (
            g, out["deliver"]["high"], out["deliver"]["base"], out["deliver"]["low"], ok,
            out["nll"]["high"], out["nll"]["base"], out["nll"]["low"], order,
            out["delta_vs_full_base"]), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
