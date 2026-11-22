#!/usr/bin/env python3
"""Head-to-head probe: does deliver_mass add value over PLAIN selection overlap?

rev-da finding I1 (the sharpest adversarial objection): deliver_mass is defined
as leader-selected positions' mass under the follower's dense distribution, so
it may just be a decorative restatement of "the follower also likes where the
leader looks" -- i.e. plain overlap. If overlap predicts the reuse cost just as
well, deliver_mass has no incremental value.

This probe computes BOTH quantities on the SAME dense pass, per (source,target):
  deliver(s->t) = mean_q sum_{p in I^s_q} P^t_q[p]                    (our diagnostic)
  jaccard(s,t)  = mean_q |I^s_q ∩ I^t_q| / |I^s_q ∪ I^t_q|             (plain overlap)
where I^l_q is layer l's top-k key set for query position q. The index sets are
collected read-only by leader_mechanism.patch_collect; no reuse mask is applied.

A separate GPU-free script joins both to vs_own and asks whether deliver keeps
a partial correlation after controlling jaccard (and vice versa).
"""
import argparse
import json
import os
import torch

from leader_mechanism import patch_collect, deliver_matrix, STATE


def jaccard_matrix(sel_by_layer, max_distance=6):
    """jaccard[s][t] = mean over batches/heads/query-rows of the top-k set
    overlap between leader s and follower t. Both sets have size k; top-k has
    no duplicate positions, so |intersection| = count of exact index matches
    and |union| = 2k - |intersection|."""
    layers = sorted(sel_by_layer)
    out = {}
    for s in layers:
        for t in layers:
            d = t - s
            if not (1 <= d <= max_distance):
                continue
            vals = []
            for sel_s, sel_t in zip(sel_by_layer[s], sel_by_layer[t]):
                # each (B,H,T,k)
                k = sel_s.shape[-1]
                match_per_s = (sel_s.unsqueeze(-1) == sel_t.unsqueeze(-2)).sum(-1)  # (B,H,T,k)
                inter = match_per_s.sum(-1).float()  # (B,H,T) = |intersection|
                jac = inter / (2 * k - inter)
                vals.append(float(jac.mean()))
            assert vals
            out[(s, t)] = sum(vals) / len(vals)
    return out


def selftest():
    # 1. jaccard of identical sets = 1; disjoint sets = 0.
    a = torch.tensor([[[[0, 1, 2, 3]]]])  # (1,1,1,4)
    b = torch.tensor([[[[2, 3, 4, 5]]]])
    j_same = jaccard_matrix({0: [a], 1: [a]})[(0, 1)]
    assert abs(j_same - 1.0) < 1e-6, j_same
    j_half = jaccard_matrix({0: [a], 1: [b]})[(0, 1)]
    assert abs(j_half - (2.0 / 6.0)) < 1e-6, j_half  # |inter|=2, |union|=6
    c = torch.tensor([[[[10, 11, 12, 13]]]])
    j_none = jaccard_matrix({0: [a], 1: [c]})[(0, 1)]
    assert abs(j_none - 0.0) < 1e-6, j_none

    # 2. distance filter: only 1..max_distance kept.
    m = jaccard_matrix({0: [a], 1: [a], 2: [a], 5: [a]}, max_distance=2)
    assert (0, 1) in m and (0, 2) in m and (1, 2) in m
    assert (0, 5) not in m and (2, 5) not in m, sorted(m)
    print("SELFTEST OK  j_same=%.3f j_half=%.3f j_none=%.3f" % (j_same, j_half, j_none))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="/root/attn-reuse/models/Qwen2.5-0.5B")
    ap.add_argument("--corpus", default="/root/attn-reuse/wt2_test.txt")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--max-distance", type=int, default=6)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="/root/attn-reuse/out/mech_h2h_05.json")
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
    jm = jaccard_matrix(STATE["sel_idx"], a.max_distance)
    assert set(dm) == set(jm), (len(dm), len(jm))
    pairs = [{"source": s, "target": t,
              "deliver": dm[(s, t)], "jaccard": jm[(s, t)]}
             for (s, t) in sorted(dm)]
    result = {"model": a.model, "dtype": a.dtype, "seq": a.seq, "k": a.k,
              "n_batches": len(bs), "max_distance": a.max_distance,
              "n_layers": n_layers, "pairs": pairs}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", a.out, "pairs=", len(pairs))


if __name__ == "__main__":
    main()
