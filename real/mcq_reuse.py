#!/usr/bin/env python3
"""Does the NLL cost of cross-layer top-k reuse reach *decisions*?

`reports/06-reuse-real-cost.md` measures the cost as a change in token-level
NLL. §9 lists "no downstream task" as an explicit boundary, and that boundary is
the difference between "reuse costs 0.06 nats" and "reuse changes answers".
This script closes exactly that gap and nothing more: multiple-choice accuracy
on ARC-Easy / MMLU, with the same readings as the NLL grid.

Design notes that decide whether the numbers mean anything
----------------------------------------------------------
* **Scoring, not generation.** Every option is scored by the summed
  log-probability of its own tokens given the prompt. Generation would add a
  decoding policy on top of the attention intervention, and then an accuracy
  change could not be attributed to the reuse path.
* **No KV cache.** Each scoring forward is one full pass over
  `prompt + option`. Returning a fake cache from a replaced attention forward
  silently corrupts every later step; not using one is slower and correct.
  The sparse path therefore also applies to the option tokens, identically in
  every reading.
* **Per-item paired outcome.** We keep per-item correctness under each reading,
  so the interesting quantity is the flip counts (right→wrong vs wrong→right),
  not two accuracies. A cost that only shows up as "both wrong" is noise.
* **Control at each item's own length.** With `k >= seq_len` the sparse path
  keeps every key, so it must reproduce the dense per-item outcome exactly.
  Otherwise the harness is wrong and the run exits 2 — same discipline as
  `reuse_real.py`, but the comparison is per-item outcomes rather than 1e-9 on
  the loss, because a padded batch has no bit-identity to appeal to.

Run:
  /opt/venv/bin/python mcq_reuse.py --model models/Qwen2.5-7B --bench arc_easy \
      --n 600 --ks 128,256,384 --groups 2,3,6 --out out/mcq_7b_arc.json
  /opt/venv/bin/python mcq_reuse.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# With the mask formulation (see patch_model) k == len is bit-exact, so the
# control can be as tight as reuse_real.py's 1e-9 on the loss. 1e-6 leaves room
# for the option log-prob accumulation over different token counts.
CTRL_TOL = 1e-6

STATE = {"mode": "full", "k": None, "group": None,
         "own_sel": {}, "reuse_sel": {}}


def lead_of(layer_idx, group):
    return (layer_idx // group) * group


def resolve_sel(scores, borrowed, kk, qlen, layer_idx, group):
    """Pick the selection this layer uses. Never silently use a wrong one.

    Inside a *reuse* pass the group's lead layer computes its own selection and
    publishes it; the other layers of that group read it. That is the whole
    experiment — an earlier draft only published at the end of the item, so
    every non-lead layer found nothing, fell back to its own selection, and the
    "reuse" reading was a bit-exact copy of the "own" reading. The control could
    not catch it (the control is about k >= len). Hence `borrows` is counted
    here and `run()` refuses a reuse pass that borrowed nothing.

    A borrowed selection is only valid at the same query length: `gather` with a
    shorter index does not raise, it reads the wrong columns.
    """
    if STATE["mode"] == "own":
        return scores.topk(kk, dim=-1).indices

    lead = lead_of(layer_idx, group)
    if layer_idx == lead:                      # seed the group
        sel = scores.topk(kk, dim=-1).indices
        STATE["reuse_sel"][lead] = sel
        return sel

    sel = STATE["reuse_sel"].get(lead)
    if sel is None or sel.shape[-2] != qlen or sel.shape[-1] != kk:
        STATE["fallbacks"] = STATE.get("fallbacks", 0) + 1
        return scores.topk(kk, dim=-1).indices
    STATE["borrows"] = STATE.get("borrows", 0) + 1
    return sel


def patch_model(model):
    """Replace each decoder layer's attention forward with the instrumented one.

    Only the mask/selection step changes: q/k/v projection, RoPE and the output
    projection stay the model's own code. Reimplementing those would make the
    comparison about my arithmetic instead of about reuse.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        def make(idx, attn=attn):
            def forward(hidden_states, position_embeddings, attention_mask=None,
                        past_key_value=None, cache_position=None, **kwargs):
                bsz, qlen, _ = hidden_states.shape
                # transformers 4.57 exposes head counts on config, not on the
                # attention module (attn.num_heads does not exist).
                nh = attn.config.num_attention_heads
                nkv = attn.config.num_key_value_heads
                q = attn.q_proj(hidden_states).view(bsz, qlen, nh,
                                                    attn.head_dim).transpose(1, 2)
                k = attn.k_proj(hidden_states).view(bsz, qlen, nkv,
                                                    attn.head_dim).transpose(1, 2)
                v = attn.v_proj(hidden_states).view(bsz, qlen, nkv,
                                                    attn.head_dim).transpose(1, 2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
                v = v.repeat_interleave(attn.num_key_value_groups, dim=1)

                scores = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
                neg = torch.finfo(scores.dtype).min
                causal = torch.triu(torch.ones(qlen, qlen, dtype=torch.bool,
                                               device=scores.device), diagonal=1)
                scores = scores.masked_fill(causal, neg)
                if attention_mask is not None and attention_mask.dim() == 4:
                    scores = scores + attention_mask

                if STATE["mode"] == "full":
                    weights = torch.softmax(scores, dim=-1)
                else:
                    kk = min(STATE["k"], qlen)
                    sel = resolve_sel(scores, None, kk, qlen, idx, STATE["group"])
                    # Mask, do not gather. Gather+scatter reorders the softmax
                    # reduction (topk returns indices sorted by value), and in
                    # bf16 addition is not associative: the same values summed in
                    # a different order differ in the last bits, and 28 layers
                    # amplify that to ~0.17 in an option log-prob at k == qlen.
                    # Keeping the column order and masking dropped keys makes
                    # k == qlen a bit-exact no-op, which is what a control needs.
                    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(3, sel, True)
                    scores = scores.masked_fill(~keep, neg)
                    weights = torch.softmax(scores, dim=-1)
                    if kk >= qlen:
                        # Structural assertion: "k >= len keeps every key" must
                        # mean exactly that, independent of any arithmetic.
                        assert keep.all(), "k >= qlen must keep every key"

                out = torch.matmul(weights, v)
                out = out.transpose(1, 2).reshape(bsz, qlen, -1).contiguous()
                # The Qwen2DecoderLayer unpacks exactly two values
                # (`hidden_states, _ = self.self_attn(...)`); a 3-tuple raises
                # "too many values to unpack" inside transformers, not here.
                return attn.o_proj(out), None
            return forward

        attn.forward = make(i)


def build_items(bench, n, tok, seq, shots=0):
    """[(head_ids, [option_ids...], gold)] with the question truncated, not the cue.

    `shots` prepends few-shot exemplars. This is not cosmetic: ARC-Easy questions
    are ~22 tokens long, and a top-k of 128 over a 22-token prefix is not a
    sparse-attention experiment at all — it keeps every key. The few-shot prefix
    is what puts the prompts into the length regime where cross-layer reuse is
    actually proposed to save work, so the comparison has a subject.
    """
    from datasets import load_dataset
    if bench in ("arc_easy", "arc_challenge"):
        # ARC-Challenge is the same format and the same instrument with a harder
        # split: ARC-Easy answers are mostly recall, Challenge ones need a step
        # of reasoning. Dense accuracy on Easy has almost no headroom left on
        # 1.5B/7B (0.76 / 0.79), and a ceiling is the worst place to measure a
        # perturbation — damage saturates at "cannot get worse".
        cfg = "ARC-Easy" if bench == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        raw = [(r["question"], r["choices"]["text"],
                "ABCDE".index(r["answerKey"]) if r["answerKey"] in "ABCDE" else None)
               for r in ds]
    elif bench == "mmlu":
        ds = load_dataset("cais/mmlu", "high_school_mathematics", split="test")
        raw = [(r["question"], r["choices"], r["answer"]) for r in ds]
    else:
        raise SystemExit(f"unknown bench {bench}")

    # Exemplars: take the first `shots` items with a known answer as demonstrations.
    # Each one is kept as a separate string so an item can use as many as fit:
    # MMLU questions are long enough that 5 shots exceed a seq-384 budget, and
    # truncating the *prompt* (which is what `head_ids[-budget:]` does) throws
    # away the question being asked and leaves only demonstrations — the model
    # then scores options against a prompt that never states the problem. That
    # is a measurement of nothing, and it reports a plausible-looking accuracy.
    demos = []
    if shots:
        for q, choices, gold in raw[:shots]:
            if gold is None or len(choices) < 3:
                continue
            demos.append(f"Question: {q}\nAnswer: {choices[gold]}\n")

    # Two passes. The first finds the largest demonstration count that every
    # kept item can afford; the second builds all items with exactly that count.
    # A per-item count would make prompt lengths incomparable across the test
    # set, and the number of shots would then differ between items that reuse
    # cost happens to damage and items it does not — a confound in the only
    # comparison the experiment makes.
    plan = []
    for q, choices, gold in raw[shots:]:
        if gold is None or len(choices) < 3:
            continue
        question = f"Question: {q}\nAnswer:"
        opt_ids = [tok(" " + c, add_special_tokens=False).input_ids for c in choices]
        if any(len(o) == 0 for o in opt_ids):
            continue
        # Keep prompt + longest option <= seq so the whole scored sequence fits.
        budget = seq - max(len(o) for o in opt_ids)
        if budget < 8:
            continue
        q_ids = tok(question, return_tensors="pt").input_ids[0]
        if len(q_ids) > budget:
            continue
        allowance = 0
        for i in range(1, len(demos) + 1):
            cand = tok("".join(demos[:i]) + question,
                       return_tensors="pt").input_ids[0]
            if len(cand) <= budget:
                allowance = i
        plan.append((question, opt_ids, gold, allowance))
        if len(plan) >= n:
            break

    used = min((p[3] for p in plan), default=0)
    STATE["shots_used"] = used
    prefix = "".join(demos[:used])
    items = []
    for question, opt_ids, gold, allowance in plan:
        if allowance < used:
            # cannot happen: `used` is the minimum allowance. Asserted rather
            # than silently truncated, because a shorter prompt for one item is
            # exactly the confound the uniform count is there to remove.
            raise SystemExit(f"shot-count plan inconsistent: {allowance} < {used}")
        head_ids = tok(prefix + question, return_tensors="pt").input_ids[0]
        items.append((head_ids, opt_ids, gold))
    if shots:
        print(f"shots requested={shots} used={used} (uniform across "
              f"{len(items)} items)", flush=True)
    return items


@torch.no_grad()
def score_item(model, head_ids, opt_ids, device, length_norm=False):
    """Score each option by the sum of its own token log-probabilities.

    `length_norm` divides by the option length. Sum-of-logprob has a documented
    bias toward short answers (each factor is negative, so fewer factors score
    higher). On ARC-Easy the four options are near-equal in length and the bias
    is negligible; on MMLU the options range from 2 to 20 tokens, and a model
    scoring below chance on the un-normalised metric is the symptom. The
    default stays `False` so the ARC numbers already reported are not silently
    recomputed under a different metric.
    """
    scores = []
    for o in opt_ids:
        full = torch.cat([head_ids, torch.tensor(o)]).unsqueeze(0).to(device)
        logits = model(full).logits[0]
        base = len(head_ids) - 1
        lp = torch.log_softmax(logits[base:base + len(o)].float(), dim=-1)
        total = sum(lp[t, tid].item() for t, tid in enumerate(o))
        scores.append(total / max(len(o), 1) if length_norm else total)
    best = int(max(range(len(scores)), key=lambda i: scores[i]))
    return best, scores


def run(model, items, mode, k=None, group=None, device="cuda", length_norm=False):
    """Return (per-item correctness, per-item option score vectors)."""
    STATE.update({"mode": mode, "k": k, "group": group,
                  "own_sel": {}, "reuse_sel": {}, "fallbacks": 0, "borrows": 0})
    per_item, score_rows = [], []
    for head_ids, opt_ids, gold in items:
        # reuse_sel lives for exactly one item: a selection made while scoring
        # item n must never be read while scoring item n+1 (different tokens).
        STATE["reuse_sel"] = {}
        best, sc = score_item(model, head_ids, opt_ids, device, length_norm)
        per_item.append(1 if best == gold else 0)
        score_rows.append(sc)
    if mode == "reuse":
        STATE["noop"] = STATE["borrows"] == 0
    return per_item, score_rows


def flips(a, b):
    return {"right_to_wrong": sum(1 for x, y in zip(a, b) if x and not y),
            "wrong_to_right": sum(1 for x, y in zip(a, b) if not x and y)}


def selftest():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 7, 4
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    kk = torch.randn(B, H, T, D, dtype=torch.float64)
    sc = torch.matmul(q, kk.transpose(2, 3))
    neg = torch.finfo(sc.dtype).min
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    sc = sc.masked_fill(causal, neg)
    dense = torch.softmax(sc, dim=-1)

    # 1. k == T reproduces dense weights exactly
    sel = sc.topk(T, dim=-1).indices
    w = torch.zeros_like(sc).scatter_(3, sel, torch.softmax(torch.gather(sc, 3, sel), dim=-1))
    assert torch.allclose(w, dense, atol=1e-12), (w - dense).abs().max()

    # 2. a wrong-length borrowed selection must be *replaced*, not used.
    #    gather with a shorter index does not raise on ROcm/torch — it reads the
    #    wrong columns — so the guard has to be an explicit length check.
    # group=2 so layer 1 is a *follower* of lead 0 (with group=1 every layer is
    # its own lead and there is nothing to borrow from).
    STATE.update({"mode": "reuse", "k": 3, "group": 2, "reuse_sel": {0: sel[..., :4, :]},
                  "own_sel": {}, "fallbacks": 0})
    used = resolve_sel(sc, None, 3, T, layer_idx=1, group=2)
    assert used.shape[-2] == T and used.shape[-1] == 3, used.shape
    assert STATE["fallbacks"] == 1, "wrong-length borrow must be counted as a fallback"

    # 2b. THE no-op bug: with a group, the lead layer must seed and the others
    #     must borrow. An earlier draft published only at item end, so every
    #     layer fell back to its own selection and "reuse" == "own" bit for bit.
    T2, D2 = 6, 4
    sc2 = torch.randn(1, 2, T2, T2, dtype=torch.float64)
    STATE.update({"mode": "reuse", "k": 3, "group": 3, "reuse_sel": {},
                  "own_sel": {}, "fallbacks": 0, "borrows": 0})
    for layer in range(T2):
        resolve_sel(sc2, None, 3, T2, layer, 3)
    assert STATE["borrows"] > 0, "non-lead layers must borrow within one item"
    assert STATE["fallbacks"] == 0
    assert set(STATE["reuse_sel"]) == {0, 3}, STATE["reuse_sel"]

    # 2c. group=1 means every layer is a lead: nothing can be borrowed, and that
    #     must be detectable rather than silently reported as a reuse result.
    STATE.update({"mode": "reuse", "k": 3, "group": 1, "reuse_sel": {},
                  "own_sel": {}, "fallbacks": 0, "borrows": 0})
    for layer in range(T2):
        resolve_sel(sc2, None, 3, T2, layer, 1)
    assert STATE["borrows"] == 0, "group=1 cannot borrow"

    # 3. a correct-length borrow is used as-is (so the fallback counter is not
    #    permanently on, which would make the guard untestable)
    good = sc.topk(3, dim=-1).indices
    # group=3 makes layer 1 belong to the group led by layer 0, so the key exists
    STATE.update({"mode": "reuse", "k": 3, "group": 3, "reuse_sel": {0: good},
                  "own_sel": {}, "fallbacks": 0})
    assert torch.equal(resolve_sel(sc, None, 3, T, 1, 3), good)
    assert STATE["fallbacks"] == 0

    # 4. sparsification cannot invent mass (mask formulation)
    keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(3, good, True)
    w2 = torch.softmax(sc.masked_fill(~keep, neg), dim=-1)
    assert torch.allclose(w2.sum(-1), torch.ones_like(w2.sum(-1)))
    assert w2.masked_fill(keep, 0).abs().max() == 0

    # 4b. Why the mask formulation is mandatory. On synthetic tensors gather+
    #     scatter and masking agree to within a few ulp, so this test cannot
    #     reproduce the real failure — but it CAN pin the property that matters:
    #     with every key kept, masking returns the *identical* tensor entry for
    #     entry, so softmax reduces over the same element order. The observed
    #     7B failure (max |Δ option log-prob| = 0.17 at k = 284 > len) came from
    #     the gathered layout, where softmax sees a permuted vector and bf16
    #     rounding differs per element. Recorded here so the next person does not
    #     "simplify" back to gather.
    xb = sc.to(torch.bfloat16)
    keep_all = torch.zeros_like(xb, dtype=torch.bool).scatter_(
        3, xb.topk(T, dim=-1).indices, True)
    masked_all = xb.masked_fill(~keep_all, torch.finfo(xb.dtype).min)
    assert torch.equal(masked_all, xb), "keep-all masking must be a no-op, bit for bit"
    assert torch.equal(torch.softmax(masked_all, dim=-1), torch.softmax(xb, dim=-1))

    # 5. lead_of must put every layer of a group on its first layer
    assert [lead_of(i, 3) for i in range(7)] == [0, 0, 0, 3, 3, 3, 6]

    # 6. flip accounting is directional and does not double count
    assert flips([1, 1, 0, 0], [0, 1, 1, 0]) == {"right_to_wrong": 1, "wrong_to_right": 1}
    print("selftest OK (k==T bit-exact via mask, keep-all masking is a no-op, "
          "wrong-length borrow replaced + counted, lead seeds and others "
          "borrow, group=1 cannot borrow, right-length borrow honoured, mass "
          "conserved, lead mapping, flip accounting)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-7B")
    ap.add_argument("--bench", default="arc_easy",
                    choices=["arc_easy", "arc_challenge", "mmlu"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--shots", type=int, default=0,
                    help="few-shot exemplars; 0 gives ~22-token prompts where "
                         "k>=64 keeps every key and there is nothing to sparsify")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--ks", default="128,256,384")
    ap.add_argument("--groups", default="2,3,6")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--length-norm", action="store_true",
                    help="divide each option's log-prob sum by its token count; "
                         "needed when options differ a lot in length (MMLU)")
    ap.add_argument("--out", default="out/mcq.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), device_map="cuda",
        attn_implementation="eager")
    model.eval()
    patch_model(model)

    items = build_items(a.bench, a.n, tok, a.seq, a.shots)
    length_norm = a.length_norm
    lens = [len(h) for h, _, _ in items]
    full_lens = [len(h) + len(o) for h, opts, _ in items for o in opts]
    med = int(torch.tensor(lens).median())
    print(f"items={len(items)} prompt_len median={med} max={max(lens)} "
          f"full_len max={max(full_lens)}", flush=True)

    res = {"model": a.model, "bench": a.bench, "seq": a.seq, "n_items": len(items),
           "shots": a.shots,
           # `shots` above records what was ASKED for. When a long stem leaves
           # no room, build_items drops exemplars rather than truncating the
           # question, so the two can differ — and a reader comparing two runs
           # would then compare a benchmark change with a shot change. Recorded
           # separately so the confound is visible in the artifact itself.
           "shots_used": STATE.get("shots_used"),
           "prompt_len_median": med, "prompt_len_max": max(lens),
           "full_len_max": max(full_lens),
           "dtype": a.dtype, "length_norm": a.length_norm,
           "readings": {}, "per_item": {}}
    if not items:
        print("no usable items", flush=True)
        return 1

    full, full_sc = run(model, items, "full", length_norm=length_norm)
    res["readings"]["full"] = sum(full) / len(full)
    res["per_item"]["full"] = full
    print(f"full acc={res['readings']['full']:.4f}", flush=True)

    # Control: keep every key. Must reproduce dense outcomes item-for-item.
    # The scored sequence is prompt + option, so the length that matters is the
    # LONGEST full sequence, not the longest prompt — using max(lens) let one
    # item with a long option exceed the budget and silently drop keys, which is
    # exactly what a control is supposed to catch (it did).
    ctrl_k = max(len(h) + len(o) for h, opts, _ in items for o in opts)
    ctrl, ctrl_sc = run(model, items, "own", k=ctrl_k, length_norm=length_norm)
    res["readings"][f"control_own_k{ctrl_k}"] = sum(ctrl) / len(ctrl)
    # The control's subject is "the sparse path kept every key". That is a
    # statement about option *scores*, not about decisions: bf16 softmax over a
    # re-sorted index set differs in the last bits, and a decision near equal
    # option scores can flip on that alone. So compare scores with a tolerance
    # and report the deviation; only a real key-drop moves scores by more.
    dev = max(abs(a - b) for r1, r2 in zip(full_sc, ctrl_sc)
              for a, b in zip(r1, r2)) if full_sc else 0.0
    ctrl_flips = flips(full, ctrl)
    margins = []
    for r1, r2, f1, f2 in zip(full_sc, ctrl_sc, full, ctrl):
        if f1 != f2:
            srt = sorted(r1, reverse=True)
            margins.append((f1, f2, srt[0] - srt[1], max(abs(a - b) for a, b in zip(r1, r2))))
    res["control"] = {"k": ctrl_k, "max_abs_score_deviation": dev,
                      "same_per_item": ctrl == full, "flips": ctrl_flips,
                      "flip_margins": margins,
                      "fallbacks": STATE.get("fallbacks", 0),
                      "tolerance": CTRL_TOL}
    if dev > CTRL_TOL:  # CTRL_TOL is now 1e-6: mask formulation is bit-exact
        print(f"CONTROL FAILED at k={ctrl_k}: max |score deviation| = {dev:.3e} "
              f"> {CTRL_TOL:.1e} (keys were dropped)", flush=True)
        json.dump(res, open(a.out, "w"), indent=2)
        return 2
    print(f"control ok: k={ctrl_k} max|Δscore|={dev:.2e} flips={ctrl_flips}", flush=True)

    for k in [int(x) for x in a.ks.split(",")]:
        own, own_sc = run(model, items, "own", k=k, length_norm=length_norm)
        res["readings"][f"own_k{k}"] = sum(own) / len(own)
        res["per_item"][f"own_k{k}"] = own
        res.setdefault("flips_vs_full", {})[f"own_k{k}"] = flips(full, own)
        res.setdefault("score_dev_vs_own", {})[f"own_k{k}"] = max(
            abs(a - b) for r1, r2 in zip(own_sc, own_sc) for a, b in zip(r1, r2))
        for g in [int(x) for x in a.groups.split(",")]:
            re_, re_sc = run(model, items, "reuse", k=k, group=g, length_norm=length_norm)
            if STATE.get("noop"):
                print(f"NO-OP REUSE at k={k} g={g}: no layer borrowed a selection; "
                      f"the reading would be a copy of own_k{k}", flush=True)
                json.dump(res, open(a.out, "w"), indent=2)
                return 3
            res["readings"][f"reuse_k{k}_g{g}"] = sum(re_) / len(re_)
            res.setdefault("borrows", {})[f"k{k}_g{g}"] = STATE["borrows"]
            res.setdefault("fallbacks", {})[f"k{k}_g{g}"] = STATE["fallbacks"]
            res.setdefault("score_delta_vs_own", {})[f"k{k}_g{g}"] = sum(
                y - x for r1, r2 in zip(own_sc, re_sc) for x, y in zip(r1, r2)) / \
                sum(len(r1) for r1 in own_sc)
            res["per_item"][f"reuse_k{k}_g{g}"] = re_
            res.setdefault("flips_vs_own", {})[f"k{k}_g{g}"] = flips(own, re_)
        print(f"k={k}: own={res['readings'][f'own_k{k}']:.4f} "
              + " ".join(f"g{g}={res['readings'][f'reuse_k{k}_g{g}']:.4f}"
                         for g in [int(x) for x in a.groups.split(",")]), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("WROTE", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
