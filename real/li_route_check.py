"""Manipulation check: does forcing ROUTE_OFFSET actually change what a
follower selects?

The four-cell run reported `offset0 - own = +0.000105` with a CI crossing
zero.  Two readings fit that number equally well:

  a) the intervention took effect and the follower's own score is worth
     nothing measurable, or
  b) the intervention never reached the model.

This distinction is the same one that cost two wrong conclusions earlier in
this project (a `torch.roll` that could not change a boolean mask, and a
probe aimed at a chunk that could not discriminate).  So: capture indices
with own scoring, capture again with the offset forced, and count how many
selected sets actually differ per chunk.
"""
import argparse, json, os, sys
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/DSV2-Lite-LatentIndex-Sparse-1250")
    ap.add_argument("--corpus", default="/root/wt2_test.txt")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--gpu-gib", default="34GiB")
    ap.add_argument("--out", default="/root/attn_newstack/li_route_check.json")
    a = ap.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

    sys.path.insert(0, "/tmp")
    import patch_li
    print("instrument:", patch_li.patch(os.path.join(a.model, "modeling_dots_dsa.py")), flush=True)
    import glob, shutil
    for d in glob.glob(os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/*LatentIndex*")):
        shutil.rmtree(d, ignore_errors=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager", device_map="auto", low_cpu_mem_usage=True,
        max_memory={0: a.gpu_gib, "cpu": "260GiB"})
    model.eval()
    MM = sys.modules[type(model).__module__]
    assert "ROUTE_OFFSET" in open(MM.__file__, encoding="utf-8").read()
    print("module:", MM.__file__, flush=True)

    text = open(a.corpus, encoding="utf-8").read()
    bb = tok(text, return_tensors="pt").input_ids[0][:a.seq + 1]
    L = model.config.num_hidden_layers

    def grab():
        MM.CAPTURE.clear(); MM.DO_CAPTURE[0] = True
        with torch.no_grad():
            model(input_ids=bb[None].cuda(), use_cache=False)
        MM.DO_CAPTURE[0] = False
        return {l: {k: v.clone() for k, v in d.items()} for l, d in MM.CAPTURE.items()}

    followers = {}
    for l in range(L):
        att = model.model.layers[l].self_attn
        if att.offset > 0 and att.group_size > 1:
            followers[l] = att
    print("followers", len(followers), flush=True)

    own = grab()
    for l in followers:
        MM.ROUTE_OFFSET[l] = 0
    forced = grab()
    MM.ROUTE_OFFSET.clear()

    MM.CAPTURE.clear()
    again = grab()

    def same_set(x, y):
        """True when x and y select the same set of positions.

        The first version of this function compared x to itself, so every
        repeat looked identical and a real question ("did the second
        capture reproduce the first?") became a tautology.  It also
        returned None on a shape mismatch, which `all()` reads as False --
        so "cannot compare" and "compared and differed" were the same
        answer.  Both are fixed here, and the asserts below prove the
        comparator can say no: it is checked with two DIFFERENT sets.
        """
        if x.shape != y.shape:
            raise ValueError(f'shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}')
        return bool((torch.sort(x.reshape(-1)).values ==
                     torch.sort(y.reshape(-1)).values).all())

    _a = torch.arange(4).view(1, 1, 4)
    _b = torch.arange(4).view(1, 1, 4) + 1
    assert same_set(_a, _b) is False, 'comparator cannot tell two sets apart'
    assert same_set(_a, _a.clone()) is True, 'comparator rejects identical sets'

    rows = []
    for l in sorted(followers):
        for (s, e) in sorted(own[l]):
            o = own[l][(s, e)]
            f = forced[l][(s, e)]
            # per chunk, not summed over chunks: summing first mixes chunks
            # into one intersection and can exceed 2k, which is how this
            # first version printed a negative Jaccard
            inter = (o.unsqueeze(-1) == f.unsqueeze(-2)).any(-2).sum(-1).float()
            kk = o.shape[-1]
            jac = (inter / (2 * kk - inter)).mean().item()
            rows.append({"layer": l, "chunk_start": s, "jaccard_own_vs_offset0": jac,
                         "identical": same_set(o, f),
                         "chunk_is_live": s + 1 > model.config.index_topk})

    live = [r for r in rows if r["chunk_is_live"]]
    inert = [r for r in rows if not r["chunk_is_live"]]
    # Repeatability across two independent captures.  `same_set` raises on a
    # shape mismatch now, so a structural difference shows up as an exception
    # instead of being silently folded into False.
    det = all(same_set(own[l][(s, e)], again[l][(s, e)]) for l in sorted(followers) for (s, e) in own[l])
    det_all = all(same_set(own[l][k], again[l][k]) for l in sorted(own) for k in own[l])
    # Restricted to chunks that can discriminate.  On inert chunks top-k runs
    # over positions that are all masked to -inf, so which ones survive is a
    # tie-break, not a decision; a disagreement there says nothing about
    # determinism of the measurement.
    det_live = all(same_set(own[l][(s, e)], again[l][(s, e)])
                   for l in sorted(followers) for (s, e) in own[l]
                   if s + 1 > model.config.index_topk)
    det_inert = all(same_set(own[l][(s, e)], again[l][(s, e)])
                    for l in sorted(followers) for (s, e) in own[l]
                    if s + 1 <= model.config.index_topk)

    res = {"model": a.model, "seq": a.seq, "topk": model.config.index_topk,
           "n_followers": len(followers), "rows": rows,
           "capture_repeatable": det, "capture_repeatable_all_layers": det_all,
           "capture_repeatable_live": det_live, "capture_repeatable_inert": det_inert,
           "live_chunks": len(live), "inert_chunks": len(inert),
           "live_jaccard_mean": sum(r["jaccard_own_vs_offset0"] for r in live) / max(1, len(live)),
           "live_identical_share": sum(1 for r in live if r["identical"]) / max(1, len(live)),
           "inert_jaccard_mean": sum(r["jaccard_own_vs_offset0"] for r in inert) / max(1, len(inert)),
           "inert_identical_share": sum(1 for r in inert if r["identical"]) / max(1, len(inert))}
    json.dump(res, open(a.out, "w"), indent=1)
    print("capture repeatable  all:", det, " live-only:", det_live,
          " inert-only:", det_inert, flush=True)
    print(f"live chunks {len(live)}  jaccard(own,offset0) mean {res['live_jaccard_mean']:.4f}"
          f"  identical {res['live_identical_share']*100:.1f}%", flush=True)
    print(f"inert chunks {len(inert)}  jaccard mean {res['inert_jaccard_mean']:.4f}"
          f"  identical {res['inert_identical_share']*100:.1f}%", flush=True)

    # the gate: if forcing the offset leaves the selected sets untouched on the
    # chunks that CAN discriminate, the intervention did not bite and the
    # +0.000105 reading must not be reported as a finding
    if res["live_identical_share"] > 0.5:
        raise SystemExit("INTERVENTION DID NOT BITE on live chunks")
    print("MANIPULATION CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
