"""Paired measurement on the LatentIndex checkpoint.

Four cells, all in one run, all on the same batches:

  own        -- unmodified: every follower reads group[offset], i.e. it
                scores with its own projection (the published behaviour).
  offset0    -- every follower reads group[0], i.e. the leader's projection.
                This is exactly how an IndexCache follower behaves, so
                own - offset0 is what layer-specific scoring buys.
  null_own   -- follower keeps its own score, but the SET of selected
                indices is replaced with a random set of the same size.
  null_off0  -- follower reads group[0] AND the set is replaced.

Each cell is one extra forward per batch.  Deterministic algorithms are
on; a repeat of the base pass has to reproduce it exactly.
"""
import argparse, json, os, sys, time
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/DSV2-Lite-LatentIndex-Sparse-1250")
    ap.add_argument("--corpus", default="/root/wt2_test.txt")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--n-batches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gpu-gib", default="34GiB")
    ap.add_argument("--out", default="/root/attn_newstack/li_run.json")
    a = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

    sys.path.insert(0, "/tmp")
    import patch_li
    modeling = os.path.join(a.model, "modeling_dots_dsa.py")
    print("instrument:", patch_li.patch(modeling), flush=True)

    # transformers copies trust_remote_code files into its own module cache and
    # imports THAT copy, not the one in the model directory.  Patching only the
    # model directory therefore produces "already patched" followed by a stack
    # trace pointing at the unpatched cached file.  Drop the cache so it is
    # rebuilt from the patched source, and verify after loading that the module
    # actually imported is the instrumented one.
    import glob, shutil
    for d in glob.glob(os.path.expanduser(
            "~/.cache/huggingface/modules/transformers_modules/*LatentIndex*")):
        shutil.rmtree(d, ignore_errors=True)
        print("purged module cache:", d, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager", device_map="auto", low_cpu_mem_usage=True,
        max_memory={0: a.gpu_gib, "cpu": "260GiB"})
    model.eval()
    MM = sys.modules[type(model).__module__]
    _src = open(MM.__file__, encoding="utf-8").read()
    if "ROUTE_OFFSET" not in _src or "DO_CAPTURE" not in _src:
        raise SystemExit(f"the module actually imported is NOT instrumented: {MM.__file__}")
    print("module instrumented:", MM.__file__, flush=True)

    text = open(a.corpus, encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    stride = a.seq * 7
    batches = [ids[i * stride:i * stride + a.seq + 1] for i in range(a.n_batches)]
    batches = [b for b in batches if len(b) == a.seq + 1]
    print("batches", len(batches), "layers", model.config.num_hidden_layers,
          "topk", model.config.index_topk, "method", model.config.dots_dsa_method, flush=True)

    L, KTOP, CHUNK = model.config.num_hidden_layers, model.config.index_topk, model.config.indexer_query_chunk_size

    def nll(bb):
        """Mean NLL, accumulated in position slices.

        A full 4096-position forward produces a 4096 x vocab logits tensor;
        casting all of it to float32 at once needs ~1.7 GiB on top of the
        bf16 copy and is what OOM'd this box on the first attempt.  Slicing
        keeps the extra footprint at ~200 MiB and does not change the value:
        the mean is taken over the same number of positions.
        """
        tgt = bb[1:].cuda()
        total, n = 0.0, 0
        with torch.no_grad():
            out = model(input_ids=bb[None].cuda(), use_cache=False)
            lg = out.logits[0, :-1]
            for i in range(0, lg.shape[0], 512):
                seg = lg[i:i + 512].float()
                total += torch.nn.functional.cross_entropy(
                    seg, tgt[i:i + 512], reduction="sum").item()
                n += seg.shape[0]
                del seg
            del lg, out
        torch.cuda.empty_cache()
        return total / n

    def capture(bb):
        MM.SUBSET.clear(); MM.ROUTE_OFFSET.clear(); MM.CAPTURE.clear()
        MM.DO_CAPTURE[0] = True
        with torch.no_grad():
            model(input_ids=bb[None].cuda(), use_cache=False)
        MM.DO_CAPTURE[0] = False
        caps = {l: {k: v.clone() for k, v in d.items()} for l, d in MM.CAPTURE.items()}
        MM.CAPTURE.clear()
        return caps

    def rand_for(cap, start, end, key_length, layer):
        qpos = torch.arange(start, end, device="cuda")
        g = torch.Generator(device="cuda").manual_seed(a.seed * 100003 + layer + start)
        noise = torch.rand((end - start, key_length), generator=g, dtype=torch.float32,
                           device="cuda")
        noise.masked_fill_(torch.arange(key_length, device="cuda")[None, :] > qpos[:, None], -1.0)
        sel = noise.topk(min(KTOP, key_length), dim=-1).indices
        sel = torch.minimum(sel, qpos[:, None])
        return sel.unsqueeze(0)   # topk returns (batch, chunk, k); batch is 1 here

    base = []
    for bb in batches:
        base.append(nll(bb))
        print("  base", len(base), f"{base[-1]:.9f}", flush=True)

    repeat = nll(batches[0])
    _rd = abs(repeat - base[0])
    _tag = "DETERMINISTIC" if _rd < 1e-9 else "DRIFTS"
    print("  repeat", f"{repeat:.9f}", "range", f"{_rd:.3e}", _tag, flush=True)

    caps = [capture(bb) for bb in batches]
    print("captured", len(caps), "batches; layers holding a route:",
          len(caps[0]), flush=True)

    followers = {}
    for l in range(L):
        att = model.model.layers[l].self_attn
        if att.offset > 0 and att.group_size > 1:
            followers[l] = att
    print("followers", len(followers), "leaders", len(set(v.leader for v in followers.values())), flush=True)

    cells = {"own": [], "offset0": [], "null_own": [], "null_off0": []}

    # own: nothing touched -- this is a control and must come back as exactly 0
    for i, bb in enumerate(batches):
        cells["own"].append(nll(bb) - base[i])

    # offset0: every follower reads the leader's offset-0 projection
    MM.ROUTE_OFFSET.clear()
    for l in followers:
        MM.ROUTE_OFFSET[l] = 0
    for i, bb in enumerate(batches):
        cells["offset0"].append(nll(bb) - base[i])
    MM.ROUTE_OFFSET.clear()

    # null_own: own score, random set
    for i, bb in enumerate(batches):
        MM.SUBSET.clear()
        for l in followers:
            for (s, e) in caps[i][l]:
                MM.SUBSET.setdefault(l, {})[(s, e)] = rand_for(None, s, e, e, l)
        cells["null_own"].append(nll(bb) - base[i])
    MM.SUBSET.clear()

    # null_off0: leader's projection AND random set
    MM.ROUTE_OFFSET.clear()
    for l in followers:
        MM.ROUTE_OFFSET[l] = 0
    for i, bb in enumerate(batches):
        MM.SUBSET.clear()
        for l in followers:
            for (s, e) in caps[i][l]:
                MM.SUBSET.setdefault(l, {})[(s, e)] = rand_for(None, s, e, e, l)
        cells["null_off0"].append(nll(bb) - base[i])
    MM.SUBSET.clear(); MM.ROUTE_OFFSET.clear()

    for name, vals in cells.items():
        print(f"  {name:9s}", " ".join(f"{v:+.6f}" for v in vals),
              "mean", f"{sum(vals)/len(vals):+.6f}", flush=True)

    out = {"model": a.model, "method": model.config.dots_dsa_method, "seq": a.seq,
           "topk": KTOP, "chunk": CHUNK, "n_batches": len(batches), "seed": a.seed,
           "layers": L, "n_followers": len(followers),
           "leaders": sorted(set(v.leader for v in followers.values())),
           "base_per_batch": base, "repeat_delta": abs(repeat - base[0]),
           "deltas": cells}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("WROTE", a.out, flush=True)

if __name__ == "__main__":
    main()
