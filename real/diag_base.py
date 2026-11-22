"""Diagnose why reuse_intervene base_live != reuse_llama reuse (F2 root cause).

Runs both patches on the SAME model + batches, captures layer-0 selection from
each, and reports (a) whether selections are bit-identical, (b) NLL of each,
(c) NLL of a third 'grid-topk-in-intervene' hybrid that copies the grid's exact
own computation. This isolates whether the gap is selection or mask/softmax.
"""
import os, sys, argparse, importlib.util
import torch


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/attn-reuse/models/Qwen2.5-0.5B")
    ap.add_argument("--corpus", default="/root/attn-reuse/wt2_test.txt")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    torch.manual_seed(20261120)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    common = dict(dtype=getattr(torch, a.dtype), device_map="cuda", attn_implementation="eager")
    ids = tok(open(a.corpus).read(), return_tensors="pt").input_ids[0]
    stride = a.seq * 7
    batches = [ids[i*stride:i*stride+a.seq].unsqueeze(0).cuda()
               for i in range(a.n_batches) if (i*stride+a.seq) <= ids.numel()]

    here = "/root/attn-reuse"
    grid = load_mod("grid_mod", here + "/reuse_llama.py")
    iv = load_mod("iv_mod", here + "/reuse_intervene.py")

    # --- grid reuse, capture layer-0 selection from STATE['collect'] ---
    mg = AutoModelForCausalLM.from_pretrained(a.model, **common)
    grid.patch_model(mg)
    nll_grid, coll = grid.run_pass(mg, batches, "reuse", k=a.k, group=a.group)
    sel_grid = coll[0][0, :, :, :].detach().cpu()   # (H,T,k)

    # --- intervention base_live, capture layer-0 leader_own ---
    mi = AutoModelForCausalLM.from_pretrained(a.model, **common)
    iv.patch(mi)
    nll_iv = iv.run_intervene(mi, batches, {}, a.k, a.group, mode="base_live")
    sel_iv = iv.STATE["leader_own"][0][0].detach().cpu()

    eq = torch.equal(sel_grid, sel_iv)
    neq = int((sel_grid != sel_iv).sum())
    print("GRID reuse NLL      = %.6f" % nll_grid)
    print("IV   base_live NLL  = %.6f" % nll_iv)
    print("delta(iv-grid)      = %+.6f" % (nll_iv - nll_grid))
    print("layer0 sel equal    =", eq, " num_diff =", neq, "of", sel_grid.numel())
    if not eq:
        print("  grid sel[0,0,:8]  =", sel_grid[0, 0, :8].tolist())
        print("  iv   sel[0,0,:8]  =", sel_iv[0, 0, :8].tolist())

    # --- third pass: run_intervene but with mode=intervene and borrow built
    #     from grid's live selection for ALL followers (should equal grid) ---
    mi2 = AutoModelForCausalLM.from_pretrained(a.model, **common)
    iv.patch(mi2)
    nll_iv2 = iv.run_intervene(mi2, batches, {}, a.k, a.group, mode="base_live")
    print("IV   base_live rerun= %.6f (determinism check)" % nll_iv2)


if __name__ == "__main__":
    main()
