import argparse, json, os, sys, time
import torch
sys.path.insert(0, "/root")
from transformers import AutoModelForCausalLM, AutoTokenizer
import reuse_llama as R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen2.5-14B-base")
    ap.add_argument("--corpus", default="/root/wt2_test.txt")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--gpu-gib", default="14")
    ap.add_argument("--out", default="/root/attn_newstack/pairs14b.jsonl")
    a = ap.parse_args()

    # --- resume state ---
    done, base = set(), None
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            if r.get("kind") == "pair":
                done.add((r["source"], r["target"]))
            elif r.get("kind") == "base":
                base = r
    print("resume: %d pairs already done, base=%s" % (len(done), bool(base)), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager",
        low_cpu_mem_usage=True, max_memory={0: a.gpu_gib + "GiB", "cpu": "120GiB"})
    print("load_s", round(time.time()-t0, 1), flush=True)
    R.patch_model(model)
    L = model.config.num_hidden_layers
    ids = tok(open(a.corpus).read(), return_tensors="pt").input_ids[0]
    stride = a.seq * 7
    batches = [ids[i*stride:i*stride+a.seq].unsqueeze(0).cuda() for i in range(a.n_batches)]
    batches = [b for b in batches if b.shape[1] == a.seq]
    print("layers", L, "batches", len(batches), flush=True)

    own_nll, _ = R.run_pass(model, batches, "own", k=a.k, group=1)
    print("own_k%d nll=%.8f" % (a.k, own_nll), flush=True)
    if base is not None:
        d = abs(own_nll - base["own_nll"])
        print("own vs base absdiff=%.3e %s" % (d, "OK" if d < 1e-6 else "MISMATCH-USING-BASE"), flush=True)
        own_nll = base["own_nll"]  # keep vs_own口径一致

    todo = []
    for source in range(L-1):
        for dd in range(1, a.window+1):
            target = source + dd
            if target >= L: continue
            if (source, target) in done: continue
            todo.append((source, target, dd))
    print("todo pairs", len(todo), flush=True)

    fh = open(a.out, "a")
    n = 0
    for source, target, dd in todo:
        t1 = time.time()
        nll, _ = R.run_pass(model, batches, "reuse", k=a.k, group=1, force={target: source})
        rec = {"kind":"pair","k":a.k,"target":target,"source":source,"distance":dd,
               "nll":nll,"vs_own":nll-own_nll}
        fh.write(json.dumps(rec)+"\n"); fh.flush()
        n += 1
        print("pair %d s=%d t=%d d=%d vs_own=%+.6f %.0fs" % (n, source, target, dd, nll-own_nll, time.time()-t1), flush=True)
    fh.close()
    print("DONE resume pairs", n, "total_s", round(time.time()-t0, 1), flush=True)

main()
