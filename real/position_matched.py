"""Re-measure the dense reuse cost on the rows that can actually discriminate.

At seq 512 with k=256, a query at position q sees q+1 keys. For q+1 <= 256 the
top-256 set IS every visible key, so the borrowed set and the own set coincide
and that row contributes zero to the averaged delta -- yet it still counts in
the denominator. Half the sequence is inert by construction, in both the dense
grid (512/256) and the trained-sparse grid (4096/2048).

This asks whether the reported +0.0242 / +0.0114 / +0.0736 would change if the
average were taken over discriminating positions only. It does not claim those
numbers are wrong: they are defined over all scoring positions, as the paper
says. It measures how much of them is dilution.
"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reuse_llama as RL
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/tmp/qwen05_full")
ap.add_argument("--corpus", default="morphagent-textbook/Book.txt")
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--n-batches", type=int, default=20)
ap.add_argument("--ks", default="256")
ap.add_argument("--groups", default="2,3,6")
ap.add_argument("--dtype", default="float16")
ap.add_argument("--out", default="/tmp/r20/position_matched.json")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype),
        low_cpu_mem_usage=True, attn_implementation="eager").cuda().eval()
RL.patch_model(model)

ids = tok(open(a.corpus, encoding="utf-8", errors="ignore").read(), return_tensors="pt").input_ids[0]
if ids.shape[0] < a.seq * a.n_batches:
    raise SystemExit("corpus too short: %d tokens for %d batches of %d"
                     % (ids.shape[0], a.n_batches, a.seq))
stride = a.seq * 7
batches = []
for i in range(a.n_batches):
    off = i * stride
    seg = ids[off:off + a.seq]
    if seg.shape[0] == a.seq:
        batches.append(seg.unsqueeze(0).cuda())
print("model", a.model, "batches", len(batches), "seq", a.seq, "dtype", a.dtype, flush=True)


def per_row_nll(bb, mode, k=None, group=None):
    """Cross-entropy per scoring position, not averaged.

    The forward pass is untouched; we re-derive the per-token loss from the
    logits so the same pass yields both the all-row and the discriminating-row
    means. Loss at position t predicts token t+1, so the row that *made* the
    prediction for target t is query position t-1.
    """
    RL.STATE.update({"mode": mode, "k": k, "group": group or 1,
                     "leader_sel": {}, "collect": {}, "rand": {}})
    hs = RL._hooks(model, group or 1)
    with torch.no_grad():
        out = model(input_ids=bb)
    for h in hs:
        h.remove()
    logits = out.logits[0, :-1]                      # predicts tokens 1..T-1
    tgt = bb[0, 1:]
    nll = torch.nn.functional.cross_entropy(logits.float(), tgt, reduction="none")
    return nll.detach()


ks = [int(x) for x in a.ks.split(",")]
groups = [int(x) for x in a.groups.split(",")]

allrows, active = {}, {}
base_all = []
base_by_k = {k: [] for k in ks}          # the baseline must be cut per k, not once
for bb in batches:
    n = per_row_nll(bb, "full")
    base_all.append(n.mean().item())
    for k in ks:
        keep = torch.arange(n.shape[0], device=n.device) + 1 > k
        base_by_k[k].append(n[keep].mean().item())
mean_base_all = sum(base_all) / len(base_all)
mean_base_by_k = {k: sum(v) / len(v) for k, v in base_by_k.items()}

for k in ks:
    for g in groups:
        ra, ac = [], []
        for bb in batches:
            n = per_row_nll(bb, "reuse", k=k, group=g)
            ra.append(n.mean().item())
            # the threshold is this cell's own k; using ks[0] for every cell was
            # the bug that produced nonsense multipliers (0.87x, 19.05x, -0.91x)
            keep = torch.arange(n.shape[0], device=n.device) + 1 > k
            ac.append(n[keep].mean().item())
        key = "k%d_g%d" % (k, g)
        allrows[key] = sum(ra) / len(ra) - mean_base_all
        active[key] = sum(ac) / len(ac) - mean_base_by_k[k]
        print("%-8s all_rows %+0.5f  active_rows %+0.5f  multiplier %.3fx"
              % (key, allrows[key], active[key],
                 allrows[key] / active[key] if active[key] else float("nan")), flush=True)

res = {"model": a.model, "corpus": a.corpus, "seq": a.seq, "n_batches": len(batches),
       "dtype": a.dtype, "k": ks, "groups": groups,
       "base_all": mean_base_all, "base_active_by_k": mean_base_by_k,
       "deltas_all_rows": allrows, "deltas_active_rows": active,
       "active_row_share_by_k": {str(k): (a.seq - 1 - k) / (a.seq - 1) for k in ks},
       "expected_multiplier_by_k": {str(k): (a.seq - 1) / (a.seq - 1 - k) for k in ks}}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print("WROTE", a.out)
