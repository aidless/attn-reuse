# Independent hardware + corpus recompute (local RTX 3060)

Round16. AMD Radeon Cloud (`<cloud-host>`, used for every real-model
run in Rounds 2-12) went down (`Connection refused`, probed 2026-11-20). The
local machine's GPU -- long assumed unavailable in stale notes -- was confirmed
working: RTX 3060 Laptop / 6144 MiB, `torch 2.14.1+cu130`,
`cuda.is_available()=True`, fp16 2048x2048 matmul in 0.22 s. A complete
Qwen2.5-0.5B weight (943 MB) was downloaded from hf-mirror into `/tmp/qwen05_full`.

This is a **robustness side-check, not a manuscript result.** It deliberately
changes two things the paper held fixed -- hardware (local RTX 3060 vs AMD) and
corpus (a textbook vs WikiText-2) -- so the NLL levels are not comparable to the
paper's tables and no number here is pinned into `main.tex` or the audit.

## Setup

- model: Qwen2.5-0.5B (24 layers), FP16, eager attention, `HF_HUB_OFFLINE=1`
- local script `real/reuse_llama.py --selftest` passed before the run
- control `own(k=seq)` == dense exactly (|diff| < 1e-6) in both runs
- grid: seq 512, k 256, groups 2/3/6, 20 batches (`data/local_gpu_recompute_05_20b.json`)
- per-pair: seq 512, k 256, window 6, 4 batches (`data/pairs_local_05.jsonl`, 123 pairs)

## Result 1: reuse penalty direction reproduces on new hardware + corpus

vs_full NLL delta, k=256 (AMD bfloat16/WikiText vs local float16/textbook):

| cell | AMD (+0.00x nats) | local RTX 3060 | same sign |
|---|---|---|---|
| g=2 | +0.02381 | +0.01834 | yes |
| g=3 | +0.01105 | +0.00876 | yes |
| g=6 | +0.07317 | +0.04385 | yes |

All three group sizes stay positive: borrowing a group's routing costs NLL on
new hardware and a different corpus. The local magnitudes are smaller -- but
the two runs differ in dtype, corpus, and corpus length, so the gap is not
interpretable as an effect difference and is not claimed as one.

## Result 2: the leader anomaly reproduces (weaker, as expected at 4 batches)

Per-pair, source<=1 (n=12) mean vs_own +0.002224 vs source>=2 (n=111) +0.000489
= **4.54x** (source: `real/data/pairs_local_05.jsonl`); two-tailed label permutation (10k, seed 20261119) p=0.0096 (source: `real/sweep2.py`).
By source, the first two layers are the costliest (0: +0.002607, 1: +0.001841).

The paper's AMD 0.5B figure is 11.93x, but that is a 12-batch WikiText run;
this is 4 batches on a textbook. The paper already records that 4-batch
per-pair figures are a sample-size control and not a reported result (the
earlier 4-batch table showed flip-prone signs that stabilized at 20 batches).
So 4.54x is read only as **direction and significance agree**, not as a
magnitude to compare.

## What this does and does not establish

- Establishes: the group-reuse penalty and the first-layers-as-leader anomaly
  are reproducible on a second, independent hardware+corpus stack (control
  passes, all group cells positive, leader effect p<0.01).
- Does not establish: any new magnitude, any new model scale, or anything about
  the deliver-mass mechanism (not probed here). Not merged into the paper.

## Artifacts

- `real/data/local_gpu_recompute_05_20b.json` (grid)
- `real/data/pairs_local_05.jsonl` (per-pair)
- weights are in `/tmp/qwen05_full` (not committed; re-download from hf-mirror
  `Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors` if regenerating)

Reproduce:
```
HF_HUB_OFFLINE=1 python3 real/reuse_llama.py --model /tmp/qwen05_full \
  --corpus <textbook.txt> --seq 512 --n-batches 20 --ks 256 --groups 2,3,6 \
  --dtype float16 --out real/data/local_gpu_recompute_05_20b.json
HF_HUB_OFFLINE=1 python3 real/sweep2.py --mode pairs --k 256 --window 6 \
  --seq 512 --n-batches 4 --model /tmp/qwen05_full --corpus <textbook.txt> \
  --dtype float16 --out real/data/pairs_local_05.jsonl
```
