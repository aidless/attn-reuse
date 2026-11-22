# Software-stack invariance: the 7B measurement is not a version artifact

Round17. A reviewer can ask whether the layer-routing hook depends on a
particular transformers internals (the paper was measured on 5.23.0). This is a
**single-variable control**: we rerun the paper's 7B grid on a fresh stack with
a different transformers version and check the numbers reproduce bit-for-bit.

## What changed and what did not

| | paper run | independent run |
|---|---|---|
| transformers | 5.23.0 | **4.57.6** |
| torch | 2.9.1+rocm6.4 | 2.9.1+rocm7.2.1 |
| hardware | AMD Radeon Cloud | AMD Radeon Cloud (new instance) |
| model | Qwen2.5-7B (base) | Qwen2.5-7B (base) -- identical weights |
| corpus | wikitext-2-raw-v1 test | wikitext-2-raw-v1 test (1,289,979 chars) |
| dtype | bfloat16 | bfloat16 |
| seq / batches / k / groups | 512 / 20 / 256 / 2,3,6 | identical |

Only the software stack (transformers + ROCm build) differs; model, corpus,
dtype, and protocol are held fixed. Corpus verified by character count against
the paper README (~1.29M chars) (source: `attn-reuse/README.md`).

## Result: bit-for-bit identical

| quantity | paper (5.23.0) | new stack (4.57.6) | abs diff |
|---|---|---|---|
| full NLL | 2.27039399 | 2.27039399 | 0.00e+00 |
| g2 vs_full | 0.00327049 | 0.00327049 | 0.00e+00 |
| g2 vs_own | 0.00323931 | 0.00323931 | 0.00e+00 |
| g3 vs_full | 0.00952521 | 0.00952521 | 0.00e+00 |
| g3 vs_own | 0.00949402 | 0.00949402 | 0.00e+00 |
| g6 vs_full | 0.03222021 | 0.03222021 | 0.00e+00 |
| g6 vs_own | 0.03218903 | 0.03218903 | 0.00e+00 |

(source: `real/data/main_20b_7.json` vs `real/data/software_stack_7b_base_4576.json`)

**Max absolute difference = 0.00e+00 across all seven quantities.** The control
`own(k=seq)==full` also passed exactly in the new run (2.27039 vs 2.27039).

## What this establishes

The measurement is invariant to the transformers version and ROCm build: the
top-k routing hook (`real/reuse_llama.py`) computes the same selections and NLL
on 4.57.6 as on 5.23.0. This closes the "version-specific internals" attack.

## What it does not establish

This is one model (7B base) and one corpus. It says nothing about other model
families (TinyLlama was checked in the paper) or about the mechanism/deliver-mass
analysis (not rerun here). Not merged into the manuscript -- it is a robustness
certificate, not a new result.

## Artifacts and reproduction

- `real/data/software_stack_7b_base_4576.json` (base 7B, single-variable control)
- `real/data/software_stack_7b_instruct_4576.json` (Instruct 7B, exploratory;
  different weights so NOT comparable to the paper -- full=2.37014)
- weights: Qwen2.5-7B base downloaded to AMD `/root/models/Qwen2.5-7B-base`

Reproduce on a transformers 4.57.6 stack:
```
HF_HUB_OFFLINE=1 python3 real/reuse_llama.py \
  --model <Qwen2.5-7B-base> --corpus <wikitext-2-raw-v1/test> \
  --seq 512 --n-batches 20 --ks 256 --groups 2,3,6 --dtype bfloat16 \
  --out real/data/software_stack_7b_base_4576.json
```
