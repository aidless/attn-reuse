# Does layer-specific scoring buy anything?

`fxmeng/DSV2-Lite-LatentIndex-Sparse-1250` (step 1250, 27 layers, `FSS` route
pattern repeated nine times, `index_topk` 2048). The checkpoint's selling
point -- and the selling point of the paper it comes from
(arXiv:2610.04635) -- is that followers do **not** borrow the leader's score.
The leader shares only its latent cache; each follower reads its own offset
out of the leader's route list (`route = group[self.offset]`) and therefore
scores the same tokens with its own projection.

That gives a within-checkpoint comparison that no earlier round could make:

    offset0   follower scores with the leader's projection (IndexCache-like)
    own       follower scores with its own projection (published behaviour)

and then, on top of each, substitute the selected set with a random set of
the same size. `null_own / null_off0` is what layer-specific scoring buys.

## Result: nothing measurable

One run, three batches, `seq` 4096, bfloat16, deterministic algorithms on,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Repeat of the base pass reproduced it to
`0.000e+00`.

| cell | mean dNLL |
|---|---|
| `own` (control) | `+0.000000` |
| `offset0` | `+0.000105` |
| `null_own` | `+0.069073` |
| `null_off0` | `+0.069093` |

- `null_own / null_off0` = **0.99959** (mean of per-batch ratios) and
  **0.99970** (ratio of means). Both essentially 1.
- `offset0 - own` = **+0.000105**, CI `[-0.001347, +0.001556]` **crossing
  zero**, per-batch `+0.000674 / +0.000133 / -0.000493` -- not even
  consistently positive.
- The cost of substituting the set is **0.984 x** the IndexCache cost
  (`+0.069073` vs `+0.070199`). This comparison is only defensible because
  the two checkpoints are the same architecture, the same base weights, the
  same protocol and the same three batches, and their base NLLs agree to
  `1.9958` vs `1.9965`. Different training runs, so different weights -- but
  the controlled difference is the scoring rule, not the family.

## The intervention did bite -- this is not a dead probe

The reading "no effect" and the reading "probe never reached the model" fit
the same number, so the probe has to be checked separately. Forcing every
follower to `offset0` and re-capturing its selected indices:

| chunks | Jaccard(own, offset0) | identical |
|---|---|---|
| 306 live (`first query sees > 2048`) | **0.7799** (median 0.7878, min 0.4322) | **0.0%** |
| 288 inert | 1.0000 | 97.6% |

Every live chunk changed; not one was left identical. Inert chunks stayed
identical at 97.6% -- which is exactly what they should do, because their
`scores` are all `-inf` and top-`k` is a tie-break there, not a decision.
That split is also why the "is capture repeatable?" field reads `False` when
computed over all chunks and `True` when restricted to live ones: the
non-repeatability lives entirely in the tie-break region.

So: **the sets really do differ, and the cost really is the same.** That is a
statement about the quantity, not about the instrument.

## What this does and does not establish

Establishes: on this checkpoint, at this keep fraction and sequence length,
giving each follower its own score changes which tokens it selects
(Jaccard 0.78, i.e. roughly a fifth of each set differs) but does not change
the NLL cost of replacing the set. The claimed recovery of attention-mass
recall does not show up as a decision-level saving.

Does not establish: that layer-specific scoring is useless. Attention-mass
recall is a property of the selected set measured against the dense
attention; NLL after substitution is a different quantity. A method can
recover recall and still not move NLL. What this measurement rules out is the
specific inference "better recall therefore lower substitution cost" -- this
project has been rejected once for exactly that shape of inference.

Also does not establish: anything across keep fractions. `index_topk` is
2048 of 4096 here, the same 50.0% fraction at which the dense row-dilution
work applies, and only that fraction was measured.

## Reproduction

    scp real/patch_li.py real/li_run.py real/li_route_check.py <host>:/tmp/
    /opt/venv/bin/python /tmp/li_run.py --model ... --gpu-gib 30GiB
    /opt/venv/bin/python /tmp/li_route_check.py --model ... --gpu-gib 30GiB
    python3 real/li_analysis.py --selftest
    python3 real/li_analysis.py

Artifacts: `real/data/li_run.json`, `real/data/li_route_check.json`,
`real/data/li_analysis.json`.

## Three things the instrument took

1. **`torch.roll` cannot change a boolean mask.** An earlier round spent a
   probe on reordering indices; `scatter_` turns them into a mask and the
   mask does not see the order. Change the set, not the list.
2. **`transformers` imports a copy, not the file you patched.** Patching
   `models/<ckpt>/modeling_dots_dsa.py` prints "patched" and then the stack
   trace points at `~/.cache/huggingface/modules/transformers_modules/...`.
   The run now purges that cache before loading and then *asserts* that the
   module actually imported contains the hook markers -- the assertion is
   what makes the failure visible instead of silent.
3. **A comparator that compares x to itself is a tautology.** The first
   version of `same_set` did `x.sort() == x.sort()`, so every repeat looked
   identical; it also returned `None` on a shape mismatch, which `all()` reads
   as `False`, folding "cannot compare" into "compared and differed". Both are
   fixed, and the function is now checked with two different inputs before it
   is used.
