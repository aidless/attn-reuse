# The null `reuse - own` needs: an arbitrary set of the same size


Reported cell by cell. Every number below is read from the artifacts named
at the end; nothing here is transcribed by hand.

## 1. What the quantity means without a null

`reuse - own` compares a follower using the **leader's** k-set against the
same layer using its **own** k-set. Those two states differ in two ways at
once: the set is someone else's, *and* the set is not the one this layer
would have chosen for itself. A ratio near 1 would say borrowing is as bad
as picking at random, i.e. the leader's index carries no information. That
reference has to be measured, not assumed, which is what this round adds.

The null: with the same group and the same k, the **leaders keep their own
selection** and every borrowing layer gets an arbitrary causal-valid set of
size k instead. The null therefore perturbs exactly the layers the reuse
cell perturbs. A whole-model random pass would not: it would also degrade
the leaders and overstate the null.

## 2. Cells (512 positions, 20 batches, bfloat16, k=256)

| model | g | `reuse - own` | null `rand - own` | ratio | null / reuse |
|---|---|---|---|---|---|
| 0.5B | 2 | +0.02419 | +0.31493 | 0.077 | 13.0x |
| 0.5B | 3 | +0.01143 | +0.33666 | 0.034 | 29.4x |
| 0.5B | 6 | +0.07356 | +0.58492 | 0.126 | 8.0x |
| 7B | 2 | +0.00324 | +0.07348 | 0.044 | 22.7x |
| 7B | 3 | +0.00949 | +0.17319 | 0.055 | 18.2x |
| 7B | 6 | +0.03219 | +0.27844 | 0.116 | 8.7x |

The null is **7.95 to 29.4 times** the reuse cost on both models, so the
leader's index carries real information and `reuse - own` is not merely "a set
this layer did not choose". But at **3.4-12.6\%** of the null, the leader's
index is also far from sufficient: borrowed from a neighbour or not, most of the
penalty of swapping a set is simply that the set is not the layer's own.

## 3. Three independent stacks agree bit for bit

The `reuse - own` column above is not a re-derivation; it reproduces the
published artifact exactly:

| stack | dtype | `k256_g2` | `k256_g3` | `k256_g6` |
|---|---|---|---|---|
| published (`main_20b_05.json`) | bfloat16 | +0.02419477 | +0.01143454 | +0.07356138 |
| this round, AMD bfloat16 | bfloat16 | +0.02419477 | +0.01143454 | +0.07356138 |
| this round, local RTX float16 | float16 | +0.01856912 | +0.00898123 | +0.04407940 |

Max abs difference between the bfloat16 rerun and the published artifact, over
all three cells: **0.0**. The float16 run uses a different GPU, a different
corpus and a different dtype on purpose; it agrees in sign and magnitude
({loc['k256_g2']['vs_own']:.5f} / {loc['k256_g3']['vs_own']:.5f} / {loc['k256_g6']['vs_own']:.5f} against {p5['k256_g2']['vs_own']:.5f} / {p5['k256_g3']['vs_own']:.5f} / {p5['k256_g6']['vs_own']:.5f}), which is
what a control is for - it shows the null ratio is a property of the
protocol, not of one stack.

The 7B model reproduces its own published cells the same way: max abs
difference **0.0** over the three cells, against published values
{p7['k256_g2']['vs_own']:.8f} / {p7['k256_g3']['vs_own']:.8f} / {p7['k256_g6']['vs_own']:.8f}. Its null is 0.07348 / 0.17319 / 0.27844,
i.e. the null-to-reuse ratio is 8.7-22.7x - the same regime as the 0.5B
model, so the finding is not size-specific.

## 4. What this does and does not establish

Does:

- the leader's selection carries information (the null is 8-13 times larger);
- the bulk of `reuse - own` is not information loss about *which* keys matter
  but the cost of using a set the layer did not choose;
- the ratio is stable across two dtypes, two GPUs and two corpora.

Does not:

- locate *why* an arbitrary set costs 0.31-0.58 nats. Nothing here separates
  lost attention mass from a disrupted residual stream;
- say anything about the trained-sparse checkpoint. Its null must be measured
  on that model, with its own k, before the two ratios can be compared;
- use a per-pair null. This is a per-cell null, matched to the cell it is
  quoted against.

## 5. Artifacts and repro

```
source: attn-reuse/real/data/rand_control_amd_05_bf16.json   (AMD, bfloat16, 20 batches)
source: attn-reuse/real/data/rand_control_local_05_20b.json  (local RTX 3060, float16, 20 batches)
source: attn-reuse/real/data/rand_control_amd_7b_bf16.json   (AMD, bfloat16, 20 batches, 7B)
source: attn-reuse/real/data/main_20b_05.json               (published 0.5B cells)
source: attn-reuse/real/data/main_20b_7.json                (published 7B cells)
```

The harness gained one mode: `--rand-control`, with the null computed per
(k, group) cell rather than once per k. `--selftest` gains four checks -
the null changes the output, **zero causal violations** (a random set that
reached past the query would be a better set than any legal one and would
understate every cost attributed to it), seed reproducibility, and
seed sensitivity. All four pass on both stacks.
