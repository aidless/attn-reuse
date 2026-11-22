# How much of the leader anomaly does deliver-mass explain? (rev-da I6)

## The objection

Round 5 adversarial review, rev-da finding I6: the selection *shape* is nearly
identical across layers — top-k mass 0.9928–0.9995, local share ≈0.279,
position-0 share ≈0.003 (`real/data/leader_512.json`) — yet early leaders'
deliver-mass sits only ~6 points below late leaders' (0.9087 vs 0.9722 on 0.5B),
while their reuse cost is 7.5–35× higher. A small mass gap driving a large cost
gap is an extreme nonlinearity the paper never explained. If deliver-mass were
the mechanism, moving it by the observed amount should move the cost by the
observed amount. This report tests that quantitatively.

## Method (source: `real/leader_magnitude.py`, GPU-free)

Join the same artifacts as the correlation analyses (`mech_*.json` ×
`pairs20_*.jsonl`). Split pairs into early leaders (source ≤ 1) and late leaders
(source ≥ 2). Fit OLS `cost ~ deliver` on the **late group only** — the regime
where reuse is cheap and deliver varies — then extrapolate that slope to the
early group's mean deliver. The fraction of the actual early-minus-late cost gap
that this predicts is `explained`; the rest is `residual`. This is deliberately
generous to deliver: it assumes the late-group linear response holds unchanged in
the early regime. We also report `support`, the fraction of early pairs whose
deliver lies inside the late group's range; when support is low the extrapolation
is out-of-sample and `explained` is only an upper bound on deliver's reach.

The selftest pins three fixtures: a law identical in both groups must give
explained = 1.0000; a cost that depends only on group identity must give 0.0000;
a linear-plus-offset law must match its analytic fraction (0.5789). All pass, so
the instrument can go red.

## Result (source: `real/data/leader_magnitude.json`, sha256 `dde880e3…238f`)

| scale | n early/late | deliver gap | cost ratio | late slope | explained | residual | early-in-late-support |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 12/111 | 0.0636 | 7.5× | −0.0264 | 0.323 | 0.677 | 33% |
| TinyLlama-1.1B | 12/99 | 0.1432 | 28.7× | −0.2067 | 0.599 | 0.401 | 8% |
| Qwen2.5-1.5B | 12/135 | 0.0881 | 35.0× | −0.0191 | 0.283 | 0.717 | 0% |
| Qwen2.5-7B | 12/135 | 0.0639 | 11.8× | −0.0364 | 0.375 | 0.625 | 42% |

Every cell is out-of-support (< 50%): at 1.5B the two groups do not overlap in
deliver at all (early max 0.9242 < late min 0.9379), so no matched comparison is
possible there and deliver is almost collinear with the binary leader flag.

## Reading

Even under the generous linear extrapolation, deliver-mass accounts for only
28–60% of the leader cost gap; 40–72% is left unexplained. Combined with the
near-zero common support, this says deliver-mass is a **correlated diagnostic,
not a sufficient account** of why the first layers are expensive. The bulk of the
anomaly lives in something other than the delivered mass — plausibly the early
layers' role in the residual stream or position handling, which this measurement
does not isolate. This is consistent with, and sharpens, the Round 5 downgrade:
we make no claim that deliver-mass identifies the mechanism, and now we quantify
how far short of the magnitude it falls.

## What this does not establish

- It is not a causal decomposition: the extrapolation is out-of-support and
  shares the follower's q/k with the masked-NLL cost.
- It does not identify what the residual 40–72% *is*; it only bounds what
  deliver-mass can explain. Testing position-encoding ablations or matched-deliver
  subsamples would be the next step, and neither is done here.