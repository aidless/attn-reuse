# Deliver-mass: a mechanism candidate for the leader anomaly

**Status**: positive result, observational. Two model families, both controls
survive. NOT a causal claim: deliver-mass is dense-attention (no reuse mask)
while the cost is masked-NLL, so they are related by construction. The
informative finding is that this *directional* mass measure predicts the
leader-anomaly cost where three *similarity* measures failed.

Date: 2026-11-20. Goal Round 3.

## Question

The real-weight experiment (report 06) found that borrowing the selected token
set from layers 0–1 is 7.5×–35× costlier than from layers 2+, across four
models (Qwen2.5-0.5B/1.5B/7B, TinyLlama-1.1B). That is a description of *who*
is expensive, not *why*. Three candidate explanations were all rejected:

| measure | corr with reuse cost | verdict |
|---|---|---|
| index overlap with target's own selection | −0.055 | no |
| linear CKA of attention outputs | −0.341 | no |
| contribution magnitude `‖W_U·h_l‖` | −0.071, flips to +0.604 once layer is controlled | no (layer proxy) |

## Hypothesis

A leader is expensive not when its selection is *dissimilar* to the follower's,
but when the positions it selects carry little mass under the follower's own
attention. Define, for leader `s` and follower `t` at distance `d = t − s ≤ 6`:

```
deliver_mass(t | s) = mean_t' [ Σ_{j ∈ TopK_s(t')} softmax(A_t')[j] ]
```

i.e. take the leader's selected positions per token, and read them out against
the follower's **dense** softmax (no reuse). High deliver-mass means the
leader happens to pick where the follower was already looking.

Prediction: early leaders (s ≤ 1) have lower deliver-mass, and across pairs,
lower deliver-mass predicts larger reuse cost.

## Instruments

- `real/leader_mechanism.py` — dense-attention hook collects per-layer TopK
  indices and the dense softmax row; `deliver_matrix()` gathers the leader's
  indices in the follower's row; `summarize()` does an early/late split and a
  10,000-permutation two-tailed test. `--selftest`: flat→ratio 1.0, skew→0.083.
- `real/leader_deliver_analysis.py` — GPU-free join of `mech_*.json` (deliver)
  with `pairs20_*.jsonl` (cost `vs_own`, distance). Full control battery A–H.
  `--selftest`: positive fixture forces A < −0.8; confound fixture (both
  deliver and cost scale only with layer) forces partial-by-layer < 0.3 and
  within-source ≈ 0 — i.e. the guard can go red on a layer proxy.

Protocol (source: `real/leader_mechanism.py` + `real/leader_deliver_analysis.py`): seq 512, k 256, 20 batches, max_distance 6, window 6, eager attention, seed 20261119. Identical for Qwen2.5-0.5B and TinyLlama-1.1B.

## Results

### 1. Early leaders deliver less mass

| model | n_layers | early (s≤1) | late (s≥2) | early/late | permutation p |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 0.9087 (n=12) | 0.9722 (n=111) | **0.935** | 0.00010 |
| TinyLlama-1.1B | 22 | 0.8352 (n=12) | 0.9784 (n=99) | **0.854** | 0.00010 |

Deliver-mass rises monotonically from source 0 to source 3 then plateaus
(0.897→0.970→~0.98 for 0.5B; 0.818→0.986→~0.985 for TinyLlama).

### 2. Not a shape effect

Self-mass (each layer's own TopK read against its own dense row) is flat:
0.9928–0.9996 across all 24 Qwen2.5 layers; TinyLlama L0=0.9044, L1=0.9798,
rest 0.982–0.9995. **Early layers' selections are high-mass under their own
distribution** — low deliver-mass is a genuine cross-layer mismatch, not early
attention being diffuse.

### 3. Predicts reuse cost at pair level

| control | Qwen2.5-0.5B (n=123) | TinyLlama (n=111) |
|---|---|---|
| A. corr(deliver, vs_own) | **−0.5124** (p=1.4e−9) | **−0.6912** (p=4.6e−17) |
| B. corr(deliver, source layer) | +0.300 | +0.615 |
| C. partial by layer | **−0.4679** (p=4.8e−8) | **−0.6290** (p=1.4e−13) |
| D. partial by distance | −0.5233 | −0.6878 |
| E. late-only (s≥2) | −0.2957 (n=111) | −0.7108 (n=99) |
| F. shuffled deliver (negative control) | −0.0911 (p=0.32) | −0.0431 (p=0.65) |
| G. within-source mean r | −0.312 (20 sources) | −0.236 (18 sources) |

Permutation p for A (10,000 label shuffles, two-tailed): 0.00010 both models.

Every distance stratum is negative (0.5B −0.53…−0.77; TinyLlama −0.49…−0.92).

## Reading the controls

- **B vs C**: deliver-mass *does* correlate with layer number (+0.30/+0.615),
  but far more weakly than the rejected contribution magnitude (log c vs layer
  r = +0.895). After partialing out layer, C stays strongly negative
  (−0.47/−0.63), whereas the contribution magnitude *flipped positive*
  (+0.604) under the same control. This is the decisive difference.
- **E (late-only)**: inside layers 2+ the effect stays for TinyLlama (−0.71)
  but weakens for 0.5B (−0.30). The leader anomaly is mostly a 0/1-vs-rest
  contrast, so a weaker within-plateau slope is expected; report it plainly.
- **F (shuffled)**: negative control is near zero and non-significant for both
  models — the battery is not manufacturing correlation.
- **G (within-source)**: moderate (−0.31/−0.24). The measure is stronger at
  separating *leaders* than at ranking followers for a fixed leader, which is
  exactly the granularity of the phenomenon it is meant to explain.
- **Distance**: effect holds at every distance 1–6, so it is not driven by
  adjacent-layer proximity alone.

## Honest limitations

1. **Related by construction.** deliver-mass reads the follower's dense row at
   the leader's positions; reuse cost is the NLL after masking to those
   positions. They share the follower's q/k. The claim is *predictive account*,
   not mechanism-in-the-causal sense. An intervention that changes deliver-mass
   while holding layer/distance fixed (e.g. steering the leader's selection
   toward high-follower-mass positions) would be needed for causality.
2. **Two families, one protocol.** Only 0.5B and TinyLlama have dense hooks
   dumped. Extending to 1.5B/7B is cheap (no new training) but not yet done.
3. **Small n.** 123/111 pairs; permutation p is floored at 1/10001 by design.
4. **fp16 entropy bug fixed in a sibling probe.** `leader_anomaly.py:75`
   `clamp_min(1e-12)` is a no-op in fp16 and produced all-NaN entropy; that
   probe's other columns were flat and are not used here. This probe uses
   mass (softmax already normalised), so it is unaffected.

## Decision

Promote to a labelled mechanism-candidate section in `paper/main.tex`: it
fills the paper's weakest point (a "who", with no "why"). Keep the language
observational ("predicts", "is consistent with"), carry the shuffled negative
control and the late-only weakening into the paper, and pin every number to
`data/deliver_corr.json` through `audit_numbers.py`.

Artifacts: `real/data/mech_05.json`, `real/data/mech_tiny.json`,
`real/data/deliver_corr.json`. Scripts: `real/leader_mechanism.py`,
`real/leader_deliver_analysis.py`. Both `--selftest` pass; `pw` ALLOW.
