# Prereg: does raising deliver-mass of the SHARED index reduce reuse cost?

Status: FROZEN before any intervention run. Goal Round 4, 2026-11-20.

## Background
Round 3 found `deliver_mass(t|s)` = mass that leader s's selected positions carry
under follower t's DENSE softmax. Observational result: early leaders (s<=1)
deliver less mass (0.5B 0.9087 vs 0.9722; TinyLlama 0.8352 vs 0.9784,
permutation p=0.0001), and deliver-mass predicts reuse cost after controlling
for layer depth (partial r = -0.47 / -0.63). The paper's Discussion says
"a remedy must move the shared index toward those positions". That is a
causal claim, currently untested.

## Hypothesis (falsifiable)
H1: Manipulating the SHARED index set so that it carries MORE follower mass
LOWERS reuse NLL, and manipulating it to carry LESS follower mass RAISES reuse
NLL. If true, deliver-mass is a lever (sufficient manipulation), not merely a
 correlate.

## Design (oracle, two-pass)
The manipulation changes ONLY the index set broadcast by the group leader to
its followers. Each layer's own computation (its own q/k/v and its own top-k
mask when it is a leader) is left untouched; only what followers BORROW changes.

Pass A (dense, read-only): for each layer l collect dense attention and
aggregate to `mass[l][h,q,p]` = attention mass at key position p.

For group with leader s and window W (followers s+1..s+W-1), build the
consensus mass `C[h,q,p] = mean_{t in followers} mass[t][h,q,p]`.

Conditions (same model, same batches, same k, same group layout):
- BASE: followers borrow leader s's own top-k (current reuse behavior).
- HIGH: followers borrow `topk_p C[h,q,p]` (the positions followers already
  look at -> high deliver-mass).
- LOW (negative control): followers borrow the k lowest-consensus positions
  among p<=q (same k, same group, opposite manipulation).

Manipulation check (must pass or the run is inconclusive): measured
deliver-mass HIGH > BASE > LOW on the same (source,target) pairs.

## Predicted ordering
NLL(HIGH) < NLL(BASE) < NLL(LOW).

## Decision rule (set before running)
- SUPPORTED: both models show HIGH < BASE < LOW AND manipulation check passes.
- PARTIAL: ordering holds in one model only, or only HIGH<BASE (LOW flat).
- REFUTED: ordering reversed or flat in both models with check passing.
- INCONCLUSIVE: manipulation check fails, or harness control (own(k=seq)==full)
  fails.

## Scope / caveat (declared in advance)
The guided set is built from follower DENSE attention, which is not available
before the followers run in a causal decoder. This is therefore an ORACLE
upper-bound / proof-of-lever, NOT a deployable remedy. The claim being tested
is sufficiency ("if the shared index carries follower mass, cost drops"),
not a practical routing algorithm.

## Negative controls
- own(k=seq) must equal full to 1e-7 (harness adds no error).
- LOW must not accidentally equal BASE (check set overlap < k).
- Seeds/batches identical across all three conditions.
