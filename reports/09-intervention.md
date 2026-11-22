# Steering the shared index changes the cost: an intervention

**Status**: positive result, interventional (ORACLE sufficiency). Pre-registered
before any run in `prereg/intervention-prereg.md` (FROZEN, 61 lines). Mechanical
verdict `SUPPORTED`: in both model families and all three group widths, moving
the borrowed index toward positions of high follower mass *lowers* NLL, and
moving it toward low-mass positions *raises* it, with every control passing.
This upgrades the Round-3 deliver-mass finding from *predicts* to *intervenes*.

Date: 2026-11-20. Goal Round 4.

## Question

Round 3 showed that the leader anomaly is explained by a directional quantity,
deliver-mass: the positions an early leader selects carry little mass under its
followers' attention. But that was observational — deliver-mass is read on dense
attention, cost on masked attention, and both share the follower's q/k. A
reviewer's obvious objection: *correlation is not control — if you move the
index, does the cost move?* This round answers that with a two-pass ORACLE
intervention.

## Pre-registered design (verbatim from the frozen prereg)

H1: raising the follower-mass of the borrowed index lowers NLL; lowering it
raises NLL. Two passes, both with each layer's own q/k/v untouched:
- **Pass A (collect)**: run dense attention, record `mass[l][h,q,p]` per layer.
- **Pass B (intervene)**: change *only* the borrowed index that followers use;
  leaders keep their own top-k. Three conditions:
  - **BASE** — followers borrow the leader's own top-k (the status quo).
  - **HIGH** — followers borrow the top-k of the *followers' consensus* mass
    (positions high under the followers' own dense rows).
  - **LOW** — followers borrow the bottom-k (negative control).
Decision rule: `SUPPORTED` iff both models show `NLL(HIGH) < NLL(BASE) <
NLL(LOW)` in every group, with the manipulation check and both negative
controls passing; otherwise `PARTIAL` / `REFUTED` / `INCONCLUSIVE`.

## Instruments

- `real/reuse_intervene.py` — two-pass ORACLE. `collect_dense` records dense
  mass; `build_borrow(direction=high|low)` selects consensus top/bottom-k with
  causal clamping `p≤q`; `run_intervene` swaps only the borrowed index;
  `measured_deliver` reads delivered mass with `scatter_` (no double-counting).
  `--selftest` (source: `real/reuse_intervene.py`): synthetic even/odd peaks give
  `d_high=0.7427 > d_base=0.3624 > d_low=0.2443`, and no causal violation.
- `real/intervention_verdict.py` — applies the frozen decision rule mechanically.
  `--selftest` (source: `real/intervention_verdict.py`): SUPPORTED / PARTIAL /
  REFUTED / three INCONCLUSIVE fixtures all fire, so the gate can go red.

Protocol (source: `real/reuse_intervene.py` + `real/data/intervene_05.json` +
`real/data/intervene_tiny.json`): seq 512, k 256, 20 batches, groups 2/3/6,
eager fp16, HF_HUB_OFFLINE=1, identical for Qwen2.5-0.5B and TinyLlama-1.1B.

## Results

### 1. Ordering holds in every cell

NLL (lower is better); h/b/l = HIGH / BASE / LOW.
Source: `real/data/intervene_05.json`, `real/data/intervene_tiny.json`.

| model | group | HIGH | BASE | LOW | order |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 2 | 2.82444 | 2.87525 | 4.26165 | HIGH<BASE<LOW |
| Qwen2.5-0.5B | 3 | 2.82434 | 2.83650 | 5.03828 | HIGH<BASE<LOW |
| Qwen2.5-0.5B | 6 | 2.82418 | 2.90997 | 5.22306 | HIGH<BASE<LOW |
| TinyLlama-1.1B | 2 | 2.42572 | 2.43887 | 4.77614 | HIGH<BASE<LOW |
| TinyLlama-1.1B | 3 | 2.42590 | 2.43960 | 5.34296 | HIGH<BASE<LOW |
| TinyLlama-1.1B | 6 | 2.42644 | 2.85459 | 5.30630 | HIGH<BASE<LOW |

Delivered mass moves in the same direction (source: same two JSONs): HIGH
0.980–0.986, BASE 0.935–0.967, LOW 0.556–0.577 across all six cells.

### 2. HIGH nearly recovers the no-reuse baseline

Full (no reuse) NLL is 2.82451 on 0.5B and 2.42524 on TinyLlama
(source: `real/data/intervene_05.json`, `real/data/intervene_tiny.json`). The
HIGH condition lands at 2.82418–2.82444 and 2.42572–2.42644 respectively —
within 0.0003 / 0.0012 of full. Redirecting the borrowed index to positions the
followers already attend to removes essentially all of the reuse penalty.
LOW instead inflates NLL by 1.39–2.31 (0.5B) and 2.34–2.90 (TinyLlama) nats.

### 3. Both negative controls pass

- Harness control: `own(k=seq)` with an empty borrow dict reproduces full NLL
  to `absdiff = 0.00e+00` on both models (source: `real/data/intervene_05.json`,
  `real/data/intervene_tiny.json`, key `harness_abs_diff`).
- LOW distinct from BASE: overlap `0.251` in all six cells, below the 0.9
  threshold (source: same JSONs, key `overlap_low_base`).
- Manipulation check: delivered mass ordered HIGH>BASE>LOW in all six cells
  (source: same JSONs, key `manipulation_check`).

### 4. Mechanical verdict

`python3 real/intervention_verdict.py --inputs intervene_05.json
intervene_tiny.json` → **SUPPORTED**, all six (model, group) cells SUPPORTED
(source: `real/data/intervention_verdict.json`).

## What this does and does not establish

- **Establishes sufficiency**: changing the follower-mass of the borrowed index,
  holding layer/distance/own-q-k fixed, is enough to move the cost in the
  predicted direction. It rules out "deliver-mass is merely a readout of depth".
- **Does not give a deployable remedy**: the HIGH guide set uses follower *dense*
  attention, unavailable in a causal decoder before the fact. This is a
  proof-of-lever, not a fast index-selection algorithm. The paper says
  "sufficient to change the cost", never "a method to fix it".
- **Scope**: two families, one protocol, ORACLE upper bound. Extending to
  1.5B/7B is cheap but not required by the prereg's decision rule.

## Artifacts

Data: `real/data/intervene_05.json` (sha256 `05ea2aff1e76…`),
`real/data/intervene_tiny.json` (sha256 `9a47b30e479a…`),
`real/data/intervention_verdict.json`. Scripts: `real/reuse_intervene.py`,
`real/intervention_verdict.py`. Prereg: `prereg/intervention-prereg.md`.
Both scripts' `--selftest` pass; `pw` ALLOW.

## Revision after adversarial review (Round 5, 2026-11-20)

This section records what the review changed about the claims above. The
measurements in the body of this report remain as run; the interpretation
does not survive unchanged.

- **BASE was not the grid's condition.** A same-batch diagnostic
  (`real/diag_base.py`) shows the v1 BASE index is not bit-identical to the grid
  reuse condition: grid NLL `2.852547` versus intervention BASE `2.873979`
  (delta `+0.021432`), with layer-0 selection differing at `1352658/1835008`
  (`73.7%`) positions. The v1 run was fp16 while the grid is bfloat16, so the
  "HIGH lands within 0.0012 of full" comparison crossed both a precision and a
  construction boundary. The absolute magnitude is withdrawn.
- **v2 rerun (bfloat16, live BASE).** `real/data/intervene_05_v2.json`, full
  `2.82874` (bit-identical to the grid's dense reading), BASE NLL
  `2.874`/`2.874`/`2.911` and LOW `4.26`--`5.23` over groups 2/3/6. Ordering
  HIGH$<$BASE$<$LOW holds in every cell and both harness controls pass
  (`source: real/reuse_intervene.py`, `--dtype bfloat16`).
- **Ordering is partly an artifact of construction.** HIGH and LOW are the top
  and bottom of the same follower dense distribution that defines deliver-mass,
  so the bracket is guaranteed by how the arms are built. What the check shows
  is that the borrowed index is a live lever on cost, not that deliver-mass
  identifies a mechanism, and not that some deployable rule achieves it: the
  guide set is dense attention, unavailable in an autoregressive decoder.
- **Role in the paper.** Down from "establishes sufficiency of the mass
  account" to "a construct check". The mechanism section is retitled
  "A predictive diagnostic, and a construct check", and the Limitations section
  carries the four caveats explicitly.
