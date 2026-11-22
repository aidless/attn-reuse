# Deliver-mass vs plain overlap: an incremental-value test

Round 6. This report answers the sharpest surviving objection from the Round 5
adversarial review (rev-da finding I1): *deliver-mass has never been compared
fairly against plain selection overlap*. If deliver-mass is merely a decorative
restatement of "the follower also looks where the leader looks", then plain
overlap predicts the reuse cost just as well, and the mass diagnostic adds
nothing.

## Design

Both quantities are read on the **same dense (unmasked) forward pass**, per
`(source s, target t)` pair, using the index sets already collected read-only by
`real/leader_mechanism.py::patch_collect` (`STATE["sel_idx"]`):

- `deliver(s->t) = mean_q sum_{p in I^s_q} P^t_q[p]` — our diagnostic, the dense
  mass the follower `t` places on the leader `s`'s top-k positions.
- `jaccard(s,t) = mean_q |I^s_q ∩ I^t_q| / |I^s_q ∪ I^t_q|` — plain overlap of the
  two top-k index sets, a mass-free set-overlap control.

`I^l_q` is layer `l`'s top-k key set for query position `q`; top-k has no
duplicate positions, so `|∩|` is the exact index-match count and `|∪| = 2k − |∩|`.
Only offsets `1 <= t−s <= 6` are kept. (source: `real/overlap_head2head.py`)

The GPU-free join then asks, on the same pair set as the masked-NLL cost
(`source: real/data/pairs20_0.5B.jsonl`, `real/data/pairs20_TinyLlama.jsonl`):

- A: `corr(deliver, vs_own)`
- B: `corr(jaccard, vs_own)`
- C: `partial(deliver, vs_own | jaccard)` — does deliver add anything beyond overlap?
- D: `partial(jaccard, vs_own | deliver)` — does overlap add anything beyond deliver?
- ΔR²: variance gained by adding each predictor to the other
  (source: `real/h2h_incremental.py`)

## Result

Both models, seq 512, k 256, 20 batches, float16 dense pass (source: `real/data/mech_h2h_05.json`,
`real/data/mech_h2h_tiny.json`, `real/data/h2h_corr.json`).

| quantity | Qwen2.5-0.5B (n=123) | TinyLlama-1.1B (n=111) |
|---|---|---|
| corr(deliver, cost) | −0.5124 | −0.6912 |
| corr(jaccard, cost) | −0.3632 | −0.3450 |
| **partial(deliver \| jaccard)** | **−0.4218** (p=1.2e−6) | **−0.6469** (p=1.7e−14) |
| partial(jaccard \| deliver) | −0.1795 (p=0.047) | +0.1379 (p=0.149) |
| R²(jaccard) | 0.1319 | 0.1190 |
| R²(deliver) | 0.2626 | 0.4777 |
| R²(both) | 0.2863 | 0.4876 |
| **ΔR²(+deliver)** | **+0.1544** | **+0.3687** |
| ΔR²(+jaccard) | +0.0237 | +0.0099 |

(source: `real/data/h2h_corr.json`)

Reading: plain overlap does carry signal (B is negative, p<1e−4 on both), so the
objection is well-founded in part. But deliver-mass keeps a strong partial
correlation after controlling jaccard (−0.42 / −0.65) and adds substantial
variance (ΔR² +0.15 / +0.37), while jaccard's partial collapses and even flips
positive on TinyLlama (p=0.149). Under this test deliver-mass is not a
decorative restatement of overlap (source: `real/data/h2h_corr.json`).

## Negative control (the gate can go red)

`real/h2h_incremental.py --selftest` builds a synthetic fixture in which deliver
is a **pure monotone function of jaccard** (i.e. literally decorative). There the
partial correlation collapses to +0.028 and ΔR² to −0.000 (source:
`real/h2h_incremental.py`, selftest output `decorative: C=0.028 dR2=-0.000`).
A positive fixture with an independent deliver component yields partial −0.989
and ΔR² +0.358. So the instrument distinguishes a real incremental signal from
a decorative one; the real-data numbers above are not an artifact of the metric.

## What this does and does not establish

- Does: on the same dense pass and pair set, deliver-mass predicts reuse cost
  beyond plain set overlap, across two model families.
- Does not: identify a mechanism. Both quantities are read on dense attention
  and share the follower's q/k with the masked-NLL cost; this is a predictive
  comparison, not a causal one (consistent with the Round 5 ruling). The
  ORACLE intervention remains a construct check, not deployable.
- Scope: per-pair level only; the continuous-layer and binary-leader confounds
  from Round 5 still apply to the raw deliver correlation, though the
  jaccard-controlled partial reported here is a distinct, additional control.
