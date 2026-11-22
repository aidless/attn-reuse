# Sharing a Route Is Not Free — cross-layer top-$k$ reuse cost

Code and stored artifacts behind the paper **"Sharing a Route Is Not Free: Measuring the
Cost of Cross-Layer Top-$k$ Reuse in Real Weights"** (TMLR-style submission; the manuscript
in `paper/` is the anonymized review build).

> One sentence: neighbouring layers *do* pick nearly the same tokens, but picking the same
> tokens is not free — the cost of borrowing another layer's top-$k$ set is a per-pair
> quantity that concentrates on the first two layers, is invisible to every similarity-shaped
> predictor we tried, and is small at the decision level once the set is wide enough.

## Question

Families of sparse attention let one layer reuse the token selection another layer computed.
That rests on an empirical claim about **selection** (adjacent layers choose nearly the same
tokens) and a silent assumption about **cost** (reusing the set is therefore harmless).
`IndexCache` measures quality at benchmark granularity on *trained sparse* models; this repo
isolates the selection-set variable on **dense pretrained** weights, where the set can be
swapped without retraining.

Premise, verified on real weights: overlap is **1.19×** chance at 512 tokens / $k{=}128$ and
**1.96×** at 2048, and it does not decay with depth (layer 8–9, 16–17, 24–25 all enriched).

## Harness: three readouts, one identity

Every protocol runs the same patched attention pass three ways; per-layer $q,k,v$, batch,
positions and the token sequence are pinned, so the only thing that moves is **which** tokens
a layer attends to:

| readout | attended set |
|---|---|
| `dense` | all positions (no top-$k$) |
| `own` | this layer's own top-$k$ |
| `reuse` | the group leader's top-$k$ |

Design sweep is depth × group leader × $k$ → 25 ARC cells.

The harness **refuses to report** unless per-layer selection at $k = $ sequence length
reproduces the dense pass. That control is not decoration: it caught a real defect (a missing
causal mask) that would otherwise have entered the tables as a "reuse cost".

An identity the tables are consistent with, over 40 cells with residual `0.000`:

```
b - c = N * (own - reuse)
```

## Readouts

| # | quantity | value | artifact |
|---|---|---|---|
| 1 | mean cost per **borrowed pair**, source = layer 0/1 vs ≥ 2 | **11.9×** | `paper/main.tex` (20 batches × 512 positions) |
| 2 | whole-model mean ratio, layer 0/1 vs deep — 0.5B / 1.5B | **13.98× / 23.66×** | `real/data/`, `reports/06-reuse-real-cost.md` |
| 3 | cost when groups are shifted by one layer | **2–3.4× lower** at 3 of 4 widths (`0.19–0.51×`) | `real/data/` |
| 4 | cost growth, 512 → 2048 positions | **3.6–21.6×** | `reports/02-k-sweep.md` |
| 5 | cost vs an arbitrary $k$-set of the same size | **0.034–0.126** of that null | `real/data/` |
| 6 | prediction from selection overlap — 23 Qwen2.5-0.5B pairs | $r=-0.357$ | `data/cka_vs_reuse.json` |
| 7 | prediction from CKA / weight-level contribution | $r=-0.341$ / $r=-0.071$ | `data/cka_corr.json`, `data/contribution.json` |
| 8 | decision level, ARC-Easy 0.5B, 25 cells | ≤ **25.5** points, five cells at $p<0.05$ | `real/data/arc_easy_*.json` |
| 9 | same, at $k{=}256$ (67% of 384 positions) | largest difference **1** point | `real/data/arc_easy_*.json` |
| 10 | asymmetry, own vs reuse per hundred at-risk (Easy / Challenge) | **10.1 vs 7.9** / **12.4 vs 6.9** (1.3–1.8×, not 5:1) | `real/data/arcc_*.json` |
| 11 | the same asymmetry flips with exemplar count | 8-shot **0.91** → 5-shot **1.17** | `real/data/arc_easy_5shot.json` |
| 12 | NLL gap vs accuracy gap across 25 cells | $r=-0.936$ (band level holds, per-cell does not) | `real/data/` |
| 13 | minimum detectable effect, 0.5B / 1.5B / 7B | 3.0–9.5 / 3.5–7.0 / **4.7** points | `real/data/` |
| 14 | leadership replication: mean cost vs rest — 0.5B / 1.5B | 1.96× / 3.31× | `real/data/` |
| 15 | DeepSeek-sparse-attention weights, 141 pairs | ratio **0.8877** (shape replicates, magnitude does not) | `real/data/dsa_report.json` |
| 16 | 14B extrapolation, 267 pairs, 48 layers | ratio **12.67**, bootstrap CI **[5.50, 22.04]**, permutation $p=10^{-4}$ | `real/data/scale_leader_14b.json` |

Readout 16 is a robustness certificate kept out of the manuscript tables on purpose: it costs
20 batches of 512 positions per arm on a 14B model and moves none of the paper's claims.

## What this does *not* claim

- Not "cross-layer reuse is bad". At the decision level, on three model sizes, the loss at
  wide $k$ is within 1 point of the own-set pass. The claim is about **where** the cost sits
  and **what predicts** it, not that sharing should be avoided.
- Not a predictor paper with a negative result bolted on: three similarity-shaped predictors
  (overlap, CKA, weight-level contribution) do not rank the cost, and their sign flips with
  the universe and the preprocessing. Normalising by attention mass makes the whole predictor
  story disappear ($r=+0.509$ for starved attention mass, $+0.258$ without layers 0/1) — the
  correlation was absorbing the readout, not explaining it.
- Not a sparse-attention quality study: trained sparse checkpoints are a different substrate
  and are cited, not re-measured.
- Single-tokenizer model families; main protocol is 512 positions (the 512 → 2048 sweep is a
  separate readout, not the default).

## Repo map

```
paper/       main.tex (anonymized TMLR build), main.bib, tmlr.sty/bst, main.pdf
paper/audit_numbers.py       number gate: every printed number re-derived from an artifact
paper/evidence/              gate reports, derived facts, figure-trace manifests, src hashes
real/        the real-weight harness (patched attention, three readouts, per-experiment drivers)
real/data/   stored artifacts every table and prose number is re-derived from
scanner/     fast proxy experiments on pretrained weights (layer reuse, CKA, contribution)
data/        proxy artifacts (GPT-2 / distilgpt2 probes)
reports/     experiment notes in the order the claims were built (01 … 09)
prereg/      pre-registrations written before the intervention and proxy runs
```

## Quickstart

No third-party dependencies for the gate — Python 3 standard library only:

```bash
python3 paper/audit_numbers.py            # every printed number vs its artifact
python3 paper/audit_numbers.py --selftest # proves the gate can fail
```

Current state: `checks=428 mismatches=0`, `table cells=141 mismatches=0`,
coverage `397 of 403` printed numbers backed, `6` declared as notation
(`0.05, 0.2, 1.1, 1250, 1920, 5.1`), `VERDICT: every printed number is an artifact or
declared notation`.

The `--selftest` path derives its expectations from the artifacts rather than typing them, and
proves red on seven kinds of corruption: a corrupted manuscript literal, a wrong printed
digit, a shifted artifact, a wrong table cell, a moved table artifact, a missing table fact,
and an undeclared printed number.

Rebuild the PDF (two passes, TMLR style file included):

```bash
cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

The real-weight harness needs PyTorch and a local checkpoint (paths are set per script in
`real/`); the stored artifacts under `real/data/` are what the paper's numbers come from, so
the gate runs without any model download.

## Provenance

Every artifact under `real/data/` records the experiment that produced it; `real/data/scale_leader_14b.json`
carries the `sha256` of its source pair file (`pairs14b.jsonl`) so the 14B numbers can be tied
to the exact input. Gate reports in `paper/evidence/` are the ones referenced by the manuscript.

## License

MIT (see `LICENSE`). The manuscript is an anonymized review build: no author metadata is
present in this repository.
