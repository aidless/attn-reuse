# The headline is an average over rows that cannot discriminate

Written 2026-11-21. Answers a question a referee would ask about `tab:grid` and
every other number in this paper: what exactly is being averaged?

## 1. The observation

At sequence length $S$ and top-$k$, a query at position $q$ can see $q+1$ keys.
While $q+1 \le k$ the top-$k$ set **is** every visible key. A borrowing layer
therefore receives exactly the set it would have chosen, and that row
contributes precisely zero to `reuse - own`. Those rows nevertheless count in
the denominator, because the loss is a mean over all $S-1$ scoring positions.

The share of inert rows is $\min(k, S-1)/(S-1)$: at 512 positions with
$k{=}256$ it is **50.1%** (source: `attn-reuse/real/data/dilution_analysis.json`,
field `cells.k256_g2.inert_row_share`).

## 2. The identity, with no free parameter

If the inert rows contribute exactly zero, then the delta restricted to
discriminating rows must be

$$
\delta_{\text{active}} = \delta_{\text{all}} \cdot \frac{S-1}{S-1-k}
$$

exactly. This is not fitted; it follows from "those rows are zero" and nothing
else. Measured on 0.5B, 20 batches, three group sizes, four $k$ values:

| $k$ | inert share | predicted | observed (g=2,3,6) |
|---|---|---|---|
| 64  | 12.5% | 1.1432 | 1.1432 · 1.1432 · 1.1432 |
| 128 | 25.0% | 1.3342 | 1.3342 · 1.3342 · 1.3342 |
| 192 | 37.6% | 1.6019 | 1.6019 · 1.6019 · 1.6019 |
| 256 | 50.1% | 2.0039 | 2.0039 · 2.0039 · 2.0039 |

(source: `attn-reuse/real/data/dilution_analysis.json`, computed by
`attn-reuse/real/dilution_analysis.py` from
`attn-reuse/real/data/position_matched_05_20b.json`)

Worst deviation across all twelve cells: **5.38e-06**. Twelve out of twelve
cells reproduce the derivation to five decimals, and the multiplier depends
only on $k$ — not on the group size, not on the corpus (this run uses a textbook
corpus, not WikiText), not on the dtype.

That agreement is the point. It says the inert rows really do contribute
nothing, so the dilution is a property of the *measurement*, not a quirk of one
model. If a cell had disagreed, the zero-contribution assumption would be wrong
and every number in the grid would need re-deriving.

## 3. What it changes and what it does not

**Does not change**: the ordering of cells, the sign, the dependence on $k$ and
$g$, the leader anomaly ratios, the null ratios. Every comparison is a
comparison of deltas measured on the same row set, so a common factor cancels.

**Does change**: the size of the number a reader quotes. At $k{=}256$ the
headline $+0.0242$ is $+0.0368$ per discriminating token; at $k{=}64$ the
correction is only 14% (source: `attn-reuse/real/data/dilution_analysis.json`,
`cells.k64_g2.predicted_multiplier`). So the published numbers are *conservative*, and they
are conservative by more where $k$ is a larger fraction of the sequence — which
is exactly the regime the paper argues is dangerous.

## 4. Does this affect the trained-sparse result?

Yes, identically: 4096 positions with `index_topk=2048` also has 50.0% inert
(source: `attn-reuse/real/data/dsa_group_rand.json`, fields `seq` and `topk`)
rows, so the same factor of 2.0039 applies there. Because both the reuse delta
and the null delta scale by the same factor, the ratio between them (0.034 to
0.126) is unaffected — a ratio of two quantities measured on the same rows is
dilution-free.

## 5. How it was checked

`attn-reuse/real/position_matched.py` re-derives per-position loss from the same
forward pass (no extra compute) and reports both means. `attn-reuse/real/dilution_analysis.py`
checks the identity and carries two directional negative controls: a fixture
that violates the identity must be flagged, and a fixture that obeys it must be
accepted — otherwise the check would be always-red or always-green.

```bash
python3 attn-reuse/real/position_matched.py --n-batches 20 --ks 64,128,192,256 \
    --out attn-reuse/real/data/position_matched_05_20b.json
python3 attn-reuse/real/dilution_analysis.py --selftest
python3 attn-reuse/real/dilution_analysis.py
```

## 6. One bug worth recording

The first version of `position_matched.py` applied `ks[0]`'s threshold to every
cell, so a run with `--ks 64,128,192` used $k{=}64$ everywhere. It produced
multipliers of 0.87×, 19.05× and −0.91× — a negative multiplier, which is
impossible for a share-of-rows correction. The absurd sign is what exposed it.
A per-k baseline had to be computed too; the baseline was also being cut once
with `ks[0]`.
