# Is distance an axis, or a composition effect?

The manuscript's `Distance is not the axis; the leader is` section reports a
**marginal** profile: the median cost at distance one, five and ten, pooling
every source that can reach that distance. The profile is what the data says,
but the argument built on it is weaker than it looks, for one arithmetic
reason.

## The profile is confounded by construction

In an $L$-layer stack a pair at distance $d$ can only have source
$0 \ldots L-1-d$, so the *mean source* at distance $d$ is exactly
$(L-1-d)/2$ whenever the column is complete. Checked against all four sweeps:

| sweep | L | d=1 | d=6 | d=12 | arithmetic $(L-1-d)/2$ |
|---|---|---|---|---|---|
| qwen05_w12 | 24 | 11.00 | 8.50 | 5.50 | 11.00 / 8.50 / 5.50 |
| qwen7_w12 | 28 | 13.00 | 10.50 | 7.50 | 13.00 / 10.50 / 7.50 |
| qwen14_w6 | 48 | 23.00 | 20.50 | — | 23.00 / 20.50 |
| tiny_w6 | 22 | 10.00 | 7.50 | — | 10.00 / 7.50 |

The observed column matches the arithmetic column to two decimals in every
case. That is not a finding about the model; it is geometry. So the deep
columns of the published profile are populated by shallower sources, and
"cost at distance twelve" and "cost at distance one" are averages over
*different source populations*.

## What survives once the composition is held fixed

Two stratified estimators, both computed from the published sweeps:

* **within-source**: hold the source fixed, ask whether the cost rises from
  that source's nearest to its farthest target.
* **within-target**: hold the target fixed, ask the same question with the
  source varying the other way.

| sweep | within-source rho | p (20000 stratified permutations) | within-target rho | R^2 source | R^2 distance increment |
|---|---|---|---|---|---|
| qwen05_w12 | +0.172 | 0.0609 | +0.480 | 0.0906 | 0.0021 |
| qwen7_w12 | +0.168 | 0.0354 | +0.264 | 0.0456 | 0.0086 |
| qwen14_w6 | +0.191 | 0.0057 | +0.176 | 0.0480 | 0.0224 |
| tiny_w6 | +0.236 | 0.0381 | +0.616 | 0.1420 | 0.0015 |

Three things, in order of what they change:

1. **Distance is not nothing.** Once source is held fixed, rho is +0.17 to
   +0.24 with p between 0.0057 and 0.0609 — three of four sweeps clear 0.05,
   0.5B does not. So the paper's "twelve times the distance buys less than
   twice the cost" is true in magnitude but understates the direction: the sign
   is consistently positive, not flat.
2. **Source dominates it, but the size of the gap depends on which column
   goes first (see the Round 28 correction at the end).** Ordering by source
   gives a distance increment of 0.0015 to 0.0224 against source's 0.0456 to
   0.1420; ordering by distance gives a position increment of 0.0436 to
   0.1341. Both are the same two-variable fit. The order-free version is the
   marginal pair: source alone 0.0456 to 0.1420, distance alone 0.0012 to
   0.0268.
3. **RETRACTED (Round 28).** "Later targets pay more is a cleaner statement"
   was wrong. `t = s + d` holds for every pair, so with the source held fixed
   the distance and the target differ by a constant: `rho(distance, cost)` and
   `rho(target, cost)` are the *same number* (+0.132172 on 0.5B, identical to
   1e-12 on all four sweeps). Likewise with the target held fixed,
   `rho(distance, cost)` and `rho(-source, cost)` are the same number (all four
   identical). The "within-target" number is therefore the leader effect read
   backwards — a shallower source costs more — not an independent regularity
   about targets. Checked row by row on target=12 in 0.5B: cost climbs
   +0.001701 (source 11) to +0.003920 (source 0) as the source shallows.

## The mean and the median disagree in sign on one sweep

| sweep | near mean | far mean | mean Δ | near median | far median | median Δ |
|---|---|---|---|---|---|---|
| qwen05_w12 | +0.001247 | +0.001045 | **−0.000202** | +0.000635 | +0.000992 | **+0.000357** |
| qwen7_w12 | +0.000719 | +0.000968 | +0.000249 | +0.000375 | +0.000831 | +0.000456 |
| qwen14_w6 | +0.000165 | +0.000734 | +0.000568 | +0.000176 | +0.000325 | +0.000149 |
| tiny_w6 | +0.000954 | +0.004970 | +0.004016 | +0.000367 | +0.000779 | +0.000412 |

On 0.5B the mean delta is **negative** and the median delta is **positive**.
One source (s=2) contributes +0.0146 at its nearest target and +0.0013 at its
farthest, which drags the mean. A mean-only summary of this effect would
report the wrong sign. Both are now reported, and the near/far check has a
fixture in the selftest that plants exactly this shape.

## Does and does not establish

* Establishes: the published distance profile is a composition effect in its
  deep columns; the composition is arithmetic, not empirical. Distance still
  carries a detectable positive signal once source is fixed, but it accounts
  for 0.15–2.2% of variance against source's 4.6–14.2%.
* Does not establish that distance is irrelevant. rho +0.17 to +0.24 with
  p ≤ 0.06 on three sweeps is a signal, and the paper now says "small" rather
  than "not an axis" where the numbers require it.
* Does not establish a mechanism for the within-target regularity. "Later
  targets pay more" could be target depth, could be the shallower source that
  necessarily accompanies it at fixed target.

## Reproduce

    python3 real/distance_axis.py --selftest
    python3 real/distance_axis.py

Artifact `real/data/distance_axis.json` (20000 permutations, seed 20261121;
~27 s GPP-free). Selftest proves six things can fail: composition vs
arithmetic on a complete sweep, disagreement on a holed sweep, a planted trend
detected, flat data staying inside its null, the decomposition splitting by
which variable drives, and near/far exposing a mean-median sign disagreement.

## Two instrument lessons

1. **A marginal profile over a stratified sweep is a composition report.**
   Any "axis" claim drawn from a per-distance or per-bucket average of a
   cross-product sweep needs the per-level composition printed beside it. Here
   the composition was arithmetic and matched to two decimals, so the confound
   was exact rather than approximate — which is what made it easy to miss.
2. **A single summary statistic can flip the sign.** The near/far table has a
   mean-negative, median-positive case. My first selftest asserted a magnitude
   threshold (`abs(median_delta) < 0.2`) and the fixture landed exactly on
   0.2 — a threshold artefact, not a failing gate. The assertion is now on
   sign disagreement, which is the property that actually matters.


## Round 28 correction: the decomposition is basis-dependent

`t = s + d` is an identity, not a fitted relation, so `(source, target,
distance)` spans only two dimensions and any two of the three generate the
same fit. Reporting "the distance increment is 0.0015–0.0224 while source is
0.0456–0.1420" is therefore a statement about **which column was put first**,
not about the model:

| sweep | R²(s) | R²(t) | R²(d) | +d\|s | +d\|t | +pos\|d | 2-var | unexplained | corr(s,t) |
|---|---|---|---|---|---|---|---|---|---|
| qwen05_w12 | 0.0906 | 0.0482 | 0.0182 | 0.0021 | 0.0446 | 0.0746 | 0.0927 | 90.7% | +0.817 |
| qwen7_w12 | 0.0456 | 0.0536 | 0.0012 | 0.0086 | 0.0006 | 0.0530 | 0.0542 | 94.6% | +0.869 |
| qwen14_w6 | 0.0480 | 0.0390 | 0.0268 | 0.0224 | 0.0314 | 0.0436 | 0.0704 | 93.0% | +0.991 |
| tiny_w6 | 0.1420 | 0.1201 | 0.0095 | 0.0015 | 0.0234 | 0.1341 | 0.1435 | 85.6% | +0.952 |

The three orderings sum to the same total: `R²(s) + inc(d|s) = R²(t) +
inc(d|t) = R²(d) + inc(pos|d) = the two-variable R²`, which the script asserts
to 1e-9 on a fixture. So the honest statements are:

- **No two-variable fit of these two indices explains more than 14.4%.** 86–95%
  of the variance in pair cost is carried by something neither index names.
  This is the strongest survivor and it needs no ordering.
- **The leader is the larger axis in the marginal sense only** — source alone
  0.0456–0.1420 vs distance alone 0.0012–0.0268. That comparison is
  order-free because each column is fitted by itself.
- **The split is unidentifiable from this data.** `corr(source, target)` runs
  +0.82 to +0.99, and at fixed distance the two are collinear by construction
  (`t - s == d` on every row, verified).

Two stratification identities were also mistaken for findings. Both are
proved in `basis_check` and asserted in `selftest`.

The paper was rewritten accordingly: the section heading became "Two indices,
neither of which is the axis", the abstract's "far more than" became "and by
inter-layer distance much less", and the decomposition paragraph now reports
all three orderings plus the ceiling.
