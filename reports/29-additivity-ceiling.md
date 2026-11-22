# The additivity bound, and what the single-pair protocol can be used for

## What was already there

`\subsection{Borrowed costs add up}` compared the measured group cost against
the sum of its members' single-pair costs, and reported factors for 0.5B at
g=2,3,6 and for 1.5B and 7B at the same three widths. Every one of those
numbers reproduces exactly from `real/data/main_20b_05.json` and the
`real/data/pairs20w12_*.jsonl` sweeps; the own_k256 baseline the two
protocols subtract is the same to eight decimals
(0.5B 2.82835256, 1.5B 2.49151429, 7B 2.27042517), so the subtraction is
against the same reference on both sides.

## The defect

The paragraph summarised 0.5B as "within 32%", quoting g=2, g=3 and g=6 (`replication` of the old sentence, source: `git show HEAD:attn-reuse/paper/main.tex`).
`tab:grid` prints a g=12 column, and `real/data/main_20b_05.json` already
carries that cell: measured +0.44930 against a summed single-pair prediction
of +0.15502, a factor of **2.90** (source: `real/data/additivity.json`).
So the qualitative summary was falsified by a cell in the same paper, three
pages earlier. This is the same failure mode as the stale Limitations
sentence in Round 25 and the "sign is the opposite" error in Round 10: a
summary written when fewer cells existed, never revisited when the table
grew.

## The corrected numbers

Measured over summed-predicted, k=256, twenty batches, 512 positions
(source: `real/data/additivity.json`):

| model | g=2 | g=3 | g=4 | g=6 | g=12 |
|---|---|---|---|---|---|
| 0.5B | 1.03 | 1.12 | 1.16 | 1.32 | **2.90** |
| 1.5B | 1.89 | 1.72 | - | 2.32 | - |
| 7B   | 0.40 | 0.62 | - | 0.71 | - |

Additivity holds to three percent at g=2 on 0.5B and degrades monotonically
with width. On 7B the sum *overstates* the group cost at every width
measured, so the sign of the departure is a property of the model, not of
the protocol.

## The part that is new

The rank correlation between the summed single-pair cost and the measured
group cost is **+1.000** on all three models (0.5B over five widths, 1.5B
and 7B over three each; source: `real/data/additivity.json`). Ordering
transfers even where magnitude does not: on 7B the single-pair protocol
ranks the three widths correctly while understating the cost of sharing by
a factor of two, and on 0.5B it ranks all five correctly while
understating the widest by a factor of three.

That makes the single-pair protocol a usable *ranking* instrument and an
unusable *magnitude* instrument, which is what every mechanism result in the
paper rests on. Nothing in the paper said so before.

## The explanation I can offer and the one I cannot

At g=12 the 24 layers form two groups, so eleven consecutive layers borrow
from layer 0 simultaneously and each corrupted output is the next layer's
input. At g=2 no follower of layer 0 has another follower of layer 0
upstream. The departure is largest exactly where that chain is longest.

But chain depth and group width move together in this grid: the number of
followers grows 12, 16, 18, 20, 22 across the five widths and the number of
those borrowing from layer 0 grows 1, 2, 3, 5, 11, so the two cannot be
separated from these cells. The paper records the pattern and says
explicitly that it is not a decomposition. The single-pair costs at source
0 are themselves extremely uneven (+0.00121 to +0.04065 over eleven
followers, source: `real/data/pairs20w12_0.5B.jsonl`), so a three-value sum
carries a lot of the prediction, which is a further reason not to read a
mechanism into the ratio.

## Verification

- `python3 real/additivity.py --selftest` -> five directions, including a
  super-additive fixture that must give 2.0, a sub-additive one that must
  give 0.5, and a rank fixture where ranks are perfect while every ratio is
  wrong.
- `python3 paper/audit_numbers.py` -> `checks=428 mismatches=0`, coverage
  397 of 403, six declared notation.
- prove-red on both new literals: corrupting 2.90 to 2.91 and 1.000 to
  0.998 each yields `NO SOURCE` and `VERDICT: unbacked numbers without a
  declared reason`.
- `real/additivity.py` and the inline block in `paper/audit_numbers.py` are
  separate implementations and agree cell for cell.

## A lesson I earned the hard way this round

I re-derived the whole additivity table from scratch before noticing the
paper already contained it. The waste was not the recomputation — that
produced an independent check, which is worth having — it was that I did not
look at the subsection I was about to add to. The cheap first move, before
any new script, is to grep the manuscript for the quantity you think you are
discovering.
