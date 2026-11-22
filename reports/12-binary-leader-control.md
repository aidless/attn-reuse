# The binary early-leader control (rev-da I2)

Round 7. The Round 5 adversarial review's devil's advocate (finding I2) was the
last objection with no number behind it: the manuscript controlled the
*continuous* source-layer index, but the theory's actual confound is the
*binary* leader identity (source $\le 1$). If deliver-mass is only a repackaging
of "this pair's source is an early leader" -- a variable the leader anomaly
already establishes as dominant -- then its correlation with cost is not a new
quantity. This report computes that control from the artifacts on disk.

## Design (source: `real/dummy_controls.py`)

Join the same two probes used everywhere else -- the dense deliver-mass probe
(`real/data/mech_05.json`, `real/data/mech_tiny.json`) and the masked-NLL pairs
(`real/data/pairs20_0.5B.jsonl`, `real/data/pairs20_TinyLlama.jsonl`) -- by
`(source, target)`. Define `dummy = 1[source <= 1]`. Report:

- **R** = corr(deliver, dummy): how much of the mass readout is the leader flag.
- **P** = partial(deliver, cost | dummy): does the correlation survive the
  correct (binary) control?
- **full** = partial(deliver, cost | dummy, source, distance): full stack.
- **dR2** = R2(cost ~ dummy + deliver) - R2(cost ~ dummy): variance deliver adds
  over the leader flag alone.

Three synthetic fixtures pin the instrument: a *dummy-only* fixture (deliver is
a pure function of the flag) collapses P to $-0.03$ and dR2 to $0.000$; an
*independent* fixture (flag silent, deliver tracks cost) keeps P $=-0.998$; a
*layered* fixture (half flag, half independent) keeps P $=-0.984$ with dR2
$=+0.802$. The comparison can go red. (source: `real/dummy_controls.py`,
`--selftest`)

## Result (source: `real/data/dummy_controls.json`)

| quantity | Qwen2.5-0.5B (n=123) | TinyLlama-1.1B (n=111) |
|---|---|---|
| R = corr(deliver, leader dummy) | −0.7038 | −0.9002 |
| **P = partial(deliver, cost \| dummy)** | **−0.3338** (p=1.6e−4) | **−0.4253** (p=3.3e−6) |
| full stack partial | −0.3650 | −0.4197 |
| dR2 of deliver over dummy | +0.0915 | +0.1149 |

## Scoring the same control at 1.5B and 7B (source: `real/data/dummy_controls.json`)

| quantity | Qwen2.5-1.5B (n=147) | Qwen2.5-7B (n=147) |
|---|---|---|
| R = corr(deliver, leader dummy) | −0.9093 | −0.7731 |
| P = partial(deliver, cost \| dummy) | −0.2369 (p=3.9e−3) | −0.4887 (p=3.4e−10) |
| full stack partial | −0.2205 | −0.4956 |
| R2(dummy) | 0.4984 | 0.3056 |
| dR2 of deliver over dummy | +0.0281 | +0.1659 |

The control is scale-dependent, and we report that rather than only the
favourable cells. At 1.5B the league flag already explains half the cost
variance, so deliver-mass adds almost nothing (+0.028) and the partial, while
still negative and significant, is weak (−0.237). At 7B the picture is close to
0.5B (+0.166 added variance, partial −0.489). The honest claim is "survives the
binary control at every scale, weakly at 1.5B", not "a stable increment".

## Reading

The reviewer is right that the correlation shrinks materially under the binary
control (0.5B $-0.5124 \to -0.3338$; TinyLlama $-0.6912 \to -0.4253$) and that
deliver-mass is heavily entangled with leader identity (R $=-0.70/-0.90$). But
it does **not** collapse: the partial stays significant at both p $< 10^{-3}$
and deliver adds 9--11 points of explained variance beyond the flag. The honest
summary is "partly entangled with leader identity, not a restatement of it" --
which is what the manuscript now says.

This is still observational: deliver-mass and the masked-NLL cost share the
follower's queries and keys, so this control removes one confound (leader
identity) but does not identify a mechanism. (source: `paper/main.tex`,
§The mass diagnostic is not an identified mechanism)
