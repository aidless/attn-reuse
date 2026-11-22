# Adversarial review synthesis — attn-reuse

Date: 2026-11-20. Target: `paper/main.tex` (776 lines). Reviewers: rev-domain (novelty/positioning), rev-method (statistics/controls), rev-da (devil's advocate). Protocol: academic-paper-reviewer v1.11.1, Phase 2 editorial synthesis. This is an editorial decision record, not a rewrite of reviewer opinions.

## Overall verdict

Three independent seats converge: the empirical grid (sections 3.1–3.4, 4.1) is publishable as a controlled measurement; the mechanism+intervention chain (section 3.5) and several absolute novelty phrasings are not defensible as currently written. Decision: **accept the empirical core, substantially downgrade the mechanism claim, withdraw non-comparable quantitative statements, and reposition novelty as controlled replication + two-scale separation.**

## DA CRITICAL items — ruled one by one

### I1 (CRITICAL) — mechanism/intervention circularity: UPHELD, requires structural rewrite
- Claim attacked: deliver-mass (§3.5) explains the leader cost; ORACLE HIGH restores the no-reuse baseline; therefore deliver-mass is the right axis.
- Reviewer's reasoning: deliver-mass = leader positions under follower *dense* softmax; HIGH = follower consensus *dense* mass top-k; LOW = bottom-k. HIGH<BASE<LOW is guaranteed by construction of the same dense distribution and does not require the deliver-mass construct. No head-to-head comparison against simpler follower-leader overlap was run.
- Editorial ruling: **UPHELD**. The three objects share the follower dense distribution. The data cannot support "deliver-mass is the right axis" or a mechanism claim; they support a weaker, observational, construct-consistency statement. Actions: (a) retitle §3.5 from a mechanism to a "predictive diagnostic and a construct check"; (b) delete "a remedy must move the shared index toward those positions"; (c) state explicitly that HIGH/LOW bracket the ordering by construction (HIGH=consensus dense top-k, LOW=bottom-k) and are a sanity check that the index can move cost, not evidence of mechanism identification; (d) report deliver-mass as an attention-mass-recall variant that is *correlated* with cost, with the confound that both use follower q/k; (e) add the honest comparison to the already-measured overlap r=-0.357 (Qwen, 23 pairs) and state deliver-mass r=-0.529 is on a different pair set and is **not** established as incrementally better.
- Reviewer's suggested "non-dense intervention" and head-to-head incremental regression are noted as genuine future work, not performed here; the paper must say this rather than imply it.

### F1 / I3 (CRITICAL) — abstract coefficients from undisclosed GPT-2 artifacts contradict body: UPHELD
- `paper/audit_numbers.py:691` binds abstract `r=-0.055` (overlap) to `ov.gpt2_ex0_r` from GPT-2 old artifacts (n=10, 2 texts, k=8); `r=-0.341` (CKA) and `r=-0.071` (contribution) are the same GPT-2 lineage. Body §"What does not predict" (main.tex:443) reports `r=-0.357` for overlap on Qwen (23 pairs). Abstract says "across pairs" without naming the model.
- Ruling: **UPHELD**. Change the abstract to name GPT-2 explicitly and say the three negative results are from an earlier 2-text GPT-2 probe; or remove the three coefficients from the abstract and point to the body. Chosen action: keep them but label the source ("in an earlier GPT-2 probe ...") and add the Qwen overlap number in the body as a separate, same-direction result; do not mix them.

### F2 / I5 (CRITICAL) — intervention BASE not comparable to grid, cross-precision leak: UPHELD
- Diagnostic `real/diag_base.py` on AMD, same model and batches: grid `reuse_llama.py` reuse NLL = 2.852547, intervention `base_live` NLL = 2.873979, delta = +0.021432; layer-0 selection differs at 1,352,658 / 1,835,008 (73.7%) positions; rerun deterministic. v1 intervention was fp16 while grid and main text are bfloat16. Round4 claim "HIGH lands within 0.0012 of the no-reuse baseline, removes essentially the whole penalty" crossed both precision and index construction; the leaked ~0.02 nats is comparable to the effect being explained (~0.024).
- Ruling: **UPHELD**. Withdraw the quantitative "HIGH ≈ no-reuse baseline" and "removes essentially the whole penalty" statements. Report only the within-intervention ordering HIGH<BASE<LOW (same artifact, same dtype) as a construct check. If retaining the v2 bfloat16 numbers, state that BASE in the ORACLE harness is not bit-identical to the grid reuse condition and give the diagnostic delta; do not treat them as a single continuous story. Unit the dtype declaration across the paper (bfloat16 for Qwen2.5 grid and intervention).

### rev-domain Critical#1 — absolute existence claim falsified by IndexCache: UPHELD
- IndexCache (arXiv:2603.12201, verified abs+HTML,号-题一致) §4.3 Table 2 gives training-free cross-layer index reuse quality on long-context benchmarks and shows sharp drops without search; §4.4 discusses inherited-index distributional shift; Appendix C is "why similarity fails as a proxy". main.tex:47 "we could not find it measured on real weights" and main.tex:92-95 "We could not find this step measured ... the sparse-attention papers report throughput" are false.
- Ruling: **UPHELD**. Replace with "IndexCache measures the quality of training-free index reuse at benchmark granularity on DSA models; we isolate the selection-set variable under a controlled three-readout design on dense pretrained weights and separate token-level from decision-level cost." Add IndexCache/Kascade to the related-work paragraph with what they measure vs what this paper measures.

### rev-domain Critical#2 — deliver-mass is a variant of prior attention-mass recall: UPHELD (merged into I1)
Kascade (arXiv:2512.16391, verified) §3.2 similarity ratio is the normalized form of deliver-mass; §3.1 notes layer-0 flat distribution => layer 0 stays dense (same source-0 signal). Ruling: treat as additional support for I1 downgrade; cite Kascade as prior art for the quantity and for the layer-0 observation. Do not call deliver-mass a novel mechanism.

## MAJOR items — rulings

- **I2 early-dummy control**: UPHELD as a reporting deficiency. The paper controls continuous source-layer index (partial -0.47/-0.63) but the reset is binary (source≤1). Reviewer's recomputation (0.5B -0.5124→-0.3338; TinyLlama -0.6912→-0.4253) shows ~1/3 shrinkage. Action: state this in §3.5 as a confound; present the already-stored E_late_only (0.5B -0.2957, n=111) in the main text, not only as a parenthetical "weaker on 0.5B". Do not claim depth was fully controlled.
- **I4 layer-3 story**: PARTIALLY UPHELD. The layer-3 = 18.9× median claim is numerically correct on `pairs20w12_0.5B.jsonl`, but the "width non-monotonicity mechanism" built on it is 0.5B-specific and should be labelled as an observation on one model. Keep the descriptive arithmetic; drop or hedge the mechanistic explanation.
- **I6 shape vs mismatch explanation**: UPHELD as a limitation to state. Selection-shape descriptors (top-k mass 0.9928–0.9995, local 0.2773–0.2800, pos0 0.0029–0.0039) are nearly flat while deliver differs by 6.4 points for 11.9–12.4× cost. No position-embedding ablation or matched-deliver conditioning was performed. Add to Limitations; do not claim the mismatch mechanism is identified.
- **F3 statistical units**: UPHELD as a caveat. The leader permutation treats 210 single pairs as exchangeable though source and target are depth-coupled; the damage/rescue Clopper-Pearson intervals treat paired b/c as independent; 20-batch and first-4-batch results are nested; MDE is post-hoc. The paper already flags MDE as design aid; extend the same caveat to the permutation and CI. No new experiment this round.
- **F4 / I7 decision-level robustness**: UPHELD as a hedge. 8-shot vs 5-shot sign reversal and 4-batch vs 20-batch sign flips must be visible in the abstract/body, not buried.
- **F5 sample coverage**: ACKNOWLEDGED. 7B token grid is k∈{128,256,384}; TinyLlama is single-pair only; state coverage explicitly.
- **F6 / I8**: minor — fix label "flipped includes argmax changes" and distinguish seq=384 decision from seq=512 token tables (source: `paper/main.tex`, the decision-level and token-level table captions).

## Citations to add (all verified 号-题 via https web_fetch, 2026-11-20)
- Kascade 2512.16391 — §3.2 similarity ratio, §3.1 flat layer-0.
- KVSharer 2410.18517 — "sharing dissimilar KV caches better preserves performance"; similarity-fails prior.
- TidalDecode 2410.05076 — training-free token/layer selection reuse (systems line).
- HySparse 2602.03560 — sparse attention systems line.
- StreamingLLM 2309.17453 — attention sink, for "measuring what a component is needed for".
- Kornblith et al. ICML 2019 — CKA original (bibtex from official proceedings, verify before adding; if unverified this round, cite in text by name without fabricated bibitem).
- IndexCache 2603.12201 — already in `main.bib`, upgrade the related-work text.

## What remains defensible after ruling
1. Controlled three-readout design on dense pretrained weights; only the attended-position set changes, own(k=seq)=dense control is bit-identical.
2. Grid: cost governed by k and source-layer identity, not distance (leader anomaly 11.9× per-pair, 7.5–35× across four models, permutation p≤0.0002 with the 0.5B value corrected).
3. Three similarity-shaped predictors fail on GPT-2 (n≈10–11) and overlap fails on Qwen (n=23); consistent with KVSharer/IndexCache/Kascade prior art, we replicate rather than discover this.
4. Decision-level: at k=67% budget differences ≤1 point; damage>repair 1.3–1.8×, direction unstable to shot count and sample size (source: `real/data/sensitivity.json`, `real/data/atrisk_512.json`, and the decision-level tables in `paper/main.tex`).
5. Deliver-mass correlates with cost and an ORACLE intervention moves cost in the predicted direction — reported as a construct check, not a mechanism identification, not a deployable remedy (source: `real/data/deliver_corr.json`, `real/data/intervene_05_v2.json`, `real/leader_deliver_analysis.py`).
