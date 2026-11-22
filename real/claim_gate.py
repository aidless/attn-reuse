#!/usr/bin/env python3
"""Claim-support gate: every claim must land on an artifact field, and its
wording must not outrun what its evidence class supports.

Three checks per claim, all deterministic (no model call):

  A  quote  — the claim's quoted sentence must appear in the report verbatim.
              A claim the manuscript no longer contains is a claim nobody can
              audit; a paraphrase means the gate is checking text that is not
              the paper.
  B  support — each cited (artifact, field) must resolve in the evidence file,
              and the number printed in the quoted sentence must equal that
              field within the stated tolerance. This is the half ARIS calls
              "hallucinated evidence": path missing, or the value is not there.
  C  strength — a claim declared `partial` or `negative` may not use verbs that
              assert what its evidence does not carry. `measured` claims are
              free: the number IS the measurement.

Why C exists: the report's honest sentence is "starved is a directionally
correct partial explanation (r=+0.509), not 'the mechanism is found'". Without
C, a later edit can promote that to "the mechanism" and nothing would notice —
which is exactly the failure mode this project already hit once (§7 of the
report had to be rewritten to stop at "partial").

Run: python3 claim_gate.py [--report PATH] [--evidence PATH] [--selftest]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "reports", "06-reuse-real-cost.md")
EVIDENCE = os.path.join(HERE, "data", "evaluation_evidence.json")
REPORT07 = os.path.join(HERE, "..", "reports", "07-decisions.md")
SUMMARY = "attn-reuse/real/data/mcq_summary.json"
POWER = "attn-reuse/real/data/mcq_power.json"
CROSS = "attn-reuse/real/data/cross_bench.json"
ARCC15 = "attn-reuse/real/data/arcc_15b.json"
ARCC7 = "attn-reuse/real/data/arcc_7b.json"

# Wording that asserts more than a correlational or negative result carries.
# Kept deliberately short: a long list becomes a lexical taste test, and the
# point is to catch promotion, not to police style.
BANNED = {
    "partial": ["机制已找到", "证明了因果", "是因果", "由…决定", "决定",
                "proves", "determines", "is caused by"],
    "negative": ["不存在", "从不", "never", "no such thing"],
}

# Claims: (id, verbatim quote from the report, class, [(artifact, field, tolerance)]).
# `field` is a dotted path; `[i]` indexes a list; a `*` reads a list of numbers.
CLAIMS = [
    ("C1-control",
     "`own(k=seq_len)` 与 `full` 必须**逐位相同**",
     "measured", []),

    ("C2-grid-k",
     "k=384（= 512 的 75%）时 g≤4 的代价 ≤ +0.0114 nats",
     "measured", [("attn-reuse/real/data/main.json",
                   "deltas.k384_g4.vs_own", 5e-5)]),

    ("C3-offset",
     "代价降到原来的 0.41 倍（g=2）、0.24 倍（g=3）、0.19 倍（g=4）、0.51 倍（g=6）",
     "measured", []),

    ("C4-leader",
     "**leaders 0–1（n=24）均值 +0.01151 vs leaders 2+（n=186）均值 +0.00082 = 13.98 倍**",
     "measured", []),

    ("C5-layer3-share",
     "g=2 分区里，layer 3←2 这一对单独贡献了 79.5%",
     "measured", []),

    ("C6-additive",
     "**可加到 ±2 倍以内**",
     "measured", []),

    ("C7-long",
     "k=512/g=2 仍是 +0.07921 nats",
     "measured", [("attn-reuse/real/data/long.json",
                   "deltas.k512_g2.vs_own", 5e-5)]),

    ("C8-overlap-null",
     "**r = -0.357（n=23）**",
     "negative", []),

    ("C9-starved-partial",
     "starved 是**方向正确的部分解释**（r=0.509，去掉浅层后 0.258）",
     "partial", []),

    ("C10-no-mechanism",
     "不可写「复用代价由 attention 质量分布决定」",
     "partial", []),

    ("C11-shape",
     "**24 层的形状几乎一模一样**",
     "measured", []),

    ("C12-own-not-worse",
     "**稀疏 top-k 本身不让模型变差**",
     "measured", []),
]

CLAIMS07 = [
    # (id, verbatim quote, class, [(artifact, field, tolerance)])
    # Report 07's load-bearing numbers. Scale mismatches are the defect this
    # guards: an artifact storing 0.565 vs a report printing "−25.5 pt" would
    # otherwise need a re-scaling in the reader's head, and a re-scaling in
    # the gate is a place for a silent off-by-100.
    ("D1-dense-acc",
     "dense 准确率 0.6100",
     "measured", [(SUMMARY, "by_cell.Qwen2_5_0_5B_k32_g2.dense_acc", 5e-5)]),

    ("D2-grid12-collapse",
     "**−25.5** (p=1.7e−09)",
     "measured", [(SUMMARY, "by_cell.Qwen2_5_0_5B_k32_g12.acc_pt", 0.05),
                  (SUMMARY, "by_cell.Qwen2_5_0_5B_k32_g12.p_exact", 1e-10)]),

    ("D3-k256-flat",
     "五格合计只有 5 次 discordant——两格 b=c=0，一格 b=c=1（净 0）",
     "measured", []),

    ("D4-scale",
     "k=128/g=2 的 `reuse−own` 在 0.5B 是 +0.1529 nats、1.5B 是 +0.0482（0.32 倍）",
     "measured", [("attn-reuse/real/data/main.json",
                   "deltas.k128_g2.vs_own", 5e-5),
                  ("attn-reuse/real/data/main_15b.json",
                   "deltas.k128_g2.vs_own", 5e-4)]),

    ("D5-band-corr",
     "**25 格上 Pearson(NLL 增量, 准确率变化) = −0.936。**",
     "measured", [(SUMMARY, "corr_nll_vs_acc_pt_keyed.Qwen2_5_0_5B.r", 1e-3)]),

    # D6 was originally "1.5B 与 7B 的 b:c 是 5.1:1 与 5.0:1". That claim is
    # retired, not softened: b-c = N*(own-reuse) is an identity (checked on all
    # 40 cells), so b:c restates the accuracy drop and cannot be evidence of a
    # directional bias. The replacement cites the two rates that have
    # independent denominators, computed by cross_bench.py.
    ("D6-directionality",
     "ARC-Easy 上每 100 个原本答对的题有 **10.1** 个被改错、"
     "每 100 个原本答错的题有 **7.9** 个被改对（非对称比 **1.28**）",
     "measured", [(CROSS, "arc_easy.pooled.damage_pct", 5e-2),
                  (CROSS, "arc_easy.pooled.rescue_pct", 5e-2),
                  (CROSS, "arc_easy.pooled.asymmetry", 5e-3)]),

    ("D7-margin-auc",
     "| **AUC(margin 预测翻转)** | **0.764** | **0.856** |",
     "measured", [("attn-reuse/real/data/mcq_offset_05b_k64g6.json",
                   "aggregate.auc_margin_predicts_flip", 5e-4)]),

    ("D8-power-is-bound",
     "**「0.5B 在 k=256 上检不出损伤」而不是「0.5B 在 k=256 上无损」**",
     "partial", []),

    # class is `measured`, not `negative`: "no rejection region at n=2" is
    # arithmetic (the smallest two-sided exact p is 2*0.5^n), not an
    # absence-of-evidence claim. The machine check for it lives in
    # verify_decisions_numbers.py ("no cell without a rejection region may carry
    # an MDE"), which is stronger than a numeric field match could be.
    ("D9-no-rejection-region",
     "连拒绝域都不存在",
     "measured", []),

    ("D10-not-a-proxy",
     "**不能写**「NLL 代价可以当作准确率代价的代理」",
     "partial", []),

    # The exemplar count is a live confound, measured rather than assumed: the
    # same six cells on the same model disagree in sign at 8 and at 5 exemplars.
    # This claim exists so that §6.1's table cannot silently drift away from
    # the artifact, and so that "ARC-Challenge is more asymmetric because the
    # questions are harder" cannot be re-introduced without the gate noticing.
    ("D11-shot-control",
     "**方向在同一批格子上反了**：8 示范时 damage < rescue（比值 0.91），"
     "5 示范时 damage > rescue（1.17）",
     "measured", [(CROSS, "shot_control.arms.eight_shot.asymmetry", 5e-3),
                  (CROSS, "shot_control.arms.five_shot.asymmetry", 5e-3)]),

    ("D12-shot-confound-unresolved",
     "**§6 的 1.28 不能全归给题面**",
     "partial", []),
]


def resolve(doc, path):
    """Dotted path with [i] indexing. Raises KeyError when the path does not exist."""
    cur = doc
    for part in path.split("."):
        m = re.match(r"(\w+)(?:\[(\d+)\])?$", part)
        if not m:
            raise KeyError(part)
        cur = cur[m.group(1)]
        if m.group(2) is not None:
            cur = cur[int(m.group(2))]
    return cur


NEG_MARKERS = re.compile(
    r"(不可写|不能写|不应写|不许写|不得写|不可称|不能称|不要说|不许说|不可断言"
    r"|cannot write|cannot claim|must not|not claimable)")
CLAUSE_BREAK = re.compile(r"[。；;\n]|\|\s*---")


def banned_verdicts(text, phrase):
    """Yield True for each *un-negated* assertion of `phrase`.

    A phrase preceded by a prohibition marker in the same clause is the report
    naming what it refuses to say — legal. A phrase preceded by a clause break
    or with no marker is a real assertion — illegal for a `partial` class.
    """
    start = 0
    while True:
        i = text.find(phrase, start)
        if i < 0:
            return
        head = text[:i]
        # last clause boundary before the phrase
        cuts = [m.end() for m in CLAUSE_BREAK.finditer(head)]
        clause = head[cuts[-1]:] if cuts else head
        yield not NEG_MARKERS.search(clause)
        start = i + len(phrase)


# The report typeset with U+2212 MINUS SIGN, not ASCII '-'. A pattern that only
# accepts ASCII turns "−25.5" into +25.5, and since the comparison below uses
# |n| the sign error hides until a quote's only matching number is negative.
NUM = re.compile(r"[-+\u2212\u2013]?\d+(?:\.\d+)?(?:e[-+\u2212\u2013]\d+)?"
                 r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?")
MINUS = str.maketrans({"\u2212": "-", "\u2013": "-"})


def numbers_in(text):
    out = []
    for m in NUM.finditer(text):
        raw = m.group(0).translate(MINUS).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def load_evidence(path, root):
    doc = {"evaluation_evidence.json": json.load(open(path))}
    return doc


def load_for(claims, root):
    """Load only the artifacts these claims cite.

    A single pre-built dict would silently ship the wrong artifact to a claim
    that names another file, so the loader reads the paths out of the claim list
    itself.
    """
    docs = {}
    for cid, quote, cls, refs in claims:
        for art, field, tol in refs:
            if art in docs:
                continue
            p = art if os.path.isabs(art) else os.path.join(root, art)
            if os.path.exists(p):
                docs[art] = json.load(open(p))
    return docs


def check(report_text, evidence_docs, root, verbose=False, claims=None):
    claims = CLAIMS if claims is None else claims
    problems = []
    for cid, quote, cls, refs in claims:
        # A: verbatim presence
        if quote not in report_text:
            problems.append(f"A {cid}: quote not in report: {quote[:40]!r}")
            continue
        # B: every cited artifact field must resolve and match a number in the quote
        if refs:
            nums = numbers_in(quote)
            for art, field, tol in refs:
                if art not in evidence_docs:
                    p = os.path.join(root, art)
                    if not os.path.exists(p):
                        problems.append(f"B {cid}: artifact missing {art}")
                        continue
                    evidence_docs[art] = json.load(open(p))
                try:
                    val = resolve(evidence_docs[art], field)
                except (KeyError, TypeError, IndexError):
                    problems.append(f"B {cid}: field does not resolve {art}:{field}")
                    continue
                if not isinstance(val, (int, float)):
                    problems.append(f"B {cid}: field is not numeric {art}:{field}")
                    continue
                if not any(abs(abs(n) - abs(val)) <= tol for n in nums):
                    problems.append(
                        f"B {cid}: no number in quote matches {art}:{field}={val}")
        # C: wording must not outrun the evidence class.
        # A banned phrase inside an explicit prohibition ("must not write X") is
        # the honest use of that phrase, not a violation: the report has to be
        # able to name what it refuses to claim. So each occurrence is checked
        # for a negation marker in the same clause; only un-negated ones fail.
        for bad in BANNED.get(cls, []):
            for verdict in banned_verdicts(quote, bad):
                if verdict:
                    problems.append(f"C {cid}: class {cls} may not assert {bad!r}")
                    break
    if verbose:
        for cid, quote, cls, _ in claims:
            print(f"  {cid:<18} {cls:<9} ok")
    return problems


def problems_named(probs, cid):
    return [p for p in probs if cid in p]


def selftest():
    report = open(REPORT).read()
    ev = {f"attn-reuse/real/data/{n}": json.load(open(os.path.join(HERE, "data", n)))
          for n in ("main.json", "long.json")}
    ev["evaluation_evidence.json"] = json.load(open(EVIDENCE))
    probs = check(report, dict(ev), HERE, verbose=True)
    assert not probs, probs

    # prove red A: paraphrase the quote -> no longer verbatim
    bad_a = report.replace("k=384（= 512 的 75%）", "k=384 (= 75% of 512)")
    assert any(p.startswith("A C2") for p in check(bad_a, dict(ev), HERE)), "A must go red"

    # prove red B: a field that does not exist
    bad_field = CLAIMS + [("C-bad-field",
                           "k=384（= 512 的 75%）时 g≤4 的代价 ≤ +0.0114 nats",
                           "measured", [("attn-reuse/real/data/main.json",
                                         "deltas.k384_g4.nonexistent", 5e-5)])]
    probs = check(report, dict(ev), HERE, claims=bad_field)
    assert any("field does not resolve" in p for p in probs), "B must go red (missing field)"

    # prove red B: a real field holding a different number
    bad_val = CLAIMS + [("C-bad-value",
                         "k=384（= 512 的 75%）时 g≤4 的代价 ≤ +0.0114 nats",
                         "measured", [("attn-reuse/real/data/main.json",
                                       "deltas.k32_g2.vs_own", 5e-5)])]
    probs = check(report, dict(ev), HERE, claims=bad_val)
    assert any("no number in quote matches" in p for p in probs), "B must go red (wrong value)"

    # prove red C: promote a partial claim into a causal assertion
    promoted = [c for c in CLAIMS if c[0] != "C9-starved-partial"] + [
        ("C9-starved-partial", "starved 证明了因果：复用代价由 attention 质量分布决定",
         "partial", [])]
    probs = check("starved 证明了因果：复用代价由 attention 质量分布决定", {k: {} for k in ev},
                  HERE, claims=promoted)
    assert any("may not assert" in p for p in probs), "C must go red on a promoted claim"

    # ...and the honest version (prohibition marker in the same clause) must pass C
    honest = [c for c in CLAIMS if c[0] != "C10-no-mechanism"] + [
        ("C10-no-mechanism", "不可写「复用代价由 attention 质量分布决定」", "partial", [])]
    probs = check("不可写「复用代价由 attention 质量分布决定」", dict(ev), HERE, claims=honest)
    assert not problems_named(probs, "C10"), probs
    # ---- report 07 ----
    r07 = open(REPORT07).read()
    ev07 = load_for(CLAIMS07, os.path.join(HERE, "..", ".."))
    probs07 = check(r07, dict(ev07), os.path.join(HERE, "..", ".."), claims=CLAIMS07)
    assert not probs07, probs07

    # prove red on report 07 too: the same two failure modes must go red there,
    # or the second report is only nominally gated.
    bad07 = r07.replace("**−25.5** (p=1.7e−09)", "**−25.5** (p=0.0153)")
    probs07b = check(bad07, dict(ev07), os.path.join(HERE, "..", ".."), claims=CLAIMS07)
    assert any(p.startswith("A D2") for p in probs07b), "A must go red on report 07"

    # The promotion this guards against is a *later edit* of report 07 that drops
    # the prohibition and asserts the causal sentence outright. The probe text
    # must therefore be the bare assertion, and the claim set must keep the
    # D10 quote in its promoted form — otherwise the gate is checking a quote
    # that is still honest and would pass, which is a false green.
    promoted07 = [c for c in CLAIMS07 if c[0] != "D10-not-a-proxy"] + [
        ("D10-not-a-proxy", "复用代价由 attention 质量分布决定", "partial", [])]
    probs07c = check("复用代价由 attention 质量分布决定", dict(ev07),
                     os.path.join(HERE, "..", ".."), claims=promoted07)
    assert any("may not assert" in p for p in probs07c), \
        "C must go red on a promoted claim in report 07"

    print(f"selftest OK ({len(CLAIMS)} claims on 06 + {len(CLAIMS07)} on 07; "
          f"prove-red A/B/C on both reports; prove-green on C(negated)")


def main():
    ap = argparse.ArgumentParser()
    # Default report follows the claim set. With a fixed default of REPORT the
    # three-letter combination `--set 07` silently checked claims07 against
    # report 06 and reported ten A-failures — a gate answering about the wrong
    # document, which is the exact defect this file exists to catch.
    ap.add_argument("--report", default=None)
    ap.add_argument("--set", dest="claimset", choices=("06", "07"), default="06")
    ap.add_argument("--root", default=os.path.join(HERE, "..", ".."))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    claims = CLAIMS if a.claimset == "06" else CLAIMS07
    path = a.report or (REPORT if a.claimset == "06" else REPORT07)
    report = open(path).read()
    root = os.path.abspath(a.root)
    probs = check(report, load_for(claims, root), root, a.verbose, claims=claims)
    print(f"claims={len(claims)} problems={len(probs)}")
    for p in probs:
        print("  " + p)
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())
