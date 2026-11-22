#!/usr/bin/env python3
"""Emit evaluation_evidence.json for the cross-layer reuse experiments.

Why this file exists
--------------------
Upstream (SakanaAI/AI-Scientist-v3 docs/evaluation_evidence.md) draws a line that
this project needed: an evaluator must publish *per-episode* evidence, and an
evaluator that cannot must report `unavailable` rather than pass silently. The
failure it prevents is a downstream phase reading "the run completed" as "the
numbers are supported".

Our situation is worse than a missing duration: our readings live in several
json files written by three different scripts, and the report cited them
through a hand-maintained fact table. So this script does three things:

1. one episode per (model, seq, k, group, mode) cell actually measured, with
   the NLL, the delta, and the control verdict that cell depends on;
2. an `availability` block that names, per field, whether it was RECORDED
   (present in the artifacts), DERIVED (computed from recorded fields, with the
   formula given), or UNAVAILABLE (not recorded — null, never a guess);
3. a hard rule enforced in code: a cell whose control did not pass is emitted
   with `correct: false` and is refused by `check()` for citation, so no report
   can quote it.

`solution` is our equivalent of v3's retained output: for a reuse cell it is
the pair (source layer, target layers) that was actually borrowed, and for an
own-* cell it is the k that was kept. `candidate_diagnostics` carries what the
harness measured about the selection itself.

Run: python3 evidence.py --out evaluation_evidence.json
     python3 evidence.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Fields the schema promises, and where each one comes from. Anything not in
# RECORDED is reported UNAVAILABLE in the output rather than synthesised.
RECORDED = {
    "score": "readings[*] / deltas[*vs_own] in {main,main_15b,long}.json",
    "control": "readings['control_own_kqlen'] - readings['full']",
    "n_layers": "top-level n_layers",
    "seq": "top-level seq",
    "n_batches": "top-level n_batches",
    "model": "top-level model path",
}
DERIVED = {
    "ppl": "exp(score); unit conversion only, no new information",
    "delta_ppl_pct": "100*(exp(delta)-1) when the row is a delta",
    "envelope_tex": "label from (model, seq, k, group) — identifier, not a measurement",
}
# Deliberately null: we never timed the forward passes and never recorded VRAM.
UNAVAILABLE = ["duration_seconds", "peak_vram_bytes", "wall_clock_started_at"]


def _load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def _sha(path):
    with open(os.path.join(DATA, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p for b right->wrong and c wrong->right flips."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if n <= 900:
        return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    lg = math.lgamma
    tail = sum(math.exp(lg(n + 1) - lg(i + 1) - lg(n - i + 1) - n * math.log(2.0))
               for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def _control_ok(doc):
    """The one assertion everything else depends on: with k == seq_len the
    sparse path must reproduce the dense path bit-for-bit."""
    r = doc["readings"]
    return abs(r["control_own_kqlen"] - r["full"]) < 1e-9, r["control_own_kqlen"] - r["full"]


def build():
    docs = {"main": _load("main.json"), "main_15b": _load("main_15b.json"),
            "long": _load("long.json")}
    episodes, skipped = [], []

    for tag, doc in docs.items():
        ok, dev = _control_ok(doc)
        model = os.path.basename(doc["model"])
        base = {"envelope_tex": f"{tag}", "model": model, "seq": doc["seq"],
                "n_layers": doc["n_layers"], "n_batches": doc["n_batches"],
                "control_deviation": dev, "control_passed": ok}
        for key, cell in doc["deltas"].items():
            if not key.startswith("k"):
                continue  # only the vs_own grid is a reuse measurement
            k_str, g_str = key.split("_")
            k, group = int(k_str[1:]), int(g_str[1:])
            own = doc["readings"][f"own_{k_str}"]
            rec = dict(base, k=k, group=group, mode="reuse",
                       score=cell["vs_own"],
                       score_absolute=doc["readings"][f"reuse_{k_str}_{g_str}"],
                       own_absolute=own,
                       ppl=math.exp(own + cell["vs_own"]),
                       delta_ppl_pct=100.0 * (math.exp(cell["vs_own"]) - 1.0),
                       solution={"borrowed_from_layer": (0 // group) * group,
                                 "target_layers": [i for i in range(1, doc["n_layers"])
                                                   if i % group != 0]},
                       candidate_diagnostics={"own_nll": own,
                                              "reuse_nll": doc["readings"][f"reuse_{k_str}_{g_str}"]},
                       correct=ok)
            episodes.append(rec)
            if not ok:
                skipped.append(f"{tag}:{key}")

    # own-* cells: the sparsification-only reading (no reuse). Distinct mode so a
    # reader cannot mistake them for a reuse result.
    for tag, doc in docs.items():
        ok, dev = _control_ok(doc)
        model = os.path.basename(doc["model"])
        for key, val in doc["readings"].items():
            if not key.startswith("own_k"):
                continue
            k = int(key[len("own_k"):])
            episodes.append({
                "envelope_tex": tag, "model": model, "seq": doc["seq"],
                "n_layers": doc["n_layers"], "n_batches": doc["n_batches"],
                "control_deviation": dev, "control_passed": ok,
                "k": k, "group": 1, "mode": "own",
                "score": val - doc["readings"]["full"],
                "score_absolute": val,
                "own_absolute": val,
                "ppl": math.exp(val),
                "delta_ppl_pct": 100.0 * (math.exp(val - doc["readings"]["full"]) - 1.0),
                "solution": {"kept": k, "of": doc["seq"]},
                "candidate_diagnostics": {"full_nll": doc["readings"]["full"]},
                "correct": ok,
            })

    # Single-layer-pair borrowings: one episode per (target, source) pair. These
    # carry the leader-identity result, so they must live in the same evidence
    # file as the group cells or the claim gate cannot cite them uniformly.
    pairs = []
    with open(os.path.join(DATA, "pairs_wide.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "pair":
                continue
            pairs.append(r)
    margin = {}
    mpath = os.path.join(DATA, "margin_512.json")
    if os.path.exists(mpath):
        for p in _load("margin_512.json")["pairs"]:
            margin[(p["target"], p["source"])] = p

    for r in pairs:
        t, s = r["target"], r["source"]
        m = margin.get((t, s), {})
        own256 = docs["main"]["readings"]["own_k256"]
        episodes.append({
            "envelope_tex": "pairs_wide",
            "model": "Qwen2.5-0.5B", "seq": 512, "n_layers": docs["main"]["n_layers"],
            "n_batches": docs["main"]["n_batches"],
            "control_deviation": _control_ok(docs["main"])[1],
            "control_passed": _control_ok(docs["main"])[0],
            "k": r["k"], "group": 1, "mode": "pair_reuse",
            "target_layer": t, "source_layer": s, "distance": r["distance"],
            "score": r["vs_own"], "score_absolute": r["nll"],
            "own_absolute": own256,
            "ppl": math.exp(r["nll"]),
            "delta_ppl_pct": 100.0 * (math.exp(r["vs_own"]) - 1.0),
            "solution": {"borrowed_from_layer": s, "target_layers": [t]},
            "candidate_diagnostics": {"starved": m.get("starved"),
                                      "own_nll": own256, "reuse_nll": r["nll"]},
            "correct": _control_ok(docs["main"])[0],
        })

    # Decision-level cells (report 07). These carry a different control verdict —
    # the MCQ instrument's own k==full-length control — so the episode records it
    # separately instead of inheriting the NLL envelope's.
    for name, tag in (("mcq_05b.json", "mcq_05b"),
                      ("mcq_15b.json", "mcq_15b"),
                      ("mcq_7b.json", "mcq_7b"),
                      ("arcc_15b.json", "arcc_15b"),
                      ("arcc_7b.json", "arcc_7b"),
                      ("arcc_7b_k64.json", "arcc_7b_k64"),
                      # The shot-matched control: ARC-Easy at the 5 exemplars
                      # that actually fit when the stems are long. It is a
                      # control, not a fourth benchmark, so its envelope tag
                      # says so.
                      ("arc_easy_5shot.json", "arc_easy_5shot_control")):
        if not os.path.exists(os.path.join(DATA, name)):
            continue
        doc = _load(name)
        ctl = doc["control"]
        dev = ctl.get("max_abs_score_deviation", 0.0)
        ok = abs(dev) < 1e-9 and not ctl.get("flips", {}).get("right_to_wrong") \
            and not ctl.get("flips", {}).get("wrong_to_right")
        base = {"envelope_tex": tag, "model": os.path.basename(doc["model"]),
                "seq": doc["seq"], "bench": doc["bench"], "shots": doc["shots"],
                "n_items": doc["n_items"],
                "control_deviation": dev, "control_passed": ok}
        for key, fl in doc["flips_vs_own"].items():
            k_str, g_str = key.split("_")
            k, group = int(k_str[1:]), int(g_str[1:])
            own_acc = doc["readings"][f"own_k{k}"]
            reuse_acc = doc["readings"][f"reuse_{k_str}_{g_str}"]
            b, c = fl["right_to_wrong"], fl["wrong_to_right"]
            episodes.append(dict(
                base, k=k, group=group, mode="decision",
                score=reuse_acc - own_acc,          # accuracy delta
                score_absolute=reuse_acc, own_absolute=own_acc,
                dense_absolute=doc["readings"]["full"],
                right_to_wrong=b, wrong_to_right=c, discordant=b + c,
                net_loss=b - c,
                p_exact=mcnemar_exact(b, c),
                # NOT an NLL delta: this is the mean per-option log-prob shift the
                # MCQ instrument records. Naming it `nll_delta` would have made a
                # reader compare it against the wikitext token-NLL grid, which is a
                # different quantity on a different corpus.
                mean_opt_logprob_delta_vs_own=
                    (doc.get("score_delta_vs_own") or {}).get(key),
                borrows=(doc.get("borrows") or {}).get(key),
                fallbacks=(doc.get("fallbacks") or {}).get(key),
                solution={"borrowed_from_layer": "group leader", "group": group},
                candidate_diagnostics={"prompt_len_median": doc["prompt_len_median"],
                                       "full_len_max": doc["full_len_max"]},
                correct=ok,
            ))
            if not ok:
                skipped.append(f"{tag}:{key}")

    availability = {}
    for field in RECORDED:
        availability[field] = {"status": "recorded", "from": RECORDED[field]}
    for field in DERIVED:
        availability[field] = {"status": "derived", "formula": DERIVED[field]}
    for field in UNAVAILABLE:
        availability[field] = {"status": "unavailable",
                               "detail": "not measured by the harness; emitted as null, "
                                         "never estimated"}
    for ep in episodes:
        for field in UNAVAILABLE:
            ep[field] = None

    per_env = {}
    for ep in episodes:
        per_env.setdefault(ep["envelope_tex"], []).append(ep["correct"])

    return {
        "schema_version": 1,
        "generated_by": "attn-reuse/real/evidence.py",
        "artifacts": {n: {"path": f"attn-reuse/real/data/{n}", "sha256": _sha(n)}
                      for n in ("main.json", "main_15b.json", "long.json",
                                "mcq_05b.json", "mcq_15b.json", "mcq_7b.json",
                                "arcc_15b.json", "arcc_7b.json",
                                "arcc_7b_k64.json", "arc_easy_5shot.json",
                                "probe_arcc_05b.json", "mmlu_probe.json",
                                "cross_bench.json")
                      if os.path.exists(os.path.join(DATA, n))},
        "availability": availability,
        "episodes": episodes,
        "summary": {
            "n_episodes": len(episodes),
            "n_decision": sum(1 for e in episodes if e["mode"] == "decision"),
            "n_reuse": sum(1 for e in episodes if e["mode"] == "reuse"),
            "n_own": sum(1 for e in episodes if e["mode"] == "own"),
            "n_pair": sum(1 for e in episodes if e["mode"] == "pair_reuse"),
            "control_failed_cells": skipped,
            "all_controls_passed": not skipped and all(per_env.values()),
        },
        "citation_rule": ("A cell may be quoted only when correct == true. "
                          "correct is false whenever |control_own_kqlen - full| >= 1e-9, "
                          "because the harness would then be manufacturing its own error."
                          ),
    }


def check(doc):
    """Refuse to let an incorrect cell be cited. Returns (ok, reasons)."""
    reasons = []
    for ep in doc["episodes"]:
        if ep["mode"] in ("reuse", "pair_reuse") and not ep["control_passed"]:
            reasons.append(f"{ep['envelope_tex']} k={ep['k']} g={ep['group']}: "
                           f"control deviation {ep['control_deviation']:.3e}")
    for field, meta in doc["availability"].items():
        if meta["status"] not in ("recorded", "derived", "unavailable"):
            reasons.append(f"availability[{field}] has unknown status {meta['status']!r}")
    if not doc["summary"]["all_controls_passed"]:
        reasons.append("summary says not all controls passed")
    return (not reasons), reasons


def selftest():
    doc = build()
    ok, reasons = check(doc)
    assert ok, reasons

    # 1. every reuse episode carries the control verdict that produced it
    assert all("control_passed" in e for e in doc["episodes"])
    # 2. unrecorded fields are null everywhere, not zero, not absent
    for f in UNAVAILABLE:
        assert all(e[f] is None for e in doc["episodes"]), f
    # 3. an own-k cell must not masquerade as a reuse cell
    assert all(e["group"] == 1 and e["mode"] == "own"
               for e in doc["episodes"] if e["mode"] == "own")
    # 4. derived ppl must agree with exp(score) applied to the recorded absolute
    for e in doc["episodes"]:
        if e["mode"] in ("reuse", "pair_reuse"):
            assert abs(e["ppl"] - math.exp(e["score_absolute"])) < 1e-9, e
    # 5. prove-red: corrupt the control and the checker must refuse
    bad = json.loads(json.dumps(doc))
    bad["episodes"][0]["control_passed"] = False
    bad["episodes"][0]["control_deviation"] = 11.6
    assert check(bad)[0] is False, "corrupted control must be refused"
    # 6. prove-red: an unknown availability status must be refused
    bad2 = json.loads(json.dumps(doc))
    bad2["availability"]["score"] = {"status": "guessed"}
    assert check(bad2)[0] is False, "guessed availability must be refused"
    # 7. the same value must not be silently reused across envelopes: the three
    #    envelopes differ in seq, and a reader relies on that to tell them apart
    seqs = {e["envelope_tex"]: e["seq"] for e in doc["episodes"]}
    assert len(set(seqs.values())) >= 2, seqs
    # 8. decision episodes must not be mistakable for NLL episodes: their score is
    #    an accuracy difference bounded by 1, while an NLL delta can exceed it
    dec = [e for e in doc["episodes"] if e["mode"] == "decision"]
    assert dec, "decision episodes must be present"
    assert all(abs(e["score"]) <= 1.0 for e in dec)
    assert all(e["mode"] != "reuse" or abs(e["score"]) < 100 for e in doc["episodes"]
               if e["mode"] == "reuse")
    # 9. a decision cell's flip counts must reproduce its own p-value
    for e in dec:
        assert e["net_loss"] == e["right_to_wrong"] - e["wrong_to_right"]
        assert e["discordant"] == e["right_to_wrong"] + e["wrong_to_right"]
        assert abs(e["p_exact"] - mcnemar_exact(e["right_to_wrong"],
                                                e["wrong_to_right"])) < 1e-12
    print(f"selftest OK ({len(doc['episodes'])} episodes "
          f"({len(dec)} decision, {doc['summary']['n_reuse']} reuse), "
          f"{len(UNAVAILABLE)} fields honestly null, 2 prove-red cases)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "evaluation_evidence.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    doc = build()
    ok, reasons = check(doc)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"WROTE {a.out}")
    print(f"episodes={doc['summary']['n_episodes']} "
          f"(reuse={doc['summary']['n_reuse']} own={doc['summary']['n_own']} "
          f"pair={doc['summary']['n_pair']} decision={doc['summary']['n_decision']}) "
          f"controls={doc['summary']['all_controls_passed']}")
    for r in reasons:
        print(f"  REFUSED {r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
