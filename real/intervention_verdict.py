"""Mechanical verdict for prereg/intervention-prereg.md (Goal Round 4).

Reads the two intervention result JSONs and applies the decision rule that was
FROZEN before any run. No hand-picked verdict. Decision rule (verbatim):
- INCONCLUSIVE: manipulation check fails, harness own(k=seq)==full control
  fails, or LOW accidentally equals BASE (negative controls).
- SUPPORTED: both models show NLL(HIGH)<NLL(BASE)<NLL(LOW) in every group,
  with manipulation checks passing.
- PARTIAL: ordering holds in one model only, or only HIGH<BASE (LOW flat).
- REFUTED: ordering reversed or flat in both models with checks passing.
"""
import argparse, json

EPS = 1e-6


def group_verdict(out):
    """Verdict for one (model, group) cell, given one groups[g] dict."""
    if not out.get("manipulation_check", False):
        return "INCONCLUSIVE", "manipulation check failed"
    if not out.get("low_distinct_from_base", True):
        return "INCONCLUSIVE", "LOW equals BASE (negative control failed)"
    hi, ba, lo = out["nll"]["high"], out["nll"]["base"], out["nll"]["low"]
    if hi < ba - EPS and ba < lo - EPS:
        return "SUPPORTED", "HIGH<BASE<LOW"
    if hi < ba - EPS and abs(ba - lo) <= EPS:
        return "PARTIAL", "only HIGH<BASE (LOW flat)"
    if hi < ba - EPS and ba > lo + EPS:
        return "PARTIAL", "HIGH<BASE but reversed LOW"
    if hi > ba + EPS and ba > lo + EPS:
        return "REFUTED", "ordering fully reversed"
    return "REFUTED", "flat or unsupported ordering"


def model_verdict(result):
    """Aggregate a model's groups; harness control gates everything."""
    if not result.get("harness_pass", False):
        return "INCONCLUSIVE", "harness own(k=seq)!=full", {}
    cells = {}
    for g, out in result["groups"].items():
        cells[g] = group_verdict(out)
    vs = [v for v, _ in cells.values()]
    if all(v == "SUPPORTED" for v in vs):
        return "SUPPORTED", "all groups HIGH<BASE<LOW", cells
    if all(v == "INCONCLUSIVE" for v in vs):
        return "INCONCLUSIVE", "all groups inconclusive", cells
    if any(v == "SUPPORTED" for v in vs):
        return "PARTIAL", "some groups supported, some not", cells
    if any(v == "PARTIAL" for v in vs):
        return "PARTIAL", "partial ordering in some groups", cells
    return "REFUTED", "no group supports the ordering", cells


def overall(results):
    """results: list of (name, result_dict)."""
    per = {}
    for name, res in results:
        per[name] = model_verdict(res)
    vs = [v[0] for v in per.values()]
    if any(v == "INCONCLUSIVE" for v in vs):
        verdict = "INCONCLUSIVE"
    elif all(v == "SUPPORTED" for v in vs):
        verdict = "SUPPORTED"
    elif all(v == "REFUTED" for v in vs):
        verdict = "REFUTED"
    else:
        verdict = "PARTIAL"
    return verdict, per


def _cell(hi, ba, lo, manip=True, lowdist=True):
    return {"nll": {"high": hi, "base": ba, "low": lo},
            "manipulation_check": manip, "low_distinct_from_base": lowdist}


def selftest():
    ok = {"harness_pass": True,
          "groups": {"2": _cell(2.82, 2.87, 4.26), "3": _cell(2.82, 2.83, 5.03)}}
    bad_harness = dict(ok, harness_pass=False)
    bad_manip = {"harness_pass": True, "groups": {"2": _cell(2.82, 2.87, 4.26, manip=False)}}
    bad_low = {"harness_pass": True, "groups": {"2": _cell(2.82, 2.87, 4.26, lowdist=False)}}
    refuted = {"harness_pass": True, "groups": {"2": _cell(4.26, 2.87, 2.82)}}
    partial = {"harness_pass": True, "groups": {"2": _cell(2.82, 2.87, 2.87)}}
    # positive
    assert overall([("a", ok), ("b", ok)])[0] == "SUPPORTED"
    # one model missing -> PARTIAL
    assert overall([("a", ok), ("b", refuted)])[0] == "PARTIAL"
    # negative controls -> INCONCLUSIVE even if ordering perfect
    assert overall([("a", bad_harness), ("b", ok)])[0] == "INCONCLUSIVE"
    assert overall([("a", bad_manip), ("b", ok)])[0] == "INCONCLUSIVE"
    assert overall([("a", bad_low), ("b", ok)])[0] == "INCONCLUSIVE"
    # reversed -> REFUTED
    assert overall([("a", refuted), ("b", refuted)])[0] == "REFUTED"
    # only HIGH<BASE -> PARTIAL
    assert overall([("a", partial), ("b", partial)])[0] == "PARTIAL"
    print("SELFTEST OK: SUPPORTED / PARTIAL / REFUTED / 3x INCONCLUSIVE all fire")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    results = []
    for path in a.inputs:
        d = json.load(open(path))
        results.append((d["model"].split("/")[-1], d))
    verdict, per = overall(results)
    out = {"verdict": verdict, "models": {}, "eps": EPS,
           "inputs": a.inputs}
    for name, (mv, why, cells) in per.items():
        out["models"][name] = {"verdict": mv, "reason": why,
                               "groups": {g: {"verdict": v, "reason": r}
                                          for g, (v, r) in cells.items()}}
    print("VERDICT:", verdict)
    for name, m in out["models"].items():
        print("  %s: %s (%s)" % (name, m["verdict"], m["reason"]))
        for g, c in m["groups"].items():
            print("    g=%s %s: %s" % (g, c["verdict"], c["reason"]))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
