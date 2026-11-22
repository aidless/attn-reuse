#!/usr/bin/env python3
"""按预注册判据判定 c_l 与复用代价的关系，并跑层深混淆检查。

判据来源：`attn-reuse/prereg/reuse-proxy-prereg.md` §4。
本脚本只做判定，不改判据。

用法：python3 attn-reuse/scanner/analyze_contribution.py [--json]
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    return pearson(rank(x), rank(y))


PERMUTATIONS = 1000
PERM_SEED = 42


def perm_control(c, d, n_perm=None, seed=None):
    """预注册负向控制 1：打乱 c_l 的层序，实际 r 须在零分布 99 分位外。

    返回 (实际 r, 零分布 99 分位, 是否通过)。置换只打乱 c 的顺序，d 不动，
    等价于「把 c_l 与它所属的层解耦」，破坏任何真实的层-代价对应关系。
    """
    n_perm = PERMUTATIONS if n_perm is None else n_perm
    seed = PERM_SEED if seed is None else seed
    actual = pearson(c, d)
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        shuffled = list(c)
        rng.shuffle(shuffled)
        null.append(pearson(shuffled, d))
    null_sorted = sorted(null)
    q99 = null_sorted[int(0.99 * (len(null_sorted) - 1))]
    # 两侧都要：一个显著为负的相关同样在零分布的 1 分位之外
    q01 = null_sorted[int(0.01 * (len(null_sorted) - 1))]
    passed = actual > q99 or actual < q01
    return actual, q99, q01, passed


def analyse():
    contrib = json.loads((DATA / "contribution.json").read_text())
    delta = json.loads((DATA / "reuse_single.json").read_text())["per_layer_mean_delta"]
    cka = json.loads((DATA / "cka_vs_reuse.json").read_text())["openai-community--gpt2"]["pairs"]
    ckamap = {p["reused_layer"]: p["cka"] for p in cka}

    g = contrib["openai-community--gpt2"]["c"]
    layers = sorted((int(k) for k in delta), key=int)
    c_vals = [g[str(l)] for l in layers]
    d_vals = [delta[str(l)] for l in layers]

    def stats(layers_subset):
        c = [g[str(l)] for l in layers_subset]
        d = [delta[str(l)] for l in layers_subset]
        lc = [math.log(v) for v in c]
        return {
            "n": len(c),
            "pearson_raw": pearson(c, d),
            "pearson_log": pearson(lc, d),
            "spearman": spearman(c, d),
            "corr_log_layer": pearson(lc, [float(l) for l in layers_subset]),
            "corr_raw_layer": pearson(c, [float(l) for l in layers_subset]),
        }

    full = stats(layers)
    excl = stats(layers[1:])

    # 增量：c_l 对 CKA 的独立信息
    ck = [ckamap[l] for l in layers if l in ckamap]
    cc = [g[str(l)] for l in layers if l in ckamap]
    inc = {"corr_c_cka": pearson(cc, ck)}

    # 判据
    if full["pearson_raw"] >= 0.75 and full["spearman"] >= 0.70:
        verdict = "成立"
    elif 0.55 <= full["pearson_raw"] < 0.75:
        verdict = "边界"
    else:
        verdict = "否决"

    guard3 = abs(full["corr_log_layer"]) < 0.5

    actual, q99, q01, perm_pass = perm_control(c_vals, d_vals)

    return {"full": full, "excl_first": excl, "incremental": inc,
            "verdict": verdict, "guard3_layercorr_pass": guard3,
            "perm_control": {"actual_r": actual, "q99": q99, "q01": q01,
                             "n_perm": PERMUTATIONS, "seed": PERM_SEED,
                             "pass": perm_pass},
            "threshold": 0.75}


def selftest():
    # 零分布必须真的能红：给一对与 d 无关的 c，置换控制不许通过
    d = [float(i) for i in range(12)]
    rng = random.Random(7)
    c_null = [rng.random() for _ in range(12)]
    actual, q99, q01, passed = perm_control(c_null, d, n_perm=400, seed=1)
    assert not passed, f"unrelated c must not clear the null: r={actual:+.3f}"
    # 而一对强相关必须能过
    c_strong = [3.0 * v + 1.0 for v in d]
    actual, q99, q01, passed = perm_control(c_strong, d, n_perm=400, seed=1)
    assert passed, f"perfectly correlated c must clear the null: r={actual:+.3f}"
    # 置换确实改变了 r（不是空转）
    c = [float(i) ** 1.5 for i in range(12)]
    null = []
    rng = random.Random(42)
    for _ in range(50):
        sh = list(c)
        rng.shuffle(sh)
        null.append(pearson(sh, d))
    assert len({round(v, 9) for v in null}) > 10, "permutations must actually permute"
    print("selftest OK (permutation control goes red on unrelated c and green on "
          "correlated c; shuffling really changes r)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    r = analyse()
    if a.json:
        print(json.dumps(r, indent=1))
        return 0

    print("预注册判据（prereg/reuse-proxy-prereg.md §4）")
    print(f"  主判据阈值：Pearson r >= 0.75 且 Spearman >= 0.70")
    print()
    for name, s in (("全部层对", r["full"]), ("排除 0->1", r["excl_first"])):
        print(f"{name}（n={s['n']}）")
        print(f"  Pearson raw   {s['pearson_raw']:+.3f}")
        print(f"  Pearson log   {s['pearson_log']:+.3f}")
        print(f"  Spearman      {s['spearman']:+.3f}")
        print(f"  corr(log c, layer) {s['corr_log_layer']:+.3f}   [控制 3 要求 < 0.5]")
        print()
    print(f"增量检查：corr(c, CKA) = {r['incremental']['corr_c_cka']:+.3f}")
    print()
    pm = r["perm_control"]
    print(f"控制 1（置换零分布，{pm['n_perm']} 次，seed={pm['seed']}）："
          f"实际 r={pm['actual_r']:+.3f}，零分布 99 分位={pm['q99']:+.3f} / "
          f"1 分位={pm['q01']:+.3f} -> {'通过' if pm['pass'] else '失败'}")
    print(f"控制 3（层深混淆）：{'通过' if r['guard3_layercorr_pass'] else '失败'}")
    print(f"判定：{r['verdict']}")
    if r["verdict"] == "否决":
        print("  -> 按预注册 §6，这条线停，不跑 stitching / patching。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
