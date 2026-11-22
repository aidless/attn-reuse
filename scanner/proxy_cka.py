#!/usr/bin/env python3
"""CKA 作为复用代价代理量的检验。

动机（`reports/03-reuse-cost.md` §6）：索引重叠在 0.29–0.56 区间内
不预测复用代价（r = −0.055）。那么什么预测？

他山之石：Kornblith et al. 2019 的 CKA 是「表示相似度」的标准工具，
Bansal et al. 2021 的 model stitching 表明表示相似度高不等于可替换。
本脚本把 CKA 搬到「相邻层 attention 输出的可替换性」上：若 CKA 能预测
复用代价，它就是可用的廉价代理量（不必前向算两次 loss）；若不能，
则是对 MLI/CLSA「重叠高即可复用」的第二条独立反驳。

用法：
    python3 proxy_cka.py --selftest
    python3 proxy_cka.py --run          # 输出 data/cka_vs_reuse.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

TEXT = (
    "The question of how much computation a language model actually needs to "
    "answer a given prompt has become central to inference-time efficiency. "
    "Sparse attention reduces the cost of the quadratic interaction by "
    "selecting a small subset of keys for each query, and the selection is "
    "computed by a lightweight indexer that itself costs very little. Recent "
    "work observes that adjacent layers often select nearly the same set of "
    "tokens, and proposes to reuse one layer's selection in the next several "
    "layers. The efficiency argument is straightforward: one indexer serves "
    "many layers, so the per-layer selection cost disappears. What is not "
    "established is whether the selected set is also the set that matters for "
    "the computation performed by those layers. Selection overlap is a "
    "property of positions; the attention output is a weighted combination of "
    "values, and two layers can pick the same positions while assigning them "
    "very different weights or reading different value projections. This "
    "distinction matters because a reuse scheme that is safe in one regime "
    "can silently degrade another, and the degradation does not show up in "
    "the overlap statistic that motivated the scheme in the first place. "
) * 4


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """线性核 CKA（Kornblith 2019, eq. 3 的线性特例）。

    x, y: (n, d1), (n, d2)，n 为样本数（同一批 token 位置）。
    """
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    k = x @ x.T
    l = y @ y.T
    kt_l = (k * l).sum()          # <K, L>_F，K/L 对称
    kt_k = (k * k).sum()
    lt_l = (l * l).sum()
    denom = (kt_k * lt_l).sqrt()
    if denom <= 0:
        return 0.0
    return float(kt_l / denom)


def local_snapshot(repo: str) -> str | None:
    """直接给出本地 snapshot 目录。

    HF_HUB_OFFLINE=1 时 transformers 5.x 仍会尝试解析 refs/main，
    某些缓存缺 refs 就会误报「不在缓存里」。给定目录绕过这一步。
    """
    import glob as _glob
    import os as _os
    hits = _glob.glob(_os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo}/snapshots/*/"))
    return hits[0] if hits else None


def collect(model_name: str, max_tokens: int = 1000):
    """返回每层 attention 输出的 (n, d) 采样。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = local_snapshot(model_name) or model_name
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, output_hidden_states=False)
    model.eval()

    ids = tok(TEXT, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids
    n_layers = model.config.n_layer
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx):
        def hook(_mod, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            store[idx] = o.detach()[0]          # (seq, d)
        return hook

    for i in range(n_layers):
        handles.append(model.transformer.h[i].attn.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    seq = ids.shape[1]
    step = max(1, seq // 800)
    sub = slice(0, seq, step)
    return model_name, n_layers, {i: v[sub] for i, v in store.items()}, seq


def run() -> dict:
    out = {}
    for name in ("openai-community--gpt2", "distilbert--distilgpt2"):
        mname, n_layers, acts, seq = collect(name)
        pairs = []
        for l in range(1, n_layers):
            pairs.append({"source_layer": l - 1, "reused_layer": l,
                          "cka": linear_cka(acts[l - 1], acts[l])})
        out[name] = {"n_layers": n_layers, "seq": seq, "pairs": pairs}
        print(f"{name}: {n_layers} layers, {len(pairs)} adjacent pairs")
        for p in pairs:
            print(f"  {p['source_layer']:2d}->{p['reused_layer']:2d}  CKA={p['cka']:.4f}")
    return out


def selftest() -> None:
    # n 必须显著大于 d：线性 CKA 在有限样本下对独立随机矩阵有正偏
    # （n=200,d=64 时偏到 0.232；n=2000,d=64 时降到 0.023），
    # 报 CKA 必须同时报 n 与 d，否则数字不可比。
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2000, 64, generator=g)
    print(f"  ok    CKA(x,x) = {linear_cka(x, x):.6f}")
    assert abs(linear_cka(x, x) - 1.0) < 1e-6, "自相似必须为 1"

    print(f"  ok    CKA(x, 3x) = {linear_cka(x, 3 * x):.6f}")
    assert abs(linear_cka(x, 3 * x) - 1.0) < 1e-6, "线性 CKA 对尺度不变"

    y = torch.randn(2000, 64, generator=g)
    c = linear_cka(x, y)
    print(f"  ok    CKA(x, random y) = {c:.4f}")
    assert abs(c) < 0.05, f"n=2000 时独立随机应接近 0，得 {c}"

    # 负向控制：只对一半维度做置换，CKA 仍应显著下降
    z = x.clone()
    z[:, 32:] = z[torch.randperm(2000, generator=g), 32:]
    cp = linear_cka(x, z)
    print(f"  ok    CKA(x, half-permuted x) = {cp:.4f}")
    assert cp < 0.8, f"半置换必须明显低于 1，得 {cp}"

    # 平移不变
    s = x + 5.0
    cs = linear_cka(x, s)
    print(f"  ok    CKA(x, x+5) = {cs:.6f}")
    assert abs(cs - 1.0) < 1e-6, "均值中心化后平移不变"

    print("SELFTEST PASSED")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.run:
        res = run()
        DATA.mkdir(exist_ok=True)
        (DATA / "cka_vs_reuse.json").write_text(json.dumps(res, indent=1))
        print(f"wrote {DATA / 'cka_vs_reuse.json'}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
