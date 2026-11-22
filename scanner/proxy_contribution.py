#!/usr/bin/env python3
"""预注册检验：功能性贡献量 c_l 能否预测单层复用代价。

预注册文档：`attn-reuse/prereg/reuse-proxy-prereg.md`（判据先于数据，阈值 r >= 0.75）。

定义（与预注册一致，不得事后更改）：

    c_l = mean_t || W_U · h_l[t] ||_2

h_l 是第 l 层 attention 模块的输出（已经过它自己的 c_proj，即 O_l），
W_U 是 unembedding（GPT-2 为 tied 的 lm_head）。c_l 只依赖 O_l 与 W_U，
与「h_l 像不像 h_{l-1}」无关 —— 这是它能区别于 CKA 与索引重叠的全部理由。

用法：
    python3 scanner/proxy_contribution.py --selftest
    python3 scanner/proxy_contribution.py --run
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


def local_snapshot(repo: str) -> str | None:
    import glob as _glob
    import os as _os
    hits = _glob.glob(_os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo}/snapshots/*/"))
    return hits[0] if hits else None


def contribution(h: torch.Tensor, wu: torch.Tensor) -> float:
    """c = mean_t || W_U h[t] ||_2。h: (seq, d)，wu: (vocab, d)。"""
    logits = h.double() @ wu.double().T
    return float(logits.norm(dim=1).mean())


def collect(model_repo: str, max_tokens: int = 1024):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = local_snapshot(model_repo) or model_repo
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()

    ids = tok(TEXT, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids
    n_layers = model.config.n_layer
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx):
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            store[idx] = o.detach()[0]
        return hook

    for i in range(n_layers):
        handles.append(model.transformer.h[i].attn.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    wu = model.lm_head.weight.detach()
    seq = ids.shape[1]
    step = max(1, seq // 800)
    sub = slice(0, seq, step)
    cvals = {i: contribution(store[i][sub], wu) for i in store}
    return {"repo": model_repo, "n_layers": n_layers, "seq": seq,
            "c": {str(k): v for k, v in cvals.items()}}


def run():
    out = {}
    for repo in ("openai-community--gpt2", "distilbert--distilgpt2"):
        r = collect(repo)
        out[repo] = r
        print(f"{repo}: {r['n_layers']} layers, seq={r['seq']}")
        for k in sorted(r["c"], key=int):
            print(f"  layer {k:>2}  c = {r['c'][k]:.4f}")
    return out


def selftest():
    g = torch.Generator().manual_seed(0)
    seq, d, vocab = 200, 64, 1000
    h = torch.randn(seq, d, generator=g)
    wu = torch.randn(vocab, d, generator=g)

    z = torch.zeros_like(h)
    print(f"  ok    c(0) = {contribution(z, wu):.6f}")
    assert contribution(z, wu) == 0.0, "零输出必为零贡献"

    c1, c3 = contribution(h, wu), contribution(3 * h, wu)
    print(f"  ok    c(h) = {c1:.4f}, c(3h) = {c3:.4f} (ratio {c3 / c1:.6f})")
    assert abs(c3 / c1 - 3.0) < 1e-6, "对缩放必须线性"

    perm = torch.randperm(seq, generator=g)
    print(f"  ok    c(permuted) = {contribution(h[perm], wu):.6f}")
    assert abs(contribution(h[perm], wu) - c1) < 1e-9, "mean 必须对顺序不变"

    import statistics
    rs = []
    for _ in range(200):
        a = torch.randn(11, generator=g)
        b = torch.randn(11, generator=g)
        rs.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    med = statistics.median(rs)
    print(f"  ok    null median r = {med:+.4f}")
    assert abs(med) < 0.06, f"独立随机的相关中位数应接近 0，得 {med}"

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
        (DATA / "contribution.json").write_text(json.dumps(res, indent=1))
        print(f"wrote {DATA / 'contribution.json'}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
