#!/usr/bin/env python3
"""把「选择重叠」接到「复用的代价」上：组内跨层复用 attention 输出，测语言建模损失的变化。

为什么需要这一步：
    reports/01、02 只证明了一件否定的事——「相邻层选同样的 token」这个动机论据缺少对照，
    它能被同层现象解释。但动机弱不等于效果差：即使重叠高得可疑，复用也可能几乎无损。

    所以必须直接测效果。DeepSeek-V4.1 的 CSA2 配置是每 6 层一组、首层 Full、其余 5 层 Reuse
    （见 deepseek-research/evidence/v41_architecture_facts.md）。在没有真实稀疏 indexer 的
    情况下，能在 dense 模型上做的最接近的代理是：把组内后续层的 attention 输出直接替换成
    组首层的输出——这正是「复用」在计算图上的含义，只不过被复用的是一个 dense 全量输出而
    不是一组 top-k 索引。

    这个代理偏保守：dense 全量输出携带的信息比 top-k 索引下的加权和更多，所以真实系统里
    复用索引的代价预计不会比这里更大。反过来也不成立——这里的绝对损失值不能外推到 552B 模型。

用法:
    python3 reuse_effect.py --selftest
    python3 reuse_effect.py --model openai-community/gpt2 --group 6 --json data/reuse_gpt2_g6.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

TEXTS = [
    ("The history of the printing press begins with the recognition that a single "
     "reusable matrix of metal type could replace the labor of copying a manuscript by hand. "
     "Within fifty years of Gutenberg's first experiments, presses were operating in more than "
     "two hundred European towns, and the price of a printed book had fallen far enough that "
     "ordinary merchants could own one."),
    ("When a compiler encounters a function it has already analysed, it does not begin the "
     "analysis again. It looks up the earlier result, checks that the assumptions still hold, "
     "and reuses the conclusion. This works because most functions are small, their inputs are "
     "described by types that rarely change, and the cost of rechecking the assumptions is much "
     "lower than the cost of redoing the work."),
]


def load(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    return tok, model


def blocks_of(model):
    for attr in ("transformer", "gpt_neox", "model"):
        base = getattr(model, attr, None)
        if base is None:
            continue
        h = getattr(base, "h", None) or getattr(base, "layers", None)
        if h is not None:
            return list(h)
    raise RuntimeError("cannot locate transformer blocks")


def attn_of(block):
    for attr in ("attn", "attention", "self_attn"):
        m = getattr(block, attr, None)
        if m is not None:
            return m
    raise RuntimeError("cannot locate attention module in block")


def loss_and_logits(model, input_ids, groups: list[tuple[int, int]] | None):
    """groups = [(start, end_exclusive), ...]；组内第 2 层起复用组首层的 attention 输出。"""
    handles = []
    cache: dict[int, torch.Tensor] = {}
    if groups:
        blocks = blocks_of(model)
        for start, end in groups:
            for idx in range(start, end):
                attn = attn_of(blocks[idx])

                def make_hook(i=idx, s=start):
                    def hook(_mod, _inp, out):
                        if i == s:
                            cache[i] = out[0] if isinstance(out, tuple) else out
                            return out
                        reuse = cache.get(s)
                        if reuse is None:
                            return out
                        if isinstance(out, tuple):
                            return (reuse,) + tuple(out[1:])
                        return reuse
                    return hook

                handles.append(attn.register_forward_hook(make_hook()))
    try:
        out = model(input_ids=input_ids, labels=input_ids)
        return float(out.loss), out.logits
    finally:
        for h in handles:
            h.remove()


def run(model_name: str, group: int) -> dict:
    tok, model = load(model_name)
    blocks = blocks_of(model)
    n_layers = len(blocks)
    groups = [(s, min(s + group, n_layers)) for s in range(0, n_layers, group)]
    results = []
    for text in TEXTS:
        ids = tok(text, return_tensors="pt").input_ids
        base_loss, base_logits = loss_and_logits(model, ids, None)
        reuse_loss, reuse_logits = loss_and_logits(model, ids, groups)
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(reuse_logits, -1),
            torch.softmax(base_logits, -1),
            reduction="batchmean",
        )
        results.append({
            "tokens": int(ids.shape[1]),
            "loss_base": base_loss,
            "loss_reuse": reuse_loss,
            "delta_loss": reuse_loss - base_loss,
            "ppl_base": float(torch.exp(torch.tensor(base_loss))),
            "ppl_reuse": float(torch.exp(torch.tensor(reuse_loss))),
            "kl_base_to_reuse": float(kl),
        })
    return {
        "model": model_name,
        "n_layers": n_layers,
        "group_size": group,
        "n_groups": len(groups),
        "groups": groups,
        "texts": results,
        "mean_delta_loss": sum(r["delta_loss"] for r in results) / len(results),
    }


def selftest() -> None:
    tok, model = load("sshleifer/tiny-gpt2")
    blocks = blocks_of(model)
    assert len(blocks) >= 2, len(blocks)
    ids = tok("hello world this is a test", return_tensors="pt").input_ids
    l0, logits0 = loss_and_logits(model, ids, None)
    # 正向控制（机制层）：把所有组内后续层的 attention 输出替换成组首层，
    # 输出必须确实改变。断言方向而不只是断言不同会失败——tiny-gpt2 只有 2 层、
    # hidden=2，替换后损失的变化在浮点噪声量级，不能作为「变差」的证据。
    groups = [(0, len(blocks))]
    l1, logits1 = loss_and_logits(model, ids, groups)
    assert not torch.equal(logits0, logits1), "hook did not modify the forward pass"
    # 负向控制：组长为 1（没有任何层被复用）-> 输出必须逐位复现 baseline
    groups1 = [(i, i + 1) for i in range(len(blocks))]
    l2, logits2 = loss_and_logits(model, ids, groups1)
    assert torch.equal(logits0, logits2), "group=1 must be a no-op"
    assert abs(l2 - l0) < 1e-9, (l0, l2)
    print(f"  ok    baseline {l0:.4f}")
    print(f"  ok    reuse hook changes logits (Δloss {l1 - l0:+.2e})")
    print(f"  ok    group=1 is bitwise no-op (loss {l2:.6f})")
    print("SELFTEST PASSED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--json")
    ap.add_argument("--all-single", action="store_true",
                    help="逐层对测量：只把第 i 层的 attention 输出替换成第 i-1 层的")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.all_single:
        tok, model = load(a.model)
        blocks = blocks_of(model)
        n = len(blocks)
        per_layer = []
        for text_i, text in enumerate(TEXTS, 1):
            ids = tok(text, return_tensors="pt").input_ids
            base_loss, _ = loss_and_logits(model, ids, None)
            for i in range(1, n):
                l, _ = loss_and_logits(model, ids, [(i - 1, i + 1)])
                per_layer.append({"text": text_i, "reused_layer": i,
                                  "source_layer": i - 1,
                                  "loss_base": base_loss, "loss_reuse": l,
                                  "delta_loss": l - base_loss})
        summary = {}
        for i in range(1, n):
            ds = [r["delta_loss"] for r in per_layer if r["reused_layer"] == i]
            summary[i] = sum(ds) / len(ds)
        out = {"model": a.model, "n_layers": n, "mode": "single-layer",
               "per_layer_mean_delta": summary, "rows": per_layer}
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1))
        for i in range(1, n):
            print(f"reuse layer {i:2d} <- {i-1:2d}: mean Δloss {summary[i]:+.4f}")
        return

    res = run(a.model, a.group)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    for i, t in enumerate(res["texts"], 1):
        print(f"text {i}: tokens={t['tokens']} loss {t['loss_base']:.4f} -> {t['loss_reuse']:.4f} "
              f"(Δ {t['delta_loss']:+.4f}, ppl {t['ppl_base']:.2f} -> {t['ppl_reuse']:.2f})")
    print(f"group={res['group_size']} of {res['n_layers']} layers, {res['n_groups']} groups, "
          f"mean Δloss {res['mean_delta_loss']:+.4f}")


if __name__ == "__main__":
    main()
