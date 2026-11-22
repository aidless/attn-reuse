# 贵的带宽有多深：不是架构无关（2026-11-21，Goal Round19）

**性质：对论文 §leader 最后一句的限定证据。** 它推翻了论文里一句强主张的一半，
所以这轮的重点不是「又测了一个模型」，而是「论文那句话该改」。

## 1. 论文那句主张

`attn-reuse/paper/main.tex:343-345`：

> The leader anomaly survives a change of model family: the asymmetry and the
> source-layer ordering do not depend on the architecture, although the absolute
> magnitudes do.

这句话含两个可分的主张：(a) 不对称性（lead vs rest）跨族保留；(b) **源层次序**跨族不变。
(a) 有四模型支撑。(b) 只有两个数据点可比（0.5B 与 TinyLlama 报过 source 0/1 的均值），
且从未做过「这两层彼此可分吗」的检验。

## 2. 先立一个可比性闸（否则 14B 进不了这张表）

`pairs20w12_*.jsonl` 扫到 distance 12，而 14B 与 TinyLlama 只扫到 6。
直接比会混入协议差。所以所有文件都截到 `distance<=6`，并用**能否复现已发表的
cross-family ratio（7.54 / 34.99 / 11.82 / 28.71，容差 0.02）**作为闸：
（source: `real/band_depth.py` 的 `PUBLISHED` 与 `--selftest`）

| 模型 | distance<=6 重算 | 已发表 | 吻合 |
|---|---|---|---|
| Qwen2.5-0.5B | 7.54 | 7.54 | ✔ |
| Qwen2.5-1.5B | 34.99 | 34.99 | ✔ |
| Qwen2.5-7B | 11.82 | 11.82 | ✔ |
| TinyLlama-1.1B | 28.71 | 28.71 | ✔ |

四个全部吻合到 0.02 内 ⇒ 截断后的表与论文主表同口径，14B 可以进这张表。
`--selftest` 同时证明这个闸能红：把 0.5B 的截断错成 12 时，重算值偏离已发表值 ≥0.02。
（source: `real/data/band_depth.json` 的 `models.*.reproduces_published`）

## 3. 结果一：贵的带宽在模型间差约 45 倍

source 2 的均值 / source≥3 的均值，全部 distance<=6：

| 模型 | src2 | src3+ | ratio | 95% CI（组内 bootstrap, 20000 次重采样，源 `real/band_depth.py`） |
|---|---|---|---|---|
| Qwen2.5-0.5B | +0.003264 | +0.000653 | 5.00 | [0.36, 12.80] |
| Qwen2.5-1.5B | +0.000089 | +0.000180 | **0.49** | [−0.47, 1.76] |
| Qwen2.5-7B | +0.000463 | +0.000578 | **0.80** | [0.08, 1.68] |
| TinyLlama-1.1B | +0.017370 | +0.000778 | **22.33** | [9.14, 41.09] |
| Qwen2.5-14B | +0.002066 | +0.000277 | **7.47** | [2.33, 19.11] |

（source: `real/data/band_depth.json` 的 `models.*.src2_over_tail` 与 `src2_over_tail_ci`）

读法：1.5B 与 7B 上**第 2 层已经和深尾没有区别**（ratio 0.49 / 0.80，CI 含 1）；
0.5B / TinyLlama / 14B 上第 2 层仍明显更贵（5.00 / 22.33 / 7.47）。
即「贵的带宽」在有的架构上止于第 2 层，在有的到第 3 层才止，
跨度 0.49 → 22.33 约 **45 倍**。

## 4. 结果二：14B 上 source 0 与 source 1 谁更贵，测不出来

我上一版报告 21 写过「14B 上 source 1 比 source 0 更贵，与 0.5B 相反」——
那只是读了两个均值（+0.004706 vs +0.003024），没有做检验。补上检验后：

- 六个一对的置换：**p = 0.2267**（单尾，src1 > src0）
- 差值 bootstrap CI：**[−0.002874, +0.005977]**，跨零（源 `real/data/scale_leader_14b.json` 的 `top_two_order.boot_ci`）
- `distinguishable = False`

（source: `real/data/scale_leader_14b.json` 的 `top_two_order` 字段）

四个已发表模型上同一检验的方向都是 src0 > src1（0.5B −0.005136、1.5B −0.003751、
7B −0.004413、TinyLlama −0.025749），但那些也是**点估计方向，没有配对检验**。
所以正确的说法是：第一层组内部（source 0 vs 1）的序关系在本数据上不可判，
不能拿它当「跨架构不变」或「随规模换位」的证据。

## 5. 由此对论文的两条修改建议

1. 把那句拆开：不对称性跨族保留（四模型，已发表、已验）；**源层次序与贵带宽深度
   不跨族不变**（本条给出 0.49–22.33 的带宽跨 45 倍、以及 top-2 序不可判）。
2. 不要引入新数字到正文（guard：`audit_numbers.py` 要求每个印刷数字有 artifact pin）。
   最小改法是把「ordering / depth」这类词降级为「asymmetry survives; finer structure varies」。

## 6. 复现

```bash
python3 attn-reuse/real/band_depth.py --selftest      # 闸能红
python3 attn-reuse/real/band_depth.py --out attn-reuse/real/data/band_depth.json
python3 attn-reuse/real/scale_leader_14b.py --selftest
python3 attn-reuse/real/scale_leader_14b.py --pairs attn-reuse/real/data/pairs14b.jsonl \
    --out attn-reuse/real/data/scale_leader_14b.json
```
（source: `real/band_depth.py` + `real/scale_leader_14b.py`）

## 7. 教训

**读两个均值之间的序关系，是最容易被自己骗的一步**：0.003024 与 0.004706 看起来
是「反过来了」，但 n=6 时这个差别完全在噪声里。上一个版本我把它写成了发现。
判据应该是「我先验地要求可分辨性检验」，而不是「差值看起来不小」。
