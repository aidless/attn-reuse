# 四规模复用代价趋势：0.5B → 14B（2026-11-21，Goal Round18）

**性质：robustness / scale certificate，不进正文、不挂 audit pin。** 原因见 §4 的协议差异。

## 1. 动机：论文 Limitations 明写的空白

`attn-reuse/paper/main.tex:762-766`：

> Three model sizes, three data regimes. … Whether the leader anomaly persists
> at sizes where the first layers differ structurally is untested.

论文正文只到 7B（`main.tex:186-194`：Qwen2.5-0.5B 24 层 / 1.5B 28 层 / 7B 28 层）。
14B（48 层）是本机（6GB RTX 3060）与原 AMD 实例都跑不了的规模，只有这台 48GB
实例能做。这是唯一不是「又一个小模型旁证」的增量。

## 2. 协议

同一份 20-batch / seq 512 / k=256 / bfloat16 / wikitext-2-raw-v1 test 网格，
每一规模独立跑 full、control、own、reuse{g2,g3,g6}。

- 0.5B / 1.5B / 7B：`real/data/main_20b_{05,15,7}.json`（论文主表数据源，`device_map="cuda"`）
- 14B：`real/data/scale_14b_base_20b.json`（`device_map="auto"` + `max_memory={0:"14GiB"}` CPU offload，
  48 层里 38 层在 GPU、10 层在 CPU；权重 `models/Qwen2.5-14B-base`，8 分片）
- 复算脚本：`real/scale_trend.py`（`--selftest` 三条 prove-red 全过）

14B 的 control 与 full **逐位相等**（|d| < 1e-9），与其余三规模同判据。

## 3. 结果（k=256，delta 相对 full）

| model | layers | g2 | g3 | g6 | g6/g2 | control |
|---|---|---|---|---|---|---|
| 0.5B | 24 | +0.02381 | +0.01105 | +0.07317 | 3.1× | OK |
| 1.5B | 28 | +0.00779 | +0.01293 | +0.09110 | 11.7× | OK |
| 7B | 28 | +0.00327 | +0.00953 | +0.03222 | 9.9× | OK |
| 14B | 48 | +0.00228 | +0.00544 | +0.01735 | 7.6× | OK |

（数据源：`real/data/scale_trend.json`）

三条判据同时成立：

1. **相邻组（g2）代价随参数量单调下降**：0.02381 → 0.00779 → 0.00327 → 0.00228。
2. **组大小放大在所有规模都存在**：g6 > g2 > 0，比值 3.1 / 11.7 / 9.9 / 7.6。
3. **14B 保留符号**：三个 group 的 delta 全为正；g6 代价从 7B 到 14B 缩小 1.86×。

## 4. 这**不**证明什么（协议差异）

14B 用 CPU offload（`device_map="auto"`），与论文三规模的 `device_map="cuda"`
不同。虽然：

- control 判据同样逐位通过（offload 不改变 own(k=seq)==full 的恒等性）；
- 同一脚本、同一语料、同一 dtype、同一 batch 数；

但 offload 会让部分层的计算图跨越 host/device 边界，**理论上可能改变 eager
attention 的数值路径**。因此这组数字只作「趋势方向」的旁证，不写进正文、不进
audit pin、不改变任何已发表数字。若要把 14B 写成正文数字，需要一台能全 GPU
装下 28G 权重的机器重跑。

另外 14B 的 NLL 绝对值（full 1.8524）显著低于 7B（2.2704），是模型更强的正常
表现，不能横比。

## 5. 与既有 leader anomaly 的关系

本报告只测**网格层面的组代价**，没测 leader anomaly 的 per-pair 结构
（source≤1 vs source≥2 的代价比）。后者的 14B 版本由 `pairs14b.jsonl` 单独跑
（window 6、k=256、20 batches），结果见后续报告。

## 6. 复现命令

```bash
# 14B 网格（AMD 48GB 实例，需 CPU offload）
HF_HUB_OFFLINE=1 /opt/venv/bin/python reuse_llama_offload.py \
  --model /root/models/Qwen2.5-14B-base --corpus /root/wt2_test.txt \
  --seq 512 --n-batches 20 --ks 256 --groups 2,3,6 --dtype bfloat16 --offload \
  --out /root/attn_newstack/results_14b_base_20b.json
# 四规模趋势复算（无需 GPU）
python3 real/scale_trend.py --selftest
python3 real/scale_trend.py --out real/data/scale_trend.json
```
