# Tested on a model that was *trained* to be sparse, not a dense model with a mask

报告 16/18/19/20/21 全部建立在 dense 权重上做 post-hoc 复用。用户对那条线的判断是
「模型太老」「没研究价值」。本报告是对该判断的技术回应：把同一个问题搬到**真正训练过
稀疏注意力的当代 checkpoint** 上，问 leader anomaly 是否仍然存在。

- 模型：`fxmeng/DSV2-Lite-DSA-Sparse-1250`（HF，2026-10-08 更新，`model_type=dots_dsa`，
  base = `deepseek-ai/DeepSeek-V2-Lite`，带 `index_topk=2048`、`attention_mode=sparse`、
  每层独立 learned DSA indexer；27 层 / hidden 2048 / 16 heads / 64 routed + 2 shared experts）。
  它是一个训练过的稀疏注意力模型，不是加载后再加 mask 的 dense 模型。
- 语料 WikiText-2 test（`/root/wt2_test.txt`，1,289,979 chars），seq 4096，3 个 batch，
  bf16，eager attention，`torch.use_deterministic_algorithms(True, warn_only=True)`
  + `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- 复用方式与论文一致：把 source 层自己选出的 2048 个位置**强行**装进 target 层，target
  仍用自己的 q/k/v，只换集合。141 对（source 0–25，distance 1–6）。

## 1. 先修仪器：这条路径原本不可复现

第一次跑出 `ratio 0.54×`，但同一输入的两次前向 NLL 差 **1.418e-3**，与要测的 delta
（1e-3~1e-4）同量级 ⇒ 那份结论是噪声，已作废。归因链：

| 检查 | 结果 | 含义 |
|---|---|---|
| 同进程 4 次同输入 | range **1.418e-3** | 漂移真实存在（`real/data/dsa_drift.json`）|
| 两次 capture 的索引 | 逐层不同（单 chunk 20.9 万 / 26.2 万条目差异） | 索引选择也不稳 |
| `attention_mode="dense"`（indexer 不进路径） | range 仍 **1.240e-3** | 漂移不来自选索引 |
| 非 MoE 的 Qwen2.5-7B 同机同 seq | range **0.000e+00** | 不是硬件/驱动的通病 |
| 加 `use_deterministic_algorithms` | range **0.000e+00** | **修好了** |

修好后补两项仪器自证（`dsa_order.py`，同一进程）：

- `base_pre` 3 次 / capture 一次 / `base_post` 3 次 / 自借 2 次 / `base_after` 2 次，
  **10 次全是 `1.8414348363876343`，所有 delta `+0.000e+00`** ⇒ capture 与自借都不移动基线。
- 两次 capture 的**逐行集合比较**：110,592 个 query-row 中 110,587 行完全相同
  （99.995%），余下 5 行是等分 tie 的排序差异，集合本身一致（`real/data/dsa_sets_clean.json`）
  ⇒ 索引选择在修好后是确定的。

**已知残余限制**：跨进程的 MoE 分组 GEMM 仍可能因显存压力选不同 kernel
（`dsa_controls.json` 的 CONTROL A 在一次独立进程里读出自借 `+6.243e-4`，与同进程
`+0.000e+00` 冲突）。所以本轮所有 141 对**在同一进程内跑完**，跨进程数值不混用。

## 2. 结果：leader anomaly 在训练过稀疏的模型上**不存在**

（source: `real/data/dsa_multi.json`，sha256 见文件；分析 `real/data/dsa_analysis.json`）

| 组 | n | 平均 vs_own |
|---|---|---|
| lead（source≤1） | 12 | **+0.000482** |
| rest（source≥2） | 129 | **+0.000543** |
| 比值 | | **0.89×** |

置换检验（10,000 次重标号，seed 20261119，**双尾**）**p = 0.6132**（`real/data/dsa_analysis.json`）；差值
lead−rest = **−6.09e-5**，bootstrap 95% CI **[−4.68e-4, +3.98e-4] 跨零**（同上 `dsa_analysis.json`），
38.2% 的重采样为正。边界从 ≤0 挪到 ≤3：1.04× / 0.89× / 0.81× / 0.91×（`real/data/dsa_analysis.json` 的 `boundary_sensitivity`），
**没有一个显著偏离 1**。

对照五个 dense 模型（同一 distance≤6 截断，各自仪器）：

| 模型 | 层 | 对 | lead | rest | 比值 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 123 | 0.005986 | 0.000794 | **7.54×** |
| Qwen2.5-1.5B | 28 | 147 | 0.006141 | 0.000176 | **34.99×** |
| Qwen2.5-7B | 28 | 147 | 0.006772 | 0.000573 | **11.82×** |
| TinyLlama-1.1B | 22 | 111 | 0.051198 | 0.001784 | **28.71×** |
| Qwen2.5-14B | 48 | 72 | 0.003865 | 0.000456 | **8.48×** |
| **DSV2-Lite-DSA（训练过稀疏）** | 27 | 141 | 0.000482 | 0.000543 | **0.89×** |

（source: `real/data/real_vs_dense.json`）

**这是本轮唯一的硬结论**：在 dense 权重上「前两层当 leader 特别贵」是 7.5–35× 的稳定
现象，在一个被训练成每层自己选索引的模型上**消失**（比值落在 1 附近）。论文 §leader
那句「the asymmetry does not depend on the architecture」的适用范围因此必须限定为
**未训练共享/复用的 dense 权重**，不能外推到训练过稀疏选择的模型。

## 3. 但复用代价本身是**正的、且近似均匀的**

容易被 0.89× 遮住的一点：141 对里 **127 对为正**，均值 **+5.376e-4**，
bootstrap 95% CI **[+4.72e-4, +6.07e-4]** 不含零，中位数 +5.33e-4（`real/data/dsa_multi.json` 的 141 对）。

所以训练过稀疏的模型不是「借用免费」，而是「借用一个并非自己的索引集合要付一份小而
均匀的代价，而且这份代价不因源层是不是前两层而变」。这与 dense 模型的区别正好在
**分布形状**：dense 上是厚尾（少数前两层对特别贵），这里是近似平坦。

按相对量看更清楚：lead delta / base NLL 在 dense 五个模型是 2.1e-3 – 2.1e-2，
在训练过稀疏的模型上是 **2.42e-4**，低约一个数量级
（source: `real/data/real_vs_dense.json` 的 `lead_rel` 字段）。

## 4. 索引重叠依然零预测力——这是跨两条线的同一结论

对 141 对逐 query 算真实 Jaccard（不是抽样的近似）：

- 全体 **0.8631**（min 0.8015 / max 0.8983）；lead 0.8251，rest 0.8666
- 按 distance：1→0.8725，2→0.8672，3→0.8645，4→0.8607，5→0.8564，6→0.8547（几乎不动）
- **corr(overlap, cost) = 0.0008**；控制 source 层后偏相关 **0.0129**
  （corr(overlap, source)=0.3655，corr(cost, source)=−0.0305）

两个独立的 2048-of-4096 随机集合的期望 Jaccard 是 **0.3333**，实测 0.86 说明这些层的
选择确实高度相似——**但相似程度对代价没有任何解释力**。这与 dense 权重上
（索引重叠 r=−0.055、CKA −0.341、贡献量 −0.071）是同一个结论，现在在一个
「选择由训练得来」的模型上再次成立，不再能归因于「dense 模型的 top-k 是伪选择」。

（source: `real/data/dsa_overlap.json`）

## 5. 这改写了什么、没改写什么

改写（**已并入正文，commit `614bf9c8`**）：

- 论文 §\label{sec:leader} 的「跨架构不变」已改为「across the five dense checkpoints」，
  并点名 Qwen2.5-14B（48 层，ratio 8.48）。原句「the asymmetry does not depend on the
  architecture」已从稿件删除（grep 计数 0）。
- 新增一整段 trained-sparse（同一 §leader 末尾），末尾结论句「the asymmetry is a
  property of dense weights measured post hoc, not of attention as such」。
- 「What this paper is not」与 Limitations 的 Post hoc sharing 两句同步改写，否则新段落
  会让它们变成假话。**但「Every number here borrows routing from a model trained
  without sharing」这句逐字保留**——trained-sparse 学的是每层自选、不是学分享，原文仍成立；
  删掉它会让 figure_trace 的 LIMITATION-DROPPED 在 tab:grid/tab:leader/tab:long 三处报红。
- 「重叠高 ⇒ 冗余大 ⇒ 可复用」这条动机链在两个完全不同的模型上都断了，其中一个的
  索引是学出来的。这是对 MLI/CLSA/IndexCache 一类方法更强的反驳：不是 dense 模型的
  伪选择问题。
- 复用代价在训练过稀疏的模型上没有消失（+5.4e-4, CI 不含零），只是变小和变平。

没改写：

- 只有一个 checkpoint（HF 140 下载、0 likes），一个模型族，27 层，seq 4096，3 batch。
- 没有对话/生成任务，没有下游 benchmark，只有 NLL。
- 没有对比「训练时就知道要共享」的模型（本模型每层独立，没有跨层共享的训练信号）。

## 5b. 数字门的自证（`real/dsa_report_check.py`）

`--selftest` 四门，其中两门是 prove-red：

1. 五个 dense 模型比值在**同一 distance≤6 截断**下重算必须复现已发表值（7.54 / 34.99 /
   11.82 / 28.71 / 8.48）——这条把「拿已发表数和自测数比」换成「两边都现算」。
2. **prove-red**：把同一文件截到 distance≤12 时 0.5B 必须**不再**复现（7.54 → 11.93，
   偏离 4.39）——证明这道闸会红，不是恒绿。
3. 训练过稀疏的模型必须「不可拒绝」（双尾 p > 0.05 且差值 CI 跨零）。
4. **prove-red**：合成的强 lead 效应必须被判显著。

（source: `real/dsa_report_check.py`）

## 6. 复现命令

```
# 确定性模式（原文件已在模型目录内注入 CAPTURE/SUBSET 钩子）
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1
/opt/venv/bin/python /root/dsa_multi.py    # → dsa_multi.json（141 对，3 batch）
/opt/venv/bin/python /root/dsa_overlap.py  # → dsa_overlap.json（逐 query Jaccard）
/opt/venv/bin/python /root/dsa_order.py    # → dsa_order.json（仪器自证，10 次同值）
/opt/venv/bin/python /root/dsa_sets_clean.py  # → dsa_sets_clean.json（集合级确定）
python3 attn-reuse/real/dsa_report_check.py --selftest   # 本地门（GPU-free）
```

## 7. 教训（本轮踩到的）

1. **先证仪器再测东西**。第一次的 0.54× 与第二次的 0.89× 都是「测出来的」，但只有后者
   有资格被引用——差别不在数字，在仪器是否先被证明可复现。
2. **`dsa_sets.py` 第一版有个静默零**：`same_set += ... if x.numel() <= 200000 else 0`，
   而每个 chunk 恰好 262144 个元素 ⇒ 该字段结构性地永远为 0，看上去像「集合完全不同」
   这个惊人结论。加一行 `assert sv > 0` 才暴露。
3. **跨进程不等于同进程**。同一进程内 10 次同值（`real/data/dsa_order.json`）；跨进程的 MoE 路径可能因显存压力换
   kernel。要么在单进程内跑完整个网格，要么把跨进程差异当噪声源显式建模。
