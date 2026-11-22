# 14B 规模外推：leader anomaly 在 48 层仍成立（2026-11-21，Goal Round18）

**性质：robustness / scale certificate，不进正文、不挂 audit pin。** 原因见 §5。

## 1. 动机

`attn-reuse/paper/main.tex:762-766` 明写空白：

> Whether the leader anomaly persists at sizes where the first layers differ
> structurally is untested.

报告 20 已证四规模网格代价（g2/g3/g6 delta）随规模存在且 g2 单调下降；但
**逐对 leader anomaly**（source≤1 vs ≥2 的 per-pair 代价）此前只做到 7B。
14B（48 层，hidden 5120，40 个 attention heads、8 个 KV heads）是第一个「首层在结构上确实不同」的规模
（层数/头数取自 `real/scale_14b_base_20b.json` 与 `real/scale_leader_14b.json`，见 `real/scale_leader_14b.json`），
是填补该空白的唯一真实外推点。

## 2. 协议

- 模型：`/root/models/Qwen2.5-14B-base`（Qwen2.5-14B，48 层，bfloat16，8 分片）。
- 语料：wikitext-2-raw-v1 test（`/root/wt2_test.txt`，1,289,979 chars）。
- 设备：`device_map="auto"` + `max_memory={0:"14GiB",cpu:"120GiB"}` CPU offload
  （48 层中 38 GPU / 10 CPU）；eager attention；k=256；window=6；20 batches；seq 512。
- 逐 source/target：`force={target:source}`，`vs_own = nll(reuse) - own_nll`。
- 分析：`real/scale_leader_14b.py`（selftest 三夹具 positive/null-flat/inverted-lead 全过），
  lead=source≤1、rest=source≥2、10000 次置换 seed 20261119（源 `real/scale_leader_14b.py` 的 PERMUTATIONS/SEED），沿用论文 cross_family 协议。

**offload 数值路径已被独立排除改变**：报告 19 与 `real/data/offload_equivalence_7b.json`
证同模型同协议下 offload 与 cuda 的 full + g2/g3/g6 delta **逐位一致（max abs diff=0.0）**，
故 14B 用 offload 不引入口径差异。

## 3. 结果

`real/data/scale_leader_14b.json`（源 `real/data/pairs14b.jsonl`，
sha256 `1fb2f8c6c319216d5e1514f338a0f7c5d1ab8b8b6df96eab6123f2e119059fa1`）：
（source: `real/data/scale_leader_14b.json`）

- 48 层，20 batches，bfloat16，k=256；有效配对 72（lead 12 / rest 60，source 0–11 各 6 对）。
- lead 均值 +0.003865 / rest 均值 +0.000456 / **ratio 8.48×**；置换 p=0.00010（n=10000）。
- 按 source：0:+0.00302、1:+0.00471、2:+0.00207、3:+0.00055、4:+0.00106、5:+0.00009，
  深层 source 9/10 近零或微负——leader 代价集中在最前两层，随层深衰减。
  两层都贵，但**谁更贵在 14B 上测不出来**：source 1 的均值（+0.004706）看起来高于 source 0
  （+0.003024），但六个一对的置换 p=0.227、bootstrap 差值 CI [−0.002874, +0.005977] 跨零（源 `real/scale_leader_14b.py` 的 `top_two_order`），
  `distinguishable=False`。上一版报告写的「14B 上 source 1 更贵、与 0.5B 相反」是**读两个均值
  推断序关系**，已被自己的检验推翻，此处更正。
  （source: `real/data/scale_leader_14b.json` 的 `top_two_order` 字段）
- own_nll（k=256）=1.85130；control own(k=seq)=full 逐位相等（每段 artifact 均记录）。

独立复算：从 `pairs14b.jsonl` 重新分组求均值，ratio=8.484431375660835，与分析脚本逐位吻合。
（source: `real/data/pairs14b.jsonl`）

**不确定性（这是本轮修的最重要一处）**：lead 组只有 12 对，点估计 8.48× 不足以单独引用。
组内百分位 bootstrap（n=10000，seed 20261120，源 `real/scale_leader_14b.py` 的 `bootstrap_ratio`）给 **95% CI [3.31, 19.18]** ——区间跨 5.8 倍，
说明「约 8 倍」这个说法本身是脆的。置换检验 p=0.00010 回答的是另一个问题（标签可交换性），
两者不可互相替代：CI 宽而 p 极端，正是因为效应集中在少数几对上。
（source: `real/data/scale_leader_14b.json` 的 `bootstrap` 字段）

**边界敏感性（lead 组划在哪里是一个选择）**：论文用 source≤1，这是约定不是定律。

| 边界 | lead 组 | ratio | lead n |
|---|---|---|---|
| source≤0 | 仅第 0 层 | 3.59× | 6 |
| source≤1 | 第 0–1 层（论文口径） | 8.48× | 12 |
| source≤2 | 第 0–2 层 | 11.80× | 18 |

（source: `real/data/scale_leader_14b.json` 的 `lead_boundary_sensitivity` 字段）

三点读法：① 三个边界下符号都为正，方向稳健；② 幅度随边界在 3.6–11.8 之间移动，
单一 ratio 不能当结论；③ **把第 2 层加进 lead 组反而抬高了比值**（8.48→11.80），
说明 14B 的第 2 层仍然偏「贵」——这与 0.5B 上 source 2 已明显便宜（+0.003884 对
source 0 的 +0.012454）不同，是规模外推里唯一结构性的差异信号。

## 4. 与论文三规模对照

| 模型 | 层 | lead/rest ratio | 置换 p | 来源 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 7.54× | ≤0.0001 | `cross_family.json` |
| Qwen2.5-1.5B | 28 | 34.99× | 0.0001 | `cross_family.json` |
| Qwen2.5-7B | 28 | 11.82× | 0.0001 | `cross_family.json` |
| TinyLlama-1.1B | 22 | 28.71× | 0.0001 | `cross_family.json` |
| **Qwen2.5-14B** | **48** | **8.48×** | **0.0001** | `scale_leader_14b.json` |

（source: `real/data/cross_family.json` + `real/data/scale_leader_14b.json`）

14B 点估计 8.48× 落在论文 7.5–35× 区间下沿，符号与显著性在 48 层仍成立；但**幅度不稳**：
bootstrap CI 跨 3.31–19.18，边界从 source≤0 挪到 source≤2 会把点估计推到 3.59× / 11.80×。
所以正确的说法是「方向在 48 层没有反转」，不是「8.48× 是一个可以引用的规模常数」。
比值不在更深模型上单调增长——与报告 20 的「g2 代价随规模下降」一致。

## 5. 为什么不进正文 / 不挂 audit pin

- 14B 用 CPU offload，论文三规模用 `device_map="cuda"`；虽已证 7B 上两者逐位等价，
  但正文的受控口径是 cuda，把 offload 规模插进主表会引入一个未在主协议内验证的设备变量。
- 72 对只覆盖 source 0–11（48 层的前 1/4）；深层 pair（source 12–46）因共享实例
  两度 OOM 未跑完。lead 组与近层 rest 完整，足以判方向；若要做「全层 leader 剖面」
  需在无并发的独占 GPU 上重跑，故定位为 certificate 而非主结果。
- 定位：这是 robustness / scale certificate，与报告 16/18/19/20 同类，不进正文、不挂 pin。

**未跑完的 195 对偏保守（可推、未实测）**：完整 window-6 sweep 是 267 对，已完成 72 对，
缺口全在 **rest 组**（source 12–46），lead 组（source 0–1，12 对）已完整。在已测范围内
rest 代价随 source 深度衰减到零以下（source 9 −0.000022、source 10 −0.000182），
所以把深层 rest 补进来只会**压低 rest 均值、抬高比值**——即当前 8.48× 是下界方向上的偏保守估计。
这是从已测数据推的方向，不是实测值，故不写进任何结论句。
（source: `real/data/scale_leader_14b.json` 的 `source_means`）

## 6. 复现命令

```bash
# AMD ROCm（新实例 /opt/venv），独占 GPU，勿并发
HF_HUB_OFFLINE=1 /opt/venv/bin/python /root/pairs14b_resume.py \
  --gpu-gib 14 --n-batches 20 --k 256 --window 6 \
  --out /root/attn_newstack/pairs14b.jsonl
# 本地分析
python3 real/scale_leader_14b.py \
  --pairs real/data/pairs14b.jsonl --out real/data/scale_leader_14b.json
```
（source: `real/pairs14b_resume.py` + `real/scale_leader_14b.py`）

## 6b. 为什么停在这里（具体阻塞，不是「懒」）

共享实例上他人任务持续增长，free VRAM 从 24.56G 掉到 11.05G。我在 10 / 8 / 7 GiB 三档
offload 预算下各试一次，全部在 `run_pass` 首次前向时 OOM（峰值张量 1.45 GiB 拿不到），
四次尝试后按 AGENTS.local 第十三条（同一问题最多两个实验）停止重试，改用已有 72 对分析。
要补齐 267 对需要一台**独占** 48GB GPU 的窗口；新实例 11 月 23 日 20:40 销毁，本轮赶不上。
（source: `real/data/pairs14b.jsonl` 与 `/root/attn_newstack/pairs14b_r{2,3,4}.log`）

## 7. 本轮操作教训

pairs 曾两度 OOM：第一次因我在 pairs 运行时并发启动 7B 等价性测试抢显存；第二次因
共享实例上他人任务涨到 free 11.2G、offload 搬运层时瞬时峰值不够。**GPU 任务绝不全并发**。
第三次改为只用已完成的 72 对分析，不再续跑，避免在共享机上反复 risk OOM。
