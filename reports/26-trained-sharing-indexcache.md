# 在训练过跨层共享的权重上，打破共享要付多少

## 1. 为什么这是论文 Limitations 明写的空白

`attn-reuse/paper/main.tex` 的 `\paragraph{Post hoc sharing}` 写：deployed 系统
的正确对照是 trained-with-sharing 基线，本轮就用它。此前用过的
`DSV2-Lite-DSA-Sparse-1250` 是每层各自选（`layer_assignments: dsa -> [(i,0,1)]`），
不是"训练过共享"。本轮找到 `fxmeng/DSV2-Lite-IndexCache-Sparse-1250`：
`dots_dsa_method=indexcache`、`dsa_indexcache_layer_types="FSS"×9`、
`full_stride=3`，即 27 层里 9 个 full leader 各自为后面两层算路由，README 原话
"Each full layer computes routes reused by its two shared layers"。

## 2. 主结果

| reading | 平均 Δ (nats) | 95% t CI | 3/3 batch 同号 |
|---|---|---|---|
| 随机集（打破共享） | **+0.070199** | [+0.060463, +0.079935] | 是 |
| 邻组集（真实但错组） | +0.005763 | [−0.003176, +0.014703] | 是 |

**随机集是邻组集的 12.18 倍。** 两个 cell 都在同一次 run、同一批 batch 上测，
配对。

## 3. 三个负向/结构性对照

- **只改顺序 = 0**：`torch.roll` 索引列表后 delta 恰为 `+0.000e+00`。这是对的，
  不是 bug —— 索引最终经 `scatter_()` 变成布尔 mask，mask 对顺序不敏感。
  **第一版 sanity 用 roll 当"改了集合"，等于什么都没测**，它报的 0 让我误判
  机制没接上。
- **惰性 chunk 逐位无贡献**：把 SUBSET 只加在能区分的 chunk（`start+1 > topk`，
  16/32 个）与加在所有 chunk 上，三条 batch 的 delta **完全相同**
  （`ic_dilution.json`）。惰性 chunk 贡献恰为 0，这是实测不是推导。
- **行稀释 2.0005×**：4095 个 scoring 行里 2048 行 `q+1 <= topk` 无法区分，
  故可区分行上的均值应为全体均值的 `4095/2047`。实测
  1.9932 / 1.9913 / 2.0136，均值 1.9994，与 2.0005 差 **0.06%**。

## 4. 与 dense 权重的对照，以及为什么不能直接比

dense 0.5B 上任意集 null 是复用代价的 7.95–29.4 倍；这里打破一个**训练过的**
共享要 `+0.070199`，相对该模型自己的 base NLL 1.9964 是 3.5%。
两个模型不是同一模型的两个尺度，**绝对值与倍数都不可跨族比较**——只能各自
说"在自己的 null 下占比多少"。

## 5. 我自己的三个错（全部由荒谬值暴露）

1. **roll 当集合改动**：顺序不影响 mask，delta 恒 0。判据：一次"改动"必须
   改变集合本身，或改用 mask/set 层面的比较。
2. **sanit 探针打在惰性 chunk 上**：chunk (0,128) 的 query 最多看 128 个 key，
   `topk(2048)` 返回全部可见 key，强制集合法理上等于原集合。判据：先算
   `chunk_start + 1 > topk` 再选探针位置。
3. **把"惰性 chunk 无贡献"当成"稀释不成立"**：两者是不同的问题，前者测分子
   后者测分母。改测逐行 loss 后 2.0005 立刻出现。

## 6. 复现

```bash
# AMD <cloud-host>, /opt/venv, transformers 4.57.6
/opt/venv/bin/python /root/ic_null.py        # 主读数
/opt/venv/bin/python /root/ic_dilution.py    # 惰性 chunk 逐位无贡献
/opt/venv/bin/python /root/ic_rows.py        # 行稀释 2.0005x
python3 attn-reuse/real/ic_analysis.py       # 汇总 + selftest
```

跑前必须 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 且 `torch.use_deterministic_algorithms(True)`；
共享实例上必须先等到 free ≥30 GiB（`retry_ic.sh` 的轮询写法）。
