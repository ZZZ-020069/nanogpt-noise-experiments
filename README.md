# NanoGPT：LM Head-only 实验

**当前分支：`design/lm-head-only-20261114` · 方案版本：`LH-QP-20261114-draft2`**

本分支发布“**冻结随机初始化 backbone、只训练完整 LM Head**”的实验代码与 H100 运行体系，用于研究能否借助 Hessian 信息构造二次代理并复现优化器排序。仓库内已提供独立的 `G` / `T` / `K` 训练实现、Hessian 子块曲率拟合与留出验收模块、Layer 1 Triton 融合加速以及 H100 启动脚本。

- **固定执行布局**：`MICRO_BATCH=16`，单卡梯度累积 32 次，`G` / `T` / `K` 统一布局。
- **监控与启动入口**：H100 入口（`scripts/run_lm_head_h100.sh`）默认开启 SwanLab（项目 `nanogpt-lm-head`），需在任务环境配置 `SWANLAB_API_KEY`，先通过 `smoke` 并在网页确认指标送达后再启动正式任务。
- **按顺序操作**：准备与 `K` 统计量校准（`prepare-k`）→ 联合短测（`smoke-all`）→ `G→T→K` 自动顺序正式 `clean`（`clean-all`）→ 审阅结果后再逐目标执行 `grid`，详见 [H100 操作手册](LM_HEAD_RUN.md) 与 [操作参考](LM_HEAD_REFERENCE.md)。

当前修复版使用新的 `20261116-r1` 目录，下载后冻结实际提交。`prepare-k` 先检查拟合数据稳定性及块间二阶谱矩变异系数，冻结通过的拟合规模后才打开留出；10 块拟合仍配 5 块留出。已有任务继续原冻结代码，不更新运行中的 checkout。手册排期与[计算清单](docs/experiments/lm-head-only/runtime-estimates.json)统一保留为加速前的工程参照；新 T/K 的 H100 用时待实测。

---

## 文档与入口导航

| 文档 / 路径 | 说明 |
|---|---|
| [H100 操作手册](LM_HEAD_RUN.md) | 输入核对、`smoke`、`clean`、曲率拟合/留出、`grid` 与断点续训命令 |
| [操作参考](LM_HEAD_REFERENCE.md) | SwanLab 指标字段、输出文件结构、扩样规则与耗时更新公式 |
| [实现与验证记录](docs/experiments/lm-head-only/IMPLEMENTATION_VALIDATION.md) | 当前修复的 CPU、恢复与启动流程验收，另列历史验证 |
| [一页审阅摘要](docs/experiments/lm-head-only/20261114-draft2/README.md) | 研究动机、基础对齐与必要改动摘要 |
| [完整实验设计](docs/experiments/lm-head-only/20261116-audit-fixes/EXPERIMENT_DESIGN.md) | 主设计第 14 章最新发布快照，含实现修订记录；旧 draft2 快照保留 |
| [当前实现校验清单](docs/experiments/lm-head-only/implementation-manifest.json) | 当前代码身份、来源及文件 SHA-256；不把旧 v3 标签当作本次发布 |

---

## 研究内容

1. **全量 Head 训练**：从统一随机初始化出发，冻结除 LM Head 外的全部参数，仅训练完整 `50304 × 768` 的 LM Head（`38,633,472` 个参数）。
2. **四优化器对照**：在 `SGD`、`Adam`、`AdamW` 与直接更新 Head 的 `Muon-head` 之间比较收敛与排名。
3. **曲率测量**：在初始化与训练关键节点采样 LM Head 的 Hessian 子块；采样仅用于分析与构造 `K`，不缩小训练参数范围。
4. **二次代理检验**：分别构建初始化处精确二阶泰勒展开 `T` 与统计结构二次代理 `K`，检验其能否保留真实模型 `G` 上的优化器排序。

> Head 与输入 `embedding` 必须以同值副本解除权重共享（untie），确保 `embedding` 始终冻结；`Muon` 仅作用于 2D Head 参数组并保留其余超参数，因此记为 `Muon-head`。

---

## 已合入的 Layer 1 加速与验证状态

为消除 `T` 与 `K` 在前向与反向中构建并保留完整 `[16384, 50304]` logits autograd 计算图带来的显存与访存瓶颈，本分支已合入针对 `K` / `T` 的 **Layer 1 Triton 融合快路径**（自动兼容 PyTorch 分块回退）：

- **实现位置**：`nanogpt_noise/lm_head/kernels.py`（`triton_k_fused`、`triton_t_fused`、`triton_ready`）、`model.py`（`forward_fast_chunk`、`backward_fast_chunk`、`finalize_K_step`）及 `train.py`（`fast_objective` 循环）。
- **加速机制**：在 `world == 1` 且 CUDA `bfloat16` autocast 下，通过单次 online softmax Triton kernel 融合完成 logits 统计、损失归约与 pre-GEMM 梯度 `dz` 生成，并由显式矩阵乘 `dz^T @ h` 完成 FP32 梯度累积；非 BF16、CPU 或无 Triton 环境自动回退至数学等价的 PyTorch 行分块路径（`row_slice=1024`）。
- **协议保持**：不改变 `MICRO_BATCH=16`、单卡累积 32 次、BF16 autocast、FP32 主参数与梯度、`G` 训练逻辑、固定评估、`K` 曲率测量流程，并将 `kernels.py` 纳入 `code_identity()` 代码指纹校验。

### 验证范围

- **当前修复**：CPU 回归及新增扩块门槛检查见[当前验证记录](docs/experiments/lm-head-only/IMPLEMENTATION_VALIDATION.md)；本轮没有启动 GPU 任务。
- **此前加速提交的 RTX 记录**（本轮未重新实测）：CUDA BF16、`micro=16`、32-step smoke，含 eval 与 checkpoint。下表保留原报告，不作为当前 H100 验收凭证。

| 目标 / 优化器 | 加速后 32-step 耗时 | 峰值显存 | 加速前 32-step 参考 |
|---|---:|---:|---:|
| `G`（`adamw` / `muon`） | `41.42s` / `41.68s` | `9.04 GiB` / `8.90 GiB` | `51.41s` / `47.05s` |
| `K`（`adamw` / `muon`） | `32.11s` / `31.64s` | `15.62 GiB` / `15.47 GiB` | `87.19s` / `85.75s`（提速约 **2.7x**） |
| `T`（`adamw` / `muon`） | `39.42s` / `39.79s` | `18.54 GiB` / `18.40 GiB` | `146.13s` / `141.36s`（提速约 **3.6–3.7x**） |

- **此前报告的 CUDA BF16 断点续训检查**：RTX PRO 6000 的 `K` 与 `T`，`4-step`、`pause_after=2`，恢复与连续运行 bit-exact 一致。本轮未复测；当前版本的 H100 smoke 和实际 SwanLab 送达仍待验收。

---

## 与 `last-k` 的基础对齐

| 项目 | 对齐要求 |
|---|---|
| 模型与数据 | 保持 GPT 架构、语料、初始化数值及数据窗口不变 |
| 预算与搜索 | 每候选约 2B tokens / 3815 步；5 个初始 + 3 个追加学习率 |
| Batch 与调度 | 全局 batch 524,288 tokens（`512 × 1024`），沿用原 warmup / cosine |
| Micro 布局 | 固定 `MICRO_BATCH=16`，单卡累积 32 次；`G` / `T` / `K` 统一执行 |
| 噪声与随机性 | 原 35 点网格、噪声注入公式及单 seed |
| 精度与超参数 | BF16 autocast、FP32 主参数与梯度、原优化器超参数、无梯度裁剪 |
| 主要评价指标 | 完整预算终点 validation CE（选最优学习率与优化器排名） |

高精度 Hessian 分析（FP64）仅作为离线诊断，不改变训练精度。

---

## 仓库结构与隔离约束

- `nanogpt_noise/lm_head/`：独立 LM Head 包（模型、Triton 核、训练循环、曲率测量、SwanLab 遥测与运行编排）。
- `scripts/run_lm_head_h100.sh`、`scripts/run_lm_head.sh`：H100 与底层单目标启动入口；旧版 [RUN_TASK.md](RUN_TASK.md) 仅用于已有 `v1.8` 实验。
- **与旧实验严格隔离**：本分支不修改 `main`、原有 `v1.8` 训练代码及 [v1.8 设计](docs/nanogpt_experiment_design.md)；LM Head 运行强制使用独立 `lm_head_only` / `LH-QP` 身份与全新输出根目录，不可接续旧实验 checkpoint。旧版首页保留于[基线提交](https://github.com/ZZZ-020069/nanogpt-noise-experiments/blob/bb3cf72e1113d0e9ceb903515772bfb1d1161df9/README.md)。

> 主设计唯一编辑来源仍是工作区主设计第 14 章；`docs/experiments/lm-head-only/20261114-draft2/` 保存发布快照，后续涉及设计条文变更应先更新主设计再导出新版本。模型实现参考 nanoGPT，保留 [MIT 声明](third_party/nanogpt-LICENSE)。
