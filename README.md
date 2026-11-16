# NanoGPT：LM Head-only 实验

**当前分支：`design/lm-head-only-20261114` · 方案版本：`LH-QP-20261114-draft2`**

本分支发布“**冻结随机初始化 backbone、只训练完整 LM Head**”的实验代码与 H100 运行体系，用于研究能否借助 Hessian 信息构造二次代理并复现优化器排序。仓库内已提供独立的 `G` / `T` / `K` 训练实现、Hessian 子块曲率拟合与留出验收模块、Layer 1 Triton 融合加速以及 H100 启动脚本。

- **固定执行布局**：`MICRO_BATCH=16`，单卡梯度累积 32 次，`G` / `T` / `K` 统一布局。
- **监控与启动入口**：H100 入口（`scripts/run_lm_head_h100.sh`）默认开启 SwanLab（项目 `nanogpt-lm-head`），需在任务环境配置 `SWANLAB_API_KEY`，先通过 `smoke` 并在网页确认指标送达后再启动正式任务。
- **按顺序操作**：准备 → `G` → `T` → `K` → 汇总对比，详见 [H100 操作手册](LM_HEAD_RUN.md) 与 [操作参考](LM_HEAD_REFERENCE.md)。

> **提示**：当前已将 Layer 1 `K`/`T` Triton 融合加速合入 `design/lm-head-only-20261114`；若需使用加速版进行 H100 实验，请直接检出本分支最新提交（`LM_HEAD_RUN.md` 中的 `v3` 标签克隆示例与 `runtime-estimates.json` 仍对应加速前基线，待 H100 首轮 `smoke` 后再替换用时估算）。

---

## 文档与入口导航

| 文档 / 路径 | 说明 |
|---|---|
| [H100 操作手册](LM_HEAD_RUN.md) | 输入核对、`smoke`、`clean`、曲率拟合/留出、`grid` 与断点续训命令 |
| [操作参考](LM_HEAD_REFERENCE.md) | SwanLab 指标字段、输出文件结构、扩样规则与耗时更新公式 |
| [实现与验证记录](docs/experiments/lm-head-only/IMPLEMENTATION_VALIDATION.md) | 早期实现验证基线记录 |
| [一页审阅摘要](docs/experiments/lm-head-only/20261114-draft2/README.md) | 研究动机、基础对齐与必要改动摘要 |
| [完整实验设计](docs/experiments/lm-head-only/20261114-draft2/EXPERIMENT_DESIGN.md) | 训练、Hessian 采样、二次代理及验收判据（第 14 章发布快照） |
| [来源与校验清单](docs/experiments/lm-head-only/20261114-draft2/manifest.json) | 主设计来源、基线提交与文件 SHA-256 |

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

### 已完成验证（本地 CPU + 远端 RTX PRO 6000）

- **本地单元测试**：覆盖 `telemetry`、`lm_head_telemetry`、`lm_head`、`lm_head_micro`、`lm_head_shell`、`lm_head_h100` 及 `lm_head_speedup` 共 **49 项测试全部通过**（CPU 合成数据与模拟上传）。
- **远端 RTX PRO 6000（CUDA BF16，`micro=16`，32-step 端到端 smoke，含 eval 与 checkpoint）**：

| 目标 / 优化器 | 加速后 32-step 耗时 | 峰值显存 | 加速前 32-step 参考 |
|---|---:|---:|---:|
| `G`（`adamw` / `muon`） | `41.42s` / `41.68s` | `9.04 GiB` / `8.90 GiB` | `51.41s` / `47.05s` |
| `K`（`adamw` / `muon`） | `32.11s` / `31.64s` | `15.62 GiB` / `15.47 GiB` | `87.19s` / `85.75s`（提速约 **2.7x**） |
| `T`（`adamw` / `muon`） | `39.42s` / `39.79s` | `18.54 GiB` / `18.40 GiB` | `146.13s` / `141.36s`（提速约 **3.6–3.7x**） |

- **CUDA BF16 断点续训一致**：在 RTX PRO 6000 上对 `K` 与 `T` 进行 `4-step`、`pause_after=2` 的真实暂停与恢复测试，与连续运行结果达到 **bit-exact 一致**；目前尚未在 H100 上启动正式实验。

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
