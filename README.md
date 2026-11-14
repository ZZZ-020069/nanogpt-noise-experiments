# NanoGPT：LM Head-only 实验设计

**当前分支：design/lm-head-only-20261114 · 方案版本：LH-QP-20261114-draft2**

本分支发布“冻结随机初始化的 backbone，只训练完整 LM Head”的实验方案，并研究能否用 Hessian 信息构造二次函数，复现优化器排序。现已提供独立的 G/T/K 训练代码、Hessian 子块测量和启动脚本。实现检查与目标 GPU 验收状态见 [验证记录](docs/experiments/lm-head-only/IMPLEMENTATION_VALIDATION.md)。

LM Head 的训练和离线测量现在默认 **MICRO_BATCH=16**，支持用 16/32 在独立临时目录上卡测速。全局 batch 和训练预算保持不变；同一实验根目录锁定一个 micro，G/T/K 及跨目标报告均校验一致性。H100 最终档位仍待三个目标共同验收，步骤见[启动说明](LM_HEAD_RUN.md#h100-选档分别测-16-和-32)。

## 从这里开始

| 文档 | 内容 |
|---|---|
| [启动与恢复说明](LM_HEAD_RUN.md) | 输入核对、smoke、clean、曲率测量和后续阶段命令 |
| [一页审阅摘要](docs/experiments/lm-head-only/20261114-draft2/README.md) | 研究路线、基础对齐与必要改动 |
| [完整实验设计](docs/experiments/lm-head-only/20261114-draft2/EXPERIMENT_DESIGN.md) | 训练、Hessian 采样、二次代理及验证方案 |
| [来源与校验清单](docs/experiments/lm-head-only/20261114-draft2/manifest.json) | 主设计来源、基线提交与快照 SHA-256 |

## 这次研究什么

1. 从统一未训练初始化开始，冻结其余参数，训练整个 50304×768 的 LM Head。
2. 比较 SGD、Adam、AdamW 和直接更新 Head 的 Muon-head。
3. 在初始化与训练过程中采样 LM Head 的 Hessian；采样只用于测量，不缩小训练范围。
4. 构造二次参照和统计结构代理，检验优化器排序能否保留。

Head 与输入 embedding 必须同值解除权重共享，保证 embedding 始终冻结。Muon 的参数分组需要单独调整，因此记录为 Muon-head。

## 与 last-k 保持基础对齐

| 项目 | 对齐要求 |
|---|---|
| 模型与数据 | 保持架构、语料、初始化数值和数据窗口 |
| 预算与搜索 | 每候选约 2B tokens / 3815 步，五初始加三追加学习率 |
| Batch 与调度 | 全局 524288 tokens，原 warmup/cosine |
| micro 执行布局 | 默认 16，单卡累积 32 次；micro=32 时累积 16 次；G/T/K 锁定同一档 |
| 噪声与随机性 | 原 35 点网格、噪声公式、单 seed |
| 精度与配方 | BF16 autocast、FP32 主参数和梯度、原超参数、无裁剪 |
| 主要评价 | 完整预算终点 validation loss 选学习率和排名 |

只对训练范围、权重共享、Muon 适用参数组和曲率观测作必要修改。高精度 Hessian 分析是离线诊断，不改变训练精度。采样规模与代理验收阈值仍以完整方案中的草案标识为准。

## 与旧实验隔离

- 新方案位于独立分支和独立版本目录，不覆盖旧的 [v1.8 设计](docs/nanogpt_experiment_design.md)。
- main、原有 v1.8 训练代码、配置与 [RUN_TASK.md](RUN_TASK.md) 保持原样；RUN_TASK.md 仍只用于已有 v1.8 实验，不是 LM Head-only 的启动说明。
- 本分支首页更新为新方案导航；[旧版首页](https://github.com/ZZZ-020069/nanogpt-noise-experiments/blob/bb3cf72e1113d0e9ceb903515772bfb1d1161df9/README.md)保留在基线提交中。
- 新实现和运行使用独立的 lm_head_only / LH-QP 实验身份及输出目录，不能接续 last1 的 checkpoint 或混用结果。
- 实现检查采用小规模合成 CPU 数据；本次没有部署或启动正式实验。

唯一编辑来源是工作区主设计的第 14 章。本仓库版本目录保存发布快照，后续先修改主设计，再生成新版本；首页仅提供导航与摘要。

模型实现参考 nanoGPT，保留 [MIT 声明](third_party/nanogpt-LICENSE)。
