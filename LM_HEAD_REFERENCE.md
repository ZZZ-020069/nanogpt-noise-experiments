# LM Head 操作参考

首次运行请按[操作手册](LM_HEAD_RUN.md)逐步执行。本页补充查询信息，所有命令仍需先执行手册中的共同三行。

## SwanLab 看哪些指标

| 字段 | 用途 |
|---|---|
| `config.objective / optimizer / peak_lr / node` | 按 G/T/K、优化器、峰值 LR 和噪声节点过滤实验 |
| `progress/step / progress/tokens` | 实际训练进度；横轴用实际 step |
| `diagnostic/train_loss / diagnostic/lr` | 训练目标 loss、当前 LR |
| `evaluation/train / evaluation/val` | 固定评估池的真实 CE，对应 `evaluation/step` |
| `diagnostic/train_objective / diagnostic/val_objective` | G/T/K 各自目标值，对应 `diagnostic/objective_eval_step` |
| `diagnostic/gradient_norm / diagnostic/update_norm` | 梯度与更新诊断 |
| `status/completed / status/numerical_failure` | 完成、数值失败标志 |
| `diagnostic/infrastructure_failure` | 暂停或基础设施异常 |

第 1 步、每 10 步、评估与终态上传。候选及每次恢复各有独立 attempt，用 `training_run_id` 关联、`resumed_from_step` 定位恢复点；上传序号不等于训练步。强制杀进程可能来不及上传终态。

缓存位于 `$LH_RUN_ROOT/swanlab/`。上传异常不终止训练，本地记录可能比网页完整。显式 `NANOGPT_SWANLAB=0` 可关闭；手册默认保持开启。SDK 安装与 Key 环境变量用法见 [SwanLab 官方说明](https://docs.swanlab.cn/en/guide_cloud/integration/integration-any-library.html)。

## 文件在哪里

以下路径相对 `$LH_RUN_ROOT`：

| 路径 | 内容 |
|---|---|
| `lm_head_run.json` | 代码、输入与 micro 身份 |
| `smoke/G/passed.json` | G 的训练 smoke 凭证；T/K 同理，不证明云端送达 |
| `logs/G-job-0000.log` | 正式任务的 worker 日志；目标和 job 编号按实际替换 |
| `G/summary.json` | G 已完成搜索数与汇总 |
| `G/jobs/job_0000/search.json` | 一个优化器、一个节点的 8 候选与赢家 |
| `G/jobs/job_0000/candidate_00/` | 单候选 metadata、status、progress、evaluations 与诊断 |
| `candidate_*/analysis/`（位于各 job 下） | G clean 的 Head 中间快照 |
| `G/jobs/job_0000/retained.json` | 该搜索保留权重的哈希清单 |
| `comparison.csv / comparison.json` | 已完成搜索的 CE 排名与 G→T、T→K 比较 |

未完成搜索不会进入 comparison；先用 status 或候选的 progress 查看进度。正式搜索会压缩逐步日志并保留赢家恢复包，不能按“每个候选都有 final.pt”判断是否丢数据。完整单目标噪声面仅赢家恢复包约 54 GB；曲率、分析快照、日志及临时写入空间另计。

## 拟合扩样

只在拟合池上决定规模，冻结后才打开留出。主设计第 14.6.4 节规定：数值正常但稳定性未通过时，按 65536、131072 tokens 扩容，保持原采样块；到上限仍不稳定就记录未测稳，暂停 K。

例如从 32768 扩为 65536，每布局仍用 5 块：

```bash
LH_TOKENS=65536 LH_BLOCKS=5 \
LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261115-v3/init-fit-64k \
bash scripts/run_lm_head_h100.sh curvature-fit
```

需要 131072 时，同时替换 token 数和目录名为 `init-fit-128k`。只有数据扩容已稳定、同一布局的全参数二阶谱矩估计在五块间的变异系数仍大于 30%，才将拟合块数改为 `LH_BLOCKS=10`，另用新目录（例如 `init-fit-64k-b10`）。该触发条件由报告人工审查，脚本不会自动扩块。

查看 `report.json` 的 `moments_by_layout`，各布局 `second_moment.std / abs(second_moment.mean)` 即该变异系数；均值为零时不能直接用比值判定。`numeric_passed`、`geometry_passed`、`compatibility.passed` 和 `fixed_gradient_checks_passed` 对应数值、结构与梯度兼容性检查；这些失败不能直接解释为样本不足。

例如最终采用 65536 tokens、5 块的拟合并已通过，留出命令为：

```bash
LH_TOKENS=65536 LH_BLOCKS=5 \
LH_FIT_DIR=/data/260010171/results/lm-head-curvature-m16-20261115-v3/init-fit-64k \
LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261115-v3/init-check-64k \
bash scripts/run_lm_head_h100.sh curvature-check
```

若拟合实际采用 10 块，将 `LH_FIT_DIR` 改为对应拟合目录；留出仍为每布局 5 块。留出通过后，把实际生成的 `K-approved.pt` 路径写入共同 `.env` 的 `K_STATISTICS`。测量设置变化使用新目录，保留旧报告；不能修改 metadata 或通过标志复用旧产物。

## 训练后曲率漂移

这一步使用 G clean 的 CE 赢家，在 step 191、1908 和终点分别测量；沿用初始化确定的 token 数，每布局默认 2 块。它衡量漂移，不生成新的 K 训练产物。

先从对应 `search.json` 找到赢家 candidate，在平台任务环境设置 `LH_CHECKPOINT` 为该赢家的 `analysis/head-000191.pt`（其他时点相应替换，终点也可用赢家 `final.pt`）。在任务中执行：

```bash
: "${LH_CHECKPOINT:?设置为实际 G clean 赢家的 Head 快照绝对路径}"
LH_TOKENS=32768 LH_BLOCKS=2 \
LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261115-v3/drift-G-winner-step191-fit \
bash scripts/run_lm_head_h100.sh curvature-fit
```

这里的 32768 应替换为初始化最终规模；每个优化器、时点和阶段都使用不同目录。拟合完成后，保持同一 `LH_CHECKPOINT`，执行：

```bash
LH_TOKENS=32768 LH_BLOCKS=2 \
LH_FIT_DIR=/data/260010171/results/lm-head-curvature-m16-20261115-v3/drift-G-winner-step191-fit \
LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261115-v3/drift-G-winner-step191-check \
bash scripts/run_lm_head_h100.sh curvature-check
```

后期 `passed` 主要反映数值检查；结构误差保留为漂移，不用于重新校准初始化代理。做完后清除任务环境中的 `LH_CHECKPOINT`，避免后续初始化测量误用训练后的权重。
