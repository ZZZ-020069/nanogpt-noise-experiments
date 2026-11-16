# LM Head 操作参考

首次运行请按[操作手册](LM_HEAD_RUN.md)逐步执行。本页补充查询信息；每个命令块均已包含目录切换和环境加载，可整块复制。先按相应说明确认采样规模和 checkpoint 路径。

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

## 拟合扩块

数据扩样的 65536 / 131072 tokens 完整命令见[操作手册第 3 步](LM_HEAD_RUN.md#3-生成并校验-k-统计量只用第-1-张卡)。只有拟合数据稳定且 `fit_scale.action=expand_blocks`，才执行本节；数值、结构与梯度相关检查还须正常。只选与刚才拟合相同 token 数的一块，不顺次运行三块。

`moments_by_layout` 中各布局的 `second_moment.std / abs(second_moment.mean)` 是块间变异系数。程序把两个布局的值和 30% 门槛写入 `fit_scale`；零均值、非有限或无法计算的值不会自动通过。5 块超过门槛时停止并建议扩到 10 块；10 块仍超过门槛时记录 `unresolved_at_block_limit` 并停止。脚本不会自行扩大 token 数或块数。

**32768 tokens、每布局 10 个拟合块；用时：暂无可靠小时数，按本机日志校准。拟合通过后另需留出测量时间。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
LH_TOKENS=32768 LH_BLOCKS=10 bash scripts/run_lm_head_h100.sh prepare-k
```

**65536 tokens、每布局 10 个拟合块；用时：暂无可靠小时数，按本机日志校准。拟合通过后另需留出测量时间。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
LH_TOKENS=65536 LH_BLOCKS=10 bash scripts/run_lm_head_h100.sh prepare-k
```

**131072 tokens、每布局 10 个拟合块；用时：暂无可靠小时数，按本机日志校准。拟合通过后另需留出测量时间。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
LH_TOKENS=131072 LH_BLOCKS=10 bash scripts/run_lm_head_h100.sh prepare-k
```

拟合输出为 `$LH_CURVATURE_ROOT/init-fit-<tokens>-b10/`，留出输出为 `init-check-<tokens>-fit-b10/`。`fit.pt` 只有在全部拟合门槛通过时才登记 `frozen_fit_id`；它绑定代码、输入、token 数、窗口和实际参数块。初始化留出始终每布局 5 块，沿用相同 token 数。

如果需要在拟合结束后人工查看报告，再单独打开留出，下面给出 **65536 tokens、10 块拟合** 的完整示例。它与上方同规模 `prepare-k` 共用目录。选择一种运行方式即可；已完成的相同设置会读取已有报告。

**单独拟合用时：待本机实测。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
unset LH_CHECKPOINT
LH_TOKENS=65536 LH_BLOCKS=10 \
LH_CURVATURE_DIR="$LH_CURVATURE_ROOT/init-fit-65536-b10" \
bash scripts/run_lm_head_h100.sh curvature-fit
```

只有该拟合 `passed=True`、`fit_scale.action=ready` 且 `frozen_fit_id` 非空时，执行下面留出；未通过就停止。

**单独留出用时：待本机实测；固定每布局 5 块。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
unset LH_CHECKPOINT
LH_TOKENS=65536 LH_BLOCKS=5 \
LH_FIT_DIR="$LH_CURVATURE_ROOT/init-fit-65536-b10" \
LH_CURVATURE_DIR="$LH_CURVATURE_ROOT/init-check-65536-fit-b10" \
bash scripts/run_lm_head_h100.sh curvature-check
```

留出失败保留为失败，不根据同一留出结果再调代理。初始化留出通过后，当共同 `.env` 导出了 `LH_ENV_FILE` 且尚未绑定其他 `K_STATISTICS` 时，程序会自动写入实际 `K-approved.pt` 路径。若提示已保留其他路径，先核对所用代码、输入和正式运行身份，停止处理，勿强制覆盖后续训练正在使用的 K。后期携带 `LH_CHECKPOINT` 的漂移检查不会改写 `.env`。

## 训练后曲率漂移

这一步使用 G clean 的 CE 赢家，在 step 191、1908 和终点分别测量；沿用初始化确定的 token 数，每布局默认 2 块。它衡量漂移，不生成新的 K 训练产物。

先从对应 `search.json` 找到赢家 candidate，在平台任务环境设置 `LH_CHECKPOINT` 为该赢家的 `analysis/head-000191.pt`（其他时点相应替换，终点也可用赢家 `final.pt`）。在任务中执行：

**本步骤用时（后期漂移拟合）：暂无可靠小时数；先参考同机器初始化拟合的实测用时，再用首个时点校准。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
: "${LH_CHECKPOINT:?设置为实际 G clean 赢家的 Head 快照绝对路径}"
LH_TOKENS=32768 LH_BLOCKS=2 \
LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261116-r1/drift-G-winner-step191-fit \
bash scripts/run_lm_head_h100.sh curvature-fit
```

这里的 32768 应替换为初始化最终规模；每个优化器、时点和阶段都使用不同目录。拟合完成后，保持同一 `LH_CHECKPOINT`，执行：

**本步骤用时（后期漂移留出）：暂无可靠小时数；先参考同机器初始化留出的实测用时，再用首个时点校准。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
: "${LH_CHECKPOINT:?设置为与拟合相同的 G clean 赢家 Head 快照绝对路径}"
LH_TOKENS=32768 LH_BLOCKS=2 \
LH_FIT_DIR=/data/260010171/results/lm-head-curvature-m16-20261116-r1/drift-G-winner-step191-fit \
LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261116-r1/drift-G-winner-step191-check \
bash scripts/run_lm_head_h100.sh curvature-check
```

后期 `passed` 主要反映数值检查；结构误差保留为漂移，不用于重新校准初始化代理。做完后清除任务环境中的 `LH_CHECKPOINT`，避免后续初始化测量误用训练后的权重。

## 用时估算与更新

### 当前数字从哪里来

原始依据是 2026-11-15 的 RTX PRO 6000、micro=16 硬件短测，使用旧输入，只用于工程用时估算。[原始汇总](docs/experiments/lm-head-only/history/20261115-selection/rtx-benchmark-result.json)和[本次计算清单](docs/experiments/lm-head-only/runtime-estimates.json)保留实测字段、源文件哈希与外推公式。K 短测使用经验统计包，不包含正式 Hessian 子块与留出验收。

| 目标 | 四优化器顺序短测实测合计 | 按 RTX 速度外推，四卡 clean | 按同速度，clean 后其余 34 节点 |
|---|---:|---:|---:|
| G | 3.25 分钟 | 11.4 小时 | 16.2 天 |
| T | 9.63 分钟 | 35.9 小时 | 50.9 天 |
| K | 5.72 分钟 | 21.2 小时 | 30.1 天 |

短测合计是 candidate 内部计时，包含评估与保存，不含完整命令启动/输入验证；手册的 smoke 预留值另给启动和上传余量。准备、汇总与恢复初始化的分钟数也是操作预留值，不是服务器测量结果。

对每个优化器，从 32 步的 `performance.json` 取总耗时 W、评估耗时 E、保存耗时 C：

- 训练与其余开销的每步近似：`s = (W − E − C) / 32`。
- 一条完整候选近似：`t = 3815 × s + 21 × (E + C) / 2`。短测评估两次，正式预算评估 21 次。
- 四卡 clean：四优化器各占一张卡、各顺序跑八候选，取 `8 × max(t)`。
- 已完成 clean 的其余网格：约 `34 × clean用时`；从零运行完整 grid 则约 `35 × clean用时`。

这是全候选正常跑满的外推。短测同步开销、正式诊断频率、额外 Head 快照、归档与慢盘都会影响误差；数值失败提前结束会缩短部分任务，故障重跑会延长。H100 的 0.5–1.5 倍范围只是排期敏感性假设，不是置信区间或保证范围，不应用理论峰值算力直接换算。四卡情况下 G/T/K 按目标顺序运行；两卡总体工作量粗略约乘二，单卡约乘四，任务不均衡另计。

### 有 H100 日志后怎样更新

先用 H100 四优化器 smoke 的 `performance.json` 代入同一公式。正式训练运行到至少一次完整评估之后（约 step 191），再用覆盖训练、评估和保存的一段时间除以实际完成步数，更新每步耗时。对暂停恢复后的新 attempt 只取该段实际步数，不把历史已完成步算进去。

四卡 clean 的剩余时间，按每个优化器分别加总“当前候选剩余步数耗时 + 后续候选完整耗时”，最后取四者最大值。grid 有多个排队 job，需按实际 GPU 分配累计，不能只用当前单候选的进度条代表整次搜索。

### K 测量为什么暂不给固定小时数

当前还没有完整 FP64 测量的硬件记录。代码含 GPU token 累积、CPU 子块特征分解和固定梯度检查，耗时不能从训练 smoke 推导。

日志依次出现 `curvature fit/check: done/total tokens` 和 `fixed gradients fit/check: i/n`。token 累积进度到 100% 后仍有谱分解与梯度检查，任务并未结束。拟合的梯度阶段是 `n=16` 个完整 batch，留出是 `n=2` 个评估池，不能把 fit 和 check 的单组时间互换。

在梯度阶段连续两组完成后，可用该阶段平均每组耗时 × 剩余组数估剩余时间，再留出最终汇总/写盘余量；整个阶段第一次完成后，用 `report.json.elapsed_seconds` 登记真实用时，并补计命令启动和输入验证。扩到 65536/131072 tokens 只会增加部分测量工作，固定梯度检查并不随之成倍增长；每布局从 5 块增至 10 块会增加块计算和谱分解。未拆分计时前，不把整轮简单乘二或乘四。

训练后曲率每布局只有 2 块，但仍做对应梯度检查，先按同机器初始化阶段的实测用时预留，再根据首个时点更新。当前同样没有独立的可靠小时估计。
