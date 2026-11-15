# LM Head-only：H100 操作手册

按顺序完成：**准备 → G 短测与正式训练 → T 短测与正式训练 → K 拟合、验收与训练 → 查看结果**。同一组 GPU 上，前一步退出后再运行下一步。

G 是真实 GPT 损失；T 是初始化处的二次展开；K 是用 Hessian 统计构造的二次代理。三者都只更新完整 LM Head，固定 micro=16。本文使用代码版本 `lm-head-h100-m16-20261115-v3`。

## 用时先看这里

以下按 **4 张空闲 H100、micro=16** 估计墙钟时间（不是 GPU 小时）；smoke 和曲率只用第一张卡，formal/grid 才用四卡并行。不含平台排队、输入制备与人工审查时间。**尚无本版 H100 实测**：训练范围是假设 H100 端到端耗时为 RTX 实测外推的 0.5–1.5 倍所得的情景估计，不是已测加速比；首轮 H100 数据到手后替换。计算依据见[用时估算与更新](LM_HEAD_REFERENCE.md#用时估算与更新)。

| 流程 | 暂定用时 / 排期 |
|---|---|
| 下载代码、配置路径与 Key | 预留 5–15 分钟 |
| 安装 SDK、check、plan | 预留 5–15 分钟，受下载和磁盘影响 |
| G smoke → G formal | smoke 预留 3–8 分钟；formal 约 6–18 小时 |
| T smoke → T formal | smoke 预留 6–18 分钟；formal 约 18–54 小时 |
| K 拟合 → 独立留出 | 暂无可靠小时数；各阶段首轮计时，不能用 K smoke 代替估算 |
| K smoke → K formal | smoke 预留 4–12 分钟；formal 约 11–32 小时 |
| status + report | clean 阶段通常预留 1–3 分钟；完整 grid 汇总预留 5–20 分钟 |
| clean 已完成后，G / T / K 的其余 34 节点 | 分别约 8–25 / 25–77 / 15–46 天；按正常候选全预算外推 |
| resume / pause | 恢复启动预留 1–5 分钟，之后只跑剩余工作；暂停请求数秒，实际退出还需等待更新边界和保存 |

三目标 clean 的训练部分合计约 **35–104 小时**，另加 K 测量及审查。网格远长于 clean，先看 clean 结果再决定是否提交；少于四张卡时 formal/grid 会更慢，不能直接沿用本表。

## 1. 准备代码、路径和 SwanLab（只做一次）

**操作预留：5–15 分钟。**假设正式输入已准备好，不含重新制备或传输数据。

在 H100 平台任务环境中配置保密变量 `SWANLAB_API_KEY`，后续每次任务都要继承它。然后下载固定代码：

```bash
set -euo pipefail
cd /data/260010171
git clone --branch lm-head-h100-m16-20261115-v3 --single-branch \
  https://github.com/ZZZ-020069/nanogpt-noise-experiments.git \
  nanogpt-lm-head-h100-m16-20261115-v3
cd nanogpt-lm-head-h100-m16-20261115-v3
cp -n scripts/lm_head_h100.env.example /data/260010171/lm-head-h100-m16-v3.env
```

打开复制出的 `.env` 文件，确认以下配置，再继续：

| 配置 | 填什么 |
|---|---|
| `PYTHON` | 实验 Python 的命令或绝对路径；要求 Python≥3.11、PyTorch≥2.6,<3、NumPy≥1.26,<3 |
| `INPUT_ROOT` | 已登记的正式 v1.8 输入目录；默认 `/data/260010171/inputs-v1.8` |
| `LH_RUN_ROOT` | 全新的结果目录；默认 `/data/260010171/results/lm-head-m16-20261115-v3` |
| `GPUS` | 本次分配的 GPU 编号，四卡默认 `0,1,2,3` |
| SwanLab | 保持 `NANOGPT_SWANLAB=1`；项目默认 `nanogpt-lm-head`，团队账号另填 `SWANLAB_WORKSPACE` |

**以下每个命令块都已包含目录切换和环境加载，可以整块复制到一个平台任务中。按顺序执行，通过该步骤的检查后，再复制下一块。**

路径不同就统一替换文中的路径。已有 v3 checkout 可直接使用；已有任务继续原版本，运行中不更新代码。

## 2. 检查环境和输入

**预留：5–15 分钟。**仅 plan 通常是秒级，安装与输入哈希核验占主要时间。

整块复制执行。安装 SDK 只需一次，使用独立实验环境，勿升级旧任务正在使用的共享环境。

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
"$PYTHON" -m pip install "swanlab==0.10.5"
bash scripts/run_lm_head_h100.sh check
bash scripts/run_lm_head_h100.sh plan
```

**继续条件：**没有报错；plan 显示 `micro=16`、`accumulations=32`、`steps=3815`、每搜索 8 候选。check 只检查 SwanLab 配置和 SDK，下一步才验收云端连接。报错时先按文末处理表修复。

## 3. 跑 G：先短测，再正式训练

**用时：smoke 预留 3–8 分钟；四卡 formal 暂估 6–18 小时。**

先做 smoke（四优化器各 32 步，依次使用第一张 GPU）：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=G bash scripts/run_lm_head_h100.sh smoke
```

**继续条件：**命令正常结束，`$LH_RUN_ROOT/smoke/G/passed.json` 的 `passed` 为 `true`；SwanLab 项目中四优化器均有数据，`progress/step` 到 32。两项都满足后，另开任务运行：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=G bash scripts/run_lm_head_h100.sh formal
```

formal 执行零噪声 clean：四优化器各搜索 8 个学习率，共 32 条候选，每条正常完成 3815 步。脚本自动搜索学习率，无需手动调参。

**结束后检查：**`$LH_RUN_ROOT/G/summary.json` 中 `completed_jobs=4`。这是四个搜索完成；个别候选数值失败会保留记录。若某优化器全部候选失败，先审查失败记录，不给它强排最后一名。

## 4. 跑 T：重复相同验收流程

**用时：smoke 预留 6–18 分钟；四卡 formal 暂估 18–54 小时。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=T bash scripts/run_lm_head_h100.sh smoke
```

确认 `smoke/T/passed.json` 和 SwanLab 四优化器数据后，另开任务：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=T bash scripts/run_lm_head_h100.sh formal
```

结束后检查 `T/summary.json` 的 `completed_jobs=4`。G/T 均不需要 K 文件。

## 5. 准备 K：先拟合，再独立验收

**用时：拟合与留出均待首轮计时，当前没有可信的数值估计。**两者包含 FP64 子块、谱分解和梯度探针；拟合测 16 个完整训练 batch，留出测 train/val 各 200 条序列，不能假定两阶段一样久。日志如何估剩余时间见[参考页](LM_HEAD_REFERENCE.md#用时估算与更新)。

等待训练退出。先在初始化处测量并拟合，使用第一张 GPU：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
unset LH_CHECKPOINT
LH_TOKENS=32768 LH_BLOCKS=5 \
LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261115-v3/init-fit-32k \
bash scripts/run_lm_head_h100.sh curvature-fit
```

打开该目录的 `report.json`。**只有 `passed=true` 才继续。**若失败，先看文末处理表；扩样规则与命令见[拟合扩样](LM_HEAD_REFERENCE.md#拟合扩样)。

拟合通过后，用相同 token 数做独立留出验收：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
unset LH_CHECKPOINT
LH_TOKENS=32768 LH_BLOCKS=5 \
LH_FIT_DIR=/data/260010171/results/lm-head-curvature-m16-20261115-v3/init-fit-32k \
LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261115-v3/init-check-32k \
bash scripts/run_lm_head_h100.sh curvature-check
```

**继续条件：**留出目录的 `report.json` 为 `passed=true`，并生成 `K-approved.pt`。将该文件的绝对路径填入共同 `.env` 文件中的 `K_STATISTICS`，后续任务重新 source。如果拟合扩样过，留出命令的 token 数和路径也必须对应修改。

留出失败就保留失败结果，暂停 K；不要根据已看过的留出结果反复调到通过。曲率阶段暂不上传 SwanLab，进度看平台日志，结果看 `report.json`。

## 6. 跑 K：先短测，再正式训练

**用时：smoke 预留 4–12 分钟；四卡 formal 暂估 11–32 小时。**不包含第 5 步的曲率测量。

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=K bash scripts/run_lm_head_h100.sh smoke
```

确认 `smoke/K/passed.json` 和 SwanLab 四优化器数据后，另开任务：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=K bash scripts/run_lm_head_h100.sh formal
```

结束后检查 `K/summary.json` 的 `completed_jobs=4`。

## 7. 查看结果，再决定是否跑噪声网格

**汇总预留：clean 1–3 分钟，完整 grid 5–20 分钟。**这是文件扫描的操作预留；存储慢时可能更久。

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
bash scripts/run_lm_head_h100.sh status
bash scripts/run_lm_head_h100.sh report
```

看 `$LH_RUN_ROOT/comparison.csv`：每行是一项目标、一个噪声节点、一个优化器的搜索结果。clean 对应 `node=0`；G/T/K 均完成时有 12 行。主要比较终点 `ce`（越低越好），同时查看 `status`、`valid_candidates` 和 `boundary_unclosed`。**排序不一致或最优 LR 仍在边界时，先报告结果，不修改已完成实验。**

先审查 clean。决定开展完整噪声网格后，按需要选择下面的目标；每块单独提交，前一个任务退出后再提交下一个。

**G 噪声网格：clean 已完成后，剩余暂估 8–25 天。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=G bash scripts/run_lm_head_h100.sh grid
```

**T 噪声网格：clean 已完成后，剩余暂估 25–77 天。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=T bash scripts/run_lm_head_h100.sh grid
```

**K 噪声网格：clean 已完成后，剩余暂估 15–46 天（正式 K 文件已配置，K smoke 已通过）。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=K bash scripts/run_lm_head_h100.sh grid
```

grid 每目标 35 节点 × 4 优化器 × 8 候选，复用同身份已完成的 clean。grid 不会因 clean 完成而自动启动。

## 中断与异常：按结果处理

| 看到什么 | 怎么处理 |
|---|---|
| 输入、路径或 Key/SDK 报错 | 修复路径或任务环境，重新执行 check。输入必须含正式登记；不能拿旧测速输入替代 |
| CUDA OOM | 先检查 GPU 是否被其他任务占用；释放占用或换到空闲 H100。保持 micro=16；空闲卡仍 OOM 时保留日志并暂停排查 |
| smoke 成功但网页没数据 | 修复 Key、项目/空间或网络。已完成 smoke 重跑会复用结果；改用全新 `LH_RUN_ROOT` 重新 check 和 smoke，确认上传后再 formal |
| 正式训练中 SwanLab 断更 | 检查平台与本地 `progress.json`/日志，训练可能仍在继续；不要因为网页断更重复启动任务 |
| K 拟合 `stability_passed=false` | 数值检查正常时按 32768→65536→131072 扩样，每次使用新目录；先不打开留出 |
| K 的数值、结构或零空间检查失败 | 保存 `report.json`，先审查原因；不放宽阈值、加 ridge 或修改梯度来强行通过 |
| 退出码 75、平台中断或基础设施故障 | 等原控制器和 worker 全部退出，保持代码、输入、结果根与 micro 不变，按下方命令恢复 |

## 断点续训：只复制对应的一块

**恢复初始化预留 1–5 分钟；训练耗时按未完成候选和剩余步数计算，不会重新跑完整预算。**若只剩一个候选，最多只会占一张卡，不能把它的剩余时间再除以四。

确认原控制器和所有 worker 已退出，并继续使用原代码、输入、结果目录和 micro。已完成候选自动跳过，被中断候选从最近检查点恢复；强制中断可能重算检查点后的步骤。K 恢复仍需原来匹配的 `K_STATISTICS`。

### 恢复 clean 正式训练（原来执行 formal）

**恢复 G clean：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=G LH_PHASE=clean bash scripts/run_lm_head_h100.sh resume
```

**恢复 T clean：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=T LH_PHASE=clean bash scripts/run_lm_head_h100.sh resume
```

**恢复 K clean：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=K LH_PHASE=clean bash scripts/run_lm_head_h100.sh resume
```

### 恢复 完整噪声网格（原来执行 grid）

**恢复 G grid：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=G LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**恢复 T grid：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=T LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**恢复 K grid：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
OBJECTIVE=K LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

### 主动暂停与 smoke 中断

**暂停请求通常数秒返回；训练实际退出还需等待当前更新与检查点写入，可能是几十秒到数分钟。**确认退出后再恢复。

有可用交互终端时，复制下面整块请求当前结果根的训练保存并暂停：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115-v3
source /data/260010171/lm-head-h100-m16-v3.env
bash scripts/run_lm_head_h100.sh pause
```

平台直接停止任务后，也使用上面对应的 resume 命令。smoke 中断则重新复制第 3、4 或 6 步对应目标的完整 smoke 命令；曲率测量不使用训练 resume。不要并开两个同根任务或手工删锁。

更多指标、文件位置、扩样和训练后曲率命令见[参考页](LM_HEAD_REFERENCE.md)。
