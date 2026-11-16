# LM Head-only：H100 操作手册

按顺序完成：**准备与检查 → K 初始化拟合与留出 → G/T/K 联合短测（`smoke-all`） → G→T→K 一键顺序正式 clean（`clean-all`） → 停下查看与分析结果 → 按需逐目标启动 `grid`（保持 G → T → K 人工门控）**。同一组 GPU 上，前一步退出后再运行下一步。

G 是真实 GPT 损失；T 是初始化处的二次展开；K 是用 Hessian 统计构造的二次代理。三者都只更新完整 LM Head，固定 `micro=16`（单卡累积 32 次）。当前分支 `design/lm-head-only-20261114` 已合入 `K` / `T` 的 **Layer 1 Triton 融合快路径**（自动兼容 PyTorch 行分块回退），不改变任何科学协议或优化超参数。

## 用时先看这里

以下按 **4 张空闲 H100、micro=16** 估计墙钟时间（不是 GPU 小时）；smoke 和曲率只用第一张卡，clean/grid 才用四卡并行。不含平台排队、输入制备与人工审查时间。**尚无本版 H100 实测**：

- `G` 沿用 RTX PRO 6000 四优化器基准外推（[runtime-estimates.json](docs/experiments/lm-head-only/runtime-estimates.json)）；
- `T` 与 `K` 已按合入 Layer 1 Triton 加速后的 RTX PRO 6000（CUDA BF16，`adamw` / `muon` 32-step E2E 实测：`T` `39.42s / 39.79s`、`K` `32.11s / 31.64s`）套用同一公式与 `0.5–1.5x` H100 排期敏感性区间更新；若环境退回 PyTorch 分块回退（未启用 Triton），`T` / `K` 将退回加速前耗时（见[参考页](LM_HEAD_REFERENCE.md#用时估算与更新)）。首轮 H100 `smoke-all` 数据到手后按真实测速替换。

| 流程 | 暂定用时 / 排期（已含 Layer 1 加速） |
|---|---|
| 下载代码、配置路径与 Key | 预留 5–15 分钟 |
| 安装 SDK、check、plan | 预留 5–15 分钟，受下载和磁盘影响 |
| K 初始化拟合 → 独立留出 | 暂无可靠小时数；各阶段首轮计时，不能用 K smoke 代替估算 |
| G/T/K 联合冒烟（`smoke-all`） | 预留 10–25 分钟（三目标 × 四优化器在第一张卡上顺序执行） |
| G → T → K 自动顺序 clean（`clean-all`） | 合计约 **15–45 小时**（其中 `G` 6–18h、`T` 5–15h、`K` 4–12h） |
| status + report | clean 阶段通常预留 1–3 分钟；完整 grid 汇总预留 5–20 分钟 |
| clean 已完成后，G / T / K 的其余 34 节点（`grid`） | 分别约 8–25 / 7–21 / 5–17 天；**必须人工审查后分目标提交** |
| resume / pause | 恢复启动预留 1–5 分钟，之后只跑剩余工作；暂停请求数秒，实际退出还需等待更新边界和保存 |

正式 `clean` 阶段三目标均需完整运行以建立排序基准，因此通过 `clean-all` 连续自动执行，跑完后统一停下人工分析；后续 `grid` 耗时以天计，仍严格保持 `G → 审查 → T → 审查 → K` 的门控顺序，少于四张卡时 clean/grid 会更慢，不能直接沿用本表。

## 1. 准备代码、路径和 SwanLab（只做一次）

在 H100 平台任务环境中配置保密变量 `SWANLAB_API_KEY`，后续每次任务都要继承它。然后下载最新正式分支代码：

**本步骤用时：下载与配置预留 5–15 分钟，不含输入制备或传输。**

```bash
set -euo pipefail
cd /data/260010171
git clone --branch design/lm-head-only-20261114 --single-branch \
  https://github.com/ZZZ-020069/nanogpt-noise-experiments.git \
  nanogpt-lm-head-h100-m16-20261116
cd nanogpt-lm-head-h100-m16-20261116
cp -n scripts/lm_head_h100.env.example /data/260010171/lm-head-h100-m16-20261116.env
```

打开复制出的 `.env` 文件，确认以下配置，再继续：

| 配置 | 填什么 |
|---|---|
| `PYTHON` | 实验 Python 的命令或绝对路径；要求 Python≥3.11、PyTorch≥2.6,<3、NumPy≥1.26,<3（Linux CUDA 版 PyTorch 通常自带 Triton） |
| `INPUT_ROOT` | 已登记的正式 v1.8 输入目录；默认 `/data/260010171/inputs-v1.8` |
| `LH_RUN_ROOT` | 全新的结果目录；默认 `/data/260010171/results/lm-head-m16-20261116` |
| `GPUS` | 本次分配的 GPU 编号，四卡默认 `0,1,2,3` |
| SwanLab | 保持 `NANOGPT_SWANLAB=1`；项目默认 `nanogpt-lm-head`，团队账号另填 `SWANLAB_WORKSPACE` |

**以下每个命令块都已包含目录切换和环境加载，可以整块复制到一个平台任务中。按顺序执行，通过该步骤的检查后，再复制下一块。**

路径不同就统一替换文中的路径。不要在已开始正式训练的结果目录中途升级代码版本（`code_identity` 会严格校验代码指纹）。

## 2. 检查环境和输入

整块复制执行。安装 SDK 只需一次，使用独立实验环境，勿升级旧任务正在使用的共享环境；顺带检测 Triton 融合核是否可用（无 GPU 节点时自动跳过）：

**本步骤用时：安装与检查预留 5–15 分钟；仅 plan 通常为秒级。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
"$PYTHON" -m pip install "swanlab==0.10.5"
"$PYTHON" -c "import torch, nanogpt_noise.lm_head.kernels as k; print('cuda=', torch.cuda.is_available(), 'triton_ready=', k.triton_ready(torch.device('cuda')) if torch.cuda.is_available() else 'skip_no_cuda')"
bash scripts/run_lm_head_h100.sh check
bash scripts/run_lm_head_h100.sh plan
```

**继续条件：**没有报错；plan 显示 `micro=16`、`accumulations=32`、`steps=3815`、每搜索 8 候选；若当前在 GPU 任务节点上，确认输出 `triton_ready= True`（若为 `False` 会自动回退到等价 PyTorch 分块，正确性不受影响但 `T` / `K` 会明显变慢）。check 只检查 SwanLab 配置和 SDK，第 4 步才验收云端连接。报错时先按文末处理表修复。

## 3. 准备 K：初始化处拟合与独立留出验收

初始化处的曲率拟合与留出测量只使用第一张 GPU、共同初始权重和 `INPUT_ROOT`，**不依赖 `G` 或 `T` 先完成**；提前完成并产出自检合格的 `K-approved.pt` 后，第 4 步即可把 `G / T / K` 一起冒烟，第 5 步即可一键顺序跑完 `clean`。

> 注：拟合与留出包含 FP64 子块、谱分解和梯度探针；拟合测 16 个完整训练 batch，留出测 train/val 各 200 条序列，不能假定两阶段一样久。日志如何估剩余时间见[参考页](LM_HEAD_REFERENCE.md#用时估算与更新)。

先在初始化处测量并拟合：

**本步骤用时（K 拟合）：暂无可靠小时数，需首轮计时；不能用 K smoke 估时。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
unset LH_CHECKPOINT
LH_TOKENS=32768 LH_BLOCKS=5 \
LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261116/init-fit-32k \
bash scripts/run_lm_head_h100.sh curvature-fit
```

打开该目录的 `report.json`。**只有 `passed=true` 才继续。**若失败，先看文末处理表；扩样规则与命令见[拟合扩样](LM_HEAD_REFERENCE.md#拟合扩样)。

拟合通过后，用相同 token 数做独立留出验收：

**本步骤用时（K 独立留出）：暂无可靠小时数，需单独计时；不能假定与拟合等时。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
unset LH_CHECKPOINT
LH_TOKENS=32768 LH_BLOCKS=5 \
LH_FIT_DIR=/data/260010171/results/lm-head-curvature-m16-20261116/init-fit-32k \
LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261116/init-check-32k \
bash scripts/run_lm_head_h100.sh curvature-check
```

**继续条件：**留出目录的 `report.json` 为 `passed=true`，并生成 `K-approved.pt`。将该文件的绝对路径写入共同 `.env` 文件中的 `K_STATISTICS`（取消注释并核对路径），后续任务自动加载。如果拟合扩样过，留出命令的 token 数和路径也必须对应修改。

留出失败就保留失败结果，暂停 K；不要根据已看过的留出结果反复调到通过（此时仍可用单目标命令 `OBJECTIVE=G` / `OBJECTIVE=T` 单独跑 `G` 和 `T`）。曲率阶段暂不上传 SwanLab，进度看平台日志，结果看 `report.json`。

## 4. 联合冒烟（G / T / K 一起做 smoke）

`K_STATISTICS` 已配置后，整块运行即可在第一张 GPU 上顺序完成 `G → T → K` 三目标的 32-step smoke 验收：

**本步骤用时（`smoke-all`）：预留 10–25 分钟，三目标各跑 4 个优化器 × 32 步。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
bash scripts/run_lm_head_h100.sh smoke-all
```

**继续条件：**命令正常退出，`$LH_RUN_ROOT/smoke/{G,T,K}/passed.json` 的 `passed` 均为 `true`；在 SwanLab 网页项目中确认 `G` / `T` / `K` 各 4 个优化器均有数据且 `progress/step` 到 32，同时可检查三目标 `performance.json` 确认 H100 单步耗时与 Triton 加速生效（`T`/`K` 不应显著慢于 `G`）。若只想单测某一目标，仍可使用 `OBJECTIVE=G bash scripts/run_lm_head_h100.sh smoke`（`T` / `K` 同理）。

## 5. 自动顺序执行正式 clean（G → T → K 连续跑完）

确认三目标 `smoke-all` 与 SwanLab 数据均通过后，提交一条命令即可按 `G → T → K` 顺序自动跑完全部正式 `clean` 实验（每目标四优化器各搜索 8 个学习率，共 32 条候选 × 3815 步，脚本自动扩边搜索），**中间无需人工干预**：

**本步骤用时（`clean-all`）：四卡合计暂估 15–45 小时（G 6–18h + T 5–15h + K 4–12h，未启用 Triton 时会更久），待 H100 首轮实测校准。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
bash scripts/run_lm_head_h100.sh clean-all
```

- 启动前会自动检查 `K_STATISTICS` 以及 `$LH_RUN_ROOT/smoke/{G,T,K}/passed.json` 均已就绪；任一目标出现暂停（退出码 75）或异常退出时，脚本立即停止，不会继续下一目标。
- 若仍想拆开单跑某一目标（例如只重跑或单跑某个目标），保留 `OBJECTIVE=G|T|K bash scripts/run_lm_head_h100.sh formal`（完全等价）。

**全部结束后检查：**`$LH_RUN_ROOT/{G,T,K}/summary.json` 中 `completed_jobs` 均为 `4`。个别候选数值失败会保留记录；若某优化器全部候选失败，先审查失败记录，不给它强排最后一名。

## 6. 停下来查看结果并审查，再决定是否跑噪声网格

`clean-all` 全部跑完后，先停下来汇总并人工审查 `clean` 结果（以及按需做后期曲率漂移测量，见[训练后曲率漂移](LM_HEAD_REFERENCE.md#训练后曲率漂移)）：

**本步骤用时：clean 结果汇总预留 1–3 分钟；完整 grid 汇总预留 5–20 分钟，慢盘另计。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
bash scripts/run_lm_head_h100.sh status
bash scripts/run_lm_head_h100.sh report
```

看 `$LH_RUN_ROOT/comparison.csv`：每行是一项目标、一个噪声节点、一个优化器的搜索结果。clean 对应 `node=0`；G/T/K 均完成时有 12 行。主要比较终点 `ce`（越低越好），同时查看 `status`、`valid_candidates` 和 `boundary_unclosed`。**排序不一致或最优 LR 仍在边界时，先报告并分析结果，不修改已完成实验。**

决定开展完整噪声网格后，**`grid` 阶段仍严格保持 `G → 审查 → T → 审查 → K` 的门控顺序**；每块单独提交，前一个任务退出并人工确认后再提交下一个：

**G 噪声网格：**

**本步骤用时（G grid）：clean 已完成时，四卡剩余暂估 8–25 天，待 H100 实测校准。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=G bash scripts/run_lm_head_h100.sh grid
```

**T 噪声网格：**

**本步骤用时（T grid）：clean 已完成时，四卡剩余暂估 7–21 天（开启 Triton），待 H100 实测校准。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=T bash scripts/run_lm_head_h100.sh grid
```

**K 噪声网格（正式 K 文件已配置，K smoke 已通过）：**

**本步骤用时（K grid）：clean 已完成时，四卡剩余暂估 5–17 天（开启 Triton），待 H100 实测校准。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=K bash scripts/run_lm_head_h100.sh grid
```

grid 每目标 35 节点 × 4 优化器 × 8 候选，复用同身份已完成的 clean。grid 不会因 clean 完成而自动启动。

## 中断与异常：按结果处理

| 看到什么 | 怎么处理 |
|---|---|
| 输入、路径或 Key/SDK 报错 | 修复路径或任务环境，重新执行 check。输入必须含正式登记；不能拿旧测速输入替代 |
| `triton_ready` 为 `False` 或 `T`/`K` 速度偏慢 | 检查环境是否安装 Triton、CUDA driver/stub（`kernels.py` 会自动扫描 `/usr/local/cuda*/lib64/stubs`）及是否被强制禁用；缺失时会自动回退 PyTorch 行分块，结果一致但耗时更长 |
| CUDA OOM | 先检查 GPU 是否被其他任务占用；释放占用或换到空闲 H100。保持 micro=16；空闲卡仍 OOM 时保留日志并暂停排查 |
| smoke 成功但网页没数据 | 修复 Key、项目/空间或网络。已完成 smoke 重跑会复用结果；改用全新 `LH_RUN_ROOT` 重新 check 和 smoke，确认上传后再 formal |
| 正式训练中 SwanLab 断更 | 检查平台与本地 `progress.json`/日志，训练可能仍在继续；不要因为网页断更重复启动任务 |
| K 拟合 `stability_passed=false` | 数值检查正常时按 32768→65536→131072 扩样，每次使用新目录；先不打开留出 |
| K 的数值、结构或零空间检查失败 | 保存 `report.json`，先审查原因；不放宽阈值、加 ridge 或修改梯度来强行通过 |
| 退出码 75、平台中断或基础设施故障 | 等原控制器和 worker 全部退出，保持代码、输入、结果根与 micro 不变，按下方命令恢复 |

## 断点续训：只复制对应的一块

若只剩一个候选，最多只会占一张卡，不能把它的剩余时间再除以四。

确认原控制器和所有 worker 已退出，并继续使用原代码、输入、结果目录和 micro。已完成候选自动跳过，被中断候选从最近检查点恢复；强制中断可能重算检查点后的步骤。K 恢复仍需原来匹配的 `K_STATISTICS`。

### 恢复 clean 正式训练（整串 `clean-all` 或单目标 `formal`）

**一键恢复并跑完剩余 G → T → K clean（原来执行 `clean-all`）：**

**本步骤用时（恢复 `clean-all`）：初始化预留 1–5 分钟；已完成目标自动快速跳过，仅执行剩余目标与剩余步数。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
LH_PHASE=clean-all bash scripts/run_lm_head_h100.sh resume
```

若只想恢复单一目标 clean：

**恢复 G clean：**

**本步骤用时（恢复 G clean）：初始化预留 1–5 分钟，之后仅执行剩余候选和步数；剩余时长按当前进度估算。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=G LH_PHASE=clean bash scripts/run_lm_head_h100.sh resume
```

**恢复 T clean：**

**本步骤用时（恢复 T clean）：初始化预留 1–5 分钟，之后仅执行剩余候选和步数；剩余时长按当前进度估算。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=T LH_PHASE=clean bash scripts/run_lm_head_h100.sh resume
```

**恢复 K clean：**

**本步骤用时（恢复 K clean）：初始化预留 1–5 分钟，之后仅执行剩余候选和步数；剩余时长按当前进度估算。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=K LH_PHASE=clean bash scripts/run_lm_head_h100.sh resume
```

### 恢复 完整噪声网格（原来执行 grid）

**恢复 G grid：**

**本步骤用时（恢复 G grid）：初始化预留 1–5 分钟，之后仅执行剩余候选和步数；剩余时长按当前进度估算。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=G LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**恢复 T grid：**

**本步骤用时（恢复 T grid）：初始化预留 1–5 分钟，之后仅执行剩余候选和步数；剩余时长按当前进度估算。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=T LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**恢复 K grid：**

**本步骤用时（恢复 K grid）：初始化预留 1–5 分钟，之后仅执行剩余候选和步数；剩余时长按当前进度估算。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=K LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

### 主动暂停与 smoke 中断

有可用交互终端时，复制下面整块请求当前结果根的训练保存并暂停：

**本步骤用时：暂停请求通常数秒返回；实际退出还需几十秒至数分钟等待更新与保存。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
bash scripts/run_lm_head_h100.sh pause
```

平台直接停止任务后，也使用上面对应的 resume 命令。smoke 中断则重新复制第 4 步 `smoke-all`（或单目标 `smoke`）命令；曲率测量不使用训练 resume。不要并开两个同根任务或手工删锁。

更多指标、文件位置、扩样和训练后曲率命令见[参考页](LM_HEAD_REFERENCE.md)。
