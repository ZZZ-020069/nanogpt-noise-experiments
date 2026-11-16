# LM Head-only：H100 运行操作手册

在 4 卡 H100 上按以下 5 步顺序执行，即可完成 `clean` 阶段并自动生成汇总结果表；**同一组 GPU 上，务必等前一个任务彻底退出后再提交下一个任务**。

```text
1. 下载与配置  →  2. 环境检查  →  3. 准备 K (prepare-k)  →  4. 联合短测 (smoke-all)  →  5. 自动顺序跑完 clean (clean-all)
```

> `clean-all` 全部跑完后会自动汇总并停止；确认结果后，再决定是否按第 6 步逐个提交 `grid` 任务。

运行期间请保持代码与目录锁定：不要在已经跑过任务的目录里执行 `git pull`，也不要混用其他目录或旧批次的中间产物。

## 耗时参考（4 张 H100）

| 步骤 | 说明 | 预估墙钟时间 |
|---|---|---|
| 1–2. 配置与检查 | 克隆代码、配置 `.env`、检查输入与环境 | 各 5–15 分钟 |
| 3. `prepare-k` | 生成并校验 K 统计量（只用第 1 张卡） | 待本机首轮实测（按日志进度估算） |
| 4. `smoke-all` | G / T / K 联合 32 步短测（只用第 1 张卡） | 约 10–25 分钟 |
| 5. `clean-all` | 自动顺序 G → T → K，每目标四卡并行 | 旧版本参照合计 35–104 小时（G 6–18 / T 18–54 / K 11–32 小时）；当前 `T`/`K` 已加速，具体耗时以第 4 步 `smoke-all` 实测为准 |
| 6. `grid`（可选） | `clean` 后其余 34 个噪声节点，逐目标提交 | 旧版本参照 G 8–25 / T 25–77 / K 15–46 天；当前版待 `clean` 实测后校准 |
| 暂停 / 恢复 / 查进度 | 见文末对应命令 | 暂停请求与状态查询约 1 分钟内返回；恢复初始化约 5–15 分钟，之后只跑剩余步数 |

注：上表训练耗时为旧版硬件基准按 0.5–1.5 倍波动外推得到的保守参照（[计算清单](docs/experiments/lm-head-only/runtime-estimates.json)），不含平台排队、人工审查与故障重跑时间。新 `T`/`K` 加速后的 H100 实际速度，可在第 4 步 `smoke-all` 结束后按输出耗时直接换算。

---

## 1. 下载代码与配置（只做一次）

**本步骤用时：预留 5–15 分钟（取决于网络和已有环境）。**

先在平台任务环境变量中配置 **`SWANLAB_API_KEY`**（后续每个任务均需继承该变量），然后执行以下整块命令克隆并锁定代码版本：

```bash
set -euo pipefail
cd /data/260010171
git clone --branch design/lm-head-only-20261114 --single-branch \
  https://github.com/ZZZ-020069/nanogpt-noise-experiments.git \
  nanogpt-lm-head-h100-m16-20261116-r1
cd nanogpt-lm-head-h100-m16-20261116-r1
git checkout --detach HEAD
git rev-parse HEAD
cp -n scripts/lm_head_h100.env.example /data/260010171/lm-head-h100-m16-20261116-r1.env
source /data/260010171/lm-head-h100-m16-20261116-r1.env
```

随后打开 `/data/260010171/lm-head-h100-m16-20261116-r1.env`，按当前 H100 平台实际路径核对并修改以下配置：

| 变量 | 填写说明 |
|---|---|
| `PYTHON` | 实验 Python 路径（要求 Python ≥ 3.11、PyTorch ≥ 2.6 且 < 3、NumPy ≥ 1.26 且 < 3） |
| `INPUT_ROOT` | 已就绪的正式输入目录（默认 `/data/260010171/inputs-v1.8`） |
| `LH_RUN_ROOT` | 本次全新结果目录（默认 `/data/260010171/results/lm-head-m16-20261116-r1`） |
| `LH_CURVATURE_ROOT` | 本次 K 统计量输出目录（默认 `/data/260010171/results/lm-head-curvature-m16-20261116-r1`） |
| `GPUS` | 本次使用的 GPU 编号（4 卡填 `0,1,2,3`） |
| `SWANLAB_PROJ_NAME` | 默认 `nanogpt-lm-head`；如使用团队工作空间，取消注释并填写 `SWANLAB_WORKSPACE` |

> **`K_STATISTICS` 暂保持注释状态即可**，第 3 步通过后会自动写入该文件。若更改过目录名，请同步替换下文命令块中的对应路径；正式训练开始后请勿改动代码或 `.env`。

---

## 2. 检查环境与输入

**本步骤用时：预留 5–15 分钟（首次安装依赖时可能稍久）。**

整块复制执行，安装 `swanlab` 并自动核对 CUDA/Triton 状态、输入数据完整性与执行计划：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
"$PYTHON" -m pip install "swanlab==0.10.5"
"$PYTHON" -c "import torch, nanogpt_noise.lm_head.kernels as k; print('cuda=', torch.cuda.is_available(), 'triton_ready=', k.triton_ready(torch.device('cuda')) if torch.cuda.is_available() else 'skip_no_cuda')"
bash scripts/run_lm_head_h100.sh check
bash scripts/run_lm_head_h100.sh plan
```

**通过标准**：命令正常退出；`plan` 输出 `micro: 16`、`accumulations: 32`、`steps: 3815`；若当前节点带 GPU，终端输出 `triton_ready= True`（若为 `False` 不影响正确性，但 `T`/`K` 会明显变慢，建议先排查 Triton/CUDA 环境）。

---

## 3. 生成并校验 K 统计量（只用第 1 张卡）

**本步骤用时：暂无固定小时数，以本机实测与日志进度为准。**

运行后会自动先做拟合与稳定性检查；**只有全部拟合门槛通过**，才会继续执行独立留出校验，并在通过后**自动将 `K-approved.pt` 路径写入 `.env` 中的 `K_STATISTICS`**，最后打印汇总：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
LH_TOKENS=32768 LH_BLOCKS=5 bash scripts/run_lm_head_h100.sh prepare-k
```

**通过标准**：命令正常退出（退出码 0），终端末尾 `=== PREPARE-K SUMMARY ===` 显示 `fit: passed=True`、`check: passed=True`，且 `.env` 行已显示 `K_STATISTICS=.../K-approved.pt`。

**如果未通过（退出非零），请按终端输出的 `fit_scale.action` 与报告处理，不要跳过检查**：
- 若仅数据稳定性未通过（`fit_scale.action=expand_tokens` 且数值/结构/梯度检查均通过），按下面两档依次扩样（先跑 `65536`，仍提示此项再跑 `131072`）：
  **扩到 65536 tokens；本步骤用时：待本机实测。**
  ```bash
  set -euo pipefail
  cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
  source /data/260010171/lm-head-h100-m16-20261116-r1.env
  LH_TOKENS=65536 LH_BLOCKS=5 bash scripts/run_lm_head_h100.sh prepare-k
  ```
  **若 65536 仍仅稳定性未通过，扩到 131072 tokens（本轮上限）；本步骤用时：待本机实测。**
  ```bash
  set -euo pipefail
  cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
  source /data/260010171/lm-head-h100-m16-20261116-r1.env
  LH_TOKENS=131072 LH_BLOCKS=5 bash scripts/run_lm_head_h100.sh prepare-k
  ```
- 若数据已稳定但提示块间方差超阈值（`fit_scale.action=expand_blocks`），按[参考页 · 拟合扩块](LM_HEAD_REFERENCE.md#拟合扩块)在当前 token 数上改用 `LH_BLOCKS=10` 运行对应命令。
- 若出现数值/结构/梯度检查失败、`review_undefined_moments`、`unresolved_at_block_limit`、`131072` 仍未稳定，或独立留出（`check`）失败：**停止并保存对应目录下的 `report.json` 用于排查，不要继续执行 K 任务。**

> 拟合报告位于 `$LH_CURVATURE_ROOT/init-fit-<tokens>-b<块数>/report.json`，留出报告位于 `init-check-<tokens>-fit-b<拟合块数>/report.json`。不同规模自动分目录存放；若中途意外中断，重跑同一命令会自动从头重算未完成部分（已完整结束的同设置会直接复用结果）。

---

## 4. 联合短测（`smoke-all`，只用第 1 张卡）

**本步骤用时：预留 10–25 分钟，结束后需人工登录 SwanLab 网页核对曲线。**

在第一张卡上连续运行 `G`、`T`、`K` 三目标各 32 步短测，并在结束时自动打印耗时、显存与通过状态汇总：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
bash scripts/run_lm_head_h100.sh smoke-all
```

**通过标准（需同时满足以下两项）**：
1. 命令正常退出，终端末尾 `=== SMOKE-ALL SUMMARY ===` 中 **`G` / `T` / `K` 全部为 `passed=True`**，且单步耗时与显存正常；
2. 登录 SwanLab 网页进入项目 `nanogpt-lm-head`，确认 **`G`、`T`、`K` 各自 4 个优化器（共 12 条运行）均已收到指标，且 `progress/step` 到达 `32`**。

---

## 5. 自动顺序执行正式 `clean`（4 卡并行）

**本步骤用时：旧版参照合计 35–104 小时（开启当前 `T`/`K` 加速后预计明显缩短，具体请按第 4 步实测耗时估算）。**

确认第 4 步两项通过标准均已满足后，提交以下命令即可自动按 **`G → T → K` 顺序跑完整个 `clean` 阶段**，中间无需人工干预；全部结束后会自动汇总并输出 `comparison.csv`：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
bash scripts/run_lm_head_h100.sh clean-all
```

**完成标准**：
- 命令正常退出，终端末尾 `=== CLEAN-ALL SUMMARY ===` 显示 **`G`、`T`、`K` 的 `completed_jobs` 均为 `4`**，并已打印出 `$LH_RUN_ROOT/comparison.csv` 的全部 12 行汇总结果；
- 将 `$LH_RUN_ROOT/comparison.csv` 导出或发回分析，**到此先停下来等待结果审阅，不要直接提交第 6 步**。

---

## 6. 完整噪声网格（仅在 `clean` 结果确认后再逐个执行）

`grid` 阶段运行周期较长，须按 **`G` → 确认 → `T` → 确认 → `K`** 顺序分别提交；**务必等前一个任务彻底退出并人工确认后，再提交下一个**（同目录下已完成的 `clean` 会自动复用，无需重跑）：

**1) 提交 G grid；本步骤用时：旧版参照约 8–25 天，建议按本机 `clean` 实际耗时估算。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=G bash scripts/run_lm_head_h100.sh grid
```

**2) G 结束确认后再提交 T grid；本步骤用时：旧版参照约 25–77 天，当前加速版按本机实测校准。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=T bash scripts/run_lm_head_h100.sh grid
```

**3) T 结束确认后再提交 K grid；本步骤用时：旧版参照约 15–46 天，当前加速版按本机实测校准。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=K bash scripts/run_lm_head_h100.sh grid
```

---

## 中断恢复与常见问题

### 平台中断 / 超时 / 主动暂停后的恢复

确认原任务及其 GPU 子进程已完全退出后，按中断前所处阶段复制下面对应的一块继续运行（已完成候选自动跳过，未完成候选从最近检查点续跑）：

**恢复 `clean-all`（接着跑完剩余 `G → T → K` clean，并在结束时出表）；本步骤用时：初始化预留 5–15 分钟，之后只跑剩余训练（强杀中断可能重算最近未保存的至多约 191 步）。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
LH_PHASE=clean-all bash scripts/run_lm_head_h100.sh resume
```

如果是单个 `grid` 任务中断，只复制对应目标的那一块：

**恢复 G grid；本步骤用时：初始化预留 5–15 分钟，之后只跑 G 剩余部分。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=G LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**恢复 T grid；本步骤用时：初始化预留 5–15 分钟，之后只跑 T 剩余部分。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=T LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**恢复 K grid；本步骤用时：初始化预留 5–15 分钟，之后只跑 K 剩余部分。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=K LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**主动安全暂停当前训练；本步骤用时：请求通常 1 分钟内返回，之后需等待当前步完成保存与 GPU 释放（返回成功不代表 GPU 已立即清空）。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
bash scripts/run_lm_head_h100.sh pause
```

**查看当前进度 / 随时刷新结果汇总；本步骤用时：clean 阶段通常 1 分钟内，完整 grid 约 1–5 分钟。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
bash scripts/run_lm_head_h100.sh status
bash scripts/run_lm_head_h100.sh report
```

### 常见异常处理

| 现象 | 处理方式 |
|---|---|
| `check` 或输入路径报错 | 核对 `.env` 中的 `INPUT_ROOT`、`PYTHON` 与 `SWANLAB_API_KEY`，修正后重跑第 2 步 |
| `triton_ready` 为 `False` | 不影响正确性，但 `T`/`K` 会明显变慢；检查环境中是否已安装 `triton` 及 CUDA 环境后重跑第 2 步 |
| `CUDA OOM` | 先确认 GPU 未被其他进程占用；空闲卡仍 OOM 时保留日志停止排查，勿擅自修改 `MICRO_BATCH=16` |
| `smoke-all` 通过但 SwanLab 网页无曲线 | 检查 `SWANLAB_API_KEY`、项目名或网络；修复后换一个全新 `LH_RUN_ROOT` 重跑第 2 步和第 4 步 |
| 正式训练中 SwanLab 网页断更 | 先跑 `status` 或查看 `$LH_RUN_ROOT/logs/`，训练可能仍在正常推进；切勿在旧任务未退出时重复提交 |
| 退出码 `75` 或任务意外中止 | 等待原任务及所有子进程完全退出，使用上方对应 `resume` 命令恢复（切勿手工删锁或并发跑同目录任务） |

更多日志路径、扩块命令与耗时校准公式见[参考页](LM_HEAD_REFERENCE.md)。
