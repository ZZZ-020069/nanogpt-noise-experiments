# LM Head-only：H100 运行操作手册

在 4 卡 H100 上按以下 5 步顺序执行，即可完成 `clean` 阶段并产出汇总表；**同一组 GPU 上，等前一个任务彻底退出后再提交下一个任务**。

```text
1. 下载与配置  →  2. 环境检查  →  3. 准备 K (prepare-k)  →  4. 联合短测 (smoke-all)  →  5. 自动顺序跑完 clean (clean-all)
```

> `clean-all` 跑完会自动生成汇总表并停止；确认结果后再决定是否逐个启动第 6 步的 `grid`。

## 耗时参考（4 张 H100）

| 步骤 | 说明 | 预估墙钟时间 |
|---|---|---|
| 1–2. 配置与检查 | 克隆代码、配置 `.env`、检查输入与环境 | 各 5–15 分钟 |
| 3. `prepare-k` | 生成并校验 K 统计量（只用第 1 张卡） | 待首轮实测（无固定小时数） |
| 4. `smoke-all` | G / T / K 联合 32 步短测（只用第 1 张卡） | 约 10–25 分钟 |
| 5. `clean-all` | 自动顺序跑完 G → T → K 正式 `clean`（4 卡并行） | **合计约 15–45 小时**（G 6–18h / T 5–15h / K 4–12h） |
| 6. `grid`（可选） | 确认 `clean` 后按 G → T → K 逐目标提交 | 约 8–25 / 7–21 / 5–17 天（各目标单独跑） |

注：上述训练耗时由单卡短测外推（含 0.5–1.5x 波动范围），不含平台排队时间；实际速度以第 4 步 `smoke-all` 输出为准。

---

## 1. 下载代码与配置（只做一次）

在平台任务环境变量中配置 **`SWANLAB_API_KEY`**（后续每个任务都需保留该变量），然后执行：

```bash
set -euo pipefail
cd /data/260010171
git clone --branch design/lm-head-only-20261114 --single-branch \
  https://github.com/ZZZ-020069/nanogpt-noise-experiments.git \
  nanogpt-lm-head-h100-m16-20261116
cd nanogpt-lm-head-h100-m16-20261116
cp -n scripts/lm_head_h100.env.example /data/260010171/lm-head-h100-m16-20261116.env
```

打开 `/data/260010171/lm-head-h100-m16-20261116.env`，按当前平台路径核对或修改以下变量：

| 变量 | 说明 |
|---|---|
| `PYTHON` | 实验环境 Python 路径（要求 Python ≥ 3.11，PyTorch ≥ 2.6 且 < 3，NumPy ≥ 1.26 且 < 3） |
| `INPUT_ROOT` | 已登记的正式输入目录（默认 `/data/260010171/inputs-v1.8`） |
| `LH_RUN_ROOT` | 本次全新输出目录（默认 `/data/260010171/results/lm-head-m16-20261116`） |
| `LH_CURVATURE_ROOT` | K 统计量输出目录（默认 `/data/260010171/results/lm-head-curvature-m16-20261116`） |
| `GPUS` | 本次使用的 GPU 编号（4 卡填 `0,1,2,3`） |
| `SWANLAB_PROJ_NAME` | 保持 `nanogpt-lm-head`；团队账号可另取消注释并填写 `SWANLAB_WORKSPACE` |

> **`K_STATISTICS` 暂时保持注释状态**，第 3 步通过后会自动写入。若修改了存放目录，请同步修改下面命令块中的目录路径；已开始正式训练后请勿再修改代码或 `.env`。

---

## 2. 检查环境与输入

整块复制运行（安装依赖并核对 SwanLab 配置、输入哈希、Triton 状态与运行计划）：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
"$PYTHON" -m pip install "swanlab==0.10.5"
"$PYTHON" -c "import torch, nanogpt_noise.lm_head.kernels as k; print('cuda=', torch.cuda.is_available(), 'triton_ready=', k.triton_ready(torch.device('cuda')) if torch.cuda.is_available() else 'skip_no_cuda')"
bash scripts/run_lm_head_h100.sh check
bash scripts/run_lm_head_h100.sh plan
```

**通过标准**：命令无报错退出；`plan` 输出 `micro: 16`、`accumulations: 32`、`steps: 3815`；若当前节点带 GPU，确认输出 `triton_ready= True`（若为 `False` 不影响结果正确性，但后续会显著变慢，建议排查 Triton / CUDA 环境）。

---

## 3. 生成并校验 K 统计量（只用第 1 张卡）

该命令会自动依次完成拟合与独立校验，并在通过后**自动将生成的 `K-approved.pt` 路径写入 `.env` 中的 `K_STATISTICS`**，最后打印汇总信息：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
bash scripts/run_lm_head_h100.sh prepare-k
```

**通过标准**：命令退出码为 0，终端末尾 `=== PREPARE-K SUMMARY ===` 显示 `fit: passed=True`、`check: passed=True`，且 `.env` 中已激活 `K_STATISTICS=.../K-approved.pt`。

- **若报错退出**：不要强行继续 `K`，先看终端输出与对应目录 `report.json`；若提示 `stability_passed=False`（且其他检查正常），按 `65536` → `131072` 扩样重跑即可：
  ```bash
  LH_TOKENS=65536 bash scripts/run_lm_head_h100.sh prepare-k
  ```
- 如扩样至 `131072` 仍未通过，保留输出与日志停止排查。

---

## 4. 联合短测（`smoke-all`，只用第 1 张卡，约 10–25 分钟）

在第一张卡上连续完成 `G`、`T`、`K` 三个目标的 32 步短测、SwanLab 上传验收与耗时汇总：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
bash scripts/run_lm_head_h100.sh smoke-all
```

**通过标准（缺一不可）**：
1. 命令正常退出，终端末尾 `=== SMOKE-ALL SUMMARY ===` 中 **`G` / `T` / `K` 均为 `passed=True`**，且耗时与显存正常；
2. 打开 SwanLab 网页项目 `nanogpt-lm-head`，确认能看到 **`G`、`T`、`K` 各 4 个优化器（共 12 组）且 `progress/step` 均到达 `32`**。

---

## 5. 自动顺序执行正式 `clean`（4 卡并行，约 15–45 小时）

确认第 4 步两项通过标准均满足后，提交一条命令即可自动顺序跑完 **`G → T → K` 全部正式 `clean`**，期间无需人工干预；全部结束后会自动生成并打印汇总结果表：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
bash scripts/run_lm_head_h100.sh clean-all
```

**完成标准**：
- 命令正常退出，终端末尾 `=== CLEAN-ALL SUMMARY ===` 中 **`G`、`T`、`K` 的 `completed_jobs` 均为 `4`**，并已输出 `$LH_RUN_ROOT/comparison.csv` 的全部 12 行汇总表；
- 将 `$LH_RUN_ROOT/comparison.csv` 导出或发回核对分析，**到此先暂停等待结果确认，不要直接提交第 6 步**。

---

## 6. 完整噪声网格（仅在 `clean` 结果确认后再逐个执行）

`grid` 运行时间长，需按 **`G` → 确认 → `T` → 确认 → `K`** 顺序分别提交；**前一个任务彻底结束并确认后再提交下一个**（会自动跳过同目录下已完成的 `clean`）：

**1) 提交 G grid（约 8–25 天）：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=G bash scripts/run_lm_head_h100.sh grid
```

**2) G 结束确认后再提交 T grid（约 7–21 天）：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=T bash scripts/run_lm_head_h100.sh grid
```

**3) T 结束确认后再提交 K grid（约 5–17 天）：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=K bash scripts/run_lm_head_h100.sh grid
```

---

## 中断恢复与常见问题

### 平台中断 / 超时 / 主动暂停后的恢复

确认原任务进程已完全退出，直接按中断前所处阶段复制对应的一块恢复（已完成部分自动跳过，只跑剩余步数）：

**恢复 `clean-all`（接着跑完剩余 G → T → K clean，并在结束时出表）：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
LH_PHASE=clean-all bash scripts/run_lm_head_h100.sh resume
```

**恢复某个单目标 `grid`（把 `OBJECTIVE` 改为实际中断的目标 `G` / `T` / `K`）：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
OBJECTIVE=G LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**主动安全暂停当前训练：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
bash scripts/run_lm_head_h100.sh pause
```

**查看当前进度 / 随时刷新结果表：**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116
source /data/260010171/lm-head-h100-m16-20261116.env
bash scripts/run_lm_head_h100.sh status
bash scripts/run_lm_head_h100.sh report
```

### 常见异常处理

| 现象 | 处理方式 |
|---|---|
| `check` 或输入路径报错 | 检查 `.env` 中 `INPUT_ROOT`、`PYTHON` 与 `SWANLAB_API_KEY` 配置后重跑第 2 步 |
| `triton_ready` 为 `False` | 不影响正确性但 `T`/`K` 会变慢；检查环境是否安装了 `triton` 及 CUDA 环境后再重跑第 2 步 |
| `CUDA OOM` | 先检查 GPU 是否被其他任务占用；空闲卡仍 OOM 时保留日志停止（勿擅自改小 `MICRO_BATCH=16`） |
| `smoke-all` 通过但 SwanLab 网页无曲线 | 检查 `SWANLAB_API_KEY`、项目名或网络；修复后换一个全新 `LH_RUN_ROOT` 重新跑第 2 步和第 4 步 |
| 正式训练中 SwanLab 网页断更 | 先跑 `status` 或看本地日志，训练通常仍在继续；切勿在旧任务未退出时重复提交 |
| 退出码 `75` 或任务意外中止 | 等待原任务与子进程完全退出，使用上方对应 `resume` 命令恢复（勿手工删锁或并行启动两个同目录任务） |

更多日志字段、扩样细节与后期分析命令见[参考页](LM_HEAD_REFERENCE.md)。
