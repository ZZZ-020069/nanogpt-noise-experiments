# LM Head-only：H100 运行操作手册

在 4 卡 H100 上按以下 5 步顺序执行，即可完成 `clean` 阶段并产出汇总表；**同一组 GPU 上，等前一个任务彻底退出后再提交下一个任务**。

```text
1. 下载与配置  →  2. 环境检查  →  3. 准备 K (prepare-k)  →  4. 联合短测 (smoke-all)  →  5. 自动顺序跑完 clean (clean-all)
```

> `clean-all` 跑完会自动生成汇总表并停止；确认结果后再决定是否逐个启动第 6 步的 `grid`。

本次修复使用全新的 `20261116-r1` 代码、环境和结果目录。已有任务继续使用原冻结版本和原环境恢复；不要在运行目录 `git pull`，也不要将旧 K 产物或 smoke 凭证搬入新目录。

## 耗时参考（4 张 H100）

| 步骤 | 说明 | 预估墙钟时间 |
|---|---|---|
| 1–2. 配置与检查 | 克隆代码、配置 `.env`、检查输入与环境 | 各 5–15 分钟 |
| 3. `prepare-k` | 生成并校验 K 统计量（只用第 1 张卡） | 待首轮实测（无固定小时数） |
| 4. `smoke-all` | G / T / K 联合 32 步短测（只用第 1 张卡） | 约 10–25 分钟 |
| 5. `clean-all` | 顺序 G → T → K，每个目标四卡并行 | 旧实现排期参照合计 35–104 小时（G 6–18 / T 18–54 / K 11–32 小时）；加速版待实测 |
| 6. `grid`（可选） | clean 后其余 34 节点，逐目标提交 | 旧实现参照 G 8–25 / T 25–77 / K 15–46 天；加速版待实测 |
| 暂停、恢复、查进度 | 见下方对应命令 | 暂停请求与查询通常 1 分钟内；释放需等完整步和保存；恢复预留 5–15 分钟初始化，再加剩余训练时间 |

训练排期沿用主设计 14.24 和[计算清单](docs/experiments/lm-head-only/runtime-estimates.json)：旧实现 RTX 短测外推，再取 0.5–1.5 倍情景；这不是 H100 实测或完成时限。新 T/K 加速版尚无同口径 H100 外推清单，先用 smoke 和首段正式日志重新估算，不能承诺旧参照就是上限。不含排队、K 测量、人工审查和故障重跑。

---

## 1. 下载代码与配置（只做一次）

**本步骤用时：预留 5–15 分钟，取决于网络和已有环境。**

在平台任务环境变量中配置 **`SWANLAB_API_KEY`**（后续每个任务都需保留该变量），然后执行：

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

打开 `/data/260010171/lm-head-h100-m16-20261116-r1.env`，按当前平台路径核对或修改以下变量：

| 变量 | 说明 |
|---|---|
| `PYTHON` | 实验环境 Python 路径（要求 Python ≥ 3.11，PyTorch ≥ 2.6 且 < 3，NumPy ≥ 1.26 且 < 3） |
| `INPUT_ROOT` | 已登记的正式输入目录（默认 `/data/260010171/inputs-v1.8`） |
| `LH_RUN_ROOT` | 本次全新输出目录（默认 `/data/260010171/results/lm-head-m16-20261116-r1`） |
| `LH_CURVATURE_ROOT` | K 统计量输出目录（默认 `/data/260010171/results/lm-head-curvature-m16-20261116-r1`） |
| `GPUS` | 本次使用的 GPU 编号（4 卡填 `0,1,2,3`） |
| `SWANLAB_PROJ_NAME` | 保持 `nanogpt-lm-head`；团队账号可另取消注释并填写 `SWANLAB_WORKSPACE` |

> **`K_STATISTICS` 暂时保持注释状态**，第 3 步通过后会自动写入。若修改了存放目录，请同步修改下面命令块中的目录路径；已开始正式训练后请勿再修改代码或 `.env`。

---

## 2. 检查环境与输入

**本步骤用时：预留 5–15 分钟；首次依赖安装可能更久。**

整块复制运行（安装依赖并核对 SwanLab 配置、输入哈希、Triton 状态与运行计划）：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
"$PYTHON" -m pip install "swanlab==0.10.5"
"$PYTHON" -c "import torch, nanogpt_noise.lm_head.kernels as k; print('cuda=', torch.cuda.is_available(), 'triton_ready=', k.triton_ready(torch.device('cuda')) if torch.cuda.is_available() else 'skip_no_cuda')"
bash scripts/run_lm_head_h100.sh check
bash scripts/run_lm_head_h100.sh plan
```

**通过标准**：命令无报错退出；`plan` 输出 `micro: 16`、`accumulations: 32`、`steps: 3815`；若当前节点带 GPU，确认输出 `triton_ready= True`（若为 `False` 不影响结果正确性，但后续会显著变慢，建议排查 Triton / CUDA 环境）。

---

## 3. 生成并校验 K 统计量（只用第 1 张卡）

**本步骤用时：尚无正式 Hessian 同口径实测，暂无可靠小时数；按日志进度校准。**

先在拟合池检查数据稳定性和两种布局的块间二阶谱矩变异系数。全部门槛通过后，程序在 `fit.pt` 中冻结规模及身份，再自动执行独立留出；任何拟合失败都先停止，不读取留出数据。留出通过后，自动把 `K-approved.pt` 路径写入尚未绑定其他 K 的 `.env`，最后打印汇总。

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
LH_TOKENS=32768 LH_BLOCKS=5 bash scripts/run_lm_head_h100.sh prepare-k
```

**通过标准**：命令退出码为 0，终端末尾 `=== PREPARE-K SUMMARY ===` 显示 `fit: passed=True`、`check: passed=True`，且 `.env` 中已激活 `K_STATISTICS=.../K-approved.pt`。

**失败后按报告处理，不能跳过门槛：**

- 数值、结构、梯度兼容性或固定梯度检查失败：保留报告，停止排查，不把它们解释成样本不足。
- 只有拟合数据稳定性失败时，按下方两档依次扩样；`fit_scale.action=expand_tokens` 是规模建议，还需核对其他门槛。
- 数据稳定但 `fit_scale.action=expand_blocks`：按[参考页的完整命令](LM_HEAD_REFERENCE.md#拟合扩块)在相同 token 数上改用每布局 10 块。留出仍是每布局 5 块。
- `review_undefined_moments`、10 块后仍 `unresolved_at_block_limit`，或 131072 tokens 仍不稳定：保留未测稳结果，停止 K，不继续增大规模。
- **留出失败时停止。** 不根据留出结果再扩样、扩块或改代理，直到同一份留出通过。

**扩到 65536 tokens 用时：待实测；这是一次新的拟合，拟合通过才会继续留出。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
LH_TOKENS=65536 LH_BLOCKS=5 bash scripts/run_lm_head_h100.sh prepare-k
```

**65536 仍仅数据不稳定时，扩到 131072 tokens 用时：待实测；这是本轮 token 上限。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
LH_TOKENS=131072 LH_BLOCKS=5 bash scripts/run_lm_head_h100.sh prepare-k
```

默认报告位于 `$LH_CURVATURE_ROOT/init-fit-<tokens>-b<块数>/report.json`；留出位于 `init-check-<tokens>-fit-b<拟合块数>/`。不同规模自动分目录。完整已结束的同设置测量可重跑同命令读取；离线曲率未实现训练式逐步续算，未完成测量会重新计算。

---

## 4. 联合短测（`smoke-all`，只用第 1 张卡，约 10–25 分钟）

**本步骤用时：预留 10–25 分钟，另需人工查看 SwanLab 网页。**

在第一张卡上连续完成 `G`、`T`、`K` 三个目标的 32 步短测与耗时汇总；云端是否收到指标须在命令结束后人工确认：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
bash scripts/run_lm_head_h100.sh smoke-all
```

**通过标准（缺一不可）**：
1. 命令正常退出，终端末尾 `=== SMOKE-ALL SUMMARY ===` 中 **`G` / `T` / `K` 均为 `passed=True`**，且耗时与显存正常；
2. 打开 SwanLab 网页项目 `nanogpt-lm-head`，确认能看到 **`G`、`T`、`K` 各 4 个优化器（共 12 组）且 `progress/step` 均到达 `32`**。

---

## 5. 自动顺序执行正式 `clean`（4 卡并行）

**本步骤用时：旧实现排期参照合计 35–104 小时；新 T/K 加速版需根据本机 smoke 与正式吞吐更新。**

确认第 4 步两项通过标准均满足后，提交一条命令即可自动顺序跑完 **`G → T → K` 全部正式 `clean`**，期间无需人工干预；全部结束后会自动生成并打印汇总结果表：

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
bash scripts/run_lm_head_h100.sh clean-all
```

**完成标准**：
- 命令正常退出，终端末尾 `=== CLEAN-ALL SUMMARY ===` 中 **`G`、`T`、`K` 的 `completed_jobs` 均为 `4`**，并已输出 `$LH_RUN_ROOT/comparison.csv` 的全部 12 行汇总表；
- 将 `$LH_RUN_ROOT/comparison.csv` 导出或发回核对分析，**到此先暂停等待结果确认，不要直接提交第 6 步**。

---

## 6. 完整噪声网格（仅在 `clean` 结果确认后再逐个执行）

`grid` 运行时间长，需按 **`G` → 确认 → `T` → 确认 → `K`** 顺序分别提交；**前一个任务彻底结束并确认后再提交下一个**（会自动跳过同目录下已完成的 `clean`）：

**1) 提交 G grid；用时：旧实现参照约 8–25 天，优先按本机 clean 实测更新。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=G bash scripts/run_lm_head_h100.sh grid
```

**2) G 结束确认后再提交 T grid；用时：旧实现参照约 25–77 天，加速版待校准。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=T bash scripts/run_lm_head_h100.sh grid
```

**3) T 结束确认后再提交 K grid；用时：旧实现参照约 15–46 天，加速版待校准。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=K bash scripts/run_lm_head_h100.sh grid
```

---

## 中断恢复与常见问题

### 平台中断 / 超时 / 主动暂停后的恢复

确认原任务进程已完全退出，直接按中断前所处阶段复制对应的一块恢复（已完成部分自动跳过，只跑剩余步数）：

**恢复 `clean-all`（接着跑完剩余 G → T → K clean，并在结束时出表）：**

**用时：预留 5–15 分钟初始化，再加未完成的训练；强制中断可能重算最近未持久保存的至多约 191 步。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
LH_PHASE=clean-all bash scripts/run_lm_head_h100.sh resume
```

仅复制实际中断目标对应的一块。

**恢复 G grid；用时：预留 5–15 分钟初始化，再加 G 剩余训练。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=G LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**恢复 T grid；用时：预留 5–15 分钟初始化，再加 T 剩余训练。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=T LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**恢复 K grid；用时：预留 5–15 分钟初始化，再加 K 剩余训练。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
OBJECTIVE=K LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
```

**主动安全暂停当前训练：**

**用时：请求通常 1 分钟内返回；真正释放 GPU 需等待完整 optimizer step、保存和全部 worker 退出。返回成功不代表已经释放。**

```bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261116-r1
source /data/260010171/lm-head-h100-m16-20261116-r1.env
bash scripts/run_lm_head_h100.sh pause
```

**查看当前进度 / 随时刷新结果表：**

**用时：小规模通常 1 分钟内；完整网格预留 1–5 分钟，取决于日志和存储速度。**

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
| `check` 或输入路径报错 | 检查 `.env` 中 `INPUT_ROOT`、`PYTHON` 与 `SWANLAB_API_KEY` 配置后重跑第 2 步 |
| `triton_ready` 为 `False` | 不影响正确性但 `T`/`K` 会变慢；检查环境是否安装了 `triton` 及 CUDA 环境后再重跑第 2 步 |
| `CUDA OOM` | 先检查 GPU 是否被其他任务占用；空闲卡仍 OOM 时保留日志停止（勿擅自改小 `MICRO_BATCH=16`） |
| `smoke-all` 通过但 SwanLab 网页无曲线 | 检查 `SWANLAB_API_KEY`、项目名或网络；修复后换一个全新 `LH_RUN_ROOT` 重新跑第 2 步和第 4 步 |
| 正式训练中 SwanLab 网页断更 | 先跑 `status` 或看本地日志，训练通常仍在继续；切勿在旧任务未退出时重复提交 |
| 退出码 `75` 或任务意外中止 | 等待原任务与子进程完全退出，使用上方对应 `resume` 命令恢复（勿手工删锁或并行启动两个同目录任务） |

更多日志字段、扩样细节与后期分析命令见[参考页](LM_HEAD_REFERENCE.md)。
