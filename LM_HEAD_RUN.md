# LM Head-only：H100 启动说明

**最终档位固定为 micro=16。** 用户根据 RTX 实测作出此决定：32 仅快约 1%，峰值 allocated 却由 22.86 GiB 增至 44.38 GiB。G/T/K、smoke、正式训练和曲率测量统一使用 16；单卡累积 32 次，全局 batch=512×1024，每候选 3815 步。

使用 scripts/run_lm_head_h100.sh。它拒绝残留的 MICRO_BATCH=32；不会从 RTX selection.json 读回 32。训练数学和原优化器实现不变，全部 38,633,472 个 LM Head 参数更新，其余参数冻结。主设计第 14.21 节记录最终决定；[RTX 原始实测](docs/experiments/lm-head-only/RTX_SELECTION_RECORD.md)保留历史自动推荐 32。

## 1. 首次准备固定版本与共同环境

下面路径沿用既有 H100 平台的 /data/260010171。若实际挂载位置不同，统一修改代码、环境文件和输出路径。INPUT_ROOT 必须是之前已登记的正式 v1.8 输入，包含 openwebtext_subset、frozen-artifacts、registration.json 和 SHA256SUMS。结果根目录必须全新，不能复用 last-k 或 micro=32 的目录。

首次准备时运行一次：

~~~bash
set -euo pipefail
cd /data/260010171
git clone --branch lm-head-h100-m16-20261115 --single-branch \
  https://github.com/ZZZ-020069/nanogpt-noise-experiments.git \
  nanogpt-lm-head-h100-m16-20261115
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115
test "$(git rev-parse HEAD)" = "$(git rev-parse 'lm-head-h100-m16-20261115^{commit}')"
cp -n scripts/lm_head_h100.env.example /data/260010171/lm-head-h100-m16.env
source /data/260010171/lm-head-h100-m16.env
"$PYTHON" -c 'import sys,torch,numpy; assert sys.version_info >= (3,11); print("torch",torch.__version__,"numpy",numpy.__version__,"CUDA",torch.cuda.is_available())'
bash scripts/run_lm_head_h100.sh check
bash scripts/run_lm_head_h100.sh plan
~~~

环境文件默认如下，按服务器实际路径修改一次；后续每个任务重新 source 同一文件：

~~~bash
export PYTHON=python
export INPUT_ROOT=/data/260010171/inputs-v1.8
export LH_RUN_ROOT=/data/260010171/results/lm-head-m16-20261115
export GPUS=0,1,2,3
export MICRO_BATCH=16
export OBJECTIVE=G
# 正式 K 验收通过后才配置：
# export K_STATISTICS=/data/260010171/results/lm-head-curvature-m16-20261115/init-check-32k/K-approved.pt
~~~

从仓库目录直接运行，复用平台已有 Python≥3.11、PyTorch≥2.6,<3 和 NumPy≥1.26,<3 环境。若用已有虚拟环境，把 PYTHON 改为其绝对路径。不要升级正在承担旧实验的共享环境。check 核对正式输入并绑定新结果根的 micro=16；plan 只列任务和预算，不计算 GPU。

固定 checkout 后，smoke、正式运行和恢复始终用相同代码、环境、输入和输出根。运行期间不 git pull，不改变 micro。

## 2. H100 smoke

先验收 G；这一阶段不需要 K 统计文件。下面是一条完整的独立任务启动命令：

~~~bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115
source /data/260010171/lm-head-h100-m16.env
OBJECTIVE=G bash scripts/run_lm_head_h100.sh smoke
~~~

smoke 在 GPUS 的第一张卡上依次运行四优化器，各中心 LR、32 步；成功后写 smoke/G/passed.json。检查退出码、终态和 performance.json 的耗时/显存。该命令只做 smoke，不启动正式训练。

T 使用同一根目录与同一 micro：将命令中的 OBJECTIVE=G 改为 OBJECTIVE=T。K 必须先完成第 4 节的正式曲率验收，再以 OBJECTIVE=K 运行。每个目标有独立成功凭证。

如果正式 K 已准备好，可一次验收三目标：

~~~bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115
source /data/260010171/lm-head-h100-m16.env
bash scripts/run_lm_head_h100.sh smoke-all
~~~

smoke-all 要求 K_STATISTICS 已指向正式 K-approved.pt。任一目标失败就停止；全部通过也不会自动开始正式训练。RTX 的 benchmark.json 和 K-benchmark.pt 不能作为 H100 正式凭证或 K 训练输入。

## 3. 正式 clean 训练

对应目标的 smoke 成功后，单独提交正式命令。先运行 G：

~~~bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115
source /data/260010171/lm-head-h100-m16.env
OBJECTIVE=G bash scripts/run_lm_head_h100.sh formal
~~~

formal 是 clean 的别名：零噪声节点、四优化器各八个 LR，共 32 条正式候选，每条 3815 步。四张卡各领取一项优化器搜索，单个搜索内八候选顺序执行。formal 不等于完整 35 点噪声面；第 5 节单独提供 grid 命令。

T、K 也在同一根目录运行，分别提交（前提是各自 smoke 已成功）：

~~~bash
# 每次任务先执行本节相同的 cd 与 source
OBJECTIVE=T bash scripts/run_lm_head_h100.sh formal
# K 必须已通过正式曲率验收并配置 K_STATISTICS
OBJECTIVE=K bash scripts/run_lm_head_h100.sh formal
~~~

同一组 GPU 上按目标顺序运行，前一目标退出后再启动下一目标。G/T 不因 K 尚未就绪而被阻塞。运行器会核对目标、输入、训练代码、micro 和 K 哈希对应的 smoke 凭证；旧凭证不能用于改变布局后的运行。

## 4. 正式 Hessian 测量与 K 验收

采用正式 v1.8 输入，在初始化处先拟合、后独立留出。两个阶段分开运行，只有拟合报告通过后才开启留出。每阶段在 GPUS 第一张卡计算，避免与正在运行的训练占用同一张卡。

拟合命令：

~~~bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115
source /data/260010171/lm-head-h100-m16.env
export LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261115/init-fit-32k
bash scripts/run_lm_head_h100.sh curvature-fit
~~~

拟合通过后，留出命令：

~~~bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115
source /data/260010171/lm-head-h100-m16.env
export LH_FIT_DIR=/data/260010171/results/lm-head-curvature-m16-20261115/init-fit-32k
export LH_CURVATURE_DIR=/data/260010171/results/lm-head-curvature-m16-20261115/init-check-32k
bash scripts/run_lm_head_h100.sh curvature-check
~~~

初始化默认每种布局五块：128×32、512×8，块尺寸均为 4096×4096；32768 tokens、8192 前缀稳定性比较、完整词表概率和 FP64 累积保持主设计规定。固定梯度探针使用前 16 个完整训练 batch，评估池各 200 条序列。采样只作用于测量，训练始终更新完整 Head。

拟合或留出失败时退出码为 3；保留报告，不放宽阈值或伪造通过。若按设计扩样，在新目录设置 LH_TOKENS=65536 或 131072；拟合可设 LH_BLOCKS=10，留出保持 5。先冻结拟合设置，再开启对应留出，不用已看过的留出结果反复调参。

只有两阶段通过才生成 init-check-32k/K-approved.pt。此时把它的绝对路径写入共同环境文件的 K_STATISTICS，再执行 K smoke 和 formal。RTX 的经验统计文件不适用。

G clean 完成后的曲率漂移，指定 LH_CHECKPOINT 为赢家 analysis/head-000191.pt、head-001908.pt 或终点包，使用新的 fit/check 目录。后期每布局默认两块，不产生新的 K 训练产物。

## 5. 完整噪声面、恢复与报告

审查 clean 结果后，再明确启动需要的目标的 35 节点噪声面：

~~~bash
set -euo pipefail
cd /data/260010171/nanogpt-lm-head-h100-m16-20261115
source /data/260010171/lm-head-h100-m16.env
OBJECTIVE=G bash scripts/run_lm_head_h100.sh grid
~~~

grid 每目标 140 个搜索、每搜索八候选；同身份完成的 clean 直接复用。T/K 的 grid 分别改变 OBJECTIVE，仍需对应成功 smoke 与 K 前置条件。

查看、暂停和恢复命令如下。每次新的平台任务都先执行相同的 cd 与 source：

~~~bash
bash scripts/run_lm_head_h100.sh status
bash scripts/run_lm_head_h100.sh report
# 有可用交互终端时，可请求在更新边界保存并暂停
bash scripts/run_lm_head_h100.sh pause
# 等原控制器和所有 worker 退出后，选择实际被中断的阶段
OBJECTIVE=G LH_PHASE=clean bash scripts/run_lm_head_h100.sh resume
# 如果原来运行的是 grid，改用这一条
OBJECTIVE=G LH_PHASE=grid bash scripts/run_lm_head_h100.sh resume
~~~

退出码 75 表示暂停。ACP 无运行中交互终端时，使用平台停止，随后重新提交对应 resume 命令；无需在训练中另开终端。检查点约每 191 步保存，可控信号尽力保存最新状态，强制停止可能重算最近检查点之后的步骤。不要同时运行两个同根控制器，不手工删除锁或改 metadata。

## 6. 产物与 RTX 历史记录

| 路径 | 内容 |
|---|---|
| lm_head_run.json | 绑定代码、输入和 micro=16 的正式根身份 |
| smoke/G/passed.json | G 四优化器验收；T/K 各自独立 |
| G/jobs/job_0000/ | 一个优化器、一个节点的八候选与 search.json |
| candidate_*/ | 身份、终态、评估、逐步记录和恢复包 |
| candidate_*/analysis/ | G clean 的中间 Head 快照 |
| retained.json | 搜索完成后保留权重的哈希清单 |
| comparison.json / comparison.csv | 正式 CE 排名与训练目标诊断 |
| 独立曲率目录 | 采样身份、子块与谱、统计和验收报告 |

恢复包保存 Head 和优化器状态并绑定冻结 backbone；分析 Head 快照不能独立续训。每搜索完成后保留 CE 赢家完整包，必要时另保留训练目标赢家 Head。正式单目标完整面约 54 GB 的 winner 恢复包，其他快照、曲率块和临时写入空间另计；不会自动清理旧实验。

RTX 原始测速已完成，无需再跑选档。原始自动推荐 32 保存在 rtx-benchmark-result.json；最终用户决策 16 保存在 h100-micro-decision.json。为复核旧 selection.json，使用测量时的冻结提交 74af8db；当前 --handoff-micro 已停用。H100 正式入口只接受最终 16。
