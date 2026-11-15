# LM Head-only 启动说明

本入口对应 LH-QP-20261114-draft2：冻结同一随机初始化的 backbone，只更新完整 LM Head。新代码使用单卡独立候选的 GPU 队列。训练预算、八候选搜索、35 点噪声网格和 validation CE 选优继承 last-k。完整规则见[设计快照](docs/experiments/lm-head-only/20261114-draft2/EXPERIMENT_DESIGN.md)。

代码提供 G（真实 CE）、T（初始化处逐样本二次展开）、K（固定统计结构二次代理）三个入口，各阶段由命令明确启动。本地 CPU 测试与正式结果分开；目标 GPU 的 BF16、显存和耗时须通过 smoke 实测。

## 1. 准备与核对

在目标 Linux 服务器使用本分支的独立 checkout，安装依赖：

~~~bash
python -m pip install -e .
export PYTHON=python
export INPUT_ROOT=/absolute/path/to/registered-input-bundle
export LH_RUN_ROOT=/absolute/path/to/new-lm-head-results
export GPUS=0,1,2,3
export MICRO_BATCH=16
export OBJECTIVE=G
bash scripts/run_lm_head.sh check
bash scripts/run_lm_head.sh plan
~~~

INPUT_ROOT 必须包含现有、已登记的 openwebtext_subset/、frozen-artifacts/、registration.json 和 SHA256SUMS。直接读取之前的冻结输入，不重新生成初始化或训练窗口。输出目录必须是输入树之外的全新目录，不能使用 last1 输出。

check 验证哈希、输入协议和根目录 micro；plan 只打印任务、micro、累积次数及固定预算，不访问 GPU、不创建产物。正式执行时每张卡处理一个候选，八个 LR 候选顺序推进。

默认 micro=16；可用 MICRO_BATCH=32 或 CLI 的 --micro-batch 32 覆盖。全局 batch 始终为 512 条长 1024 的序列：16 对应每步累积 32 次，32 对应 16 次，均为每步 524288 tokens、每候选 3815 步。噪声仍在全部累积结束后注入一次。

G/T/K 在同一个 LH_RUN_ROOT 中锁定同一个 micro；smoke、clean、grid 和恢复都受检查。report 会逐候选验证 micro，并在 CSV 记录该值。旧根登记缺少 micro 时拒绝自动迁移；改变 micro、代码或输入必须使用全新目录。已运行的实验继续用原冻结 checkout，不在运行期间更新代码。

### RTX 双卡选档：分别测 16 和 32

2026-11-15 已在旧输入测速模式完成两档全部 24 个候选，按规则选中 32。两档总计时相差约 1%，详情见 [实测记录](docs/experiments/lm-head-only/RTX_SELECTION_RECORD.md#最终硬件测速结果)。已有结果可直接交给 H100 接手，下面命令用于复核或另起新测试。

在双 RTX PRO 6000 上完成选档，H100 只验收选中的共同 micro。监督脚本让第一张卡测 16，第二张卡测 32，各自顺序完成 G/T/K、四优化器、32 步，共 24 个候选。用户已允许本轮纯硬件测速使用旧输入，优先使用下面的 benchmark-only 路径。

~~~bash
export PYTHON=/experiments/xuyangzhang/venvs/quadratic/bin/python
export INPUT_ROOT=/experiments/xuyangzhang/nanogpt-inputs
export SELECTION_ROOT=/experiments/xuyangzhang/lm-head-legacy-selection-20261115
export GPUS=0,1
# 检查 GPU，计算经验 K，然后双卡选档；不启动正式训练
bash scripts/run_lm_head_rtx_benchmark.sh
~~~

旧语料、初始化和窗口保持原文件、原哈希，执行仍使用当前完整模型、global batch=512×1024 和 3815 步调度的前 32 步。输出为 benchmark 模式，只写 benchmark.json，不能充当正式 smoke 凭证。K-benchmark.pt 由旧数据前 32768 个固定 eval_train tokens 测得 B、pbar 和 alpha，passed=false；正式加载器拒绝它。H100 可接收推荐 micro，但需要自己的正式输入和正式 K-approved.pt。

如果已有同一正式 v1.8 输入和已验收 K，也可使用原有严格配对测速路径（不加 --benchmark-only）：

~~~bash
export PYTHON=/absolute/path/to/python
export INPUT_ROOT=/absolute/path/to/registered-input-bundle
export SELECTION_ROOT=/absolute/path/to/new-rtx-selection
export K_STATISTICS=/absolute/path/to/approved/K-approved.pt
bash scripts/select_lm_head_micro.sh --input-root "$INPUT_ROOT" \
  --selection-root "$SELECTION_ROOT" --devices 0,1 --statistics "$K_STATISTICS"
~~~

脚本先验证输入与 K，再检查两卡无计算进程、利用率为 0、每卡已占显存不超过 1 GiB。它记录 GPU UUID、软件和代码身份，并使用内核锁防止同目录双开。检测后其他用户仍可能占卡；出现资源冲突应保留失败记录，在空闲时显式重试。

若 K 尚未获得，可先执行下面的 G/T 部分；退出码 3 和 selected_micro=null 表示完整选档尚未完成。K 的测量仍使用第 3 节的真实拟合与留出流程；曲率测量应与双卡选档错开，避免同时占卡。得到 K-approved.pt 后执行上面的完整命令，已完成 G/T 经核验后自动复用。

~~~bash
bash scripts/select_lm_head_micro.sh --input-root "$INPUT_ROOT" \
  --selection-root "$SELECTION_ROOT" --devices 0,1 --stage gt
# 完成/停止后，只汇总已有证据（无 GPU 计算）
bash scripts/select_lm_head_micro.sh --selection-root "$SELECTION_ROOT" --report-only
~~~

每个 micro/目标组保留 attempt-0000 等独立目录及 console.log、execution.json。重新运行不会自动重试失败组；需要重试时在原命令加 --retry-failed。重试从新目录第 0 步开始，避免用恢复后的部分时间比较吞吐。Ctrl-C/TERM 停止后续调度并通知正在执行的 smoke；超过 30 秒仍未退出的子进程会被终止，已有产物保留。

selection.json 汇总全部状态、四优化器分项时间、峰值 allocated 和含启动开销的外层耗时。只有六组均为完整成功或明确 CUDA OOM，才判定测试完成；输入/进程错误、数值失败或 K 缺失均阻止定档。可选档位须三目标全通过，且每次 PyTorch 峰值 allocated 不超过 RTX 总显存的 90%。在合格档位中比较十二次完整 smoke 的 wall_seconds 总和，取较短者，精确相同取 16。该显存规则提供工程余量，实际设备空闲量和 H100 容量仍需实测。

T 的完整词表中间张量随 micro 增加；K 的固定二次项仍在每个 micro-batch 计算。选档数据只用于运行布局，不进入优化器正式排名。selection.json 与 K 产物一并交接到 H100；保持相同代码、输入和 K 哈希，不复制 RTX 的 smoke 凭证。H100 不再比较两个档位。

## 2. 先验收，再启动 G clean

在 H100 上，从已完成的 selection.json 读取共同 micro，在全新正式根目录先通过 G/T/K 三目标的 smoke。任一失败都先停止；只有验收成功，才显式执行 clean：

~~~bash
export LH_RUN_ROOT=/absolute/path/to/new-formal-lm-head-results
export SELECTION_ROOT=/absolute/path/to/transferred-selection-directory
export K_STATISTICS=/absolute/path/to/transferred/K-approved.pt
MICRO_BATCH="$(bash scripts/select_lm_head_micro.sh --selection-root "$SELECTION_ROOT" \
  --input-root "$INPUT_ROOT" --statistics "$K_STATISTICS" --handoff-micro)" || exit $?
export MICRO_BATCH
OBJECTIVE=G bash scripts/run_lm_head.sh smoke && \
OBJECTIVE=T bash scripts/run_lm_head.sh smoke && \
OBJECTIVE=K bash scripts/run_lm_head.sh smoke || exit $?
# 三目标验收成功后，单独执行正式训练命令
OBJECTIVE=G bash scripts/run_lm_head.sh clean
~~~

smoke 在 GPUS 中第一张卡上顺序运行四优化器，各中心 LR、32 步。成功记录绑定输入、代码、目标及 micro batch；clean 必须找到相同身份的成功记录，才会启动四个八候选搜索，每候选 3815 步。smoke 的分数不进入排名。

每个 smoke 目录的 performance.json 记录同步计时与峰值显存，包括冻结特征、前向、反向、优化器、评估和 checkpoint。特征时间含训练与评估，前向时间包含特征生成，不能将重叠项直接相加。计时只作用于 smoke，正式训练不插入这些同步点。

建议在独立 tmux 会话中运行长任务。查看与暂停：

~~~bash
bash scripts/run_lm_head.sh status
bash scripts/run_lm_head.sh pause
# 等控制器报告暂停、所有 worker 退出后恢复
LH_PHASE=clean bash scripts/run_lm_head.sh resume
~~~

退出码 75 表示已暂停。暂停请求持久保存，worker 在更新边界写 checkpoint；OOM、进程中断等也续跑同一候选。硬中断可能重放最近 checkpoint 后的步骤。不要删除锁或另开同目录 launcher；Linux 内核锁防止双写。暂停命令成功只表示请求已写入，需确认 worker 退出和显存释放。

## 3. 测量 Hessian 与验收 K

先在初始化处测拟合池，随后显式开启留出池：

~~~bash
export LH_CURVATURE_DIR=/absolute/path/to/lm-head-curvature/init-fit-32k
bash scripts/run_lm_head.sh curvature-fit
export LH_FIT_DIR="$LH_CURVATURE_DIR"
export LH_CURVATURE_DIR=/absolute/path/to/lm-head-curvature/init-check-32k
bash scripts/run_lm_head.sh curvature-check
~~~

每种布局默认五块：128×32、512×8，子块均为 4096×4096。数据前缀为 8192→32768 tokens；统计累积、子块和分解用 FP64，概率始终覆盖完整词表。参数采样只用于测量，训练参数始终是完整 Head。测量还保存梯度奇异值、固定 batch 波动、Taylor 误差、校正谱矩和 K 近似误差。

离线入口也默认 micro=16，接收相同 MICRO_BATCH，记录在自身的曲率身份中。正式分析沿用选定的共同 micro；不要用不同 micro 的计算结果宣称逐位相同。200 条固定评估序列在 16/32 下均有 8 条尾块，评价和梯度统计按实际样本数加权。tiny CPU 测试须显式传入 --micro-batch 2 或 4。

拟合阶段使用训练步骤 0–15 的完整 batch 检查平均梯度；留出阶段用固定 train/val 评估池复核零空间兼容性。这些离线计算有额外成本。拟合或留出检查失败时退出码为 3，报告仍保留。失败不能通过更改阈值或挑块绕过。

如按设计需要扩大拟合样本，在新的目录设置 LH_TOKENS=65536 或 131072；拟合块可设 LH_BLOCKS=10，留出固定为 5。先冻结拟合规模，再开启相同 token 数的留出池。已查看的留出结果不能再用于调参。

通过初始化的拟合与留出验收，才生成 K-approved.pt。训练后的漂移测量指定 LH_CHECKPOINT 为 G clean 赢家保留的 analysis/head-000191.pt、head-001908.pt 或终点包；每个参考点使用独立 fit/check 目录和对应 fit 来源。后期漂移不产生新的 K 参数，也不改变初始化代理。

指定 checkpoint 后默认使用每种布局最先抽到的两块拟合、两块留出，仍与初始化的同名块配对。后期报告的通过状态只表示数值检查完成；结构误差作为漂移保留，不否决初始化的 K。后期重新测得的统计不允许作为新的 K 训练输入。

## 4. T、K 与后续噪声面

~~~bash
OBJECTIVE=T bash scripts/run_lm_head.sh smoke
OBJECTIVE=T bash scripts/run_lm_head.sh clean
export K_STATISTICS=/absolute/path/to/lm-head-curvature/init-check-32k/K-approved.pt
OBJECTIVE=K bash scripts/run_lm_head.sh check
OBJECTIVE=K bash scripts/run_lm_head.sh smoke
OBJECTIVE=K bash scripts/run_lm_head.sh clean
bash scripts/run_lm_head.sh report
~~~

确认 clean 比较后，再对需要继续的目标显式执行：

~~~bash
OBJECTIVE=G bash scripts/run_lm_head.sh grid
# 恢复 grid 使用 OBJECTIVE=G LH_PHASE=grid ... resume
~~~

grid 包含 35 个节点；同身份完成的 clean 直接复用。T、K 分别指定 OBJECTIVE，运行规则相同。脚本不会自动从 clean 转入 grid，也不包含后续多噪声种子扩展。

## 5. 产物位置

| 路径 | 内容 |
|---|---|
| lm_head_run.json | 输出根目录的输入、代码、micro 与实验身份 |
| smoke/G/passed.json | 四优化器硬件验收记录；T、K 各自独立 |
| G/jobs/job_0000/ | 一个节点、一个优化器的八候选及 search.json |
| candidate_*/ | metadata、终态、评估曲线、压缩逐步日志 |
| candidate_*/analysis/ | G clean 的中间 Head 快照 |
| retained.json | 完成搜索后保留权重的 SHA-256 清单 |
| G/summary.json | 控制器退出时的已完成搜索汇总 |
| comparison.json、comparison.csv | report 命令生成的 CE 排名、配对关系与候选集内训练目标诊断 |
| 曲率目录 | 窗口/参数身份、特征缓存、FP64 主块、report.json、fixed-gradients.json |

恢复包只保存 Head 与优化器状态，并绑定冻结 backbone 身份。每搜索只保留 CE winner 的完整包；若训练目标 winner 不同，另留其 Head。分析快照没有优化器状态，不能独立续训。全失败、未完成、平局和未闭合 LR 边界均保留标签，不以虚构大 loss 排名。

正式运行前保留足够空间：每个目标完整面约 54 GB 的 winner 恢复包，其他 Head、曲率块、日志和原子写入空间另计，详见设计成本表。这里没有自动清理旧实验的功能。
