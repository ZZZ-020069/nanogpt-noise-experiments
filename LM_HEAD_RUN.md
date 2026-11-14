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
export MICRO_BATCH=4
export OBJECTIVE=G
bash scripts/run_lm_head.sh check
bash scripts/run_lm_head.sh plan
~~~

INPUT_ROOT 必须包含现有、已登记的 openwebtext_subset/、frozen-artifacts/、registration.json 和 SHA256SUMS。直接读取之前的冻结输入，不重新生成初始化或训练窗口。输出目录必须是输入树之外的全新目录，不能使用 last1 输出。

check 验证哈希与输入协议；plan 只打印任务，不访问 GPU、不创建产物。正式执行时每张卡处理一个候选，八个 LR 候选顺序推进。代码身份、输入或 micro batch 改变会拒绝续跑；不要在运行期间更新 checkout。

## 2. 先验收，再启动 G clean

~~~bash
bash scripts/run_lm_head.sh smoke
bash scripts/run_lm_head.sh clean
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
| lm_head_run.json | 输出根目录的输入、代码与实验身份 |
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
