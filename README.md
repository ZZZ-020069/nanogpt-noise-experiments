# NanoGPT 噪声实验：四卡运行指南

本文面向接手运行实验的人，按 **Linux 服务器、单机 4 张 H100** 说明操作。四张卡通过 DDP 共同训练一个候选，不是每张卡分别运行一个实验。当前仅实现 NanoGPT，二次函数不在本仓库的运行范围内。

**第一次运行请依次完成第 1–5 步，验收通过后再进入第 6 步。** 所有命令均在服务器的 Bash 中执行，不是在本地 Windows PowerShell 中执行。

只有 `sbatch`、没有交互式 GPU 会话时：先做第1步、第2.1节、第3步及第4步的文件校验/CPU测试，再按[Slurm 附录](#slurm)提交验收。第2.2节和第4步的 GPU 检查由四卡脚本在计算节点内自动完成，不要求登录节点能看到 GPU。

> 验证状态：26项单元测试及 Linux 两进程/四进程 CPU 分布式测试已通过，见[验证记录](docs/VALIDATION.md)。H100 的 BF16、NCCL 通信、显存、速度及恢复仍需实机验收；CPU 测试通过不代表已完成 GPU 验收。当前没有正式实验结果。

## 操作目录

1. [获取代码](#step-1)
2. [准备 Python 和 GPU 环境](#step-2)
3. [放置数据、初始权重并设置路径](#step-3)
4. [检查文件和四卡配置](#step-4)
5. [运行短程验收](#step-5)
6. [选择任务并开始正式训练](#step-6)
7. [查看进度、停止与恢复](#step-7)
8. [汇总结果并交回文件](#step-8)

附录：[Slurm 提交](#slurm) · [常见问题](#faq) · [维护者说明](#maintainers)

<a id="step-1"></a>
## 第 1 步：获取代码

首次下载：

```bash
git clone https://github.com/ZZZ-020069/nanogpt-noise-experiments.git
cd nanogpt-noise-experiments
```

如果已经下载过，只在**开始新一轮实验前、没有训练进程运行时**更新：

```bash
cd /实际的仓库路径/nanogpt-noise-experiments
git pull --ff-only
```

仓库若为私有，需要先接受协作者邀请，并使用有权限的 GitHub 账号认证。不要向其他人发送密码或访问令牌。

**检查点：** 当前目录下应有 `README.md`、`nanogpt_noise/` 和 `scripts/run_4gpu.sh`。后文命令默认都在这个仓库根目录执行。开始正式训练后固定代码版本，不要一边训练一边 `git pull`。

<a id="step-2"></a>
## 第 2 步：准备 Python 和 GPU 环境

### 2.1 激活环境并安装项目依赖

优先使用服务器已有的、包含 **Python ≥3.11 和 CUDA 版 PyTorch 2.6–2.x** 的环境。若使用 Conda，将下面的环境名换成实际名称；环境已经激活则跳过第一行。

```bash
conda activate 实际环境名
python --version
python -m pip install -e '.[plots]'
```

没有合适环境时，可先创建独立环境，按服务器驱动选择 [PyTorch 官方安装命令](https://pytorch.org/get-started/locally/)，安装符合上述版本范围的 CUDA wheel，再安装本项目。**不要把 `requirements-cpu-tested.txt` 当成 H100 安装配置。** 接手运行不需要重新下载或分词数据，因此此处不安装 `prepare` 依赖。

### 2.2 确认拿到了四张 GPU

先通过学校调度系统申请单机四卡，或使用已经分配给你的四卡服务器。不要在仅供登录的节点启动训练。需要 `sbatch` 的服务器参见[Slurm 附录](#slurm)。

在获得 GPU 的环境中检查：

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('NCCL:', torch.distributed.is_nccl_available())"
```

**检查点：** `available: True`、`GPU count: 4`、`NCCL: True`，且 CUDA 版本不是 `None`。

如果获准使用的服务器有更多可见 GPU，可以选择其中已分配的四张，例如 `export CUDA_VISIBLE_DEVICES=0,1,2,3`。编号按实际分配填写；**Slurm 已设置此变量时，保留调度器设置。** 脚本只使用可见 GPU，不会替你申请资源。

<a id="step-3"></a>
## 第 3 步：放置数据、初始权重并设置路径

### 3.1 从项目提供者处接收两组文件

`git clone` **只下载代码，不包含以下大文件**。从项目提供者分享的网盘等渠道下载、解压到服务器，保留文件内容和名称：

```text
数据目录/                      ← DATA_DIR 指向这里
├── train.bin                  # 2,000,000,000 字节
├── val.bin                    #    20,000,000 字节
└── meta.json

公共产物目录/                  ← ARTIFACT_DIR 指向这里
├── initial.pt                 # 统一的未训练初始权重
├── windows.pt                 # 固定训练/评估窗口索引
└── identity.json              # 文件哈希与身份信息

结果目录/                      ← RUN_DIR 指向这里，由程序写入
```

公共产物不是训练后的模型。所有实验共用同一套文件；如果只收到了 `train.bin` 和 `val.bin`，仍然缺文件，需补齐后继续。

### 3.2 设置三个目录

**下面三行都需要换成服务器上的真实绝对路径。** 指向包含文件的目录，不是压缩包，也不是单个 `.bin` 或 `.pt` 文件。

```bash
export DATA_DIR="/实际路径/openwebtext_subset/data"
export ARTIFACT_DIR="/实际路径/protocol-v1.4"
export RUN_DIR="/实际路径/results/nanogpt-v1.4"

mkdir -p "$RUN_DIR"
ls -lh "$DATA_DIR/train.bin" "$DATA_DIR/val.bin" "$DATA_DIR/meta.json"
ls -lh "$ARTIFACT_DIR/initial.pt" "$ARTIFACT_DIR/windows.pt" "$ARTIFACT_DIR/identity.json"
df -h "$RUN_DIR"
```

**检查点：** 六个输入文件都存在，结果目录可写、空间充足。full Adam/AdamW 每个候选的 latest/best/final 合计约4.5 GB，八个候选约36 GB，还需预留临时写入和日志空间；不能只按一个模型估算整个网格的存储量。

新开终端或调度任务时，需要重新激活同一 Python 环境并设置这三个变量。恢复训练必须指向原结果目录。

<a id="step-4"></a>
## 第 4 步：检查文件和四卡配置

先核对文件哈希；这个命令不训练，也不重新生成权重：

```bash
python -m nanogpt_noise.data \
  --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" --verify
```

**检查点：** 无报错并输出产物 ID。本项目已提供产物的 ID 为：

```text
9cabd9c54c6d717354f8f6750715f9280c5502f0a85a9032806b9c7956cbd76e
```

随后在四卡环境中执行启动检查：

```bash
python -m nanogpt_noise.preflight \
  --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" --micro-batch 4
```

**检查点：** 输出四张 GPU 的名称，以及 `world: 4`、`micro: 4`、`accumulation: 32`、`global_batch: 512`。程序还会逐卡检查 BF16 和验证文件身份。报错时先按[常见问题](#faq)排查，不继续训练。

再运行实现测试：

```bash
python -m unittest discover -s tests -v
```

应看到 `Ran 26 tests` 和 `OK`。这一步使用小模型/合成数据，不占用正式训练预算，也不能替代下一步的四卡验收。

<a id="step-5"></a>
## 第 5 步：先运行短程验收，不启动正式网格

### 5.1 运行第一个四卡 smoke test

`smoke` 是短程连通性和实现检查：使用完整124M模型、正式全局 batch 和学习率调度，但只跑8步，不参与正式结果。

首次运行这条 smoke 时，安排一次恢复验收：看到至少完成第1步、尚未到第8步时，按第7.2节停止并重启原命令，然后让它跑完。不要等所有 smoke 完成后再尝试中断；Slurm 后台作业的停止方法见附录 A。

```bash
bash scripts/run_4gpu.sh smoke
```

默认配置为 `full / Muon / 噪声节点10`，峰值 LR 使用基准值。程序先重复启动检查，然后输出 `step=.../8` 和验证 loss。第一次还会读取权重、进行 step=0 评估和写 checkpoint，可能不会立即出现训练步日志。

结束后查看状态：

```bash
python -m json.tool "$RUN_DIR/smoke/4gpu/full-muon-node10/status.json"
```

**成功标志：** `status` 为 `completed`、`mode` 为 `smoke`、`step` 为 `8`，且命令没有异常退出。`numerical_failure` 或 `infrastructure_failure` 都不算验收通过。首步 `lr=0` 是预热设计，不是故障。

### 5.2 补查其他优化器和训练范围

第一个 smoke 通过后，再逐条运行下列配置；每条都应检查对应目录的 `status.json`：

```bash
bash scripts/run_4gpu.sh smoke full adamw 0
bash scripts/run_4gpu.sh smoke last1 adam 10
bash scripts/run_4gpu.sh smoke last12 sgd 10
```

这组检查覆盖四种优化器、full/last1/last12、clean 和非零噪声。结果位于 `$RUN_DIR/smoke/4gpu/` 下的 `full-adamw-node0/`、`last1-adam-node10/`、`last12-sgd-node10/`。

首次上机还需观察显存、每步耗时和写盘耗时，确认第一个 smoke 的恢复检查已完成。**重复运行已完成的同一条 smoke 会直接返回已有结果，不会重测 GPU。** 如错过中断时机或需重测，使用新的结果根目录保存验收输出，不要删除正式 checkpoint；测完后再将 RUN_DIR 设回本轮正式结果目录。

验收通过后，保存本轮软件环境和代码版本：

```bash
python -m pip freeze > "$RUN_DIR/environment-lock.txt"
git rev-parse HEAD > "$RUN_DIR/code-commit.txt"
```

这两份文件用于记录，不会自动阻止环境变化；正式运行及恢复时仍须保持环境一致。不要在另一套环境中覆盖已有记录。

<a id="step-6"></a>
## 第 6 步：选择任务并开始正式训练

### 6.1 先理解任务编号

一个 `job` = **一种可训练范围（scope）× 一种优化器 × 一个噪声节点**。一个 job 内顺序测试八个学习率；每个正常候选完整运行4960步、约2.6B tokens。它不是8步 smoke，也不是一次命令完成所有噪声节点。

生成完整任务表只写文件，不启动训练：

```bash
python -m nanogpt_noise.jobs --output "$RUN_DIR/jobs.json"
```

| scope | job 编号范围 | 可训练参数 |
|---|---:|---|
| `full` | 0–139 | 全部参数 |
| `last1` | 140–279 | 最后1个 block |
| `last2` | 280–419 | 最后2个 block |
| `last3` | 420–559 | 最后3个 block |
| `last6` | 560–699 | 最后6个 block |
| `last12` | 700–839 | 全部12个 block，仍冻结 block 外参数 |

每个 scope 内依次是 AdamW、Adam、SGD-Momentum、Muon，各占35个编号。例如 full 的 AdamW 是0–34，Adam 是35–69，SGD 是70–104，Muon 是105–139。`smoke ... 10` 中的10是该优化器内的噪声节点号；`node 10` 中的10则是整个任务表的 job 编号，二者不要混淆。

### 6.2 先查看，再启动一个 job

以 **job 0：full / AdamW / clean 节点** 为例。先查看配置：

```bash
python -m nanogpt_noise.node --job-index 0 \
  --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" \
  --output "$RUN_DIR" --dry-run
```

确认输出包含 `scope: full`、`optimizer: adamw`、`node: 0`、`sigma1: 0.0`、`sigma2: 0.0` 后启动：

```bash
bash scripts/run_4gpu.sh node 0
```

程序自动搜索八个学习率，不需要手动填 LR。四卡默认每卡 micro-batch=4、累积32次，每步全局512条序列；不要为加速改掉正式预算、噪声定义或启用梯度裁剪。

输出放在 `$RUN_DIR/job_0000/`。更换任务时按任务表修改末尾编号；如运行 job 105，则执行 `bash scripts/run_4gpu.sh node 105`。

**长时间训练不要依赖裸 SSH 会话。** 已分配的独立服务器可使用 tmux：先执行 `tmux new -s nanogpt`，在新会话中进入仓库、激活环境、设置三个路径，再运行命令；按 `Ctrl+B` 后按 `D` 脱离，训练继续，之后用 `tmux attach -t nanogpt` 返回。学校集群按[Slurm 附录](#slurm)提交。

### 6.3 确认预算后再批量运行

先确认单个 job 的运行、恢复和存储正常，再与项目负责人确认范围。下面**仅运行 full / AdamW 的35个节点**，顺序执行，遇到进程报错立即停止：

```bash
for job in $(seq 0 34); do
  bash scripts/run_4gpu.sh node "$job" || break
done
```

full 四种优化器共140个 job；全部六个 scope 共840个 job、6720条候选，最大基础预算约17.475T tokens。不要把范围随意改成0–839后直接提交。四卡脚本一次占用四张卡，不要在同一组卡上同时启动多个四卡命令。

<a id="step-7"></a>
## 第 7 步：查看进度、停止与恢复

### 7.1 在哪里看结果

以 job 0 为例：

```text
RUN_DIR/
├── smoke/4gpu/...             # 短程验收，与正式结果分开
├── jobs.json                  # 任务编号表
└── job_0000/
    ├── search.json            # 已结束候选及当前 winner；首个候选结束后才有
    ├── candidate_00/          # 第1个学习率候选
    │   ├── metadata.json      # 配置、环境、文件身份和参数范围
    │   ├── steps/000001.json  # 每个成功更新步骤的日志
    │   ├── evaluations.json   # 训练/验证评估曲线
    │   ├── status.json        # 完成或中断/失败状态，不是实时心跳
    │   ├── latest.pt          # 恢复用 checkpoint
    │   ├── best.pt            # 最低验证 loss 的诊断 checkpoint
    │   └── final.pt           # 完整预算结束后生成
    └── candidate_01/ ... candidate_07/
```

终端在首步、每10步和末步输出训练进度，并输出评估信息。也可查看已生成的 JSON：

```bash
ls "$RUN_DIR/job_0000/candidate_00/steps/" | tail
python -m json.tool "$RUN_DIR/job_0000/candidate_00/evaluations.json"
python -m json.tool "$RUN_DIR/job_0000/search.json"
```

尚未生成的文件会报不存在，不必据此判断训练失败。某个候选结束后，到下一个 `candidate_XX` 目录查看。

**整个 job 完成的标志：** `search.json` 中 `complete: true`，并有8个候选记录。`winner` 按完整预算末尾验证 loss 选择，不使用 `best.pt` 的最低历史 loss；八个候选均数值失败时，搜索仍算结束，但 `winner` 为空。

### 7.2 停止后如何继续

需要主动停止时，在运行终端按一次 `Ctrl+C`，等待进程退出。程序尝试在完整更新结束后保存 `latest.pt`；外部强杀、断电或调度器不给足退出时间时，只能恢复到此前已经落盘的 checkpoint。

恢复不需要指定 `.pt` 文件，重新激活原环境、设置原路径，并执行**原命令**：

```bash
bash scripts/run_4gpu.sh node 0
```

已完成或数值失败的候选不会重跑，未完成候选从自己的 `latest.pt` 继续；此前没有成功写入 checkpoint 的候选会从统一初始化重新开始。不要改 job 编号来“恢复”，不要换 RUN_DIR，也不要复制其他候选的 checkpoint。

| 状态/问题 | 应如何处理 |
|---|---|
| `completed` | 候选完成；节点控制器继续下一个 LR |
| `numerical_failure` | 出现 NaN/Inf；保留 `failure.pt` 和记录，正式搜索继续其他候选，不算基础设施故障 |
| `infrastructure_failure`、OOM、存储错误、抢占 | 排查环境，恢复原候选；不换 LR 顶替、不擅自修改 batch 或噪声 |
| `.lock` 已存在 | 先确认没有活跃进程写该目录；确认是硬杀遗留锁后才手动移除那个锁文件，不删 checkpoint |
| `refusing incompatible resume` | 代码、数据、配置、PyTorch/CUDA 或布局与记录不符；恢复原环境，不改 metadata 绕过检查 |

四卡实机恢复验收也使用此流程，但应在**尚未完成的 smoke** 上操作，并重跑相同 smoke 命令。检查停止后有 `latest.pt`，且 `status.json` 中的 step 为1–7；重启时的 `[smoke] ... step=N/8` 应接续该步骤，而不是从0开始。如果只保住了 step=0 的初始 checkpoint，这次不能证明训练中途恢复成功，需要重新安排验收。继续到8步只能说明恢复流程跑通；若要证明 GPU 权重与不中断训练逐位一致，还需额外对照，不能仅凭 `completed` 宣称一致。

<a id="step-8"></a>
## 第 8 步：汇总结果并交回文件

训练后，在同一环境和仓库目录执行：

```bash
python -m nanogpt_noise.analyze \
  --runs "$RUN_DIR" --output "$RUN_DIR/analysis" --plots
```

生成 `analysis/surfaces.csv`、`analysis/surfaces.json` 和各 scope 的热图。只汇总正式结果，smoke 不参与；未完成或没有有效 winner 的位置保留空白，不补成零。只有八个候选结束后才给正式节点结果，跨优化器结果未齐时不能据不完整排名下结论。

交回时保留 `search.json`、各候选的 metadata/日志/状态、分析文件和环境/commit 记录。Checkpoint 体积较大，先与项目负责人商定存放或传输方式，**不要自行删除，也不要上传 GitHub**。

<a id="slurm"></a>
## 附录 A：服务器使用 Slurm 时

本节是直接执行四卡命令的替代方式，不是另一套实验。先完成代码、环境和数据准备。提交前激活 Python 环境，设置三个**绝对路径**，并进入仓库根目录。

在学校允许的准备节点运行第4步的文件 `--verify` 和 CPU 单元测试即可；不用在无 GPU 的登录节点运行 `preflight`。下面的批作业会在分配到四卡后自动执行 preflight，检查 GPU 数量、BF16、NCCL、布局和文件哈希。检查通过才会开始 smoke。

分区名和账户名必须替换；学校不要求 account 时删除该选项。GPU 资源写法、CPU/内存和时限按学校规定调整；示例不保证已获分配。

```bash
export PYTHON="$(command -v python)"
sbatch --job-name=nanogpt-smoke \
  --partition=实际分区 --account=实际账户 \
  --nodes=1 --ntasks=1 --gres=gpu:4 \
  --cpus-per-task=16 --mem=64G --time=02:00:00 \
  --export=ALL scripts/run_4gpu.sh smoke
```

`sbatch` 返回的是**调度器作业号**，不是实验的0–839编号。提交成功不等于验收通过；用 `squeue -u "$USER"` 看排队情况，检查 `slurm-调度器作业号.out` 和第5步的 smoke 状态文件。

**后台作业的恢复验收：** 在第一个 smoke 的日志已出现 `step=1/8`、仍未结束时，仅对这一个验收作业发送 TERM（替换为 `sbatch` 返回的实际作业号）：

```bash
scancel --signal=TERM --full 实际调度器作业号
```

`--full` 会把信号发送到批作业及子进程，参见 [Slurm 官方说明](https://slurm.schedmd.com/scancel.html)。等待旧进程退出，按第7.2节检查 checkpoint 和停止状态，再重新提交同一条 `sbatch ... smoke` 命令。新调度器作业号可以不同，但环境、RUN_DIR 和 smoke 参数必须相同。保存是否来得及仍取决于实机退出时限；未检查 checkpoint 前，不要声称已成功恢复。

首个 smoke 完成后，分别把提交命令末尾的 `smoke` 换成第5.2节的 `smoke full adamw 0`、`smoke last1 adam 10`、`smoke last12 sgd 10`，逐个提交和检查。全部通过后再做第5步的环境记录及正式训练。

验收完成后，提交一个正式节点，例如实验 job 0：

```bash
sbatch --job-name=nanogpt-job0 \
  --partition=实际分区 --account=实际账户 \
  --nodes=1 --ntasks=1 --gres=gpu:4 \
  --cpus-per-task=16 --mem=64G --time=24:00:00 \
  --export=ALL scripts/run_4gpu.sh node 0
```

24小时只是资源申请示例，不保证能完成八个候选。超时后重新提交同一命令、同一路径恢复，检查是否遗留锁。不要同时提交两份相同实验编号和 RUN_DIR 的作业。四卡入口直接使用 `run_4gpu.sh`；仓库另有 `slurm.sh` 通用模板，默认只申请一卡，不应不加修改地用于四卡训练。

<a id="faq"></a>
## 附录 B：常见问题

| 现象 | 先检查什么 |
|---|---|
| `No module named nanogpt_noise` | 是否在仓库根目录、激活正确环境、用该环境的 `python -m pip install -e '.[plots]'` 安装 |
| `CUDA unavailable` / `NCCL unavailable` | 是否装了 CPU wheel、位于已分配 GPU 的计算节点、驱动与 CUDA wheel 是否兼容 |
| `Exactly four visible GPUs required` | 实际分配数量和 `CUDA_VISIBLE_DEVICES`；不要指定未分配的卡 |
| `hash mismatch` / 找不到输入文件 | 下载或解压是否完整、路径是否多嵌套一层、数据和公共产物是否属于同一套 |
| 缺 `initial.pt`、`windows.pt` 或 `identity.json` | 向项目提供者补齐；运行者不要自行生成另一套替代 |
| `find_unused_parameters=True` 提示未发现 unused 参数 | 固定 DDP 梯度桶布局的预期开销提示；不要为了消除提示修改训练代码 |
| 开始时较久没有 step 日志 | 正在检查文件、初始评估或写 checkpoint；结合终端错误、GPU 和磁盘状态判断 |
| smoke 报错 | 停止批量启动，把完整报错和配置交回排查；不要只删状态文件后重跑 |
| GitHub 发来 CI 失败邮件 | 查看对应 commit 的 Actions；它是代码测试结果，不是服务器训练状态，详见验证记录 |

<a id="maintainers"></a>
## 附录 C：设计来源与维护者操作

本 README 说明如何运行，不独立维护实验参数。科学协议见[实验设计快照](docs/nanogpt_experiment_design.md)；作者工作区唯一编辑源是 `VSCode下载/New/nanogpt_experiment_design.md`，应先修改主文档再同步发布快照。

- [启动脚本](scripts/run_4gpu.sh)：四卡预检、smoke 和单节点搜索入口。
- `nanogpt_noise/protocol.py`、`search.py`：固定协议与 LR 搜索。
- `model.py`、`optim.py`、`noise.py`：模型、优化器和噪声。
- `data.py`、`prepare.py`：固定公共产物与可选数据制备。
- `train.py`、`node.py`：候选训练、恢复与八候选控制。
- `jobs.py`、`analyze.py`：任务清单和结果汇总。
- [验证记录](docs/VALIDATION.md)：已通过的检查和未验收项目。

仅维护者需要制备新公共产物时，才在新的空目录运行以下命令，之后统一分发、记录身份，不与当前产物混用：

```bash
python -m nanogpt_noise.data \
  --data "$DATA_DIR" --artifacts "/新空目录/nanogpt-artifacts"
```

重新制备语料也不是接手运行的必需步骤。确需重制备时，另装 `python -m pip install -e '.[prepare]'`，使用 `python -m nanogpt_noise.prepare --output /新的空语料目录`。需要网络访问，不能只凭相同 seed 认定与现有数据一致，必须核对哈希；原语料未留存的依赖版本应如实标为未知。

当前支持单机 DDP，不支持跨节点、FSDP 或分片优化器，不依赖 WandB。模型结构参考 nanoGPT，保留[原 MIT 声明](third_party/nanogpt-LICENSE)，不因此额外为整个研究项目授予统一开源许可。
