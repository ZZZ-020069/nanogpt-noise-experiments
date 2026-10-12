# NanoGPT 噪声实验：四张 H100 运行指南

本仓库当前只负责 NanoGPT 实验。默认运行方式是：**一张 H100 训练一个独立任务，四张卡同时处理四个不同任务**，不是四张卡通过 DDP 共同训练一个候选。

一个任务（job）固定一个 `scope / optimizer / 噪声节点` 组合。任务内部依次运行 8 个学习率候选：前 5 个是初始对数网格，后 3 个根据已经完成的结果自适应提出。因此，同一任务内的 8 个候选不能并行；任务之间可以并行。四卡任务池会让空闲 GPU 自动领取下一个 job。

运行顺序是：

1. 准备代码、环境、数据和公共产物；
2. 先用一张 H100 跑 smoke test，确认单卡能承载一个候选；
3. 再用四张 H100 启动独立任务池；
4. 发生中断时，用相同范围和相同输出目录重跑以恢复。

> 当前 CPU 单元测试只能验证协议和调度代码，不能证明 H100 显存、CUDA 性能或实机恢复已经通过。首次 H100 smoke test 的输出才是单卡承载能力的验收依据。smoke test 不是正式实验结果。

## 1. 获取代码

在 Linux 计算服务器上执行：

```bash
git clone https://github.com/ZZZ-020069/nanogpt-noise-experiments.git
cd nanogpt-noise-experiments
```

若仓库保持私有，运行者必须先接受 GitHub Collaborator 邀请，并用自己的 GitHub 账号完成认证；不要传递账号密码或个人访问令牌。

已经下载过时，只在没有训练进程运行时更新：

```bash
git pull --ff-only
```

后文命令都默认在仓库根目录执行。正式实验开始后固定 commit，不要一边训练一边更新代码。

## 2. 准备 Python 和 CUDA 环境

使用服务器已有的 Python 3.11 及 CUDA 版 PyTorch 2.6–2.x 环境；下面的环境名需要替换：

```bash
conda activate 实际环境名
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -e '.[plots]'
```

最后一行必须显示 CUDA 可用。若需要新建环境，请按服务器驱动选择 [PyTorch 官方 CUDA 安装命令](https://pytorch.org/get-started/locally/)，再安装本项目。不要把仓库中的 CPU 测试环境当作 H100 环境。

## 3. 放置数据和公共产物

GitHub 仓库不包含大文件。需要从项目提供者处取得并解压为以下结构：

```text
数据目录/                         -> DATA_DIR
├── train.bin                    # 2,000,000,000 bytes
├── val.bin                      #    20,000,000 bytes
└── meta.json

公共产物目录/                     -> ARTIFACT_DIR
├── initial.pt                   # 全任务统一的未训练初始权重
├── windows.pt                   # 固定训练与评估窗口
└── identity.json                # 文件哈希和协议身份

结果目录/                         -> RUN_DIR，由程序写入
```

设置绝对路径；示例路径必须替换：

```bash
export DATA_DIR="/实际路径/openwebtext_subset/data"
export ARTIFACT_DIR="/实际路径/protocol-v1.4"
export RUN_DIR="/实际路径/results/nanogpt-v1.4"

mkdir -p "$RUN_DIR"
ls -lh "$DATA_DIR/train.bin" "$DATA_DIR/val.bin" "$DATA_DIR/meta.json"
ls -lh "$ARTIFACT_DIR/initial.pt" "$ARTIFACT_DIR/windows.pt" "$ARTIFACT_DIR/identity.json"
df -h "$RUN_DIR"
```

每次新开终端或提交新调度作业，都要重新激活环境并设置这三个变量。恢复训练时必须继续指向原来的 `RUN_DIR`。

## 4. 先做不训练的检查

验证数据和公共产物的哈希：

```bash
python -m nanogpt_noise.data \
  --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" --verify
```

当前分发产物的 ID 应为：

```text
9cabd9c54c6d717354f8f6750715f9280c5502f0a85a9032806b9c7956cbd76e
```

运行 CPU 实现测试：

```bash
python -m unittest discover -s tests -v
```

末尾应为 `OK`。这一步不消耗正式训练预算，也不能替代 H100 验收。

## 5. 用一张 H100 验证单候选承载能力

先申请一张 H100，进入实际 GPU 计算节点。脚本只使用已经分配的 GPU，不会申请资源。检查：

```bash
nvidia-smi
python -c "import torch; print('available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

然后运行默认 smoke test：

```bash
bash scripts/run_4gpu.sh smoke
```

它使用完整 124M 模型、正式 micro-batch 和梯度累积方式，但只执行 8 个更新步。脚本会把单卡训练世界设为：

```text
micro-batch = 4
training_world = 1
accumulation = 128
global_batch = 512 sequences
tokens_per_step = 524,288
```

输出位于：

```text
$RUN_DIR/smoke/1gpu/full-muon-node10/
```

验收状态：

```bash
python -m json.tool "$RUN_DIR/smoke/1gpu/full-muon-node10/status.json"
```

只有 `status: completed`、`mode: smoke`、`step: 8` 才算通过。OOM、`infrastructure_failure` 或 `numerical_failure` 都不算通过。不要为绕过 OOM 擅自改变 micro-batch、模型、噪声、梯度裁剪或正式预算；先把日志和显存情况交回排查。

可选地补查其他路径：

```bash
bash scripts/run_4gpu.sh smoke full adamw 0
bash scripts/run_4gpu.sh smoke last1 adam 10
bash scripts/run_4gpu.sh smoke last12 sgd 10
```

如果当前分配了多张卡，`smoke` 默认只用可见列表中的第 0 个槽位。可用 `GPU_SLOT=2 bash scripts/run_4gpu.sh smoke` 指定第 2 个槽位。脚本会保留 Slurm 提供的真实 GPU ID 或 UUID，不会把它误写成物理卡 2。

实机验收通过后保存环境和代码版本：

```bash
python -m pip freeze > "$RUN_DIR/environment-lock.txt"
git rev-parse HEAD > "$RUN_DIR/code-commit.txt"
```

## 6. 理解 job 编号

生成完整任务表，不启动训练：

```bash
python -m nanogpt_noise.jobs --output "$RUN_DIR/jobs.json"
```

共有 840 个 job：

| scope | job 编号 | 可训练参数 |
|---|---:|---|
| `full` | 0–139 | 全部参数 |
| `last1` | 140–279 | 最后 1 个 block |
| `last2` | 280–419 | 最后 2 个 block |
| `last3` | 420–559 | 最后 3 个 block |
| `last6` | 560–699 | 最后 6 个 block |
| `last12` | 700–839 | 12 个 block，block 外参数仍冻结 |

每个 scope 内依次是 AdamW、Adam、SGD-Momentum、Muon，各有 35 个噪声节点。例如 `full` 中 AdamW 为 0–34、Adam 为 35–69、SGD 为 70–104、Muon 为 105–139。

查看某个 job 而不训练：

```bash
python -m nanogpt_noise.node --job-index 0 \
  --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" \
  --output "$RUN_DIR" --dry-run
```

注意：`smoke ... 10` 中的 `10` 是优化器内部噪声节点号；`node 10` 中的 `10` 是全局 job 编号。

## 7. 启动四张 H100 的独立任务池

### 7.1 第一次只跑一个小范围

申请同一节点上的四张 H100。若调度器已经设置 `CUDA_VISIBLE_DEVICES`，不要覆盖；独立服务器上若可见卡超过四张，只选择已分配的四张：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 仅在需要手动选择时执行
nvidia-smi
```

先让四张卡各领取一个正式 job：

```bash
bash scripts/run_4gpu.sh queue 0 3
```

`0 3` 是包含两端的区间。运行过程如下：

```text
共享队列: job 0, 1, 2, 3
GPU A -> job 0 -> 依次运行 candidate 00...07
GPU B -> job 1 -> 依次运行 candidate 00...07
GPU C -> job 2 -> 依次运行 candidate 00...07
GPU D -> job 3 -> 依次运行 candidate 00...07
```

每个 candidate 正常情况下完整运行 4,960 步，约 2.6B tokens。这里是正式实验，不是 8 步 smoke，因此可能运行很久。

### 7.2 扩大范围

确认四个 job 的显存、日志、checkpoint、恢复和存储均正常后，再提交经过确认的区间。例如运行 `full / AdamW` 的 35 个噪声节点：

```bash
bash scripts/run_4gpu.sh queue 0 34
```

四个 worker 各自完成一个完整 job；某张卡先完成，就从共享队列领取下一个 job。同一 job 的 8 个 LR 候选始终由同一张卡顺序执行，因而第 6–8 个候选的自适应依赖不会被破坏。

不要在同一个 `RUN_DIR` 中同时启动两个有重叠 job 区间的任务池。不要未经预算确认就直接运行 `queue 0 839`。

在不使用 Slurm 的独立服务器上，应把长任务放进 `tmux` 等持久会话，避免 SSH 断开结束进程。例如先运行 `tmux new -s nanogpt`，在会话中完成环境设置并启动任务池；按 `Ctrl+B`、再按 `D` 脱离，之后用 `tmux attach -t nanogpt` 返回。

### 7.3 单独运行或恢复一个 job

排查或只恢复 job 105 时，可以让一张指定卡运行：

```bash
GPU_SLOT=0 bash scripts/run_4gpu.sh node 105
```

`node` 仍会在 job 内按顺序完成 8 个候选；它不是只运行一个 LR。

## 8. 日志、状态和恢复

任务池的调度日志与训练输出分开保存：

```text
RUN_DIR/
├── launcher/job_0000.log          # pool 子进程的标准输出和错误
├── job_0000/
│   ├── search.json                # 8 个候选与 winner；完整结束后 complete=true
│   ├── candidate_00/
│   │   ├── metadata.json
│   │   ├── steps/*.json
│   │   ├── evaluations.json
│   │   ├── status.json
│   │   ├── latest.pt              # 恢复 checkpoint
│   │   ├── best.pt
│   │   └── final.pt
│   └── candidate_01/ ... candidate_07/
└── smoke/1gpu/...                 # 非正式验收结果
```

查看进度：

```bash
tail -f "$RUN_DIR/launcher/job_0000.log"
python -m json.tool "$RUN_DIR/job_0000/candidate_00/status.json"
python -m json.tool "$RUN_DIR/job_0000/search.json"
```

整个 job 完成的标志是 `search.json` 中 `complete: true` 且有 8 条候选记录。尚未生成某个 JSON 不一定代表失败，可能仍在首次评估或写 checkpoint。

中断任务池时按一次 `Ctrl+C`，或由调度器发送 `TERM`。任一子任务返回非零状态时，任务池停止分配新 job，并终止其余活动子进程，以免继续消耗预算。排除基础设施问题后，使用**完全相同的区间、路径、环境和代码**重跑：

```bash
bash scripts/run_4gpu.sh queue 0 34
```

已经完成的候选会返回既有结果；未完成候选从自己的 `latest.pt` 恢复。基础设施故障必须恢复原候选，不能换一个 LR 顶替。

| 状态或问题 | 处理方式 |
|---|---|
| `completed` | 候选完成，job 控制器继续下一个 LR |
| `numerical_failure` | 保留记录，搜索按协议继续；这不是基础设施故障 |
| `infrastructure_failure`、OOM、磁盘错误、抢占 | 修复环境后重跑原区间，恢复原候选 |
| `.lock` 已存在 | 先确认没有活动进程；只有确认是硬杀遗留锁后才移除锁，不删除 checkpoint |
| `refusing incompatible resume` | 恢复原代码、数据、配置和软件环境，不修改 metadata 绕过检查 |

## 9. 存储与结果汇总

最占空间的是 optimizer checkpoint。按当前估算，`full` 的 Adam/AdamW 单个 candidate 的 `latest/best/final` 合计可达约 4.5 GB，一个 8-candidate job 约 36 GB；四个并发 full job 需要为已完成 checkpoint 加临时写入至少预留约 144 GB。不同 scope 和优化器会变化。不要按单个模型大小估算整个网格，也不要在没有容量计划时运行全部 840 个 job。

查看剩余空间：

```bash
df -h "$RUN_DIR"
du -sh "$RUN_DIR"/job_* 2>/dev/null | sort -h | tail
```

汇总已经完成的正式结果：

```bash
python -m nanogpt_noise.analyze \
  --runs "$RUN_DIR" --output "$RUN_DIR/analysis" --plots
```

smoke 不进入正式汇总；未完成或没有有效 winner 的节点保持为空，不补成零。checkpoint 不上传 GitHub，也不要未经确认删除。

## 10. Slurm 示例

集群参数必须按学校实际分区、账号、H100 资源写法和时限调整。脚本不会替你猜这些设置。

### 10.1 单卡 smoke

先申请一张 H100 的交互式会话，再执行第 5 节命令。若只能使用 `sbatch`，可按学校规范创建一个申请 1 张 GPU、最终执行 `bash scripts/run_4gpu.sh smoke` 的短作业。

### 10.2 四卡正式任务池

仓库提供 [scripts/slurm.sh](scripts/slurm.sh)。提交前先在 shell 中设置路径和 job 区间：

```bash
export DATA_DIR="/实际绝对路径/data"
export ARTIFACT_DIR="/实际绝对路径/artifacts"
export RUN_DIR="/实际绝对路径/results"
export START_JOB=0
export END_JOB=3

sbatch --partition=实际分区 --account=实际账号 \
  --export=ALL scripts/slurm.sh
```

模板申请单节点 4 张 GPU、16 个 CPU、64 GB 主存和 24 小时，并默认给每个独立 worker 4 个 OpenMP 线程。按集群要求修改，但不要把 GPU 数改成 1 后仍调用 `queue`。超时后用原变量重新提交即可恢复。

用以下命令查看或停止：

```bash
squeue -u "$USER"
scancel --signal=TERM --full 实际Slurm作业号
```

Slurm 作业号不是实验 job 编号。

## 11. 常见问题

| 现象 | 优先检查 |
|---|---|
| `No module named nanogpt_noise` | 是否在仓库根目录、激活正确环境并执行了 `python -m pip install -e '.[plots]'` |
| `CUDA unavailable` | 是否进入 GPU 计算节点、是否误装 CPU PyTorch、驱动与 CUDA wheel 是否兼容 |
| `Expected 4 visible GPU(s)` | 是否实际申请了 4 张卡；Slurm 的 `CUDA_VISIBLE_DEVICES` 是否被手动覆盖 |
| `queue requires exactly four ... tokens` | `CUDA_VISIBLE_DEVICES` 应恰好列出调度器分配的四张卡 |
| `hash mismatch` | 下载/解压是否完整，路径是否多套一层目录，数据和公共产物是否来自同一套分发 |
| 缺 `initial.pt`、`windows.pt` 或 `identity.json` | 向项目提供者补齐，不要自行生成一套替代品 |
| smoke OOM | 停止正式启动，保留完整错误与 `nvidia-smi` 信息，不擅自改变协议参数 |
| GitHub Actions 失败邮件 | 这是代码 CPU 测试状态，不是服务器训练状态；打开对应 commit 的 Actions 日志 |

## 12. 设计与代码入口

README 只说明如何操作，不单独定义科学参数。仓库中的设计快照见 [docs/nanogpt_experiment_design.md](docs/nanogpt_experiment_design.md)；项目维护者工作区的唯一权威设计源是 `VSCode下载/New/nanogpt_experiment_design.md`。

- [scripts/run_4gpu.sh](scripts/run_4gpu.sh)：单卡 smoke、单 job 和四卡动态任务池入口。
- `nanogpt_noise/pool.py`：四个单卡 worker 的动态调度与日志。
- `nanogpt_noise/node.py`：一个 job 内 8 个 LR 候选的顺序、自适应控制。
- `nanogpt_noise/train.py`：单个候选的训练、状态和恢复。
- `nanogpt_noise/protocol.py`、`search.py`：固定协议和 LR 搜索规则。
- [docs/VALIDATION.md](docs/VALIDATION.md)：已完成和仍待实机验证的检查记录。

模型结构参考 nanoGPT，并保留[原 MIT 声明](third_party/nanogpt-LICENSE)。本仓库不依赖 WandB。底层仍保留 DDP 回归测试能力，但正式 H100 启动路径采用一张卡一个独立任务。
