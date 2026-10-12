# NanoGPT 噪声实验：四张 H100 操作手册

本仓库只负责 NanoGPT 实验。正式运行方式是：一张 H100 训练一个独立 `scope / optimizer / 噪声节点` 任务，四张卡同时处理四个不同任务；同一任务内部的 8 个学习率候选必须顺序完成。

受约 200 GB 存储限制，六个 scope 必须分开运行。一个 scope 完成后，要先生成 manifest、下载到本地并校验，再清空服务器上的该 scope，之后才能运行下一个 scope。代码不设置磁盘水位保护，运行者需要按本手册完成空间检查和交接。

> 当前只完成了 CPU 协议测试。H100 BF16、目标服务器四进程运行、实际 checkpoint 大小和峰值存储仍须在目标机器验收。Smoke test 不是正式实验结果。

## 1. 整体流程

每个 scope 都严格走完以下闭环：

1. 获取代码并准备环境、数据和公共产物；
2. 用单张 H100 运行 smoke test；
3. 为当前 scope 建立独立 `RUN_DIR`；
4. 四张 H100 完整运行该 scope 的 140 个任务；
5. 中断时使用相同 scope、相同代码和相同目录恢复；
6. 全部完成后生成 `scope_manifest.json`；
7. 将 scope 目录直接下载到本地，不在服务器另打大型压缩包；
8. 在本地逐文件校验 SHA-256；
9. 本地校验通过后，清空服务器上的该 scope 目录；
10. 再开始下一个 scope。

推荐顺序为：

```text
full -> last1 -> last2 -> last3 -> last6 -> last12
```

## 2. 获取代码

在 Linux 计算服务器执行：

```bash
git clone https://github.com/ZZZ-020069/nanogpt-noise-experiments.git
cd nanogpt-noise-experiments
```

私有仓库需要运行者先接受 GitHub Collaborator 邀请并使用自己的账号认证。不要共享密码或个人访问令牌。

更新已有仓库时，先确认没有训练进程，再执行：

```bash
git pull --ff-only
```

正式实验开始后固定 commit，不要边训练边更新代码。后文命令均在仓库根目录执行。

## 3. 准备 Python 与 CUDA 环境

优先使用服务器已有的 Python 3.11 和 CUDA 版 PyTorch 2.6–2.x 环境：

```bash
conda activate 实际环境名
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -e '.[plots]'
```

最后一行检查必须显示 CUDA 可用。不要把仓库的 CPU 测试环境当作 H100 环境。

## 4. 准备数据、公共产物与目录

GitHub 不包含大型数据文件。这里要区分“需要提前传到服务器的输入”和“在服务器上新建的输出目录”：

| 变量 | 内容 | 是否需要提前传输 |
|---|---|---|
| `DATA_DIR` | `train.bin`、`val.bin` 和 `meta.json` | **需要**。这是训练输入，必须提前上传到服务器 |
| `ARTIFACT_DIR` | `initial.pt`、`windows.pt` 和 `identity.json` | **需要**。这是统一初始化和固定窗口产物，必须提前上传到服务器 |
| `RUN_BASE` | 六个 scope 输出目录的父目录 | **不需要上传**。在服务器的大容量计算盘或项目盘上创建一个空目录即可 |
| `RUN_DIR` | 当前一个 scope 的实际输出目录，例如 `RUN_BASE/full` | **不需要提前上传**。程序运行时自动产生内容；scope 完成后再从服务器下载到本地 |

因此，“结果根目录”不是需要传给学长的数据包，也不是仓库的一部分；它只是学长在服务器上选择的一个可写位置。优先放在容量足够的 `/scratch/...` 或学校项目盘中，不要在未确认配额时放进个人 home 目录。服务器建议采用以下结构：

```text
数据目录/                              -> DATA_DIR
├── train.bin                         # 2,000,000,000 bytes
├── val.bin                           #    20,000,000 bytes
└── meta.json

公共产物目录/                          -> ARTIFACT_DIR
├── initial.pt                        # 统一未训练初始权重
├── windows.pt                        # 固定训练与评估窗口
└── identity.json                     # 文件哈希与身份

结果根目录/                           -> RUN_BASE，在服务器上新建
├── full/                             # 一次只存在当前 scope
├── last1/
└── ...
```

设置绝对路径。下面示例中，前两个目录需要提前传好内容；`RUN_BASE` 只需由学长在服务器上创建。`RUN_DIR` 每次必须指向一个具体 scope：

```bash
export DATA_DIR="/实际路径/openwebtext_subset/data"
export ARTIFACT_DIR="/实际路径/protocol-v1.4"
export RUN_BASE="/实际路径/nanogpt-v1.5"
mkdir -p "$RUN_BASE"

ls -lh "$DATA_DIR/train.bin" "$DATA_DIR/val.bin" "$DATA_DIR/meta.json"
ls -lh "$ARTIFACT_DIR/initial.pt" "$ARTIFACT_DIR/windows.pt" "$ARTIFACT_DIR/identity.json"
df -B1 "$RUN_BASE"
```

数据传输方向可以概括为：

```text
实验开始前：你的本地 DATA_DIR、ARTIFACT_DIR  ->  H100 服务器
实验运行时：H100 服务器在 RUN_DIR 中生成结果
每个 scope 完成后：服务器 RUN_DIR  ->  你的本地结果目录
本地校验通过后：清空服务器上的这个 RUN_DIR，再运行下一个 scope
```

公共产物沿用已冻结的 manifest，ID 应为：

```text
9cabd9c54c6d717354f8f6750715f9280c5502f0a85a9032806b9c7956cbd76e
```

在开始 `full` 前，结果所在文件系统的可用空间应至少为 `200000000000` bytes，并且不能混放上一 scope 的结果。

## 5. 先做不训练的检查

验证数据和公共产物：

```bash
python -m nanogpt_noise.data \
  --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" --verify
```

运行 CPU 测试：

```bash
python -m unittest discover -s tests -v
```

末尾必须为 `OK`。这一步验证协议和调度代码，不验证 H100。

查看六个 scope 的静态存储估算：

```bash
for SCOPE in full last1 last2 last3 last6 last12; do
  python -m nanogpt_noise.scope estimate --scope "$SCOPE"
done
```

## 6. 用一张 H100 验收单候选

进入实际 GPU 计算节点后检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

为 smoke 单独设置目录：

```bash
export RUN_DIR="$RUN_BASE/smoke"
mkdir -p "$RUN_DIR"
bash scripts/run_4gpu.sh smoke
```

默认 smoke 使用完整 124M 模型、正式 micro-batch 和梯度累积，但只执行 8 步：

```text
micro-batch = 4
training_world = 1
accumulation = 128
global_batch = 512 sequences
tokens_per_step = 524,288
```

验收状态：

```bash
python -m json.tool "$RUN_DIR/smoke/1gpu/full-muon-node10/status.json"
du -sh "$RUN_DIR/smoke/1gpu/full-muon-node10"
```

只有 `status: completed`、`mode: smoke`、`step: 8` 才算通过。OOM、`infrastructure_failure` 或 `numerical_failure` 均不通过。不要为了绕过 OOM 擅自修改 micro-batch、模型、噪声、梯度裁剪或正式预算。

## 7. Scope、job 与存储预算

每个 scope 有 140 个 job；每个 job 固定一个 `scope / optimizer / 噪声节点`，内部顺序运行 8 个完整预算 LR 候选。

| scope | 全局 job | 可训练参数 | winner checkpoint 估算 | 四卡峰值估算 |
|---|---:|---|---:|---:|
| `full` | 0–139 | 全部参数 | 179.64 GB | **195.58 GB** |
| `last1` | 140–279 | 最后 1 个 block | 75.60 GB | 84.03 GB |
| `last2` | 280–419 | 最后 2 个 block | 81.54 GB | 90.43 GB |
| `last3` | 420–559 | 最后 3 个 block | 87.49 GB | 96.83 GB |
| `last6` | 560–699 | 最后 6 个 block | 105.33 GB | 116.03 GB |
| `last12` | 700–839 | 12 个 block，block 外参数冻结 | 141.01 GB | 154.43 GB |

这里使用十进制 GB。`full` 的 195.58 GB 包括：

- 140 个最终 winner 的模型和完整优化器状态：179.64 GB；
- 四个活跃任务在原子 checkpoint 写入时的临时峰值：11.94 GB；
- 数据、公共产物、压缩日志和 metadata 预留：4.00 GB。

这是协议估算，不是 H100 实测值。`full` 的余量很窄，因此：

- 不得同时运行两个 scope；
- 不得把六个 scope 写进同一个 `RUN_DIR`；
- 不得在服务器保留上一 scope 的副本；
- 不得在服务器上再生成与 scope 同体积的 tar/zip；
- 启动前和运行中由运行者用 `df -B1`、`du -sh` 检查，不使用自动磁盘水位保护。

## 8. 正式运行一个 scope

以下以 `full` 为例。其他 scope 只替换 `SCOPE`。

### 8.1 选择 scope 和独立目录

```bash
export SCOPE=full
export RUN_DIR="$RUN_BASE/$SCOPE"
mkdir -p "$RUN_DIR"
```

目录名应与 scope 完全相同。启动器会写入 `scope_run.json`，并拒绝把另一个 scope 的 job 混入该目录。

### 8.2 启动四卡任务池

如果调度器已经设置 `CUDA_VISIBLE_DEVICES`，不要覆盖。独立服务器才按实际分配选择四张卡：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/run_4gpu.sh scope "$SCOPE"
```

四个 worker 各占一张卡。某张卡完成一个 job 后，从当前 scope 的共享队列领取下一个 job。同一 job 的 8 个 LR 候选始终顺序执行，不会被拆到八张卡并行。

建议在 `tmux` 中运行长任务：

```bash
tmux new -s nanogpt-full
# 在 tmux 内重新激活环境、设置变量，然后运行 scope 命令
```

### 8.3 查看进度

```bash
tail -f "$RUN_DIR/launcher/job_0000.log"
python -m json.tool "$RUN_DIR/job_0000/search.json"
du -sh "$RUN_DIR"
df -B1 "$RUN_DIR"
```

单个 job 完成的标志是 `search.json` 中 `complete: true`，且包含 8 个候选记录。

### 8.4 中断与恢复

任务被抢占、SSH 断开或发生基础设施故障后，修复环境并使用完全相同的变量重新执行：

```bash
bash scripts/run_4gpu.sh scope "$SCOPE"
```

完成候选会复用已有终态；未完成候选从自己的 `latest.pt` 恢复。基础设施故障必须恢复原候选，不能换一个 LR 顶替。不要修改 `metadata.json` 绕过身份检查。

单独排查某个 job 时可用：

```bash
GPU_SLOT=0 bash scripts/run_4gpu.sh node 105
```

## 9. 完成、下载、校验与清空

这一节必须按顺序操作。没有完成本地校验，不得清空服务器目录。

### 9.1 在服务器封存 scope

任务池正常结束后执行：

```bash
python -m nanogpt_noise.scope finalize \
  --runs "$RUN_DIR" --scope "$SCOPE"
```

该命令检查 140 个 job 是否全部完成、候选诊断是否已经压缩、checkpoint 是否只属于最终 winner，然后生成 `scope_manifest.json`。生成 manifest 后不要再向该目录写入；若检查失败，按报错恢复缺失任务，再重新 finalize。

### 9.2 直接下载到本地

在本地机器执行，替换服务器地址和绝对路径：

```bash
mkdir -p "/本地结果根目录/$SCOPE"
rsync -av --partial --info=progress2 \
  学长用户名@服务器地址:"$RUN_DIR/" \
  "/本地结果根目录/$SCOPE/"
```

如果本地 shell 没有服务器上的 `$RUN_DIR` 变量，就写服务器绝对路径，例如：

```bash
rsync -av --partial --info=progress2 \
  user@server:/scratch/project/nanogpt-v1.5/full/ \
  /local/results/nanogpt-v1.5/full/
```

连接中断后重复相同 `rsync` 命令即可续传。不要先在服务器创建 tar/zip；那会在最紧张的阶段再占一份大型空间。

### 9.3 在本地校验副本

本地仓库安装完成后执行：

```bash
python -m nanogpt_noise.scope verify \
  --runs "/本地结果根目录/$SCOPE"
```

命令成功结束，才表示文件集合、大小和逐文件 SHA-256 与服务器 manifest 一致。校验失败时继续或重新下载，不能清服务器数据。

### 9.4 清空服务器上的当前 scope

确认以下三件事：本地 `verify` 成功、本地目录可以打开、服务器 `RUN_DIR` 确实是当前 scope 的独立目录。然后回到服务器执行带 scope 专用确认串的清理命令：

```bash
python -m nanogpt_noise.scope cleanup \
  --runs "$RUN_DIR" --scope "$SCOPE" --confirm "DELETE-$SCOPE"
```

清理命令只接受已经 finalize 且自校验通过、目录名与 scope 完全一致的目录；它不会删除 `DATA_DIR`、`ARTIFACT_DIR` 或仓库。删除不可由服务器恢复，本地校验未通过时绝对不要执行。

随后确认空间已经释放：

```bash
test ! -e "$RUN_DIR" && echo "scope removed"
df -B1 "$RUN_BASE"
```

然后设置下一个 `SCOPE` 和新的 `RUN_DIR`，回到第 8 节。

## 10. 结果目录与保留规则

运行中的目录大致为：

```text
RUN_DIR/
├── scope_run.json
├── launcher/job_0000.log
├── job_0000/
│   ├── search.json
│   ├── candidate_00/
│   │   ├── metadata.json
│   │   ├── evaluations.json
│   │   ├── status.json
│   │   ├── steps.jsonl.gz
│   │   ├── steps.compaction.json
│   │   └── final.pt                  # 仅当它是当前/最终 winner
│   └── candidate_01/ ... candidate_07/
└── scope_manifest.json               # finalize 后生成
```

保留策略如下：

- 活跃候选只有一个 `latest.pt`，用于恢复；
- 候选完整结束时将最后的 `latest.pt` 重命名为 `final.pt`，不另存 `best.pt`；
- 每完成一个候选并写稳搜索记录，只保留当前 winner 的一个 `final.pt`；
- winner checkpoint 包含模型、优化器和 Muon 辅助组状态；
- 非 winner 不保留 checkpoint，但保留 metadata、状态、完整评估曲线和全部逐步诊断；
- 逐步诊断只是从大量小 JSON 无损转成 gzip JSONL，没有降采样或删字段；
- 如果 8 个候选全部数值失败，该 job 不保留 checkpoint。

## 11. 汇总六个已下载 scope

六个 scope 可以保存在六个独立本地目录，无需重新合并。全部下载后执行：

```bash
python -m nanogpt_noise.analyze \
  --runs \
    "/本地结果根目录/full" \
    "/本地结果根目录/last1" \
    "/本地结果根目录/last2" \
    "/本地结果根目录/last3" \
    "/本地结果根目录/last6" \
    "/本地结果根目录/last12" \
  --output "/本地结果根目录/analysis" --plots
```

Smoke、不完整任务和无有效 winner 的节点不会伪装成正式有限结果，也不会补为零。

## 12. Slurm 提交示例

仓库提供单节点四卡模板 [scripts/slurm.sh](scripts/slurm.sh)。按学校实际分区、账号和路径设置变量：

```bash
export DATA_DIR="/实际绝对路径/data"
export ARTIFACT_DIR="/实际绝对路径/artifacts"
export RUN_BASE="/实际绝对路径/nanogpt-v1.5"
export SCOPE=full
export RUN_DIR="$RUN_BASE/$SCOPE"

sbatch --partition=实际分区 --account=实际账号 \
  --export=ALL scripts/slurm.sh
```

模板申请单节点 4 张 GPU、16 个 CPU、64 GB 主存和 24 小时。按集群规定调整分区、账号和时限，但正式 scope 模式必须保留四张 GPU。超时后用相同 `SCOPE` 和 `RUN_DIR` 重新提交即可恢复。

```bash
squeue -u "$USER"
scancel --signal=TERM --full 实际Slurm作业号
```

Slurm 作业号不是实验 job 编号。

## 13. 常见问题

| 现象 | 处理 |
|---|---|
| `No module named nanogpt_noise` | 在仓库根目录激活正确环境并运行 `python -m pip install -e '.[plots]'` |
| `CUDA unavailable` | 确认进入 GPU 节点、安装 CUDA 版 PyTorch，检查驱动兼容性 |
| `scope requires exactly four ... tokens` | `CUDA_VISIBLE_DEVICES` 必须恰好列出调度器分配的四张卡 |
| `belongs to scope ...` | 当前 `RUN_DIR` 已属于另一个 scope；为新 scope 使用新目录 |
| `scope jobs incomplete` | 查看报错列出的缺失 job，使用原目录恢复任务池 |
| `diagnostics are not compacted` | 重新运行同一 scope；终态候选会复用结果并补做压缩 |
| manifest `hash mismatch` / `size mismatch` | 本地副本不完整或损坏，继续 `rsync`，不要清服务器数据 |
| `.lock` 已存在 | 先确认没有活跃进程；只有确认是硬杀遗留锁后才移除锁 |
| smoke OOM | 停止正式启动并保存完整错误和 `nvidia-smi`，不要擅改协议参数 |
| GitHub Actions 失败邮件 | 这是代码 CPU 测试状态，不是服务器训练状态；查看对应 commit 的 Actions 日志 |

## 14. 设计与代码入口

README 只说明操作。科学协议的唯一权威来源是工作区的 `VSCode下载/New/nanogpt_experiment_design.md`；仓库中的 [设计快照](docs/nanogpt_experiment_design.md) 随发布同步。

- [scripts/run_4gpu.sh](scripts/run_4gpu.sh)：单卡 smoke、单 job 和逐 scope 四卡入口；
- `nanogpt_noise/pool.py`：四个独立单卡 worker 的动态调度；
- `nanogpt_noise/node.py`：一个 job 内 8 个 LR 候选的顺序搜索；
- `nanogpt_noise/train.py`：单候选训练、状态与恢复；
- `nanogpt_noise/storage.py`：诊断压缩与 winner-only checkpoint；
- `nanogpt_noise/scope.py`：scope 隔离、估算、finalize、verify 和受保护清理；
- `nanogpt_noise/analyze.py`：跨多个已下载 scope 的统一汇总；
- [实现验证记录](docs/VALIDATION.md)：已完成检查和目标硬件待验项目。

模型结构参考 nanoGPT，并保留[原 MIT 声明](third_party/nanogpt-LICENSE)。本仓库不依赖 WandB。底层仍保留 DDP 回归测试，但正式 H100 路径是一张卡运行一个独立任务。
