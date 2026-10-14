# NanoGPT 噪声实验：四张 H100 运行与结果交接手册

本仓库包含一套已经固定参数的 NanoGPT 训练任务。无需自行选择噪声、优化器或学习率，也不必根据中途结果修改训练设置；工作内容是准备规定的输入文件，在 Linux 计算节点上执行统一入口脚本，并将生成的结果完整交接到持久存储。常规操作只需使用 `scripts/h100.sh`，其余命令主要用于故障排查。

> **请先阅读 [简短任务说明](RUN_TASK.md)。** 本 README 用于需要进一步查询环境、存储、恢复、传输和故障处理细节时参考。

正式运行采用四个独立单卡进程：每张 H100 同时处理一个不同任务，不采用四卡 DistributedDataParallel（DDP）共同训练一个候选。由于结果盘按约 200 GB 规划，六个训练范围必须逐个运行；完成一个范围后，先封存、传出并校验，再清空服务器副本并运行下一个范围。程序不会自动监控剩余磁盘空间。

本文使用的术语如下：

| 术语 | 含义 |
|---|---|
| scope | Nanogpt可训练参数范围，此实验Nanogpt采用12层transformer，共有 `full`、`last1`、`last2`、`last3`、`last6`、`last12` 六种，对应含义为完整Nanogpt，只学习最后一层transformer的Nanogpt，只学习最后两层…… |
| 噪声节点 | 一组固定的 `(sigma1, sigma2)` 参数；每个 scope 有 35 个，程序自动枚举 |
| job | 一个固定的 `scope + optimizer + 噪声节点` 组合；每个 scope 有 140个（4个优化器乘以35个噪声节点）  |
| candidate | 一个 job 中的一条候选学习率训练轨迹；每个 job 顺序运行 8 条完整轨迹（也就是学习率预算为8次） |
| winner | 8 个 candidate 中最终验证损失最低的有效候选 |
| checkpoint | 用于恢复或保留 winner 的模型与优化器状态文件 |
| manifest | `scope_manifest.json`，记录一个 scope 全部交付文件的路径、大小和 SHA-256 |
| smoke test | 只训练 8 步的实机承载验收，不计入正式实验结果 |
| metadata | 程序自动记录的配置、输入身份、代码和软硬件环境信息 |

> 仓库中的自动化测试目前验证了 CPU 协议实现；H100 BF16、目标服务器上的实际显存、运行时间、checkpoint 大小和存储峰值必须通过本文规定的实机 smoke test 与首批运行核验。Smoke test 不是正式实验结果。

## 1. 开始前需要具备的条件

开始操作前，应确认已经取得：

1. 本代码仓库的读取权限；
2. 私有输入仓库 `ZZZ-020069/nanogpt-noise-inputs` 的 Collaborator 邀请，并已使用运行者自己的 GitHub 账号接受；
3. 服务器上的 Git LFS；输入仓库内含 `train.bin`、`val.bin`、`meta.json`、`initial.pt`、`windows.pt`、`identity.json`；
4. 一台 Linux 计算节点上的四张 H100，四张卡必须能被同一个作业同时看见并支持 BF16；
5. Python 3.11 或更高版本，以及与服务器驱动兼容的 CUDA 版 PyTorch；
6. `full` 运行开始前至少 `200000000000` bytes 的可用文件系统空间；
7. 一个独立于临时算力盘的持久结果接收位置，以及从计算服务器向该位置传输文件的方法；
8. 若使用 Slurm，学校实际提供的 partition、account、GPU 资源写法和作业时限。

输入仓库无权访问、Git LFS 不可用、输入目录缺文件或校验不通过时应停止并联系项目提供方，不必自行重新生成、替换或修改数据与冻结产物。学校集群参数不确定时需向集群管理员确认。

运行中若出现 OOM、CUDA、文件系统或调度器错误，应保存完整日志并恢复原任务。不要通过改学习率、batch、训练步数、模型、噪声、梯度裁剪、优化器或代码来规避错误。

完整操作顺序如下：

1. 克隆代码，准备 Python/CUDA 环境；
2. 从私有输入仓库将数据和冻结产物下载到服务器的固定输入目录，并完成 LFS、SHA-256 与项目身份校验；
3. 运行输入校验和 CPU 测试；
4. 申请一张 H100，完成 8 步 smoke test；
5. 申请同一节点上的四张 H100，完整运行当前 scope 的 140 个 job；
6. 中断时使用相同 commit、环境、scope 和输出目录恢复；
7. scope 全部完成后生成 `scope_manifest.json`；
8. 将该 scope 目录直接传到约定的持久存储，不在服务器另建大型压缩包；
9. 在接收端逐文件校验 SHA-256；
10. 校验通过后清空服务器上的该 scope，再开始下一个 scope。

推荐顺序为：

```text
full -> last1 -> last2 -> last3 -> last6 -> last12
```

## 2. 获取代码

在 Linux 计算服务器执行：

```bash
cd /data/260010171
git clone https://github.com/ZZZ-020069/nanogpt-noise-experiments.git
cd /data/260010171/nanogpt-noise-experiments
```

如果仓库为私有状态，运行者需要先接受仓库管理员发送的 GitHub Collaborator 邀请，再使用自己的 GitHub 账号认证。遇到 `Repository not found` 或权限错误时先确认邀请和登录状态；不要共享密码或个人访问令牌。

更新已有仓库时，先确认没有训练进程，再执行：

```bash
git pull --ff-only
```

正式实验开始后固定 commit，不要边训练边更新代码。后文命令均在仓库根目录执行。

## 3. 准备 Python 与 CUDA 环境

优先使用服务器已有的 Python 3.11 或更高版本以及 CUDA 版 PyTorch 2.6–2.x 环境。将下面的 `实际环境名` 替换为服务器上的 Conda 环境；如果服务器不使用 Conda，则先按管理员提供的方法进入正确环境：

```bash
conda activate 实际环境名
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip install -e '.[plots]'
```

PyTorch 必须带 CUDA 支持，`torch.version.cuda` 不能为 `None`。登录节点可能没有 GPU，因此 GPU 可用性和 BF16 支持在第 6 节进入计算节点后检查。安装命令只安装本项目代码及分析依赖，不会替运行者申请 GPU。不要把仓库的 CPU 测试环境当作 H100 环境。

## 4. 准备数据、冻结产物与目录

大型输入不放在本代码仓库，而是放在独立的私有 Git LFS 仓库：

```text
https://github.com/ZZZ-020069/nanogpt-noise-inputs.git
```

这里要区分“从输入仓库下载到服务器的文件”和“在服务器上新建的输出目录”：

| 变量 | 内容 | 是否需要提前传输 |
|---|---|---|
| `DATA_DIR` | `train.bin`、`val.bin` 和 `meta.json` | **需要**。从私有输入仓库通过 Git LFS 下载 |
| `ARTIFACT_DIR` | `initial.pt`、`windows.pt` 和 `identity.json` | **需要**。从同一私有输入仓库通过 Git LFS 下载；不要自行生成替代文件 |
| `RUN_BASE` | 六个 scope 输出目录的父目录 | **不需要上传**。在服务器的大容量计算盘或项目盘上创建一个空目录即可 |
| `RUN_DIR` | 当前一个 scope 的实际输出目录，例如 `RUN_BASE/full` | **不需要提前上传**。程序运行时自动产生内容；scope 完成后再传到约定的持久结果存储 |

结果根目录不是需要提前接收的数据包，也不是仓库的一部分；它只是运行者在服务器上选择的一个可写位置。不要在未确认配额时放进个人 home 目录。本项目当前服务器采用以下结构：

```text
/data/260010171/
├── nanogpt-noise-experiments/        # 本代码仓库
└── results/nanogpt-v1.5/             # RUN_BASE，在服务器上新建
    └── 当前scope/                    # RUN_DIR，例如 full/

/data/260010189/
└── inputs/                           # 私有输入仓库的工作区
    ├── openwebtext_subset/           # DATA_DIR
    │   ├── train.bin                 # 2,000,000,000 bytes
    │   ├── val.bin                   #    20,000,000 bytes
    │   └── meta.json
    └── frozen-artifacts/             # ARTIFACT_DIR
        ├── initial.pt                # 统一未训练初始权重
        ├── windows.pt                # 固定训练与评估窗口
        └── identity.json             # 文件哈希与身份
```

### 4.1 取得私有仓库权限并检查 Git LFS

项目仓库所有者先在输入仓库的 `Settings -> Collaborators` 中添加运行者的 GitHub 用户名；运行者必须使用自己的账号接受邀请。不要共享 GitHub 密码或个人访问令牌。服务器使用 HTTPS 时应按学校允许的方式配置 GitHub CLI、credential helper 或个人访问令牌；GitHub 的密码提示框不接受账号登录密码。

在服务器检查 Git LFS：

```bash
git lfs version
```

如果命令不存在，先使用服务器的软件模块、系统包管理器或管理员提供的方法安装 Git LFS，不要继续执行普通 `git fetch` 来代替 LFS 下载。

### 4.2 首次下载六个输入文件

学长已经创建 `/data/260010189/inputs/openwebtext_subset/` 时，不要再把仓库克隆成额外一层 `nanogpt-noise-inputs/`。应直接把 `/data/260010189/inputs` 初始化成输入仓库工作区，使最终路径与程序配置一致：

```bash
cd /data/260010189/inputs
git init
git remote add origin https://github.com/ZZZ-020069/nanogpt-noise-inputs.git
git lfs install
git fetch origin main
git checkout -b main --track origin/main
git lfs pull
```

若该目录已经完成过上述初始化，则不要再次执行 `git init` 或 `git remote add`，改用：

```bash
cd /data/260010189/inputs
git pull --ff-only
git lfs pull
```

当前输入交付提交为：

```text
1b3643ac44ff2b76ed471cc130a04c2c9c52fb01
```

检查提交、LFS 对象、文件大小和仓库自带的 SHA-256 清单：

```bash
cd /data/260010189/inputs
git rev-parse HEAD
git lfs ls-files
ls -lh openwebtext_subset/train.bin \
       openwebtext_subset/val.bin \
       openwebtext_subset/meta.json \
       frozen-artifacts/initial.pt \
       frozen-artifacts/windows.pt \
       frozen-artifacts/identity.json
sha256sum -c SHA256SUMS
```

`git lfs ls-files` 应列出四个大文件，`sha256sum` 的六项均必须为 `OK`。很小且以 `version https://git-lfs.github.com/spec/v1` 开头的 `.bin` 或 `.pt` 是未下载的 LFS 指针，不能用于实验。

### 4.3 设置实验目录

设置绝对路径；`DATA_DIR` 与 `ARTIFACT_DIR` 指向刚下载的两个子目录，`RUN_BASE` 是服务器上的结果根目录：

```bash
cd /data/260010171/nanogpt-noise-experiments
export DATA_DIR="/data/260010189/inputs/openwebtext_subset"
export ARTIFACT_DIR="/data/260010189/inputs/frozen-artifacts"
export RUN_BASE="/data/260010171/results/nanogpt-v1.5"
mkdir -p "$RUN_BASE"

ls -lh "$DATA_DIR/train.bin" "$DATA_DIR/val.bin" "$DATA_DIR/meta.json"
ls -lh "$ARTIFACT_DIR/initial.pt" "$ARTIFACT_DIR/windows.pt" "$ARTIFACT_DIR/identity.json"
df -B1 "$RUN_BASE"
```

数据传输方向可以概括为：

```text
实验开始前：私有输入仓库  ->  H100 服务器的 DATA_DIR、ARTIFACT_DIR
实验运行时：H100 服务器在 RUN_DIR 中生成结果
每个 scope 完成后：服务器 RUN_DIR  ->  约定的持久结果接收位置
接收端校验通过后：清空服务器上的这个 RUN_DIR，再运行下一个 scope
```

`SHA256SUMS` 检查下载传输是否完整；`identity.json` 描述冻结产物身份。下一节的项目级校验还会同时检查数据和冻结产物的逐文件哈希，并输出预期 ID。两层校验都必须通过。

在开始 `full` 前，结果所在文件系统的可用空间应至少为 `200000000000` bytes，并且不能混放上一 scope 的结果。

## 5. 先做不训练的检查

在仓库根目录执行统一检查：

```bash
bash scripts/h100.sh check
```

脚本会依次完成以下操作：

1. 要求 Git 工作区没有未提交修改，并显示当前 commit；
2. 检查固定路径下的六个输入文件；若输入目录含 `SHA256SUMS`，同时执行逐文件校验；
3. 验证数据与冻结产物的项目身份；
4. 运行完整 CPU 测试；
5. 输出六个 scope 的存储估算，以及当前结果文件系统的 `df`、`du` 信息。

输入身份必须为：

```text
9cabd9c54c6d717354f8f6750715f9280c5502f0a85a9032806b9c7956cbd76e
```

当前版本应显示 46 项测试并在末尾输出 `OK`。这一步验证协议和调度代码，不验证 H100；检查失败时不要启动训练。

如果需要单独排查某一项，可手动执行底层命令：

```bash
python -m nanogpt_noise.data \
  --data /data/260010189/inputs/openwebtext_subset \
  --artifacts /data/260010189/inputs/frozen-artifacts --verify
python -m unittest discover -s tests -v
git status --short
git rev-parse HEAD
```

六个 scope 必须使用同一个提交；正式 smoke 开始后不要执行 `git pull` 或编辑代码。脚本只显示磁盘信息，不会设置自动停机阈值，运行者仍需自行判断空间是否足够。

## 6. 用一张 H100 验收单候选

进入实际 GPU 计算节点后检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

前两个布尔值必须都为 `True`，GPU 名称应为实际分配的 H100。

先申请只含一张 H100 的交互式计算会话，并确认当前进程只看见分配到的 GPU。进入仓库根目录后执行：

```bash
bash scripts/h100.sh smoke
```

`smoke` 会先重复第 5 节的全部检查，再运行单卡训练，因此不需要手工设置 `DATA_DIR`、`ARTIFACT_DIR`、`RUN_BASE` 或 `RUN_DIR`。

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
bash scripts/h100.sh status
python -m json.tool /data/260010171/results/nanogpt-v1.5/smoke/1gpu/full-muon-node10/status.json
```

只有终端显示 `SMOKE PASSED`，且状态为 `status: completed`、`mode: smoke`、`step: 8` 才算通过。脚本会把本次 smoke 的 commit 写入结果根目录；正式 scope 必须使用相同 commit。OOM、`infrastructure_failure` 或 `numerical_failure` 均不通过。不要为了绕过 OOM 擅自修改 micro-batch、模型、噪声、梯度裁剪或正式预算。

## 7. Scope、job 与存储预算

每个 scope 有 140 个 job：AdamW、Adam、SGD-Momentum、Muon 四个优化器分别覆盖 35 个噪声节点。每个 job 内部顺序运行 8 个完整预算学习率候选。运行者只选择 scope；job 编号、噪声参数、优化器和学习率候选都由程序生成，不需要手工填写。

| scope | 全局 job | 可训练参数 | 全部 winner checkpoint 估算 | 四卡峰值估算 |
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
- 数据、冻结产物、压缩日志和 metadata 预留：4.00 GB。

这是协议估算，不是 H100 实测值。`full` 的余量很窄，因此：

- 不得同时运行两个 scope；
- 不得把六个 scope 写进同一个 `RUN_DIR`；
- 不得在服务器保留上一 scope 的副本；
- 不得在服务器上再生成与 scope 同体积的 tar/zip；
- 启动前和运行中由运行者用 `df -B1`、`du -sh` 检查，不使用自动磁盘水位保护。

## 8. 正式运行一个 scope

以下以 `full` 为例。其他 scope 只替换 `SCOPE`。

六个 scope 建议按下列顺序逐个完成；不要跳到 `queue 0 839`，也不要并行启动两个 scope：

```text
full -> last1 -> last2 -> last3 -> last6 -> last12
```

### 8.1 启动四卡任务池

正式任务需要同一节点上的四张 H100。如果 Slurm 或其他调度器已经设置 `CUDA_VISIBLE_DEVICES`，不要覆盖。进入仓库根目录后执行：

```bash
bash scripts/h100.sh scope full
```

脚本自动使用 `/data/260010171/results/nanogpt-v1.5/full` 作为该 scope 的独立结果目录，并检查输入、smoke 状态与 commit 一致性。四个 worker 各占一张卡；某张卡完成一个 job 后，从当前 scope 的共享队列领取下一个 job。同一 job 的 8 个 LR 候选始终顺序执行，不会被拆到八张卡并行。

只有在独立服务器上、且运行者已经获准使用 0、1、2、3 号卡时，才手动指定：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/h100.sh scope full
```

建议在 `tmux` 中运行长任务：

```bash
tmux new -s nanogpt-full
# 在 tmux 内重新激活环境，然后运行 h100.sh scope 命令
```

### 8.2 查看进度

```bash
bash scripts/h100.sh status full
tail -f /data/260010171/results/nanogpt-v1.5/full/launcher/job_0000.log
```

`status` 只读取现有文件并显示任务、磁盘和目录大小，不会改变训练。单个 job 完成的标志是 `search.json` 中 `complete: true`，且包含 8 个候选记录。

### 8.3 中断与恢复

任务被抢占、SSH 断开或发生基础设施故障后，修复环境并使用相同 commit 重新执行同一条命令：

```bash
bash scripts/h100.sh scope full
```

完成候选会复用已有终态；未完成候选从自己的 `latest.pt` 恢复。基础设施故障必须恢复原候选，不能换一个 LR 顶替。不要修改 `metadata.json` 绕过身份检查。

只有在项目维护者明确要求排查某个 job 时，才使用单 job 入口。例如：

```bash
GPU_SLOT=0 bash scripts/run_4gpu.sh node 105
```

普通正式运行不需要手工调用 `run_4gpu.sh`、`node` 或 `queue`。这些都是维护者排查问题时使用的底层入口。

## 9. 完成、传出、校验与清空

这一节必须按顺序操作。结果接收端没有完成校验前，不得清空计算服务器目录。

### 9.1 在服务器封存 scope

任务池正常结束后执行：

```bash
bash scripts/h100.sh finalize full
```

该命令检查 140 个 job 是否全部完成、候选诊断是否已经压缩、checkpoint 是否只属于最终 winner，然后生成 `scope_manifest.json`。生成 manifest 后不要再向该目录写入；若检查失败，按报错恢复缺失任务，再重新 finalize。

### 9.2 直接传到持久结果存储

下面以结果接收机器主动使用 `rsync` 从计算服务器拉取为例。尖括号字段必须替换为实际登录信息和绝对路径；不要在接收端命令中使用只存在于服务器 shell 的 `$RUN_DIR`：

```bash
export REMOTE_USER="替换为服务器用户名"
export REMOTE_HOST="替换为服务器地址"
export REMOTE_SCOPE_DIR="/scratch/替换为项目目录/results/nanogpt-v1.5/full"
export RECEIVED_SCOPE_DIR="/替换为持久结果根目录/full"

mkdir -p "$RECEIVED_SCOPE_DIR"
rsync -av --partial --info=progress2 \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SCOPE_DIR}/" \
  "${RECEIVED_SCOPE_DIR}/"
```

连接中断后重复完全相同的 `rsync` 命令即可续传。如果接收端没有 `rsync`，可以使用学校允许的 SFTP、WinSCP 或对象存储传输方式，但必须保持整个 scope 的目录层级和文件名不变，并完成下一节的 manifest 校验。不要先在计算服务器创建 tar/zip；那会在最紧张的阶段再占一份大型空间。

### 9.3 在结果接收端校验副本

在结果接收机器上克隆同一仓库、安装项目，然后对收到的 scope 目录执行：

```bash
git clone https://github.com/ZZZ-020069/nanogpt-noise-experiments.git
cd nanogpt-noise-experiments
export RUN_COMMIT="替换为正式运行时记录的commit"
git checkout "$RUN_COMMIT"
python -m pip install -e '.[plots]'
python -m nanogpt_noise.scope verify \
  --runs "/持久结果根目录/full"
```

命令成功结束并输出 manifest 内容，才表示文件集合、大小和逐文件 SHA-256 与计算服务器生成的 manifest 一致。校验失败时继续或重新传输，不能清服务器数据。处理其他 scope 时，将示例中的 `full` 同时替换为对应 scope 名称。

### 9.4 清空服务器上的当前 scope

确认以下三件事：接收端 `verify` 成功、接收目录可以读取、计算服务器的 `RUN_DIR` 确实是当前 scope 的独立目录。然后回到计算服务器执行带 scope 专用确认串的清理命令：

```bash
python -m nanogpt_noise.scope cleanup \
  --runs "$RUN_DIR" --scope "$SCOPE" --confirm "DELETE-$SCOPE"
```

清理命令只接受已经 finalize 且自校验通过、目录名与 scope 完全一致的目录；它不会删除 `DATA_DIR`、`ARTIFACT_DIR` 或仓库。删除不可由计算服务器恢复，接收端校验未通过时绝对不要执行。

随后确认空间已经释放：

```bash
test ! -e "$RUN_DIR" && echo "scope removed"
df -B1 "$RUN_BASE"
```

然后设置下一个 `SCOPE` 和新的 `RUN_DIR`，回到第 8 节。

### 9.5 单个 scope 的交接完成标准

只有同时满足以下条件，当前 scope 才算完成交接：

- 计算服务器上的 `finalize` 成功；
- 持久结果存储中存在完整的 scope 目录及 `scope_manifest.json`；
- 接收端 `verify` 成功；
- 正式运行使用的 Git commit 已记录；
- 若出现过基础设施故障，相关调度日志已一并保留；
- 计算服务器副本仅在以上检查全部完成后清理。

最终需要交接六个相互独立且各自校验通过的目录：`full`、`last1`、`last2`、`last3`、`last6`、`last12`。输入数据和冻结产物不属于结果，不需要从服务器重复传回。

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

## 11. 可选：汇总六个已交接 scope

运行者完成算力任务和结果交接即可；除非项目维护者明确要求，不需要负责科学分析。若需要在持久存储上检查六个 scope 能否被统一读取，可保留六个独立目录并执行：

```bash
python -m nanogpt_noise.analyze \
  --runs \
    "/持久结果根目录/full" \
    "/持久结果根目录/last1" \
    "/持久结果根目录/last2" \
    "/持久结果根目录/last3" \
    "/持久结果根目录/last6" \
    "/持久结果根目录/last12" \
  --output "/持久结果根目录/analysis" --plots
```

Smoke、不完整任务和无有效 winner 的节点不会伪装成正式有限结果，也不会补为零。

## 12. Slurm 提交示例

仓库提供单节点四卡模板 [scripts/slurm.sh](scripts/slurm.sh)。必须在仓库根目录提交，并将所有占位符替换为学校实际分区、账号、H100 资源写法和绝对路径：

```bash
export DATA_DIR="/data/260010189/inputs/openwebtext_subset"
export ARTIFACT_DIR="/data/260010189/inputs/frozen-artifacts"
export RUN_BASE="/data/260010171/results/nanogpt-v1.5"
export SCOPE=full
export RUN_DIR="$RUN_BASE/$SCOPE"
mkdir -p "$RUN_DIR"
export SLURM_PARTITION="替换为实际分区"
export SLURM_ACCOUNT="替换为实际账号"
export SLURM_H100_GRES="替换为学校H100资源写法"

sbatch --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" --gres="$SLURM_H100_GRES" \
  --export=ALL scripts/slurm.sh
```

模板默认申请单节点 4 张 GPU、16 个 CPU、64 GB 主存和 24 小时。命令行给出的资源参数会覆盖模板中的同名设置；具体 H100 写法必须向学校集群管理员确认。正式 scope 模式必须保留同一节点四张 GPU。超时后回到同一仓库 commit，用相同 `SCOPE`、`RUN_DIR`、数据和冻结产物重新提交即可恢复。

```bash
squeue -u "$USER"
scancel --signal=TERM --full 实际Slurm作业号
```

Slurm 作业号不是实验 job 编号。

## 13. 常见问题

| 现象 | 处理 |
|---|---|
| `No module named nanogpt_noise` | 在仓库根目录激活正确环境并运行 `python -m pip install -e '.[plots]'` |
| `code worktree is not clean` | 不要带着本地修改运行正式实验；保存必要内容后恢复到约定 commit 的干净工作区 |
| `smoke commit marker is missing` | 先在一张 H100 上运行 `bash scripts/h100.sh smoke`，不要跳过实机验收 |
| smoke commit 与当前 commit 不同 | 切回通过 smoke 的 commit，或在当前 commit 重新执行 smoke；不要混用两个代码版本 |
| `CUDA unavailable` | 确认进入 GPU 节点、安装 CUDA 版 PyTorch，检查驱动兼容性 |
| 输入校验 ID 不同或 `hash mismatch` | 停止运行，重新取得项目提供方给出的完整数据和冻结产物；不要自行重建 |
| `df -B1` 显示可用空间不足 200000000000 bytes | 不要启动 `full`；改用容量更大的项目盘或先清理已经校验交接的旧 scope |
| `scope requires exactly four ... tokens` | `CUDA_VISIBLE_DEVICES` 必须恰好列出调度器分配的四张卡 |
| `belongs to scope ...` | 当前 `RUN_DIR` 已属于另一个 scope；为新 scope 使用新目录 |
| `scope jobs incomplete` | 查看报错列出的缺失 job，使用原目录恢复任务池 |
| `diagnostics are not compacted` | 重新运行同一 scope；终态候选会复用结果并补做压缩 |
| manifest `hash mismatch` / `size mismatch` | 接收端副本不完整或损坏，继续传输，不要清服务器数据 |
| `.lock` 已存在 | 先确认没有活跃进程；只有确认是硬杀遗留锁后才移除锁 |
| smoke OOM | 停止正式启动并保存完整错误和 `nvidia-smi`，不要擅改协议参数 |
| GitHub Actions 失败邮件 | 这是代码 CPU 测试状态，不是服务器训练状态；查看对应 commit 的 Actions 日志 |

## 14. 设计与代码入口

本 README 是运行与结果交接手册。仓库内的 [实验设计发布快照](docs/nanogpt_experiment_design.md) 说明固定的科学参数；运行者不需要访问仓库之外的文档，也不应自行改动协议。正式结果由 Git commit、候选 metadata、输入身份和 scope manifest 共同标识。

- [scripts/h100.sh](scripts/h100.sh)：运行者的统一入口，提供 `check`、`smoke`、`scope`、`status` 和 `finalize`；
- [scripts/run_4gpu.sh](scripts/run_4gpu.sh)：由统一入口调用的底层 GPU 启动器，仅供维护和排障；
- `nanogpt_noise/pool.py`：四个独立单卡 worker 的动态调度；
- `nanogpt_noise/node.py`：一个 job 内 8 个 LR 候选的顺序搜索；
- `nanogpt_noise/train.py`：单候选训练、状态与恢复；
- `nanogpt_noise/storage.py`：诊断压缩与 winner-only checkpoint；
- `nanogpt_noise/scope.py`：scope 隔离、估算、finalize、verify 和受保护清理；
- `nanogpt_noise/analyze.py`：跨多个已下载 scope 的统一汇总；
- [实现验证记录](docs/VALIDATION.md)：已完成检查和目标硬件待验项目。

统一入口不会登录服务器、申请 GPU、激活 Conda/虚拟环境、安装 CUDA 版 PyTorch、执行 `git pull`、传出结果、删除 scope 或因磁盘空间不足自动停止训练。这些操作跨越了不同资源会话或具有破坏性，必须由运行者按本文明确完成。

模型结构参考 nanoGPT，并保留[原 MIT 声明](third_party/nanogpt-LICENSE)。本仓库不依赖 WandB。底层仍保留 DDP 回归测试，但正式 H100 路径是一张卡运行一个独立任务。
