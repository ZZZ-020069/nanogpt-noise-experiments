# 运行任务说明

需要做的事情只有一件：

> 在一台能同时使用四张 NVIDIA H100 的 Linux 计算节点上，按仓库已经固定好的配置运行 NanoGPT 实验，并把每批结果完整传回。

不需要选择模型参数、噪声、优化器或学习率，也不需要分析实验结果。程序会自动生成任务并选择每个任务的最佳学习率。

## 1. 开始前请向项目提供方取得

1. 本代码仓库的 GitHub 访问权限；
2. 私有输入仓库 `ZZZ-020069/nanogpt-noise-inputs` 的 Collaborator 邀请，并使用自己的 GitHub 账号接受；
3. 最终结果应传到哪里，以及对应的传输方式；
4. 学校集群实际使用的 Slurm partition、account 和 H100 资源写法。

六个固定输入文件通过私有输入仓库的 Git LFS 交付。如果无权访问、Git LFS 不可用或文件校验失败，请先联系项目提供方，不要自行重新生成。

## 2. 下载代码并安装

```bash
cd /data/260010171
git clone https://github.com/ZZZ-020069/nanogpt-noise-experiments.git
cd /data/260010171/nanogpt-noise-experiments

# 先按学校服务器的方式进入 Python 3.11+、CUDA PyTorch 环境
python -m pip install -e '.[plots]'
```

正式运行开始后不要更新或修改代码。记录本次使用的提交号：

```bash
git status --short       # 正常应无输出
git rev-parse HEAD
```

## 3. 下载固定输入文件

先确认服务器安装了 Git LFS：

```bash
git lfs version
```

如果命令不存在，先按服务器管理员提供的方法安装 Git LFS。不要把 GitHub 账号密码或个人访问令牌发给其他人。

因为已创建 `/data/260010171/inputs/openwebtext_subset/`，所以直接把其父目录初始化为输入仓库工作区，不要再增加一层目录：

```bash
cd /data/260010171/inputs
git init
git remote add origin https://github.com/ZZZ-020069/nanogpt-noise-inputs.git
git lfs install
git fetch origin main
git checkout -b main --track origin/main
git lfs pull
```

私有仓库认证必须使用运行者自己的 GitHub 账号；GitHub 的密码提示不接受账号登录密码。若该目录以前已经初始化完成，则执行以下更新命令，不要重复运行 `git init` 和 `git remote add`：

```bash
cd /data/260010171/inputs
git pull --ff-only
git lfs pull
```

下载后执行：

```bash
cd /data/260010171/inputs
git rev-parse HEAD
git lfs ls-files
sha256sum -c SHA256SUMS
```

当前输入提交应为：

```text
1b3643ac44ff2b76ed471cc130a04c2c9c52fb01
```

`git lfs ls-files` 应列出四个大文件，`sha256sum` 的六项均须为 `OK`。最终目录必须为：

```text
/data/260010171/inputs/
├── openwebtext_subset/
│   ├── train.bin
│   ├── val.bin
│   └── meta.json
└── frozen-artifacts/
    ├── initial.pt
    ├── windows.pt
    └── identity.json
```

## 4. 设置三个目录

本项目当前服务器使用以下绝对路径：

```bash
cd /data/260010171/nanogpt-noise-experiments
export DATA_DIR="/data/260010171/inputs/openwebtext_subset"
export ARTIFACT_DIR="/data/260010171/inputs/frozen-artifacts"
export RUN_BASE="/data/260010171/results/nanogpt-v1.5"
mkdir -p "$RUN_BASE"
```

- `DATA_DIR` 和 `ARTIFACT_DIR` 是刚从私有输入仓库下载并校验的输入；
- `RUN_BASE` 是你在服务器大容量磁盘上新建的空结果目录；
- 不要把结果写到容量未经确认的个人 home 目录。

## 5. 先校验输入和代码

```bash
python -m nanogpt_noise.data \
  --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" --verify

python -m unittest discover -s tests -v
```

第一条命令必须输出：

```text
9cabd9c54c6d717354f8f6750715f9280c5502f0a85a9032806b9c7956cbd76e
```

测试应为 41 项并以 `OK` 结束。任一检查失败都不要开始正式训练。

## 6. 先用一张 H100 做 smoke test

在只分配一张 H100 的计算节点上执行：

```bash
export RUN_DIR="$RUN_BASE/smoke"
mkdir -p "$RUN_DIR"
bash scripts/run_4gpu.sh smoke

python -m json.tool \
  "$RUN_DIR/smoke/1gpu/full-muon-node10/status.json"
```

只有输出中同时出现以下三项才通过：

```text
status: completed
mode: smoke
step: 8
```

Smoke test 只是检查 H100 能否承载任务，不是正式实验结果。

## 7. 四张 H100 每次只运行一个 scope

六个 scope 按下面顺序分别运行：

```text
full -> last1 -> last2 -> last3 -> last6 -> last12
```

每个 scope 有 140 个 job。四张卡分别运行四个独立 job；一个 job 内的 8 个学习率候选由程序依次完成。不要同时启动两个 scope。

以下以 `full` 为例：

```bash
export SCOPE=full
export RUN_DIR="$RUN_BASE/$SCOPE"
mkdir -p "$RUN_DIR"

# 调度器通常会自动设置 CUDA_VISIBLE_DEVICES，不要覆盖。
# 只有在独立服务器上且已获准使用 0、1、2、3 号卡时，才手动执行：
# export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/run_4gpu.sh scope "$SCOPE"
```

如果任务因抢占、SSH 断开或基础设施错误停止，修复环境后保持相同代码、变量和结果目录，重新执行同一条命令即可恢复：

```bash
bash scripts/run_4gpu.sh scope "$SCOPE"
```

不要通过修改学习率、batch、训练步数、噪声、模型或梯度裁剪来处理错误。

## 8. 查看进度

```bash
tail -f "$RUN_DIR/launcher/job_0000.log"
du -sh "$RUN_DIR"
df -B1 "$RUN_DIR"
```

`full` 开始前至少应有 `200000000000` bytes 可用空间。单个 job 的 `search.json` 显示 `complete: true` 且包含 8 个候选，才表示该 job 完成。

## 9. 每个 scope 完成后的操作

先在服务器封存并生成文件校验清单：

```bash
python -m nanogpt_noise.scope finalize \
  --runs "$RUN_DIR" --scope "$SCOPE"
```

然后完成以下步骤：

1. 将整个 `$RUN_DIR` 原样传到项目方指定的持久存储；
2. 不要在服务器上另建同体积的 tar/zip；
3. 在接收端按照 [README 的校验步骤](README.md#93-在结果接收端校验副本)运行 `nanogpt_noise.scope verify`；
4. 只有接收端校验成功后，才能清理服务器上的当前 scope；
5. 再把 `SCOPE` 改成下一个值，重复第 7–9 节。

清理命令具有删除作用，必须在接收端验证成功后才运行：

```bash
python -m nanogpt_noise.scope cleanup \
  --runs "$RUN_DIR" --scope "$SCOPE" --confirm "DELETE-$SCOPE"
```

## 10. 最终需要交付

- 六个分别通过校验的结果目录：`full`、`last1`、`last2`、`last3`、`last6`、`last12`；
- 每个目录内的 `scope_manifest.json`；
- 正式运行使用的 Git commit；
- 如果出现过集群、CUDA、存储或调度故障，相应的完整日志。

单个候选出现 `numerical_failure` 是实验可能产生的有效结果，程序会记录并继续，不要手工改参数。如果任务池本身退出，或遇到 OOM、输入哈希不一致、测试失败、集群/存储故障，以及不确定是否可以删除结果时，请保留日志并联系项目提供方。更详细的环境、Slurm、传输、校验和错误处理说明见 [README.md](README.md)。
