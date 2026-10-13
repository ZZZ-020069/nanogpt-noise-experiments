# 运行任务说明

需要做的事情只有一件：

> 在一台能同时使用四张 NVIDIA H100 的 Linux 计算节点上，按仓库已经固定好的配置运行 NanoGPT 实验，并把每批结果完整传回。

不需要选择模型参数、噪声、优化器或学习率，也不需要分析实验结果。程序会自动生成任务并选择每个任务的最佳学习率。

## 1. 开始前请向项目提供方取得

1. 本仓库的 GitHub 访问权限；
2. 数据目录，包含 `train.bin`、`val.bin`、`meta.json`；
3. 冻结产物目录，包含 `initial.pt`、`windows.pt`、`identity.json`；
4. 最终结果应传到哪里，以及对应的传输方式；
5. 学校集群实际使用的 Slurm partition、account 和 H100 资源写法。

GitHub 仓库不包含上述大型输入文件。如果文件不全，请先联系项目提供方，不要自行重新生成。

## 2. 下载代码并安装

```bash
git clone https://github.com/ZZZ-020069/nanogpt-noise-experiments.git
cd nanogpt-noise-experiments

# 先按学校服务器的方式进入 Python 3.11+、CUDA PyTorch 环境
python -m pip install -e '.[plots]'
```

正式运行开始后不要更新或修改代码。记录本次使用的提交号：

```bash
git status --short       # 正常应无输出
git rev-parse HEAD
```

## 3. 设置三个目录

下面路径只是示例，必须替换为服务器上的真实绝对路径：

```bash
export DATA_DIR="/scratch/<项目目录>/inputs/openwebtext_subset"
export ARTIFACT_DIR="/scratch/<项目目录>/inputs/frozen-artifacts"
export RUN_BASE="/scratch/<项目目录>/results/nanogpt-v1.5"
mkdir -p "$RUN_BASE"
```

- `DATA_DIR` 和 `ARTIFACT_DIR` 是项目提供方传来的输入；
- `RUN_BASE` 是你在服务器大容量磁盘上新建的空结果目录；
- 不要把结果写到容量未经确认的个人 home 目录。

## 4. 先校验输入和代码

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

## 5. 先用一张 H100 做 smoke test

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

## 6. 四张 H100 每次只运行一个 scope

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

## 7. 查看进度

```bash
tail -f "$RUN_DIR/launcher/job_0000.log"
du -sh "$RUN_DIR"
df -B1 "$RUN_DIR"
```

`full` 开始前至少应有 `200000000000` bytes 可用空间。单个 job 的 `search.json` 显示 `complete: true` 且包含 8 个候选，才表示该 job 完成。

## 8. 每个 scope 完成后的操作

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
5. 再把 `SCOPE` 改成下一个值，重复第 6–8 节。

清理命令具有删除作用，必须在接收端验证成功后才运行：

```bash
python -m nanogpt_noise.scope cleanup \
  --runs "$RUN_DIR" --scope "$SCOPE" --confirm "DELETE-$SCOPE"
```

## 9. 最终需要交付

- 六个分别通过校验的结果目录：`full`、`last1`、`last2`、`last3`、`last6`、`last12`；
- 每个目录内的 `scope_manifest.json`；
- 正式运行使用的 Git commit；
- 如果出现过集群、CUDA、存储或调度故障，相应的完整日志。

单个候选出现 `numerical_failure` 是实验可能产生的有效结果，程序会记录并继续，不要手工改参数。如果任务池本身退出，或遇到 OOM、输入哈希不一致、测试失败、集群/存储故障，以及不确定是否可以删除结果时，请保留日志并联系项目提供方。更详细的环境、Slurm、传输、校验和错误处理说明见 [README.md](README.md)。
