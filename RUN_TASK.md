# H100 运行任务

首次依次运行三个命令：一次性生成并登记输入、完成 smoke test、完整运行并封存一个正式 scope。模型、数据处理规则、噪声、优化器、学习率搜索、任务编号和结果目录都已经写入程序，不需要手工设置。

## 一次性准备

代码和输入应位于：

```text
代码：/data/260010171/nanogpt-noise-experiments
输入：/data/260010171/inputs
结果：/data/260010171/results/nanogpt-v1.6
```

停止旧任务后，第一次使用这个版本时执行一次更新：

```bash
cd /data/260010171/nanogpt-noise-experiments &&
git -c safe.directory=/data/260010171/nanogpt-noise-experiments pull --ff-only
```

这里的 `-c safe.directory=...` 只用于让旧版本成功拉取本次修复，不会修改全局 Git 配置。更新后，ACP 脚本会自动为当前进程处理平台用户与仓库所有者不一致的问题；无需再执行 `git config --global`。输入生成后不要再更新或修改代码。以上准备不需要在每个实验任务前重复。

## 1. 一次性生成输入

在正式 smoke 之前执行一次：

```bash
cd /data/260010171/nanogpt-noise-experiments && bash scripts/run_acp.sh inputs
```

脚本会从固定 OpenWebText revision 生成一份包含 10M validation tokens 和 1B train tokens 的**固定数据语料**，再生成统一初始化权重和固定窗口，最后登记六文件 SHA-256 与本轮唯一 input ID。这里的 1B 是 `train.bin` 中保存的数据量，不是单条训练轨迹的预算；每个 LR 候选仍完整训练约 2.6B tokens（4960 steps，实际 2,600,468,480 tokens），训练期间会从这份固定语料重复采样。过程可能运行数小时，需要服务器访问 Hugging Face 和 Python 软件源。看到 `INPUT BUNDLE REGISTERED` 才算成功；此后六个 scope 共用 `/data/260010171/inputs`，不得重新生成。

## 2. 一张 H100：运行 smoke

申请一张 H100，ACP 启动命令填写：

```bash
cd /data/260010171/nanogpt-noise-experiments && bash scripts/run_acp.sh smoke
```

脚本会先选择兼容环境；若平台只有 Python 3.9，则通过 Conda 创建独立 Python 3.11 + CUDA PyTorch 环境，不修改 base。首次安装需要网络和系统盘额外空间，后续自动复用；任何安装失败都会停止，不会继续训练。无需在启动命令中另加 pip install。

然后自动检查代码和六个输入文件、校验数据身份、运行全部 CPU 测试、检查单卡环境，并完成 8 步 smoke。终端显示 `SMOKE PASSED` 即通过。

## 3. 四张 H100：运行正式实验

smoke 通过后，申请同一节点上的四张 H100，ACP 启动命令填写：

```bash
cd /data/260010171/nanogpt-noise-experiments && bash scripts/run_acp.sh formal full
```

不提供参数时默认运行 `full`。这一条命令会自动：

1. 复查代码、输入和 smoke 身份；
2. 让四张卡分别领取 `full` 的 140 个 job；
3. 在每个 job 内依次完成 8 个完整预算学习率候选；
4. 在中断后复用已完成结果并从 checkpoint 恢复未完成候选；
5. 全部完成后自动检查结果并生成 `scope_manifest.json`。

任务中断或基础设施故障修复后，重新运行完全相同的命令即可恢复。无需另外执行 `check`、`status` 或 `finalize`。

## 4. 一个 scope 完成以后

看到 `FORMAL SCOPE FINALIZED` 后，结果位于：

```text
/data/260010171/results/nanogpt-v1.6/full
```

此时需要人工完成三件事：

1. 把整个 `full` 目录传到项目方的持久存储；
2. 在接收端校验 `scope_manifest.json`；
3. 校验成功后再清理服务器上的 `full` 目录。

然后运行下一个 scope，例如：

```bash
bash scripts/run_acp.sh formal last1
```

依次运行：

```text
full -> last1 -> last2 -> last3 -> last6 -> last12
```

六个 scope 不能由脚本在服务器上自动连续运行，因为 `full` 单独就可能占用约 195.58 GB；每个 scope 完成后必须先传出、校验和清理，才能安全开始下一个 scope。脚本不会自动删除任何结果。

环境默认位于 `$HOME/.local/share/nanogpt-noise/env-py311`；环境和 Hugging Face 临时缓存额外占用空间，不计入结果盘约 195.58 GB 的估算。确保系统盘有空间；如需指定其他持久盘，设置 `ENV_PREFIX` 和 `HF_HOME`。输入生成、smoke 与正式运行之间不要更新代码。

更详细的传输、校验、清理和故障排查方法见 [README.md](README.md)。
