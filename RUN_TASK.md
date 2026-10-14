# H100 运行任务

正常情况下只需要运行两个脚本：一个完成 smoke test，一个完整运行并封存一个正式 scope。模型、噪声、优化器、学习率搜索、任务编号和结果目录都已经写入程序，不需要手工设置。

## 一次性准备

代码和输入应位于：

```text
代码：/data/260010171/nanogpt-noise-experiments
输入：/data/260010189/inputs
结果：/data/260010171/results/nanogpt-v1.5
```

第一次使用这个版本时，在已经激活 CUDA PyTorch 环境的终端执行一次：

```bash
cd /data/260010171/nanogpt-noise-experiments
git pull --ff-only
python -m pip install -e '.[plots]'
```

开始 smoke 后不要再更新或修改代码。以上准备不需要在每个实验任务前重复。

## 1. 一张 H100：运行 smoke

申请一张 H100，在仓库目录执行一条命令：

```bash
bash scripts/run_smoke.sh
```

脚本会自动检查代码和六个输入文件、校验数据身份、运行全部 CPU 测试、检查单卡环境，并完成 8 步 smoke。终端显示 `SMOKE PASSED` 即通过。

## 2. 四张 H100：运行正式实验

申请同一节点上的四张 H100，在相同仓库和 Python 环境执行：

```bash
bash scripts/run_formal.sh
```

不提供参数时默认运行 `full`。这一条命令会自动：

1. 复查代码、输入和 smoke 身份；
2. 让四张卡分别领取 `full` 的 140 个 job；
3. 在每个 job 内依次完成 8 个完整预算学习率候选；
4. 在中断后复用已完成结果并从 checkpoint 恢复未完成候选；
5. 全部完成后自动检查结果并生成 `scope_manifest.json`。

任务中断或基础设施故障修复后，重新运行完全相同的命令即可恢复。无需另外执行 `check`、`status` 或 `finalize`。

## 3. 一个 scope 完成以后

看到 `FORMAL SCOPE FINALIZED` 后，结果位于：

```text
/data/260010171/results/nanogpt-v1.5/full
```

此时需要人工完成三件事：

1. 把整个 `full` 目录传到项目方的持久存储；
2. 在接收端校验 `scope_manifest.json`；
3. 校验成功后再清理服务器上的 `full` 目录。

然后运行下一个 scope，例如：

```bash
bash scripts/run_formal.sh last1
```

依次运行：

```text
full -> last1 -> last2 -> last3 -> last6 -> last12
```

六个 scope 不能由脚本在服务器上自动连续运行，因为 `full` 单独就可能占用约 195.58 GB；每个 scope 完成后必须先传出、校验和清理，才能安全开始下一个 scope。脚本不会自动删除任何结果。

更详细的传输、校验、清理和故障排查方法见 [README.md](README.md)。
