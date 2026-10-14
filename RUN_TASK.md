# 运行任务说明

需要完成的事情是：在学校服务器上先用一张 H100 验证单候选可以运行，再用四张 H100 逐个完成六个 scope，并在每个 scope 完成后传出和校验结果。

模型、噪声、优化器、学习率搜索和训练预算都已固定。正常运行只使用统一入口 `scripts/h100.sh`，不需要手工填写数据路径、任务编号或学习率。

## 1. 第一次运行前

服务器上的固定位置为：

```text
代码：/data/260010171/nanogpt-noise-experiments
输入：/data/260010189/inputs
结果：/data/260010171/results/nanogpt-v1.5
```

输入目录下必须已有：

```text
/data/260010189/inputs/
├── openwebtext_subset/
│   ├── train.bin
│   ├── val.bin
│   └── meta.json
└── frozen-artifacts/
    ├── initial.pt
    ├── windows.pt
    └── identity.json
```

进入代码仓库，拉取最新版本，然后激活服务器上的 CUDA PyTorch 环境并安装项目：

```bash
cd /data/260010171/nanogpt-noise-experiments
git pull --ff-only

# 先按学校服务器的实际方式激活 Python 3.11+、CUDA PyTorch 环境
python -m pip install -e '.[plots]'

bash scripts/h100.sh --help
```

正式 smoke 开始后固定代码版本，不再执行 `git pull`，也不要编辑仓库文件。

如果希望在申请 GPU 前先检查输入和 CPU 测试，可以执行：

```bash
bash scripts/h100.sh check
```

这一步是可选的，因为下一节的 `smoke` 会自动重复全部检查。

## 2. 一张 H100：运行 smoke test

先申请只含一张 H100 的计算会话。调度器通常会自动设置 `CUDA_VISIBLE_DEVICES`，不要覆盖。进入代码仓库后只需执行：

```bash
cd /data/260010171/nanogpt-noise-experiments
bash scripts/h100.sh smoke
```

脚本会自动完成：

1. 检查代码工作区与提交号；
2. 检查六个输入文件、SHA-256 和固定输入身份；
3. 运行完整 CPU 测试；
4. 显示磁盘占用；
5. 调用单卡 8 步 smoke；
6. 验证 `status=completed`、`mode=smoke`、`step=8`；
7. 记录 smoke 使用的代码提交。

看到 `SMOKE PASSED` 才能进入正式运行。如果是独立服务器而不是调度器，并且明确获准使用某一张卡，例如 GPU 0，才使用：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/h100.sh smoke
```

## 3. 四张 H100：运行一个 scope

六个 scope 必须按以下顺序逐个完成，不要同时运行两个：

```text
full -> last1 -> last2 -> last3 -> last6 -> last12
```

申请同一节点上的四张 H100。调度器必须让当前进程恰好看到四张卡。第一个 scope 只需执行：

```bash
cd /data/260010171/nanogpt-noise-experiments
bash scripts/h100.sh scope full
```

脚本会验证输入、smoke 状态和代码提交，然后自动使用四张卡运行 `full` 的 140 个 job。每张卡一次处理一个独立 job；每个 job 内的 8 个学习率候选顺序完成。

如果是独立服务器，并且明确获准使用 0、1、2、3 号卡，才使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/h100.sh scope full
```

任务被抢占、SSH 断开或基础设施错误修复后，使用相同代码和目录重新执行同一条命令即可恢复：

```bash
bash scripts/h100.sh scope full
```

不要通过修改学习率、batch、训练步数、噪声、模型或梯度裁剪来绕过错误。

## 4. 查看状态

另开一个终端，在相同 Python 环境中执行：

```bash
cd /data/260010171/nanogpt-noise-experiments
bash scripts/h100.sh status full
```

如需查看某个 launcher 日志：

```bash
tail -f /data/260010171/results/nanogpt-v1.5/full/launcher/job_0000.log
```

脚本只显示磁盘情况，不会自动按剩余空间中止训练。运行者仍需关注 `df` 和 `du` 输出。

## 5. Scope 完成后

任务池正常结束后封存当前 scope：

```bash
cd /data/260010171/nanogpt-noise-experiments
bash scripts/h100.sh finalize full
```

然后依次完成：

1. 将整个 `/data/260010171/results/nanogpt-v1.5/full` 原样传到项目方指定的持久存储；
2. 不要在服务器上另建同体积的 tar/zip；
3. 在接收端按照 [README 的校验步骤](README.md#93-在结果接收端校验副本)验证 `scope_manifest.json`；
4. 只有接收端校验成功后，才能清理服务器上的当前 scope；
5. 把命令中的 `full` 改为 `last1`，然后依次完成其余 scope。

清理仍然是单独的受保护操作，不由统一脚本自动执行：

```bash
python -m nanogpt_noise.scope cleanup \
  --runs /data/260010171/results/nanogpt-v1.5/full \
  --scope full --confirm DELETE-full
```

## 6. 统一脚本不会代替的事情

`scripts/h100.sh` 不会自动：

- 登录服务器或申请 GPU；
- 激活 Conda/虚拟环境或安装 CUDA PyTorch；
- 执行 `git pull`；
- 从服务器传出结果；
- 删除任何 scope 结果；
- 因磁盘空间低而自动停止任务。

输入缺失、身份不一致、CPU 测试失败、smoke 失败、代码提交变化或可见 GPU 数量不正确时，脚本会拒绝继续。更详细的输入下载、环境、Slurm、恢复、传输和故障处理说明见 [README.md](README.md)。
