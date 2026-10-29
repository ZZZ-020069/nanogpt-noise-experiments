# 当前任务：先完成 last1

**v1.8：所有 scope 统一为 35 噪声节点、四优化器、五初始＋三追加学习率、每候选完整 2B tokens。当前只运行 last1，完成后先分析，不接着启动其他 scope。**

last1 从共同未训练初始化开始，只训练第 12 个 Transformer block（7,079,424 参数）；embedding、前 11 个 block、final LayerNorm 和共享输出头均冻结。每候选 3815 steps，实际 2,000,158,720 tokens，100M warmup 后余弦衰减至峰值的 10%。共有 140 jobs、最多 1120 条轨迹。四 H100 的 10–15 天是未实测的条件估算，不是工期保证。

## 1. 更新代码和选择目录

确认旧任务已停止，然后更新一次；smoke 和正式任务之间固定 commit：

```bash
cd /data/260010171/nanogpt-noise-experiments
git -c safe.directory=/data/260010171/nanogpt-noise-experiments pull --ff-only
export INPUT_ROOT=/data/260010171/inputs-v1.8
export RUN_BASE=/data/260010171/results/nanogpt-v1.8
```

默认路径就是上述值。使用其他服务器时在每次任务中设置相同的 INPUT_ROOT、RUN_BASE；已有 CUDA 环境可指定 `PYTHON=/绝对路径/bin/python`。结果目录应在仓库外，避免日志导致工作区不干净。

## 2. 输入：统一在服务器重新制备

使用新的空 INPUT_ROOT，直接运行：

```bash
bash scripts/run_acp.sh inputs
```

服务器从固定 OpenWebText revision 流式读取，生成 1B 训练/10M 验证语料、共同未训练初始化和完整 2B 调度窗口，然后登记哈希与身份。无需本机 SCP 上传，不再提供旧输入迁移入口。旧目录及历史结果保持原样；不要把旧 checkpoint、smoke 或 pilot 放入新结果目录。

首次制备可能需要数小时；输入文件约 2.53 GB，环境与下载缓存另计，可用 HF_HOME、ENV_PREFIX 指定位置。目标目录必须为空或由本版本制备标记管理；中断后可重试，若提示部分最终产物不完整，则保留原目录，改用新空目录。制备结束后保持输入固定；重复运行 inputs 只校验，不重新抽样或初始化。所有候选及未来其他 scope 复用同一 input ID。

## 3. 一张 H100：冒烟

```bash
bash scripts/run_acp.sh smoke last1
```

依次检查四优化器，各运行节点 (0.03,0.01)、中心 LR、8 steps，包含初末评估、冻结参数校验和 checkpoint；这是新 2B 调度的短前缀，不是正式结果。显示 `SMOKE PASSED` 才能继续。CPU 测试也会运行，环境安装与输入校验耗时另计。

已有兼容环境的等价入口：`bash scripts/run_last1_smoke.sh`。不自动安装环境。

## 4. 四张 H100：提交训练任务

完成数据制备和冒烟后，在 ACP 的启动命令中提交以下内容。第一次启动和平台停止后的再次启动，使用完全相同的命令：

```bash
export INPUT_ROOT=/data/260010171/inputs-v1.8
export RUN_BASE=/data/260010171/results/nanogpt-v1.8-platform
exec bash /data/260010171/nanogpt-noise-experiments/scripts/run_acp.sh formal last1
```

smoke 也必须使用这里相同的 INPUT_ROOT、RUN_BASE 和提交版本；若之前已在旧目录冒烟，先在新 RUN_BASE 重新冒烟。环境可通过相同的 PYTHON 或 ENV_PREFIX 指定。输入、结果和需复用的环境必须放在平台持久化存储中；每次启动均挂载同一个目录。不要在每次启动命令里自动 git pull、重新制备数据或更换结果目录。

平台配置为一个实例/worker分配四张H100，仓库内部启动四个独立单卡训练进程；不要再由平台复制成四个四卡实例。四卡领取jobs140–279，各job顺序完成八个候选，仅运行last1。完成后自动finalize并停止；已finalize的目录再次启动只校验交付文件，不重新训练。

这里使用 exec，使启动shell被仓库入口替换，让平台停止信号能够到达仓库的监督进程。截图中的外部 `/data/260010171/run_v18_ready.sh` 不在仓库，尚未核对；如继续使用它，应让最后一条训练启动命令也通过exec转入上述仓库入口，并保持相同路径和代码版本。

## 5. 平台停止与再次启动

学长需要临时用卡时，直接使用平台的停止任务操作。无需在运行中另开终端执行命令。

- 如果平台先发送TERM/INT并给予足够保存时间，监督脚本转发信号，训练完成当前step并保存，等待全部训练进程退出后结束；停止时不执行finalize。
- 如果平台直接强杀、宽限期太短或机器故障，只能从最近已持久保存的checkpoint恢复，通常最多重算活跃候选最近约191步/100M tokens。实际宽限期和保存耗时仍须目标平台验收。
- 重新获得算力后，启动原任务或重新提交第4节相同命令。已完成候选直接复用，未完成候选读取断点；不需要再次生成输入，也不需要手工把formal换成resume。

Linux使用操作系统flock独占锁。新版本进程被强杀后，内核锁自动释放；重启先取得同一 `.lease` 的独占权，再回收本版本遗留 `.lock` 标记。活跃写入者仍会阻止第二个任务接管。**不要删除 `.lease` 文件**；它是持久的锁载体。文件系统不支持flock会明确报错，不降级为不安全的自动删锁。旧版文本锁或来源不明的锁仍需确认原任务彻底停止后人工处理。

输入、checkpoint、代码、PyTorch/CUDA构建、micro-batch和RUN_BASE必须与原轨迹一致。正式开始后固定commit，禁止为恢复修改metadata、预算或学习率。之前版本已丢失的checkpoint不能由本修复恢复。停止75退出码属于可恢复中断，可能被平台显示为失败；如启用自动重试，人工停止应由平台保持停止状态，不要把所有非零退出无条件无限重试。

可选：有终端或能访问同一结果目录时，仍可执行 `bash scripts/run_acp.sh pause last1`，等待 `SCOPE QUIESCENT` 后交出GPU；这种显式文件暂停需用 `bash scripts/run_acp.sh resume last1` 清除暂停标记。它与平台停止流程分别使用，平台停止不会自动写入pause.request。

## 6. 下载并分析后再决定下一步

结果在 `/data/260010171/results/nanogpt-v1.8/last1`。下载整个目录并核验 `scope_manifest.json`；本机接收目录固定放 D 盘，例如：

```text
D:\港中深RA\VSCode下载\New\gpu-results\nanogpt-v1.8\last1
```

不要先下载到 C 盘，也不要在服务器另建一份大型压缩包。校验成功前保留服务器结果；目前没有自动清理或自动开展下一 scope 的安排。

更详细的存储、恢复、Slurm 与结果校验见 [README.md](README.md)。


升级本次断点修复时，如果此前已经运行过旧提交的 smoke，请保留旧结果并先选择新结果目录，再冒烟；之后所有启动、暂停、继续命令均使用这个 RUN_BASE：

```bash
export RUN_BASE=/data/260010171/results/nanogpt-v1.8-platform
bash scripts/run_acp.sh smoke last1
```

已登记的 INPUT_ROOT 不变，无需重新制备语料；正式开始后固定提交，不能靠切换 RUN_BASE 把旧轨迹恢复成新代码轨迹。
