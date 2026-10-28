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

## 4. 四张 H100：启动或恢复 last1

```bash
bash scripts/run_acp.sh formal last1
```

四卡分别领取 jobs 140–279，每个 job 内顺序完成八个候选；只运行 last1。省略 scope 也默认 last1。完成后自动 finalize 并输出 `FORMAL SCOPE FINALIZED`，不会运行下一个 scope。

已有兼容环境的等价入口：`bash scripts/run_last1.sh`。中断后使用相同 commit、输入、软件环境、micro-batch 和输出目录重新运行同一命令。micro 默认 4；全局 batch 始终为 512×1024 tokens。禁止为恢复而改 metadata、预算或学习率。

## 5. 下载并分析后再决定下一步

结果在 `/data/260010171/results/nanogpt-v1.8/last1`。下载整个目录并核验 `scope_manifest.json`；本机接收目录固定放 D 盘，例如：

```text
D:\港中深RA\VSCode下载\New\gpu-results\nanogpt-v1.8\last1
```

不要先下载到 C 盘，也不要在服务器另建一份大型压缩包。校验成功前保留服务器结果；目前没有自动清理或自动开展下一 scope 的安排。

更详细的存储、恢复、Slurm 与结果校验见 [README.md](README.md)。
