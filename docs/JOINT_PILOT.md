> v1.8 可选探索工具；当前任务请使用 RUN_TASK.md 的 last1 入口。2026-10-28 已完成的旧 pilot 使用 2.6B 调度，此处新入口不能恢复旧结果。

# 双 RTX 联合初筛：启动、恢复与交付

这是用户于 2026-10-28 授权实施的短 pilot。科学规格见工作区唯一主设计的第 12.6 节；本页是操作手册。正式 full 的 30 点、2B 和五候选缩减仍未确认。

程序复用现有训练器，以短测接口运行固定前缀。底层 `metadata/status` 的模式为 `smoke`，config 中另有 `pilot` 身份；顶层 manifest 和 summary 的模式为 `pilot_prefix`、`formal_result=false`。它不会进入正式八候选搜索。

## 1. 使用已有远程环境和输入

目标目录是已有仓库：

```text
/experiments/xuyangzhang/nanogpt-noise-experiments
```

将发布包里的以下三个新文件解压到仓库，保留层级：

```text
scripts/joint_pilot.py
scripts/run_joint_pilot.sh
docs/JOINT_PILOT.md
```

不需要覆盖现有训练代码、重新安装环境、重新下载或制备数据。发布包不含数据和训练依赖，要求远程仓库为已有 v1.6 实现，且有此前测速成功的 Python/CUDA 环境和冻结输入。

如果已把 ZIP 传到用户目录，可在远程终端解压：

```bash
cd /experiments/xuyangzhang/nanogpt-noise-experiments
/experiments/xuyangzhang/venvs/quadratic/bin/python -m zipfile -e \
  /experiments/xuyangzhang/joint-pilot-v1.8-2b.zip .
```

此 pilot 完整保留 24 条轨迹的终点模型及优化器状态；完成后约数十 GB，建议结果盘准备至少 **50 GB 可用空间**，包括并发原子写盘余量。这是预留建议，实际文件大小由目标环境决定。程序不自动删除 checkpoint，也不修改输入。

## 2. 只检查计划，不使用 GPU

```bash
cd /experiments/xuyangzhang/nanogpt-noise-experiments
INPUT_ROOT=/experiments/xuyangzhang/nanogpt-inputs \
PYTHON=/experiments/xuyangzhang/venvs/quadratic/bin/python \
bash scripts/run_joint_pilot.sh --dry-run
```

输出应是 24 个候选、每条 573 steps / 300,417,024 tokens、总预算 7,210,008,576 tokens。此命令不读取数据、不创建结果目录、不启动训练。

## 3. 挂起运行

确保当前会话获准使用两张卡；入口保留已有 `CUDA_VISIBLE_DEVICES`，未设置时使用 0、1。每卡一个独立候选，不使用 DDP。参考 micro-batch=4、全局 batch=512 条序列、单卡累积=128。

```bash
cd /experiments/xuyangzhang/nanogpt-noise-experiments
nohup env \
  INPUT_ROOT=/experiments/xuyangzhang/nanogpt-inputs \
  PYTHON=/experiments/xuyangzhang/venvs/quadratic/bin/python \
  PILOT_OUT=/experiments/xuyangzhang/nanogpt-noise-experiments/joint-pilot-v1.8-2b \
  bash scripts/run_joint_pilot.sh \
  >> joint-pilot-v1.8-2b.console.log 2>&1 < /dev/null &
```

最后显示的是 supervisor 的进程号，保存下来便于停止。默认预计约 **6 小时**，依据此前双 RTX micro4 的 20-step 测速线性外推；不是时间上限。高 LR 数值失败会记录实际消耗，可能提前结束该候选。完整输入验证、并发磁盘、持续吞吐和负载可能改变实际工期。

也可在 `tmux` 中前台执行同样的 `env ... bash ...`。不设置 `PILOT_OUT` 时每次调用会创建一个新的时间戳目录；需要恢复时必须显式使用原目录。

## 4. 查看进度

```bash
tail -f /experiments/xuyangzhang/nanogpt-noise-experiments/joint-pilot-v1.8-2b.console.log
```

总日志会打印每次领取任务、GPU token、候选日志路径及完成数量。候选日志在结果目录的 `launcher/*.log`，每 10 步打印训练进度。`summary.json` 在队列推进时和结束时更新。

在新终端对当前目录重新生成报告，不会启动训练：

```bash
cd /experiments/xuyangzhang/nanogpt-noise-experiments
INPUT_ROOT=/experiments/xuyangzhang/nanogpt-inputs \
PYTHON=/experiments/xuyangzhang/venvs/quadratic/bin/python \
PILOT_OUT=/experiments/xuyangzhang/nanogpt-noise-experiments/joint-pilot-v1.8-2b \
bash scripts/run_joint_pilot.sh --summarize
```

避免与正在生成报告的 supervisor 同时反复运行此命令；只读 `summary.json` 或 `report.txt` 通常就足够。`running_or_interrupted` 表示已有候选 metadata、尚无终态，程序不会据此宣称仍有活跃 GPU 进程。

## 5. 中断与恢复

向保存的 supervisor 进程号发送普通 TERM，例如 `kill -TERM 实际进程号`。程序停止领取新任务，通知活跃候选保存 checkpoint，并等候最多约 120 秒；仍未退出的子进程才会被强制结束。不要同时再启动第二份同目录任务。

恢复前确认原 supervisor 和子进程已退出，然后重新执行第 3 节完全相同的命令，尤其保持相同 `PILOT_OUT`、代码、输入、Python/PyTorch 版本和 micro-batch。完成/数值失败候选跳过，未完成候选从自己的 `latest.pt` 接续。

OOM、磁盘或 Python/CUDA 错误属于基础设施问题：停止后修复原问题，以原配置恢复；程序不会自动减 micro、改 LR 或换数据。若改变 micro/代码/运行环境，恢复会被拒绝，应保留旧结果并另用新输出目录。

硬杀或断电可能留下结果根目录及候选目录的 `.lock`。先确认没有对应进程，再人工处理明确的遗留锁；脚本不会自动删锁。不要删除 checkpoint、修改 manifest 或状态文件来绕过恢复检查。

## 6. 完成标志和发回的文件

总日志最后出现 `PILOT FINISHED`，且 `summary.json` 的 `complete=true`、`completed+numerical_failure=24`。数值失败是需要分析的结果，完成初筛不表示所有候选训练成功或科学假设通过。`PILOT INCOMPLETE`、非零退出、输入/身份报错需要处理后恢复。

运行结束或正常中断时，程序自动生成：

```text
joint-pilot-v1.8-2b/
├── pilot_manifest.json      # 冻结输入、脚本/训练代码哈希、计划、环境、micro
├── pilot_status.json
├── summary.json             # 汇总、有限性、LR 初始网格对照、小噪声配对
├── candidates.csv
├── evaluations.csv          # 实际评估曲线
├── report.txt
├── analysis_bundle.zip      # 发回这个小包；不含模型 checkpoint
├── launcher/*.log
└── 每候选目录/              # metadata/status/evaluations、诊断与 checkpoint
```

下载并发回 **`analysis_bundle.zip`** 即可开始分析；模型 checkpoint 继续保存在 GPU 机器上。ZIP 包含候选 metadata、状态、评估、压缩诊断和日志；不包含 `.pt` 大文件。异常强杀时这个包可能尚未生成，可先恢复/生成报告并保留所有原文件。

用户指定本机所有 GPU 下载统一使用 **D 盘**。本次接收目录固定为 `D:\港中深RA\VSCode下载\New\gpu-results\joint-pilot-v1.8-2b`；在本机 PowerShell 下载，不经过 C 盘 Downloads 或 Desktop 中转：

```powershell
$pilotReceiveDir = 'D:\港中深RA\VSCode下载\New\gpu-results\joint-pilot-v1.8-2b'
New-Item -ItemType Directory -Force -Path $pilotReceiveDir | Out-Null
scp -P 23457 'xuyangzhang@119.23.54.6:/experiments/xuyangzhang/nanogpt-noise-experiments/joint-pilot-v1.8-2b/analysis_bundle.zip' "$pilotReceiveDir\analysis_bundle.zip"
```

上述命令只下载分析包。若后续需要 checkpoint 或完整 scope，同样显式指定 D 盘目标，不能默认同步到 C 盘。50GB 的预留建议针对 GPU 服务器结果盘，不要求本机 C 盘保留相同空间。

汇总中的 LR 差值是三档初始点 winner 与五档初始点 winner 的 0.3B 前缀 loss 差，不是完整三加二与五加三搜索的误差。小噪声差值是相同中心 LR 的配对比较，不是各自调优后的 ΔL。未数值失败的候选都到同一预定终点；有限但较大的 loss 不会被中途淘汰。
