# NanoGPT 噪声实验：v1.8，先运行 last1

**全部六个 scope 统一采用 35 噪声节点 × 四优化器 × 八个学习率候选 × 2B tokens。当前执行阶段只有 last1；完成并分析后再决定下一阶段。** 运行者先阅读 [简短任务说明](RUN_TASK.md)。

## 当前协议

| 项目 | 设置 |
|---|---|
| 模型 | GPT-2 124M，12 blocks，统一未训练初始化，单 seed |
| scope | full、last1、last2、last3、last6、last12，全部使用相同预算 |
| 当前运行 | last1，仅训练最后一个 block 的 7,079,424 参数 |
| 噪声 | σ₁={0,.03,.1,.3,.9}；σ₂={0,.0001,.001,.01,.1,1,10}；35 个节点 |
| 优化器 | AdamW、Adam、SGD-Momentum、Muon；配方不变 |
| LR 搜索 | 五初始点 {0.1,0.3162278,1,3.162278,10}×基准，三次自适应扩边/加密 |
| 训练预算 | 每候选 3815 steps，实际 2,000,158,720 tokens |
| 调度 | 100M tokens 线性 warmup，再余弦衰减至峰值的 10% |
| 全局 batch | 512×1024=524288 tokens；单卡 micro=4，累积128次 |
| 评估 | 每100M、step0及终点，合计21次；最终验证 loss 选择 winner |
| 失败规则 | 只有 NaN/Inf 提前数值失败；有限但较差的候选不淘汰；无梯度裁剪 |
| 并行 | 四 H100、四个独立单卡 worker；同一 job 内八个候选顺序运行 |

每 scope 有140 jobs、最多1120条独立候选；六个 scope 总计6720条候选。2B 是本次选定的有限预算，未声称与旧2.6B等价；last1 的约10–15天只是条件估算，尚无目标硬件实测保证。last1 不是预训练模型微调；冻结的前11层、embedding、final LayerNorm与输出头均保持共同随机初始化。last12仍冻结block外参数，因此不等于full。

唯一设计来源为工作区主设计；仓库中的 [发布快照](docs/nanogpt_experiment_design.md)与之同步，不单独维护另一套协议。快照中带日期的旧讨论和pilot记录不覆盖v1.8当前决定。

## 环境与输入

需要 Linux、Python≥3.11、CUDA版PyTorch≥2.6,<3、NumPy、matplotlib及支持BF16的GPU。`run_acp.sh` 会检查指定 PYTHON；未指定且默认环境不兼容时，通过Conda创建独立Python3.11环境，安装PyTorch2.8.0/cu128等依赖，不修改base。首次安装需要网络和额外磁盘；`ENV_PREFIX`可指定环境目录。失败就停止，不跳过检查。

默认目录：

```text
/data/260010171/nanogpt-noise-experiments   代码
/data/260010171/inputs-v1.8                当前输入
/data/260010171/results/nanogpt-v1.8       当前结果
```

可用 INPUT_ROOT、RUN_BASE、PYTHON 覆盖；保持绝对路径，smoke与正式运行一致。INPUT_ROOT下为preparation.json、openwebtext_subset、frozen-artifacts、registration.json和SHA256SUMS。DATA_DIR和ARTIFACT_DIR通常不应单独覆盖，以免与注册根目录不一致。

所有服务器统一在新的空 INPUT_ROOT 中执行一次：

```bash
bash scripts/run_acp.sh inputs
```

固定 OpenWebText revision、GPT-2 tokenizer 及既定划分，直接在服务器流式生成 1B 训练语料和 10M 验证语料，再重新制备共同未训练初始化、训练与评估窗口并登记。无需 SCP 上传数据，也不接受旧输入迁移。1B 是语料容量，训练时重复采样至 2B。首次制备可能耗时数小时，需要服务器访问数据源。

目录必须为空或不存在。程序创建本版本 preparation.json 标记；中断后仅允许继续本版本制备，明确的 .incomplete 语料可从源重做，半写入的最终语料/冻结产物保留诊断并改用新目录。旧文件不会被覆盖。成功登记后，再运行 inputs 只校验身份和哈希，不重新生成；所有候选及 scope 共用这一份输入，正式训练后不得更换。

输入文件约 2.53 GB；流式读取仍可能使用缓存，可通过 HF_HOME 指定位置，环境目录由 ENV_PREFIX 指定。缓存和环境需要额外空间。只准备环境而不制备输入或训练：`bash scripts/run_acp.sh prepare`。已激活兼容环境时，完整制备的等价入口是 `python -m nanogpt_noise.prepare --input-root /绝对路径/inputs-v1.8 --restart-incomplete`。

## 冒烟和正式启动

提交前确保干净仓库；输入、结果和控制台日志放在仓库外。入口仅为当前进程配置仓库的Git safe.directory。没有任务运行时更新一次，smoke开始后固定commit：

```bash
git -c safe.directory=/data/260010171/nanogpt-noise-experiments pull --ff-only
```

一张H100，last1四优化器各8步（共32步）：

```bash
bash scripts/run_acp.sh smoke last1
# 已激活兼容环境的等价入口：bash scripts/run_last1_smoke.sh
```

smoke包括节点(.03,.01)、中心LR、初末固定窗口评估、冻结校验和及checkpoint。四项全部成功才写入代码和输入验收标记；正式入口还核对scope、micro、协议、软件版本与候选metadata。smoke保存为独立的短前缀结果，不参加正式排名。8步不检验收敛，也不足以保证多日吞吐。

四张H100，首次启动last1：

```bash
bash scripts/run_acp.sh formal last1
# 已激活兼容环境的等价入口：bash scripts/run_last1.sh
```

默认scope也是last1。四个独立worker动态领取jobs140–279，运行35节点×4优化器，各执行5＋3搜索。全部终态后自动finalize并停止，不启动下一个scope。未来其他scope仍可显式指定，但必须先完成对应scope的smoke，当前任务不包括它们。

GPU由平台分配；CUDA_VISIBLE_DEVICES必须列出四个不同的GPU编号/UUID。程序不申请GPU、不登录服务器、不自行启动下一scope。单个候选默认micro4；改micro必须在smoke前确定并保持全局batch一致，运行后不可换布局恢复。

## 存储、日志与恢复

每个job仅保留当前/最终winner的一份checkpoint（含模型与优化器状态）；非winner保留全部metadata、状态、评估及压缩逐步诊断。最后winner直接使用已有轨迹，不额外重训。每个scope完成后输出逐文件大小和SHA-256的scope_manifest.json。不要在结果盘另建同体积tar/zip。

`python -m nanogpt_noise.scope estimate --scope last1` 输出静态空间估算，smoke和首批任务还需核验真实文件大小。last1静态峰值约84.03GB；full峰值约195.58GB，不能把该值理解为last1实际使用量；该84.03GB已含公共输入/日志4GB预留，另加四项smoke checkpoint约2.16GB；环境、下载缓存及接收端空间另计。脚本报告空间，不自动设置磁盘阈值。

中断后使用相同commit、环境、输入、scope、micro与RUN_BASE恢复。主动执行 pause 后用 resume 解除暂停标记并续训；直接执行 formal 会拒绝启动。意外中断且无暂停请求时，确认原进程已退出并处理遗留锁后，可重跑 formal。两者共用同一套续训逻辑。完成候选复用终态，未完成候选恢复latest.pt。NaN/Inf保留明确失败标签并继续搜索；OOM/存储/硬件错误属于基础设施故障，恢复同一候选，不能换LR规避。如果smoke后改变代码、环境或micro，使用新的RUN_BASE重新冒烟；不要覆盖旧输出。旧2.6B、旧pilot及旧smoke不能迁入新结果目录，也不能修改metadata绕过身份检查。

进度查看（默认目录）：

```bash
bash scripts/h100.sh status last1
tail -f /data/260010171/results/nanogpt-v1.8/last1/launcher/job_0140.log
```

主动暂停（另开终端，设置与训练相同的 RUN_BASE）：

```bash
bash scripts/run_acp.sh pause last1
# 等待 SCOPE QUIESCENT 后再将 GPU 交给其他人。
# 算力恢复后，在相同代码/环境/输入/输出下：
bash scripts/run_acp.sh resume last1
```

pause 停止分配新任务，并让活跃候选完成当前 optimizer step、保存完整状态后退出；不会重新制备输入。暂停文件保留到显式 resume，后者只在获得独占调度锁后清除请求。pause 不安装依赖，不要求 CUDA；默认等待300秒，PAUSE_WAIT_SECONDS可调整，超时不会强杀进程，也不表示GPU已经空闲。训练端以75结束属于可恢复暂停，包装器可能打印 ACP FAILED，但不是数值发散，也不会执行finalize。

断电、SIGKILL或硬件错误可丢失最近一次周期checkpoint之后的进度，通常最多重算约191步/100M tokens。已完成候选和后续候选断点在搜索重放时保留。硬杀可能留下launcher、job或candidate的.lock；先确认所有相关进程都已退出，再处理对应遗留锁，程序不自动解锁。有进度但checkpoint缺失或最终winner缺失会报错，不静默从零开始。

输入、结果与环境必须放在平台停止/销毁任务后仍保留的目录。若使用nohup，将控制台日志放在RUN_BASE之外或其父目录并使用绝对路径；平台任务中通常直接用启动命令。恢复期间不得更新代码或依赖。此修复不改变v1.8输入，可复用已制备输入；从旧提交升级须在正式训练前重新冒烟，不能直接改已有轨迹的metadata来兼容新代码。

## 完成与下载

看到FORMAL SCOPE FINALIZED后，完整传输RUN_BASE/last1目录。不要只下载winner或summary而遗漏候选日志。使用WinSCP/SFTP或rsync保留目录层级，连接中断后续传。

本机接收位置必须在D盘，例如 `D:\港中深RA\VSCode下载\New\gpu-results\nanogpt-v1.8\last1`，不要经C盘Downloads/Desktop中转。在安装同一commit的接收端校验：

```powershell
python -m nanogpt_noise.scope verify --runs 'D:\港中深RA\VSCode下载\New\gpu-results\nanogpt-v1.8\last1'
```

校验成功前保留服务器副本。当前安排是在last1完成后分析，暂不启动下一个scope。若以后需人工清理已交接目录，现有scope cleanup命令要求目录匹配、finalize/self-verify通过和DELETE-scope确认串；没有自动清理。

单scope分析：

```bash
python -m nanogpt_noise.analyze --runs /data/260010171/results/nanogpt-v1.8/last1 --output /data/260010171/results/analysis-last1-v1.8 --plots
```

## Slurm 与维护

先完成相同scope的smoke，再从仓库根目录提交scripts/slurm.sh；设置INPUT_ROOT、RUN_BASE、PYTHON，默认SCOPE=last1。学校partition/account/H100 gres通过sbatch参数指定。模板单节点四卡、16CPU、64GB内存、24小时时限；超时后提交相同命令恢复，不更改实验预算。Slurm入口同样通过正式检查并自动finalize。

维护命令：`bash scripts/h100.sh check`、`status last1`、`finalize last1`。底层run_4gpu.sh/node/queue仅供维护，不替代公开入口的smoke验收。scope编号保持不变：full：0–139；last1：140–279；last2：280–419；last3：420–559；last6：560–699；last12：700–839。

本地验证：`python -m unittest discover -s tests -v`；工作区设计同步：`python scripts/check_design.py`（独立clone需用--source指定权威主文档）。[验证记录](docs/VALIDATION.md)区分CPU检查与尚待执行的目标GPU验收。可选联合pilot和双RTX测速不是本轮last1正式启动步骤；新版pilot使用独立2B身份，旧pilot结果保留其原协议标签。

模型参考nanoGPT，保留[MIT声明](third_party/nanogpt-LICENSE)；无WandB依赖。


升级本次断点修复时，如果此前已经运行过旧提交的 smoke，请保留旧结果并先选择新结果目录，再冒烟；之后所有启动、暂停、继续命令均使用这个 RUN_BASE：

```bash
export RUN_BASE=/data/260010171/results/nanogpt-v1.8-resume-fix
bash scripts/run_acp.sh smoke last1
```

已登记的 INPUT_ROOT 不变，无需重新制备语料；正式开始后固定提交，不能靠切换 RUN_BASE 把旧轨迹恢复成新代码轨迹。
