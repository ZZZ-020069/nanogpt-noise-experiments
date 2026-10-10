# NanoGPT gradient-noise experiments

实现设计 v1.4：完整 124M NanoGPT，full 和 last-{1,2,3,6,12}，四种优化器、35 个噪声节点、每节点八个完整预算 LR 候选。二次函数不在本次代码范围内。

**状态：CPU 协议/集成测试通过；H100 BF16、Linux NCCL 多卡、实际吞吐和显存尚未验证。没有正式实验结果。不要未经验收直接提交整个任务数组。**

## 四张 H100：直接启动

入口是 **`scripts/run_4gpu.sh`**，四张卡共同运行一个候选（DDP），不是每卡运行一个不同候选。先按下文安装 CUDA 版 PyTorch、传好数据和冻结产物，并在服务器申请到单机四张 GPU。脚本不会申请算力、下载数据或重新生成初始化。

```bash
git pull --ff-only
# 激活已安装本项目依赖的 Python 环境
export DATA_DIR=/实际路径/openwebtext_subset/data
export ARTIFACT_DIR=/实际路径/protocol-v1.4
export RUN_DIR=/实际路径/results/nanogpt-v1.4

# 先验收：完整模型、正式全局 batch，Muon 非零噪声，运行8步
bash scripts/run_4gpu.sh smoke

# 验收通过后：正式运行 job 0 的八个完整预算 LR 候选
bash scripts/run_4gpu.sh node 0
```

脚本要求恰好四张可见 GPU，逐卡检查 BF16、检查 NCCL 和数据/权重哈希；默认每卡 micro-batch=4、累积32次，全局 batch=512 不变。若服务器可见卡数超过四张，可在**已获分配的卡内**设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`（编号按实际分配）；Slurm 已设置该变量时应保留调度器设置。`PYTHON` 可指定环境内 Python 的绝对路径。

`smoke [scope optimizer node]` 可选择验收配置，例如：

```bash
bash scripts/run_4gpu.sh smoke full adamw 0
bash scripts/run_4gpu.sh smoke last1 adam 10
bash scripts/run_4gpu.sh smoke last12 sgd 10
```

这些输出都放在 `$RUN_DIR/smoke/4gpu/`，不进入正式结果。任意验收报错或数值失败都要先检查，不能据此继续批量启动。短程成功也不保证所有正式 LR 都稳定；正式搜索中的数值失败按协议记录并继续下一候选。恢复时执行原命令，保持代码、环境、四卡布局和 micro-batch 不变。

`node N` 每次仅启动一个节点；不会自动跑完整 840 个节点。需要顺序运行多个节点时，在验收和预算确认后显式指定，例如 `for job in $(seq 0 34); do bash scripts/run_4gpu.sh node "$job" || break; done` 只运行 full/AdamW 的35节点，不是四种优化器的全部节点。建议在服务器的 tmux 会话或学校调度任务内运行，避免 SSH 断开打断训练。Slurm 使用下文模板时，另以 `--gres=gpu:4` 申请四卡并设置 `GPUS_PER_JOB=4`，不要只申请一卡却启动四进程。

## 设计来源和目录

工作区的唯一编辑源为 `VSCode下载/New/nanogpt_experiment_design.md`。本仓库 `docs/nanogpt_experiment_design.md` 是随代码发布的同版文本快照，不是独立维护的设计；修订应先改主文档，再同步快照。本文仅描述操作与实现。

- `nanogpt_noise/protocol.py`, `search.py`：冻结参数、token 调度和 LR 搜索。
- `model.py`, `optim.py`, `noise.py`：模型、参数范围、优化器和同步后的噪声注入。
- `data.py`, `prepare.py`：数据检查、固定权重/窗口，及可选的数据重新制备。
- `train.py`, `node.py`：独立候选、恢复、节点级八次候选搜索。
- `jobs.py`, `analyze.py`：完整任务清单与结果汇总/二维图。
- `tests/`：CPU 单元、端到端、断点恢复与多进程检查。
- `scripts/`：直接启动与 Slurm 模板。

模型结构参考 nanoGPT，原 MIT 声明见 `third_party/nanogpt-LICENSE`；不因此为整个研究项目额外授予统一开源许可。

## 1. 安装（Linux / H100）

使用 Python 3.11 或以上的独立环境。先按服务器驱动安装学校认可的 CUDA 版 PyTorch（代码范围为 2.6–2.x，CPU 实测为 2.9.1），不能把 CPU wheel 安装命令直接当作 H100 环境配置。驱动/CUDA wheel 的组合需要在目标机器核验，当前不假定某个具体 CUDA 版本。

```bash
git clone https://github.com/ZZZ-020069/nanogpt-noise-experiments.git
cd nanogpt-noise-experiments
python -m venv .venv
source .venv/bin/activate
# 在此环境安装与服务器驱动匹配的 CUDA PyTorch，然后：
pip install -e '.[plots]'
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"
python -m unittest discover -s tests -v
```

首次验收后将 `pip freeze` 保存到实验输出目录的 `environment-lock.txt`，正式批量运行锁定同一环境。已有合适的学校 Conda 环境也可以直接使用，不必套建 venv。CI 的 CPU 安装与 H100 安装分开。

## 2. 固定数据和公共产物

优先将现有 `train.bin`（2,000,000,000 字节）、`val.bin`（20,000,000 字节）、`meta.json` 传到服务器同一个数据目录。不要提交到 Git。现有数据的 tokenizer/制备依赖版本若未留存，必须标为未知，不能用当前版本冒充；制备 revision 与文件哈希由程序保存。

以下 `/path/to/...` 都是需由运行者替换的目录：

```bash
export DATA_DIR=/path/to/openwebtext_subset/data
export ARTIFACT_DIR=/path/to/shared/nanogpt-v1.4-artifacts
export RUN_DIR=/path/to/results/nanogpt-v1.4
python -m nanogpt_noise.data --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR"
python -m nanogpt_noise.data --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" --verify
python -m nanogpt_noise.jobs --output "$RUN_DIR/jobs.json"
```

冻结动作只做一次，生成 `initial.pt`、`windows.pt`、`identity.json`。每个候选启动时重验文件哈希；各 rank 读取同一份公共产物。初始化和全部训练/评估窗口在 CPU 固定，正式运行不重新抽样。输出文件不会覆盖已有不匹配产物。一次未完成的制备应换新目录重做。

本地已制备一套 `artifacts/protocol-v1.4/`（被 Git 忽略），对应身份登记在主设计 §12.1。若使用这套公共产物，请将整个目录连同三份数据文件另行传输，服务器仅运行 `--verify`；不要各自重新生成不同的初始化/窗口再混用。数据依赖版本未知项保持如实记录。

如果确实需要重新制备数据，可在独立空目录运行：

```bash
pip install -e '.[prepare]'
python -m nanogpt_noise.prepare --output /path/to/new-corpus
```

重新制备需要网络和数据访问权限；固定源 revision、验证优先、GPT-2 编码和边界文档丢弃规则。不同依赖版本仍可能得到不同产物，因此不能只凭同 seed 宣称与已有语料一致；比较哈希，所有 run 必须选择同一套最终产物。本次未在线重制备 1B 语料。

## 3. 先做 H100 smoke 验收

```bash
torchrun --standalone --nproc-per-node=1 -m nanogpt_noise.train \
  --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" \
  --output "$RUN_DIR/smoke/full-muon-noisy" \
  --scope full --optimizer muon --node 10 --exponent 0 --smoke-steps 8
```

smoke 使用完整模型、完整全局 batch、正式 scheduler，只在指定步数结束；始终标记 `mode=smoke`，不能进入正式搜索或替代正式预算。至少检查四种优化器、full 与 last1/last12、clean 与非零噪声；多卡时用实际卡数重复检查。默认每卡 micro-batch=4，卡数×micro-batch 必须整除 512。布局可以在新实验启动前配置，但同一候选恢复时拒绝改变布局。

Linux CPU 分布式机制检查（小模型、合成数据，不是正式结果）：

```bash
torchrun --standalone --nproc-per-node=2 -m tests.ddp_smoke --output /tmp/nanogpt-ddp-test
# 四进程检查请使用另一空输出目录
torchrun --standalone --nproc-per-node=4 -m tests.ddp_smoke --output /tmp/nanogpt-ddp-test-4
```

该测试验证四种优化器、full/last1、梯度累积、非零噪声、中断恢复及单 rank 数值失败的全局处理；本机 Windows 的 Gloo 后端不支持建立连接，因此尚未在本机通过。CI 配有 Linux 两进程/四进程测试，但其实际通过状态需查看 GitHub Actions。这不替代 H100/NCCL 验证。

H100 验收还应检查 BF16、显存峰值、每步耗时、评估/写盘耗时，并用 SIGTERM 后重启确认同候选恢复。首步 LR=0 是设计要求，不是训练故障。

## 4. 正式运行与 Slurm

每个 job 是一个 scope / optimizer / 噪声节点，内部顺序执行五个初始 LR 和三个自适应 LR。每条候选完整 4960 步；只有数值非有限提前结束。不使用梯度裁剪，也不因为有限的大 loss 提前淘汰。

```bash
# 先查看配置，不启动训练
python -m nanogpt_noise.node --job-index 0 --data "$DATA_DIR" \
  --artifacts "$ARTIFACT_DIR" --output "$RUN_DIR" --dry-run
# 单机单卡跑一个节点的八次搜索
bash scripts/run.sh 0
# 或单机四卡分摊同一个候选
GPUS_PER_JOB=4 MICRO_BATCH=4 bash scripts/run.sh 0
```

job 0–139 为 full；后续每 140 个依次为 last1、last2、last3、last6、last12。每个 scope 内 optimizer 顺序为 AdamW、Adam、SGD、Muon；每个优化器包含 σ1-major 的 35 节点。以 `jobs.json` 为准确映射。

Slurm 模板不虚构账户/分区名称：由运行者通过 `sbatch --partition=实际分区 --account=实际账户 scripts/slurm.sh` 指定。如学校无 account 要求则省略该参数。模板默认申请 1 GPU、8 CPU、32 GB 主机内存、24 小时；应按学校限制调整资源参数，不能改变实验参数。多卡需同时调整申请的 GPU 数和 `GPUS_PER_JOB`。

验收并确认总预算后才可手动添加例如 `--array=0-139%4`（full 的140个节点、最多4个并发）。完整范围为0–839，总计6720条候选、约17.475T tokens；脚本不自动提交这个规模。不同节点可以并行，一个节点内部的追加 LR 依赖已完成候选。

## 5. 恢复、故障和文件

原命令重跑即恢复，无需另传 checkpoint。完成的候选不会重复训练；数值失败是终态，基础设施中断继续原候选。SIGTERM/SIGINT 在完整更新结束后保存 latest；硬杀/断电从最近一次原子 checkpoint 恢复。Slurm 超时可能来不及保存最后一步，此时会重算此前未提交部分，不改变正式预算。

每条候选输出：

- `metadata.json`：协议、实现哈希、配置、数据/初始化身份、环境、布局、参数清单、Muon 分组。
- `steps/000001.json` 等：逐步日志；首步/末步及每10步详细诊断。
- `evaluations.json`：固定窗口曲线；恢复时以 checkpoint 中的曲线为准。
- `latest.pt`、`best.pt`、`final.pt`：可恢复状态、诊断最佳状态、完整预算状态。
- `status.json`：处理的 tokens、成功更新步数、模式、完成或故障状态。
- `failure.pt`：数值失败现场，不覆盖 last-known-good latest。

原子写入加 `.lock` 防止重复进程写同一目录。硬杀后若遗留 `.lock`，必须先确认该目录没有活跃 worker，再仅删除该锁文件；不要删除 checkpoint，也不要盲目解锁。禁止两个 Slurm job 使用同一个 job-index 和输出目录。

恢复拒绝更换配置、数据、代码、依赖版本和并行布局。请冻结正式 commit 后再运行；更新代码后不要硬改 metadata 绕过检查。若存储/权限错误发生在进入训练循环前，进程会报错退出，同样不能当作 numerical_failure。

Checkpoint 占用很大：full Adam/AdamW 每份约1.5 GB，latest/best/final 三份约4.5 GB（不含临时写入及其他输出）。应按各 scope/优化器测量总存储需求并预留原子写入空间。本代码不会擅自删候选 checkpoint 来省空间。

## 6. 汇总

```bash
python -m nanogpt_noise.analyze --runs "$RUN_DIR" --output "$RUN_DIR/analysis" --plots
```

生成 CSV/JSON 和 σ1×σ2 热图：终点 loss、相对各自调优 clean 的 Δloss、最佳 LR、优化器名次和数值失败数。缺失/全失败保留空白；只有八次搜索完成才给出正式节点 winner，未完成跨优化器比较标为不完整。只使用 `mode=formal`，不把小模型测试或 smoke 混入结果。单 seed 的排名不代表统计显著性。

## 实现边界

当前支持单机多卡 DDP，不支持跨节点启动、FSDP 或分片优化器。为了保证噪声配对，各 rank 使用相同独立 RNG 和去重参数顺序；评估 checkpoint 时检查副本权重哈希相同。范数采用高精度 reduction 避免平方和溢出；这可能带来额外开销，应测量吞吐但不能私自省略协议诊断。运行不依赖 WandB 等在线日志服务。

DDP 使用固定初始梯度桶布局（`find_unused_parameters=True, static_graph=False`），避免中断恢复后重新经历桶重排而改变浮点归约顺序。PyTorch 可能提示没有发现 unused parameters，这是该设置的预期开销提示；代码仍禁止可训练参数缺失梯度。不要为消除这条提示擅自更改 DDP 参数。具体测试记录见 `docs/VALIDATION.md`，CPU 逐位一致性测试不能代替 H100 的恢复验收。
