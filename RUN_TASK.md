# ACP唯一操作指南：保存旧结果 → 冒烟 → 正式运行

**学长和用户只需按本文件第1至6节顺序操作，不需要再对照其他启动说明。所有代码块整段粘贴到ACP的启动命令中提交；需要的GPU数量在各节标明。**

本说明适用于已经运行32小时的旧版044d3ab任务。先取回轻量记录，保留旧目录，再从本次交付的代码开始新的last1任务。新任务复用已登记输入，从共同未训练初始化开始；旧checkpoint不放入新RUN_BASE，不修改旧metadata。

本轮训练固定使用标签 **acp-start-20261101-v2**，预期完整commit为 **241de0dcb7bfd2f3fe761ad08f9ab6302c3cc3f0**。第2、4、5节已写入这个编号，会自动打印预期与实际版本并核对，无需另找交付回复或人工逐字符比较。不一致时立即报错退出。

本指南的自动核对命令更新在main分支；训练标签仍固定在上述commit，不随说明更新而移动。请从GitHub的main分支阅读本指南并复制命令，克隆目录内旧版RUN_TASK.md不包含本次说明修订。服务器训练代码无需git pull到main。此前acp-start-20261101标签与分散启动示例属于上一交付。

所有代码块均使用Bash执行，并保留换行。第1、2节只用CPU，不调用GPU；ACP是否允许单独申请CPU任务取决于平台。若只能使用H100任务环境，也可执行，GPU将闲置，资源计费以平台为准。准备、冒烟和正式任务必须挂载同一份持久化/data/260010171存储。

## 1. 无需GPU：打包并发回旧32小时记录

旧结果根目录：`/data/260010171/results/nanogpt-v1.8-platform/last1`。

| 位置 | 发回的文件 | 用途 |
|---|---|---|
| 旧结果根目录 | scope_run.json | scope与任务范围 |
| launcher/ | 所有.log | 每个job的实际训练、评估和退出情况 |
| job_0140至job_0155/ | 各自search.json | LR候选、winner与数值失败 |
| 每个job的candidate_*/ | metadata.json、status.json、evaluations.json | 代码/输入身份、训练终点、loss曲线 |
| 每个candidate_*/ | steps.jsonl.gz与steps.compaction.json；未压缩时的steps/*.json | 梯度、噪声、LR与吞吐诊断 |
| /data/260010171/inputs-v1.8/ | registration.json、SHA256SUMS、frozen-artifacts/identity.json | 核对共同输入身份 |

以上只收已有文件；未开始的candidate或中断时尚未产生的文件无需补造。旧版本没有progress.json。已提供的ACP控制台.gz也请保留。此次传输不含模型.pt、语料.bin或环境；旧checkpoint在服务器保留，确认分析与存储安排后再决定清理。

先确保旧ACP任务已经停止。可以单独提交以下CPU打包任务，无需交互终端，也不启动训练：

```bash
set -euo pipefail
/data/260010171/envs/nanogpt-py311/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,tarfile
root=Path('/data/260010171/results/nanogpt-v1.8-platform/last1')
inputs=Path('/data/260010171/inputs-v1.8')
assert root.is_dir(), 'Old result directory is missing'
selected=[]
def add(path,arc):
    if path.is_file() and not path.is_symlink(): selected.append((path,arc))
add(root/'scope_run.json','last1/scope_run.json')
for path in sorted((root/'launcher').glob('*.log')):
    add(path,'last1/launcher/'+path.name)
for number in range(140,156):
    job=root/f'job_{number:04d}'
    add(job/'search.json','last1/'+job.name+'/search.json')
    for candidate in sorted(job.glob('candidate_[0-9][0-9]')):
        for filename in ('metadata.json','status.json','evaluations.json',
                         'steps.jsonl.gz','steps.compaction.json'):
            path=candidate/filename
            add(path,'last1/'+path.relative_to(root).as_posix())
        for path in sorted((candidate/'steps').glob('*.json')):
            add(path,'last1/'+path.relative_to(root).as_posix())
for rel in ('registration.json','SHA256SUMS','frozen-artifacts/identity.json'):
    add(inputs/rel,'inputs-v1.8/'+rel)
assert selected, 'No files selected'
output=Path('/data/260010171/results/old32h-analysis-20261101.tar.gz')
assert not output.exists(), 'Bundle already exists; preserve it or choose another output filename'
manifest=[]
with tarfile.open(output,'w:gz') as bundle:
    for path,arc in selected:
        content=path.read_bytes()
        manifest.append(dict(path=arc,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        bundle.add(path,arcname=arc,recursive=False)
manifest_path=output.with_suffix(output.suffix+'.manifest.json')
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('SEND:',output)
print('SEND:',manifest_path)
print('files:',len(selected),'bundle bytes:',output.stat().st_size)
PY
```

请将生成的tar.gz和manifest.json通过平台允许的下载方式发回。用户本机直接接收到：

```text
D:\港中深RA\VSCode下载\New\gpu-results\old32h-20261101\
```

这是一份轻量分析包，不是完整训练备份；实际大小以打包日志为准。

## 2. 无需GPU：准备本次新代码，只做一次

不覆盖旧目录nanogpt-noise-experiments-20261030。以下新目录须尚不存在；若已经使用，保留原目录并统一选择另一个新代码路径。

在ACP提交下列准备命令，不使用GPU。该任务成功结束且日志显示“版本核对通过（VERSION CHECK PASSED）”后再提交冒烟：

```bash
set -euo pipefail
CODE_ROOT=/data/260010171/nanogpt-noise-experiments-20261101-v2
EXPECTED_COMMIT=241de0dcb7bfd2f3fe761ad08f9ab6302c3cc3f0
if [[ -e "$CODE_ROOT" || -L "$CODE_ROOT" ]]; then
  printf 'ERROR: 代码路径已存在，保留原目录并选择新的路径：%s\n' "$CODE_ROOT" >&2
  exit 2
fi
git clone --branch acp-start-20261101-v2 --depth 1 \
  https://github.com/ZZZ-020069/nanogpt-noise-experiments.git \
  "$CODE_ROOT"

ACTUAL_COMMIT="$(git -c safe.directory="$CODE_ROOT" -C "$CODE_ROOT" rev-parse --verify 'HEAD^{commit}')"
printf '预期版本（Expected commit）: %s\n实际版本（Actual commit）:   %s\n' "$EXPECTED_COMMIT" "$ACTUAL_COMMIT"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  printf 'ERROR: 版本不一致（VERSION CHECK FAILED），停止；请保留日志和目录，不要启动训练。\n' >&2
  exit 2
fi
printf '版本核对通过（VERSION CHECK PASSED）\n'
```

标签克隆后处于detached HEAD是正常情况。成功日志会同时显示上述两串完整编号和通过提示；下载失败或读取commit失败同样会以非零状态退出，不会报告通过。保留任务日志即可，无需手动比较编号。后续启动只使用下方带20261101的新代码路径，不调用旧run_v18_ready.sh或旧日期副本。

## 3. 在ACP配置密钥与持久目录

在ACP任务的环境变量／密钥配置中，设置SWANLAB_API_KEY为用户提供的完整Key。smoke、formal及重启任务都带上这一配置，Key不写入命令或Git。使用个人空间时不设置SWANLAB_WORKSPACE；使用团队则另设有权限的团队username。

以下命令复用日志中确认过的Python环境与已登记输入。环境必须持久保存，SwanLab依赖由run_acp.sh安装。输入无需重做；若平台路径不同，运行者在smoke前统一调整，之后固定。

本次新RUN_BASE为/data/260010171/results/nanogpt-v1.8-20261101-v2。**首次smoke前该目录应为空或不存在**；若已用于旧提交，另选新空目录，并同步修改所有后续代码块。smoke创建目录后，formal和重启继续使用同一目录。

## 4. 一张H100：提交冒烟，确认网页收到指标

```bash
set -euo pipefail
export NANOGPT_SWANLAB=1
export SWANLAB_PROJ_NAME=nanogpt-noise
export INPUT_ROOT=/data/260010171/inputs-v1.8
export RUN_BASE=/data/260010171/results/nanogpt-v1.8-20261101-v2
export PYTHON=/data/260010171/envs/nanogpt-py311/bin/python

CODE_ROOT=/data/260010171/nanogpt-noise-experiments-20261101-v2
EXPECTED_COMMIT=241de0dcb7bfd2f3fe761ad08f9ab6302c3cc3f0
ACTUAL_COMMIT="$(git -c safe.directory="$CODE_ROOT" -C "$CODE_ROOT" rev-parse --verify 'HEAD^{commit}')"
printf '预期版本（Expected commit）: %s\n实际版本（Actual commit）:   %s\n' "$EXPECTED_COMMIT" "$ACTUAL_COMMIT"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  printf 'ERROR: 版本不一致（VERSION CHECK FAILED），停止；请保留日志和目录，不要启动训练。\n' >&2
  exit 2
fi
printf '版本核对通过（VERSION CHECK PASSED）\n'
exec bash "$CODE_ROOT/scripts/run_acp.sh" smoke last1
```

四行export配置上传开关、网页项目、输入与输出；PYTHON固定环境。每次ACP提交都是独立环境，不能只在上一个任务设置一次。

冒烟成功需要同时满足：

- 日志显示VERSION CHECK PASSED，随后ACP code commit为本指南固定的241de0dcb7bfd2f3fe761ad08f9ab6302c3cc3f0。
- 日志显示SMOKE PASSED。
- 用户登录自己的SwanLab个人空间，看到nanogpt-noise项目及四优化器smoke记录，指标包括evaluation/val、progress/step。

网页无需预建项目或组织；代码联网认证成功后创建项目。只有SMOKE PASSED但网页无数据时，先排查[swanlab]日志、密钥和联网，不直接提交正式任务。

## 5. 四张H100：提交正式last1训练

创建一个实例分配四卡，带相同API Key配置。不要由平台复制四个实例。

```bash
set -euo pipefail
export NANOGPT_SWANLAB=1
export SWANLAB_PROJ_NAME=nanogpt-noise
export INPUT_ROOT=/data/260010171/inputs-v1.8
export RUN_BASE=/data/260010171/results/nanogpt-v1.8-20261101-v2
export PYTHON=/data/260010171/envs/nanogpt-py311/bin/python

CODE_ROOT=/data/260010171/nanogpt-noise-experiments-20261101-v2
EXPECTED_COMMIT=241de0dcb7bfd2f3fe761ad08f9ab6302c3cc3f0
ACTUAL_COMMIT="$(git -c safe.directory="$CODE_ROOT" -C "$CODE_ROOT" rev-parse --verify 'HEAD^{commit}')"
printf '预期版本（Expected commit）: %s\n实际版本（Actual commit）:   %s\n' "$EXPECTED_COMMIT" "$ACTUAL_COMMIT"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  printf 'ERROR: 版本不一致（VERSION CHECK FAILED），停止；请保留日志和目录，不要启动训练。\n' >&2
  exit 2
fi
printf '版本核对通过（VERSION CHECK PASSED）\n'
exec bash "$CODE_ROOT/scripts/run_acp.sh" formal last1
```

启动时检查：新代码路径、新RUN_BASE、同一release commit。首批任务应为140/175/210/245（四优化器的节点0；控制台打印先后可不同），之后空闲GPU继续领取下一任务。不能仍按140/141/142/143作为首批。pool应自动输出[report]阶段摘要，SwanLab陆续出现formal记录；完成last1后停止，不自动运行其他scope。

## 6. 平台停止后：原样重新提交第5节命令续跑

ACP平台停止任务，等待原进程退出。重新获得四卡后，原样提交第5节整段命令和相同密钥配置。不再clone、git pull、重新制备数据、换目录或重做smoke。checkpoint仍在本轮RUN_BASE下恢复，平台强杀可能回退到最近周期checkpoint。实际保存宽限期需平台验收；不要手工删除.lease文件。

新版SwanLab每次候选恢复创建独立attempt，config通过training_run_id/resumed_from_step对应同一轨迹；模型恢复以本地checkpoint为准。完整35×4响应表和阶段报告位于RUN_BASE/reports/last1。旧32小时结果不混入新目录。
