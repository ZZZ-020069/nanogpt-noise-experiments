# 2026-11-01：保存旧结果，从本次最新交付重新启动

本说明适用于已经运行32小时的旧版044d3ab任务。先取回轻量记录，保留旧目录，再从本次交付的代码开始新的last1任务。新任务复用已登记输入，从共同未训练初始化开始；旧checkpoint不放入新RUN_BASE，不修改旧metadata。

本次交付标签为 **acp-start-20261101**，它指向包含本说明的本次最新提交；完整commit SHA以本次交付回复和准备任务日志为准。以下代码通过该标签选择版本，不再沿用此前回复中的044d3ab、a2137fa或2acd8db。smoke和formal必须打印相同的ACP code commit。

## 1. 学长先发回这32小时的哪些文件

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

## 2. 单独提交代码准备任务，只做一次

不覆盖旧目录nanogpt-noise-experiments-20261030。以下新目录须尚不存在；若已经使用，保留原目录并统一选择另一个新代码路径。

在ACP提交下列准备命令，无需申请训练GPU。该任务结束后再提交冒烟：

```bash
set -euo pipefail
git clone --branch acp-start-20261101 --depth 1   https://github.com/ZZZ-020069/nanogpt-noise-experiments.git   /data/260010171/nanogpt-noise-experiments-20261101

git -c safe.directory=/data/260010171/nanogpt-noise-experiments-20261101   -C /data/260010171/nanogpt-noise-experiments-20261101 rev-parse HEAD
```

标签克隆后处于detached HEAD是正常情况。记下打印的完整commit，应该与本次交付回复一致。后续启动只使用下方带20261101的新代码路径，不调用旧run_v18_ready.sh或旧日期副本。

## 3. 配置每个任务都需要的API Key

在ACP任务的环境变量／密钥配置中，设置SWANLAB_API_KEY为用户提供的完整Key。smoke、formal及重启任务都带上这一配置，Key不写入命令或Git。使用个人空间时不设置SWANLAB_WORKSPACE；使用团队则另设有权限的团队username。

以下命令复用日志中确认过的Python环境与已登记输入。环境必须持久保存，SwanLab依赖由run_acp.sh安装。输入无需重做；若平台路径不同，运行者在smoke前统一调整，之后固定。

本次新RUN_BASE为/data/260010171/results/nanogpt-v1.8-20261101。**首次smoke前该目录应为空或不存在**；若已用于旧提交，另选新空目录，并同步修改所有后续代码块。smoke创建目录后，formal和重启继续使用同一目录。

## 4. 一张H100：整段粘贴到ACP启动命令

```bash
set -euo pipefail
export NANOGPT_SWANLAB=1
export SWANLAB_PROJ_NAME=nanogpt-noise
export INPUT_ROOT=/data/260010171/inputs-v1.8
export RUN_BASE=/data/260010171/results/nanogpt-v1.8-20261101
export PYTHON=/data/260010171/envs/nanogpt-py311/bin/python

CODE_ROOT=/data/260010171/nanogpt-noise-experiments-20261101
EXPECTED_COMMIT="$(git -c safe.directory="$CODE_ROOT" -C "$CODE_ROOT" rev-parse 'refs/tags/acp-start-20261101^{commit}')"
ACTUAL_COMMIT="$(git -c safe.directory="$CODE_ROOT" -C "$CODE_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]]
printf 'Pinned release: %s\n' "$ACTUAL_COMMIT"
exec bash "$CODE_ROOT/scripts/run_acp.sh" smoke last1
```

四行export配置上传开关、网页项目、输入与输出；PYTHON固定环境。每次ACP提交都是独立环境，不能只在上一个任务设置一次。

冒烟成功需要同时满足：

- ACP code commit与第2节及本次交付回复一致，不能仍是044d3ab。
- 日志显示SMOKE PASSED。
- 用户登录自己的SwanLab个人空间，看到nanogpt-noise项目及四优化器smoke记录，指标包括evaluation/val、progress/step。

网页无需预建项目或组织；代码联网认证成功后创建项目。只有SMOKE PASSED但网页无数据时，先排查[swanlab]日志、密钥和联网，不直接提交正式任务。

## 5. 四张H100：整段粘贴到ACP启动命令

创建一个实例分配四卡，带相同API Key配置。不要由平台复制四个实例。

```bash
set -euo pipefail
export NANOGPT_SWANLAB=1
export SWANLAB_PROJ_NAME=nanogpt-noise
export INPUT_ROOT=/data/260010171/inputs-v1.8
export RUN_BASE=/data/260010171/results/nanogpt-v1.8-20261101
export PYTHON=/data/260010171/envs/nanogpt-py311/bin/python

CODE_ROOT=/data/260010171/nanogpt-noise-experiments-20261101
EXPECTED_COMMIT="$(git -c safe.directory="$CODE_ROOT" -C "$CODE_ROOT" rev-parse 'refs/tags/acp-start-20261101^{commit}')"
ACTUAL_COMMIT="$(git -c safe.directory="$CODE_ROOT" -C "$CODE_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]]
printf 'Pinned release: %s\n' "$ACTUAL_COMMIT"
exec bash "$CODE_ROOT/scripts/run_acp.sh" formal last1
```

启动时检查：新代码路径、新RUN_BASE、同一release commit。首批任务应为140/175/210/245（四优化器的节点0；控制台打印先后可不同），之后空闲GPU继续领取下一任务。不能仍按140/141/142/143作为首批。pool应自动输出[report]阶段摘要，SwanLab陆续出现formal记录；完成last1后停止，不自动运行其他scope。

## 6. 以后中断与续跑

ACP平台停止任务，等待原进程退出。重新获得四卡后，原样提交第5节整段命令和相同密钥配置。不再clone、git pull、重新制备数据、换目录或重做smoke。checkpoint仍在本轮RUN_BASE下恢复，平台强杀可能回退到最近周期checkpoint。实际保存宽限期需平台验收；不要手工删除.lease文件。

新版SwanLab每次候选恢复创建独立attempt，config通过training_run_id/resumed_from_step对应同一轨迹；模型恢复以本地checkpoint为准。完整35×4响应表和阶段报告位于RUN_BASE/reports/last1。旧32小时结果不混入新目录。
