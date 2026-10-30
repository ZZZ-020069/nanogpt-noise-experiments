# ACP操作说明：先冒烟，再运行last1

当前只运行last1：35噪声节点×4优化器×8个学习率候选，每候选2B tokens。last1从共同未训练初始化开始，仅训练最后一个Transformer block。完成后先分析，不自动启动其他scope；四H100约10–15天仍是未实测的条件估算。

## 1. 更新代码一次，并配置API Key

确认旧任务已停止后，由能更新代码的人执行以下命令一次。冒烟后固定代码，续训时不再git pull：

```bash
cd /data/260010171/nanogpt-noise-experiments
git -c safe.directory=/data/260010171/nanogpt-noise-experiments pull --ff-only
```

在ACP的任务环境变量／密钥配置中添加：

| 名称 | 值 |
|---|---|
| SWANLAB_API_KEY | 你自己的完整API Key，以密钥方式保存 |

冒烟任务、正式任务及之后的重启任务都必须带上这项密钥配置。可选SWANLAB_WORKSPACE指定有权限的团队；不填则使用账号默认空间。Key不写入仓库或下面的启动命令。

## 2. 确认输入和结果目录

- INPUT_ROOT复用已经通过v1.8 inputs入口成功制备并登记的目录：/data/260010171/inputs-v1.8，不需重新生成。
- 本轮RUN_BASE选择/data/260010171/results/nanogpt-v1.8-swanlab；如果它已被旧提交使用，另选空目录，并把下方冒烟和正式命令中的RUN_BASE同时改成该目录。
- 第一次冒烟确定目录后，正式训练及所有中断续跑都固定相同目录、代码、环境、micro和持久存储挂载。已有兼容环境可通过ACP环境变量设置PYTHON=/绝对路径/bin/python；每次任务保持一致。

只有尚未成功制备v1.8输入时，才单独提交下面的数据准备命令：

```bash
export INPUT_ROOT=/data/260010171/inputs-v1.8
exec bash /data/260010171/nanogpt-noise-experiments/scripts/run_acp.sh inputs
```

首次制备需网络和数小时，输入文件约2.53GB，环境缓存另计。目标目录必须为空或由本版本制备标记管理；已登记目录重复执行inputs只校验，不重新抽样或初始化。

## 3. 一张H100：提交冒烟任务

在ACP中创建一个分配一张H100的任务，带上第1节的密钥配置。将下面整段粘贴到“启动命令”，作为一个任务一起提交；export四行与最后的bash命令一起执行，不需要运行中另开终端：

```bash
export NANOGPT_SWANLAB=1
export SWANLAB_PROJ_NAME=nanogpt-noise
export INPUT_ROOT=/data/260010171/inputs-v1.8
export RUN_BASE=/data/260010171/results/nanogpt-v1.8-swanlab

exec bash /data/260010171/nanogpt-noise-experiments/scripts/run_acp.sh smoke last1
```

四行export依次设置：开启SwanLab、网页项目名称、已登记输入目录、本轮结果目录。每次提交任务是独立环境，所以正式任务和重启任务也必须重复写这些设置。

入口在启用上传时安装SwanLab。冒烟依次运行四优化器，各8步，并检查固定窗口评估、冻结参数和checkpoint。必须同时确认两项后再正式运行：

1. ACP日志出现SMOKE PASSED。
2. SwanLab网页的nanogpt-noise项目出现四个优化器的smoke记录，能看到evaluation/val、progress/step指标。

仅SMOKE PASSED不能证明云端认证和联网成功。网页无数据时先检查密钥配置、网络和日志中的[swanlab]提示；暂不提交正式任务。

## 4. 四张H100：提交正式任务

创建一个实例分配四张H100，带上同一API Key配置，将下面整段粘贴到ACP“启动命令”：

```bash
export NANOGPT_SWANLAB=1
export SWANLAB_PROJ_NAME=nanogpt-noise
export INPUT_ROOT=/data/260010171/inputs-v1.8
export RUN_BASE=/data/260010171/results/nanogpt-v1.8-swanlab

exec bash /data/260010171/nanogpt-noise-experiments/scripts/run_acp.sh formal last1
```

与冒烟使用同一INPUT_ROOT和RUN_BASE。仓库内部启动四个独立单卡worker；不需要平台复制四个实例。任务按固定sigma1、遍历七个sigma2的顺序入队，每节点依次排列四优化器搜索。每个搜索内部顺序完成八候选，空闲GPU立即领取下一搜索，不等待整行。last1全部结束后自动finalize并停止。

## 5. 中途停止与断点续跑

需要交出算力时，使用ACP“停止任务”。再次获得四张H100后，原样重新提交第4节完整命令及密钥配置即可；不更换RUN_BASE，不重新制备输入，不再更新代码或依赖，也不重跑smoke。

平台先发TERM/INT且保存宽限期足够时，程序完成当前step、保存并退出；强杀或断电时回退到最近周期checkpoint，通常最多重算活跃候选约191步/100M tokens。实际平台保存宽限期仍需验收。已完成候选复用，未完成候选续训；已finalize目录再次启动只校验文件。持久存储必须跨任务保留并挂载相同目录。

Linux内核flock在进程结束后释放，重启取得独占权后接管本版本遗留.lock；不要删除.lease文件。旧格式锁或不支持flock的存储需要核查。有额外命令入口时才使用可选pause/resume，详见README。

## 6. 查看阶段报告和下载结果

自动阶段报告随正式任务一起启动，无需在ACP中途输入命令。噪声行固定按sigma1={0,.03,.1,.3,.9}外层、sigma2={0,.0001,.001,.01,.1,1,10}内层升序执行。不同节点可以同时运行，完成顺序可以不同于排队顺序；整行齐全后才报告完整行排名，但不阻塞GPU领取下一任务。报告由后台线程生成，实际工期需结合实测更新。

报告位于 `RUN_BASE/reports/last1/`：默认每5分钟更新 `latest.zip`、`latest.html` 和 `latest.json`，每行结束及可控退出时额外更新。ZIP内有可离线打开的HTML曲线/诊断、35行总表CSV、候选CSV和完整JSON；每个完成行保留一份 `row_00.zip` 至 `row_34.zip`。不含模型权重。需要在中途分析时，使用平台允许的持久存储访问/下载方式取回latest.zip，解压打开report.html；本机仍直接接收到D盘。平台若不提供运行中的文件下载，可以先看平台日志中的 `[report]` 摘要，并在停止后读取最近报告。

未完成格子只展示进度，八候选搜索完成后才填最终loss/LR；四种优化器均完成该行后才展示完整行排名。报告包括生成时间、generation和诊断所属步数；不是同步的跨GPU checkpoint，不将前缀排名当作最终结论。`report_error.json`或 `[report] ERROR` 表示报告更新失败，上次快照保留，训练继续按原协议执行。平台强杀可能来不及生成退出报告，下次同命令启动会重建。latest.zip是同一generation的完整快照，优先下载该文件。



最终结果在/data/260010171/results/nanogpt-v1.8-swanlab/last1。看到FORMAL SCOPE FINALIZED后下载整个目录，按scope_manifest.json校验；本机直接接收至D盘，例如D:\港中深RA\VSCode下载\New\gpu-results\nanogpt-v1.8-swanlab\last1，不经过C盘。不要在服务器另建一份大型压缩包，校验成功前保留服务器副本。存储和校验命令详见[README](README.md)。

## SwanLab记录含义

每候选每次启动一个独立上传记录，名称包括mode/scope/optimizer/node/candidate/attempt；config记录training_run_id和resumed_from_step。恢复后以training_run_id对应同一科学轨迹，用attempt区分回退重算，不能把上传记录数量当候选数量。横轴默认上传序号，分析时改用progress/tokens；evaluation指标附带其自身step/tokens，diagnostic/step标明诊断时刻。smoke与formal通过mode区分。上传指标为观察数据，最新评估值在两次评估之间保持；精确固定窗口曲线及35×4完整排名仍以本地reports包为准。

上传在独立进程，队列有界；断网、上传落后或关闭超时会提示，云端可能缺点，训练继续。本地日志与checkpoint是恢复和科学分析依据，不以云端记录恢复模型。SDK日志在RUN_BASE/swanlab，scope目录之外；额外缓存空间须实机检查。云端只显式记录配置和标量，不上传模型/语料。当前本地测试不构成ACP账号与联网验收。
