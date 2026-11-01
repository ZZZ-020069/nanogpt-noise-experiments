# NanoGPT 噪声实验：v1.8，先运行 last1

**运行者只需阅读 [RUN_TASK.md：唯一操作指南](RUN_TASK.md)。** 按其中第1至6节依次完成：旧结果打包 → 新代码准备 → 配置密钥 → 一卡冒烟 → 四卡正式运行 → 中断续跑。所有ACP启动命令、路径和版本检查都在该文件中。

本README提供项目与实现概况，不维护第二套启动流程。

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

## 实现与验收

- 需要Linux、Python≥3.11、CUDA版PyTorch≥2.6,<3、NumPy、matplotlib和支持BF16的GPU。ACP入口检查环境，启用SwanLab后安装其SDK。
- 四张GPU运行四个独立候选，空闲GPU从按噪声行优先排列的队列领取下一搜索；每搜索内部顺序完成八个LR候选。
- 数据输入首次在服务器制备并登记，所有候选复用同一份输入。已登记v1.8输入无需重做。
- 本地checkpoint用于恢复，模型、优化器、步数和配对随机性随断点恢复。Linux使用持久.lease上的内核flock，禁止手工删除.lease。
- 每job最终仅保留winner的完整checkpoint，候选诊断无损压缩。last1静态空间估算84.03GB，另计smoke、阶段报告、SwanLab缓存与环境；实际存储需目标平台核验。
- 自动报告每5分钟及完整行完成后更新RUN_BASE/reports/last1/latest.zip，含离线HTML、总表与候选诊断。SwanLab上传失败保留本地记录，完整科学分析依据本地文件。
- 最终scope_manifest.json用于逐文件SHA-256校验；用户下载直接接收D盘，校验成功前保留服务器结果。只在last1完成并分析后决定下一阶段。

本地回归命令为`python -m unittest discover -s tests -v`。[验证记录](docs/VALIDATION.md)区分CPU/CI检查与尚待执行的目标硬件验收。联合pilot与双RTX测速是历史或探索工具，不属于当前正式启动流程。

模型参考nanoGPT，保留[MIT声明](third_party/nanogpt-LICENSE)。
