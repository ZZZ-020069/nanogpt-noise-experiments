# LM Head-only 实现与验证记录

> 历史实现/验证快照。当前最终档位为 micro=16（用户决定），RTX 实测已完成，H100 验收待执行；最新命令见仓库 LM_HEAD_RUN.md。下文保留原日期记录。

2026-11-14，依据工作区主设计第 14 章实现 LH-QP-20261114-draft2。启动入口见 [LM_HEAD_RUN.md](../../../LM_HEAD_RUN.md)。实现身份与文件哈希见 [implementation-manifest.json](implementation-manifest.json)。

## 已交付

- 完整 LM Head 的 G/T/K 训练、同值解除共享、冻结校验和 Muon-head。
- 原预算及 B=8 搜索、逐候选恢复、动态 GPU 队列、持久暂停、独立输出身份。
- FP64 Hessian 主子块、全词表概率、谱矩抽样校正、拟合/留出检查、K 产物门槛及后期漂移测量。
- 真实 validation CE 排名、平局/失败/边界标记、候选集内训练目标诊断与 CSV/JSON 比较。
- Bash 启动入口、README、操作说明和 smoke 性能记录。

原 v1.8 源码、脚本、配置、RUN_TASK.md 与设计快照保持不变。新增功能集中在 nanogpt_noise/lm_head/，运行目录不接受旧 last1 checkpoint。

## micro batch 调整的检查

根据用户提供的 micro_batch_adjustment_guide.md，将所有 LM Head 入口默认 micro 改为 16，并新增运行根目录及跨目标报告的一致性校验。主设计登记在第 14.17 节，发布快照见 [IMPLEMENTATION_RECORD.md](IMPLEMENTATION_RECORD.md)。T/K 算子、梯度累积与噪声注入位置未改。

**最终完整回归：154 项，147 通过、7 项 POSIX 检查跳过，314.299 秒，退出码 0。** 结构化记录及日志哈希见 [micro-batch-validation.json](micro-batch-validation.json)。

新增 8 项 Python 检查和 1 项 Bash 路由检查已通过，覆盖：

- 训练、队列、根登记、CLI、离线测量与脚本默认值，以及显式 micro=32；
- 512 条全局 batch、16/32 对应的累积次数、524288 tokens/step 和 3815 步；
- G/T/K 混用 micro、旧登记缺少 micro、无效 micro 和改变 micro 续训时的拒绝行为；
- 报告核对全部候选，包括未获胜的候选；拒绝后不写入新的比较结果；
- 200 条评估序列的实际尾块加权，以及固定 16 个完整 batch 的梯度诊断；
- micro=16/32 的冻结特征和 Hessian 主子块数值对照，含非满分块。

首次全量回归共 154 项：146 通过、7 项 POSIX 检查跳过、1 项原有并发顺序断言失败，用时 330.617 秒。失败项为 test_idle_gpu_runs_later_rows_while_first_row_is_blocked：最后完成行为 25，而断言要求为 0。该测试、旧 pool.py 和 progress.py 均未修改。测试只在第 34 行完成时释放第 0 行，不能保证其余行此前都完成，因而该断言依赖线程时序。

同一失败项独立连续复跑 10 次均通过，耗时 0.091 秒。首次失败与复跑结果均保留，未改旧调度逻辑来制造通过结果。

随后在代码不变的条件下重新执行完整 unittest discovery，得到上面的最终通过结果。运行时将 Git Bash 放在 PATH 首位，使用可正常执行原子替换的工作区临时目录；这轮 Bash 测试没有使用 Windows Apps 占位入口。

H100 上 micro=16/32 的 G/T/K 共同选档尚未执行；本地 CPU 与脚本 stub 只验证实现，不证明显存足够或吞吐提升。

## 初次实现的本地检查

环境：Windows，Python 3.11，PyTorch 2.9.1+cpu、NumPy 2.1.3、Git Bash。合计 **138 项通过，7 项 Linux/POSIX 专属检查跳过**，按不同测试项去重计数。

| 检查 | 结果 |
|---|---|
| 全量发现，当时共 140 项 | 103 通过、7 跳过；30 项旧 Bash 测试遇到本机 Windows Apps 占位入口错误 |
| 指定 Git Bash 重跑上述 30 项 | 全部通过，59.114 秒 |
| 新增 LM Head 数学、恢复、队列与测量测试 | 全量扫描中的 11 项通过；补充后期 checkpoint 测量 1 项通过 |
| 新增 Bash 阶段/路径/GPU/K/暂停恢复路由 | 4 项通过，1.078 秒；使用打印参数的 stub |
| 子块零范数处理及偶数块中位数修正后 | 针对性曲率流程测试复跑通过 |
| Bash 语法、help、Python CLI help、只读 plan | 通过 |

新增数学检查以自动微分的完整小 Hessian 核对解析主块；枚举矩形抽样核对 trace 与二阶谱矩的纳入概率。T 在起点匹配梯度和 Hessian，K 匹配起点梯度及定义的 Kronecker Hessian。12 组 G/T/K × 四优化器在同一含噪节点上比较中断与连续训练的最终 Head、优化器状态和验证 CE。

队列测试实际启动 CPU 子进程，完成四个 B=8 搜索，再复跑同一任务，确认复用终态、保留赢家和哈希不变。Windows 虚拟环境启动器带来的中间父进程问题已修复；Windows 用持有的进程句柄监测控制器，Linux 沿用父进程检查。

诊断测试还覆盖数据前缀、参数块前缀与 RNG 隔离、输入/输出隔离、失败 K 不得训练、后期参考点使用初始化的块前缀且不能生成新的 K-approved.pt。

复现命令：

~~~bash
python -m unittest discover -s tests -v
bash -n scripts/run_lm_head.sh
python -m nanogpt_noise.lm_head --help
~~~

Windows 应先将 Git Bash 加入 PATH，避免 Windows Apps 的 Bash 占位入口；测试临时目录应可正常执行原子文件替换。上述环境故障未通过修改旧研究代码绕过。

## 尚待目标硬件验收

本地结果不覆盖 GPU BF16 数值、完整 Head 的显存和吞吐、Linux SIGTERM/SIGKILL 恢复、实际磁盘峰值。7 项跳过均来自原有 POSIX 生命周期检查。下一步在独立服务器 checkout 中依次执行 check、对应目标的四优化器 32 步 smoke；通过后才能启动 clean。

smoke 记录分项同步计时及峰值显存；曲率测量保存耗时与完整采样身份。完整尺寸的子块构造/分解性能仍需实测，不能用本地小模型耗时估算正式工期。

本次没有访问服务器、部署、启动正式训练或修改 Obsidian。20261114-draft2/ 内的“仅设计”状态描述的是当时的发布快照，当前实现状态以本记录和工作区主设计 14.16 为准。
