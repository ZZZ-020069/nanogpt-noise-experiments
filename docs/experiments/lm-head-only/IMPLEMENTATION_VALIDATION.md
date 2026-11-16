# 当前实现验证

2026-11-16：以远端 `67913b0` 为修复基线，本地 **58 项相关检查全部通过，耗时 89.001 秒**。其中新增 6 项覆盖两个布局的 30% 扩块门槛、数据稳定性优先、零值/非有限谱矩拒绝、CLI 拟合失败退出、冻结拟合前拒绝留出，以及 10 块拟合配 5 块留出的实际路由。保留的检查覆盖完整 Head 冻结范围、G/T/K 数学、T/K 快路径 CPU 对照、12 组目标×优化器断点恢复、micro、队列、上传模拟与 Bash 启动入口。

复现命令（在仓库根目录的兼容 CPU 测试环境中运行；Windows 需 Git Bash 位于 PATH）：

```text
python -B -m unittest tests.test_telemetry tests.test_lm_head_telemetry tests.test_lm_head tests.test_lm_head_micro tests.test_lm_head_shell tests.test_lm_head_h100 tests.test_lm_head_speedup tests.test_lm_head_fit_gate -v
```

本轮设置 `CUDA_VISIBLE_DEVICES` 为空，使用 CPU PyTorch 2.9.1 和合成数据；上传测试使用模拟 SDK。没有访问训练服务器、启动 GPU、读取真实 Key 或验证 SwanLab 云端。Windows 的 Git Bash 测试使用已获准的 MSYS 执行环境。

操作手册共 23 个 Bash 命令块、两个启动脚本和环境示例均通过 `bash -n`；每块有完整环境前缀和步骤前用时说明。该检查只解析命令，没有执行手册中的 GPU 工作。新旧路径核对、当前代码与文件 SHA-256 核对及主设计快照来源核对一并完成。训练耗时统一标记为有计算清单的旧实现排期参照，新 T/K 的 H100 吞吐尚待实测。

修复版的代码身份改变，当前清单以内容哈希标识，不再指向旧 v3 发布标签。新部署使用 `20261116-r1` 输出树，重新生成 K 和 smoke 凭证；已有任务继续其原代码、原环境与原目录。原 last-k 的代码、发布设计及旧 LM Head 历史快照保持原字节。

H100 仍需用正式输入完成当前代码的 G/T/K smoke，并人工确认 SwanLab 网页收到各四优化器指标。CPU 回归不替代 Triton CUDA、H100 显存/吞吐和真实云端送达验收。

历史验证：2026-11-15 的 SwanLab 接入版本通过 45 项 CPU/模拟检查（59.898 秒）；此前加速提交另报告了 RTX BF16 短测和恢复结果，见 README。本轮未复测这些 GPU 记录，不把它们计入上述 58 项。
