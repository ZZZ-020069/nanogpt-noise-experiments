# RTX 选档执行记录 — 2026-11-15

当前状态：双卡选档代码已实现，真实选档尚未启动。缺少与 H100 last-k 相同的已登记 v1.8 输入包；服务器现有 v1.4 包不能替代。K-approved.pt 也尚未生成，需要正确输入到位后执行初始化拟合与独立留出验收。

## 已核对的服务器状态

北京时间 11:13 和 11:20 通过 SSH 检查 enine，两次均无 GPU 计算进程。11:20:46 的状态如下：

| GPU | 型号 | 总显存 MiB | 已占 MiB | 利用率 |
|---|---|---:|---:|---:|
| 0 | RTX PRO 6000 Blackwell Workstation Edition | 97887 | 200 | 0% |
| 1 | RTX PRO 6000 Blackwell Workstation Edition | 97887 | 15 | 0% |

驱动 580.173.02；现有 Python 环境为 /experiments/xuyangzhang/venvs/quadratic/bin/python，实测 PyTorch 2.9.1+cu128、CUDA 可用。/experiments 可用约 11 TB。这是检查时的状态，实际启动前脚本会再次检查资源。

/experiments/xuyangzhang/nanogpt-inputs/frozen-artifacts/identity.json 的 design=v1.4、steps=4960，input_id=9cabd9c54c6d717354f8f6750715f9280c5502f0a85a9032806b9c7956cbd76e，目录内没有 registration.json。本机两个旧冻结输入副本也为 v1.4。已向用户请求当前 H100 的已登记输入位置，未修改旧包身份，也未重新生成初始化来代替。

## 本次实现

scripts/select_lm_head_micro.sh 调用独立 Python 监督脚本，使用第一张卡运行 micro=16、第二张卡运行 micro=32。每档依次执行 G/T/K，各目标四优化器、32 步。训练实现、优化器、噪声、曲率阈值和 last-k 代码均保持原版本。

输出根登记输入、训练代码、监督脚本哈希、软件与 GPU UUID；每次失败重试创建新 attempt，保留旧日志与权重。汇总重新核对四优化器凭证、run ID、终点状态及 0→32 步的完整时间。缺少 K、部分时间、数值失败或基础设施错误都不会产生 selected_micro。明确 OOM 可排除该档位；满足三目标全通过和 allocated 峰值不超过总显存 90% 的档位中，取十二次 smoke 总耗时较短者。

H100 接手命令先验证 selection.json 的输入、训练代码、监督脚本及 K 哈希，再输出选定 micro。H100 在全新正式根目录完成 G/T/K 验收后，才由明确命令启动正式训练。操作命令统一见仓库 LM_HEAD_RUN.md。

本记录来自工作区唯一主设计第 14.18 节，属于实现与硬件检查记录，不是正式结果或已完成的选档报告。

## 本地验证

现有 LM Head 数学、恢复、micro 布局与 Bash 路由 25 项，加上最初 9 项选档检查，共 34 项通过，耗时 68.007 秒。增加 H100 接手哈希核对后，选档 10 项全部通过，耗时 4.224 秒；本轮合计 35 个不同检查。未修改训练代码，无需将此前 154 项全仓回归重复计入本次检查。

新检查覆盖不完整 K、两档总耗时比较、显存余量、明确 OOM 与其他失败的区别、恢复后的部分计时拒绝、错误 K 哈希、缺失优化器、繁忙 GPU、旧输入在 GPU 检查前拒绝、两卡命令分配、显式失败重试和 H100 接手。这些是合成证据与 CPU 测试，不是 RTX 吞吐测量。

## RTX 部署验收

实现提交 2e8c723be45dba655d12192f1e978639f8e151da 已推送独立分支，并部署到 /experiments/xuyangzhang/nanogpt-lm-head-rtx-20261115-2e8c723。该目录为固定代码副本；未修改旧 NanoGPT、二次实验或其他运行目录。上传 tar 的 SHA256 为 62c32eef8053ac1c36d835e83b5bf578bef4c117d73bc34b534732946497d5ab，服务器验证一致。训练代码哈希仍为 b5244224a903fd65ec1c76f44f6ff0a4c50ea25a814b75cc27aab5921e51c480。

服务器以 CUDA_VISIBLE_DEVICES 为空运行的 10 项选档测试全部通过，耗时 0.225 秒；Bash 语法和 CLI help 通过。实际输入预检报缺少 registration.json，退出后未创建 /experiments/xuyangzhang/lm-head-selection-20261115，nvidia-smi 无计算进程。没有训练或曲率计算在后台运行。

两份原始日志已保存到工作区 VSCode下载/New/timing-evidence/lm-head-rtx-20261115。下一步取得当前 H100 的完整已登记输入包，先核对 input_id 与 SHA256SUMS，再按启动说明运行选档。
