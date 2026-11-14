# LM Head-only：实验设计审阅摘要

**版本：LH-QP-20261114-draft2。这里只发布设计，不提供新的可运行训练入口。**

研究问题：冻结随机初始化的 backbone，只训练整个 LM Head；采样它的 Hessian，并检验二次函数能否复现优化器排序。

## 与 last-k 保持一致

保持架构、语料、初始化数值、数据窗口、全局 batch、每候选约 2B tokens / 3815 步、八次学习率搜索、原调度、35 个噪声节点、单 seed、BF16 autocast、FP32 主参数与梯度、无裁剪以及原优化器超参数。

仍按完整预算终点的 **validation loss** 选学习率和排名。高精度 Hessian 计算是离线诊断，不改变训练精度。

## 本次必要变化

1. 训练范围改为整个 50304×768 的 LM Head，共 38,633,472 个参数。
2. Head 从输入 embedding 同值复制并解除参数共享，保证 embedding 真正冻结。
3. Muon 直接更新整个 Head，记为 Muon-head；其余超参数沿用，明确这项参数组差异。
4. 新增 Hessian 采样及必要快照。采样不会把训练范围缩成子块。

## 执行顺序

先完成四优化器的 clean 对照，再测初始化和训练过程中完整 Head 的曲率。参数子块只用于测量；其谱不直接等于全 Head 的谱。

二次部分先建立保留完整矩阵形状的 Taylor 参照，再检验统计结构代理。全维代理可以把权重放回原模型，用同一 validation CE 做主要比较；内部二次目标另作诊断。测量规模及代理验收阈值仍是草案建议。

clean 是原 35 点之一，不额外重复训练。后续噪声面使用原网格；多种子确认属于可选扩展，不计入本轮单 seed 基础预算。

## 隔离与文件入口

- [完整方案](EXPERIMENT_DESIGN.md)：所有参数、公式、采样、评价与阶段判据。
- [来源及校验清单](manifest.json)：唯一来源、基线提交、快照 SHA-256。
- [旧 v1.8 设计](https://github.com/ZZZ-020069/nanogpt-noise-experiments/blob/bb3cf72e1113d0e9ceb903515772bfb1d1161df9/docs/nanogpt_experiment_design.md)：保留原文。
- [旧运行指南](https://github.com/ZZZ-020069/nanogpt-noise-experiments/blob/bb3cf72e1113d0e9ceb903515772bfb1d1161df9/RUN_TASK.md)：现有 last1 入口保持原样。

新方案位于 design/lm-head-only-20261114 分支的独立版本目录。main、旧设计、训练代码和配置均不修改，本次不合并、不部署。未来实现与结果必须使用独立实验身份，不能复用 last1 输出目录或 checkpoint 续训。

唯一编辑来源仍是工作区主设计第 14 章；这里是生成的发布快照。后续先改主设计，再导出新版本，避免两套设计分别演进。
