# NanoGPT–Quadratic Proxy 实验设计主文档

> **状态**：实验设计基线 / 实施前验证  
> **版本**：v1.4  
> **最后更新**：2026-10-08  
> **用途**：本文件是 NanoGPT–Quadratic Proxy 项目的第一份独立实验设计基线，也是后续实现与修订的唯一设计依据。读者无需查阅对话或其他实验文档即可理解本文规定的参数与流程。设计设定不代表已取得实验结果；实施检查和后续研究事项见第 12 节。

---

## 1. 项目核心研究目标

本项目的目标不是单纯比较不同优化器在 NanoGPT 上的性能，而是：

> **把 124M NanoGPT 作为真实神经网络参照系，寻找一类更简单的二次函数，使这些二次函数能够复现 NanoGPT 在不同噪声条件下表现出的优化性质、优化器偏好和稳定性结构。**

希望最终回答的问题包括：

1. 真实神经网络在什么条件下偏好不同优化器？
2. SGD、Adam、AdamW、Muon 对不同类型梯度噪声的敏感性如何？
3. 不同噪声机制之间是否存在显著交互作用？
4. 当可训练参数从局部层逐渐扩展到全参数时，优化器偏好如何变化？
5. 能否通过一个具有合适 Hessian spectrum、condition number、anisotropy、dimension 和 noise geometry 的二次函数，复现 NanoGPT 的上述性质？
6. 如果可以，哪些二次结构是决定神经网络优化行为的关键因素？

因此，NanoGPT 的主要角色是提供 **高可信度的神经网络 ground truth / response surface**，二次函数则承担后续低成本、可解释的大规模探索。

研究背景是量化与低精度计算：通过可控的梯度误差环境，研究哪些优化器对不同误差机制更稳健，再寻找能复现真实网络响应的二次函数。人工噪声是低精度误差的代理模型，不直接等同于某个具体位宽。比较时对不同优化器采用相同的噪声参数，而让实际噪声相对于其梯度的比例成为观测结果。

---

## 2. NanoGPT 基础设定

当前确定基础模型：

- 模型：NanoGPT / GPT-2 124M 级别
- 训练规模：约 **2.6B tokens**
- 优化器：
  - SGD
  - Adam
  - AdamW
  - Muon
- 梯度噪声由两个独立控制量描述：
  - σ₁
  - σ₂
- 噪声网格：
  - σ₁：5 个水平
  - σ₂：7 个水平
  - 总计 **5 × 7 = 35 个二维噪声节点**
- 每个噪声节点对不同优化器都允许寻找各自最合适的学习率。

七个 σ₂ 水平采用覆盖较宽范围的固定绝对数值 `{0, 0.0001, 0.001, 0.01, 0.1, 1, 10}`。二次函数与 NanoGPT 共用同一组数值，各优化器及 LR 候选也使用相同网格；不按梯度范数换算或动态缩放。二维网格第二轴直接为绝对 σ₂。

现阶段只运行一个 seed；每个节点的每个候选学习率均按完整 2.6B tokens 分配预算；所有实验均不使用梯度裁剪。

重要原则：

> 35 个节点不是一维“噪声逐渐增大”的扫描，而是一个真正的 **σ₁ × σ₂ 二维 response surface**。

### 2.1 模型、数据与训练基础

所有 NanoGPT 候选从统一未训练初始权重开始，使用相同数据与完整训练预算。scope 指可训练参数集合，取 full 或 last-k；run 指一个 scope、优化器、噪声坐标和候选学习率确定的一条独立训练轨迹。

| 项目 | 当前固定设置 |
|---|---|
| 架构 | 12 blocks、12 heads、768 hidden、1024 context、50304 vocab，`bias=False`、`dropout=0.0`，token embedding 与 LM head 权重共享 |
| 数据 | 数据集标识 `openwebtext_subset`：OpenWebText 训练子集 1,000,000,000 tokens，验证子集 10,000,000 tokens；GPT-2 tiktoken 编码、文档末尾 EOT、uint16 存储；制备 seed=2357、shuffle buffer=10000 |
| 数据身份 | 统一固定 train.bin/val.bin/meta.json，记录文件哈希、源数据 revision 和 tokenizer/依赖版本；全部 run 读取同一份产物，不按 run 重新抽取子集 |
| 全局 batch | 每 optimizer step 固定 524,288 tokens，即 512 条长 1024 的序列；参考 micro-batch=4，4 卡时每卡累积 32 次 |
| 训练预算 | 目标 2,600,000,000 tokens；按整步向上取整，共 4960 steps，实际 2,600,468,480 tokens。所有正常候选相同，评估 tokens 不计入训练预算 |
| 初始化 | `scratch`，seed=1337；Linear/Embedding 权重按均值 0、标准差 0.02 的正态分布初始化，残差 c_proj 权重标准差为 `0.02/sqrt(24)`，LayerNorm scale=1；各 scope 从同一份未训练初始权重开始，优化器状态重新初始化 |
| 数值精度 | BF16 autocast，模型主参数和累积梯度 FP32；噪声在 FP32 梯度上注入；Muon 的 Newton–Schulz 使用 BF16 运算。记录硬件、依赖版本和实现哈希 |

数据源为 `Skylion007/openwebtext` 的 `train` split，固定 source revision 后流式读取，使用 seed=2357、buffer=10000 打乱文档。按 `tiktoken:gpt2` 的 `encode_ordinary` 编码，在每篇文档末尾追加 EOT；先取得 10,000,000 个验证 token，再从后续文档取得 1,000,000,000 个训练 token。达到任一配额时截断边界文档并丢弃余下部分，保证同一源文档不跨训练/验证划分。GPT-2 tokenizer 有 50257 个 token，模型词表维度补齐为 50304。数据产物在所有任务启动前制备并固定。

训练 token 数包含对固定训练子集的重复采样，不表示有 2.6B 个互异数据 token。调整卡数或 micro-batch 时保持全局 batch 精确一致，数据配对按第 6 节实现。

### 2.2 优化器配方与搜索中心

表中 η₀ 是峰值学习率的搜索中心，不预设为最优值。所有 scope 使用同一组搜索中心，各节点独立选择完整预算末尾验证损失最小的有效候选，称为 winner。

| 优化器 | 峰值基准 η₀ | 固定配方 |
|---|---:|---|
| AdamW | 0.0006 | betas=(0.9,0.95)，eps=1e-8，解耦 weight decay=0.1，非 AMSGrad |
| Adam | 0.0006 | betas=(0.9,0.95)，eps=1e-8，coupled weight decay=0.1，非 AMSGrad |
| SGD-Momentum | 0.01 | momentum=0.9，dampening=0，nesterov=False，coupled weight decay=1e-4 |
| Muon | 0.02 | momentum=0.95，nesterov=True，5 次 Newton–Schulz，解耦 weight decay=0.01；辅助 AdamW 峰值基准=0.0006、betas=(0.9,0.95)、eps=1e-8、weight decay=0.1 |

文中 SGD 均指上述 SGD-Momentum。Adam/AdamW/SGD 仅对 ndim≥2 的可训练参数施加 decay，向量参数不 decay。Muon 仅处理 Transformer block 内二维矩阵，其余可训练参数交给辅助 AdamW；辅助组采用同样的矩阵/向量 decay 分组。last-k 的 block 内 LayerNorm 仍属于辅助组。Muon 使用矩阵形状 LR 修正 `sqrt(max(1, rows/cols))`、Newton–Schulz 系数 `(3.4445,-4.7750,2.0315)` 和归一化 epsilon=1e-7；该内部归一化属于 Muon 算法，不是额外梯度裁剪。

Adam、AdamW 和辅助 AdamW 使用带一阶/二阶矩偏差修正的更新；epsilon 加在偏差修正后二阶矩平方根之外。SGD 的动量递推为 `m_t=0.9*m_(t-1)+(g̃_t+λθ_t)`，初始动量为零，更新 `θ_(t+1)=θ_t-lr_t*m_t`。coupled decay 在噪声注入之后、动量/矩估计之前加入；解耦 decay 直接作用于参数。

Muon 动量初始为零，递推 `m_t=0.95*m_(t-1)+0.05*g̃_t`，Nesterov 输入为 `u_t=0.05*g̃_t+0.95*m_t`。将 u_t 转为 BF16，若行数大于列数则转置，以 Frobenius 范数加 1e-7 归一化为 X；重复五次 `A=X*Xᵀ; X=a*X+(b*A+c*A*A)*X`，其中 `(a,b,c)=(3.4445,-4.7750,2.0315)`，最后还原转置得到 Q。参数更新为 `θ_(t+1)=(1-lr_t*λ)*θ_t-lr_t*sqrt(max(1,rows/cols))*Q`，其中形状取原矩阵维度，decay 使用未做形状修正的 lr_t。

Muon 搜索一个共同倍数，同时缩放主 LR 和辅助 LR，固定辅助/主 LR=0.03；该绑定应用于全部 scope、噪声节点和 LR 候选，不额外搜索辅助 LR 或执行比例标定。比较结论针对这些固定优化器配方，包括其 weight decay 与辅助更新机制。

---

## 3. 实验族与可训练参数

实验包含四类条件：

1. 从初始化开始不加噪声，全参数训练；
2. 从初始化开始不加噪声，仅训练最后 k 层；（作为Baseline）
3. 从初始化开始加噪声，全参数训练；
4. 从初始化开始加噪声，仅训练最后 k 层。

按可训练参数集合划分为以下两个实验族；每个实验族的零噪声节点提供对应 baseline。

### 3.1 Full-parameter family

模型从统一初始化开始，全参数训练并对全部参数的梯度加噪，包括 12 个 block、token/position embedding、final LayerNorm 和共享 LM head；共享参数只计一次。按第 2 节架构共有 124,373,760 个独立参数。

每个实验节点为：

\[
(\sigma_1,\sigma_2,\text{optimizer},\eta)
\]

其中：

- σ₁ × σ₂ 构成 5×7 网格；
- optimizer ∈ {SGD, Adam, AdamW, Muon}；
- η 通过该节点自己的学习率搜索确定。

最终完整结果数量：

\[
35\times4=140
\]

目标为 **140 条选出的 2.6B-token 训练轨迹**。每个节点搜索八个学习率，候选本身均跑完整预算，winner 直接从这些轨迹中选出，不需为同一配置额外重复训练；若某节点八次均失败，则保留失败标签而没有有效 winner。

5×7 网格中保留 clean baseline：

\[
(\sigma_1,\sigma_2)=(0,0),
\]

该点本身就是“不加噪声、全参数训练”的 baseline，不需要再额外训练一套独立模型。

---

### 3.2 Last-k-layer family

从统一参数初始化开始，仅允许最后 k 个 Transformer block 学习，其余参数冻结。

本阶段采用稀疏深度节点：

\[
k\in\{1,2,3,6,12\}
\]

Transformer block 按 0–11 编号，last-k 的可训练集合为 `transformer.h.{12-k}.` 至 `transformer.h.11.` 下的全部参数，包括注意力、MLP 和 block 内 LayerNorm。仅这些参数设置 `requires_grad=True`，参与优化和梯度噪声注入。

token embedding、position embedding、final LayerNorm 以及与 token embedding 共享权重的 LM head 全部冻结在统一初始化值。冻结参数仍参与前向运算及必要的梯度传播，但不产生参数更新。**last-12 训练全部 12 个 block，仍冻结 block 外参数，因此与 full-parameter 是两个独立条件。** 实现必须输出并验证可训练参数名称、形状、数量和冻结参数校验和。

对于每个 k：

- 仍然研究 σ₁ × σ₂；
- 仍然使用 4 个优化器；
- 仍然允许每个节点独立选择最合适学习率。

网格中的 (0,0) 节点就是“不加噪声、仅训练最后 k 层”的 baseline，无需额外重复。

---

## 4. 二维噪声网格的地位

### 4.1 Full-parameter family：完整运行 5×7

Full-parameter NanoGPT 是后续 quadratic proxy 的最重要 ground truth，已确定：

> **最终完整保留 5×7=35 个噪声节点。**

对于每个优化器 O，希望得到二维响应：

\[
R_O(\sigma_1,\sigma_2).
\]

响应量可以包括：

- validation loss；
- training loss；
- convergence speed；
- gradient norm；
- update norm；
- parameter displacement；
- noise-to-gradient ratio；
- divergence / stability；
- divergence time；
- 最优学习率；
- optimizer ranking；
- 其它后续确定的诊断指标。

重点观察二维曲面的：

- ridge；
- valley；
- phase transition；
- stability boundary；
- optimizer-ranking boundary；
- σ₁ 与 σ₂ 的 interaction effect。

---

### 4.2 Last-k family：每个 k 完整运行 5×7 网格

Last-k 的主要作用是研究：

> 当神经网络“可优化自由度”逐渐增加时，哪些优化器偏好开始出现，以及它们如何逐渐接近 full NanoGPT。

对每个已确定的 `k ∈ {1,2,3,6,12}`，直接运行完整的 5×7=35 个噪声节点，每个 k、optimizer 和噪声节点仍独立搜索学习率，采用与 full-parameter family 相同的训练预算、单 seed、no-clipping 和评估口径。

因此，每个 k 包含：

\[
35\times4=140
\]

个 optimizer/noise 配置；五个 k 共计 700 个配置。完整网格使不同 k、full-parameter 与 quadratic proxy 能在完全相同的 35 个坐标上直接比较响应面、优化器排序和稳定性边界。

### 4.3 σ₂ 网格：两类任务共用宽范围绝对数值

已确定采用固定绝对 σ₂：二次函数与 NanoGPT 使用同一组七点，各 optimizer、LR 候选及训练时间均不改变某一节点的 σ₂。σ₂ 不依赖该轨迹的梯度范数。所有 last-k 设置均直接使用完整公共网格，不另行缩放。

七个正式节点冻结为：

\[
\sigma_2\in\{0,10^{-4},10^{-3},10^{-2},10^{-1},1,10\}.
\]

零点保留 clean/纯相对噪声对照，其余六点按十倍间隔覆盖五个数量级。该范围用于同时覆盖两类任务的多种绝对噪声尺度；稳定区域与边界位置由实验测量确定。高端节点可能成为强噪声压力测试；十倍间距也可能遗漏较窄的排序变化区间，后续补点须遵循统一绝对坐标并单独记录预算。

解释上，σ₁ 控制随梯度尺度变化的误差，σ₂ 保持绝对误差尺度。数学定义见第 4.4 节；σ₂ 表示 additive 项的总均方根范数，各向同性构造下该项每元素标准差为 `σ₂ / sqrt(D)`。相同 σ₂ 不表示不同维度下逐元素误差相同，也不直接表示相同量化位宽。

梯度范数、实际 noise-to-gradient ratio 继续作为诊断记录，不能用于回调 σ₂。范数在所选可训练参数上、梯度累积/同步/unscale 后且人工噪声注入前测量。梯度随训练变小时，固定 σ₂ 的相对影响自然增强，这是需要保留的优化现象。

---

### 4.4 σ₁ 五点与噪声公式

σ₁ 固定为 `{0,0.03,0.1,0.3,0.9}`，覆盖零相对噪声及逐渐增强的相对噪声。σ₂=0 时，理论噪声 RMS 分别为梯度范数的 0%、3%、10%、30% 和 90%。与第 4.3 节七个 σ₂ 做笛卡尔积，两类任务、全部 scope 和优化器共用 35 节点，按 σ₁-major、σ₂-minor 排序。

令 g_t 为当前可训练参数拼接后的数据损失梯度，D 为其独立标量元素总数；冻结参数排除，共享权重去重，不含 optimizer 内部的 weight decay。采用各向同性 Gaussian 模型：

\[
z_t\sim\mathcal N(0,I_D),\qquad
\xi_t=\sqrt{\frac{\sigma_1^2\lVert g_t\rVert_2^2+\sigma_2^2}{D}}\,z_t,\qquad
\widetilde g_t=g_t+\xi_t.
\]

因此 `E[||ξ_t||² | g_t]=σ₁²||g_t||²+σ₂²`。两项按方差相加，用一个标准 Gaussian 向量采样，跨候选配对规则见第 6 节；不逐层归一化，不强行把采样噪声范数归一到理论值。D 在每条轨迹内固定；正常应有梯度的可训练参数若出现 grad=None，应作为实现异常检查，不静默改变 D。二次函数采用相同公式，其 D 为实际优化变量数。

每步在全部 micro-batch 累积、DDP 梯度平均和必要的 unscale 后注入一次；各 rank 注入相同噪声，然后执行 optimizer（及其内部 decay），无额外裁剪。clean 节点仍记录梯度诊断，但不添加噪声。

---

## 5. 学习率搜索：每节点八次完整预算

### 5.1 科学要求

必须允许：

> 每一个 (σ₁, σ₂, optimizer) 节点拥有自己的最佳学习率。

不能因为给某个 optimizer 一个不合适的 LR，而错误判断其在该噪声条件下不稳定或性能较差。

**每个候选学习率均分配完整约 2.6B tokens**，每节点统一测试八个候选，即五个初始点加三次追加；单条 run 的 scheduler 见第 5.5 节。

### 5.2 五个初始节点

以第 2.2 节各 optimizer 的基准学习率 `η₀` 为中心，初始范围固定覆盖 `[0.1η₀, 10η₀]`；所有 scope 共用搜索中心，独立选择各自的最佳 LR。

初始五点为：

\[
x=\log_{10}(\eta/\eta_0)\in\{-1,-0.5,0,0.5,1\},\qquad
\eta/\eta_0\in\{0.1,10^{-0.5},1,10^{0.5},10\}.
\]

五点在 log10 坐标中等距，步长 Δx=0.5，相邻学习率相差 `sqrt(10)≈3.16228` 倍。Adam/AdamW 的峰值 LR 约为 `{0.00006,0.000189737,0.0006,0.00189737,0.006}`，SGD-Momentum 约为 `{0.001,0.00316228,0.01,0.0316228,0.1}`，Muon 主组约为 `{0.002,0.00632456,0.02,0.0632456,0.2}`；Muon 辅助组与 AdamW 使用相同五档，并与主组按共同倍数配对。实现按 `η=η₀*10^x` 计算，展示的小数仅作近似。

### 5.3 三次追加预算：扩边与加密规则

初始五点与本节追加策略均为正式运行规则。每个节点保留三次追加、合计八次完整预算，逐次根据已完成结果提出下一个候选。

在 `x=log10(η/η₀)` 空间内，以初始步长 Δx=0.5 向外延伸；即上界乘 sqrt(10)，下界除以 sqrt(10)。连续扩边时保持 Δx=0.5，不自动减半；区间加密采用对数中点。具体规则为：

1. 按第 10 节的完整预算终点验证 loss 选择 winner；边界指所有已测 LR（包括数值失败点）的最小/最大值。winner 位于最小或最大已测 LR 时，向该侧扩展一个 Δx=0.5 的步长。
2. 扩边点完成后，若它以超过 1e-6 的 loss 改善成为新的边界 winner，下一次继续同方向延伸 0.5。若它变差、数值失败或仅与原 winner 数值平局，下一次优先测试原边界 winner 与该扩边点的对数中点；不把平局当作继续向外扩展的证据。该中点测试后恢复一般 winner/区间规则。
3. 没有上述扩边反馈需要处理且 winner 位于内部时，比较其与左右最近已测点的 log10 间距，在较宽区间取中点；等宽优先较小 LR 一侧。对数中点对应 `η_new=sqrt(η_left*η_right)`。数值失败点可以作为探索区间端点，但不参加 winner 评选，也不表示其外侧所有 LR 必然失败。
4. 若全部已测候选均数值失败，向最小已测 LR 以下延伸 0.5，即除以 sqrt(10)，继续寻找有效候选；这只是统一探索方向，不保证减小 LR 能解决失败。剩余预算用完仍无有效结果时，记录“本搜索范围未找到有效 LR”。
5. 基础设施故障恢复同一候选，不作为数值失败，不用新的 LR 顶替。所有候选使用相同 scheduler、训练预算和配对随机性。每次追加都重新计算区间，不重复测试同一 LR；三次机会用完即停止。

例如以上界 `10η₀` 为 winner，下一点为 `31.6228η₀`；若该点更差或失败，下一点为 `17.7828η₀`；若其改善成为 winner，则下一点为 `100η₀`。初始两端十倍范围不是硬边界；连续三次同向扩边最多探至 `316.228η₀` 或 `0.00316228η₀`。本规则兼顾范围探索与局部加密，不保证三次追加就能定位最优值。

报告的“最优 LR”限于已测试有效候选。预算用尽时若 winner 仍在已测边界，标注“搜索边界未闭合”；若一侧仅有失败点，另记录相邻失败 LR，避免把几何上有端点误解为已经精确定位稳定性边界。相邻噪声节点的信息只用于分析，不替代本节点五点初始搜索。

### 5.4 最优学习率本身作为研究量

对每个 optimizer，可以研究：

\[
\eta_O^*(\sigma_1,\sigma_2).
\]

即最优学习率本身形成二维 surface。

可关注：

- η* 是否随 σ₁ / σ₂ 系统变化；
- 是否存在显著 interaction；
- 不同 optimizer 的 η* surface 是否平滑程度不同；
- LR sensitivity 是否可被二次函数复现。

---

### 5.5 单条轨迹的学习率调度

采用按已处理训练 tokens 定义的线性预热和余弦衰减。搜索变量 η 是峰值 LR，前 100,000,000 tokens 从 0 线性预热至 η，随后余弦衰减，到实际预算 T=2,600,468,480 tokens 时降至 0.1η。令 t 为本次更新前已完成 tokens：预热段 `lr=η*t/100000000`；之后 `lr=η*[0.1+0.45*(1+cos(π*(t-100000000)/(T-100000000)))]`，超出 T 时取 0.1η。t=0 时首步 LR=0，仍计算梯度并更新优化器状态；不另外增加训练步数。Muon 主/辅助 LR 使用同一调度乘子。

二次函数逐节点独立调节自己的 LR，不固定为 NanoGPT 的 winner；其迭代预算和搜索中心属于 quadratic 建模协议，不把 NanoGPT 的 token 单位或 LR 数值直接移植过去。

---

## 6. Paired Randomness / Common Random Numbers

为了减少实验间随机性造成的比较方差，固定：

- 相同模型初始化；
- 相同 token sequence；
- 相同 mini-batch order；
- 相同数据预处理；
- 相同底层随机噪声 realization；
- 相同 evaluation protocol。

初始化和训练采样使用 seed=1337、noise seed=20260902、eval seed=20260829；数据制备 seed=2357。噪声使用独立 RNG，按公式 `step_seed=(20260902+1000003*completed_step) mod (2^63-1)` 派生；同 scope 下按固定去重参数顺序生成第 4.4 节 z_t，各 optimizer、节点、LR 和 DDP rank 共用底层 z_t。不同 scope 的 D 不同，不声称逐坐标噪声一一配对；两类任务也不要求跨维度相同 realization。实现不能因噪声重置全局 RNG 而改变数据顺序。

训练窗口按全局更新步骤派生：全局第 s 步固定生成 512 个窗口起点，CPU Generator seed=`(1337+97409*s) mod (2^63-1)`，从 `[0,len(train)-1024)` 均匀有放回采样，再按固定全局序列顺序分发给各 rank/micro-batch；换卡数仅改变分发，不改变全局 batch 内容。冻结实际窗口 manifest/哈希，断点恢复按全局 step 接续。共享未训练初始权重与数据索引不属于共享训练轨迹。

现阶段只跑一个 seed，不把 multi-seed confirmation 列为当前必做预算。初始化、数据顺序及噪声随机流仍须分别记录；单 seed 不表示必须让所有随机流使用相同整数。所得排序属于单次随机实现下的结果，不据此宣称跨 seed 稳健性。

---

## 7. 稳定性处理：所有实验不做梯度裁剪

### 7.1 为什么关闭梯度裁剪

所有实验（包括 NanoGPT、quadratic、clean baseline、LR 候选、标定及调试）统一设置：

\[
\texttt{grad\_clip}=0.
\]

人工噪声在梯度累积、同步和 mixed-precision unscale 完成后注入，随后不再对 noisy gradient 做范数裁剪，直接交给 optimizer。

这一决定来自实验目标本身。项目需要识别不同 optimizer 对噪声幅值、方向和稳定性边界的真实响应，而 clipping 会在 optimizer 之前额外引入一个非线性映射，压缩 noisy-gradient 的径向差异，并可能把本应出现的 divergence boundary 强行拉回稳定区。此时实验对象将不再是单独的 SGD、Adam、AdamW 或 Muon，而是各 optimizer 与 clipping 的复合算法，也难以和未使用相同裁剪规则的 quadratic proxy 对齐。

本设计不安排独立梯度裁剪对照。全部候选使用统一无裁剪协议，稳定性通过数值失败记录和完整训练结果评估。

### 7.2 Divergence Early Stop

每个候选运行完整预算。统一提前终止仅限训练/评估 loss、原始/加噪梯度、模型参数或 optimizer 状态出现 NaN/Inf；在更新前后及评估时检查，保存首个失败 step、tokens、字段与诊断。有限但很大的 loss、参数范数或 update norm 只记录告警，不凭未经验证的阈值提前淘汰，不启用 target-val-loss 达标停止。仅诊断比值因分母为零得到 Inf 不算数值失败。

失败轨迹记为 numerical_failure，实际已处理 tokens 与已成功更新步数分开记录，不算完成 2.6B 的轨迹；记录首次非有限数值的时间 T_diverge。它表示数值失败边界，不把所有有限但学不好的轨迹称为发散。

OOM、抢占、网络/存储或硬件故障记为 infrastructure_failure，恢复同一候选，不计为优化器发散，不改变 LR、噪声或 batch 以“修复”结果。若以后需要加入有限值发散阈值，必须单独修订协议，不能在看到某优化器结果后临时增加。

---

## 8. 实验阶段

当前代码实现与验证范围为 NanoGPT，包括 full-parameter 与全部 last-k 设置的数据处理、训练、学习率搜索、任务管理、日志、恢复和结果分析。以下阶段描述研究任务，不要求在开展 last-k 前完成二次函数建模；二次函数的设计与实现暂不纳入当前交付范围。

### Phase 0 — 实施准备与协议核验

依据第 2、4–7、10 节，核对实现并固定以下实验产物与配置：

- 124M architecture；
- dataset / tokenizer；
- 2.6B token budget；
- effective batch；
- initialization；
- data order；
- σ₁ / σ₂ 数学定义；
- 已冻结的七个绝对 σ₂ 数值 `{0, 0.0001, 0.001, 0.01, 0.1, 1, 10}`；
- 噪声总范数与逐元素尺度的统一维度口径；
- noise injection location；
- LR scheduler；
- evaluation protocol；
- mixed precision；
- 所有实验 `grad_clip=0`；
- weight decay；
- optimizer implementation；
- logging；
- failure criterion；
- checkpoint policy。

### Phase 1 — Full-parameter NanoGPT

目标：建立真实神经网络的完整二维 ground-truth surface。

最终：

- 35 noise nodes；
- 4 optimizers；
- 每节点独立 LR tuning；
- 不使用梯度裁剪；
- 单 seed，每个 LR 候选训练完整 2.6B tokens。

LR 搜索采用：

- 各优化器基准 LR 周围五个初始候选；
- 预留三次完整训练用于边界拓宽或区间加密；
- 总计每节点八次，不使用短预算淘汰。

### Phase 2 — Quadratic Proxy 初步建模

研究：

\[
f(x)=\frac12x^\top Hx+b^\top x
\]

需要怎样的：

- Hessian spectrum；
- condition number；
- eigenvalue distribution；
- anisotropy；
- dimension；
- initialization；
- noise geometry；

才能在相同的 no-clipping 口径下复现 NanoGPT 的 optimizer ranking、stability region、convergence pattern、η* surface、σ₁/σ₂ sensitivity、interaction effect 与 divergence boundary。

### Phase 3 — Last-k Mechanism Study

采用已确定的 k：

\[
k\in\{1,2,3,6,12\}
\]

所有这些 k 均不同于 full-parameter（见第 3.2 节）。每个 k 直接运行完整 35 点二维网格和 4 个优化器，每个节点执行相同的八个完整预算 LR 候选。

### Phase 4 — 完整网格后的针对性补点

依据 Full NanoGPT、Last-k 与 Quadratic prediction，在：

- ranking transition；
- divergence boundary；
- 强 interaction 区域；
- quadratic 与 NanoGPT 不一致区域；

在基础 35 点全部完成后，才考虑增加网格外的新 NanoGPT 实验点。补点用于提高边界分辨率，不替代任何已确定的基础节点。

### Phase 5 — 最终 Confirmation

对核心结论节点使用：

- 完整 124M；
- 完整 2.6B tokens；
- 最佳 LR；
- 当前阶段单 seed，后续多 seed 验证需另行决策；
- 固定正式实验协议。

---

## 9. 算力结构与调度原则

Full-parameter 加五个 last-k 设置共形成 6 个 NanoGPT trainable scope。每个 scope 有 35 个噪声节点和 4 个优化器，因此基础实验共有：

\[
6\times35\times4=840
\]

个 optimizer/noise/scope 配置。每个配置统一测试八个完整预算 LR，共计 6720 条候选轨迹，按约 2.6B 计名义总预算 **17.472T tokens**；按整步实际预算计为 **17.4751481856T tokens**。这不包括网格外补点与基础设施重试；winner 不额外重训，数值失败按实际消耗计。该预算是实验规模核算，不代表已分配或已获准启动这些算力任务。

实验天然属于 embarrassingly parallel workload。

不同：

- (σ₁, σ₂)；
- optimizer；
- LR；
- k；

之间大多是独立 job。

因此优先考虑多张 GPU 同时分别运行多个独立训练任务，而不一定把所有 GPU 用于一个 NanoGPT 的 DDP。

后续代码设计应支持：

- config-driven launch；
- job array；
- 自动 checkpoint / resume；
- 独立结果目录；
- 统一 metadata。

---

## 10. 评估、记录与恢复协议

### 10.1 固定评估与选择指标

评估设置为 eval_iters=50、参考 batch_size=4、block_size=1024，即每次评估使用固定的 200 条序列，共 204,800 个预测 token；这是全局评估样本量，不乘卡数。CPU Generator 对 train/val 分别使用 seed=20260829/20260830，一次性生成并保存窗口起点。所有 scope、优化器、节点、LR 和评估时刻使用相同样本，eval 模式下计算平均 token cross-entropy，不添加人工噪声。

每约 100,000,000 tokens 评估一次：在首次越过每个整数倍阈值的完整 step 后评估，另评估 step=0 和最终 step=4960；重合时只评估一次。训练和验证窗口各自固定，不能使用当步随机训练 loss 代替验证指标。

以**完整预算末尾的 validation loss**选择 LR 和节点内优化器排序，越低越好；不以训练过程中最低验证 loss 选 winner，避免不同时间点比较。若候选最终 loss 差异≤1e-6，视为选择用的数值平局，取较小 LR；跨优化器在此容差内标为并列。该容差不表示统计显著性。数值失败候选不参与有限 loss 排名；全部失败则保留失败标签，不赋一个人为的大 loss。报告每个节点的八次候选、有效数量、winner 与边界未闭合标志。

每个 optimizer/scope 的 clean 节点同样独立搜索 LR。噪声影响报告 `ΔL=L*(σ₁,σ₂)-L*(0,0)`，两项均为各自调优后的终点 loss；此量反映允许调参后的噪声影响，不是固定 LR 的纯噪声干预。收敛速度辅助指标采用相同评估时刻、以 tokens 为横轴的验证 loss 梯形 AUC/T，仅比较完整轨迹，不把失败后的空白补为零。

### 10.2 日志与诊断

所有候选保存完整评估曲线。每步保存 step、累计 tokens、当步训练 loss、主/辅助 LR、有限性状态和加噪前梯度范数；每 10 步及首末步记录完整诊断：理论噪声 RMS、实际噪声范数、加噪后梯度范数、两种 noise-to-gradient ratio、梯度夹角余弦、D、逐元素噪声标准差、可训练参数范数、实际参数更新范数、相对初始化的参数位移、`||Δθ||/||g||`。update norm 指包含 decay 和所有辅助组的实际参数变化，不是 η||g||；最后这个比值只是诊断，不等同于自适应优化器的统一有效 LR。

零分母显式记录 undefined/Inf 及原因；clean 噪声范数为零。统计均在去重后的可训练参数集合计算，参数位移相对于统一未训练初始状态。full 与 last-k 同时保存 names/shapes/count、冻结参数校验和，确认冻结参数不变；full D=124,373,760，last-k D=7,079,424×k。Muon 另存主组和辅助组的参数清单与 LR。

metadata 包括设计版本、代码/配置哈希、数据和初始权重哈希、全部随机种子、硬件/软件版本、并行布局、精度、预算与实际 tokens、失败类型、每个 LR 的提出原因。sharpness/Hessian 等额外曲率估计暂不纳入本轮必做日志；quadratic 匹配先比较节点排序、ΔL、稳定性、收敛曲线形状及最佳 LR 的相对变化，不直接等同语言模型 loss 与二次函数误差。

### 10.3 Checkpoint 与断点恢复

Checkpoint 分为 latest、best 和 final：每次评估后保存可恢复的 latest，另保留最低验证 loss 的 best 供诊断，完整预算保存 final；LR 选择只看 final。Checkpoint 包含模型、全部优化器及辅助组状态、scheduler/step/tokens、随机流状态或可重建索引、配置和数据身份；采用原子写入。只从同一候选的 latest 接续；跨 scope、optimizer、噪声节点或 LR 不共享训练后 checkpoint。成功的 winner 直接使用已完成轨迹，无需再次跑相同 seed。

### 10.4 交付物与实现约束

每个候选至少交付配置与 metadata、训练诊断日志、固定窗口评估曲线、完成/失败状态及 checkpoint。每个 optimizer/noise/scope 配置额外交付八个候选的汇总、选择依据、winner 和边界标志；实验总体交付 35 点响应面、相对 clean 的变化、优化器排序和数值失败分布。

本文给出的公式、参数集合、数值设置和执行顺序是实现验收依据。运行结果须注明设计版本；代码实现完成后记录其提交或文件哈希。尚未通过一致性检查的实现与 smoke test 输出应分别标注，不能当作正式结果。

---

## 11. 当前设计核心原则

1. **124M NanoGPT 是真实网络参照，不随意缩成 toy model。**
2. **Full-parameter 5×7 二维噪声网格完整运行。**
3. **35 点是二维 response surface，不是一维噪声扫描。**
4. **用有限 LR 候选数控制预算，不缩短正常候选训练，也不削减任何 NanoGPT scope 的基础 35 节点。**
5. **每个噪声节点仍允许每个 optimizer 独立获得最佳 LR。**
6. **每节点八个 LR 候选，每个候选完整训练约 2.6B tokens（4960 个完整步骤）。**
7. **五点初始搜索加三次预留预算，用于拓宽边界或对数区间加密。**
8. **Full-parameter 与每个 last-k 设置均完整运行相同的 35 节点。**
9. **尽可能使用 paired randomness 减少比较方差。**
10. **二次函数与 NanoGPT 共用七个固定绝对 σ₂ 水平，各优化器共用数值，不进行比例标定或映射。**
11. **所有实验关闭梯度裁剪，不增加裁剪对照。**
12. **明确发散后允许统一规则 early stop，并记录 divergence time。**
13. **不能通过共享不同噪声/优化器的训练轨迹来伪造节省。**
14. **核心观测由完整 124M、完整训练协议确认；本阶段单 seed，不宣称跨 seed 稳健性。**
15. **NanoGPT 提供高价值观测点，quadratic model 承担后续大规模机制探索。**

---

## 12. 实施检查与后续研究事项

NanoGPT 主网格的参数、Muon 学习率绑定与第 5.3 节追加策略均已确定，按本文实现。正式运行前的实施检查包括：

1. 核验/固定数据 revision、文件哈希、tokenizer 与依赖版本，以及统一初始化权重；这些实际产物标识须由文件和环境生成，不能在设计文档里编造。
2. 验证 full/last-k 参数集合、共享权重去重、Muon 分组、D、噪声注入时序及所有路径 grad_clip=0。
3. 以 smoke test 验证整步 token 预算、LR 候选生成、配对数据/噪声、评估样本一致性、数值失败标签与断点恢复；smoke 结果单独保存。
4. 二次代理的谱、条件数、维度、初始化、独立 LR 中心与迭代预算属于后续建模工作，需在对应实验运行前补充独立小节。本文固定其与 NanoGPT 共用的噪声定义、坐标和无裁剪要求，但尚不构成完整的二次函数执行协议。

当前正式预算只覆盖基础 35 节点，不预拨额外噪声补点或多 seed 重复。基础网格完成后，如需提高边界分辨率，再基于统一坐标提出追加节点和预算并修订版本；不按优化器私自调整 σ₂，也不替换基础节点。

---

### 12.1 实施产物登记 — 2026-10-08

NanoGPT 实现位于 `VSCode下载/New/nanogpt-noise-experiments/`，协议仍为 v1.4；仓库中的 `docs/nanogpt_experiment_design.md` 仅为本文件的发布快照，不另行维护。代码提交身份和实现文件哈希由每条 run 的 metadata 记录。本次没有启动正式训练或取得正式实验结果。

已对固定数据进行尺寸、token 范围和元数据核验，并生成以下 SHA-256：

| 产物 | SHA-256 |
|---|---|
| train.bin | `39c5a0ed2bf12d057704d0852fe0371dfcacc9ceeb3ffb77fec937dc6cd8a2ed` |
| val.bin | `abf1621d4671024bd734e989d4345ad49db6faf2d0fac177654713fb3e941ed4` |
| meta.json | `07ab48ebfcf871eba38be6c7b066b06c07c40d4cf8f865c9475079c7e3874c0d` |
| initial.pt | `a80eda8605609401d6775edc05f7a8e54d7ffd7cdfad0227bdbcf5b323be6988` |
| windows.pt | `b325a4f6bd65ca034398481d7203c8afca2619e4eb3b7f601fce7cf5f03d8cab` |

源数据 revision 为 `79d93d786212f7344586290adb811d4ae6a1762c`。公共产物 manifest ID 为 `9cabd9c54c6d717354f8f6750715f9280c5502f0a85a9032806b9c7956cbd76e`；初始化与窗口制备环境为 PyTorch 2.9.1+cpu、NumPy 2.1.3。语料原制备时的 tokenizer/依赖版本未保存在其 meta.json 中，当前不得补写为已知版本。该缺项与文件身份分开记录。

本地 CPU 测试只验证实现，包括模型参数集合、噪声、搜索、失败标签和恢复一致性；H100 BF16 与 Linux NCCL 分布式验收仍需在目标硬件完成。Windows Gloo 连接测试受当前构建限制未通过，不视为分布式验收通过。公共权重和数据产物不属于研究结果。

---

## 13. 基线版本与修订规则

### v1.4 — 2026-10-05

冻结第 5.3 节：采用 0.5 个 log10 单位的扩边步长及扩边反馈后的中点加密规则。确认 Muon 辅助/主 LR 固定为 0.03，不增加比例标定。当前实现范围限定为 NanoGPT 的 full 与全部 last-k 实验，二次函数留待后续设计和实现。

### v1.3 — 2026-10-05

删除算力平台方向章节及实施检查中的平台选择事项，顺延章节编号并更新引用。

### v1.2 — 2026-10-05

删除重复列举不采用做法的“实验范围与可比性约束”章节，顺延后续章节编号并更新文内引用。本次为文档精简，实验参数与执行协议不变。

### v1.1 — 2026-10-05

初始 LR 五点确定为相对基准的 `{10^-1,10^-0.5,1,10^0.5,10}`，同步更新各优化器候选值。第 5.3 节列出待确认的配套扩边草案：按 0.5 个 log10 单位延伸，扩边无改善时取对数中点，再按 winner 邻域加密。正式运行前需确认该追加策略。

### v1.0 — 2026-10-04

建立 NanoGPT–Quadratic Proxy 实验设计基线，完整规定研究目标、模型与数据、参数训练范围、噪声网格与分布、优化器、学习率搜索及调度、随机配对、训练预算、失败处理、评估、日志和恢复协议。

本版本作为后续实验设计与实现的起点。后续修改直接更新本文件并增加版本记录，写明修改内容、原因和受影响的实验配置；每次运行在 metadata 中固定所依据的设计版本。数据哈希、软件版本及二次代理的待定研究参数按第 12 节在实施时补齐。
