# 长时域序列架构：对日尺度碳算子漂移问题的文献核验

核验日期：2026-11-11。范围是原始论文/会议论文集/DOI 元数据；这不是截至该日的穷尽性最新综述。Nature 页面遇到自动访问挑战，GraphCast 的 Science 页面也受站点拦截；二者的正式书目信息用 DOI/Crossref 出版商元数据交叉核验，方法细节读 arXiv 原文。ACE2 的期刊书目信息由 Crossref 核实，方法细节参考其预印本。其余链接访问原文或正式会议论文集。

## 结论摘要

Transformer 长上下文设计可能降低“读入长历史”的成本，但这和“把一个日步算子稳定地自回归几十年”是不同问题。前者改进输入表示/注意力计算；后者取决于闭环动力学、误差增长、守恒与强迫响应。再者，长 rollout 若只在推理时进行，不能提供远期损失到早期参数的信用分配；真正跨很多步反传的激活/梯度内存与计算开销，通常仍随展开长度增长。FlashAttention 保持精确注意力、减少 HBM IO，不改变注意力关于 token 数的二次算术复杂度，也不减少物理时间步数。线性时间序列模型减轻序列运算成本，也不自动令长程反传便宜或稳定。

对目前共享训练入口限定为 T1/T7/T30 的日尺度碳算子，建议将问题拆开验证；这不是反传存在 30 天理论硬上限的证明：

1. **稳定性**：在外强迫、初态、PFT/掩膜分层下，长 rollout 是否漂移，哪些状态/通量先偏离；架构改动要直接受长 rollout 检验。
2. **学习信号**：保留 T1/T7/T30 作为可承受的梯度窗口，并测不同窗口如何作用于短期过程与慢变量；长测试 rollout 不能冒充长时域梯度证据。多尺度/长时域损失、课程、梯度截断或停止梯度是训练目标/优化策略的选择，不是 Transformer 的自然附带效果，且停止梯度会改变信用分配。
3. **时间离散**：粗时间步/时间聚合可省算，但可能漏掉日通量、事件和快速扰动响应；必须以日步基线检验聚合的过程误差和状态守恒。

ACE2 是最贴近“长期闭环模拟”的天气/气候类比：它展示了小时步长机器学习大气模拟可稳定运行数十年，但论文同时指出对 SST/CO₂ 独立变化的敏感性并非完全真实；后续耦合 slab ocean 的 ACE2-SOM 对突变 CO₂ 的非平衡响应还出现不物理快速跃迁和能量守恒违背。因此，“能长时间运行”不等于“正确响应外强迫”，更不能直接外推到确定性陆地碳状态轨迹。天气预报论文的目标主要是有限时效预报技巧；天气气候模拟强调统计分布/响应；碳算子则要正确推进被强迫的物理状态及通量，评价标准不能互换。

## 核心来源

### 1. Pangu-Weather：层次化时间聚合是预报时效设计，不是长年积分证明

- **正式发表书目**：“Accurate medium-range global weather forecasting with 3D neural networks”，Nature 619, 533–538 (2023)。DOI：<https://doi.org/10.1038/s41586-023-06185-3>。Crossref 确认正式题名和出版日期 (2023-07-05)。**预印本题名**为“Pangu-Weather: A 3D High-Resolution System for Fast and Accurate Global Weather Forecast”，arXiv:2211.02556：<https://arxiv.org/abs/2211.02556>。
- **原始证据**：Nature 论文介绍 3D Earth-specific Transformer 与 hierarchical temporal aggregation；论文 III-D 节说明分别训练 1h、3h、6h、24h lead-time 模型并用不同时间间隔接续预报，以避免单一模型跨不同 lead time 的性能退化。
- **对本问题的含义/限制**：这是按预报 lead time 组织模型、减少长时效推理调用的方案。它不是同一个日步算子在几十年闭环推进，也没有证明碳状态的多年误差不会累积。对陆地碳而言，学习 1/7/30 日的转移或多时间尺度模块可作为表示/效率候选，但拼接不同步长算子必须验证通量积分、守恒、季节边界和扰动响应连续性。
- **核验**：Nature 出版商页触发 Client Challenge；Crossref DOI 记录确认正式题名、卷页和出版日期，原论文 arXiv HTML 可读。访问日期 2026-11-11。

### 2. GraphCast：有明确的多步反传课程，但展开只有 12 步

- **书目**：“Learning skillful medium-range global weather forecasting”，Science 382(6677), 1416–1421 (2023-12-22)。DOI：<https://doi.org/10.1126/science.adi2336>；论文原文：<https://arxiv.org/abs/2212.12794>。
- **原始证据**：论文 §4.2 定义 12 步（3 日）预报目标；§4.5 的三阶段课程先以 1 个 autoregressive step 训练，再将步数从 2 增到 12（每 1,000 次更新加一步）；§4.6 为容纳 12 步展开使用 bfloat16 和 gradient checkpointing，并明确 checkpointing 以更低训练速度换内存。
- **对本问题的含义/限制**：这是最直接的“增加展开长度、让训练看到 rollout 误差”的类比，也说明课程和重计算能帮助管理有限 horizon 的训练成本。但 12 个 6 小时步仍远短于多年日尺度碳 rollout；论文没有声称精确训练几十年反传。它支持把 T1/T7/T30 或更长可承受窗纳入有计划的 rollout 课程/损失研究，不支持未经验证地将一个较短梯度窗解释成长时域信用分配。
- **核验**：Science 页面受反爬挑战；出版商 DOI 元数据由 Crossref 返回 title、卷期、页码和 2023-12-22 出版日期。方法逐节读 arXiv HTML。访问日期 2026-11-11。

### 3. ACE2：多年级闭环模拟的强类比，也暴露长期模拟与强迫响应的区别

- **正式发表书目**：“ACE2: accurately learning subseasonal to decadal atmospheric variability and forced responses”，npj Climate and Atmospheric Science (2025-05-29)，DOI：<https://doi.org/10.1038/s41612-025-01090-0>。Crossref 核实正式刊名、题名、DOI 和日期；Nature 页面触发 Client Challenge。方法细节引用其预印本 arXiv:2411.11268 (2024-11-18)：<https://arxiv.org/abs/2411.11268>。
- **原始证据**：摘要称 ACE2 使用 6 小时步长，并报告可稳定推进任意多步、约 1,500 simulated years / wall-clock day；同时称其对分别改变 SST 与 CO₂ 的敏感性“not entirely realistic”。
- **补充原始来源**：“ACE2-SOM: Coupling an ML atmospheric emulator to a slab ocean and learning the sensitivity of climate to changed CO₂”，arXiv:2412.04418 (2024-12-05)：<https://arxiv.org/abs/2412.04418>。摘要报告在缓慢增加 CO₂ 场景可拟合全球年均趋势，但突增 CO₂ 时响应过快、违反全球能量守恒。
- **对本问题的含义/限制**：这是长期稳定 rollout 可行的证据，不是只靠 Transformer 上下文修复漂移的证据。它的场是大气天气/气候变量，状态更新为 6 小时；结果涉及统计变率与气候强迫响应。陆地碳的日通量、土壤/植被慢状态以及确定性轨迹需各自验证，尤其不可用平均趋势正确替代逐日状态/碳守恒。
- **核验**：Crossref 核实 ACE2 正式发表记录；arXiv API/摘要原文可访问，ACE2 与 ACE2-SOM 标识符及提交日期由 arXiv 元数据核对。访问日期 2026-11-11。

### 4. FlashAttention：长序列注意力的 IO 优化，不会消除长反传或长 rollout 成本

- **书目**：Dao et al., “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”，NeurIPS 2022 主会论文集。正式页面：<https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html>。
- **原始证据**：论文摘要明确指出标准 self-attention 在序列长度上时间和内存复杂度均为 quadratic；FlashAttention 通过 tiling 降低 GPU HBM 与 SRAM 间读写，仍是 exact attention。
- **对本问题的含义/限制**：若碳模型确实有长 token 输入，减少注意力 IO 和激活占用可能帮助增加可处理历史；但这不是线性注意力，也不减少日步状态递推数量。更重要的是，长时间展开反传的状态激活/梯度链问题独立于单层 attention 的 IO 优化。不要把显存优化等同于低成本的多年信用分配。
- **核验**：NeurIPS 正式论文集题名、作者、卷、年份和摘要均可直接读取。访问日期 2026-11-11。

### 5. Mamba 与 Mamba-2：线性序列计算可作架构候选，但隐藏状态不是物理状态

- **书目**：Gu & Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”，arXiv:2312.00752 (2023-12-01)。<https://arxiv.org/abs/2312.00752>。此引用以可核验预印本为准。
- **原始证据**：摘要称选择性 SSM 允许信息依输入内容沿序列传播，并以线性序列长度扩展；原文讨论其 recurrent mode 与 parallel scan 的硬件算法。
- **对本问题的含义/限制**：作为压缩长历史、用较小序列成本的候选结构值得局部比较；但“state-space model”里的状态是神经网络隐状态，不等价于有来源定义、守恒约束和单位的植被/土壤物理状态。线性前向序列复杂度也不自动证明梯度跨数万日稳定、低内存，或模型能正确响应强迫。若把它作为日算子骨干，应保留显式物理状态更新接口，并将远期闭环与梯度能力分开报告。
- **核验**：arXiv 摘要与 HTML 原文标题、版本及方法可读。访问日期 2026-11-11。

- **已发表的后续架构来源**：Dao & Gu, “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality”，Proceedings of Machine Learning Research 235, ICML 2024。正式论文集页：<https://proceedings.mlr.press/v235/dao24a.html>；预印本 arXiv:2405.21060：<https://arxiv.org/abs/2405.21060>。PMLR 页面确认正式标题和 ICML 2024 论文集信息。论文介绍 Mamba-2 / SSD 的结构化状态空间对偶与高效算法；它是序列计算架构结果，不是长时域物理状态闭环或长期碳通量正确性的证据。访问日期 2026-11-11。

## 术语边界

- **长输入上下文**：给单次预测更多历史 token；解决“模型看到多远过去”。它可能改变学到的动力学，但不会自动解决闭环误差增长。
- **长输出 horizon**：模型可以连续运行很多步；这是推理时能力，单独不说明长期准确或训练时梯度贯穿同样多步。
- **长信用分配**：远期误差通过未截断的计算图影响早期参数；这才需要跨时间步反传，成本受步数、状态和网络规模制约。对完整且未 detach 的 rollout 只在终点计算 loss，仍会通过整条计算图反传；但这是终点目标，与逐步/区间 loss 的定义不同。停止梯度或截断反传才会切断相应时间边界的信用链，必须单独说明。
- **状态空间架构**：序列模型内部状态用于压缩序列信息；除非明确映射并验证，否则不能将其当作地球系统状态变量或守恒状态。

本次来源支持“长时域训练课程、记忆/重计算手段、长 rollout 稳定性测试值得分别研究”，不支持“换成长上下文 Transformer/线性注意力即可解决数十年日算子漂移”。

**项目适用提醒：**`failed_architecture_lessons` 已记录，早期 stop-gradient pushforward 虽提升 seen 指标，却使 NPP/growthresp 与空间/joint 结果恶化；mixed-horizon 虽改善30天总体误差，signed tendency和关键碳量退化，223项相对门槛中114项失败。这是旧架构负证据，不能直接判定当前原型失败；任何复用都应说明与已知失败机制的差别。
