# 长期自由滚动：训练目标、分布偏移与梯度边界的原论文核对

核对日期：2026-11-11。范围：学习动力系统/PDE surrogate 的滚动训练与稳定性；不审计本项目实现，不提出已获批准的改架构方案。所有下列网址均在本次工作中实际联网访问；没有运行训练、GPU、服务器作业或安装软件。主论文 1–4 的方法正文来自作者在 arXiv 的 HTML，会议身份由出版社/会议论文集或作者提交信息核实。第 5 篇只核对出版社摘要，不把未读的算法细节作为证据。第 6 篇是更新到 2026 年的可核对预印本，不能当成顶会录用论文或成熟共识。这里不是全面检索或截至今日的排名。

## 给主任务的直接判断

**不应仅因 1/7/30 日训练窗就断言必须换成长上下文 Transformer；也不能由这些窗上的好结果断言几十年自由滚动已受约束。** 对完整状态驱动的日转移 `s[t+1] = Fθ(s[t], forcing[t], CO2[t], ...)`，关键首先是状态是否充分、慢碳库和净收支是否可辨识、训练输入是否覆盖模型自己的偏移状态，以及损失和梯度是否覆盖希望修正的慢偏差。上述是本次证据的综合推论，不是某篇流体论文对生态系统的证明。

论文支持把四件事分开：

1. **短窗拟合**：真值起点上的一步/数步误差。
2. **输入分布恢复能力**：从模型自产生的偏移状态继续预测，减轻 exposure/distribution shift。
3. **长期稳定/统计正确**：不爆炸、吸引子和不变测度/频谱等合理。
4. **有指定外驱的长期科学响应**：例如 changing CO2 与气象下 AGB/BGB 的多年趋势、碳库增量及累积净碳收支。第 3 项不能替代第 4 项。

这批论文没有证明第 1、2 项会自动给出第 4 项；相反，PDE-Refiner 和 DySLIM 都给出短期与长期表现可分离的实例。

## 1. Brandstetter, Worrall, Welling：Message Passing Neural PDE Solvers

- 会议/年份：**ICLR 2022**。作者 arXiv 摘要页 Comments 明写 Published at ICLR 2022 (Spotlight paper)。正文当前版本为 arXiv v3，2023-03-20；须区别会议年与修订年。
- 主来源：[arXiv 摘要/会议信息](https://arxiv.org/abs/2202.03376)、[全文 HTML](https://arxiv.org/html/2202.03376)。访问：2026-11-11。
- 精确位置：§3.1 Training framework → The pushforward trick；§4.2；附录 F 的 Algorithm 1/训练细节。
- 来源短句："only backpropagating errors on the last unroll step"；接着明确 "not to backpropagate through the first unroll step"。

**方法事实。** 自回归测试时输入来自模型推前后的分布，而非真值分布。作者把稳定性视为域适应问题，用前一步模型预测作为对当前输入的、由模型产生的扰动，并在这个偏移输入上训练下一步。§3.1 的基本例子展开两步，但只对最后一步回传；作者报告这种截断更快且经验上更稳定，没有把它解释为长轨迹的精确梯度。Temporal bundling 一次输出一组连续时间步，减少推断时的自回归调用；它与 pushforward 是两项不同的技巧。

**适用思路。** 可评估从本项目模型产生的偏移碳库/水分状态继续训练，区分真值起点与偏移起点性能；这是训练分布修正，不要求先换网络家族。

**边界。** 该文缓和 PDE 数值误差增长，未给数十年生态碳趋势保证。其存活时间定义为超过特定误差阈值之前的时间；不能把稳定性用语读成任意长时间精确。更不能说用 364 天 detached prefix 后，年末罚项已对前 364 次转移进行了链式求导。

## 2. Lippe et al.：PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers

- 会议/年份：**NeurIPS 2023，Advances in Neural Information Processing Systems 36，Main Conference Track**。
- 会议主来源：[正式论文集](https://papers.nips.cc/paper_files/paper/2023/hash/d529b943af3dba734f8a7d49efcb6d09-Abstract-Conference.html)。
- 正文来源：[arXiv HTML](https://arxiv.org/html/2308.05732)，v2，2023-10-21。访问：2026-11-11。
- 精确位置：§2；§3；§4.1 Alternative losses；附录 D.1、E.4、E.8。
- 来源短句（§4.1）："no gradients are backpropagated through the predicted input"。
- 来源短句（附录 E.8）："stable for much longer than they remain accurate to the ground truth"。

**方法事实。** 在 KS 与 Kolmogorov-flow 任务中，通常的一步 MSE 优先拟合高幅值空间模态，可能忽略幅值较低、但通过非线性交互影响长期动力学的模态。PDE-Refiner 用带不同噪声尺度的去噪训练和每一步内部的多次 refinement 覆盖不同幅值信息。训练时随机抽 refinement step，对带噪真值目标去噪；这些 refinement steps **不是**几十/几百个物理时间步的 BPTT。

**对 pushforward 的重要反证。** §4.1 报告，pushforward 在该设置中没有改善性能；作者明确指出它虽处理输入分布偏移，但因不向预测输入回传梯度，不能在其设置中学会准确长期预测所需的低幅信息。这支持“on-policy 偏移训练有用但非充分”，不能解释为 pushforward 在所有问题都无效。

**对更多历史的反证。** 附录 E.4 中，更多历史输入显著改善一步误差却使自回归误差更快增长；这是特定显式 PDE 预测器的结果，足以反驳“上下文更长必然更稳”，不足以否定所有历史状态/Transformer。

**适用思路。** 检查大通量/大碳库尺度下的损失是否淹没低幅但长期相关的净增量与慢过程；对不同科学量分开量纲、尺度和长期敏感性验证。这里的“低幅”启发不等于可直接照搬空间 Fourier 频谱权重到生态碳库。

**边界与代价。** 附录 E.8 的很长滚动仍可能累积高频误差；稳定不等于轨迹准确。每个物理步增加 refinement 调用，不能当作零代价训练技巧，更不是已经适合当前吞吐约束的建议。

## 3. Herde et al.：Poseidon: Efficient Foundation Models for PDEs

- 会议/年份：**NeurIPS 2024，Advances in Neural Information Processing Systems 37，Main Conference Track**。
- 会议主来源：[正式论文集](https://papers.nips.cc/paper_files/paper/2024/hash/84e1b1ec17bb11c57234e96433022a9a-Abstract-Conference.html)。
- 正文来源：[arXiv HTML](https://arxiv.org/html/2405.19101)，v2，2024-11-05。访问：2026-11-11。
- 精确位置：§2 的 all2all loss，式 (6)–(8)；补充材料 all2all ablation、autoregressive inference 与时间外推讨论（可按段首 "However, there is a caveat with the all2all strategy" / "Here, we consider the NS-SL downstream task" 定位）。
- 来源短句："all possible data pairs"；"semi-group property"。

**方法事实。** 基于时间条件化多尺度 operator Transformer。训练方面，利用自治 PDE 解算子的半群结构，对同一真值轨迹的所有 `t_k ≤ t_l` 时间对训练：输入真值状态与 lead time，预测另一时刻真值。K+1 个时间快照含对角线可形成 `(K+1)(K+2)/2` 个时间对；文中每条预训练轨迹 11 个快照得到 66 对。这是多起点、多 lead-time 监督和样本复用，不等于在每个样本内微分整段自由滚动。

**是否跨 prefix 回传？** all2all 配对的输入来自给定真值轨迹，不存在由本文 loss 自动贯通的长 predicted prefix；与 MP-PDE 截断前缀、完整 BPTT 都应分清。推断可以直接指定 lead time，也可以自回归组合。作者报告最佳选择依任务而异。

**适用思路。** 在已有真值轨迹中用多个起点与日/周/月/年等不同时间跨度的状态、增量/累积量目标，提高慢时间尺度监督，而非只增加相邻日数据。若保留纯日一步架构，不能直接声称已采用 Poseidon all2all：要解释如何连接多跨度监督与实际训练图。

**重要适用条件。** 当前生态系统有非平稳外部驱动。若用可变 lead-time 算子，必须条件化这段时间内的驱动路径/阶段，而非只给时间差后套用自治半群。不应未经科学决策就把原日转移改成跳日算子。

**边界。** 补充实验时间外推 T=0.7→1.0 中，两模型都比区间内预测更差；作者未证明任意远时间泛化。Transformer 在本文主要处理空间多尺度算子，并非“读几十年 token 的长上下文”证明。

## 4. Schiff et al.：DySLIM: Dynamics Stable Learning by Invariant Measure for Chaotic Systems

- 会议/年份：**ICML 2024，PMLR 235:43649–43684**。
- 会议主来源：[PMLR](https://proceedings.mlr.press/v235/schiff24b.html)。
- 正文来源：[arXiv HTML](https://arxiv.org/html/2402.04467)，作者 2024-06-05 修订版。访问：2026-11-11。
- 精确位置：§2 Invariant Measures and Attractors；§3 Multi-step Objectives；§4，尤其式 (9)–(13)；§5；附录 A。
- 来源短句："point-wise accurate for short-term predictions and statistically accurate for long-term ones"。

**方法事实。** 在局部轨迹损失之外加入分布匹配正则，学习长期吸引子的 invariant measure。使用 MMD 的可微、批量可估计分布距离，并用从参考分布出发的有限模型滚动近似模型不变测度；包括 unconditional 与 conditional 两种有限样本正则。可搭配单步、多步 curriculum、pushforward 等基础目标。§3 的 `sg(Sθ^k(u))` 对 pushforward 的 stop-gradient 定义非常明确。

**为什么不是必须训几十年逐点 MSE。** 混沌系统的长程轨迹点对点误差会失去信息、梯度可能爆炸/消失；长期统计目标提供另一类有意义约束。论文实验区分短程 cosine similarity 和长期分布/频谱/协方差/时间相关等指标。

**不能过度读成什么。** 不是“任何短窗 + MMD 就保证无限时间统计精确”；有限滚动近似、吸引子存在、参考测度等条件仍在，且测度保留本身允许平凡映射，因此文中仍保留轨迹目标。不能因为正则可搭配 Pfwd，就声称其每一种实验都使用跨全部长 prefix 的精确梯度。

**适用思路。** 可把长期评估中的统计/物理约束与逐日局部误差明确拆开；在固定外驱或定义清楚的季节/状态条件下研究分布偏移。

**生态科学边界。** 该文主要是有不变测度的自治耗散混沌系统（Lorenz63、KS、Kolmogorov flow）。changing CO2、气候变化、扰动和林分演替下的碳库没有自动成立的单一固定不变分布。原文摘要把 slowly-variant weather/climate 的推广写为希望/未来扩展。未经处理直接把 AGB/BGB 拉回历史分布，可能反而压制真实趋势。即使边际分布接近，也不识别相同外驱下的单条多年碳库响应和累积净碳收支；这最后一点是对本项目的科学外推边界，不是作者做过的生态实验结论。

## 5. Zhuang et al.：Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE

- 会议/年份：**ICML 2020，PMLR 119:11639–11649**。
- 已核对主来源：[PMLR 正式摘要](https://proceedings.mlr.press/v119/zhuang20a.html)。访问：2026-11-11。
- 核对深度：出版社摘要；没有依赖该文尚未核对的定理、具体复杂度或实现。
- 来源短句："the adjoint method has numerical errors in reverse-mode integration"。

**可支持的窄结论。** 作者指出 Neural ODE 的反向积分梯度估计存在数值误差，并以记录正向轨迹的 checkpoint 策略改善精度。因此不能把“连续时间 adjoint + 反向重建状态”无条件等同于实际日离散程序的精确离散梯度，也不能许诺常量内存下无代价地得到任意长精确梯度。

**对本项目的工程区分（链式法则推论）。** Checkpoint/recompute 可以保留原离散计算链，少存中间激活、需要时重算，以额外算力交换内存；这与把状态 stop-gradient、丢掉长程链式项不同。具体内存/计算节约取决于 checkpoint 划分和实际状态，本文摘要不提供本项目的代价承诺。已有本项目原生 JAX checkpointed AD 证据应由主任务复用，不应为解释概念又启动一轮训练。

## 6. 更新动态：Xu et al.：ReAl: Refine neural operators via aligning dynamical flow

- 身份/年份：**arXiv 预印本，2610.20369v1，2026-10-14**。本次原始页面未显示顶会/期刊接收，明确不把它列为已发表顶会成果。
- 主来源：[arXiv 摘要](https://arxiv.org/abs/2610.20369)、[全文 HTML](https://arxiv.org/html/2610.20369)。访问：2026-11-11；摘要信息也在 arXiv 站内检索结果实际取得。
- 精确位置：§3.3 Trajectory Alignment for PDE Forecasting；§4.1 Experiment setup；§4.2 Ablation of Reward Horizon and Metric。
- 来源短句："Same trajectory-level reward ... is applied to every autoregressive step"（省略中间公式）。

**可核实的新方向。** 先冻结 MSE 预训练 base model，训练随机 flow refiner；再从同一起点采样多条候选自由滚动轨迹，用末端 MSE 等 trajectory reward，对每步局部 flow-matching 更新施加同一条轨迹的奖励权重。其主实验每条 5 个自回归 block，每 block 10 帧，4 条候选轨迹；长期评估延到 200 帧。论文还比较 reward horizon 1/5/10 blocks，并报告不同 horizon 对相应评估尺度最有利。

**与年末罚项的关系。** 它展示“把末端质量通过显式设计的 trajectory-level credit 信号送到整条轨迹局部更新”这一研究思路，但这里是奖励加权的随机 refiner 后训练。不能把它称为穿过全部 deterministic daily transitions 的精确离散 adjoint；不能因使用末端 reward 就假定无偏、低方差、常量内存、任意长有效；也不意味着冻结当前生态 base model 有科学依据。

**权重。** 作为最新、直接相关的预印本动态补充，可提醒年末反馈需要明确归因机制；尚不足以取代上面已发表研究，更不足以推荐立刻改造成 flow/RL 模型。

## 梯度问题的最小公式：年底的误差到底传到哪里

令日转移为 `s[t+1]=Fθ(s[t],u[t])`，年末目标为 `L(s[T])`，`A[t]=∂Fθ/∂s[t]`，`B[t]=∂Fθ/∂θ`。固定初态时，完整离散梯度包含

`dL/dθ = Σ[t=0..T-1] (∂L/∂s[T]) · A[T-1] ... A[t+1] · B[t]`。

这是日离散模型的链式法则，不是新增算法。若在 `s[K]` detach 后只微分后面的窗口，上式中 `t<K` 的路径项就被删掉。参数共享意味着更新后的 θ 将来会影响所有日子，但**这不等于年末 loss 已把前段每天的累积贡献纳入该次梯度**。同理，每月真值重置后拟合得很好，并不等于训练了从自身全年状态出发的自由滚动。

因此，要让更远终点/年际趋势成为真正训练信号，至少须说清采用哪一类机制：完整离散 BPTT/adjoint（可 checkpoint/recompute）、有明确定义的近似或合成梯度、跨段约束/多重射击优化、或者像第 6 篇那样另外设计轨迹级 credit。它们的目标、梯度偏差和资源成本不相同。**本次没有核对一篇足够贴合的多重射击原论文，所以这里仅列机制类别，不引用其效果、收敛或成本结论。**

## 可以交给主任务的证据排序

1. 最强、直接且不依赖改架构的已发表依据：MP-PDE 的模型分布偏移训练；PDE-Refiner 对一步 MSE/更多历史/截断 pushforward 的局限证据。
2. 多尺度监督依据：Poseidon all2all；须说明非自治外驱条件和与现有日转移的连接，不能直接视作已批准方案。
3. 长期目标的分类依据：DySLIM 的短期轨迹与长期分布分开；不能替代 AGB/BGB 的趋势/收支科学检验。
4. 梯度实施依据：ACA 只支持连续 adjoint/反向重建有数值风险的窄提醒；本项目离散梯度判断以实际图、原生 AD 与已完成证据为准。
5. 2026 动态：ReAl 是有明确轨迹奖励归因机制的预印本例子，不是既定最佳方案。

下一步决策应由当前代码审计回答：现在 7/30 日损失是否真的穿过模型状态，初态来自真值还是 on-policy prefix，哪些地方 detach/reset，AGB/BGB 年际净增量和驱动响应有没有被单独约束/验证。文献本身不能替代这些事实，也没有授权新的正式训练、付费作业或架构改变。
