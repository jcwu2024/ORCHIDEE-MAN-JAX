# 当前方法与长时域边界审计（2026-11-11）

范围：现行 daily/shared 实现、当前入口所指的具名报告；只读源码与既有证据。
未运行模型图、GPU、训练或新实验。daily 工作树不在已知主图索引覆盖内，
采用确切源码核对；本审计不是全仓库穷尽性证明。

**1. 模型推进什么？** 当前原型是3854维连续物理状态的逐日算子，
输入包括当日原生六小时forcing、PFT/static及annual_conditions上下文。
日内forcing编码/注意力不等于跨多年时间Transformer；跨日记忆由连续/离散物理状态携带。
canonical step每一天用预测carry重建state-derived视图，未读取后续Teacher状态作为carry
（[canonical_rollout.py:26](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/canonical_rollout.py:26)；[successor_training.py:161](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/successor_training.py:161)）。
head输出先进入守恒更新器，再经过保留的exact tail，产生下一状态、离散状态及水/碳预算诊断
（[model.py:53](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/model.py:53)、[model.py:68](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/model.py:68)）。annual_conditions是输入上下文，
不能据此声称有学习的年度跳步替代模型。当前64.88M原版与66.12M latent-reader诊断原型
应与正式fresh-init/full-network联合训练协议区分；后者尚未端到端验收。

**2. 实际损失不是只有末日误差。** 共享联合目标的T1项为
`process_day/3 + (flux_1+stock_1)/9`，T7/T30各为`(flux_T+stock_T)/9`
（[shared_recurrent_training.py:243](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/shared_recurrent_training.py:243)、[shared_recurrent_training.py:197](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/shared_recurrent_training.py:197)）。
`process_day=0.5*(必要过程标签损失+一天状态变化anchor)`；尺度须来自冻结TRAIN统计
（[shared_training.py:128](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/shared_training.py:128)、[shared_training.py:148](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/shared_training.py:148)）。
flux包括GPP及maintenance/growth/heterotrophic respiration；采用逐日误差和每一天累计前缀误差，
后者按`scale*sqrt(elapsed)`归一化，各半混合。stock对七个具名库存采用逐日变化残差与
整条库存轨迹残差，各半混合，再平衡living/nonliving组
（[shared_trajectory_loss.py:27](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/shared_trajectory_loss.py:27)、[shared_trajectory_loss.py:313](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/shared_trajectory_loss.py:313)）。
因此末日已包含在轨迹/前缀内，但不是唯一监督；内部`terms.total`也不能替代公开runner的`/9`标量。
旧`long_horizon_objective`是累计通量与terminal stock组件接口
（[long_horizon_objective.py:295](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/long_horizon_objective.py:295)），旧target-oriented组合也不能冒充当前shared公式。
所审现行目标未实现年度/季节/多年趋势专用loss或年度surrogate。

**3. AD与30天限制在哪里？** 当前shared窗口准备器明确只接受7或30天，
初态取窗口第一天，后续状态自由递推（[shared_recurrent_training.py:52](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/shared_recurrent_training.py:52)）。
公共rollout使用`lax.scan`传递物理carry，对每日transition作`jax.checkpoint`，并绑定native AD
（[rollout_loss.py:79](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/rollout_loss.py:79)、[rollout_loss.py:129](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/rollout_loss.py:129)）。
这条路径没有把活跃连续carry截断或stop_gradient；离散状态/forcing/目标固定不求导。
checkpoint重算日内激活，不意味着跨日状态、完整观测轨迹、loss或参数梯度常数内存。
“能逐日前向跑几十年”与“几十年完整反向图可编译、可承受、梯度可信”是不同命题；
7/30是当前适配器约束，现有证据不能推出AD存在30天理论硬上限，也不能推出多年反向已获支持。

**4. 实测证据与失败不能互换。** 具名T30报告的原版/候选warm AD中位数分别
6.650839/5.189475s；backend compile约2487.81/5853.29s，compiler live bytes约
3.491/1.196GB。NVML原版compile6555MiB、候选5767MiB，但候选期间有外部allocation；
matched8275MiB同时保留两图，不能当作各模型无争用峰值。此为FP32 neural/FP64 physics
组件诊断，初态梯度strict自重复原版2/3、候选3/3失败；不是完整optimizer update或长期科学验收
（[shared_latent_reader_results_20261004.md:349](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/docs/research/daily_coarse_graining/shared_latent_reader_results_20261004.md:349)）。
最新FP64诊断job16060441仅完成lower，3429秒compile timeout，completed compile/calls均0，
52次数值检查未启动；MLIR已保存，GPU释放已验证。
见[终态证据](../T7-current-model-precision-shared-graph-execution-20261024/terminal/SUMMARY.md)。
不能将其解释为数值对照通过或由节点拥挤导致：现有终态只证明编译超时。

早期保守算子协议明确写7/30日truncated BPTT及365日前向验证、不保留全年图
（[conservative_daily_process_operator_v1.md:192](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/docs/research/daily_coarse_graining/conservative_daily_process_operator_v1.md:192)），与当前native checkpointed路径不同。
旧架构anonymous-state漂移、stop-gradient pushforward改善seen但NPP/growthresp与空间/joint退化，
mixed-horizon30日aggregate改善却signed tendency/key-carbon退化（114/223相对gate失败）
已有记录（[failed_architecture_lessons.md:36](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/docs/research/daily_coarse_graining/failed_architecture_lessons.md:36)）。这些是旧架构负证据，不能直接判当前66M失败，
也不能把单纯加long loss/pushforward重新称为未试过的新方案。
旧long-horizon v1文档明确未开启365日评估；669点/33450点年是Teacher数据规模，
不证明神经模型多年自由滚动准确。当前长期精度及最终可复现训练方法执行仍待验收。

**5. 可复用接口与待讨论项。** 复用canonical step与公共checkpointed loss，
保留状态/参数梯度、source masks、守恒和finite gates；科学轨迹提取器已有五通量/七库存接口
（[successor_training.py:409](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/successor_training.py:409)），shared组件投影已有PFT面积/库存语义
（[shared_trajectory_loss.py:207](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/shared_trajectory_loss.py:207)）。历史梯度配比校准接口明确train/train及系数区间，
是可参考接口，不是当前shared目标已自动采用的训练规则
（[successor_training.py:703](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/successor_training.py:703)、[successor_training.py:781](E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/research/daily_coarse_graining/daily_operator/successor_training.py:781)）。
**待讨论**：先明确长期验证与训练梯度跨度的独立目标、具名碳库/通量与signed tendency验收，
再比较保持现行监督的长窗/分段方案；任何截断、pushforward、年度辅助目标、采样或架构调整
须显式说明偏差、与旧失败的差异及对照。本文不提出自动实施、重跑或付费授权。
