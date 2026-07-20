# 日尺度粗化研究开发规范

状态：初版规范，供可行性分析和原型开发使用。

上位构想：[`paper_concept_daily_coarse_graining.md`](../../paper_concept_daily_coarse_graining.md)

## 1. 目标与边界

本研究在同一仓库内维护两个严格区分的模型：

1. **Teacher**：`jax_orchidee/` 中与 Fortran PFT14 对齐的完整半小时模型。
2. **Coarse model**：待开发的日尺度近似模型，以 Teacher 为监督真值。

以下表述不得混用：

- `Fortran-equivalent` 只用于通过既定等价门禁的 Teacher 路径。
- `teacher-matched`、`surrogate` 或 `coarse` 用于日尺度模型。
- 日尺度模型即使通过长期科学容差，也不得称为与 Fortran 数值等价。

在 669-landpoint 最终验收完成前，Teacher 生成的数据必须标记为
`provisional_teacher`。可开展低成本可行性实验，但不得冻结正式训练集或发布最终
粗化模型。

## 2. 首版替代边界

首版日尺度模型只替代以下边界：

```text
日初完整物理状态
  + 当天 48 步 forcing
  -> 48 次 SECHIBA 状态转移
  -> STOMATE 所需日累计量
  -> 日末 SECHIBA 状态 + 日累计边界
```

以下过程继续由 Teacher 的原始 JAX 实现执行：

```text
OK_LEAK -> STOMATE 日碳过程 -> modelout -> 日末写回 -> restart/跨年交接
```

采用该边界的理由：

- 计算瓶颈主要来自 48 次半小时物理转移；
- STOMATE 长期碳库和慢状态是气候记忆研究的核心，不应在首版同时近似；
- 日累计量是半小时过程与 STOMATE 的明确接口，可以独立验收；
- 保留原始日碳和 restart 过程可显著降低长期漂移的归因难度。

如果后续证据表明 STOMATE 日过程成为主要瓶颈，应建立新的边界提案和独立验收，
不得直接扩大首版网络输出。

## 3. 必须先完成的状态总账

在实现任何神经网络前，必须建立机器可读的 daily-boundary ledger。每个字段至少
记录：

- 稳定字段 ID 和数组形状；
- 物理单位、有效范围和缺失值规则；
- `prognostic`、`diagnostic`、`accumulator`、`static`、`discrete` 分类；
- 日初 producer、半小时 consumer、日末 producer、日过程 consumer；
- 是否进入 restart、modelout 或下一日状态；
- 对应 Fortran/JAX owner；
- 水量、碳量或能量预算中的角色；
- 模型是直接预测、解析计算、守恒投影还是原样携带。

状态分类规则：

- **Prognostic**：必须由粗化转移更新并传给下一日。
- **Diagnostic**：优先从状态和 forcing 解析重算，不作为自由预测目标。
- **Accumulator**：必须与 48 步 Teacher 累计定义一致，并满足单位和时间归一化。
- **Static**：参数和网格属性只作为输入，不得写回动态 state。
- **Discrete**：mask、索引、开关和阶段状态必须精确更新，不允许连续网络自由拟合。

不得用无结构 `dict` 作为正式训练或 rollout carry。正式边界使用固定、版本化的
PyTree/dataclass，并提供 Teacher state packet 与 coarse state 之间的单一适配器。

## 4. 开发目录约束

Teacher 和研究代码必须物理隔离：

```text
jax_orchidee/                         # Teacher；默认正式运行路径
research/daily_coarse_graining/       # 粗化研究代码和实验入口
docs/research/daily_coarse_graining/  # 规范、边界、决策和结果摘要
manifests/coarse_graining/            # 数据集、split 和实验 manifest
```

约束如下：

- 粗化代码不得复制或改写 Teacher 科学过程公式。
- Teacher 只允许加入一个经过测试的 daily-transition 注入接口，不得散布
  `if coarse_model` 条件。
- 默认 `orchidee-jax run` 永远使用 Teacher；粗化模式必须显式选择。
- 粗化依赖放入独立 optional dependency group，不污染基础运行环境。
- 临时 notebook、checkpoint、训练日志和数据不得进入源码目录。
- 不使用 `scripts/dev/experiment_1.py` 一类无登记的一次性脚本作为正式实验入口。

建议在完成速度上限实验、确认继续该路线后再创建
`research/daily_coarse_graining/`，避免预先生成空框架。

## 5. 数据资产规范

Teacher 数据必须外置，建议使用：

```text
$ORCHIDEE_COARSE_DATA_ROOT/
  datasets/<dataset_id>/
    manifest.yaml
    schema.json
    splits.json
    shards/
  runs/<run_id>/
  checkpoints/<experiment_id>/
```

每个日样本包含：

```text
day_start_state
forcing_48
teacher_end_sechiba_state
teacher_daily_accumulators
teacher_budget_terms
static_landpoint_parameters
date/landpoint/parameter metadata
```

数据要求：

- 不使用 pickle 作为长期科学数据格式；优先 Zarr/NetCDF 等带 schema 的格式。
- 按 landpoint-year 或相近大小分 shard，避免逐日小文件。
- manifest 记录 Teacher git commit、配置 hash、状态 schema hash、数据生成命令和
  变量单位。
- 数据生成和训练读取分离；训练不得隐式启动 Teacher 模拟。
- normalization 统计量只从训练 split 计算，并作为版本化资产保存。
- 原始 Teacher target 不得被覆盖；派生特征写入新的 dataset ID。

## 6. 数据划分规范

必须在生成训练数据前冻结 split，不得看到测试结果后重新划分：

- 空间留出：训练、验证和测试 landpoint 不重叠；
- 时间留出：至少一组完整年份不参与训练；
- 事件留出：干旱、强降水、显式雪、潮汐和高盐度设置专门测试组；
- 参数留出：保留未见参数组合，区分插值与外推；
- 生命周期留出：包含 cold start、成熟状态和跨年 restart 状态。

随机打散所有“landpoint-day”后再划分是禁止方案，因为会泄漏空间、季节和长期
状态信息。

## 7. 模型结构原则

首个可学习原型采用以下结构：

```text
显式日尺度基线
  + 可学习的亚日尺度 residual
  -> 守恒/范围投影
  -> 固定 daily-boundary state
```

结构要求：

- 首先实现无神经网络的显式日尺度 baseline，作为必要对照组。
- forcing 表示至少比较日统计量与完整 48 步轻量编码器两种方案。
- 网络只预测无法可靠解析计算的 residual，不自由预测所有状态。
- 水、碳和可解析库存关系优先硬约束；软损失只用于无法解析投影的量。
- 离散状态通过规则或分类门更新，不用连续值四舍五入代替过程语义。
- 非负性不能只依赖输出裁剪；必须检查裁剪是否破坏预算闭合。
- 网络结构、损失和 projection 必须各自可做消融。

## 8. 实验登记与可复现性

每个正式实验必须有唯一 `experiment_id` 和不可变配置，至少记录：

- 研究问题和单一主要假设；
- Teacher 版本、dataset ID 和 split ID；
- 模型、forcing encoder、projection 和损失配置；
- 随机种子、硬件、JAX/XLA 版本和精度策略；
- 训练预算、停止规则和 checkpoint 选择规则；
- 预先声明的主要指标和通过阈值；
- wall time、编译时间、峰值内存和推理吞吐；
- 生成结果的 git commit 和完整命令。

结果目录由配置 hash 生成，不允许使用 `final2`、`new_best` 等人工命名。失败实验
保留摘要和失败原因；大型 checkpoint 可删除，但 manifest 不删除。

## 9. 分阶段门禁

### Gate 0：Teacher 冻结

- PFT14 Teacher 的当前正式路径、状态 schema 和性能 baseline 有版本号；
- 669 点验收结果或明确的 provisional 标记可追溯；
- 粗化研究不得改变 Teacher 默认数值结果。

### Gate 1：速度上限

- 从内存回放 Teacher 的预日碳边界，不进行逐日磁盘 IO；
- 保留 OK_LEAK、STOMATE、modelout 和完整状态写回；
- 分别报告编译、热运行、数据准备和 IO；
- 对 1 年和 50 年估算端到端速度上限。

若跳过半小时过程后没有足够的数量级收益，停止粗化主路线。

### Gate 2：边界与数据

- daily-boundary ledger 无未分类字段；
- Teacher adapter 往返无字段丢失；
- replay 模式与 Teacher 后续日过程在相同边界输入下数值一致；
- 数据 schema、单位、预算项和 split 均冻结。

### Gate 3：单日可学习性

- 无神经 baseline、黑箱 baseline 和守恒 residual 模型使用相同 split；
- 报告所有 prognostic state、accumulator 和 budget residual；
- 在未见日期和未见 landpoint 上通过预设阈值。

### Gate 4：自由 Rollout

- 依次通过 7 日、30 日和 365 日自由 rollout；
- 每天使用模型自己的上一日状态，不注入 Teacher 状态；
- 检查状态范围、预算、事件恢复和误差单向漂移；
- restart 拆分运行与连续运行一致。

### Gate 5：长期与科学应用

- 50 年关键库存和通量通过预设科学阈值；
- 未见空间、年份、参数和事件组分别报告；
- 梯度归因由有限幅度扰动独立验证；
- 性能提升包含训练成本与实际推理吞吐，不只报告单 kernel 时间。

任何 Gate 失败都先分析状态表示、边界和守恒，不以扩大网络规模作为默认修复。

## 10. 指标规范

至少报告以下四类指标：

1. **状态误差**：逐字段 MAE/RMSE、归一化误差、偏差和极值误差。
2. **长期误差**：7 日、30 日、1 年和 50 年轨迹漂移及事件后恢复。
3. **物理误差**：逐日和累计水/碳预算 residual、非法状态次数。
4. **性能**：首次编译、热运行、每 landpoint-year 时间、内存和 CPU/GPU 成本。

AGB/BGB/GPP/NPP 不能代替内部状态验收。总量误差不能掩盖垂直土壤层、雪层、
碳库或日累计量中的补偿误差。

所有容差在运行测试集前固定。修改容差需要新版本 acceptance policy，不能覆盖旧
实验结果。

## 11. 代码评审规则

每个粗化研究 PR 必须回答：

- 改变的是 Teacher、边界适配、数据、模型还是分析？
- 是否改变 Teacher 默认运行结果？
- 新状态字段的 owner、单位和预算角色是什么？
- 数据 split 是否保持冻结？
- 是否给出相同硬件和命令下的 A/B 性能？
- 是否运行对应 Gate，而不是只给训练 loss？
- 是否增加隐式 fallback、重复数据副本或未登记输出？

禁止事项：

- 为了让粗化模型通过而调整 Teacher 科学参数；
- 用 Teacher 测试集生成 normalization 或做模型选择；
- 将约束失败样本静默过滤；
- 只保存图，不保存机器可读指标和配置；
- 将 notebook 作为唯一实现；
- 在未通过一年自由 rollout 前启动大规模 50 年训练。

## 12. 下一步分析顺序

当前不应立即选择网络结构。推荐顺序是：

1. 从 compiled complete-day carry 和 restart ledger 提取 daily-boundary 字段清单。
2. 明确 48 步边界内的水、能量和日累计预算；标出首版无法硬约束的字段。
3. 实现只用于测量的内存 replay ceiling，不加入神经网络依赖。
4. 根据速度上限决定是否继续粗化主路线。
5. 若继续，冻结 state schema、数据 manifest 和空间/时间/事件 split。
6. 实现无神经日尺度 baseline，再实现最小守恒 residual 原型。

这六步完成前，不建立大规模训练集、不提交 GPU 长任务，也不开展气候记忆梯度
分析。
