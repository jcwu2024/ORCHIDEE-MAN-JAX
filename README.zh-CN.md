# ORCHIDEE-MAN JAX

[English](README.md) | 简体中文

本项目包含两个相互衔接的阶段：将 ORCHIDEE-MAN 论文版本的 PFT14 红树林
物理模型移植到可微框架 JAX，以及利用该模型训练基于 Transformer/Attention
的日尺度神经算子。JAX 教师模型已实现并在声明范围内通过验证；日尺度替代模型
仍在研发中。

## 第一阶段：可微 JAX 教师模型

教师模型（Teacher）保留 Fortran 的科学过程逻辑，并记录对应文件、过程和
行号范围。它每天仍然执行 48 次半小时状态推进，以及原有的日尺度植被和碳过程。

- 已实现 PFT14 单 landpoint 的 Driver、SECHIBA、HYDROL、DIFFUCO、
  ENERBIL、THERMOSOIL、CONDVEG、STOMATE、restart 年际交接和年度
  modelout。
- 完整的后续日状态转移以编译后的 7 日块运行，同时保持 Fortran 半小时过程和
  日过程的状态更新顺序。
- 当前正式运行路径已经通过 cold-start 14 日、restart 14 日、365 日，以及
  跨 landpoint 复用同一编译 executable 的语义门禁。
- 7 个此前未用于定向修复的 landpoint 已完成 1961-2010 年验证；其 2010 年
  AGB/BGB/GPP/NPP 的最大相对误差为 `8.5e-5`。
- 已通过教师模型的物理参数梯度验收 Gate D1：在声明的过程、多日传播和重启
  案例中，前向、反向自动微分与有限差分一致。具体范围见
  [梯度验收记录](docs/research/daily_coarse_graining/gate_d1_physical_parameter_gradients_v1.md)。
- 已开展日/多日块编译、编译结果复用、向量化和可断点恢复的生产运行优化。
  这些优化保留原来的物理时间步，并不意味着教师模型已经变成日尺度代理模型。

科学支持目前限于 PFT14 和已声明的验收矩阵，不代表所有 PFT、所有物理参数的
导数均已验证。完整 669 点的 Fortran 等价性发布验收，与下面的教师数据集验收
是不同的门槛。

## 第二阶段：日尺度神经过程算子

当前研究目标是用一次日尺度状态转移替代亚日尺度过程序列，同时保证 GPP、
呼吸、NPP、地上生物量和地下生物量等关键输出在长期模拟中保持准确。当前候选
采用 Transformer/Attention 网络，并与守恒状态更新器、保留的精确日过程耦合：

```text
当天起始状态 + 原生 6 小时气象驱动 + 相关参数/静态条件
  → Transformer/Attention 过程头：预测日尺度过程量
  → 一次受水、碳、能量约束的状态更新
  → 保留的精确日过程
  → 下一天状态
```

网络通过既有 `bind_daily_operator` 边界接入，使用规范输入组装器和保留日过程
适配器。参数按所属过程传入。最终日尺度推理不得重建 48 个半小时气象输入，
也不得执行 48 步状态扫描；通过传递物理上的下一日状态，连接成多年模拟。

**v6 教师数据集已完整生成，并通过神经训练数据验收**：覆盖 669 个陆地点、
1961—2010 年，共 33,450 个点年数据分片、12,208,581 次日状态转移。详见
[数据验收报告](docs/research/daily_coarse_graining/teacher_v6_full_production_acceptance_20260807.md)。
后续训练使用这一连续一致的数据版本，不重新生成，也不混入历史 v5 数据。
数据集验收通过，不等于日尺度模型的精度已经达标。

### 研究状态与研发入口

日尺度模型、物理过程耦合和训练/多日模拟基础设施已经实现。当前架构采用带
Q/K 归一化的 Transformer/Attention，训练目标包含单日状态和 7/30 日科学指标。
**目前仍没有通过所需长期精度验收的日尺度模型。**

当前任务、性能实测和未完成的验收统一记录在
[下一步](docs/NEXT_STEPS.md)与[当前状态](docs/current-status.md)。
这些精简入口替代不断累积的实验流水账。按照
[研发上下文维护规则](docs/RESEARCH_WORKFLOW.md)选择本次任务所需的代码和证据；
历史实验通过[历史索引](docs/research/daily_coarse_graining/HISTORY_INDEX.md)追溯。

365 日模拟、长期留出点验证、日尺度模型的物理参数梯度验收（Gate D2）和产品
验收仍是后续里程碑。教师模型的 Gate D1 不能替代神经模型的梯度验收。
GPU 用于训练；最终 CPU/GPU 推理性能和加速幅度需要通过完整工作负载实测确认。

## 分支与研究状态

- `main` 是冻结的、面向用户的 PFT14 半小时 Teacher 基线。
- `research/daily-coarse-graining` 包含完整 Teacher，以及仅用于研究的数据捕获、
  数据集生成和神经代理代码。在日尺度研究完成前，它与 `main` 保持独立。捕获开关
  默认关闭，普通 Teacher 模式必须与 `main` 保持行为对齐；详见
  `docs/branch-alignment-20260802.md`。

新接手者先阅读 [`docs/START_HERE.md`](docs/START_HERE.md)，再用
[`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) 确认当前工作顺序，用
[`docs/CODE_MAP.md`](docs/CODE_MAP.md) 确认代码归属。
[`docs/current-status.md`](docs/current-status.md) 记录详细证据，
[`docs/DOCUMENT_STATUS.md`](docs/DOCUMENT_STATUS.md) 标明当前有效的合同及历史报告。
当前事实在原位置更新，不再逐次追加旧的工作安排。超长的 `HANDOFF.md` 保留为
实验与服务器任务历史，不再规定下一步。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `jax_orchidee/` | 正式 JAX 模型、Driver、过程模块和 CLI 运行器 |
| `research/daily_coarse_graining/` | 日尺度神经算子、教师数据接口、损失、训练与多日模拟研究 |
| `configs/` | 可移植的 PFT14 论文案例配置 |
| `manifests/` | 外置数据契约和 669 个 landpoint 的元数据 |
| `scripts/hpc/` | Explore1000 教师运行、数据准备、神经训练和诊断启动脚本 |
| `scripts/dev/` | Oracle、审计、诊断和性能工具；部分辅助函数也被研究代码复用 |
| `tests/` | 单元、数值等价、Oracle 证据和集成测试 |
| `docs/` | 用户文档和由源码驱动的等价性证据 |
| `fortran_run_scripts/` | 不可变的论文历史运行协议溯源 |
| `fortran_source/` | 有许可的 Fortran 源码本地挂载点；源码树不进入 Git |
| `data/`、`reference/` | 本地或外置科学输入及 Fortran 真值；内容不进入 Git |
| `outputs/`、`traces/` | 生成结果、缓存和原始诊断；内容不进入 Git |

根目录中的 `AGENTS.md`、`SERVER_ACCESS.md`、`.agents/`、`.venv/` 和工具
缓存属于本地工作区控制文件，不在公开 Git 候选中。

## 安装

推荐使用 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone <repository-url> orchidee-man-jax
cd orchidee-man-jax
uv sync --frozen
uv run orchidee-jax --help
```

安装开发工具：

```bash
uv sync --frozen --extra dev
```

CUDA 是可选项，并且必须与服务器驱动版本匹配：

```bash
uv sync --frozen --extra cuda12
```

Explore1000 使用与 CentOS 7 兼容的固定依赖环境，而不是通用 lock 文件。参见
[`docs/deployment-explore1000.md`](docs/deployment-explore1000.md)。

## 外置数据

大型 forcing、静态输入、restart 和参考文件不存储在 Git 中。运行前设置：

```bash
export ORCHIDEE_DATA_ROOT=/shared/orchidee-data
export ORCHIDEE_REFERENCE_ROOT=/shared/orchidee-reference
export ORCHIDEE_OUTPUT_ROOT=/scratch/$USER/orchidee-jax
```

所需目录结构记录在 `manifests/paper_pft14_data.yaml` 中。使用以下命令检查：

```bash
uv run orchidee-jax inventory --config configs/orchidee_man_250919.yaml
```

如果没有设置这些环境变量，开发工作区会回退到仓库根目录下的 `data/`、
`reference/` 和 `outputs/`。

## 运行一个 Landpoint

下面的公开 CLI 运行物理半小时教师模型，不会选择实验中的日尺度神经模型。

```bash
uv run orchidee-jax run \
  --landpoint-id 001.0-071.0 \
  --start-year 1961 \
  --years 50 \
  --initial-state cold-start \
  --year-checkpoint-dir "$ORCHIDEE_OUTPUT_ROOT/checkpoints/001.0-071.0" \
  --resume-checkpoints on \
  --output "$ORCHIDEE_OUTPUT_ROOT/001.0-071.0.json"
```

面向用户的运行器默认启用已经验收的 7 日完整日块。正式运行器位于
`jax_orchidee/runners`；开发使用的 strict/A-B 诊断保留在 `scripts/dev`。

## 验收 Landpoint

```bash
uv run orchidee-jax validate-landpoints \
  --selection manifests/landpoints_669.json \
  --start-year 1961 \
  --years 50 \
  --initial-state cold-start \
  --strict-on-failure off \
  --output-dir "$ORCHIDEE_OUTPUT_ROOT/acceptance"
```

在生产集群上，应让每个进程或调度器 array task 只运行一个 landpoint。这样可以
隔离 JAX/XLA 内存、让 restart checkpoint 相互独立，并避开 Windows 上单进程连续
编译多个 landpoint 时曾出现的原生崩溃。

## 科学原则

- Fortran 源码是过程 truth。
- 运行时 reference 用于验证数值和状态行为，但不能替代由源码驱动的分支覆盖。
- 不使用近似过程或参数校准掩盖教师模型的缺失过程。日尺度代理则是明确的学习型
  近似，需要单独通过精度、物理预算和梯度验收。
- 离散状态必须精确相等；浮点验收使用记录在 `docs/source_audits` 中、按字段明确
  定义的容差。

## 源码与许可

公开发布前，项目所有者仍需选择 JAX 仓库许可证并补充引用信息。当前工作区中的
Fortran 源码树没有经过确认的顶层再分发许可证；除非确认拥有再分发权限，否则
公开发布时必须继续排除该源码树。

部署细节参见 `docs/installation.md`、`docs/data-layout.md`、`docs/running.md` 和
`docs/deployment-explore1000.md`。`docs/release-checklist.md` 记录了剩余的许可证、
引用、集群审查和 669 点验收门禁。
