# ORCHIDEE-MAN JAX

[English](README.md) | 简体中文

这是面向 PFT14 红树林配置、以源码为依据的 ORCHIDEE-MAN 论文版本 JAX
实现。所有科学过程逻辑均从 Fortran 源码移植，并记录对应文件、过程和行号范围。

## 当前范围

- 已实现 PFT14 单 landpoint 的 Driver、SECHIBA、HYDROL、DIFFUCO、
  ENERBIL、THERMOSOIL、CONDVEG、STOMATE、restart 年际交接和年度
  modelout。
- 完整的后续日状态转移以编译后的 7 日块运行，同时保持 Fortran 半小时过程和
  日过程的状态更新顺序。
- 当前正式运行路径已经通过 cold-start 14 日、restart 14 日、365 日，以及
  跨 landpoint 复用同一编译 executable 的语义门禁。
- 7 个此前未用于定向修复的 landpoint 已完成 1961-2010 年验证；其 2010 年
  AGB/BGB/GPP/NPP 的最大相对误差为 `8.5e-5`。
- 669 个 landpoint 的完整验收尚未完成。

这里的范围声明刻意不扩展到其他 PFT 或当前不支持的 ORCHIDEE 配置。

## 分支与研究状态

- `main` 是稳定、面向用户的 PFT14 半小时 Teacher 和正式 CLI。
- `research/daily-coarse-graining` 包含完整 Teacher，以及仅用于研究的数据捕获、
  数据集生成和神经代理代码。捕获开关默认关闭，不改变正式 Teacher 行为。

日尺度神经链路已经在技术上接通，但还不是经过科学验收或面向用户的模型。目前已完成
Linux CPU 与 V100 兼容性、编译后的 Teacher 标签捕获，以及可恢复的 landpoint-year
数据分片生成。仍需完成有边界的试点数据集、单步学习验证和自由 rollout 门禁。GPU
目前主要用于批量神经网络训练；神经网络推理继续同时支持 CPU 和 GPU，最终根据完整
工作负载的实测性能选择，而不是预先规定必须使用 GPU。

权威的当前状态见 [`docs/current-status.md`](docs/current-status.md)。带日期的研究报告和
源码审计是历史证据快照，可能描述较早的阶段。

新对话如需接手 `research/daily-coarse-graining`，应首先阅读
[`docs/research/daily_coarse_graining/HANDOFF.md`](docs/research/daily_coarse_graining/HANDOFF.md)。
其中记录当前运行任务、严格验收门、唯一下一步，以及不应重新争论或重做的既定决策。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `jax_orchidee/` | 正式 JAX 模型、Driver、过程模块和 CLI 运行器 |
| `configs/` | 可移植的 PFT14 论文案例配置 |
| `manifests/` | 外置数据契约和 669 个 landpoint 的元数据 |
| `scripts/hpc/` | 每个 landpoint 独立运行的通用调度器模板 |
| `scripts/dev/` | Oracle、审计、诊断和性能工具；不是正式运行依赖 |
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
- 不使用近似过程或参数校准掩盖缺失过程。
- 离散状态必须精确相等；浮点验收使用记录在 `docs/source_audits` 中、按字段明确
  定义的容差。

## 源码与许可

公开发布前，项目所有者仍需选择 JAX 仓库许可证并补充引用信息。当前工作区中的
Fortran 源码树没有经过确认的顶层再分发许可证；除非确认拥有再分发权限，否则
公开发布时必须继续排除该源码树。

部署细节参见 `docs/installation.md`、`docs/data-layout.md`、`docs/running.md` 和
`docs/deployment-explore1000.md`。`docs/release-checklist.md` 记录了剩余的许可证、
引用、集群审查和 669 点验收门禁。
