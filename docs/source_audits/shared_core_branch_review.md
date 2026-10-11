# 共享物理核心与研究分支核查

核查日期：2026-10-11。当前结论：共同物理模型只需维护一套；研究中的共享代码
差异可以拆成三项通用修复、一个可选 Teacher 接口和两项批处理运行能力。
初始核查完成分类、证据复核与轻量测试。C1、C2、C3 已分别以独立提交
`db4dd8f`、`e317fa8`、`2a3e303` 接入主分支并通过针对性回归。日尺度已有
这三项同等实现；全球效率线已有 C1，仍需由其维护线程同步 C2、C3。
可选接口和批处理能力继续按下述范围分别评估。

## 初始核查基线

| 工作线 | 核查提交 | 固定目录 |
| --- | --- | --- |
| main | `7cae3e02adad23674767a590fdf4d27f239ce065` | `E:/ORCHIDEE-MAN-JAX` |
| 日尺度 | `7463642eb3ff20f2352bc5732e9904b5181f11ff` | 根目录下 `runtime/worktrees/daily-coarse-graining` |
| 全球效率 | `52326bd5e67629269a4a4a1c39885276eefd5ec8` | 根目录下 `runtime/worktrees/global-throughput` |

以三个提交的文件内容直接比较，`jax_orchidee/` 的差异只有以下四个路径。
初始 main 基线最近的物理源码提交为 `5910ba9`，其后至该基线的提交是管理文档更新。
研究工作树中其他任务的未提交修改不属于本核查基线。

| 路径 | 日尺度相对 main | 全球效率相对 main |
| --- | --- | --- |
| `jax_orchidee/driver/orchestration.py` | +4 行：可选通量捕捉与回调 | +63 / -23 行：提取可转换的日块函数 |
| `jax_orchidee/stomate/carbon_kernels.py` | +36 / -1 行：两项数值 / 梯度修复 | 相同 |
| `jax_orchidee/sechiba/diffuco.py` | 相同 | +7 / -1 行：缓存修复 |
| `jax_orchidee/driver/global_batch.py` | 不存在 | 新增 325 行：批处理研究适配器 |

其他差异包括研究脚本、测试和研究数据合同。日尺度还增加 SciPy 依赖、研究
配置 / manifests，并更新 manifests 导航；这些不等于共同物理公式发生变化。
既有生产配置和数据合同文件、`uv.lock` 没有出现在此次差异中。
提交、逐文件统计和配置差异记录在
[inventory.json](../../outputs/reviews/shared-core-20261011/inventory.json)。

## 逐项结论与接入顺序

| 编号 | 改动 | 应归属 | 本轮结论 |
| --- | --- | --- | --- |
| C1 | DIFFUCO PFT 掩码改为缓存不可变 NumPy 数组 | 通用执行正确性修复 | main `db4dd8f`；日尺度 `120c637` 已同步且 14 项通过；全球已有同等实现 |
| C2 | NPP 维持呼吸份额使用已有稳定比值原语 | 通用物理核 AD 修复 | main `e317fa8`，12 项通过；保留极小分母导数策略；日尺度已有，全球待同步 |
| C3 | 叶龄周转份额消去共同叶质量因子 | 通用物理核数值修复 | main `2a3e303`，20 项通过；保留原测试、适用范围和源身份；日尺度已有，全球待同步 |
| I1 | 日块可选捕捉 `daily_flux_labels` | 共同 Teacher / 数据导出接口 | 先处理回调兼容性，再与日块函数提取协调接入 |
| R1 | 提取 `_paper_later_day_block_transition_factory` | 共同 CPU / GPU 运行接口候选 | 保持原标量路径，按既定前向合同核查提取后的行为 |
| R2 | `global_batch.py` 的 lane 组装与 `vmap` | 全球运行能力候选 | 继续留在全球研究线；满足支持范围、生命周期和运行接口要求后可进入 main |

C1–C3 已接入 main；I1、R1、R2 仍是接入候选。三个通用修复均按具体源码和
回归单独整理，没有整体合并研究检查点；C3 原提交 `5fb886c` 包含大量其他研究
改动，C1 原提交 `ad6c0e1` 也包含容量实验准备材料，均不应整体用于同步修复。

### C1：与 GPU 批处理无关的缓存缺陷

核查基线 main 的 `_diffuco_pft_column_mask` 在 `lru_cache` 内创建 JAX 数组。若首次调用
发生在追踪中，缓存会保留该次 `DynamicJaxprTracer`。全球分支仅改为创建相同
布尔值的只读 NumPy 数组，由消费处在各自追踪中转换；没有改变物理公式或掩码值。

本轮用从分支回归改编的小数组测试检查 14 个 PFT 和两种独立追踪形状。
main 在 `jax.checking_leaks` 下 14 项全部复现泄漏；在同一 main 模块内注入
精确候选实现后 14 项全部通过，包括列替换结果、dtype、掩码内容与只读属性。
这不是“所有正常主模型运行都会失败”的断言。

源码及正式回归已作为独立提交
`db4dd8f1cc86a33dc8e343725fda2803f63fb358` 接入 main。该提交只包含
`jax_orchidee/sechiba/diffuco.py` 和 `tests/unit/test_diffuco_mask_cache.py`，
便于日尺度分支单独同步。修复后的 DIFFUCO 文件与全球分支 `52326bd` 完全一致。

在实际修复后的 main 上执行
`conda run -n ORCJAX python -B -m pytest -q -p no:cacheprovider tests/unit/test_diffuco_mask_cache.py`，
14 项全部通过；测试文件 Ruff 检查和 `git diff --check` 通过。
测试使用 `PYTHONDONTWRITEBYTECODE=1`；关闭 cacheprovider 产生的 `cache_dir`
配置警告不影响结果。记录见
[主分支回归日志](../../outputs/reviews/shared-core-20261011/main-diffuco-fix-pytest.log)。
预审[补丁](../../outputs/reviews/shared-core-20261011/global/diffuco-mask-host-cache.patch)
和 [global-evidence.md](../../outputs/reviews/shared-core-20261011/global/global-evidence.md)
保留初始复现过程。

日尺度已由其维护线程同步为 `120c637`，保留来源记录并通过同一组 14 项回归；
核对 main / 日尺度的 DIFFUCO 文件无差异。全球分支已有同等修复。
三条开发线的这项修复已一致，历史实验执行版本仍保留原身份。

### C2、C3：共同修复，不依赖神经网络架构

C2 来自 `9023de2a3fbb3ddb49f4476d9bdac4b003502d1b`。它保持 NPP 前向源商、
泵出碳库及掩码，改用 main 已有的稳定比值原语。该原语在
`abs(denominator) < sqrt(finfo(dtype).tiny)` 时把商的切线设为零；FP64 阈值约
`1.49e-154`。这是一项明确的导数策略，不是任意小尺度下精确求导的证明。
独立方向探针中 `n=d/4, dn=d, dd=0` 的数学方向导数为 1，而实现于阈值下为 0、
阈值及以上为 1，前向值均为 0.25。本次接入保留原有阈值和上述导数策略。

C3 保留 Fortran 顺序的叶、根、果等 stock / turnover 更新，只在有效正叶质量
区域以相对质量计算叶龄份额，避免数学上会抵消的巨大梯度项。实数代数一致，
浮点份额的计算顺序有变化；零、负、阈值以下及退化情况保持原路径。

本轮复核了 Fortran `stomate_npp.f90::npp_calc` 299–386、447–531 和
`stomate_turnover.f90::turn` 569–665，以及命名报告和已保存的数值证据。
C3 当前函数源身份与已接受执行版本一致。四个实际边界的 stock / turnover
未变，份额前向最大差异不超过 `3.33e-16`；后续四项完整损失的 CPU JVP 与
CUDA 目标坐标 VJP 对照通过原容差。全参数及前向回归通过，但其他状态差异仍保留。
它们不证明所有 PFT、所有状态或完整神经训练通过。

初始本地复核通过 6 项碳过程回归及全部 8 项周转条件测试。当前神经合批的
梯度失败不应覆盖 C3 已独立通过的证据，也不能由 C3 的通过反推整个神经训练合格。
源版本、原始证据路径和测试命令见
[daily-evidence.md](../../outputs/reviews/shared-core-20261011/daily/daily-evidence.md)。

两项修复随后在实际 main 导入路径下分别接入并验证：

- C2：`e317fa81b97b8b8c959f2e7dfdfba525bb54dec8`，仅修改
  `jax_orchidee/stomate/carbon_kernels.py` 和
  `tests/unit/test_stomate_carbon_kernels.py`。12 项 NPP 源表达式、负碳库阈值、
  掩码、微小维持呼吸梯度及生产适配器回归通过。
- C3：`2a3e303845154576c6f4b5c0b8155106e5bcc70b`，仅修改同一碳过程源码并新增
  `tests/unit/test_turnover_fraction_conditioning.py`。20 项周转、临界叶龄、
  非活动掩码、微小叶质量及 turnover / gap / kill 适配器回归通过，其中包含
  日尺度已有的全部 8 项条件测试。

接入后的碳过程源码和原碳过程测试文件与日尺度 `120c637` 完全一致；新增测试仅做
格式及等价可调用语法整理，归一化 AST 与原测试一致。NPP 函数 AST 与来源提交一致；
周转函数 SHA256 为已接受的
`f6240cf5ce64a47b9bbe0647b5e5715305e9a9230ea77f2bd9f42981ae35a93c`，
去除新增份额计算后，原物理 stock / rate / mask / 更新顺序语句的 AST 未变。
原测试模块的 Ruff 检查未新增诊断，新增周转测试文件 Ruff 检查通过。

命令、运行环境、源身份及原始日志见
[接入证据](../../outputs/reviews/shared-core-20261011/carbon-integration/integration-evidence.md)
和[机器记录](../../outputs/reviews/shared-core-20261011/carbon-integration/integration-receipt.json)。
本次复用上述已接受的有界 CPU / CUDA 证据，没有重跑历史 GPU 或完整模型任务。

### I1：四行改动包含回调接口变化

I1 来自 `2eb63842aefc5a685451e502815039bdc24b0ae1`，把日通量标签接入已有的
compiled block，并把捕捉开关加入缓存 key。底层标签类型与单日捕捉能力 main
已经存在。它是可选的 Teacher 导出能力，不需要搬入神经网络实现。

但日尺度版本无条件把 `training_output_projector` 从两个位置参数改为三个，
即使通量捕捉关闭也传入第三个 `None`。日尺度保留的
`scripts/dev/probe_ok_leak_outer_block_reentry.py::target_projector` 仍只有两个
参数；签名检查可确认三参数调用不兼容。这是日尺度旧诊断调用点的具体缺口，
该脚本不在 main，不能表述为 main 现有生产调用已经失败。

集成时需显式保持旧回调合同或采用可选 / 版本化的新接口，并检查相关调用点；
不要用捕获任意 `TypeError` 的方式掩盖回调内部错误。I1 与 R1 修改同一段日块
组织，应一起设计接口，但可以分别验证和提交。

### R1、R2：已有真实前向证据，尚不是完整生产入口

全球分支保持物理过程方程，把原日 scan 提成函数并在外层批量调用。
当前适配器只接受每点 `npts=1`、48 半小时 / 日、连续 7 日和兼容的静态配置，
拒绝 peat / TOPMODEL，且模块明确未提供生产 CLI 和持久可执行缓存。

单元测试使用模拟标量函数检查组装、拒绝条件与 lane 独立性；研究线另有真实
GPU 前向证据，不能把两者混为一谈：

- 固定 `bd9e4c04...` 的 job15662711：B512 / B1024 两阶段各 1027 门槛通过，
  保留 760 个原生输出叶和 1024 次 FULL64 对照。
- 固定 `8aa2e05a...` 的 job15671732：B1024 对照阶段通过 2051 门槛及 2048 次
  FULL64 对照；B2048 在第二连续块执行时 OOM，没有完整验收结果。

以上为 64 个真实点重复扩展的容量负载，不能替代真实 669 异质点、独立集合成员、
长期状态 / restart 和生产 IO 的对应验收。原证据分别见全球工作树的
[B1024 结果](../../runtime/worktrees/global-throughput/outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/CURRENT_RESULT.md)
和 [B2048 结果](../../runtime/worktrees/global-throughput/outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/CURRENT_RESULT.md)。

未来接入 main 应按拟支持的运行合同复用已有前向状态、输出、掩码、原生 dtype
证据并补实际缺口。全球当前研究正向吞吐，完整梯度验收不是本轮新加的接入
前置条件；若声明批量可微性，再按其明确用途补相应证据。

## 数据、执行版本与同步

固定 v6 Teacher 的生产报告记录 producer
`19087b171564f44b49eb96fa1217169deca5d57a`，其 `carbon_kernels.py` 与初始 main
核查基线 `7cae3e0` 相同，早于本次 C2、C3 接入。报告里的生产 / 哈希验收和 CPU smoke，不等于 main README
中仍待完成的 669 点长期 Fortran 等价验收；两种验收的目标不同。

Teacher 生成版本、prepared 对象构建版本、当前执行版本可以不同，必须分别保留
身份和消费者相容性。C2 保持源前向商，C3 允许上述有界浮点差异；这些事实本身
既不要求重新生成 Teacher，也不证明所有 prepared 包可无条件混用。本轮没有
核定当前活动 prepared 包的完整生产清单。既有 v6 / prepared 复用工作单、类和
模块序列化身份继续保留；不要改写历史 producer 标签。

同步状态：C1 在三条开发线上已有同等实现；C2、C3 已进入 main，日尺度原本已有
对应实现，不需盲目重复应用。全球效率线 `52326bd` 尚缺 C2、C3，由其维护线程在
完整阶段边界依次同步 `e317fa8`、`2a3e303`，并核查实际依赖的前向行为。正在执行、
排队或用于历史对照的代码和可执行文件继续保留原版本，不在原地替换。

主分支的这轮已知通用修复收尾完成，后续可开始按 Fortran 来源和分支覆盖账本
检查已有物理过程代码。I1 / R1 的兼容接口及 R2 的全球运行支持范围仍由相应
工作线推进，尚未自动纳入 main；本轮不能称为三个分支全部文件已经一致。

## 本轮验证与局限

本轮使用 ORCJAX / Python 3.11.15 / JAX 0.10.1 CPU。DIFFUCO 原 main 的 14 项
失败是有意复现；候选及接入 main 后均为 14 项通过，日尺度同步回归也为 14 项通过。
碳过程初始复核为 6 + 8 项通过；实际 main 接入后，C2 的 12 项、C3 的 20 项通过。
独立解析探针还核查了导数阈值、旧回调签名和已接受周转源身份。
保存证据链接位于上述分项报告和接入记录；
这些 `outputs/` 文件按项目规范不入 Git，本文保留可审查的结论、版本与复现入口。

知识图仅覆盖 main，generation `2026-10-10T08:46:50Z`；目标文件元数据已变更，
研究新增文件不在 main 图中，因此关键结论使用直接 Git 差异和源码回退核查。
本轮不是全模型 / 全分支科学验收，没有重新执行历史 GPU 任务、训练或生成数据。
本页随共同修复接入更新，不追加研究实验时间线。
