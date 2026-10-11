# 下一次 JAX CPU48 / GPU B64 同量对照

更新：2026-10-11。当前单卡工程选择继续为已接受的 **B1024**。CPU 主候选为 **单进程 native FULL64 B64 batch/vmap、48 个不同物理核**。CPU4 仅是可选短入口检查，不是研究终点或强制前置。当前范围是可审阅的执行准备；尚未部署、提交作业或测量真实 CPU48 性能，旧预算不复用。

实现、封装和检查见 [CPU48 准备结果](../cpu48-b64-preparation-20261011/CURRENT_RESULT.md)、[版本 pin](../cpu48-b64-preparation-20261011/execution-code-pin.json) 和 [具体运行提案](../cpu48-b64-preparation-20261011/PAID_PROPOSAL.md)。

## 比较对象和不可变科学合同

| 既有证据 | 可引用测量 | 边界 |
| --- | --- | --- |
| CPU job15591356 | scalar1、16真实点×14天；wrapper `7.052818167587637 point-days/s`；compile `688.378443662 s` | 两项 cross-GPU rveget 容差失败，科学接受stage为0；不能代替CPU48实测或完整B64内部参考。 |
| 已接受GPU B64，job15579869 | unique64×14天=`896 point-days`；wrapper median `6.222332641482353 s`，`143.99744462818447 point-days/s` | execution `2b14506992e9c9c7c851cef7b905606e55ac4e04`，science `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`；是下一次CPU48的同总工作量比较对象。 |
| 当前GPU B1024实用配置，job15671732 | FULL64 repeat32，共`28672 lane-days`；B1024×2 wrapper `299.1398250995909 lane-days/s` | 大批量归一化参考，不是CPU B64与该结果严格同总量的端到端比；后续B2048 OOM及parent失败费用保留。 |

GPU B64的64条 `native_input_lanes` 与既有FULL64 collector的 `landpoint_id` / `inputs_sha256` 逐序64/64相同，science与baseline相同。这是保存metadata的绑定证据，真实CPU读取及运行资源仍须入场检查。

执行保持原science `11b6…`、baseline `133e35aec7b83b9d79888c46092f64d518e97299`、64真实独立点、连续Days2–8 / 9–15、48半小时步/天、760完整native输出叶及masks。开启JAX x64，保留原dtype，包括地理payload和solar子计算的FP32。固定 `atol=1e-8, rtol=1e-10`，离散字段精确相同；不改物理核心、序列或容差。

当前研究树已有碳过程修复，不能冒充旧科学身份。执行版本从历史 `8aa2e05aa24b65dc4d30a27d4ee857bba6312221` 创建，只加入明确审核的harness/tests；整个 `jax_orchidee` 子树与base完全相同，当前研究分支及碳修复保留。确切执行SHA、patch/bundle摘要和路径以版本pin为准。

## 一个父 allocation 内的协议

1. 单核step建立同版本FULL64 scalar snapshot。旧B16不能代替它。沿用现有 `cpu-internal` 协议，5次同步core + 5次wrapper；cold compile、first call、输入准备和输出IO分列。scalar最多2700秒。
2. 完整scalar结束且剩余协调器窗口至少1800秒后，启动48核step：`--mode batch --cpu-budget 48 --sizes 64 --warm-repeats 5 --validation-mode exploratory --cpu-reference-role cpu-internal`，显式绑定scalar目录。一个进程、一个native CPU JAX device；B64是lane数，不宣称等于48个有效线程。
3. 保留GPU每点比较、CPU内部完整整组比较、permutation、replacement/isolation和frozen-reset门。只有有限值跨GPU小容差失败可沿用exploratory timing；CPU内部或其他结构/数值门失败硬停止。内部对照不晋升科学接受。
4. 独立collector验证seal、输入/科学身份、两块完整输出、原gate及运行观测，从保存数组重算GPU与CPU内部比较。参考不完整、超时、候选窗口不足、版本/资源不符或步骤失败时停止；无自动retry/requeue或恢复旧GPU任务。

完整2048重复负载的CPU适配只在未来需要该级别端到端对照时再选，不是B64前置；本轮不重跑GPU或建立CPU核数扫描。

## 核、线程和内存

父allocation为cnall、1节点、1 task、48 allocated CPUs、无GPU、90分钟 / 5400秒、非exclusive、no-requeue。所有阶段按父48核计费，scalar1 step没有单核折扣。

CPU48在JAX导入前建立48个不同物理核的精确affinity：`XLA_FLAGS=--xla_cpu_multi_thread_eigen=true`、`OMP_NUM_THREADS=48`，`MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`。编译后及warm后保存线程mask；每次first/core/wrapper记录OS进程CPU时间、wall、busy-core等价数和step/parent利用率。环境/affinity不证明实际利用全部48核，收益仍待实测。

scalar MemAvailable guard为24GiB，batch48为64GiB；保存compile前后、first/warm后RSS/HWM和MemAvailable。guard不是内存reservation或OOM保证。历史单进程峰值约19.925GiB、compile-end约4.776GiB；48份独立进程朴素常驻外推约229GiB，超过节点192GB / 约187GiB可见量，因此主候选采用单进程vmap。Slurm RealMemory=1失真，不编造 `--mem` 或申请whole-node exclusive。

只读快照找到健康 `IDLE` 的 `ibc12b08n35`：56物理核、CPUAlloc=0、load=0.01；旧n04已满、n01为DRAIN，不采用。快照不预留节点；提交前须重新检查健康state、occupancy和负载，启动再次保存实际环境。见 [只读准备收据](../cpu48-b64-preparation-20261011/target-readonly-preparation.json)。

## 成本和审批边界

父allocation上限：主情景 `48×1.5×0.05=CNY3.60`，保守 `48×1.5×0.07=CNY5.04`。费率是未核实账单的情景，不以scalar1耗时降低父费用。

core/wrapper热单位费用与整作业分列。完整作业估算优先使用终态sacct的Elapsed/AllocTRES；父作业只计一次，失败费用保留。没有终态accounting时，协调器start/end只能生成所覆盖区间的48核费用估算，整作业费用保持pending，不称完整账单或Slurm全部wall。

具体code/cwd、canonical Python、冻结输入/reference、新输出路径、命令、停止规则和费用上限见 [运行提案](../cpu48-b64-preparation-20261011/PAID_PROPOSAL.md)。付费提交和远端部署仍须主管取得明确授权；本准备不启动物理编译、GPU或集群计算。源码发现使用Verify意图和限定source fallback，graph transport不可用，不声称完整图覆盖。
