# JAX CPU48 / GPU B64 同量对照合同

更新：2026-10-12。当前单卡工程选择继续为已接受的 **B1024**。CPU 主候选为 **单进程 native FULL64 B64 batch/vmap、48 个不同物理核**。CPU4仅是可选短入口检查。用户已明确[本次v2授权](../cpu48-b64-execution-v2-20261012/AUTHORIZATION.md)：修复资源读取并验证后，以原资源、费用上限及科学/测量合同新增一次作业。允许同规格健康低负载CPU节点和正常排队，不锁n35；`IDLE+PLANNED`和正常PENDING不拒绝提交或关闭。canonical runtime及未变科学预检复用，新执行版本单独审核部署。前次[未提交结果](../cpu48-b64-execution-20261011/CURRENT_RESULT.md)0次sbatch/无job/费用0保留，不复用旧授权。

实现、封装和检查见 [CPU48 准备结果](../cpu48-b64-preparation-20261011/CURRENT_RESULT.md)、[版本 pin](../cpu48-b64-preparation-20261011/execution-code-pin.json) 和 [具体运行提案](../cpu48-b64-preparation-20261011/PAID_PROPOSAL.md)。
前次唯一job**15782454**在`ibc12b07n02`获得48CPU/0GPU后启动失败，
`FAILED / ExitCode 2:0 / Elapsed 00:00:01`。launcher读取`AllocTRES`为空并退出；
原始scontrol输出未保存，字段为何不可读未知。无run root、模型执行、stage或性能/数值测量。
完整父计费`billing=48,cpu=48,node=1`只计一次，费用情景CNY0.000666667/0.000933333，非核实账单。
[前次终态证据](../cpu48-b64-execution-20261012/CURRENT_RESULT.md)已保存；旧单次授权耗尽、不重提。
用户随后明确“好的，继续”，批准[新v2修复与单次重跑](../cpu48-b64-execution-v2-20261012/AUTHORIZATION.md)：
先定位资源读取、保存原始错误并验证真实启动分支，再以相同资源/费用上限新增一次作业，不恢复旧额度。
新版本基于旧执行SHA、物理子树不变，仅纳入验证修复，使用全新immutable checkout和run root。
下列科学/测量协议继续不变。[修复](../cpu48-b64-repair-v2-20261012/REPAIR_REPORT.md)已通过独立源码审查和22项本地针对性测试。immutable v2 `b1b5cf0835da298bddb388e61d754feeb6c14cf2`已部署，104个物理源码文件与v1 base完全相同；目标ibtest01/GNU Bash4.2.46六类模拟检查及原始记录已验收，模拟本身未运行真实allocation或模型。见[部署回执](../cpu48-b64-repair-v2-20261012/deployment-receipt.json)。
唯一v2付费job**15805735**在`ibc12b08n35`终态`FAILED / ExitCode 1:0 / Elapsed 00:01:49`，实际48CPU/0GPU。真实allocation校验通过、scalar单核step已派发；lower后`_aval_signature`对Python `float`读取`.shape`触发`AttributeError`。scalar lower为54.8616s，未完成stage、科学gate或热样本，B64/48核step未运行。完整48核父作业109s只计一次，费用情景CNY0.072666667/0.101733333，非核实账单。见[v2终态结果](../cpu48-b64-execution-v2-20261012/CURRENT_RESULT.md)。本次授权已消耗，无自动重提；CPU48性能、数值接受和同量CPU/GPU比仍不可用。

## 比较对象和不可变科学合同

| 既有证据 | 可引用测量 | 边界 |
| --- | --- | --- |
| CPU job15591356 | scalar1、16真实点×14天；wrapper `7.052818167587637 point-days/s`；compile `688.378443662 s` | 两项 cross-GPU rveget 容差失败，科学接受stage为0；不能代替CPU48实测或完整B64内部参考。 |
| 已接受GPU B64，job15579869 | unique64×14天=`896 point-days`；wrapper median `6.222332641482353 s`，`143.99744462818447 point-days/s` | execution `2b14506992e9c9c7c851cef7b905606e55ac4e04`，science `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`；是下一次CPU48的同总工作量比较对象。 |
| 当前GPU B1024实用配置，job15671732 | FULL64 repeat32，共`28672 lane-days`；B1024×2 wrapper `299.1398250995909 lane-days/s` | 大批量归一化参考，不是CPU B64与该结果严格同总量的端到端比；后续B2048 OOM及parent失败费用保留。 |

GPU B64的64条 `native_input_lanes` 与既有FULL64 collector的 `landpoint_id` / `inputs_sha256` 逐序64/64相同，science与baseline相同。这是保存metadata的绑定证据，真实CPU读取及运行资源仍须入场检查。

执行保持原science `11b6…`、baseline `133e35aec7b83b9d79888c46092f64d518e97299`、64真实独立点、连续Days2–8 / 9–15、48半小时步/天、760完整native输出叶及masks。开启JAX x64，保留原dtype，包括地理payload和solar子计算的FP32。固定 `atol=1e-8, rtol=1e-10`，离散字段精确相同；不改物理核心、序列或容差。

当前研究树已有碳过程修复，不能冒充旧科学身份。v1从历史 `8aa2e05aa24b65dc4d30a27d4ee857bba6312221` 创建；v2以v1执行SHA `4f6b2fc906c453b9030acff489637a8f83605dca`为base，仅加入审核的启动/资源修复及必要helper/tests。整个 `jax_orchidee` 子树须与base完全相同，当前研究分支及碳修复保留。确切执行SHA、patch/bundle摘要和路径以新版本pin为准。

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

准备期只读快照找到健康 `IDLE` 的 `ibc12b08n35`：56物理核、CPUAlloc=0、load=0.01；旧n04已满、n01为DRAIN，不采用。见 [只读准备收据](../cpu48-b64-preparation-20261011/target-readonly-preparation.json)。前次v1实查选择`ibc12b07n02`，提交前56物理核、CPUAlloc=0、load=0.03；其启动失败未生成进程运行观测。v2实查后选择健康低负载`ibc12b08n35`，job15805735也实际运行在该节点；这些快照均不预留节点，B64/48核进程未运行。

## 成本和审批边界

父allocation上限：主情景 `48×1.5×0.05=CNY3.60`，保守 `48×1.5×0.07=CNY5.04`。费率是未核实账单的情景，不以scalar1耗时降低父费用。

core/wrapper热单位费用与整作业分列。完整作业估算优先使用终态sacct的Elapsed/AllocTRES；父作业只计一次，失败费用保留。没有终态accounting时，协调器start/end只能生成所覆盖区间的48核费用估算，整作业费用保持pending，不称完整账单或Slurm全部wall。

canonical Python、冻结输入/reference、科学协议、停止规则和费用上限见 [运行提案](../cpu48-b64-preparation-20261011/PAID_PROPOSAL.md)；v2使用新审核版本/路径及单次授权，节点按实查健康同规格选择。旧提交已关闭，v2失败/超时也不自动补交或扩大资源。源码核对使用Verify意图和限定source fallback：可用图索引属于根main且排除本研究worktree，不声称完整图覆盖。
