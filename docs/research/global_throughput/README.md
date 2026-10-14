# 全球物理模型吞吐研究

更新：2026-10-15。**当前CPU目标是在48核allocation内并发调度成熟独立单点程序，测完整任务吞吐和单位成本。停止推进64点vmap大程序路线。** GPU实用配置仍为已测B1024；不要求CPU凑相同batch形状，不把历史B64末位差异归因闭合作为前置。

最新审查对象先读[遥测修复与新候选](CPU_SCALAR_POOL_TELEMETRY_FIX.md)：按主管新工作单完成可选遥测修复、新pin封定、独立部署及免费非模型验证。最终pin`bae03d9ef78726f00987737cdc58eac705a922ae`，目标job15866807在cntest/1CPU/10秒COMPLETED，14项真实启动stub测试、launcher/parser dry-run和admission身份匹配通过；未运行物理模型。相同4CPU/4worker/90分钟单次pilot仅备妥，新增付费授权为0。旧[首次pilot结果](CPU_SCALAR_POOL_PILOT_RESULT.md)及免费参考加载15866448原样保留，复用81项接口检查和参考证据。另读[原方案](CPU_SCALAR_POOL_PLAN.md)、[接口验证](CPU_SCALAR_POOL_VALIDATION.md)、[服务器合同](SERVER_BENCHMARK.md)及[主管交接](SUPERVISOR_HANDOFF.md)。

## 当前可复用内容与缺口

- `SingleRunner`已有一次编译B1后串行遍历点的成熟执行路径；`PoolRunner`已有spawn持久worker、独立状态、单核绑定、轮转分配与完整唯一点汇总。本次扩展这条pool路径，按实际allocation限制worker数，分批启动编译，保留完整输出和严格门。
- 旧scalar job15820467的FULL64串行热core/wrapper中位127.074984/121.382450秒，896 point-days。它是一核探索记录，具有原GPU有限容差失败与科学接受false，不能称48核扩展结果。
- 该进程postcompile RSS约5.18GB、warm约5.57GB、HWM约19.94GiB；包含编译和整组输入/输出。持久化复用能摊薄启动，磁盘缓存能否跨worker命中仍待实证。没有证据保证48个进程能装进单个192GB节点。
- 首次4核/4worker paid job15866488止于launcher遥测：原RUN_ROOT仅launch-context，无cache/pool/worker/report/stage。本轮修复及非模型验证通过，但仍无物理worker新证据；相同4核/90分钟pilot（最坏¥0.30/¥0.42）等待主管审查，不自动执行。没有48worker内存支持证据，也没有证明192GB不足。

## 已保存证据

| 证据 | 可用结论与边界 |
| --- | --- |
| [pool4首次终态](CPU_SCALAR_POOL_PILOT_RESULT.md) | job15866488 FAILED1:0/1s，ibc11b04n04/4CPU；物理worker与所有数值门均not reached。完整父0.001111111核时，未核实费用¥0.00005556/¥0.00007778，旧scalar不重复计费。 |
| [v5保存数组短分析](CPU48_V5_SAVED_ANALYSIS.md) | 760叶全树复核；CPU内部3个有限超差元素/2点，无额外结构、dtype、离散值或非有限问题；strict FAIL与科学false保留。v5是单CPU JAX进程/device的64点vmap，未取得warm吞吐。 |
| [v5终态](../../../outputs/research/global_throughput/cpu48-b64-execution-v5-20261014/CURRENT_RESULT.md) | job15857765 FAILED1:0/819s；48核父10.92核时，费用情景CNY0.546/0.7644，已关闭。旧scalar父2197s的CNY1.464666667/2.050533333另计一次。原输出、pin和失败原因不改。 |
| [GPU B1024终态](../../../outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/CURRENT_RESULT.md) | job15671732的B1024×2：64真点×32重复、28672 lane-days；wrapper中位95.848153921秒，299.139825 lane-days/s。容量背景，非2048异质点或独立member验收。 |
| [既有CPU1/GPU背景比较](../../../outputs/research/global_throughput/cpu-gpu-b1024-comparison-20261011/CURRENT_RESULT.md) | 不同总工作量的归一化参考；其历史B64下一步已被本方案替代。费率0.05/0.07元allocated-core-hour与GPU2.20元/card-hour均保留为未核实情景，GPU辅助CPU另计。 |
| [碳核心同步](../../../outputs/research/global_throughput/carbon-core-sync-20261011/CURRENT_RESULT.md) | 研究分支保留两项碳修复；历史性能/物理对照pin不随分支HEAD改变，不将旧验收转授新科学版本。 |

严格同量CPU/GPU比较仍缺：CPU独立任务在同总任务、相同输入分布/时间窗下的实测，以及对应代码/科学身份与完整输出审查。不同分块合法；现有GPU重复样本不能证明真实669点或集合成员扩展。CPU4/48扩展、年度/50年、同机Fortran加速、10×、训练/AD及普遍科学验收均未取得。

## 范围与保留合同

本线面向论文669个独立PFT14网格的完整物理正向效率；原时间递推、48半小时步/天、native dtypes、760完整输出与masks、`atol=1e-8/rtol=1e-10`保持。x64不等于全算术FP64。静态Fortran分支ledger与运行科学验收仍分开；不能省过程、改容差或减少科学有效输出来提速。详见[精度边界](PRECISION_BOUNDARY.md)、[执行审计](EXECUTION_AUDIT.md)和[Fortran科学验收政策](../../source_audits/paper_scientific_acceptance_policy.md)。

固定cwd `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`、branch `research/global-throughput`；开工核对top-level/branch/实际import，不写main或daily树。[工作树管理](../../WORKSPACE_MANAGEMENT.md)和[共享部署入口](../../../../../../docs/deployment-explore1000.md)仍有效。全部旧单次paid额度均已关闭；本轮只授权修复、部署和免费验证，未增加paid余额，不自动重提或扩48。交付commit/push后停止，不联系、轮询或唤醒主管，不建替代聊天或自动任务。

历史原始制品和专门报告原样保留；本页及另外三份入口压缩前全文可从Git `8fac02c1d5ac67d9a7e829a8d3f47fbf73f29ad6` 对应路径恢复。历史工作单不自动恢复授权。
