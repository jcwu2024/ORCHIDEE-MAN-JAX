# 服务器吞吐合同：独立单点CPU worker与GPU B1024

更新：2026-10-15。当前CPU路线为成熟B1程序在48核allocation内并发调度独立任务，旧CPU48/B64 vmap路线关闭。[遥测修复与新候选](CPU_SCALAR_POOL_TELEMETRY_FIX.md)记录最终pin`bae03d9ef78726f00987737cdc58eac705a922ae`已独立部署、免费job15866807完成1CPU/10秒启动矩阵与身份/dry-run检查；没有新物理运行或paid提交。[首次pool4终态](CPU_SCALAR_POOL_PILOT_RESULT.md)保留旧job15866488在可选GPU遥测处FAILED1:0/1秒、零worker的原证据；该paid授权关闭。相同4CPU/90分钟新pilot只备妥，等待主管审查。[原方案](CPU_SCALAR_POOL_PLAN.md)与[验证记录](CPU_SCALAR_POOL_VALIDATION.md)仍作历史准备证据。

## 不变的科学与任务合同

每个任务独立持有自身状态、forcing和参数，先按原序完成Days2–8，再把该任务第一块末态传入Days9–15；每日48半小时步、日STOMATE更新不变。不得把时间段当独立任务，不得广播另一点状态、漏任务、重复完成同一task_id或用多点vmap替代B1 worker。

原生输出为两块共760叶，保留全部状态、日输出、masks和dtype；整数/布尔精确比较。浮点保持`atol=1e-8, rtol=1e-10`；只允许合同声明的bare-soil roughheight matched-NaN。结构、shape、dtype、非有限、任务隔离和冻结初态reset为硬门。原始误差、strict状态、参考来源及科学false均保留。当前CPU对旧scalar的strict gate不放宽；已存在的GPU有限差异探索策略不等于科学通过。未来如需单列有限超差后的探索计时，须另作明确门策略决定，本次未实现。

x64开启，地理payload/solar等原生FP32仍保留；不能声称全算术FP64。完整Fortran科学与分支覆盖另按[政策](../../source_audits/paper_scientific_acceptance_policy.md)，[精度边界](PRECISION_BOUNDARY.md)与[冻结输入准入](FROZEN_ARGUMENTS_ADMISSION.md)不变。

## CPU执行与资源边界

复用`benchmark_global_server_cpu.py`的`SingleRunner`/`pool_worker`/`PoolRunner`。每个spawn worker只编译B1可执行，持久复用处理分配的点；一个worker内部顺序处理任务。controller按唯一任务索引收齐完整输出，core计时包含派发/完成屏障；wrapper另包含输入传输、完整host输出及IPC。不能用controller CPU time替代worker CPU总和。

真实Slurm allocation、精确CPU预算和sysfs物理核去重先于JAX准入；48预算要求48个不同物理核。每worker固定一核，OMP/MKL/OPENBLAS/NUMEXPR为1，controller也留在同allocation内，若共享worker0核则明确记录。线程数量和affinity只证明边界，不能当有效核利用率或吞吐。

worker数显式受allocation预算和任务数约束；48核模式不会默认开48个进程。启动并发默认1，避免同时冷编译。持久cache只在专用namespace启用，逐worker记录StableHLO、cache事件、lower/compile与ready wall；不能因shape相同宣称cache hit或共享内存可执行。正常退出与失败均回收本pool的子进程。

总RSS采样abort、host MemAvailable入场、启动/命令超时提供受限停止边界；采样阈值不是Slurm/OS内存reservation，不能保证截获瞬时峰值。没有依据为JAX强制任意虚拟地址空间上限，RLIMIT_AS默认不设。

## 已知内存与启动证据

旧scalar job15820467一次B1 lower55.8773秒、compile684.2656秒；64点串行first137.3622秒。postcompile RSS5,180,694,528B、warm5,565,206,528B、Linux HWM21,406,240,768B（19.94GiB）。这些包含进程编译残留、全组输入与输出，既不是单点状态大小，也不是48份实际并发测量。

每个spawn仍独立调用lower/compile；磁盘cache命中未经跨worker实测。持久worker可以摊薄启动但不会消除既有编译成本。48倍RSS/HWM仅能作风险算术，不能证明可行或不可行。首次4worker pilot未启动任何worker，RUN_ROOT仅launch-context、无cache/pool/report/stage，因此没有新增驻留或编译数据，不能据失败认定192GB不足。可选遥测的窄修复与非模型控制流检查现已完成；新admission只改执行commit/目录。相同4CPU/90min pilot（最坏¥0.30/¥0.42）须经主管审查，本轮不提交，不承诺48资源。新检查不重算旧scalar或参考数组，也不替代完整科学/资源准入。

## 工作量与GPU比较

| 项目 | 既有可用证据 | 当前缺口 |
| --- | --- | --- |
| CPU1 FULL64 | job15820467，64真点×14天=896 point-days；core127.074984秒、wrapper121.382450秒；有原GPU有限差异和scientific false | 不是CPU4/48结果，不能乘48当实测 |
| GPU B1024×2 | job15671732，64真点×32重复=2048任务，28672 lane-days；core81.772918770秒，wrapper95.848153921秒（299.139825 lane-days/s） | 是重复容量样本，不代表2048异质点/独立member；历史pin验收不能转授当前研究科学树 |
| 旧GPU B64 | job15579869，64真点/896 point-days；wrapper6.222332641秒 | 可作同量小工作量背景，已被B1024实用选择取代为主问题；不重跑 |
| 首次CPU pool4 pilot | job15866488，计划64真点/896 point-days；launcher遥测失败，零物理worker | 无完成工作、数值门或性能数据，不构成B1024同总量比较或48核结果 |

正式CPU/GPU吞吐比较允许不同分块，但总任务数、真实点分布、初态、forcing/参数和时间窗口须匹配；重复任务要分别记task_id与source point index，不能当独立member。若使用GPU现有64×32背景，CPU也需相同有序2048任务实测与来源审查；本次接口仍以64真实点pilot为边界，未擅自扩大或重跑GPU。真正669异质点、不同成员或长期运行须另定义输入并验证。

## 指标与成本

逐次记录完成任务数、完成point-days、总wall、first-call与至少5个同步warm样本的中位/范围，完整输出checksum/门状态、逐worker CPU user/system、affinity、启动负载和cache证据。总RSS采样峰值与各进程Linux HWM分列；各HWM峰值可能不同时，不能求和冒充同时RSS峰值。core与wrapper是独立计时边界，不能相减中位数估算开销。

费用用完整父allocation的实际ElapsedRaw和allocated CPU/GPU计一次，含启动、编译、门、IO与失败；warm单位成本另算`allocated CPUs × warm wall/3600 × CPU rate / completed point-days`。GPU另加辅助CPU，资源请求量不等于实际busy-core数。记录CPU核时/GPU卡时及完成工作，费用情景CPU CNY0.05/0.07每allocated-core-hour、GPU CNY2.20/card-hour，账单均未核实。不得把编译当热速度、不同总量冷/热混比，或按worker数线性外推。

本次pool4完整父4CPU×1秒=0.001111111核时，费用情景¥0.00005556/¥0.00007778；无warm单位成本。免费预检15866448为cntest/1CPU/63秒，单列且没有物理执行；旧scalar完整父另保留、不重复计费。原始sacct全部行与唯一父行见终态报告。

## 历史与来源

[v5分析](CPU48_V5_SAVED_ANALYSIS.md)和[v5原终态](../../../outputs/research/global_throughput/cpu48-b64-execution-v5-20261014/CURRENT_RESULT.md)保留单进程vmap失败，所有旧授权均关闭。已有server-cost-results只包含GPU B16/B64 accepted和CPU1 B16 exploratory，不包含可直接用的pool4吞吐结论。

历史对照物理tree为`75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`、science SHA`11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`。研究分支另保留NPP/leaf-fraction两项碳修复；新CPU方案必须用精确对照物理pin加明确harness，不能隐式用HEAD科学树借旧参考通过。当前源码与执行pin分开见[方案](CPU_SCALAR_POOL_PLAN.md)。

固定工作树和共享部署规范见[入口](README.md)。没有年度/50年、匹配Fortran、10×或普遍科学接受结论。压缩前合同可从`8fac02c1d5ac67d9a7e829a8d3f47fbf73f29ad6`的本路径恢复；历史命令不构成新授权。
