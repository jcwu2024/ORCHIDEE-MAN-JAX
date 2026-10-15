# CPU独立B1 pool v3单次pilot终态

更新：2026-10-15。**job15869035 FAILED1:0，完整父4秒；0个worker启动。**
本次明确授权的唯一提交已用完，remaining=0，不重提、不扩配、不取消其它作业。
可选GPU遥测修复已在真实付费启动中生效；失败点现为JAX导入前的物理核资源准入。
这不是数值/科学失败、编译超时或内存不足结果。

## 实际执行与失败边界

| 项目 | 实际记录 |
| --- | --- |
| pin/ref | `bae03d9ef78726f00987737cdc58eac705a922ae` / `codex/global-cpu-scalar-pool-v3` |
| cwd | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu-scalar-pool-v3`；回收时仍detached clean |
| 提交 | 人类新授权下仅一次 `bash .../cpu-scalar-pool-telemetry-fix-20261015/v3/PROPOSED_SUBMISSION.sh`；独占attempt与新registry record先于sbatch落盘 |
| 请求 | cnall，1node/1task/4allocated CPUs/4worker/0GPU，5400秒，nonexclusive/no-requeue，自动放置，无新增mem/nodelist |
| 不变界限 | 并发启动1，worker启动1200秒，命令300秒，RSS abort64GiB，host guard64GiB，RLIMIT_AS unset；canonical CPU环境及原数据/参考/科学门未变 |
| 实际allocation | `ibc11b02n24`，TRES `billing=4,cpu=4,node=1`，Slurm `CPU_IDs=47-50`，Requeue0，完整父ElapsedRaw4，ExitCode1:0 |
| 实际时间 | Submit09:13:37，Start09:14:30，End09:14:34（集群显示UTC+8，2026-10-15） |
| 实际子进程 | affinity `[24,25,79,80]`；`exec_boundary_ready`后`resource_admission_failed`；两份receipt均`JAX_imported=false/model_execution=false` |

`nvidia-smi`仍因无可用驱动返回非零，但warning后已输出`starting mode=pool`，
真实srun进入规范化Python执行边界。随后
`global_server_resources.admit`拒绝该affinity；报错为：

```text
allocation affinity must contain exactly the declared unique physical cores; require Slurm core binding, SMT is not another core
```

准确执行pin的`global_server_resources.py:61-67`以
`(package_id, physical_core_id)`去重，包含socket/package，非只按core编号误判。
本次同一launch-context内的`lscpu -e=CPU,CORE,SOCKET,ONLINE`表为：

| Linux logical CPU | CORE | SOCKET | 准入物理key |
| --- | ---: | ---: | --- |
| 24 | 24 | 0 | (0,24) |
| 25 | 25 | 0 | (0,25) |
| 79 | 23 | 0 | (0,23) |
| 80 | 24 | 0 | (0,24) |

所以仅3个唯一物理核，24与80属于同一核；这是实际拓扑表证据，非按CPU编号推断。
Slurm的CPU_IDs与Linux logical CPU编号不能直接等同。
后续只读`scontrol show node`显示`CPUTot=56/ThreadsPerCore=1/CoresPerSocket=28/Sockets=2`，
但启动时OS表公开112个logical CPUs；配置为`select/cons_tres`、`CR_CORE`、
`task/affinity,task/cgroup`，TaskPluginParam为null。
这些事实表明本次Slurm计量/绑定与OS实际物理拓扑并未满足准入合同；
尚未确定具体配置或flag原因，也不声称所有cnall节点都如此。

启动负载为53.85/54.57/52.39。终态后的node快照CPUAlloc51/CPULoad54.79仅为后续事实，
不能反推编译竞争；本次根本未编译，失败由资源准入明确抛出。

## 已取得与未到达的指标

RUN_ROOT为`.../runtime/outputs/research/global_throughput/server-benchmark/cpu4-15869035`，
实际只有`launch-context.txt`。pool4、worker目录、cache、report、raw output及stage均不存在。

| 要求指标 | 本次结果 |
| --- | --- |
| 4worker真实启动 | 0；pool创建边界not reached |
| 逐worker lower/compile/cache-hit | not reached；不能报告cache命中或共享编译 |
| 逐worker RSS增量/HWM、同时总RSS峰值 | not reached；无192GB支持/否定48worker的证据 |
| FULL64，完整760 native outputs/masks/dtypes，CPU strict | not reached；没有输出数组或数值门 |
| GPU探索门、隔离/reset | not reached；未改容差或科学策略 |
| first core/wrapper、5次warm core/wrapper | not reached；无完成任务吞吐/单位point-day成本 |
| 真实pool CPU利用率 | not reached；只有失败前启动的CPU记账 |

`sacct`完整父TotalCPU=0.812秒、wall4秒，比例0.203 busy-core equivalents，
相对4allocated CPUs为5.075%；这是短失败启动的记账，不能代表worker效率。
`.0` step TotalCPU=0.465秒/wall3秒；`/usr/bin/time`记录user0.24/system0.21/wall2.88秒，
约0.15625 busy-core equivalents，原打印CPU15%。time的MaxRSS18780KiB仅为
pre-JAX子进程峰值，不能作为物理worker、编译峰值或pool内存测量。

计划FULL64=64真点×14天=896 point-days，实际完成0。
这与GPU B1024×2的28672 lane-days不是同总工作量；无新CPU/GPU同量比较、
48worker可驻留、48核扩展、10×或科学接受结论。

## 费用与已关闭授权

完整父只计一次：`4 × 4 / 3600 = 0.004444444` allocated-core-hours。
未核实费率CNY0.05/0.07对应**CNY0.000222222/0.000311111**。
不把batch/extern/.0重复相加，不重计旧scalar、旧pilot15866488、旧vmap或免费预检。
本次最坏6核时/¥0.30/¥0.42的上限没有成为实际费用。
新授权submission_count=1、remaining=0；没有SSH不确定状态重提、付费重试或取消操作。

最小下一步是另行定义并审批**无模型physical-core绑定诊断**，先核对真实cnall
Slurm计量、进程affinity、sysfs socket/core与继承cpuset，再决定是否需要新pin。
`--hint`或`--threads-per-core`只能是待验证候选，不保证4billable CPUs即4physical cores。
如必须用同类cnall allocation，候选上限4billable CPUs×300秒为0.333333核时，
未核实最坏CNY0.0166667/0.0233333；未准备或执行该诊断，不能沿用本次额度。
不降低现有物理核门，不在此轮修改生产代码或启动下一次物理编译。

## 可复核制品

新证据目录：[cpu-scalar-pool-pilot-v3-20261015](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015)。

- [单次attempt](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/paid-submission-attempt.json)、[原提交回执](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/paid-sbatch.stdout)与原子登记收据记录一次提交及精确pin/资源。
- [原job日志](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/job-15869035.raw.log)、[原launch-context](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/launch-context.raw.txt)、[scontrol](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/terminal-scontrol.txt)保留实际失败边界。
- [sacct全部行](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/terminal-sacct-all.psv)与[唯一父行](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/terminal-sacct-parent.psv)分开保存；[分析JSON](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/result-analysis.json)记录成本、拓扑和not reached。
- [目录/缺失制品/SHA收据](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/terminal-inventory.json)、[节点快照](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/terminal-node-snapshot.txt)、[scheduler配置](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/scheduler-config.raw.txt)保留原始事实；最终[核验收据](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015/integrity-verification.json)核对源/目标字节及所有旧registry记录不变。

原job日志SHA256为`07fa451f2d0dc26c3b9d2bf8efc7b1fc9c5c23422feb00dbfb81d10950fabbb5`；
原launch-context为`3a268c5c85bacc8e86d47b6c877ff7bf6fdec0019b2fbc8412f21101bce29f3e`。
未重新加载/审计科学pickle、重跑参考预检、81/18项检查或模型测试；复用15866448、15866807旧证据。
图Verify沿用已记录generation，工作树excluded/not_tracked；本次只对资源准入准确源行作fallback，未重建图。
