# 独立单点CPU worker：48核目标与受限pilot

2026-10-14状态更新：此精确对象已获主管批准，部署与免费参考预检已完成，唯一付费job15866488已FAILED1:0/1秒，止于CPU launcher可选GPU遥测，未启动物理worker。本次授权消耗、remaining=0，不修复重提；见[完整终态](CPU_SCALAR_POOL_PILOT_RESULT.md)。**以下保留原审批对象文本与当时准备状态，不能用其中“批准后”命令恢复执行权限。** 没有新scalar生成；历史严格失败不撤销，不将其深层溯因当CPU吞吐前置。

## 怎样使用48核

直接扩展现有`benchmark_global_server_cpu.py::PoolRunner`：按allocation内不同物理核启动B1单点worker，每worker固定一核、OMP/BLAS/NUMEXPR为1；按唯一task index轮转分配，各自保持状态并完成两段连续7日窗口，逐点复用自己的resident B1 executable。controller共享allocation内一核，按原task顺序汇总全部760 native输出、masks/dtypes与完成计数。没有多点vmap worker，也没有额外第49核。

原scalar/PoolRunner、完整两窗口`run_two`、prepared输入、旧scalar与GPU数组、StageStore和reference admission均复用。启动默认串行，可显式有界；48模式强制显式worker数，不默认48。错误、超时、RSS超限均回收本pool，保留失败报告；固定atol1e-8/rtol1e-10不改。

## 内存和启动是否可控

可以约束启动并发、总RSS采样阈值、host入场门与超时；**尚不能证明48个进程在192GB节点驻留可行**。旧scalar HWM19.94GiB含编译/整组输出；postcompile5.18GB、warm5.57GB同样不能当真正单点worker占用。每worker仍调用lower/compile，跨worker persistent cache hit没有实证。缓存、稳定HLO签名与逐worker增量RSS必须在同一次受限运行里测量，不能把父进程编译结果视为spawn共享。

最小pilot采用4个worker串行ready，每次新增worker记录前后RSS/HWM与cache事件，再完成一次64点任务组及5+5热样本。总RSS abort阈值64GiB、host入场guard64GiB、启动并发1；RLIMIT_AS不设。采样不是OS硬限制，瞬时峰值仍可能先OOM；遇资源或数值硬门失败即停止，不增加worker、不重试。若4worker证据不能支持安全48驻留，交付测量和新的资源选择供审查，不启动48个并发冷编译。

## 下一次具体审批对象

候选执行pin：`50864f065363f6cd0de685c18d9c257723bf2838`，ref `codex/global-cpu-scalar-pool-v1`。只将本轮harness/test覆到历史对照物理tree `75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`；研究分支的NPP/leaf-fraction碳修复完整保留。新candidate绑定原scalar独立审计收据，不重算数组或模型；不沿用v5 candidate admission冒充新pin。

- partition cnall；1node、1task、4allocated CPUs、0GPU；nonexclusive、no-requeue、Slurm自动放置，无旧nodelist或新增mem reservation。
- 请求wall90分钟（5400秒）；每worker启动timeout1200秒、命令timeout300秒；父allocation硬上限始终有效，超时只保留已完成证据。
- 费用：4×1.5×CNY0.05=CNY0.30；保守4×1.5×0.07=CNY0.42。两费率均未核实，完整父作业只计一次；这不是旧B64授权余额。
- 拟目标cwd：`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu-scalar-pool-v1`；该目录本阶段未部署。Python为项目`.venvs/orcjax_cpu/bin/python`，不重建环境。
- 64真点、Days2–8后9–15、896 point-days；复用canonical prepared SHA`be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`、GPU数组SHA`a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6`、旧scalar SHA`1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa`。

本地已封定[execution-code-pin.json](../../../outputs/research/global_throughput/cpu-scalar-pool-preparation-20261014/execution-code-pin.json)、[新external admission](../../../outputs/research/global_throughput/cpu-scalar-pool-preparation-20261014/scalar-admission.json)和[精确提交脚本](../../../outputs/research/global_throughput/cpu-scalar-pool-preparation-20261014/PROPOSED_SUBMISSION.sh)。admission SHA为`fafd685bb54c7bb366f0d10d38cba31957ba754b9a5bcd8801e75abfd7907883`；它只重绑既有审计收据，保留strict false/reset限制/旧父费用。8个harness/test blob来自研究提交`8925ec506012da8f9155c8e90b7c282889e32369`；执行pin的全部物理blob与0100c8历史base完全相同。增量bundle已在本地verify，未传输或部署。

批准后，提交入口的精确命令为：

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu-scalar-pool-preparation-20261014/PROPOSED_SUBMISSION.sh
```

该脚本固定cwd/pin、4核/90min、全部reference SHA和pilot参数，清除继承的`SBATCH_*`额外约束；不附带mem、exclusive或node限制。实际执行`slurm_global_server_cpu_4core.sh`且只走pool，RUN_ROOT为`.../server-benchmark/cpu4-${SLURM_JOB_ID}`，cache为该新目录下`scalar-worker-cache`。完整展开命令见脚本的`--dry-run`；本阶段只在本地执行这一打印分支。目标Bash实际参数展开与相同pool身份的参考加载预检仍是部署后的边界，不能跳过。

[本地准备验证收据](../../../outputs/research/global_throughput/cpu-scalar-pool-preparation-20261014/preparation-validation.json)保存提交脚本纯打印、实际launcher的精确参数展开、环境归一化wrapper与真实Python parser的dry-run结果：pool4/4worker、启动1200秒/命令300秒、RSS/guard64GiB，JAX未导入、模型和提交均为false。它补充[81项接口回归](CPU_SCALAR_POOL_VALIDATION.md)，不重复物理编译或用toy证明760叶/内存性能。候选代码已通过独立只读并发/回收审查。

预检拟限`cntest/ibtest`、1task/1CPU/300秒，只运行`diagnose_global_cpu_references.py`，传相同pool4/4worker/FULL64及新admission参数，cache改为off、新output目录；该入口在共享`prepare_references`后退出，不创建物理runner。它不是4核/48核性能准入或新的scalar生成。

批准对象包括该精确pin的必要部署、目标shell纯dry-run/保存reference加载预检和**一次**上述pilot；当前交付不执行这些动作。目标shell/部署runtime仍须验证，不能用本地toy代替。由主管按用户既有委托审查产物后决定，不重复向用户申请同类授权。

## 怎样解释结果与比较成本

记录逐workerlower/compile/ready/cache、启动总wall、first-call、5个core和5个wrapper样本、完成task数/point-days、CPU user/system、总RSS采样峰值和每进程HWM，保留严格门和完整输出。core包含dispatch/barrier，wrapper包含完整计算/传输/IPC；两组独立计时中位数不相减。父sacct的完整allocated-core-hours单列；warm费用按实际allocation与warm wall/完成工作量另算，不能按忙核数打折。

本次pilot仅验证worker现实性，不能声称48核吞吐。最终48核目标必须实际完成同量任务并测wall，不由48 affinity/线程数或线性乘法推算。GPU现有B1024×2样本为64真点×32重复、28672 lane-days，299.139825 lane-days/s；CPU64点pilot不是同量比较。后续可按相同有序2048任务、相同分布与连续窗口测CPU，允许B1调度与GPU分块不同；repeat要标为容量任务，不能当新异质点/独立member。当前没有擅自扩展任务集或重跑GPU。

如果CPU/GPU有限差异仍出现，保留strict FAIL、全部误差与scientific false；是否在结构/dtype/非有限/isolation/reset硬门成立后另采exploratory timing是待主管明确决定的候选，不是本次实现。历史B64不再追因，不恢复其额度。[保存数组结论](CPU48_V5_SAVED_ANALYSIS.md)、[验证记录](CPU_SCALAR_POOL_VALIDATION.md)与[合同](SERVER_BENCHMARK.md)共同限定本交付。
