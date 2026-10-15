# 新项目开场指令

在现有`E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`本地项目中使用；无需新clone或切换他人的工作树。

```text
你接替全球物理模型吞吐主管。固定cwd/top-level E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput，branch research/global-throughput。先核对Git与ORCJAX实际jax_orchidee.__file__；不写main或daily树，不切他人分支。先读AGENTS.md及docs/research/global_throughput/{CPU_SCALAR_POOL_PILOT_V3_RESULT,README,SERVER_BENCHMARK,SUPERVISOR_HANDOFF}.md；CPU_SCALAR_POOL_TELEMETRY_FIX及旧PILOT_RESULT/PLAN/VALIDATION按需查证。

主管主模型GPT-6 Astra，独立执行聊天GPT-6.1 Sol，各自effort沿用用户选择；不以子代理或替代聊天改变关系。主管01a128d2-3ec2-77a1-8d51-1665c220d2ca；独立执行01a128df-4038-72b0-83a7-144dd939cd76。执行交付后已停止；不擅自联系、轮询、唤醒或创建自动任务。

最新用户方向优先：CPU复用成熟独立单点B1程序，在真实48核allocation边界内并发调度独立任务，测完整任务吞吐和单位成本。停止CPU64点vmap大程序后续准备，不把B64/scalar末位差异归因闭合作为前置。GPU实用配置仍B1024，CPU可不同分块，但同量比较须同总任务/输入分布/时间窗。

已有SingleRunner/PoolRunner及本轮必要harness/轻量协议验证。每worker独立状态、原时间递推、单核线程变量1，任务唯一分配、完整输出汇总；启动编译有界并记录cache真实命中，controller也在预算内。没有48worker可驻留或有效48核吞吐的证据。旧scalar19.94GiB HWM包含编译/整组输出，不能线性承诺48进程。

最新唯一paid pilot已经终态：新明确授权下仅提交job15869035，最终pin bae03d9ef78726f00987737cdc58eac705a922ae，ref codex/global-cpu-scalar-pool-v3，目录.../runtime/worktrees/global-throughput-cpu-scalar-pool-v3。实际cnall/ibc11b02n24/4allocated CPUs，FAILED1:0/4秒；attempt和原子registry新record先于sbatch落盘，remaining0/no retry。nvidia-smi失败warning后真正到srun/pre-JAX入口，实际affinity[24,25,79,80]被物理核门拒绝，JAX_imported=false/model_execution=false，0worker。RUN_ROOT仅launch-context，所有lower/compile/cache/RSS/HWM、760 strict/reset、first/warm与pool利用率均not reached；无192GB支持/否定48worker的证据。

按同次launch-context的lscpu实际表：CPU24->core24/socket0，25->25/0，79->23/0，80->24/0，仅3个(socket,core)key；24与80同核。hard gate以(package_id,physical_core_id)去重，包含socket。Slurm CPU_IDs47-50不能等同OS logical CPU。node配置CPUTot56/ThreadsPerCore1，而OS表公开112logical CPUs；CR_CORE、task/affinity,task/cgroup原始配置已保存。尚不归因具体flag或竞争，不能承诺4billableCPU=4physical。父TotalCPU0.812s/wall4s=0.203busy core、归一4CPU为5.075%，仅短失败启动不是模型效率。父费用只计一次4×4/3600=0.004444444核时，未核实CNY0.000222222/0.000311111，不重计step/旧scalar/旧pilot。

本次相对旧50864只改launcher和startup测试，jax_orchidee/configs/scripts/dev未变；资源/部署/tee/srun错误仍fatal。admission SHA259936bb09f5d0006895b1f5ae17fa298c0baada07edcb23b2355ed892ef796a仅改候选execution_commit/dir。旧免费15866807的14项startup/identity/dry-run、81/18回归与参考加载15866448直接复用，本次没有重跑预检、反序列化科学数组或跑模型。中间失败与旧证据均保留。

旧唯一paid pilot已关闭：pin50864f065363f6cd0de685c18d9c257723bf2838、物理tree75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba；job15866488在ibc11b04n04/4CPU FAILED1:0/1秒。可选nvidia-smi失败后未到run_case，RUN_ROOT仅launch-context，零物理worker。逐worker编译/cache/内存、first/warm、strict CPU/reset全not reached，没有48驻留支持证据，也没有证明192GB不足。

免费参考预检job15866448在cntest/ibtest01/1CPU/63秒COMPLETED，GPU/scalar各760叶加载通过、无物理runner。旧付费父4×1秒=0.001111111核时与未核实¥0.00005556/¥0.00007778另列一次；原收据见CPU_SCALAR_POOL_PILOT_RESULT.md。旧额度和v3新额度均submission_count1/remaining0，不恢复或自动提交。最小下一对象是另审无JAX/模型的physical-core绑定与Slurm计量诊断，先核affinity/sysfs/billing；free cntest须注明对cnall的可比性限制，若必要另批cnall4CPU/300秒，则最坏0.333333核时、未核实CNY0.0166667/0.0233333。hint/threads-per-core只是待验证候选；本轮未实施、未修改生产代码或新pin，后续paid必须另审。

完整物理、48半小时步/天、Days2–8接9–15、760 native outputs/masks/dtypes、atol1e-8/rtol1e-10保持。严格失败、错误、来源与scientific false全部保留，不改科学合同，不省输出提速。有限差异后的单列探索计时只是待主管决定候选，须先过结构/dtype/非有限/isolation/reset硬门，本轮未实现。JAX x64不是全算术FP64。

v5保存证据已短结论收尾于CPU48_V5_SAVED_ANALYSIS.md：760叶复核只见原3元素/2点CPU内部超差，无额外结构/dtype/离散/非有限问题。job15857765是单CPU JAX进程/device对64点vmap，FAILED1:0/819s，无warm吞吐；原输出/pin/成本CNY0.546/0.7644不变。旧scalar父job15820467的2197s与CNY1.464666667/2.050533333单列一次，61/64GPU门、3点4处失败、reset仅封存证明不升级。

GPU B1024×2 job15671732为64真点×32重复=28672 lane-days，wrapper95.848153921s/299.139825 lane-days/s；是旧pin容量背景，不是2048异质点或独立member验收，CPU64点pilot尚非同量对比。B2048已OOM关闭；新碳核心生命周期job15769234数值失败也关闭，不重跑/追深层归因。历史对照tree75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba与研究分支两项碳修复分开保留，旧科学接受不转授新tree。

全部旧及v3 paid额度关闭，不retry/延长/恢复余额，不重生成scalar、不跑GPU/B2048、不重建环境。本开场不授权后续诊断执行、额外部署或paid提交。服务端只写/WORK/liwei_work/jcwu/；登录节点不计算，复用项目canonical CPU/GPU环境。遵守docs/WORKSPACE_MANAGEMENT.md及main共享docs/deployment-explore1000.md。当前只接班和审查具体下一对象，任何后续执行按明确授权。
```
