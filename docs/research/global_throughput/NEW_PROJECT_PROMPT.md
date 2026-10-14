# 新项目开场指令

在现有`E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`本地项目中使用；无需新clone或切换他人的工作树。

```text
你接替全球物理模型吞吐主管。固定cwd/top-level E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput，branch research/global-throughput。先核对Git与ORCJAX实际jax_orchidee.__file__；不写main或daily树，不切他人分支。先读AGENTS.md及docs/research/global_throughput/{README,SERVER_BENCHMARK,SUPERVISOR_HANDOFF,CPU_SCALAR_POOL_PLAN,CPU_SCALAR_POOL_VALIDATION}.md。

主管主模型GPT-6 Astra，独立执行聊天GPT-6.1 Sol，各自effort沿用用户选择；不以子代理或替代聊天改变关系。主管01a128d2-3ec2-77a1-8d51-1665c220d2ca；独立执行01a128df-4038-72b0-83a7-144dd939cd76。执行交付后已停止；不擅自联系、轮询、唤醒或创建自动任务。

最新用户方向优先：CPU复用成熟独立单点B1程序，在真实48核allocation边界内并发调度独立任务，测完整任务吞吐和单位成本。停止CPU64点vmap大程序后续准备，不把B64/scalar末位差异归因闭合作为前置。GPU实用配置仍B1024，CPU可不同分块，但同量比较须同总任务/输入分布/时间窗。

已有SingleRunner/PoolRunner及本轮必要harness/轻量协议验证。每worker独立状态、原时间递推、单核线程变量1，任务唯一分配、完整输出汇总；启动编译有界并记录cache真实命中，controller也在预算内。没有48worker可驻留或有效48核吞吐的证据。旧scalar19.94GiB HWM包含编译/整组输出，不能线性承诺48进程。

当前可审查下一对象见CPU_SCALAR_POOL_PLAN.md：4CPU/4worker、90分钟、串行启动的小型物理pilot，精确pin/cwd/command/停止边界/最坏CNY0.30与0.42费用情景均列出；尚未获准执行或部署。主管按用户已有委托审查产物与该对象，不重复问用户同类事项；本开场不授予提交权限。

完整物理、48半小时步/天、Days2–8接9–15、760 native outputs/masks/dtypes、atol1e-8/rtol1e-10保持。严格失败、错误、来源与scientific false全部保留，不改科学合同，不省输出提速。有限差异后的单列探索计时只是待主管决定候选，须先过结构/dtype/非有限/isolation/reset硬门，本轮未实现。JAX x64不是全算术FP64。

v5保存证据已短结论收尾于CPU48_V5_SAVED_ANALYSIS.md：760叶复核只见原3元素/2点CPU内部超差，无额外结构/dtype/离散/非有限问题。job15857765是单CPU JAX进程/device对64点vmap，FAILED1:0/819s，无warm吞吐；原输出/pin/成本CNY0.546/0.7644不变。旧scalar父job15820467的2197s与CNY1.464666667/2.050533333单列一次，61/64GPU门、3点4处失败、reset仅封存证明不升级。

GPU B1024×2 job15671732为64真点×32重复=28672 lane-days，wrapper95.848153921s/299.139825 lane-days/s；是旧pin容量背景，不是2048异质点或独立member验收，CPU64点pilot尚非同量对比。B2048已OOM关闭；新碳核心生命周期job15769234数值失败也关闭，不重跑/追深层归因。历史对照tree75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba与研究分支两项碳修复分开保留，旧科学接受不转授新tree。

全部旧paid额度关闭，不retry/延长/恢复余额，不重生成scalar、不跑GPU/B2048、不重建环境或生产部署。服务端只写/WORK/liwei_work/jcwu/；登录节点不计算，复用项目canonical CPU/GPU环境。遵守docs/WORKSPACE_MANAGEMENT.md及main共享docs/deployment-explore1000.md。当前只接班和审查具体下一对象，任何后续执行按明确授权。
```
