# 全球吞吐主管交接

更新：2026-10-14。用户已纠正CPU方案：**复用成熟独立单点程序，把独立任务并发调度到48核，测完成吞吐与单位成本；停止CPU B64 vmap大程序路线。** GPU工程选择仍为B1024。有限末位差异深层归因不是CPU基线前置；科学严格失败、历史费用和输出完整保留。

先读[pool4首次终态](CPU_SCALAR_POOL_PILOT_RESULT.md)、[当前入口](README.md)、[原审批对象](CPU_SCALAR_POOL_PLAN.md)、[接口验证](CPU_SCALAR_POOL_VALIDATION.md)、[服务器合同](SERVER_BENCHMARK.md)。主管已批准并完成部署/免费预检和唯一paid提交：job15866488 FAILED1:0/1秒，止于CPU launcher可选GPU遥测；没有物理worker/lower/compile/run。本次remaining=0，已关闭，不修复重提。[保存数组短分析](CPU48_V5_SAVED_ANALYSIS.md)仍不继续扩展。

## 角色与工作树

当前主管聊天`01a128d2-3ec2-77a1-8d51-1665c220d2ca`，独立Sol执行聊天`01a128df-4038-72b0-83a7-144dd939cd76`。主管主模型GPT-6 Astra、执行主模型GPT-6.1 Sol，各自effort沿用用户选择；内部子代理不替代这两个独立聊天。本轮执行交付commit/push后停止，不发消息、轮询或自动唤醒主管，不创建替代聊天/自动任务。

固定本地cwd/top-level `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`、branch `research/global-throughput`；actual `jax_orchidee.__file__`必须在同工作树。不得写main、daily或切另一树分支。规则见[工作树管理](../../WORKSPACE_MANAGEMENT.md)、[共享部署入口](../../../../../../docs/deployment-explore1000.md)。旧主管/执行聊天只作历史保留。

## 当前交付与下一决策

生产pin为`50864f065363f6cd0de685c18d9c257723bf2838`，实际节点ibc11b04n04/4CPU；run目录只有launch-context.txt，无cache/pool4/worker/report/stage。177行裸nvidia-smi在set-e/pipefail管道中失败，194行pool dispatch未到达；并非数值/编译/内存失败。免费job15866448在cntest/ibtest01/1CPU/63秒通过保存GPU/scalar各760叶参考加载，未创建物理runner。

原始日志、sacct完整父行和所有step、目录与缺失制品收据已回传并核SHA，具体路径见终态报告。完整父4×1秒只计一次：0.001111111核时、未核实¥0.00005556/¥0.00007778；免费预检和旧scalar单列。逐worker编译/cache/内存、first/warm、strict CPU/reset均not reached；缺失证据不等于已证明192GB不足。

最小下一对象是审查CPU可选GPU遥测失败处理建议（launcher177行显式条件捕获非零）并做目标CPU非模型控制流检查，之后审查新immutable pin与相同4CPU/4worker/90分钟单次pilot、最坏¥0.30/¥0.42；本次没有改生产代码或新增额度。只有小pool实际证据后再决定48核worker/资源，不线性外推或强开48冷编译。

比较目标是CPU独立B1任务与GPU B1024在同总任务/输入分布/连续窗口下的完成吞吐及费用；允许不同分块。已有GPU B1024×2来自64真点×32重复、28672 lane-days，wrapper299.139825 lane-days/s，仅容量背景。CPU64点pilot未达同总工作量；真正同量CPU数据、48核结果和异质点/member扩展尚缺。不得把重复样本升级为独立member或把旧性能/科学验收转授新碳核心。

## 科学与证据保留

原完整物理/时间顺序、48半小时步/天、Days2–8后9–15、760 native leaves/masks、`atol=1e-8/rtol=1e-10`与离散精确比较保持。x64并非全算术FP64。当前strict CPU gate不改变；保留错误、来源和scientific false。如需在有限数值FAIL后另报探索计时，只能在结构/dtype/非有限/隔离/reset硬门成立后作独立科学策略决定，本次未实施。

- [v5终态](../../../outputs/research/global_throughput/cpu48-b64-execution-v5-20261014/CURRENT_RESULT.md)：job15857765 FAILED1:0/819s，单CPU进程/device的64点vmap，internal strict在3元素/2点失败；无reset/warm seal/CPU48吞吐。父成本情景CNY0.546/0.7644。
- [v3 scalar来源](../../../outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/CURRENT_RESULT.md)：job15820467，完整父2197s/48CPU只计一次，CNY1.464666667/2.050533333。scalar core/wrapper127.074984/121.382450秒，HWM19.94GiB；61/64 GPU门、3点4处有限失败和reset仅原封存证明保持。
- [GPU B1024](../../../outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/CURRENT_RESULT.md)：旧pin B1024完整stage严格准入，B2048第二块OOM，原父费用与失败保留，预算关闭。
- [新碳核心生命周期失败](../../../outputs/research/global_throughput/gpu-lifecycle-execution-20261011/CURRENT_RESULT.md)：job15769234首臂rveget容差失败、第二臂未运行，本轮已关闭，原因未归属；不重跑GPU或扩大诊断。
- [两项碳修复](../../../outputs/research/global_throughput/carbon-core-sync-20261011/CURRENT_RESULT.md)：研究分支保留NPP与leaf-fraction修复。历史对照tree `75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`另行保留，不回退研究分支，不借旧参考声明新科学版本通过。

所有旧单次额度均已消耗/关闭；不retry、延长、恢复余额、重生成scalar、跑GPU/B2048或重建环境。只写jcwu远端树，登录节点不计算，canonical环境不重建；下一次真实作业及部署必须纳入具体审批。

历史专门报告/原始制品仍在原路径；四入口压缩前全文可由Git `8fac02c1d5ac67d9a7e829a8d3f47fbf73f29ad6`恢复。按问题读命名报告，勿把历史“下一步”当当前工作单。新主管可用[开场指令](NEW_PROJECT_PROMPT.md)。
