# 新项目开场指令

在以 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput` 为目录的新 Codex 项目中复制下段。
添加现成目录后选择本地/Local 使用已有工作树；无需 clone 或从 main 派生分支。

```text
你接替全球吞吐主管，当前主管聊天 ID：01a128d2-3ec2-77a1-8d51-1665c220d2ca。
本线研究完整物理模型正向吞吐及单卡点位×独立集合成员的单位成本；不接手日尺度神经研究。
沿用用户指定的两个独立聊天：主管主模型 GPT-6 Astra，独立执行主模型 GPT-6.1 Sol，各自 effort 沿用用户设置。
主管负责研究方向、派工、用户沟通和实际证据验收；独立 Sol 执行负责研发、测试、服务器执行和交付。
subagent-steward 已安装，只辅助各线程内部子代理安排，不替代独立执行聊天或修改主模型/effort。

先确认实际 cwd、git top-level、branch 和 ORCJAX 的 jax_orchidee.__file__。
固定 cwd E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput，branch research/global-throughput。
不往根 main 写研究，不切换别人的工作树；添加 UI 项目不证明旧聊天 cwd 改变。
先读 AGENTS.md，以及 docs/research/global_throughput/SUPERVISOR_HANDOFF.md、README.md、SERVER_BENCHMARK.md。
当前候选证据入口：outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014/CURRENT_RESULT.md、preparation-result.json、execution-code-pin.json、scalar-admission.json及PAID_PROPOSAL.md；本地artifact-review.json是补充审阅记录。完整free report保留在本地与服务器：/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014/free-check/15857559/report.json，SHA256 d70dd295df32227c6050db890c7732a1e54a6fe70bf7d5ea631a07b6a772247f。
候选已固定并部署：执行 SHA 0100c8b09be5137b822173e0a0be66cb64939727，ref codex/global-cpu48-b64-v5，parent 为 60dccbddde10be4295aab7e7fe55fa7864d3c3f3。仅六个harness/test文件改变；104物理文件、science SHA 11b6…、actual source SHA b334…与tree 75d8…均不变。登录节点部署收据记录bundle SHA和git bundle verify通过。
免费 job15857559 在 cntest/ibtest02 的300秒上限内 COMPLETED 0:0/65s，实际1CPU、affinity [0]；JAX/jaxlib0.4.38 CPU/x64读取64点FULL64输入及GPU/旧scalar参考（各760叶），四项身份比较全部一致，目标脚本耗时16.98秒。
诊断在生产共用的 prepare_references 后退出；未建 physical runner，未执行模型 lower/compile/run，phases/gates/timing为空，scientific_acceptance=false。allocation验证了pin/admission/proposal及六个harness哈希和launcher/coordinator/runner dry-run参数展开、计划job-ID路径替换；真实启动时的目录不存在门未在dry-run中执行，不代表48CPU准入。
提案为 cnall/1node/1task/48CPU/0GPU/3600s、nonexclusive/no-requeue/无新增mem或placement限制，未核实费率情景CNY2.40/3.36；付费命令未授权、未提交，旧额度不恢复。由主管审查该精确候选和命令后决定下一步。
参考身份修复的因果证据仍见 outputs/research/global_throughput/cpu-reference-identity-repair-20261014/CURRENT_RESULT.md；job15855041/15855062和55项本地测试不作为v5的48核执行或性能证明。
原scalar 61/64 GPU门、3点4处有限超差、reset只有封存证明和scientific_acceptance=false保留；没有新GPU运行或scalar生成。
历史执行证据入口：outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/CURRENT_RESULT.md、AUTHORIZATION.md、runtime-evidence-manifest.json、artifact-review/review.json。
固定 pin 与原命令在 cpu48-b64-preparation-v4-20261014/execution-code-pin.json、PROPOSED_SUBMISSION.sh；准备证据已被实际终态入口接替。
读 docs/WORKSPACE_MANAGEMENT.md 和 handoff 链接的共享主树 deployment-explore1000.md；其他历史按问题查阅。

以下为 v4 历史付费终态，不是当前候选：唯一付费 job15853683 已 FAILED 1:0/50s，该次授权消耗并关闭。
执行 SHA60dccbddde10be4295aab7e7fe55fa7864d3c3f3，ref codex/global-cpu48-b64-v4；原命令 SHAb86068c62c96464319175135e12e5d84ce2e494a00875aceaa01ee0b1a48c160 只执行一次。
实际代码路径 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v4，checkout 仍干净。
104 物理 blob/tree75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba 与 v3 相同，science11b6…/actual sourceb334…不变，研究树两项碳修复保留。
精确部署及免费 production 预检 job15853600（cntest/ibtest01/1CPU/10s）完成，实际 candidate identity/seal、15项 harness hash 和原路径参数展开通过；没有模型计算。
唯一付费作业按 cnall/1node/1task/48CPU/0GPU/3600s、非exclusive/no-requeue/默认 placement/无新增 mem 执行，实际节点 ibc11b09n36。
真实 child 的 post-transport 环境、48 个不同物理核和内存准入通过；JAX/jaxlib0.4.38 CPU/x64 随后初始化。
首故障在 benchmark_global_server_cpu.py:411 调用 global_server_reference.py:283 的 load_reference：scalar admission candidate provenance 与 current execution identity 不一致。
失败比较的完整 expected/actual 未保存，不把 admission job_id null 与实测15853683的表面差异认定为原因；预检通过不代表该 runner 参考加载通过。
原报告 phases=[]、gates=[]、timing={}。没有 B64 lower/compile、候选保存数组、五次 core/wrapper 热样本或 CPU48/GPU 吞吐/热成本比。
免费冻结 collector job15853855（cntest/ibtest01/1CPU/4s）完成，仅接纳 failure accounting；没有模型重生成。
新父50s×48=0.666666667核时，未核实费用 CNY0.033333333/0.046666667。
旧 scalar 父job15820467 的2197s×48=29.293333333核时、CNY1.464666667/2.050533333 已发生，只计一次，不含入新父费用。
本轮 batch-only 绑定旧 scalar checkpoint SHA1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa 和独立 admission SHAf71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d；没有 scalar 重跑。
scalar 原61/64 GPU门通过、3个有限 failing points/4处比较和 scientific_acceptance=false 保留；reset只有原封存证明，不能称 reset arrays 独立重算。

同量 GPU B64 job15579869 为64真实点/896 point-days，core中位5.632416466251016s、wrapper中位6.222332641482353s；各五次样本、输入身份、费用限制和缺失原远端文件边界见执行目录 GPU_COMPARATOR_EVIDENCE.md。
没有 CPU48 分母，不能发布同量速度或单位热成本比；GPU整job还含B16/编译/验证，不能把整job费用除以896冒充热B64费用。
用户当前单卡工程选择仍是旧科学pin已接受的 B1024；job15671732 的299.139825 lane-days/s 是不同总量的工程参考，不能与B64混样。
B2048历史job在wrapper第二块OOM；新碳核心生命周期job15769234 的单点 rveget/PFT14 超差定位已接受且执行关闭，原因未归属。
旧性能/科学验收不转授新科学pin；1280/1536、显存优化、超差溯因、GPU/B2048 重跑不是本次前置。
保留完整物理、原序递推、48半小时步/天、Days2–8及9–15、760 native leaves/masks/dtypes 和固定 atol1e-8/rtol1e-10；x64不等于全算术FP64。
没有真实669异质点/独立member、年度/50年、匹配Fortran加速或普遍科学验收结论。

当前请用中文完成只读接班、实际终态和证据审阅，再提出供讨论的具体下一步；本开场不授权新修复、部署、作业、环境/科学/容差变更或扩大溯因。
所有旧单次付费额度保持关闭，不修复重提、不延长或复用余额；不创建自动任务、轮询主管、发送消息或自动唤醒旧聊天。
当前独立执行【用Sol接手JAX全球吞吐研究】ID01a128df-4038-72b0-83a7-144dd939cd76，交付后停止工作，等待派工。
旧主管01a1075c-aa28-7c11-a242-9ca5128673bc、旧执行01a1077d-30a0-7792-b7d1-a67660a7b889仅保留历史。
后续派工以用户授权为准；需要新独立聊天时由用户创建或明确授权创建，不能自行替换协作关系。
普通文档/日志维护复用已完成证据，不重复全检、物理编译或重开旧 DIFFUCO ABI 故障。
```
