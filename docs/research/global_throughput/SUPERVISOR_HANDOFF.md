# 全球 JAX 运行效率：新项目主管交接

更新：2026-10-14。CPU48 v4 唯一付费 job **15853683** 已 **FAILED 1:0/50s**，本次尝试关闭。
先读[v4 实际终态](../../../outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/CURRENT_RESULT.md)、
[单次执行授权](../../../outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/AUTHORIZATION.md)、
[v4 准备交付](../../../outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/CURRENT_RESULT.md)、
[固定 pin](../../../outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/execution-code-pin.json)、
[协议](../../../outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/EXECUTION_PROTOCOL.md)
和[精确提案/命令](../../../outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/PAID_PROPOSAL.md)。
执行 SHA `60dccbddde10be4295aab7e7fe55fa7864d3c3f3`，ref `codex/global-cpu48-b64-v4`；
来源研究检查点 `493252e418f3fd827af4a6f7748a63d3157e680d`，104 物理 blob 与 v3 基底相同，研究碳修复保留。
候选 B64-only，绑定旧 job 15820467 的 scalar 和独立 admission SHA；3 个有限跨 GPU 超差和科学 false 不变。
64 输入身份/点序、760 叶/GPU 门已独立重算；reset 只有原封存证明。
真实 production checkout 已部署；免费预检 job15853600 在 ibtest01 自然完成，实际 candidate identity、参考封存、15项 harness hash 和原路径 dry-run 通过。
唯一原命令 SHA256 `b86068c62c96464319175135e12e5d84ce2e494a00875aceaa01ee0b1a48c160` 已执行；
cnall/1 node/1 task/48 CPUs/0 GPUs/3600s，非 exclusive/no-requeue/无新增 mem 或节点限制。
实际节点 ibc11b09n36；付费 child 环境、48 个不同物理核和内存准入通过，随后初始化 JAX0.4.38 CPU/x64。
第一实质故障为 `load_reference` 的 scalar admission candidate provenance 与当前 execution identity 不一致；
失败比较双方未完整序列化，未归属具体差异原因。生产 identity/seal 预检不覆盖该 runner 的完整参考加载路径。
原报告 phases/gates 为空、timing 为空；无 B64 lower/compile、候选数组、热样本或 CPU48/GPU 速度/热成本比。
免费 collector job15853855 仅接纳失败记账；新父50s×48=0.666666667核时，未核实费用 CNY0.033333333/0.046666667。
旧父 job15820467 的2197s×48=29.293333333核时、CNY1.464666667/2.050533333 单列且只计一次。
同量 GPU B64 样本见[比较证据](../../../outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/GPU_COMPARATOR_EVIDENCE.md)；
缺少 CPU48 分母，不发布速度或单位热成本结论。B1024 仍仅为不同总量参考。
没有取消、延长、修复或重提，无 scalar 重生成、GPU/B2048、环境或科学合同变更；旧额度均不恢复。
执行聊天交付后停止工作，等待派工，不联系/轮询主管，不建新聊天或自动任务。

历史 v3 终态先读[CPU48 v3新单次授权](../../../outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/AUTHORIZATION.md)、
[v3实际执行入口](../../../outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/CURRENT_RESULT.md)、
[CPU48 v3准备结果](../../../outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/CURRENT_RESULT.md)、
[新的单次作业提案](../../../outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/PAID_PROPOSAL.md)、
[已接受的有界AOT修复](../../../outputs/research/global_throughput/cpu-aot-signature-repair-20261012/REPAIR_REPORT.md)
及[CPU/GPU现有证据比较](../../../outputs/research/global_throughput/cpu-gpu-b1024-comparison-20261011/CURRENT_RESULT.md)。
用户2026-10-13授权主管批准，主管已完成实际产物审查并批准精确v3的一次CPU48作业及必要部署、预检、运行、采集。
执行聊天已跟进到job **15820467**终态FAILED1:0/2197s：scalar FULL64完成，B64资源入场报告OMP线程预算不符，
历史child实际OMP值未记录；未进入B64 lower/compile或热计时，无CPU48吞吐/CPU48-GPU比值。v3单次授权已消耗，旧v1/v2额度均不恢复，不重提。
scalar原始5core/5wrapper中位127.074984s/121.382450s，896 point-days；61/64GPU参考门通过、3有限容差失败，冻结reset通过。
collector仅失败记账，未独立接纳scalar数组/吞吐。父allocation2197s×48=29.293333核时，未核实费用情景CNY1.464667/2.050533；scalar HWM19.94GiB。
用户已选择接受的B1024作为当前单卡工程选择，主问题转为CPU/GPU速度与单位成本；
不把1280/1536探索、显存优化或跨版本小容差失败作为前置。
旧[B1024双臂获批执行](../../../outputs/research/global_throughput/gpu-lifecycle-execution-20261011/CURRENT_RESULT.md)、
[本地准备](../../../outputs/research/global_throughput/gpu-lifecycle-preparation-20261011/CURRENT_RESULT.md)
及[碳核心同步结果](../../../outputs/research/global_throughput/carbon-core-sync-20261011/CURRENT_RESULT.md)按问题查阅，原pin和证据保留。
job15769234已终态FAILED1:0/1140s，追加保存数组CPU/NumPy只读定位已获主管接受；
本轮科学执行关闭，原因未归属；
**不恢复任何 GPU 诊断、付费作业或旧额度**。
可复制[新项目开场指令](NEW_PROJECT_PROMPT.md)。

## 1. 角色、目录与开场检查

当前主管聊天为 `01a128d2-3ec2-77a1-8d51-1665c220d2ca`；
当前独立执行聊天【用Sol接手JAX全球吞吐研究】为
`01a128df-4038-72b0-83a7-144dd939cd76`。
二者实际固定 cwd 均为 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`，
分支均为 `research/global-throughput`。

旧主管聊天【研究 JAX 全球运行效率】`01a1075c-aa28-7c11-a242-9ca5128673bc`，
以及旧执行聊天【用Sol接手研究JAX全球运行效率】`01a1077d-30a0-7792-b7d1-a67660a7b889`
仅作历史保留；旧执行已结束 B2048 OOM 并交接。两旧聊天保持休眠，不作为当前派工或接续入口。
旧聊天 cwd 为根 main；添加项目不自动修改已有聊天 cwd。
把现成 global-throughput 目录添加为项目，新聊天选择本地/Local，直接使用已有工作树；
无需重新clone或从main派生新分支。若检查不符，先定位到指定工作树，不切换他人的树。
本项目固定 cwd 为 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`，
分支 `research/global-throughput`。日尺度神经算子在另一个工作树，非本线任务。

新主管先核对 top-level、branch 和实际 module 路径，不在根 main 或他人工作树编辑：

```powershell
git rev-parse --show-toplevel
git branch --show-current
conda run -n ORCJAX python -B -c "import jax_orchidee; print(jax_orchidee.__file__)"
```

用户指定的协作结构是**两个独立的 Codex 聊天**：主管线程主模型使用 GPT-6 Astra，
独立执行线程主模型使用 GPT-6.1 Sol，各自 effort 沿用用户设置；以后模型调整以用户选择为准。
主管负责研究方向、任务分配、用户沟通和实际证据验收；Sol 执行线程负责具体研发、
修复、测试、服务器执行、文档和结果交付，复用已完成的证据。
这与日尺度线【日尺度神经算子研发】指挥【用Sol接手日尺度神经算子研发】的组织方式相同；
本线当前对应上述主管与独立执行【用Sol接手JAX全球吞吐研究】。

`subagent-steward` 已安装，是各线程内部安排子代理的辅助规则；上述独立主管/执行关系
来自用户的明确选择。不得把它改写成主管只派临时 Sol 子代理，也不需要重装或重定义技能。
后续获准开始研究时，主管向用户指定的独立执行线程下达任务、读取进展、验收实际文件。
当前主管与独立执行均使用上述固定全球工作树；旧 main 聊天继续保持休眠。
若改用新的独立执行聊天，由用户创建或明确授权创建，并将其主模型设为 GPT-6.1 Sol。
不要假设添加新项目就迁移了旧聊天，不自动改变已有线程的模型/effort。

本次开场先用中文确认理解、提出具体下一步和待决事项，不自动唤醒旧执行聊天或启动新实验。
文档/日志小修不反复全检或编译，也不绕回已修旧故障。

## 2. 研究问题与不可变合同

用户最关心**单卡点位×独立集合成员的吞吐和单位成本甜点**，不是 GPU 利用率、
堆卡或重复输入数量。现有669点是2°×2° forcing格点上的独立单点模拟；更细forcing
需配套 mapping 和验证。MAN 的 PFT、soiltiles、垂直层及格点内部交换不等同横向
嵌套地理子网格；本任务不自动修改单点 synthetic 面积。669不是永久并行上限。

当前 capacity screen 是64个真实 FULL64 点重复构造设备负载，不能称真实669异质点
或独立集合成员已验收。保留完整物理正向、原序时间递推、48半小时步/天、连续
Days2–8及9–15、全部 native outputs/masks。JAX x64开启，但原地理payload和solar
子计算含FP32，不能称全算术FP64。固定 `atol=1e-8, rtol=1e-10`，离散字段精确相同。
不得以改精度、放宽容差、省略过程或削减科学有效输出提速。
AD/校准价值另论；当前没有梯度/训练、匹配Fortran加速比或普遍科学验收。

## 3. 当前单卡工程选择：接受的B1024；B2048执行OOM

本节性能与科学验收只属于第6节的历史执行pin；不能转授已同步的新碳核心。

先读[fixed1024结果](../../../outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/CURRENT_RESULT.md)
和[fixed2048终态](../../../outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/CURRENT_RESULT.md)。

| 对比 | 同工作量 / 批布局 | 五次同步 wrapper 中位数 | 吞吐 lane-days/s |
| --- | --- | ---: | ---: |
| job15662711，同job/process/code | 14336 lane-days，B512×2 | 52.195214s | 274.661200 |
| 同上 | 同量，B1024×1 | 47.405972s | 302.409157 |
| job15671732，独立新job control | 28672 lane-days，B1024×2 | 95.848153921s | 299.139825 |

旧同job对比 B1024 吞吐提高10.1026%、热单位费用降低9.1756%；是最佳已测批大小，
不是已证明全局最优。新control不能与旧302.409157混样或跨job精确归因。
两job均为1 V10032GB+4 Slurm CPUs；不是多卡收益。

job15671732 的 B1024×2 是64真实点×32重复=2048 lanes×14天，严格collector通过
2051 gates、760 native leaves、2048 FULL64 saved-output comparisons；
core中位81.772918770s=350.629529 lane-days/s；热wrapper主费用
`2.22861220984e-6 CNY/lane-day`。B2048编译成功，但prepared-wrapper第二连续块申请
9,161,696,464B（约8.53GiB）时执行OOM；不是guard拒绝、673/536 ABI故障或科学gate失败。
没有sealed B2048 checkpoint，不发布其速度比；postcompile analysis未落盘，真实
peak VRAM未知。失败不证明所有B2048布局都不可能，也不证明批量曲线已到平台。

该job parent3800s、终态FAILED1:0，估计全job费用主/保守CNY2.533333333/2.617777778，
低于批准cap4.80/4.96；预算已关闭，不复用余额。冷compile约15–19分钟另计。
prepared-wrapper/热费用不含完整年度/50年运行、生产IO、启动/编译或失败候选成本。
主费率CPU0.05元/allocated-core-hour、GPU2.20元/card-hour，4辅助CPU另加，合计
2.40元/hour；保守CPU0.07则2.48元/hour。账单未核实。

[CPU1已存结果](../../../outputs/research/global_throughput/cpu-performance-results-15591356/CURRENT_RESULT.md)
wrapper为7.052818167587637 point-days/s。对最新接受的B1024×2
299.1398250995909 lane-days/s，归一化比例42.414226；CPU串行的16真实点是GPU FULL64的前缀，GPU为64点×32重复，
不是严格同工作量加速比。CPU仍有两项cross-GPU严格容差失败，accepted CPU stage count为0。
主情景热wrapper每百万点日/lane-days为CPU1 CNY1.969268、GPU CNY2.228612；
CPU单价0.07的保守情景变为2.756975/2.302899，初步费用方向反转，均非核实账单。
主情景GPU加4辅助CPU共2.40元/小时，假设同价48CPU线性吞吐338.5353，打平需88.362971%并行效率；
CPU4/48实际性能均未测。48核只是算术假设，不能作为48独立进程单节点方案：
CPU compile-end RSS约4.776GiB，按此作48份常驻进程的朴素外推约229GiB，超过192GB节点；单进程峰值约19.925GiB。
原始样本、版本/输入边界、冷热费用与缺口见[本轮比较](../../../outputs/research/global_throughput/cpu-gpu-b1024-comparison-20261011/CURRENT_RESULT.md)。

## 4. v4 当前终态与 v3 历史边界

当前 v4 实际执行和采集已完成，以本文开头链接的终态入口为准。主管按用户已有委托权限批准精确提案，
原固定命令只产生 job15853683 一次；生产路径预检和真实48核入场通过，实际 runner 在参考 provenance 门失败。
这没有新增 B64 物理、科学或性能证据；源代码未修复，执行 checkout 仍封存在原 pin。本次仅交付后等待派工。
下面的 v3 pin、5400s 提案、scalar 测量和终态是历史证据，不是当前 v4 命令或恢复额度。

v3阶段已完成**终态交付**。主管按用户授予的审批权限批准该精确提案，唯一job15820467已FAILED1:0/2197s；
必要精确部署、正常预检、单次提交、终态跟进和采集按 v3 执行入口核查。旧额度不重提；v4 定向修复来自上方新派工。
主候选仍为单进程native FULL64 B64 batch/vmap、48个不同物理核；CPU4不是强制前置，也不采用48份独立scalar进程。

主管已接受研究提交`d33fa65bb263b1564ca98f72b81a69e93aabc33f`的有界AOT修复。
首个实际失败叶子`.values_by_component[4][67]`是Python float；helper改用JAX `shaped_abstractify`，
tree/shape/dtype/weak_type审计和native call不变。复用[确切Linux候选证据](../../../outputs/research/global_throughput/cpu-aot-signature-repair-20261012/linux-jax-0.4.38-candidate-full-metadata.json)：
JAX/jaxlib0.4.38/x64，两组64点各536叶元数据一致；实际SingleRunner scalar/B2连续两块core/wrapper用物理transition替身，完整carry保留。
lane0/4的42处weak_type差异仍由严格构造审计拒绝，native call原生接受；仅元数据/接口获接受，完整物理和科学结果未测。

执行pin `03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`，ref `codex/global-cpu48-b64-v3`，
直接以v2 `b1b5cf0835da298bddb388e61d754feeb6c14cf2`为parent，从研究检查点
`8a860c144f6263fed5b9bf246a88f6d83d36797f`只移入六个必要harness/tests。
全部104个物理文件及整个`jax_orchidee`树与v2相同，tree OID `75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`，
science证书`11b6…`/actual source hash `b334…`不变，研究树两项碳修复保留。
AOT文件SHA `bc8ce1d62f6947228c8c8438344987345bf18095f307c90a2f45dc4655c6891e`与已接受候选相同。
[pin清单](../../../outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/execution-code-pin.json)
记录bundle/hash/size/v2 prerequisite、科学和运行绑定。ORCJAX/JAX0.10.1/x64准备15项测试通过；正确v3的source_identity通过，
原10项容量protocol/launcher失败按具体ID全部通过；另10项AOT/source-certificate筛选通过（50未选）。
[原始回归记录](../../../outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/regression-recheck-20261013/REGRESSION_RECEIPT.md)
区分两组检查，不宣称整模块60项重跑或目标JAX0.4.38 runtime接受。
原研究模块50通过/10失败只因两项碳修复的源码身份差异，旧报告保留；不改guard、回退碳修复或重开DIFFUCO。

实际独立代码目录为`P/runtime/worktrees/global-throughput-cpu48-b64-v3`，输出为
`P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/${SLURM_JOB_ID}`，
其中`P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`。canonical CPU Python仍为`P/.venvs/orcjax_cpu/bin/python`。
v3按新授权精确部署，部署/运行状态以本次执行入口为准；旧pin/目录/结果不改。准备产物及小型入口检查边界见
[准备结果](../../../outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/CURRENT_RESULT.md)。

[新提案](../../../outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/PAID_PROPOSAL.md)
固定cnall/1node/1task/48CPU/0GPU/5400s、非exclusive/no-requeue/无新增`--mem`；健康匹配节点正常排队，不锁n35，
提交前清除继承的SBATCH节点、exclusive、memory等设置，按新SHA/代码路径绑定入口。
完整父allocation含scalar均按48核计费，worst-case为48×1.5×0.05=CNY3.60，保守0.07情景CNY5.04，费率未核实。
先完整scalar参考再B64，各5 core+5 wrapper；scalar最多2700s，候选须至少剩1800s。
64 unique FULL64点、Days2–8和9–15、48步/天、896 point-days，对已接受GPU B64作同量比较；
冻结payload/reference、760完整native outputs/masks、x64/原dtype、`atol=1e-8/rtol=1e-10`和原硬门不变。
资源/版本/参考不符、窗口不足、硬数值门失败或超时即停，保留原始scontrol stdout/stderr/status与完整父终态记账，只提交一次。
B1024的299.139825仅作不同总量归一化参考；CPU2048、匹配Fortran基准、GPU重跑及核数网格非本轮前置。

旧v2 job**15805735**在`ibc12b08n35`以48CPU/0GPU运行109s后FAILED1:0；真实allocation通过，
scalar lower54.8616s后在签名helper失败，compile/B64未运行，见[v2终态](../../../outputs/research/global_throughput/cpu48-b64-execution-v2-20261012/CURRENT_RESULT.md)。
48核父一次费用情景CNY0.072666667/0.101733333，非核实账单，单次授权已消耗。
旧job15782454的FAILED2:0/1s与n35-only零提交证据原样保留，均不产生CPU48性能或新额度。
本次精确部署和唯一CPU sbatch已执行，scalar物理编译和阶段完成；B64在线程预算资源入场失败，未编译/计时。
CPU48科学/热性能未获得，完整父allocation按48核只计一次；scalar测量和未核实费率的费用情景见实际执行入口。
不改环境或科学合同、不跑GPU、不做完整预编译、不自动重试；本次仅在执行聊天交付，不联系或轮询主管线程。

以下保留已完成的生命周期、碳核心同步和失败定位证据，不作为CPU/GPU比较前置。

[保存证据与生命周期审计](../../../outputs/research/global_throughput/memory-lifecycle-audit-20261011/CURRENT_RESULT.md)
已接受：wrapper重建输入时整份core resident输入仍有持有者；约7.87GB是逻辑payload，
未保存的历史analysis、postcompile snapshot、失败repeat序号及真实峰值不能还原。
B2048×1没有前chunk，不能用跨chunk结果重叠解释其OOM。

本地[实现与验收](../../../outputs/research/global_throughput/memory-lifecycle-implementation-20261011/CURRENT_RESULT.md)
已完成，199项相关轻量测试、Ruff及diff检查通过，已提交为
`277aa35793e5b6dd3a1d436a32b4cbc97a305106`，作为碳核心同步前baseline。
该实现报告保留当时未提交状态，不回写历史验收证据；当前阶段见下述同步报告。未部署。
默认保留legacy交替顺序；新B1024-only `phased_resident_control` / `phased_release`
均先5次core后5次wrapper，仅后者在wrapper前清空core输入根引用。计时外及时保存
analysis、guard及collection/sample边界，原物理合同和内存guard未放宽。
真实GPU数值、allocator回收、显存变化与吞吐收益未验证。

[碳核心本地同步与验收](../../../outputs/research/global_throughput/carbon-core-sync-20261011/CURRENT_RESULT.md)
已完成：依次 `cherry-pick -x` main的NPP修复 `e317fa8` →本线
`c6a88afd41120dfd6434bdf5d442b72b58b34ccd`，以及leaf fraction修复 `2a3e303` →本线
`ef271cef8551f47f82cccf234949d021e958917e`。两次pick无冲突、来源独立保留。
新科学测试pin为`ef271cef`，普通记录提交的最终HEAD从Git及报告目录同步收据读取。

同fixture、各pin独立模块上下文的NPP适配器、post-NPP生产适配器和turnover_step，
在eager/JIT/vmapped-JIT三模式通过354/354个完整返回叶子记录：348个精确相同；
57个碳库/turnover及84个离散/text记录全部精确相同。仅六个leaf_frac记录有舍入差，
最大绝对差`3.3306690738754696e-16`、非零baseline最大相对差
`4.803236536610133e-16`，原`atol=1e-8, rtol=1e-10`通过，零坐标无变化。
相关新增回归37项通过，Ruff及diff检查通过；此前199项复用，不计作新核心数值验收。
范围为14个小型合成fixture、allocation之后的三个现有生产entry，CPU/x64、
JAX0.10.1；未运行外层全球driver、完整物理图、FULL64、两块连续窗口或GPU。
调用来源、实际命令、逐叶指标及图工具Transport closed的源码回退限制均在同步报告。

新核心被旧source_identity guard预期拒绝：`science source changed outside
certified execution-only cache repair`。这是版本隔离拒绝，与数值结果分开。
旧DIFFUCO-only execution certificate及helper原样保留，碳过程修复不归入旧execution-only
equivalence。专用新科学identity与显式旧冻结FULL64严格comparison admission在本轮独立准备，
旧goldens、历史pin/终态不改。新核心的完整全球/GPU数值验收仍待执行。

[GPU验证提案](../../../outputs/research/global_throughput/memory-lifecycle-implementation-20261011/GPU_VALIDATION_PROPOSAL.md)
规划同一新allocation内两个fresh B1024进程，均先五次core后五次wrapper，不能与历史交替样本混比。
本轮[B1024双臂本地准备](../../../outputs/research/global_throughput/gpu-lifecycle-preparation-20261011/CURRENT_RESULT.md)
已完成本地准备：独立新科学source guard、B1024-only runner、保存数组validator、
串行fresh-process coordinator、专用canonical launcher及实际Bash展开→真实coordinator dry-run。
common identity只排除strategy和wrapper根持有策略，保留sampling_order。新runner初始FULL64
有限数值超容差即停，不沿用旧capacity的exploratory continuation。实际初始/reset/置换/替换
数组及两个chunk的连续state handoff存证，封存后由CPU/NumPy独立重查2051 gates/760 leaves/2048 comparisons。
首臂exit0、seal、独立通过后才派第二臂；紧邻派发要求至少剩3600s；仅双臂通过发布同job比值。
本地检查实际结果及完整候选SHA见本轮报告/提交后收据，不把fake/Bash/旧参考格式检查当GPU验收。
source/runner/validator/observer有效43项（最新30项重跑，复用未变化13项），
真实旧FULL64格式与原生NamedTuple容器贯通均通过；coordinator/launcher最新14项通过，
本轮有效57项，不是单批重跑。最终code SHA/远端一致性见提交后候选回执。
Slurm日志固定到共享绝对 `runtime/logs/global_gpu_lifecycle_%j.txt`，提交前核查父目录存在且可写；
该检查须在脚本执行之前完成。Windows Bash路径/token映射与目标runtime验证的边界在本轮报告中保留。

用户已明确批准1 V10032GB+4 CPUs、120min、no-requeue、非exclusive，主/保守worst-case
CNY4.80/4.96，仅提交job15769234一次。固定执行SHA为`3b4ae35e695f85480ecb72461e4ba93f4534bbd8`，
后续记账HEAD不替代该pin。固定cwd为
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-lifecycle-v1`，
已按registry规范精确部署，干净且无overlay/alternate，旧树保留；目标Bash、库版本/绑定与源准入通过。
首臂已编译、postcompile原budget guard通过；初始返回的FULL64数组在lane45
（283.0-091.0）严格浮点容差失败，完整失败数组及hash保留；第二臂未派发。
没有warm样本/封存stage/双臂比值。整allocation估算主/保守CNY0.76/0.785333333，
账单未核实，唯一作业额度关闭，不复用余额。主管追加有界免费CPU/NumPy只读逐叶定位已在test03完成：
760leaf/2048lane全检，唯一超差为点283.0-091.0、Day8末
`diffuco_previous_step_state.rveget`、float64、inner[0,13]（PFT14）；
actual515.7648516411803/ref515.7648515535959，abs8.758433978073299e-08，
原bound6.157648515535959e-08、比1.4223666641536081。
lane45+64k（k0–31）共32次重复的坐标和数值逐字节一致；其余63真实点无失败。
无模型/JAX导入、模型/GPU重跑。主管已验读并接受该定位，关闭本轮科学执行，不再加检查或扩大溯因。
该数值字段差异不是旧ABI故障证据；新科学pin对旧冻结参考的差异尚无碳修复/生命周期因果归属，
不自动修科学代码/改容差/重跑，保留原pin与旧证据。
新核心FULL64、allocator回收、显存/吞吐收益均未验收，详见执行报告和
[保存数组解释](../../../outputs/research/global_throughput/gpu-lifecycle-execution-20261011/SAVED_ARRAY_EXPLANATION.md)。
旧fixed2048 launcher会派B2048，不能复用；旧额度保持关闭，失败不自动重试或扩大batch。

中间批量、真实669点+尾批、独立集合成员、长窗口/年和匹配Fortran基准仍为后续另议。
完成相应验证前，不作生产/50年/10×/普遍科学效率结论。

旧DIFFUCO `_diffuco_pft_column_mask` 的 `lru_cache` 曾缓存 DynamicJaxprTracer，
导致顺序shape公共536/内部673输入不匹配；已改为只读NumPy bool，mask/formula未变。
不要倒退修复或把旧ABI错误当显存极限。按需读
[因果修复](../../../outputs/research/global_throughput/gpu-metadata-repair-20261008/CAUSAL_REPAIR.md)
和[执行审计](EXECUTION_AUDIT.md)，不要把其中较早“下一步”当当前授权。

## 5. 小范围代码与证据地图

| 路径 | 用途 |
| --- | --- |
| [global_batch.py](../../../jax_orchidee/driver/global_batch.py) | 批处理完整物理transition入口 |
| [benchmark_global_gpu_capacity.py](../../../scripts/dev/benchmark_global_gpu_capacity.py) | capacity合同、计时及结果存证入口 |
| [生命周期科学身份](../../../scripts/dev/global_lifecycle_science.py)、[专用runner](../../../scripts/dev/benchmark_global_gpu_lifecycle.py) | 新科学严格准入、B1024双臂实际数组存证 |
| [生命周期validator](../../../scripts/dev/validate_global_gpu_lifecycle.py) | CPU/NumPy独立保存数组重查 |
| [生命周期coordinator](../../../scripts/hpc/global_gpu_lifecycle_coordinator.py)、[专用launcher](../../../scripts/hpc/slurm_global_gpu_lifecycle.sh) | 两strategy全新进程顺序调度，未来执行待批准 |
| [2048 coordinator](../../../scripts/hpc/global_gpu_capacity_2048_coordinator.py) | 历史两shape独立进程调度 |
| [2048 launcher](../../../scripts/hpc/slurm_global_gpu_capacity_2048.sh) | 已完成付费job的launcher，不是重提指令 |
| [SERVER_BENCHMARK.md](SERVER_BENCHMARK.md)、[PRECISION_BOUNDARY.md](PRECISION_BOUNDARY.md) | 运行合同与native dtype边界 |

最新terminal summary/artifact index/logs、完整checkpoint和collector在两份CURRENT_RESULT
链接的同目录内；881MB/440MB pickle不进Git。历史报告与原始失败证据继续保留。

## 6. 服务器、固定版本和操作边界

服务器前缀 `P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`。不可变执行目录：

| 已完成运行 | P下路径 | 执行SHA |
| --- | --- | --- |
| fixed1024 | `runtime/worktrees/global-throughput-gpu-fixed1024-v1` | `bd9e4c04af68d77e79b984bfc8aa064977f893d3` |
| fixed2048 | `runtime/worktrees/global-throughput-gpu-fixed2048-v1` | `8aa2e05aa24b65dc4d30a27d4ee857bba6312221` |

冻结输入为 `P/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl`；
服务器forcing在 `P/runtime/data`（不是root/data）。本地共享资产在`E:/ORCHIDEE-MAN-JAX/data`。
复用 `P/.venvs/orcjax_cpu` / `P/.venvs/orcjax_gpu` 的canonical JAX0.4.38部署环境；
本地ORCJAX轻量检查不替代真实部署runtime。

服务器统一入口 `P/runtime/deployment/README.md`、`registry.json`、bare `repository.git`；
本机[registry证据](../../../../../../outputs/maintenance/20261010/server/registry.json)与
[维护报告](../../../../../../outputs/maintenance/20261010/server/server-report.md)记录2026-10-10快照。
当时global镜像pin为`c537f59f33568b6e5960254fe55fb307f82ca503`；本次本地提交推进本地/GitHub，
**不更新服务器镜像或执行checkout**。当前开发SHA用`git rev-parse HEAD`读取，不能替代上述执行pin。
服务器旧Git无worktree/-C支持，按bare镜像与精确SHA bundle同步，不新造一树一版本。

遵守[共享部署入口](../../../../../../docs/deployment-explore1000.md)与
[工作树规范](../../WORKSPACE_MANAGEMENT.md)。global没有同名deployment文档，使用本线
SERVER_BENCHMARK及当前命名合同。只写jcwu树，登录节点不计算；连接身份/节点负载看当前状态，
不套旧快照。10/10账户文件数约9.18m/10m只是历史账户快照，非项目现存quota证明。
新GPU付费作业须具体批准resources、walltime、command、cwd和worst-case cost；已授权free
准备或普通可逆小修按原scope推进。本次交接不授权新实验，不恢复旧job/额度/自动任务。
