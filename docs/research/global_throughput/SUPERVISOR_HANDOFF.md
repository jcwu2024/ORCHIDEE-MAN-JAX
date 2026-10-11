# 全球 JAX 运行效率：新项目主管交接

更新：2026-10-11。先读本页、[碳核心本地同步结果](../../../outputs/research/global_throughput/carbon-core-sync-20261011/CURRENT_RESULT.md)，再按需读生命周期实现及两份GPU终态 CURRENT_RESULT；历史仅按问题查阅。
此交接保存停止点和研究问题，**不恢复任何 GPU 诊断、付费作业或旧额度**。
可复制[新项目开场指令](NEW_PROJECT_PROMPT.md)。

## 1. 角色、目录与开场检查

接替主管聊天【研究 JAX 全球运行效率】
`01a1075c-aa28-7c11-a242-9ca5128673bc`；原执行聊天【用Sol接手研究JAX全球运行效率】
`01a1077d-30a0-7792-b7d1-a67660a7b889` 已结束 B2048 OOM 并交接。
原聊天 cwd 仍为根 main；添加项目不自动修改已有聊天 cwd。
把现成 global-throughput 目录添加为项目，新聊天选择本地/Local，直接使用已有工作树；
无需重新clone或从main派生新分支。若检查不符，先定位到指定工作树，不切换他人的树。
本项目固定 cwd 为 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`，
分支 `research/global-throughput`。日尺度神经算子在另一个工作树，非本线任务。

新主管先核对 top-level、branch 和实际 module 路径，不在根 main 或他人工作树编辑：

```powershell
git rev-parse --show-toplevel
git branch --show-current
conda run -n ORCJAX python -c "import jax_orchidee; print(jax_orchidee.__file__)"
```

用户指定的协作结构是**两个独立的 Codex 聊天**：主管线程主模型使用 GPT-6 Astra，
独立执行线程主模型使用 GPT-6.1 Sol，各自 effort 沿用用户设置；以后模型调整以用户选择为准。
主管负责研究方向、任务分配、用户沟通和实际证据验收；Sol 执行线程负责具体研发、
修复、测试、服务器执行、文档和结果交付，复用已完成的证据。
这与日尺度线【日尺度神经算子研发】指挥【用Sol接手日尺度神经算子研发】的组织方式相同；
本线对应本页所列主管与【用Sol接手研究JAX全球运行效率】。

`subagent-steward` 已安装，是各线程内部安排子代理的辅助规则；上述独立主管/执行关系
来自用户的明确选择。不得把它改写成主管只派临时 Sol 子代理，也不需要重装或重定义技能。
后续获准开始研究时，主管向用户指定的独立执行线程下达任务、读取进展、验收实际文件。
原执行线程的默认 cwd 仍是根 main，接续前必须确认其命令实际在本全球研究工作树执行；
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

## 3. 最新结果：1024最佳已测，2048执行OOM

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
wrapper为7.052818 point-days/s，仍有两项cross-GPU严格容差失败，CPU48未测。
同价48核理想线性外推338.5353、对旧GPU1024的302.4092，打平需89.33%并行效率，
只是费用/缩放假设，不能宣布CPU/GPU最终赢家。GPU甜点优先，更多CPU测试非前置。

## 4. 当前阶段与下一步供主管审阅

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

当前新核心被旧source_identity guard预期拒绝：`science source changed outside
certified execution-only cache repair`。这是版本隔离拒绝，与数值结果分开。
旧DIFFUCO-only execution certificate及helper原样保留，碳过程修复不归入旧execution-only
equivalence。后续须准备独立新科学核心identity及对旧冻结FULL64的显式严格comparison
admission；旧goldens、历史pin/终态不改。新完整全球runner与GPU准入尚未完成。

[GPU验证提案](../../../outputs/research/global_throughput/memory-lifecycle-implementation-20261011/GPU_VALIDATION_PROPOSAL.md)
规划同一新allocation内两个fresh B1024进程，统一phased顺序，不能与历史交替样本混比。
拟1 V10032GB+4 CPUs、120min，主/保守worst-case CNY4.80/4.96，未获新执行授权。
本地核心同步已验收；新科学来源准入、未来执行SHA、专用两组launcher、
strategy-aware saved-array验证及目标Bash dry-run仍待准备，提案未就绪；
旧fixed2048 launcher会派B2048，不能直接复用，旧额度保持关闭。

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
| [2048 coordinator](../../../scripts/hpc/global_gpu_capacity_2048_coordinator.py) | 两shape独立进程顺序调度 |
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
