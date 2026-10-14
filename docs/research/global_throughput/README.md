# 全球物理模型吞吐研究

当前阶段（2026-10-14）：**CPU48 B64 v5 候选已固定、部署并通过一次有界免费参考准备检查；待主管审查，付费执行未授权**。
先读[准备结果](../../../outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014/CURRENT_RESULT.md)、
[机器可读摘要](../../../outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014/preparation-result.json)、
[执行 pin](../../../outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014/execution-code-pin.json)
和[付费提案](../../../outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014/PAID_PROPOSAL.md)。完整免费检查报告另保留于本地与服务器：
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014/free-check/15857559/report.json`，SHA256 `d70dd295df32227c6050db890c7732a1e54a6fe70bf7d5ea631a07b6a772247f`。
执行 commit `0100c8b09be5137b822173e0a0be66cb64939727` / `codex/global-cpu48-b64-v5`，基于 `60dccbdd...`；
只含三个已接受的生产 harness 文件和三个测试文件，104 个物理文件及 science identity 不变。
v5 新 immutable checkout 部署成功，clean 且无 alternates。登录节点部署收据记录 bundle SHA 与 `git bundle verify` 通过；实际目标 allocation 分别检查 pin、scalar admission、提案和六个 harness 哈希。
免费 job15857559 在 cntest/ibtest02 实际1 CPU、affinity `[0]`，300秒上限内 COMPLETED 0:0、父作业65秒；目标 JAX/jaxlib 0.4.38、x64 初始化，
64 个真实点的 FULL64 输入及 GPU/scalar 参考（各760叶）加载通过，四项 candidate/stage identity 相同。
实际脚本耗时16.98秒；仅运行共享 `prepare_references` 后退出，没有创建物理 runner、执行 lower/compile/run 或产生 benchmark 阶段/门/计时。
免费检查还验证 production launcher/coordinator/runner dry-run 的参数展开及 job-ID 输出路径。dry-run 只检查计划路径替换；真实启动时的目录不存在门保留但未执行。计划48 CPU参数不是48核资源准入。
本地 compact artifact review 为补充审阅记录；tracked preparation result 与 CURRENT_RESULT 是主要交接入口。
付费提案仅为 cnall/1 node/1 task/48 CPU/0 GPU/3600s、nonexclusive、no-requeue、无新增内存或 placement 限制，
费率情景 CNY2.40/3.36 未核实；必须等待主管对该精确候选和命令的审批。没有提交付费作业，旧额度均关闭。
本结果不代表48核资源入场、物理计算、科学接受或吞吐。原 scalar 61/64 GPU 门通过、3 个有限超差点/4处比较、reset 仅封存证明及 scientific acceptance false 均保留。

以下为 v4 历史付费终态；当时未保存失败比较双方，具体身份根因由上述后续免费重放确认。
**CPU48 v4 唯一付费 job15853683 已 FAILED 1:0/50s，本次尝试关闭**。
先读[v4 实际终态](../../../outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/CURRENT_RESULT.md)、
[单次授权](../../../outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/AUTHORIZATION.md)
及[固定 pin](../../../outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/execution-code-pin.json)。
执行 SHA `60dccbddde10be4295aab7e7fe55fa7864d3c3f3` / ref `codex/global-cpu48-b64-v4`；
104 个物理 blob 与 v3 基底相同，研究树 carbon 修复保留。原命令只提交一次，cnall/1 node/1 task/48 CPU/0 GPU/3600s，
非 exclusive/no-requeue/无新增 mem 或节点限制；实际节点 `ibc11b09n36`。
新 production checkout 的免费预检 job15853600 通过实际 candidate identity/seal、15项 harness hash 和原路径 dry-run；
真实付费 child 的环境、48 个不同物理核及内存准入也通过，随后 JAX0.4.38 CPU/x64 初始化。
首个实质故障为 `load_reference` 拒绝 scalar admission 的 candidate provenance 与当前 execution identity 不一致。
失败比较双方未完整序列化，未归属具体差异原因；预检通过不等于实际 runner 的参考加载通过。
报告 phases/gates 为空、timing 为空：没有 B64 lower/compile、保存候选数组、五次热样本或 CPU48/GPU 速度/热成本比。
免费 job15853855 的冻结 collector 仅接纳失败记账；新父 50s×48=0.666667 核时，未核实费用 CNY0.033333333/0.046666667。
旧 scalar job15820467 的 2197s×48=29.293333 核时、CNY1.464666667/2.050533333 单列且只计一次。
原 scalar 61/64 GPU 门通过、3 点有限超差、reset 仅原封存证明及科学 false 保留；本次无 scalar 重生成。
同量 GPU B64 原始样本见[比较证据](../../../outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/GPU_COMPARATOR_EVIDENCE.md)，
B1024 仍仅为不同总量的工程参考。原 pin、报告和日志保留；不重试、延长、取消或自行修复，不恢复旧额度。
执行聊天完成交付后停止工作，等待派工，不联系/轮询主管或创建自动任务。

以下保留 v3 终态和已完成科学证据；当前操作状态以顶部有界修复结果为准。

新项目用户/新主管先读[主管交接](SUPERVISOR_HANDOFF.md)、
[CPU48 v3新单次授权](../../../outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/AUTHORIZATION.md)、
[CPU48 v3准备结果](../../../outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/CURRENT_RESULT.md)、
[新的单次运行提案](../../../outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/PAID_PROPOSAL.md)
及[CPU/GPU现有证据比较](../../../outputs/research/global_throughput/cpu-gpu-b1024-comparison-20261011/CURRENT_RESULT.md)。
用户于2026-10-13授权主管审查批准，主管已接受实际产物并批准该精确版本的一次CPU48 v3作业及必要部署、预检、采集。
本次job **15820467**已终态FAILED1:0/2197s：scalar FULL64阶段完成，B64在启动资源入场时报告OMP线程预算不符，
历史日志未记录实际child OMP值；未进入B64 lower/compile或热计时，没有CPU48吞吐或CPU48/GPU比值。单次授权已消耗，不恢复旧v1/v2额度、不重提。实际证据见
[v3执行入口](../../../outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/CURRENT_RESULT.md)。
scalar原始记录为896 point-days、各5个样本：core中位127.074984s（7.050955 point-days/s），wrapper中位121.382450s（7.381627 point-days/s）。
64个GPU参考门中61通过、3个仅有限值容差失败，冻结reset重复通过；这些是探索记录，collector只接纳失败记账，未独立接纳scalar数组或吞吐。
父allocation只计一次：2197s×48=29.293333核时，未核实费率情景CNY1.464667/2.050533。scalar进程HWM为19.94GiB，不能用checkpoint当时5.586GB的RSS替代。
用户已选接受的B1024作为当前单卡工程选择，主问题是CPU/GPU速度与单位成本；
CPU主候选为单进程native FULL64 B64 batch/vmap、48个不同物理核，CPU4不是强制前置。

主管已接受研究提交`d33fa65bb263b1564ca98f72b81a69e93aabc33f`的有界AOT修复：
首个实际失败叶子`.values_by_component[4][67]`为Python float，共享helper改用JAX `shaped_abstractify`；
tree/shape/dtype/weak_type构造审计及native call行为保留。Linux JAX0.4.38/x64的两组64点、每点536叶元数据
和实际scalar/B2连续两块替身接口证据复用，不重复编译。该接受仅限元数据/接口，见
[修复报告](../../../outputs/research/global_throughput/cpu-aot-signature-repair-20261012/REPAIR_REPORT.md)。

新执行pin为`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`，ref `codex/global-cpu48-b64-v3`，
直接基于v2 `b1b5cf0835da298bddb388e61d754feeb6c14cf2`，从研究检查点`8a860c1…`只移入审核harness/tests。
全部104个物理文件、整个`jax_orchidee`子树和旧science证书`11b6…`与v2一致；研究树两项碳修复保留。
AOT文件精确SHA为`bc8ce1d62f6947228c8c8438344987345bf18095f307c90a2f45dc4655c6891e`。
[pin清单](../../../outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/execution-code-pin.json)
绑定完整SHA、独立v3代码/输出路径、patch及已验证bundle（依赖v2 base）。
ORCJAX/JAX0.10.1/x64本地准备测试15通过；新执行版本source_identity通过，原10项容量protocol/launcher失败按具体ID全部通过。
另10项AOT/source-certificate筛选通过（50未选）；[原始回归记录](../../../outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/regression-recheck-20261013/REGRESSION_RECEIPT.md)
区分两组选例。这不是整容量模块60项重跑或目标JAX0.4.38 runtime接受；原研究50通过/10失败记录不回写，不放宽guard或回退碳修复。

批准资源为cnall/1node/1task/48CPU/0GPU/5400s、非exclusive/no-requeue/无新增`--mem`；
全父allocation含scalar都按48核计费，未核实费率的worst-case为CNY3.60/5.04。
64 unique FULL64点、Days2–8和9–15、48半小时步/天、896 point-days，与已接受GPU B64作同量比较；
每阶段5 core+5 wrapper，scalar cap2700s，候选须至少剩1800s。760完整native outputs/masks、x64、
原生dtype、`atol=1e-8/rtol=1e-10`及原硬门保持。健康匹配节点可正常排队，不锁n35；新提案清除继承的提交限制。
B1024的299.139825只作不同工作量归一化参考。
实际scalar物理编译和完整阶段已完成；B64因线程预算入场失败，CPU48科学gate、热吞吐/RSS及同量CPU/GPU比仍未获得。
scalar原始证据按执行入口单列，不能当作48核batch或科学接受；后续修复及新作业等待新的派工。

旧v2 job**15805735**终态FAILED1:0/109s：真实allocation通过，scalar lower54.8616s后在上述签名helper失败；
compile和B64未运行。其单次付费授权已消耗，48核父一次费用情景CNY0.072666667/0.101733333，见
[v2终态](../../../outputs/research/global_throughput/cpu48-b64-execution-v2-20261012/CURRENT_RESULT.md)。
更早job15782454的启动失败及n35-only零提交证据原样保留，旧额度均不恢复。
旧GPU job15769234保存数组定位已接受：仅点283.0-091.0、Day8末rveget/PFT14超界，32重复一致；
原因未归属，本轮科学执行关闭，见[B1024双臂终态](../../../outputs/research/global_throughput/gpu-lifecycle-execution-20261011/CURRENT_RESULT.md)。
1280/1536探索、显存优化及小容差溯因不是本轮前置；B1024不称已证明全局最优。
下面保留历史实验上下文，旧“下一步”及额度不自动恢复为新授权。

更新：2026-10-13。分支：`research/global-throughput`；起点：`main@5910ba9`。
本文件是本分支的研究入口和当前工作单；[EXECUTION_AUDIT.md](EXECUTION_AUDIT.md)
记录具体实现边界。日尺度神经算子研究继续由原分支管理。

精度核查补充：冻结核心及旧四点输入原有`float32`地理payload；其中纬度进入的
`season_midwinter_solad`日照子计算保持`float32`，随后写入的`gdd_init_date`状态为
`float64`。本文历史“FP64”标签表示开启JAX x64及原运行dtype，不能作为所有物理
表达式均为FP64的证明。主管已澄清：本轮开启JAX x64、保留基线原生dtype，
已知地理payload和solar子计算含FP32；先前逐表达式全FP64措辞不属于用户要求，
此次没有精度变更。确切原生dtype输入gate已通过35项轻量测试及local16/local64生产reader；
Linux首次再生成出现16个FP64低位差异，现直接运输冻结参数。首次运输后的首点原始数值digest
通过，但旧remote4与NEW60的`sechiba_finalize_state`字段排列不同，该次合批入场严格拒绝。
后续只按字段名一一映射NEW60的state输入到未改动remote4布局，保存原始及绑定digest，
逐字段核对dtype/shape/bytes；其余参数组保持原冻结输入，现已按原预算完成Linux入场及两次GPU诊断。
证据与执行合同见[PRECISION_BOUNDARY.md](PRECISION_BOUNDARY.md)。

## 目标与范围

在保持原有物理过程、48 个半小时时步、日 STOMATE 更新、精度和验证标准的前提下，
降低论文 669 个 PFT14 网格、1961–2010 年模拟的总耗时与计算成本。
这里的“全球”首先指现有 669 个独立论文网格；不代表全陆地 PFT 或开启横向耦合的全球模型。
若任务增加 routing 或网格间交换，必须重新审计独立性和通信需求。

优先研究正向模拟。可微性的后续用途是参数校准、敏感性分析与数据同化；
这些工作需另行定义可微参数、观测损失和长期梯度验证，不能由“使用 JAX”自动推定完成。

## 初步判断

有值得检验的结构性提速机会，核心是**同时计算一批网格**，而非只替换运行设备。
JAX 编译、数组批处理和设备分片支持这一实现；它们与自动微分是不同能力。
Fortran 本身也支持向量化、MPI/OpenMP 乃至 GPU，比较对象应明确为当前论文版本和当前运行方式。

建议计算结构：

```text
网格独立状态 + 各自 forcing/参数
  -> 按兼容的静态配置分组，装成一批
  -> 同一批内并行计算网格；每个网格按原顺序推进时间
  -> 块边界汇总输出、检查状态；必要时保存恢复点
  -> 单设备收益成立后，再把不同批分配到不同 CPU/GPU
```

50 年的状态递推有因果依赖，不能直接把每年作为互不相关任务并行。
`lax.scan` 能减少调度与数据传输，却不消除这种依赖。网格、独立情景和参数集合才是
最自然的并行轴。CPU SIMD/批处理仍是候选；GPU 不保证获胜。
物理模型应按实际 FP64 算力、访存和小 kernel 调度成本选择设备，
不能借用神经网络低精度 Tensor Core 的峰值来预测速度。

## 现有证据及限制

| 证据 | 已知结果 | 不能推导的结论 |
| --- | --- | --- |
| 主分支 README 与当前入口 | 已有七天完整块、编译复用；七个新网格完成 50 年验证，最大 2010 AGB/BGB/GPP/NPP 相对误差 8.5e-5；完整 669 点等价验收待完成 | 不能称全分支、全 PFT 或全部 669 点已经通过 |
| 主分支旧 SECHIBA 编译 A/B [报告](../../source_audits/compiled_fast_path_production_readiness.md) | 三点一年 compiled 79.33–94.16 s，strict JAX 237.84–268.63 s | 2.65–3.00 倍是相对 strict JAX；不是相对 Fortran，也不是当前七天块的新实测 |
| 更晚的完整多日块验收原始记录，2026-07-19 | 本地 Windows CPU；365 日热运行 31.1364 s，冷编译加首轮 158.13 s；冷启动/全年/restart 的有界状态门槛通过 | 缺精确 commit、CPU 型号和 JAX 版本；不是本次 main 实测，也不是同机 Fortran 对比 |
| 历史物理 Teacher GPU capture，2026-07-21 | V100 / JAX 0.4.38 / FP64；15 日热运行 4.840 s，29 日 7.631 s；冷运行约 903/923 s | 带额外训练数据采集；不是当前 main 纯正向基线，短窗热速率不是 50 年速度 |
| 历史 Teacher v6 CPU capture，2026-08-06 | 五个单 CPU 常驻 worker；热年约 82 s/点年，首次 capture 约 2058 s；换网格未增加可执行缓存项 | 历史研究提交、额外输出与不同环境；不能与上述 GPU 行直接作公平 A/B |
| 立项轻量检查 | 新 worktree 导入成功；ORCJAX Python 3.11.15 / JAX 0.10.1，仅显示 CPU；公开 `run --help` 成功 | 仅为立项环境检查；后续本地完整短窗实测见下 |

历史报告可在同仓库提交 `b0080d3215bce7cae349d1bcf76b37450ac94ceb` 中恢复：

```bash
git show b0080d3:docs/research/daily_coarse_graining/teacher_capture_gpu_benchmark_20260721.md
git show b0080d3:docs/research/daily_coarse_graining/teacher_v6_reuse_benchmark_20260806.md
```

GPU 两个原始 JSON 本次已核读：原工作区
`outputs/performance/daily_coarse_graining/compiled_training_capture_in_process_gpu_15d_optimized.json`
与 `compiled_training_capture_in_process_gpu_29d_block28.json`；记录冷/热时间、350 个末态叶子、
重复运行零差异和缓存计数。CPU 数据本次依据历史报告，未重新执行该任务。
其 31.2 GiB/worker 是包含编译的进程峰值，不能直接当作稳态状态内存。

更晚的主模型证据位于原工作区
`outputs/performance/compiled_multiday_block_acceptance_20260719.json` 与
`complete_day_landpoint_reuse_20260719.json`。后者是三点两天块的复用与短窗一致性，
不是三个点都完成七天块全年验收。不要用旧性能文档覆盖这些更晚结果；具体范围见执行审计。

用户提供的 Fortran 粗基线为 2 分钟/点年、近 2 小时/点/50 年。
按 669 点估算约 1115–1338 单核小时（每点占一核的假设）；这是原始经验值的算术换算，
不是同硬件复测或 JAX 节省量。2 分钟乘 50 等于 100 分钟，两个口述数字的差距须在
基线测量中由启动、IO、年份差异等实际记录解释。

## 当前工作单

本轮从CPU1 job15591356与最新接受的GPU B1024×2 job15671732原始记录做本地比较。
CPU wrapper为7.052818167587637 point-days/s，GPU为299.1398250995909 lane-days/s，
归一化比例42.414226只作量级参考：CPU串行的16真实点是GPU FULL64的前缀，GPU为64点×32重复，尚非严格同工作量对照。
当前比较不用旧302.409157替代最新GPU速率；原生dtype、全部输出/masks和科学接受边界保留。

| 热wrapper费用情景，CNY/百万点日或lane-days | CPU1 | GPU B1024×2（含4辅助CPU） |
| --- | ---: | ---: |
| CPU 0.05元/核小时，GPU 2.20元/卡小时 | 1.969268 | 2.228612 |
| CPU 0.07元/核小时，GPU 2.20元/卡小时 | 2.756975 | 2.302899 |

单价情景使初步费用方向反转，账单未核实；冷编译、启动及全作业费用另列，不把失败B2048费用归给B1024。
主情景同价48CPU的打平效率为88.362971%，只是线性假设；按compile-end RSS作48份常驻scalar的朴素外推约229GiB，
超过192GB CPU节点。CPU4/48无实际性能结果，不提48独立进程方案。
已准备的对照合同为单进程CPU48/B64，与已接受GPU B64（job15579869、FULL64 unique64、896 point-days）
作同工作量对照；GPU B64与canonical FULL64保存metadata的64点序、完整input digests及science一致。
同一90min/48CPU父allocation先建立完整scalar参考，再测B64；每阶段5 core+5 wrapper，
scalar最多2700s，候选需至少1800s剩余窗口。整作业含scalar阶段均按48核计费，终态父sacct只计一次。
B1024是当前GPU工程选择，其大批量吞吐与B64仅作注明不同collection总量的归一化参考；
完整2048负载适配只在需要该级别端到端对照时再选。详见上方比较和CPU计划。
上方v3精确提案的单次作业已终态关闭；旧比较的线性48核数值仍仅是历史算术假设，CPU4与48独立进程方案不作为前置。
旧作业预算均关闭，本次固定科学版本、原生dtype、输出和容差不变。

以下为已保存的CPU1和GPU阶段证据，历史工作安排按其当时范围理解。

CPU1 B16 resident-scalar终态已经完成，状态为**性能已测；跨后端科学一致性未接受**。作业15591356在Explore1000节点n04以1 CPU、0 GPU运行1176秒，完成16个固定真实点、两块连续7日、每天48步的224 point-days，并保存五次同步core和wrapper样本。完整终态证据见[CPU结果](../../../outputs/research/global_throughput/cpu-performance-results-15591356/CURRENT_RESULT.md)、[原始测量](../../../outputs/research/global_throughput/cpu-performance-results-15591356/terminal-measurements.json)、[费用情景](../../../outputs/research/global_throughput/cpu-performance-results-15591356/server-cost-results.json)及[CPU-internal reader admission](../../../outputs/research/global_throughput/cpu-performance-results-15591356/cpu-internal-reader-admission.json)。

| 同工作量 B16，224 point-days | 分配CPU/GPU | Core中位数 [范围] | Wrapper中位数 [范围] | 热费用主情景 core / wrapper | GPU/CPU吞吐比 |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU1 resident scalar | 1 CPU / 0 GPU | 31.458503002 s [30.855936289, 31.737742664] | 31.760353759 s [30.915467992, 32.648889609] | CNY 0.000436924 / 0.000441116 | — |
| 已接受GPU B16 | 4 CPU / 1 GPU | 4.149942968 s [4.142994318, 4.152027067] | 4.364678632 s [4.359835308, 4.370462861] | CNY 0.002766629 / 0.002909786 | core 7.580466345×；wrapper 7.276676345× |

CPU记录17项固定gate中15项通过、2项有限tolerance差异失败：299.0-115.0的bound ratio为1.2284882227，285.0-087.0为1.7088155265。原生dtype与`atol=1e-8`、`rtol=1e-10`未变；冻结输入重复运行零差。collector通过checkpoint SHA/contract/五样本检查，完整探索性计时可发表，但accepted CPU stage count为0。费率情景为CPU CNY0.05/allocated-core-hour（保守0.07）、GPU CNY2.20/card-hour，GPU的4个辅助CPU按核时另加，因此主情景每小时GPU作业为CNY2.40（相当48个主情景CPU核时），保守情景CNY2.48。按主情景，当前B16热CPU费用低于GPU：GPU/CPU热费用比为core 6.332064、wrapper 6.596418；GPU仍分别快7.580466和7.276676倍。这里可说B16热样本的CPU1较省钱、GPU较快；不能外推年度/669点/50年或宣布全规模最终赢家，也不代表科学一致性。

作业全窗口1176秒的主/保守费用情景为CNY 0.016333333/0.022866667，父作业只计一次，含启动、编译、gate、重复和IO；不是按224 point-days拆分，也不是核实账单。lower 56.317166498秒、compile 688.378443662秒、首次调用32.158282455秒、checkpoint序列化0.005034860秒及原子写入0.025802774秒分别记录。无完整stage startup/validation accounting，不估算K复用收益。

CPU-internal reader的首次真实检查曾在record字段一致性层拒绝：checkpoint的`record.execution_origin.environment.loadavg`为tuple `(0.14, 0.06, 0.06)`，manifest JSON为相同值的list。修复只允许该字段tuple/list等价，要求长度3、finite nonnegative且每个元素type/value严格相等；其他证据字段与科学/输入/gate/output校验不放宽。使用分开的patched reader namespace对未改动CPU1 B16 snapshot重新实测，`load_reference`及全部`_validate_cpu_internal`后续检查通过：760 leaves、16 lanes、两块完整边界、5+5样本；CPU checkpoint/manifest/report SHA均未变。该角色仍是`cpu_internal_comparison`，不升级为accepted CPU或科学parity通过，也没有运行CPU4。两项原GPU有限tolerance失败仍失败。见[精确record差异](../../../outputs/research/global_throughput/cpu-reader-repair-20261006/record-differences.json)、[实际reader验证](../../../outputs/research/global_throughput/cpu-reader-repair-20261006/actual-reader-validation.json)和[修复报告](../../../outputs/research/global_throughput/cpu-reader-repair-20261006/REPAIR_REPORT.md)。vmap lane数不等于CPU核数。

### 当前结果：因果修复已验证；B512作业15657822完成并通过严格采集

v3作业15643868已终态FAILED，B256完整阶段通过并密封。新canonical JAX0.4.38顺序lower诊断自然复现了根因：DIFFUCO的Python缓存保留了前一trace的PFT布尔mask，后续137个常量被整体提升为额外输入。缓存改为只读NumPy布尔mask后，B256与B512均为536/536、保留417输入，先前stage metadata未变；原生五项输出与独立NumPy插入结果完全相同。物理公式、dtype、mask、输出和容差未改。修复后的有界B512 GPU阶段现已完成并由严格collector准入；它仍只是指定工作量、环境和时间窗下的exploratory capacity evidence。

本地103项测试、改动文件Ruff/diff检查通过；含部署/接口的canonical保守累计367.0582004秒，这部分准备/接口检查不含真实模型compile/run。实际全源码hash与基线science身份分别记录；只允许精确helper字节变换及两个审核过的完整源码字节表示，不做一般AST或换行归一化。当前science identity已准入实际sealed B128/B256与FULL64；这属于保存基线准入，和后续GPU运行证据分开。新immutable checkout已部署accepted SHA `ad6c0e1faa7e22b590627844e9cb0ad492e9deb3`，前后clean；实际CLI/source/Bash/spool九项检查均通过，direct与spool命令一致，这些接口检查未导入JAX或初始化GPU。之后job15657822已完成修复后B512 GPU compile/run并由严格collector准入；完整Linux源码hash、active certificate及[部署接口收据](../../../outputs/research/global_throughput/gpu-metadata-repair-20261008/deployment-interface-receipt.json)见当前结果和终态证据。

相同7168 lane-days，B256 core中位27.024156772s、prepared-wrapper中位30.468557430s（235.258923 lane-days/s），主/保守wrapper费用CNY2.833757201/2.928215775每百万lane-days；B128 wrapper33.838030240s。旧B128 lazy dispatch与B256 Python-bound路径不同，不能全部归因batch size。当前目标仍为未来点位×成员的单卡wrapper费用/吞吐曲线；669点需要尾批/填充、初始化及IO证据，不是容量上限；年度、Fortran、50年、10×和全局科学一致性未验收。

作业15657822从gln01经192.168.11.2只提交一次，终态`COMPLETED 0:0`，父作业1416s，实际节点ibc13b03n02，4个Slurm CPU、1张V100、1node/1task、2700s硬上限、no-requeue；集群本地起止时间为2026-10-08 20:57:08–21:20:44（UTC12:57:08–13:20:44），初始PENDING/Priority，排队不计费。严格collector在test02用已有`gpu-sweetspot-preparation-20261006/target-input-admission.json`准入1个stage、扣留0个，exit0，saved-array CPU wall80.06s；collection没有编译/运行模型或初始化GPU。GPU global ID4，UUID `GPU-949cb714-8c53-353c-1f6a-1832c629230e`，Tesla V100-SXM2-32GB，driver525.85.12；启动选定GPU进程列表为空，affinity54,55,110,111（4个计费CPU映射2个物理核），start load26.25,27.24,23.0。提交时12:48UTC快照为CPU112/112、load22.74、1/8GPU已分配且无空闲eligible节点；不能据此推断执行时竞争。

严格stage通过515个gate、760个原生输出leaf×512 lanes及512项FULL64参考比较。固定有序工作量是64个真实点×8，连续Days2–8再9–15、每天48步，共7168 lane-days，不是512个唯一点。B512 core五次计时中位22.681461047s（范围22.679539528–22.705473831），316.029024 lane-days/s；prepared-wrapper中位26.124860071s（范围26.080703341–26.167727122），274.374675 lane-days/s。未核实费率下，warm median成本为core CNY0.015120974/0.015625006、wrapper CNY0.017416573/0.017997126（主/保守）；wrapper每百万lane-days CNY2.429767492/2.510759741。相对B256 wrapper30.468557430s，观测吞吐1.166266818倍、warm成本低14.256327591%。这是不同job/code且包含helper修复后的同工作负载比较，不能单独归因batch size；B128 lazy dispatch也不同。

host stack/transfer为2.581256084/1.334585367s，lower98.739008831s，compile892.739054900s，首次完整B512调用22.867870517s，checkpoint序列化0.380422166s、atomic write0.213936657s。完整phase setup/validation未记录，因此没有K amortization估算。父作业1416s对应估算CNY0.944主口径/CNY0.975466667保守口径，在批准的CNY1.80/CNY1.86上限内；账单未核实，child rows未重复计费。compiler临时内存2288514256 bytes不是VRAM峰值；host process峰值19496943616 bytes，GPU前后快照不能作为峰值证据。详见[当前结果](../../../outputs/research/global_throughput/gpu-metadata-repair-20261008/CURRENT_RESULT.md)、[付费方案与收据](../../../outputs/research/global_throughput/gpu-metadata-repair-20261008/B512_PAID_PROPOSAL.md)、[因果证据](../../../outputs/research/global_throughput/gpu-metadata-repair-20261008/CAUSAL_REPAIR.md)及[协议和终态证据](../../../outputs/research/global_throughput/gpu-metadata-repair-20261008/B512_PROTOCOL.md)。这不是CPU比值、Fortran、年度/50年、669点、10×、训练/AD或一般科学验收；未发生retry、cancel、requeue或automation。

### 当前结果：fixed2048 B1024严格保留，B2048执行OOM，预算关闭

单次批准job15671732在n03运行3800s后`FAILED1:0`：B1024 child0:0/2287s完整密封，B2048 child1:0/1501s无checkpoint。test02 canonicalCPU/taskset0–3/OMPBLAS1/`-B`实际严格collector158.59s、exit0，准入B1024的2051gate/760leaf/2048保存输出对FULL64比较，扣留B2048；execution8aa2e05仍clean。见[终态当前结果](../../../outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/CURRENT_RESULT.md)、[summary](../../../outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/terminal-summary.json)、[实际collector](../../../outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/collected-fixed2048.json)和[原始artifact index](../../../outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/terminal-artifact-index.json)。原始881624142B完整checkpoint本地/远端保留，不纳入Git。

同28672lane-days（64真实FULL64点×32、连续Days2–8/9–15、48步/日、原生dtype/全部输出mask/固定容差）：B1024×2 core median81.772918770s、范围81.765316296–81.787853347，wrapper95.848153921、95.645140350–96.195542384；吞吐350.629529/299.139825lane-days/s。主warm单位费用core1.9013420473e-06/wrapper2.22861220984e-06CNY/lane-day，仅热collection，不含compile/IO/失败candidate。旧fixed1024计时未乘二或混入新实测。

B2048已compile，在wrapper第二连续块申请9161696464B时OOM；不是guard拒绝，不发布未seal core/first速度或两shape比。failedstage没有持久化postcompile actual analysis/snapshot，明确保留证据缺口；不泛化所有B2048布局硬件不可行或声称峰值VRAM。父raw9只计一次：1.055555556GPUh/4.222222222allocatedCPUh、CNY2.533333333主/2.617777778保守加和（账单未核实），低于4.80/4.96批准上限；28s排队不计费。该独立预算及旧预算均关闭，无retry/extension/新作业/automation；无10×/Fortran/年度669点50年/训练AD/CPU-GPU ratio/一般科学验收。

### 当前结果：fixed1024双stage准入，独立预算已关闭

2026-10-09用户批准的单次job15662711已`COMPLETED0:0`，n05上1node/1task/4Slurm CPU/1V10032GB，parent3371s，90min/no-requeue/no-exclusive；执行SHA `bd9e4c04af68d77e79b984bfc8aa064977f893d3`终态仍clean。test02 exact deployed collector用canonicalCPU环境、affinity0–3、OMPBLAS1、`-B`完成149.98s，exit0，准入2stage、扣留0、不完整0；每stage1027gate、760leaf、1024saved FULL64 comparisons通过。完整checkpoint/report/manifest/events/diagnostics及SHA已保全，见[当前结果](../../../outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/CURRENT_RESULT.md)、[终态summary](../../../outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/terminal-summary.json)和[artifact index](../../../outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/terminal-artifact-index.json)。

同一新进程中B512×2/B1024×1均为64真实FULL64点×16、连续Days2–8/9–15、48步/日、14336lane-days，五次同步core及wrapper：B512 core median45.543218s、全范围45.537509–45.564560，wrapper52.195214、52.032337–52.241993；B1024 core40.864703、40.859622–40.882893，wrapper47.405972、47.297268–47.626702。新B1024 wrapper吞吐302.409157lane-days/s，较新B512274.661200为1.101026×；median时间及热单位费用降低9.175634%。它是这两个实测shape中热单位费用较低者，不证明独立输入/production IO的最优点。旧job15657822的计时没有乘二或混入；native dtype、完整输出/mask、容差及物理合同未改。

完整parent仅计一次，0.936388889GPU card-hours、3.745555556分配CPU-hours；实际费用估算CNY2.247333主/CNY2.322244保守相加（计费未核实），低于已批3.60/3.72。lower/compile/first-call/stack/transfer/IO与全作业cost分列于结果，不算缺少完整setup归因的K。startup affinity24,25,80,81映射2物理核，load18.67,19.75,20.12；选卡global0/UUID GPU-51897aa5-dac4-9d20-523e-830c362cf7a6/V100-SXM2-32GB/driver525.85.12，进程列表为空；n05提交时MIXED，不是idle/预留。无新CPU/GPU ratio、Fortran、年度669点50年、10×、训练AD或general science acceptance。[既有paid ledger](../../../outputs/research/global_throughput/server-cost-preparation/PAID_EXECUTION_LEDGER.json)独立预算已关闭，旧预算不复用；无重提/延期/取消或新automation。

### 已完成付费轮与单点诊断

历史付费pilot曾获批CPU1、CPU4、1卡加4辅助CPU，各一小时硬上限；
原批准额度按上述比例最多2.65元，按CPU0.07元和GPU2.20元相加保守为2.83元。
CPU1付费作业15579758运行852秒后在B16第7点299.0-115.0的GPU reference门槛失败；之后获批的独立免费单点诊断已闭合，完整计算了目标点Days2–8首块380个输出叶。唯一失败为`[0][0][7]`，即末态`diffuco_previous_step_state.rveget[0,13]`（整个冠层气孔阻力，s/m）：CPU 592.8152487724788，对GPU B4 reference 592.8152488575903，signed delta -8.511153737345012e-8，absolute delta 8.511153737345012e-8，bound 6.928152488575903e-8，ratio 1.228488222708634。该叶native dtype/shape为`float64`/`(1,14)`；gate-wide `max_abs=8.940696716308594e-8`是聚合值，不是此叶的误差。165/380叶存在非零差异，另外379叶均通过固定比较（含离散值与哨兵规则）。旧付费CPU1 candidate未保存；上述candidate值只属于新单点诊断。完整首块输出已保存，见[当前结果](../../../outputs/research/global_throughput/cpu-leaf-reproduction-20261006/CURRENT_RESULT.md)、[目标叶对比](../../../outputs/research/global_throughput/cpu-leaf-reproduction-20261006/remote-evidence/shared-final/rveget_comparison.json)、[全叶汇总](../../../outputs/research/global_throughput/cpu-leaf-reproduction-20261006/remote-evidence/shared-final/all_leaf_comparisons.json)及[采集完整性核验](../../../outputs/research/global_throughput/cpu-leaf-reproduction-20261006/COLLECTION_VERIFICATION.json)。

该单点一次性诊断不产生可接受的CPU热吞吐或CPU/GPU速度/成本结论。完整窗口实际wall合计1636.648 s，保守计1653 s；最多两次的全量scalar编译额度已用完，只有一次完整首块调用。原CPU1 StableHLO原始内容未保存，当前图hash不匹配旧hash；两个保存IR的内部顺序置换及set派生字段顺序是线索，不足以证明旧图等价或根因。因2/2编译额度已用尽，未编译target-owned closure，无法区分闭合绑定、CPU scalar/GPU B4算术、backend或compiler决策的因果贡献；这些仍未隔离。详见[IR与根因边界分析](../../../outputs/research/global_throughput/cpu-leaf-reproduction-20261006/IR_ANALYSIS.md)。旧付费执行账本保持关闭；旧GPU自动跟进已结束。这不把整体吞吐研究标记为暂停，也不表示创建或续跑了新heartbeat。
CPU4前提不满足，本轮不提交；CPU1不自动重跑，容差保持原样。
两次初始启动因Slurm字段差异失败，GPU替换15579759在Python前退出6；所有失败费用
纳入原额度。GPU启动修复部署到独立checkout；作业15579869已于2026-10-06完成，B64和B16
各有五次同步计时并通过数值门槛，CPU B16仍因同一第7点门槛失败而不完整。GPU core吞吐分别为
159.079（B64）和53.977（B16）point-days/s；**在该旧付费报告中**没有CPU/GPU同工作量比值或赢家结论。当前CPU1探索性B16性能比见本节顶部；科学一致性与最终费用胜负仍未接受。
按未核实账单费率作CPU 0.05/0.07元每核小时、GPU 2.20元每卡小时的加和情景估算，
GPU whole-job主/保守口径为1.406667/1.453556元；本轮全部终态作业合计为
1.421889/1.473644元，不代表已核实的实际收费，仍在原保守批准额度2.83元内。证据见
[阶段与费用报告](../../../outputs/research/global_throughput/server-cost-results/EVIDENCE_REPORT.md)、
[汇总验证记录](../../../outputs/research/global_throughput/server-cost-results/verification.json)
及[付费执行账本](../../../outputs/research/global_throughput/server-cost-preparation/PAID_EXECUTION_LEDGER.json)。
当前平台有界资产检查尚未找到匹配Fortran可执行文件与build/input契约，Fortran对照待补。
第四轮免费账本和原始结果保持原样；旧付费基准账本保持关闭，旧GPU自动跟进已结束。上述获批单点诊断现已完成，不代表重复执行付费或批量吞吐测试。
准备验收已通过14项新增协议测试、35项受影响的既有scale测试及Ruff/diff检查；
FULL64数值参考读取兼容性核对64点和760个array leaves。四worker本地spawn检查只用合成输入。
这些本地检查使用Windows/JAX0.10.1，不替代部署0.4.38的组件检查或真实服务器物理计时。
验证记录在`outputs/research/global_throughput/server-cost-implementation/verification.json`；
后续Slurm/cache接口及GPU身份启动修复分别保存在其`startup-repair/`和`gpu-startup-repair/`。

首个本地实现已加入 `driver/global_batch.py` 与
`scripts/dev/benchmark_global_batch.py`：在 AOT 前提取原完整 transition，保持内层
`npts=1`，外层 `vmap` 的八组数值输入逐点堆叠，科学日序只在逐点相等后共享。
四个真实点 001/069/319/331 的各自 cold Day1、静态科学 closure、schema/dtype/None
审计已经通过；13 项轻量单元测试和新增文件 Ruff 检查通过。B1 原 AOT 与提取后的
两块完整状态/输出通过（760 leaves，有限值零差异，严格匹配原 bare-soil roughheight
NaN sentinel）。四点各自 context 的 scalar reference 在逐点清空内层 cache 后生成；
B4 逐点、置换和 lane 替换隔离通过，最大逐元素组合容差占比 0.08167。
物理 core 冻结在 `a33b82d`；独立参考来自 `b8bb285`，最终 harness `80831ea`
保全并复用这些参考，先验证共享 scalar 对每个独立参考零差异，再做完整 batch gate 和计时。

本地 Windows CPU、i7-13700（16物理/24可见逻辑核）、JAX/jaxlib 0.10.1、FP64，
每次从同一冻结 Day1 末态重置，连续推进两个7日块；五次热运行同步全部末态/输出和 effects：

| 同工作量 | batch 热 core 中位数 [min,max] | JAX 常驻串行中位数 [min,max] | batch 吞吐 |
| --- | --- | --- | --- |
| B1，14 point-days | 1.873 s [1.864,1.970] | 2.007 s [1.830,2.483] | 7.475 point-days/s |
| B4，56 point-days | 3.442 s [3.216,3.551] | 7.818 s [6.924,8.827] | 16.268 point-days/s |

B4 本次相对同工作量 JAX 串行的 core 比为 **2.271倍**。含已准备输入的 stack/transfer/
host materialization wrapper 中位数4.015 s；B4 lower23.575 s、compile95.345 s、
first call4.490 s。完整四份独立 scalar 编译进程峰值 RSS7.72 GiB，复用参考后最终计时
进程峰值5.30 GiB；GPU数与card-hours均0。初态准备和独立参考编译另计，wrapper不含
新cold-start、年度聚合或生产checkpoint。Fortran同机对比、GPU吞吐、年度/669/50年和AD
仍未由此短窗验收。聚合证据为 `outputs/research/global_throughput/local-milestone-summary.json`。

Explore1000免费诊断完成四点独立Day1/static/schema审计（test02，wall267.49 s）；
随后GPU `b8bb285`复用其准备输入，在gln01实际节点`ibc13b02n04`使用1张
V100-SXM2 32GB、affinity0–3、1200 s预算。原scalar reference在650.393 s保存，
extracted scalar compile在659.739 s开始；预算边界结束时尚未完成GPU batch parity或计时。
wrapper exit1，原report停在`running`，stderr仅有slow-compile alarms，不能解释为数值失败。
进程已退出；此免费GPU诊断预算1/3 card-hour，四核affinity容量4/3 core-hours，
快照实耗CPU下界0.375 core-hours、进程峰值RSS下界9.254623 GiB。1200个逐秒GPU样本
的节点card0峰值987 MiB，属于编译中设备观测，不是完整batch显存需求。
原证据在 `outputs/research/global_throughput/remote-gpu-b8bb285/`；后续launcher cache路径
隔离修复发生在运行结束后，不属于该运行条件。

此前`gpu-round2`（最终`bdd5790`）保全六份数值snapshot：B4 candidate、原001 AOT和四份
各自context独立GPU reference，15项记录gate通过；最终在B4重新编译预算边界停止，没有热
吞吐结论。cache修复后069 miss/write、319/331同key实际hit（compile/load23.939/23.910 s），
pooled001仍miss/write；根因未解析，不归因于callsite metadata。原记录保持不变，第三轮只读复用。
该轮3600 card-seconds合同按probes14+3、model1 1567、model2 695、model3保守1300计费，
合计3579（0.99417 card-hours），余额21；完整window实际wall3572.385 s，包含启动、输入读取、
编译、references/gates和IO，原Day1及旧AOT另计。最终GPU0释放；supervisor先记录survivor89971，
上海03:21:25后查无该PID/PGID，不能称即时无残留。证据`gpu-round2/`及`gpu-round2-budget.json`
均在本worktree `outputs/research/global_throughput/`，详细来源见执行审计及合并记录。

第三轮`cea0edf`采用batch-first协议，1张同节点V100-SXM2 32GB、affinity4–7、FP64、
JAX/jaxlib0.4.38；只读导入六份旧数值snapshot，新B4先编译并重新通过candidate/四reference
完整两块parity、置换及lane替换的10项gate，再完成五次同步计时：

| 56 point-days，同一GPU/输入/精度 | 中位数 [min,max] | 吞吐point-days/s |
| --- | --- | --- |
| B4热core | 3.061469 s [3.056491,3.093887] | 18.291873 |
| preprepared stack/transfer/materialize wrapper | 3.198821 s [3.180061,3.206820] | 17.506448 |
| JAX常驻scalar串行 | 10.938078 s [10.924755,10.942314] | 5.119729 |

新B4 lower93.711854 s、compile670.043815 s、首次同步双块19.111908 s，实际cache miss/write
key`643afa8f…`；完整输出、gates和原始samples于elapsed835.6367 s保存在独立
`B4_batch_only_v1`，随后才开始069 scalar编译。B4逐元素组合容差最大占比0.0453022350、
max absolute1.4908146113e-6，原精度/sentinel/schema政策不变。最终9个required stage全部
complete、24项记录gate通过，状态`passed_bounded_forward_diagnostic`。GPU B4相对同设备
同工作量JAX串行core为**3.572820倍**；本地CPU的**2.271190倍**是另一个硬件/runtime的独立
结果。单点scalar B1为2.735689 s/14 point-days、wrapper2.821319 s；不是独立vmapped B1验收。
069 pooled scalar也实际miss/write（key`5d6b4815…`），lower14.299802 s、compile623.917697 s，
没有复用旧069 disk key。根因尚未解析；两次新编译成本均不隐藏为cache load或热运行。

第三轮唯一免费window实际wall1608.972354 s，charge1609/1800 card-seconds（0.446944
card-hours），未重试或付费；余额191不授权新window。旧六份source/manifest、旧budget及
prepared input后查均未改变；本任务session为空，GPU0零MiB，其他线程GPU1 PID11889不动。
短窗warm/wrapper不含新Day1、独立references、年度聚合或生产checkpoint；完整window含
启动/编译/IO。这不是Fortran、年度/669点/50年、AD或生产端到端验收。证据目录
`outputs/research/global_throughput/gpu-round3/`，ledger是
`outputs/research/global_throughput/gpu-round3-budget.json`；保持各旧stage原origin与未记录来源限制。

本地与远端合并记录见
[MILESTONE.md](../../../outputs/research/global_throughput/MILESTONE.md)和
[milestone-summary.json](../../../outputs/research/global_throughput/milestone-summary.json)。

第四轮最终状态：B16完成，B64不完整；不发布B64吞吐。64个唯一真实点、嵌套16点前缀，连续两个7日块，
每天48步、内层`npts=1`；已接受阶段各保留五次同步样本，并通过逐lane parity、置换及替换隔离。
baseline为同点、各自forcing/参数/冻结Day1状态的实际常驻B4组，未外推旧scalar速度。

| 同工作量，两块各7日 | 同步热core中位 | 已准备wrapper中位 | 实际常驻B4组串行中位 | core点日/s | core相对B4组 |
| --- | --- | --- | --- | --- | --- |
| 本地CPU B16，224点日 | 9.068345 s | 9.012244 s | 12.447431 s | 24.701309 | 1.372624× |
| GPU B16，4组B4，224点日 | 4.095351 s | 4.278145 s | 12.208924 s | 54.696163 | 2.981166× |
| GPU B64，16组B4 | 未完成，吞吐不发布 | — | — | — | — |

GPU最终129项已记录gate通过、0失败；B64原stage仍`running`，没有complete checkpoint或warm samples。
window2监督状态`over_budget`、exit -15，是预算边界停止，不作为数值失败。
B16保留window1的原执行和测量origin；window2只复用其数值checkpoint。

单张V100-SXM2 32GB，节点`ibc13b02n04`，模型affinity4–7，JAX/jaxlib0.4.38。
两次免费window总wall 3525.762176s，charge **3527/3600卡秒（0.979722 card-hours）**；
余额73，最多两次/每次1800s的账本未重置，也不授权第三个window，无付费任务。
编译仍是边界：window1 B4/B16 compile624.034/702.748s；window2 B4/B64为623.854/780.177s。
四次实际cache均MISS/write；window2 B4 key/IR与window1不同，原因未解析，数值checkpoint复用不等于cache命中。

本地B16通过，B64未完成；累计reported wall1190.948/1200s，最后一次safe deadline超出**20s**。
Linux FULL64生产reader入场通过，最终window865.465266s，含此前失败/诊断的实测累计1127.321642s；
100s housekeeping仅为保守预留。原remote4与NEW60具名state/120块原始digest保留。
JAX x64开启并保留基线原生dtype；地理payload及solar子计算含FP32，无物理/精度/容差变更。

core与已准备wrapper分别计时，lower/compile/首调用/IO及完整window另列；均不认证生产端到端。
详见[专用汇总与raw samples](../../../outputs/research/global_throughput/round4-measurement-summary.json)及
[执行审计](../../../docs/research/global_throughput/EXECUTION_AUDIT.md)。
[旧资产保全](../../../outputs/research/global_throughput/gpu-round4/post-window-preservation-check.txt)21项hash均OK，
[运行后检查](../../../outputs/research/global_throughput/gpu-round4/post-window-checks.txt)确认checkout干净、自身PID/PGID退出及GPU0释放。
第三轮记录保留；本轮不认证Fortran加速、年度/669点/50年、AD、训练、10×或跨CPU/GPU因果加速。

输入准备由`prepare_global_scale_inputs.py`维护实际LIMIT中心和动态参数的确定性选点、
各点v2 source checksum、Day1及完整两块输入记录；
`prepare_global_scale_remote.py`在Linux重建必要context/path元数据，并消费已保存的新60点
Day1状态和完整冻结BlockArguments；保留原remote四anchor，不重新生成Day1或编译transition。
首次重算输入在215处失败；独立两块快照比较确认16个FP64叶子低位不同（forcing/查找表），
不是treedef字符串差异。运输完整原数值输入时先严格核对原digest，不把相近的重算值
视为相同输入，也不改变公式/精度/容差。Linux与本地的state字段顺序不同：原remote4
保持未变，NEW60仅按具名state字段映射到其布局，核对每个字段dtype/shape/bytes并记录
原始及绑定位置digest。其余八组参数保留原始数值；VEGETATION_FILE仍只允许已声明的
路径映射，实际local/remote文件内容及canonical static closure/schema均须通过。
此前scale的逐浮点leaf全FP64门槛拒绝旧anchor原有`float32`payload；现严格要求冻结基线的
四条确切路径/dtype/shape。其他浮点、离散、static、身份和非有限值门槛保留，
reader/gate不做cast；后续入场及同工作量实测按此合同执行。
内存入场以真实input/output字节、compiler buffer analysis和所选物理GPU free/total memory
保守投影，要求至少2GiB余量；RSS不是VRAM，入场估计也不是无OOM保证。
这些检查及新规模结果仍只认证有界完整forward，不能替代Fortran/年度/669点/50年或AD验收。

可复现命令（数据根只读，生成物在本 worktree；默认禁持久 compilation cache）：

```powershell
$env:ORCHIDEE_DATA_ROOT = 'E:/ORCHIDEE-MAN-JAX/data'
$env:ORCHIDEE_REFERENCE_ROOT = 'E:/ORCHIDEE-MAN-JAX/reference'
conda run --no-capture-output -n ORCJAX python scripts/dev/benchmark_global_batch.py --asset-root outputs/research/global_throughput/local-assets --truth-root E:/ORCHIDEE-MAN-JAX --output-dir outputs/research/global_throughput/local-audit --mode audit
conda run --no-capture-output -n ORCJAX python scripts/dev/benchmark_global_batch.py --prepared-input outputs/research/global_throughput/local-audit/prepared_inputs.pkl --truth-root E:/ORCHIDEE-MAN-JAX --output-dir outputs/research/global_throughput/local-b1-b4-final --sizes 1 4 --warm-repeats 5
```

`--asset-root` 使用 `landpoints/{id}/runtime_used_run.def` 与逐点 `reference/`，也可用
`--input-manifest` 指定每点 `landpoint_id/run_def/reference_dir`。未提供两者时使用现有
raw-server reference resolver。准备输入与 cold Day1 checkpoint 用集合 pickle 保存，
普通审计错误修复可复用；这不是生产恢复格式。`report.json` 与 `events.jsonl`
在失败时保留实际阶段、异常和峰值 RSS，不以不完整执行报告加速。
如完整独立 references 已保存而仅诊断代码失败，可在同输出目录使用
`--resume-references`；它要求原来每点的 cache-clear 记录、完整输入身份和科学源码一致。
本次最终成功是此模式修复 JAX 置换索引后得到的实测，原 report 保存为
`reference_origin_report.json`，没有重做已通过的四份独立 scalar 编译。

本轮有界GPU诊断使用 `--mode staged`，按 `--stage candidate|references|validate|all`
执行并恢复：先保存实际B4双块完整结果，再逐点清空内层cache、绑定自身context、lower/compile/
执行并保存四份独立scalar reference。第三轮`72fc47d`及路径加固`cea0edf`改变validate顺序：新进程首先
只编译B4，立即对保存candidate及四份独立reference核对实际完整两块输出，再过置换和lane
替换；随后至少五次同步56 point-days热core和preprepared wrapper，原始samples、完整输出、
gates共同atomic保存为`B4_batch_only_v1`。之后才构建/加载共享scalar、重验四份reference、
测同56 point-days串行及14 point-days单点scalar基线；后续scalar失败不破坏已完成B4资产。
本地30项及后续5项定向测试通过，第三轮B4/实际scalar及单点scalar基线全部完成GPU有界验证。
默认`--b1-mode vmapped`另编译B1 batch；
有界GPU筛选可显式选择`--b1-mode scalar`，
使用已对独立reference/原AOT验证的单点程序，单列`B1_scalar_baseline`的14 point-days与五次
同步热core/preprepared wrapper，不能称为独立vmapped B1 GPU验收。两模式使用不同完成阶段键。
本地`f1df818`新增`--pooled-scalar-point`，默认001不变；其他选择使用该点自身已审计context，
并对每个新建的实际pooled executable重新通过四份独立reference后才计时。不同selector的
scalar comparison/B1阶段分别存储，共用candidate、独立reference和已完成的selector无关B4
batch-only checkpoint；不能因选择已缓存context便推定命中。
本轮GPU实际代码`cea0edf`显式选择069；实际scalar对四reference有限值零差异，但新disk key
miss/write，不能将selector选择称作cache命中。
每个完成阶段写atomic checkpoint与`stage_manifest.json`，未完成阶段
标为running/failed/stopped；恢复只跳过完整且身份匹配的阶段。B4计时与B1阶段分别保全，
避免后续编译预算结束丢失已经完成的测量。
新输出目录可用`--mode staged --stage validate --stage-source-dir <旧model目录>`只读导入
六个完整数值prerequisites：identity及checksum严格复核、原execution origin保留，旧manifest、
reports及ledger不改写；旧B4 gate/measurement不导入，不充作第三轮新测量。
相同或任一嵌套source/destination路径在CLI输出/cache setup及StageStore创建前拒绝。

阶段身份核对完整输入的tree/path/shape/dtype/value、static closure、科学源码内容、固定baseline、
FP64、Python/JAX/jaxlib/NumPy、设备/backend及compiler环境；采用内容hash，不依赖pickle
对象共享关系。`--legacy-reference-dir`只能在复核已记录runtime、完整输入、原commit科学源码
和transition AST后导入已保存的原AOT单点reference；它不代表四份独立提取后reference。
旧记录未包含完整CUDA driver/backend和执行时源码hash，导入provenance明确保留此限制。
CPU数值gate不能替代GPU数值证据。

`--cache-mode off`仍是默认；实验性persistent模式须显式给`--cache-root`，仅复用JAX本身
按实际key匹配的编译缓存，不跨runtime导入序列化AOT。各次lower保存StableHLO内容hash和实际
cache日志，相同shape或较短耗时都不当作命中证明。小组件
`scripts/dev/probe_global_batch_cache.py --output-dir <dir> --cache-root <root> --cache-phase write|read`
在两个独立进程中核对实际miss/write与hit/read，以及runtime/device/IR/output身份；
组件命中不证明物理模型编译命中或吞吐收益。JAX0.4.38若在配置前已因物理模块导入编译，
会把无目录的cache初始化记为完成；仅晚设路径不能修复。本诊断以launcher早期cache环境变量
配合显式配置后的公开`reset_cache()`解决，既不删除磁盘cache，也不改数值程序。
`outputs/research/global_throughput/cache-init-repair-cpu-20261005/`记录test02实际0.4.38
heavy-import/late-config无reset时不写cache、修复后写入与另一新进程实际命中的控制证据；
它只验证组件cache初始化，不认证完整物理模型cache命中或速度。
新增阶段恢复/身份/小JAX执行及checkpoint IO/时序来源测试包含在30项
轻量测试中；本地记录为 `outputs/research/global_throughput/staged-implementation-checks.json`。

资源合同按supervisor完整window wall记账，包含启动、导入、复用输入读取/审计、实际lower/
compile、reference与gate执行、测量、输出/checkpoint及包装开销。原Day1准备和之前已保存
original reference的编译成本分开列，不计作本轮新执行。热core只含从冻结起态推进两个块并
同步全输出；preprepared wrapper还含stack/transfer/host materialization，仍不等于新冷启动、
公共年度CLI或生产checkpoint端到端速度。新增独立stack/transfer和snapshot序列化/atomic写入
指标只适用于采用该logging修订新执行的阶段；旧完成阶段或中断后无IO记录的恢复阶段明确
保留“未单独测量”，不得从window差值伪造阶段IO时间。
每个新阶段保存执行环境、cache配置、affinity、Git提交、launcher给出的物理GPU UUID/driver
及独立window report。`timing_origins`区分新测量与复用测量；旧阶段未记录的来源明确缺失，
指向原report/launch context复核，不能由恢复进程的环境反向填充。

1. 固定基线（本地短窗已完成）：同一组真实网格、forcing、初态、参数、输出和精度，测起点完整块。
   分开记录进程启动/数据准备、冷编译、缓存加载、热执行、数据传输、输出与恢复点 IO。
   首轮用小窗口和原有状态资产，避免先跑 669 点或 50 年。
2. 最小实现（本地已通过）：在编译前暴露纯 transition，先做 `B=1` 等价，然后 `B=4` 不同网格的批处理。
   优先考察对单点函数的 `vmap`；不要把已经 `.compile()` 的可执行文件直接套入变换。
   审计所有闭包中的静态配置，连续地点参数必须逐点传入；仅形状相同不足以证明可合批。
3. 有界吞吐筛选（本地CPU B1/B4完成，其余待验收）：真实异质点 `B=1,4,16,64`，按内存和编译结果逐级扩展；同时测 CPU/GPU。
   第三轮B4完整parity/置换/隔离、热core/wrapper、同工作量scalar对照和单点scalar基线通过，
   GPU同工作量core收益3.572820倍；编译/time-to-valid-measurement仍占主导。后续资源与
   更大规模须另定合同，不将短窗收益直接外推年度或更大batch图。
   主测七天块，先过正确性再判断是否值得测 28 天块。保留原精度、全部有效过程和输出。
   重复相同网格只可作设备负载诊断，不能代替异质全球吞吐证据。
4. 只有批处理确有收益，才扩展代表性一年、跨年恢复和多设备规模曲线；
   最后处理 669 点完整科学验收和 50 年生产规模。先明确“快在哪里”，再投入长作业。

首轮比较至少包含：现有单点 JAX 常驻串行、JAX CPU 批处理、JAX 单 GPU 批处理。
当前论文 Fortran 的同工作负载结果是最终速度基线；拿不到匹配环境的实测时，
单列用户历史值，不发布已证实的 Fortran 加速比。

## 不变的验收条件

- 批处理逐点对齐独立执行的完整末态、年度/日输出、schema、整数与逻辑 mask；
  复用原 fast-path `atol=1e-8, rtol=1e-10`，离散字段精确一致。
  Fortran 科学验收另循[现有政策](../../source_audits/paper_scientific_acceptance_policy.md)。
- 覆盖异质参数、干旱、显式积雪、边界日、冷启动及 restart；保留静态 Fortran 分支 ledger。
  批中 padding 不得污染有效网格；不得让某点的参数或初态被广播给其他点。
- 每次计时都等待设备计算完成。至少报告重复热运行的中位数与波动，
  编译缓存身份、CPU 核数/亲和性、GPU 型号/数量、主机负载、峰值内存/显存。
- 核心指标：有效点年/秒、完整任务 wall time、CPU core-hours、GPU card-hours、费用；
  将编译、IO 和 checkpoint 纳入端到端结果。多卡提速不等于单卡效率提升。
- 缓存键区分影响编译的配置/布局/批大小；连续状态与参数不应制造不必要重编译。
  不为追求占用率偷偷降低精度、省略过程、缩短递推或增加采样数据量。
- 逐块加载 forcing、常驻当前状态，按任务需要保留输出；普通正向模拟无需保存所有半小时全状态。
  输出/恢复点采用可恢复的集合文件与索引，并核算字节及文件数；不按变量或时步制造大量碎文件。

## 可微性的独立价值

若损失是单个标量，反向 AD 通常可用一次前向和一次反向传播求很多参数的梯度，
避免有限差分随参数数量成倍重复完整模拟；实际代价受重计算、存储和分支影响。
这有望使全球联合参数校准、观测约束和归因更可行，但不承诺优化迭代次数或总时间。
50 年反向传播的内存/数值稳定性、离散阈值附近的导数和参数路径完整性需单独验证。
正向运行不应为了“可微”而额外求梯度或保存反向轨迹。

## 工作目录与操作约束

本分支目录：`E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`。
根目录现为 `main`；日尺度研究已迁至同级 `daily-coarse-graining` 工作树。
每个线程固定 cwd，不在另一个工作树切分支或修改其研究状态页。
管理、同步和恢复入口见[工作树规范](../../WORKSPACE_MANAGEMENT.md)。
外部数据、Fortran source 和参考资产仍在父工作区，按已声明绝对路径只读复用；
新生成物放本分支 `outputs/`，不要复制一份大型数据树。

本次完成立项、来源审计、最小 batch 实现、本地静态输入及 B1/B4 有界数值/计时门槛。
远端免费输入审计及GPU B4完整两块parity/置换/隔离、热计时和同工作量scalar对照通过。
没有执行训练；当前付费CPU/GPU基准的授权与状态见上方当前工作单及执行账本。
后续 Explore1000 操作继续服从项目部署说明和专用技能；付费作业仍需明确资源、
时限、命令、目录和最坏费用授权。该分支不继承另一个研究线程的特定 GPU 实验额度。
