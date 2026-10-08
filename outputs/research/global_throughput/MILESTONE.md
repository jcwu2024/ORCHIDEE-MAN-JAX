# 物理模型全球吞吐：当前里程碑

## 当前状态：CPU1探索性B16计时已完成

CPU1作业15591356于Explore1000 n04完成：1 CPU、0 GPU、父作业wall1176秒；固定16真实点、两块连续7日、每日48步，共224 point-days，五次同步core/wrapper样本。**性能已测；跨后端科学一致性未接受。** 完整证据见[CPU终态结果](cpu-performance-results-15591356/CURRENT_RESULT.md)、[terminal measurements](cpu-performance-results-15591356/terminal-measurements.json)、[费用情景与本次CPU账目](cpu-performance-results-15591356/server-cost-results.json)及[CPU-internal reader admission](cpu-performance-results-15591356/cpu-internal-reader-admission.json)。

| 阶段 | 点日 | 分配CPU/GPU | Core中位数 [范围] | Wrapper中位数 [范围] | 热费用主情景 core / wrapper | GPU/CPU吞吐比 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CPU1 resident scalar，exploratory | 224 | 1 CPU / 0 GPU | 31.458503002 s [30.855936289, 31.737742664] | 31.760353759 s [30.915467992, 32.648889609] | CNY0.000436924 / 0.000441116 | — |
| 已接受GPU B16，历史同工作量参考 | 224 | 4 CPU / 1 GPU | 4.149942968 s [4.142994318, 4.152027067] | 4.364678632 s [4.359835308, 4.370462861] | CNY0.002766629 / 0.002909786 | core7.580466345×；wrapper7.276676345× |

17项固定gate中15通过、2项有限tolerance差异失败：299.0-115.0 ratio1.2284882227，285.0-087.0 ratio1.7088155265。原dtype和`atol=1e-8`/`rtol=1e-10`未变，冻结repeat零差。collector通过SHA/contract/样本检查，故完整探索性计时可发表，但accepted CPU stage count为0；比值仅说明性能，不接受科学一致性。费率情景为CPU CNY0.05/allocated-core-hour（保守0.07）、GPU CNY2.20/card-hour；GPU四个辅助CPU按核时加费，每小时GPU allocation为CNY2.40主情景或2.48保守。当前B16热样本GPU热费用高于CPU1：core为6.332064倍、wrapper为6.596418倍；GPU仍快7.580466345与7.276676345倍。可说此B16热样本CPU较省钱而GPU较快，不外推年度/669点/50年最终赢家，不制作44CPU实测结论。

父作业主/保守费用情景0.016333333/0.022866667元，按1176秒父作业计一次，不按点日摊分且不是已核实账单。lower56.317166498秒、compile688.378443662秒、first call32.158282455秒、checkpoint序列化0.005034860秒、原子写入0.025802774秒分列；无完整stage startup/validation accounting，不给K复用估计。

CPU-internal reader首次检查失败的精确原因现已定位：checkpoint的`record.execution_origin.environment.loadavg`为tuple `(0.14,0.06,0.06)`，JSON manifest为同值list。仅在该telemetry字段允许tuple/list等价，并继续要求长度3、有限非负值及每项type/value相同。patched reader通过未改动CPU1 B16快照的实际`load_reference`与全部`_validate_cpu_internal`后续检查，760叶、16 lanes、两块、5+5样本；source checkpoint/manifest/report hashes未变。reader角色是`cpu_internal_comparison`，不代表accepted CPU或科学parity通过；两项原GPU tolerance失败仍失败。见[差异证据](cpu-reader-repair-20261006/record-differences.json)、[实际reader验证](cpu-reader-repair-20261006/actual-reader-validation.json)及[修复报告](cpu-reader-repair-20261006/REPAIR_REPORT.md)。旧付费/free账本关闭，旧GPU自动跟进已结束并删除，没有续跑旧heartbeat；当前GPU15600405由独立的新heartbeat gpu-15600405 跟进。

以上CPU reader与旧heartbeat记录截至旧作业终态；`gpu-15600405`已暂停。它们是历史状态，不表示当前创建了自动化。

## 当前工作单：修复B256调用签名并准备有限延续

旧作业15600405已终态FAILED：唯一完整stage为B128/512 lanes，严格验证通过并保留；B256 compile-complete后首次`run_collection`因AOT输入数673/536不匹配而TypeError失败，无B256完整first call/checkpoint/gates/samples，不发布吞吐；该错误非数值门槛失败、OOM或timeout。旧job未查明根因。当前规范JAX0.4.38有界lower-only为123.377s（180s上限内），caller/public tree/internal AllArgsInfo均报536 inputs、417 kept，没有完整模型compile或run。小型顺序B128/B256/B512签名测试通过；无关673-input lazy-binding故障注入可重现旧异常，但`Compiled.call` bypass的输出一致。自然触发未复现；call-adapter是防御性绑定修复，不证明历史根因。冻结物理、dtype、容差未改。B128原SHA/bytes、760×512 leaves、515 gates、5+5样本及provenance重验通过；`loadavg`唯一允许tuple/list JSON等价，科学输入/schema mismatch仍拒绝。B128 lazy-call与新B256/B512 Python-bound dispatch不同，实测collection时间和费用可比较，但速率比不能单独归因batch size。部署后的最终actual-module canonical JAX0.4.38诊断通过，提交`9d19f0760b93e1186eaf796db4903ddfd16a806f`位于独立无hardlink/no-alternates checkout，旧付费checkout未变。最终Bash syntax及target参数展开→Python dry-run通过，candidate为[256,512]，未导入JAX或初始化模型/GPU。历史自然触发仍未解释，真实GPU B256/B512 collection未验证。见[修复报告](gpu-signature-repair-20261008/REPAIR_REPORT.md)、[当前结果](gpu-signature-repair-20261008/CURRENT_RESULT.md)、[模块诊断](gpu-signature-repair-20261008/final-module-canonical-diagnostic.jsonl)、[launcher验证](gpu-signature-repair-20261008/final-launcher-verification.txt)、[deployment verification](gpu-signature-repair-20261008/deployment-verification.json)及[计划审阅](gpu-signature-repair-20261008/PAID_RUN_REVIEW.md)。旧作业B512未执行。运行节点`ibc13b03n04`，1 V100和4 Slurm CPUs，观察affinity覆盖两个物理核SMT线程。FULL64源64点每点重复8次，共512 lanes。

v2付费方案经用户批准后已于2026-10-08 14:26:10 Asia/Shanghai提交一次为job15643327，startup阶段FAILED exit127:0；Slurm复制的wrapper错误从spool目录寻找child launcher。job只运行1秒，4 CPU-seconds/1 GPU card-second，主/保守费估CNY0.000666667/CNY0.000688889，账单舍入未核实；未初始化GPU/模型，无benchmark outputs/gates/samples。提交时n03八卡已占满；通知用户后同一pending job经`scontrol`改到n02，无requeue/restart。retarget snapshot `2026-10-08T06:28:47Z`：CPUAlloc12/112、load3.19、free323835MiB、AllocTRES `cpu=12,gres/gpu=1`。v2审批已消耗。[提交receipt](gpu-signature-repair-20261008/submission-receipt.txt)、[状态](gpu-signature-repair-20261008/submission-status.txt)、[retarget receipt](gpu-signature-repair-20261008/node-retarget-receipt.txt)、[启动失败](gpu-signature-repair-20261008/job-15643327-startup-failure.txt)。v3 wrapper已修正并部署验证，commit `12b495b9425de3ad1dc16efa3a5a0e69d7e89222`，checkout `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3`；新run root `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512`未使用。21项本地测试和target wrapper argument-expansion dry-run通过；不是Slurm/container startup或GPU验证。方案为1 V100、4 Slurm CPUs、90分钟，费用场景CNY3.60/CNY3.72。snapshot `2026-10-08T06:50:43Z`时没有idle eligible node；候选n02为mixed、12/112 CPU allocated、load2.85、free323601MiB、1 GPU allocated，非reservation，提交前刷新。v3尚未获批/提交，精确命令见[付费审阅](gpu-signature-repair-20261008/PAID_RUN_REVIEW.md)和[APPROVAL_COMMAND_V3.sh](gpu-signature-repair-20261008/APPROVAL_COMMAND_V3.sh)；须新明确审批，不自动重提、不创建automation。B128已封存并复用，不重编译；固定512 lanes（64个FULL64点各重复8次）仅顺序运行B256后接B512。B512须有至少1800秒加buffer/slack才准入；这是执行时间余量条件，非吞吐门槛。CPU任务不是前置。

B128完整stage严格验证通过：760 native output leaves×512、两块连续7日输出、515项gates及512项独立saved-output/source-B4 comparisons通过，5+5次完整同步collection样本已保存。SHA256为`e91db1391ccb5f525f9ec29f1cba24235c4ce9cd4153b9297fe2ed014d0c9cb5`。Core median29.617320834s [29.605399007,29.623860434]，242.020541 lane-days/s；wrapper33.838030240s [33.780365391,33.889177628]，211.832661 lane-days/s。该结果是exploratory capacity/device-load evidence，`scientific_acceptance=false`；不能据此宣称sweetspot或全球科学收益。

B128 lower/compile/first call为92.979298/744.949671/29.778435s，host stack3.176700s、device transfer1.047304s、serialization0.321704s及atomic write0.251202s分别记录。实际compiler argument/output/temp buffers为31,118,364/27,529,184/582,289,872B；compile前GPU free snapshot33,721,155,584/34,359,738,368B，job-start Host MemAvailable352,786,501,632B，阶段累计process peak RSS14,736,539,648B。GPU读数不是VRAM peak，compiler buffer analysis也不是实际peak allocation。B256仅有诊断lower63.179012s、compile752.614130s且首次调用失败，不发布吞吐或据此外推其memory。B128 warm core/wrapper费用主情景CNY0.019744881/0.022558687、保守CNY0.020403043/0.023310643，仅覆盖单次同步512-lane collection的GPU+4 allocated CPUs，不含冷启动/IO/整job。

Parent终态elapsed2135s，GPU 0.59305556 card-hours、4 CPU共2.37222222 allocated count-hours；主/保守费用估算CNY1.423333333/1.470777778，已由终态collector核对但不是实账。report window wall2121.916519s和滞后`elapsed_seconds=2060.684793`不替代parent账目。终态验证和12项artifact transfer均通过。Heartbeat `gpu-15600405`已由工具更新并读回确认`PAUSED`，旧作业终态跟进完成，已删除的旧Explore1000 heartbeat不恢复；这些是旧作业事实，不限制当前修复准备。[当前结果](gpu-sweetspot-preparation-20261006/CURRENT_RESULT.md)、[terminal validation](gpu-sweetspot-preparation-20261006/job-15600405-terminal-validation.json)、[transfer verification](gpu-sweetspot-preparation-20261006/job-15600405-terminal-transfer-verification.json)、[failure diagnostics](gpu-sweetspot-preparation-20261006/job-15600405-terminal-failure-diagnostics.json)、[raw9](gpu-sweetspot-preparation-20261006/job-15600405-terminal-sacct-raw9.txt)、[collector7](gpu-sweetspot-preparation-20261006/job-15600405-terminal-sacct-collector7.txt)、[terminal capture](gpu-sweetspot-preparation-20261006/job-15600405-terminal-evidence/capture.json)、[B128 checkpoint](gpu-sweetspot-preparation-20261006/job-15600405-terminal-evidence/capacity__stage_capacity_B128_fixed512_v1.pkl)、[ledger](gpu-sweetspot-preparation-20261006/preparation-ledger.json)。CPU reader commit `7d20a17`未改变原两项有限tolerance差异；CPU4和CPU扩展暂停且非GPU前置。1→2卡后置，年度/669点/50年科学验收未成立。

## 历史：已完成付费GPU基准与单点正确性诊断

付费GPU作业15579869已完成（2110 s）；B64和B16阶段均通过固定数值门槛，原生dtype、完整checkpoint及五次同步计时已验收。权威[阶段与费用报告](server-cost-results/EVIDENCE_REPORT.md)、[汇总验证](server-cost-results/verification.json)及[历史付费执行账本](server-cost-preparation/PAID_EXECUTION_LEDGER.json)记录该付费轮结果。该旧账本关闭；先前监视GPU15579869的automation已结束并删除。

| 阶段 | 点日 | Core中位数 [范围] | Wrapper中位数 [范围] | Core吞吐 |
| --- | ---: | ---: | ---: | ---: |
| 付费GPU B64 | 896 | 5.632416466 s [5.627945635, 5.635269955] | 6.222332641 s [6.189428551, 6.311673682] | 159.079 point-days/s |
| 付费GPU B16 | 224 | 4.149942968 s [4.142994318, 4.152027067] | 4.364678632 s [4.359835308, 4.370462861] | 53.976645 point-days/s |

费用按未核实费率CPU 0.05/0.07元每核小时、GPU 2.20元每卡小时加和估算；旧付费GPU轮全部终态作业主/保守情景为1.421888889/1.473644444元，不代表已核实账单。此处“CPU无已接受热吞吐”指accepted CPU stage count为0；新CPU1仅为exploratory。旧付费轮不支持当时的CPU/GPU比较或成本胜负、44核推断、Fortran、年度、669点50年、AD、训练或10×验收。缺少逐stage setup/validation accounting，不给K复用估计。

另行授权的免费单点CPU正确性诊断已**完成并闭合**，不改变付费轮状态，也不把整体吞吐研究标为暂停。test02/ibtest02单物理核、零GPU，在冻结Day1状态下对目标点299.0-115.0完整执行Days2–8一次；380叶严格比较有唯一失败：`[0][0][7]`即`diffuco_previous_step_state.rveget[0,13]`（整个冠层气孔阻力，s/m），float64 shape`(1,14)`。CPU值592.8152487724788，GPU B4 reference值592.8152488575903，signed delta -8.511153737345012e-8，absolute delta 8.511153737345012e-8，bound 6.928152488575903e-8，ratio1.228488222708634；其余379叶通过，165叶总计有非零差异。gate-wide `max_abs=8.940696716308594e-8`不等于目标叶差值。详情见[单点当前结果](cpu-leaf-reproduction-20261006/CURRENT_RESULT.md)、[目标叶对比](cpu-leaf-reproduction-20261006/remote-evidence/shared-final/rveget_comparison.json)、[采集校验](cpu-leaf-reproduction-20261006/COLLECTION_VERIFICATION.json)及[IR分析](cpu-leaf-reproduction-20261006/IR_ANALYSIS.md)。

这是一次性旧轮单点正确性证据，不是CPU热吞吐或CPU/GPU速度/费用比较。诊断窗口wall合计1636.648s（保守计1653s），两次全量编译已用完、仅一次完整首块调用、无热重复。原CPU1 StableHLO内容未保存，当前hash与旧hash不同；保全IR重排及set派生字段顺序只是线索，根因未证实。目标点自身closure未编译，绑定、CPU scalar/GPU B4批处理及后端/编译器原因仍未隔离。旧付费候选仍未保存，该点CPU值来自此次单点诊断，不是新作业15591356的B16值。第一次完整编译在入口重复hash检查处终止、未调用模型；修正后完成首块，其失败窗口计入总预算。后续不声称CPU4、44核、Fortran、年度、669点50年、AD、训练或10×验收。

## 后续状态

CPU1终态以本文件顶部所链`cpu-performance-results-15591356`及[本次独立执行账本](cpu-performance-preparation-20261006/PAID_EXECUTION_LEDGER.json)为准；准备审批文件中的命令和费用是历史计划记录。B16 reader准入已修复并实际通过，但CPU4没有执行或获批；不影响GPU预筛。没有新的提交或作业授权。

已接受GPU B64/B16可作为已有性能记录，B128是已密封容量筛查；B256/B512延续结果尚不存在。完成预筛后再决定是否研究1→2卡。Fortran、年度/669点50年、AD/训练与10×目标仍未验收；扩大真实点集合须先入场，再固定扩后集合比较batch。CPU费用不按核数线性外推。任何付费GPU延续都须独立审查与明确授权。

## 历史：第四轮免费GPU窗口

以下记录的是此前第四轮免费窗口，不代表当前付费结果。该轮B16完成、B64不完整，故不发布该轮B64吞吐；这不覆盖或否定上方已验收的付费GPU B64阶段。64个唯一真实点、嵌套16点前缀，连续两个7日块，
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
详见[专用汇总与raw samples](round4-measurement-summary.json)及
[执行审计](../../../docs/research/global_throughput/EXECUTION_AUDIT.md)。
[旧资产保全](gpu-round4/post-window-preservation-check.txt)21项hash均OK，
[运行后检查](gpu-round4/post-window-checks.txt)确认checkout干净、自身PID/PGID退出及GPU0释放。
第三轮记录保留；本轮不认证Fortran加速、年度/669点/50年、AD、训练、10×或跨CPU/GPU因果加速。

## 历史：第三轮免费GPU窗口

以下保留第三轮历史记录；其中的缺项描述代表当时状态，当前结论以文首付费GPU结果及专用汇总为准。

2026-10-05，`research/global-throughput`。第三轮单卡免费GPU窗口正常完成。**B4同步热core中位3.061469秒/56点日，同设备真实标量串行10.938078秒，3.572820×。** 已准备wrapper中位3.198821秒。24项已记录门槛全部通过，九个要求阶段均完整保存。

执行代码`cea0edf`，物理core保持`a33b82d`。固定真实点001/069/319/331，各自冻结Day1末态后连续推进Days2–8、Days9–15：56有效点日，FP64、每日48步、全部状态和日输出，内层`npts=1`、外层batch。容差仍为`1e-8 + 1e-10*abs(reference)`，schema/dtype/bool/int精确一致；仅源码支持的裸土`roughheight_pft` index0两类状态允许两侧完全相同NaN mask，其他NaN/Inf失败。最大逐元素组合容差占比0.045302235。

## 数值与热吞吐

当前B4程序对旧候选、四份各自context绑定并实际执行的GPU独立参考、置换及单lane替换隔离均通过。原子B4 checkpoint含760个输出叶子，每叶首轴4，以及十项新B4门槛、五次core和五次wrapper原始样本；SHA256匹配manifest。阶段在835.636729秒保存完成，随后835.724119秒才开始独立标量lower。独立只读审查确认了这个先保存后编译的顺序。

共享标量使用点069自身context，先对四独立参考通过门槛，再串行测量相同56点日。标量B1使用点001输入，另对该点独立参考和旧原AOT检查；这是14点日单点基线，不能当作独立vmapped-B1验收。

| 单V100测量 | 点日/次 | 五次中位秒 | 最小–最大秒 | 点日/秒 |
| --- | ---: | ---: | ---: | ---: |
| B4驻留输入同步core | 56 | 3.061469 | 3.056491–3.093887 | 18.291873 |
| 同设备真实标量串行 | 56 | 10.938078 | 10.924755–10.942314 | 5.119729 |
| B4已准备wrapper | 56 | 3.198821 | 3.180061–3.206820 | 17.506448 |
| 标量B1同步core | 14 | 2.735689 | 2.730862–2.737894 | 5.117540 |
| 标量B1已准备wrapper | 14 | 2.821319 | 2.798625–2.834932 | 4.962219 |

每次从同一冻结初态开始，两块内部连续推进并同步。wrapper包含已准备输入堆叠/传输和结果取回；core排除这些host开销。两者均排除新的Day1准备、生产CLI、年度汇总及生产checkpoint IO。

## 编译、启动与IO

新B4显式计时：host stack0.052592秒、device transfer0.177286秒、lower93.711854秒、compile670.043815秒、首次双块调用19.111908秒。新标量lower14.299802秒、compile623.917697秒。首次调用、编译和IO均不计入热core。进程内import/setup12.663389秒、准备输入读取/审计0.782106秒；取得并发布有效B4门槛/热测量约835.637秒。

B4 checkpoint1,749,591字节，序列化0.111594秒、原子写入0.011164秒；后续串行checkpoint7,674字节，序列化0.008312秒、写入0.004139秒。此IO计时排除manifest/report/event写入。

本轮实际B4 cache miss/write key`643afa8f…`，标量miss/write key`5d6b4815…`。选点069未命中第二轮旧key；准确key差异原因仍未建立。新完整缓存已保存，但未另开进程验证未来命中，也未执行新小组件probe。context选择或旧缓存存在均不能证明本轮命中。

## 资源与保全

Explore1000免费共享节点`ibc13b02n04`，1张V100-SXM2 32GB、GPU0、模型CPU affinity4–7；JAX/jaxlib0.4.38、FP64、driver525.85.12，UUID`GPU-daa2988f-72e7-d5e0-45e5-1bef9c1cc688`。启动前GPU0空闲，host load1.00/0.97/0.69；监督器自身记录0–55亲和性，没有Slurm分配或独占预约。

第三轮只有一个窗口，实际wall1608.972354秒，**记账1609/1800卡秒=0.446944卡时，费用0**。余191秒不授权第二窗口。window包含子进程启动、导入、输入复用/审计、lower/compile、门槛、计时、输出IO及退出；排除以前完成的Day1/参考生成、CPU准备及SSH/传输。第二轮3579/3600卡秒账本与首轮1200秒证据独立保留，不继承或修改额度。

监督器退出码0、无残余进程；后验核对准确执行session为空、GPU0释放至0MiB，GPU1其他研究PID11889未被操作。逐秒选定卡采样峰值809MiB，不是瞬时显存上界；进程峰值RSS10,573,844,480字节，进程组采样峰值10,793,304,064字节。子进程组采样CPU下界1808.34秒；回收子进程user+system1842.649891秒，监督器另219.414735秒，范围分别保留。

六份源科学checkpoint、源manifest、准备输入和第二轮budget SHA256均未变化；新九份完整checkpoint全部SHA256匹配。旧候选/原AOT/点001参考的缺失执行元数据仍明确标注unknown并保留原外部证据，不补造旧IO或设备字段。证据归档342,454字节，远端/本地SHA256同为`828e2b7ec8cbdffbdb0a2f9fe3f8af2b4d619d0a25b5550a074a071a7f764658`。新模型证据12文件/6,245,451字节、监督窗口7文件/414,966字节；复用缓存现176文件/38,706,607字节，非账户配额余量。

实现验证复用72fc47d的30项轻量测试；cea0edf重叠路径保护的5项针对性测试、Ruff/diff check通过。CPU-only合成账本验证第二窗口必拒绝且拒绝后账本字节不变。文档收尾未重跑模型或重复昂贵参考。

## CPU区别与缺项

原本地i7-13700/JAX0.10.1结果保留：B4 core3.442289秒、同56点日串行7.818091秒、**2.271190×**，wrapper4.014959秒。GPU **3.572820×**是单V100上的B4对同设备JAX串行；硬件和JAX版本不同，不能把CPU/GPU结果相除解释为纯硬件收益，两者也均不是相对Fortran的提速。

本轮仅完成四点、14个later-days的内部JAX有界forward诊断。仍缺独立vmapped-B1 GPU验收、生产端到端吞吐、耦合routing、全PFT/全分支、B16/B64、28日、年度/跨年、669点、50年、AD/训练及新的Fortran科学验收。没有开始训练或付费fallback。

证据：[当前汇总](milestone-summary.json)、[最终报告](gpu-round3/model/report.json)、[阶段manifest](gpu-round3/model/stage_manifest.json)、[单窗口budget](gpu-round3-budget.json)、[旧资产与进程核对](gpu-round3/post-window-checks.json)、[GPU释放复核](gpu-round3/post-window-cleanup.txt)、[目标shell实参](gpu-round3/target-dry-run.txt)、[CPU准备](gpu-round3-cpu-preparation-checks.json)、[实现检查](staged-implementation-checks.json)。历史：[保全第二轮汇总](round2-milestone-summary.json)、[保全第二轮里程碑](round2-MILESTONE.md)、[保全首轮汇总](round1-milestone-summary.json)。更新汇总只读证据：`conda run -n ORCJAX python outputs/research/global_throughput/summarize_gpu_round3.py`；旧轮脚本不再用于覆盖当前结果。
