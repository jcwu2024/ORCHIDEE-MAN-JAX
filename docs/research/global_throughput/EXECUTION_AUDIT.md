# 主分支物理模型执行审计与首轮基准设计

更新：2026-10-05。审计起点：`main@5910ba9af857a91e565d13a25e3de2e3019080d1`，
工作分支：`research/global-throughput`。本文件只审计 source-backed 正向模型；不依赖日尺度神经替代。
本轮保持科学过程不变，已实现编译前的外层 batch；本地 B1/B4 有界数值与计时通过。
本轮完成 Explore1000 免费 CPU 输入审计及GPU B4两块完整parity/置换/隔离和五次同步计时；
第三轮同设备同工作量JAX scalar比较与单点scalar基线通过，GPU B4 core收益3.572820倍；
没有Fortran/年度/生产端到端认证；已批准的付费服务器基准状态见下。
第四轮已完成Linux完整64点入场和两次免费GPU窗口；B16完成，B64不完整；不发布B64吞吐，最新结果见文末。

当前操作为已批准的[服务器CPU/GPU时间与成本基准执行](SERVER_BENCHMARK.md)，不是第五轮免费window。
主验收比较同16/64真实点、相同冻结输入和完整双7日块下的单物理核常驻scalar、
四个各占一核的持久worker、1/4核JAX batch及单GPU candidate。CPU affinity、物理核/SMT、
JAX初始化后的所有线程affinity和Slurm AllocTRES须记录；`vmap`不保证每lane一线程。
GPU先原子保全B64 candidate门槛与core/wrapper计时，再测B16，默认不编译新B4基线。
旧GPU FULL64参考是数值证据，CPU单核输出才是本次CPU执行对照。
保持科学core、原生dtype、容差、全部输出和224/896点日；无AD/训练或128/256点扩展。

费用主口径采用用户GPU卡/CPU核小时比44（GPU2.20元推算CPU0.05元），
部署文档CPU0.07元作保守上限，CPU/GPU相加是明确预算假设。
按allocated Slurm CPU和GPU时数计费；同工作量热费用与完整多阶段job费用分别记录。
用户已批准三个各最多一小时的pilot，原保守总额度2.83元。CPU1作业15579758运行852秒
后在B16 scalar第7点299.0-115.0的GPU参考门槛失败；叶子`[0][0][0][7]`界限占比
1.228488222708634。这是数值parity失败，没有OOM/guard证据；没有complete参考，B64
未运行，CPU4因此不提交，CPU1不自动重跑。/usr/bin/time子进程wall848.66秒、
user756.48秒、system39.82秒、peak RSS20,882,480 KiB；计费使用父作业wall852秒。
初始CPU/GPU启动均在Python前因Slurm AllocTRES/TRES字段差异失败，GPU替换15579759
在2秒内退出6、日志为空。具体身份查询原因尚未证实；修复部署到独立checkout，
后续58分钟作业15579869已排队。全部失败费用计入原额度，旧科学核心、容差及免费账本
保持原样；已有checkout不修改。权威记录为
[付费执行账本](../../../outputs/research/global_throughput/server-cost-preparation/PAID_EXECUTION_LEDGER.json)。
已结束作业合计876 allocated CPU-count-seconds及5 GPU-card-seconds，保守费用0.020088889元。
GPU跑满58分钟后全轮保守上限2.417422222元；排队不作为已消耗时长。当前对话每5分钟
静默检查已提交GPU的状态，终态后验证完整stage、更新报告并停止跟进；这不授权新增作业。
有界当前平台资产盘点未找到匹配Fortran binary及build配置，不作全账户不存在的结论。
历史2分钟/点年与本地i7计时均不能替代服务器CPU/Fortran实测。

第四轮精度补充：旧local4、remote4及新local16/local64两块输入均有相同四个group5
`float32`字段（`contfrac/lalo/lat/lon`）；其余已检查浮点输入leaf为`float64`。
冻结源中`lalo[:,0]`进入`season_midwinter_solad`，实际shape tracing结果为`float32`，
虽然后续`gdd_init_date`状态为`float64`。因此本文历史FP64运行标签不证明逐表达式
全FP64。主管已澄清本轮保持基线原生dtype、开启JAX x64；已知地理payload和solar子计算
含FP32。这是纠正先前主管措辞，未改变核心或dtype。确切原生dtype输入gate已通过35项
轻量测试及local16/local64生产reader；第四轮Linux入场改为运输原始冻结数值参数。
确切调用链、来源和执行合同见[PRECISION_BOUNDARY.md](PRECISION_BOUNDARY.md)。

## 首个实现的当前状态

纯工厂 `_paper_later_day_block_transition_factory` 原样提取原 `transition`，旧 scalar
AOT factory 继续调用它，capture/projector 契约保留。harness 从起点
`133e35aec7b83b9d79888c46092f64d518e97299` 的 Git source 载入原 AOT factory，
并检查提取前后 transition AST 完全相等。批处理变换发生在 `.lower().compile()` 前。
物理 core 冻结于 `a33b82da9a1233f38e00d09435aecde4ba088111`；独立参考 harness
为 `b8bb28574fdfd4ec335334c7f5ef936233464497`，最终通过的诊断 harness
为 `80831ea88b1107ba6d8cb1150792024788696b49`。最初本地进程在 core commit 前启动，
其 metadata HEAD 仍为 `133e35a`；运行中 commit 没有修改执行的源码文件。

`driver/global_batch.py` 独立生成每点九组输入。静态 signature 默认比较全部 context
科学值，附加 config 内容、run scalars/PFT identities、STOMATE static kwargs/parameters
扣除四组动态 vector、DIFFUCO 残余 static kwargs、vertical grids、完整 daily dispatch、
state layout 和矿质 bounds。raw run.def 未知差异拒绝；只排除已动态消费的四参数
键、选定 mangrove PFT 的 `ARJV` 和 host domain bounds。`ARJV` 并非新动态化：
`sechiba/diffuco.py:1989` 原已映射至 `trans_co2['arjv']`，九号参数已有该 leaf，
`driver/paper_binding.py:PAPER_POINT_BINDING_OWNERS` 明确其逐点 owner。

`RIVER_ROUTING=TRUE` 在此论文输入是合法静态值，未被改写。Fortran
`sechiba.f90::sechiba_main` 1227–1240 的条件是 `river_routing AND nbp_glo>1`；
内层 `nbp_glo=1` 保留显式 no-routing branch，JAX `stomate/entry.py:792` 同样拒绝
真正多点 routing。batch adapter 额外要求声明 `independent_single_landpoint`，
拒绝多点内部维度、TOPMODEL_NEW、未支持 soil-peat hydro 和缺少 compiled admission。

四点 001.0-071.0、069.0-119.0、319.0-057.0、331.0-057.0 使用每点自己
materialized run.def、restart template 和 cold Day1 state；没有广播首点模板。
本地只将已归档资产的 `VEGETATION_FILE` 路径正规化至声明 data root，科学参数不改。
初次审计保守拒绝 routing flag、第二次拒绝已动态 ARJV，事件记录保留，两者修复均来自
现有源码契约。第二次各点 Day1 及全部准备输入已保存，后续复用而不再初始化。

实测轻量证据：`tests/unit/test_global_batch.py` 13 项通过，新增模块/harness/tests
Ruff clean；现有 orchestration 存量 lint 未修改。四点 static/schema/dtype/None/day-number
审计通过：`outputs/research/global_throughput/local-audit-final/report.json`。
独立 preparation 178.15 s、峰值约 3.54 GB；其中各点 Day1 完成。
B1 原 AOT 与提取后两块完整状态/日输出已通过：760 leaves，有限值 max absolute=0。
证据 `outputs/research/global_throughput/local-b1-b4-v2/report.json`。
该过程在 B1 后主动结束以加强独立 reference，不能将其称 B4 gate 或吞吐通过。
最终完整结果 **passed_bounded_forward_diagnostic**：
`outputs/research/global_throughput/local-b1-b4-final/report.json`。其
`reference_origin_report.json` 保留 `b8bb285` 对四个点分别清空内层 cache、各自
context 编译/执行两块、每个 B4 lane 数值通过的证据；原过程在随后 JAX list-index
置换比较的诊断错误处停止。`80831ea` 将索引改为 NumPy index array，强制完整输入/
closure/baseline身份和科学 tree 无变更后复用独立参考；新实际共享 scalar 再对四个
独立参考逐点 zero finite delta，随后重新编译 B4、执行置换/替换/五次warm。
这两份报告的 source/阶段清楚分开，没有将缓存结果称作新独立 reference execution。
轻量记录存于
`outputs/research/global_throughput/implementation-checks.json`。

harness 数值参考分别由每点自己的 context/tables/dispatch 构造 scalar transition，
并在每点 scalar lower 前清空旧内层 `_COMPILED_SECHIBA_SCAN_CACHE`，记录 clear phase，
以免结构 cache 间接沿用首点 closure。生产 cache 未更改。
各自执行 Days2–8 与9–15；以完整末态和全部 daily outputs 检查 B1 原工厂/提取后、
B4 逐点、置换、完整 lane 替换。浮点组合逐元素 `1e-8+1e-10*abs(ref)`，
离散/schema/dtype 精确。只有两套明确 state component 的 `roughheight_pft`、稳定 bare-soil
identity 的 index0、预期内层 `[1,nvm]` 形状可使用双方严格同 mask 的 NaN sentinel。
此原有契约由 `condveg.py` 的 NaN 初始化和从 vegetation index1 开始的循环，以及
`tests/unit/test_condveg.py:418,486` 的明确 assertions 支持；源 provenance 是
`condveg.f90::condveg_z0cdrag` 1325–1474 与 `condveg_z0cdrag_dyn` 1578–1718，
DIFFUCO 4166–4169 在消费前替换 bare-soil 坐标。单侧 NaN、其他坐标 NaN、任意 Inf
和错误 shape 都失败，允许 sentinel 数目与路径逐项报告，没有全局 equal_nan。
参考/candidate 在 gate 前保全；routine harness 修复不会丢失已完成 reference。
计时记录 lower、compile、首次同步双块、至少五次冻结初态 reset 热双块（含 effects barrier）、
同样预 device 的 scalar serial、stack/transfer/materialize wrapper、host materialization
和完整集合输出 IO；环境、线程/affinity/cache、RSS 一并记录。短窗不含新 annual
Fortran 科学认证，不代表全部 PFT 或669点。

本地最终报告包含15项通过的数值 gates；最大 absolute delta 为
`1.103617250919342e-6`，最大逐元素容差占比为 `0.08166120891987796`。
Windows i7-13700（16物理/24可见逻辑核）、Python3.11.15、JAX/jaxlib0.10.1、
NumPy2.4.6、FP64；持久 JAX compilation cache 禁用。五次同步热运行结果如下：

| 同工作量 | batch core 中位数 [min,max] | JAX常驻串行中位数 [min,max] |
| --- | --- | --- |
| B1，14 point-days | 1.873 s [1.864,1.970] | 2.007 s [1.830,2.483] |
| B4，56 point-days | 3.442 s [3.216,3.551] | 7.818 s [6.924,8.827] |

B4 core 吞吐16.268 point-days/s，相对同工作量 JAX serial 为2.271倍；
已准备输入的 stack/transfer/materialize wrapper 中位数4.015 s。B4 lower23.575 s、
compile95.345 s、首次同步双块4.490 s；device transfer0.03556 s、host materialization
0.00470 s、完整集合输出 IO0.01766 s。独立四参考编译进程峰值 RSS7.72 GiB，
最终参考复用进程峰值5.30 GiB。wrapper不含新Day1准备、年度聚合或生产checkpoint，
不是公共CLI端到端速度。聚合证据为
`outputs/research/global_throughput/local-milestone-summary.json`。

Explore1000 免费有界诊断的证据记录如下：
`remote-cpu-audit-v2/` 的四点独立Day1、static/schema审计通过；wall267.49 s，
实耗CPU0.0809944 core-hours，四核affinity容量0.2972111 core-hours，
进程峰值6,391,112 KiB。它是共享test02输入审计，不是远端CPU吞吐基准。

GPU `b8bb285` 复用上述CPU准备输入，在gln01实际节点`ibc13b02n04`、
1张V100-SXM2 32GB、affinity0–3运行。原scalar reference在650.393 s保存，
extracted scalar compile在659.739 s开始；任务在1200 s边界结束，wrapper exit1，
未证实为GNU timeout124。`report.json`保留最后`running`状态，stderr仅见slow-compile
alarms，无数值异常；进程已退出，GPU compute processes为空。
尚无GPU batch parity或吞吐结论。预算为1/3 card-hour、四核affinity容量4/3 core-hours；
最后进程快照CPU TIME00:22:30给出实耗下界0.375 core-hours，VmHWM9,704,176 KiB
给出峰值下界9.254623 GiB。证据目录
`outputs/research/global_throughput/remote-gpu-b8bb285/` 保存report/events、launch context、
GPU memory samples、process metrics和stderr。后续launcher cache路径隔离修复发生在
此运行结束后，不属于该运行条件；本轮无付费作业。
1200个逐秒GPU样本覆盖上海时间00:04:23.237–00:24:22.647，节点card0峰值987 MiB；
该设备观测不证明完整batch编译或运行的显存需求。合并资源与验收记录见
[MILESTONE.md](../../../outputs/research/global_throughput/MILESTONE.md)和
[milestone-summary.json](../../../outputs/research/global_throughput/milestone-summary.json)。

## 有界GPU阶段协议与cache证据

`c8d118f`加入 `--mode staged --stage candidate|references|validate|all`：B4实际双块结果先保存，
再按四个点各自context/inputs清空内层cache、lower/compile/实际执行并逐点保存独立reference。
后续分别保存B4逐点/置换/替换gates、五次同步计时和B1基线。阶段checkpoint与manifest采用
atomic发布；完整stage凭schema/dtype/value/static closure/科学源码/runtime/device/compiler身份
复用，running/failed/stopped不计作完成。数值输入身份使用canonical tree/path/shape/dtype/value
hash，避免pickle对象共享关系改变旧hash；原b8 reference另按实际prepare正规化契约精确复核。

GPU首次阶段执行暴露JAX0.4.38晚配置cache失效：物理imports的早期JIT已把未设目录的
cache初始化记为完成，后设目录仍产生disabled/not initialized与不写入日志。原小组件命中
不能证明完整模型命中。`bdd5790`采用launcher早期cache环境变量配合显式配置后的公开
`reset_cache()`；该操作不删除磁盘资产、不改变IR或物理过程。实际test02/0.4.38轻量控制
在heavy import/首个JIT之后晚配置且无reset时不写cache；修复后实际miss/write，另一新进程
实际hit，runtime/device/StableHLO/output一致。证据为
`outputs/research/global_throughput/cache-init-repair-cpu-20261005/`。此CPU组件证据仍不能
替代修复后完整GPU模型cache write/hit、独立数值门槛或吞吐实测；实际后续证据如下。

`gpu-round2/model/stage_manifest.json`最终保全六个完整stage：B4 candidate、原001 AOT、
四点各自context且每次清空内层cache后实际执行的scalar reference。model1执行`c8d118f`，
model2/3执行`bdd5790`，物理core与输入保持上述冻结身份。最终15项已记录gates均通过：
保存的B4完整两块各lane对四份独立reference，最大absolute差`1.4908146113157272e-6`、
组合容差最大占比`0.04530223503755103`；共享001 scalar对四份独立reference有限值零差异，
原001 AOT与独立提取结果也有限值零差异，schema/离散/sentinel保持原政策。
第三window在重新编译B4时`over_budget`，未完成置换、lane替换隔离、五次warm或
`B1_scalar_baseline`。raw `B4_gates=running`因checkpoint尚未发布而保留，结合最终window
判为有效stopped；不改写raw记录，不称完整GPU gate通过，也无GPU throughput/speedup。

cache实际记录区别明确：model1 B4与001虽完成编译，均disabled/not initialized且没有写入；
修复后069 scalar miss/write，compile561.157389 s；319和331同key实际hit，compile/load
23.938720/23.909779 s。model3 pooled001使用另一key实际miss/write，compile562.801277 s。
参考三点raw IR hash为`5b8ab621…`，pooled001为`5e39342b…`；结构差异尚未解析。
直接核读部署JAX0.4.38的`cache_key.py`确认默认metadata关闭时先strip-debuginfo，key还包含
jaxlib/flags/options/accelerator等，不能声称callsite/source-line metadata已证明造成该miss。
第二轮结束时18 MiB的cache只有组件与两种scalar executable key，没有B4 executable。

B4 validation host stack0.054965 s、同步device transfer0.102866 s、lower41.493809 s；
compile约645 s后中断，没有compile完成时间或新首轮/热计时。旧B4 candidate lower91.254242 s、
compile683.750115 s、first call3.233149 s属于model1来源，完整状态已保存，不当作稳态性能。
新069/319/331 snapshot分别453575/453223/453223 bytes，序列化0.122514/0.090000/0.090293 s，
atomic write0.055236/0.008612/0.009638 s；不含manifest/report/events。旧三个stage的独立
checkpoint IO未记录，仍标缺失，不由总wall差值反推。

此前第二轮合同3600 card-seconds、每window≤1800，实际使用1张V100-SXM2 32GB，节点
`ibc13b02n04`、UUID`GPU-daa2988f-72e7-d5e0-45e5-1bef9c1cc688`、driver525.85.12，
JAX/jaxlib0.4.38、Python3.11.15、FP64。模型affinity4–7；supervisor可见0–55，不能将
四核affinity当作整个进程树的专属核数。probes实际wall13.350996/2.632614 s、charge14/3；
model1 wall1566.882998 s、charge1567，model2 wall694.176905 s、charge695，model3
wall1295.341835 s、按timeout reservation保守charge1300。总wall3572.385347 s、charge3579
card-seconds=0.99417 card-hours，余额21；未增加预算或启动付费任务。window计费包括
启动、复用输入读取/审计、编译、reference/gate执行和IO，原Day1与之前AOT编译另计。
model3 supervisor曾记录survivor89971；后查上海03:21:25已无该PID及89970 PGID/session，
GPU0释放，只见另一研究线程GPU1 PID84592，明确是后续cleanup证据，非即时无残留。
最终report的`elapsed_seconds=650.621450`是进入未完成compile时最后写入值，不能替代
supervisor完整wall。证据在`gpu-round2/model/`各origin report和manifest、
`gpu-round2/windows/`各稳定report/launch-status/采样、`gpu-round2-budget.json`与
`gpu-round2/post-window-cleanup.txt`，均位于本worktree `outputs/research/global_throughput/`。

默认B1为另编译的vmapped路径。有限预算下可明确指定`--b1-mode scalar`，使用已逐点对齐
独立reference且与原AOT核对的单点程序，单列`B1_scalar_baseline`完整14 point-days的五次
冻结起态热core/preprepared wrapper；它不认证独立vmapped B1 GPU路径，两种模式采用不同
完成stage键，不改变共同stage的数值身份。不得用CPUreference替代GPU独立reference。

本地修订`f1df818`提供`--pooled-scalar-point`，默认仍001。选择其他点时，由该点自己的
已审计context构建实际共享scalar；任何新构建的pooled executable必须重做四份独立
reference gates，包括只恢复B1的场景。selector改变时仅gate/timing/B1阶段换键，已完成
candidate及独立reference继续复用；execution origin记录selector，不能误复用另一selector
的测量。该补丁只通过本地30项轻量测试（4.56 s，1项禁用cacheprovider配置警告）、
变更harness/test的Ruff与diff检查；第二轮结束时尚无该补丁的GPU验收，第二轮最终代码
`bdd5790`没有中途部署该修订。不因selector对应已有cache文件就宣称实际命中。

第三轮修订`72fc47d`纠正validate执行序：新`B4_batch_only_v1`先且仅构建一个B4
executable，首次同步完整输出对保存candidate及四份独立reference，再通过置换和lane替换。
全部B4 gates通过后才记录至少五个56 point-days core及preprepared wrapper原始samples，
将samples、完整两块输出和gates共同atomic checkpoint；此前不编译独立scalar executable。之后
`B4_scalar_comparison_v1[_pool_<point>]`构建实际scalar、重验四份reference、测同56 point-days
串行，最后14 point-days scalar B1基线（显式scalar模式），分别保存，任何后续scalar失败或
预算结束不抹除B4已完成资产。B4 first-call/lower/compile与wrapper/transfer成本仍分别列。
这套新stage不能由旧`B4_gates`/`B4_measurements`记录满足；B4 stage不依赖scalar selector。

`--stage-source-dir`只供staged validate向新目录导入上述六份完整数值snapshot，不创建
可写source StageStore。先核对整个bounded import的identity/checksum/status，再atomic复制
原字节和完整来源；旧源manifest、report、ledger及execution origins不改。轻量真实JAX
测试覆盖新目录只编译一次B4、立即存好完整gates/output/五个core+wrapper samples后注入
scalar失败，确认B4完成record仍可恢复、B1未启动且source逐字节未变。30项测试5.21 s通过，
1项禁用cacheprovider警告，Ruff/diff检查及CLI help通过；没有重跑昂贵物理CPU gates。
`cea0edf`进一步在main输出/cache setup及StageStore创建前拒绝同路径或任一嵌套路径，
使无效CLI也不改源manifest/report。5项定向guard/实际小JAX recovery测试4.68 s通过、
28项未重跑、1项禁用cacheprovider警告，Ruff/diff检查通过；原30项证据复用。

第三轮实际`cea0edf`在相同V100/0.4.38/FP64和模型affinity4–7中，只读导入六份数值snapshot，
原始执行origin保留。新B4对保存candidate有限值零差异，对四份独立reference、置换及lane
替换的10项新gates均通过，组合容差最大占比0.04530223503755103、max absolute
1.4908146113157272e-6。core五次samples为3.061469/3.079388/3.056491/3.093887/3.060239 s，
中位3.061469 [3.056491,3.093887] s/56 point-days，即18.291873 point-days/s；preprepared
wrapper中位3.198821 [3.180061,3.206820] s，即17.506448 point-days/s。全部重复从冻结Day1
末态推进两个7日块并同步完整输出/effects；wrapper含stack/transfer/materialize，均不含新Day1
准备、年度聚合、生产checkpoint或独立reference编译，不能称公共CLI端到端速度。

新lower显式timer93.711854 s（phase跨度95.020494 s还含报告开销）、compile670.043815 s、
first call19.111908 s；host stack0.052592 s、同步device transfer0.177286 s。实际cache
miss/write为`jit_transition-643afa8f…`，不称命中。`B4_batch_only_v1`的原始samples/gates与完整
两块输出在elapsed835.6367 s atomic保全，snapshot1,749,591 bytes、序列化0.111594 s、
atomic write0.011164 s，不含manifest/report/events；这是新测量，非旧candidate的3.233 s首轮。
随后显式选择069 scalar，lower14.299802 s、compile623.917697 s；实际miss/write key
`jit_transition-5d6b4815…`、raw IR`c36fbc403…`，未复用旧069 reference的`6f26ea90…`key。
此前相同context选择不足以证明IR/key相同；结构根因未解析，保留旧IR/key与默认metadata
strip-debuginfo事实，不把miss解释为已证实的callsite原因。实际scalar对四份独立reference
有限值零差异；同56 point-days串行五次samples为10.940952/10.936632/10.924755/10.938078/
10.942314 s，中位10.938078 [10.924755,10.942314] s，即5.119729 point-days/s。B4相对
该同GPU、同输入、完整物理过程FP64串行core为3.572820173243233倍。单点scalar B1的
14 point-days中位2.735689 [2.730862,2.737894] s、wrapper2.821319 [2.798625,2.834932] s，
对独立001/原AOT有限值零差异；这不是独立vmapped B1路径验收。

最终`passed_bounded_forward_diagnostic`，全部9个required stage complete，24项记录gate
passed（包含7项复用来源gate）；三份新stage与完整outputs/raw samples保全。唯一免费window
完成wall1608.972354 s、charge1609/1800 card-seconds=0.446944 card-hours，无重试/付费。
余额191不开放第二window；report elapsed1581.245527 s不替代supervisor包含启动/包装的wall。
进程peak RSS10,573,844,480 bytes，group采样peak10,793,304,064 bytes；四核模型affinity
不是整个supervisor进程树的独占CPU资源。GPU0逐秒采样峰值809 MiB，非瞬时峰值保证。
`post-window-checks.json`全部通过：旧manifest、
六source checksums、prepared input及round2 budget不变，本session无进程，supervisor无survivor。
后查GPU0零MiB、GPU1其他线程PID11889保留，remote checkout仍`cea0edf`且clean。
实际证据为`gpu-round3/model/`report/manifest/三份新stage，`gpu-round3/windows/model-1/`
稳定report/launch-status/采样，后查在
`outputs/research/global_throughput/gpu-round3/post-window-checks.json`，ledger在
`outputs/research/global_throughput/gpu-round3-budget.json`。

每个新stage保存execution origin（Git/runtime/cache/affinity/launcher物理GPU UUID与driver、
PID及稳定window report），`timing_origins`明确标识复用测量并保留其原环境。旧stage未记录
的来源明确缺失，使用原report与supervisor context核对，不能反填当前环境。完整window
wall包含启动/import、输入读取审计、编译/reference/gates/计时、报告/checkpoint与包装成本；
复用的先前Day1及原reference成本分列。热core与preprepared wrapper的范围不扩张为
公共年度CLI或新cold-start端到端性能。独立stack/device transfer和checkpoint序列化/atomic
write/fsync/replace指标只对新执行的logging版本有效，旧stage未单测IO保持未知。
30项轻量测试及cache/恢复/身份/origin证据见
`outputs/research/global_throughput/staged-implementation-checks.json`；未重跑昂贵本地物理gates。

## 结论及证据边界

现有代码已经提供完整 later-day 多日纯数值扫描的实现基础，最小批处理落点是
`jax_orchidee/driver/orchestration.py:11231` 的内部 `transition`，而不是公共 CLI。
本轮已在 `.lower().compile()` 之前暴露这段函数，保留单点内部维度，以外层 `vmap`
装入异质网格。本地B1/B4短窗数值门槛和CPU吞吐诊断已通过；第三轮GPU B4两块完整
parity/置换/隔离及五次热core/wrapper和scalar对照完成。本地CPU收益2.271190倍，
GPU同设备同工作量收益3.572820倍，分别对JAX串行；不能外推为Fortran或全年生产加速。

论文 mosaic 的合同是 669 个独立单点任务，不是开启河道路由的一个全球耦合域：
`configs/orchidee_man_250919.yaml:40-51` 明确声明
`execution_unit: independent_single_landpoint`。这是本研究跨点并行的科学前提。
把每个单点的 `npts=1` 保留下来再加独立 batch 轴，可以避免未经验证地改变 grid geometry、
邻居语义和跨点 reduction。若扩展到 routing、横向交换、其他 PFT 或配置，应重新审计，
不能凭当前 669 点合同声称任意全球域独立。

代码位置以下均相对于本分支工作树。原始 `outputs/` 证据本次从
`E:/ORCHIDEE-MAN-JAX/outputs/performance/` 只读核查，没有复制大型资产；这些文件 Git-ignored，
新 worktree 本身不包含它们。文档证据与现行代码不一致时，以当前代码描述执行结构，
原始记录仅描述对应历史实验。

知识图使用 Verify tier；父任务提供双向调用和 coverage 证据，本任务再次确认 project
`E-ORCHIDEE-MAN-JAX`、generation `2026-10-04T14:30:28Z`。图索引根仍是另一个研究分支，
`runtime/worktrees` 被明确排除。对本审计代码和 tests scopes 的 coverage 结果为
`excluded/not_tracked`，因此全部关键主张采用新 worktree 直接源码回读，未把根工作区符号
当作主分支实现，也未重新索引以干扰另一任务。

## 当前执行边界

| 边界 | 当前实现及直接源码 | 对批处理的含义 |
| --- | --- | --- |
| 公共入口 | `jax_orchidee/cli/main.py:59-63` 注入 compiled SECHIBA on、block size 7；`runners/acceptance.py:48` 同样指定 7 | 公共入口已使用七日块；低层 runner parser `runners/multiyear.py:478-481` 的默认仍是 0，不能将两者混淆 |
| 初始化与每年 Day1 | cold/restart/multiday 三类 runner 先完成 Day1，再调用 `_paper_run_compiled_later_day_blocks`；例如 `orchestration.py:11621-11639` | 首日、年交接、IO 仍由 host 驱动；50 年不能当成互不相关年份并行 |
| 完整多日块 | `_paper_compiled_later_day_block_executable`，`orchestration.py:11185-11355`；内部 `lax.scan` 在 11318 | scan 顺序推进时间，跨点 batch 才增加并行轴 |
| 完整单日 | `paper_1961_driver_later_day_runtime_result`，10748-11182 | 48 步 SECHIBA → daily fold/maintenance → 半小时 OK_LEAK → STOMATE daily carbon → outputs/reset/surface/season → full state writeback；不是只有光合或水文的简化内核 |
| 日内 SECHIBA | `_paper_compiled_sechiba_scan`，10272-10453；first compact step + 47-step scan，10509-10689 | 完整块内 first compact step 也被 outer JIT 包含；块外日路径仍有 host wrapper |
| 输出与持久化 | block host wrapper 11358-11471；`runners/multiyear.py:684-710` 同步、年度汇总、checkpoint | Python daily slicing、年度聚合和文件写入应计入端到端结果，分别计时 |

块函数动态参数是九组 PyTree：

1. `values_by_component`：完整 state，字段名/layout/provenance 分离在 `DriverFastStateSpec`
   中；`driver/fast_state.py:69-87, 157-184`。不是只携带输出变量。
2. `forcing_days`：`DriverCompiledHalfHourForcing` 的 14 个叶子
   (`zlev/zlevuv/u/v/qair/temp_air/pb/precip_rain/precip_snow/lwdown/swdown/ccanopy/salinity/tide_height`)，
   每个叶子按 `[days,48,...]` 堆叠，保留原有单点维度。
3. `science_day_numbers`：真实日序，不能用编译模板的 `day_index=2` 替代。
4. `stomate_parameters`：`alloc_min/residence_time/vcmax25/maint_resp_slope` 四个连续参数向量。
5. `hydrol_table_arrays`：10 个矿质 CWRR table arrays，`orchestration.py:10212-10225`。
6. `landpoint_payload`：地理、土壤、cover 等 non-forcing numeric payload。
   字段生成规则在 10095-10109；`kjpindex/nbindex/missing_fields` 和 Driver aggregation 的
   `soilclass_sub_index/soilclass_sub_area` 被排除。payload 原定义在 `driver/sechiba_boundary.py:21-61`。
7. `stomate_restart_template`：逐点 STOMATE restart entry template。
8. `stomate_season_values`：逐点季节 template 数值，provenance 不进入动态 PyTree。
9. `diffuco_parameter_values`：`trans_co2` 的数值参数（排除 `ok_laidev`）和 `rstruct_const`。

以上由 `orchestration.py:11231-11241, 11324-11349, 11435-11445` 明确传入。
代码将 `year=1961/start_tstep=48/day_index=2` 作为结构 metadata 留在 closure 中；
科学时间由实际 forcing、CO2、`science_day_number` 输入传递（11253-11256、10810-10814、
11021、11055、11114）。这解释了跨年份复用的设计，不证明任意新增时间分支已经受此机制覆盖。

## 静态配置、异质点与缓存

连续参数不是全部都动态化。`context`、`daily_carbon_dispatch`、`config_path`、
state layout、矿质表 `imin/imax` 及多类 run-definition 控制在 closure 中。
`_paper_daily_carbon_static_dispatch`（7642-7675）还固化自然/草地/tree/peat masks、
phenology 模型、active-PFT mask，以及连续的 `tmin_crit/tcm_crit` tuple。
`_paper_later_day_slowproc_surface_update_from_runtime`（7768-7796）仍由 closure 中
run.def 解析 `EXT_COEFF_VEGETFRAC` 和 `STOMATE_OK_DGVM`。
因此“相同 shape”或“都是 PFT14”不足以保证能够合批；未列为实参的地点差异必须
确认确实一致，或在后续经过源码审计后动态传入。

外层 cache key（11217-11226）含 config **路径**、block size、mineral bounds、
state component/field layout、capture/projector 标识，没有纳入所有 closure 配置值。
内层 SECHIBA cache key（10290-10313）较详细，包含 timestep、freeze/snow/peat/PC/leak/tides、
PFT masks 与表结构。两者最多保留八个项（11350、10450）。这不是任意 run.def 改动后的
自动科学兼容性检测。首轮需从科学静态配置建立明确的兼容组，逐点核对 closure 的所有
科学有效值；缓存键硬化是候选后续工作，不应把旧跨点复用证据扩张为任意配置复用。

最小候选形态是 `jit(vmap(single_point_transition, ...))`，在 AOT 编译前完成变换：
state、forcing、逐点 parameters/tables/payload/templates 增加轴 0；相同科学日序可使用
`in_axes=None`。forcing 的内部轴仍是 `[D,48,...]`，外层为 `[B,D,48,...]`。
共享的真正静态配置只在组内复用。PyTree 每个 leaf（包括 bool/int、标量、可选 None）
须指定相应轴，不能简单把第一点 template 广播给全部点。

起点工厂返回 `jax.jit(transition).lower(...).compile()` 的 AOT executable（11324-11349），
当时不暴露原 `transition`；本轮已在编译前提取/复用原函数工厂。直接变换 AOT executable
不是本研究可依赖的路线。本地 `B=1` 输出/完整 state parity与异质 `B=4`已通过，
更大 B仍需独立门槛。直接把 `npts` 改成 B 是另一条路线，需要额外审计所有 reduction、grid、static
adapter 和 restart shapes，不能将已有 readers 的多点能力当作完整过程支持证明。

## Host 成本与不支持路径

forcing builder `orchestration.py:10131-10193` 在 host 调用 cached source-backed forcing reader、
年度 CO2、NumPy FP64 reshape/stack，并保留压力单位、降水乘 `dt_sechiba` 等 source conversion。
块 wrapper 11406-11419 逐日生成 forcing，再 stack block；11447-11460 按日 slice 输出。
`materialize_compiled_entries=False`（11273）使完整块避免单日默认路径
10664-10665 的 entry stacks `np.asarray` host materialization。
因此核心 GPU 时间、输入准备/拷贝、host 输出和完整任务时间必须分别报告。

年度 runner 明确 `_block_until_ready(run)` 后再停止年度计时（multiyear.py:684-685），
此年度计时不包含随后的年度汇总/checkpoint IO；总体 `elapsed_seconds`（755）更宽。
其数值汇总使用 `jax.device_get/np.asarray/float`（127-140）。restart state cache 使用
pickle（390-427），不等价于三文件 NetCDF restart roundtrip 的验证。
持久 compilation cache 由 `runtime.py:9-29` 放在 `ORCHIDEE_OUTPUT_ROOT/xla_cache`，
`ORCHJAX_DISABLE_COMPILATION_CACHE=1` 可禁用。记录 cache 冷/热状态，不能把磁盘命中混称冷编译。

已知 compiled admission 有严格范围：48 步日、JIT 开启、无 trace override、daily caches/tables
就绪、`not hydrol_soil_peat_hydro`、不用独立 fast-state loop（10489-10505）；
outer block admission 检查 compiled/JIT/no-trace（11563-11573）。
`PAPER_1961_HYDROL_SOIL_PEAT_HYDRO=False`（75），table wrapper `peat=None`（11271）。
不能为了批量吞吐把不支持的 soil-peat hydro、其他 PFT/config 或不完整 state 包混入同一组。
单日保留 strict fallback（10690 起）并返回 missing/stopped 诊断；batch prototype 应记录每点
ready/missing 和实际路径，不能把 fallback 后结果算作 compiled throughput 成功。
稀疏未完成批次如需 padding，首轮可重复合法点后在输出统计中排除 dummy lane，不能声称这是
Fortran mask；必须核对有效 lane 不受影响，记录真正点数及 padding 开销。

显式积雪本身不是被省略的过程；compiled SECHIBA 执行 explicit-snow 路径。
但当前测试点数/窗口不覆盖全部 mask 或 branch。维持现有 source-driven Fortran branch ledger，
对异质样本选择说明触及哪些现有分支；不能把随机运行不触发当作分支不存在。

## 已核读的最新主模型历史证据

| 原始资产（均位于父工作区 `outputs/performance/`） | 可引用事实 | 限制 |
| --- | --- | --- |
| `compiled_multiday_block_acceptance_20260719.json` | Windows CPU；compile+首个365日 158.1315 s；hot365日31.1364 s，0.0853052 s/day；建议 block7；包含完整过程与末态 | `passed_candidate`，报告当时尚 default-off/warning-policy pending；无精确 commit、CPU/JAX/线程元数据；25.947 min/50年是预测，不是实测 |
| `compiled_block_gate_cold_14_dynamic_20260719.json` | cold14、block7、strict_source_order 对照；762 float/10 discrete leaves；固定容差通过 | 总 elapsed219.13 s含baseline+candidate，不是候选吞吐；单点 |
| `compiled_block_gate_annual_365_dynamic_20260719.json` | annual365、block7、compiled_daily_no_block 对照；11292 float/10 discrete leaves；固定容差通过 | 总 elapsed368.07 s含比较双方；不是独立 Fortran 对照 |
| `compiled_block_gate_restart_14_dynamic_20260719.json` | restart14、block7、compiled_daily_no_block 对照；763 float/8 discrete leaves；通过 | `verify_compiled_block_semantic_gate.py:117-155` 从1961的14日state重基后跑1962；是有界handoff测试，不是真实365日末态的三文件roundtrip |
| `complete_day_landpoint_reuse_20260719.json` | 001.0-071.0、069.0-119.0、319.0-057.0；3日总窗、block2；outer_cache_entries始终1，短窗state/modelout容差通过 | 不是block7三点全年gate，且不是跨点同时batch证据 |
| `multiday_block_lowering_probe.json` | 不同两个2日block复用同一个executable；469 leaves；末态/modelout容差通过；hot block约0.152/0.155 s | 单点内部块测量，不能当公共CLI或IO吞吐 |
| `compiled_day_stage_timing.json` | 完整单日best hot0.0737774 s，SECHIBA-only0.0407181 s；inline-SECHIBA较慢并撤回 | 组件/单日证据，不能自动推断批量GPU瓶颈 |
| `compiled_fast_path_acceptance/final_3point_annual_compiled.json`（及readiness文档） | 旧单日scan三点年度A/B、full state、1962 continuation与strict/compiled NetCDF roundtrip | 79.33–94.16 s年速率属于旧单日scan，不能代表现行完整块 |

float gate 使用逐元素组合 `abs(delta) <= 1e-8 + 1e-10*abs(reference)`，离散/schema 精确；
不能因为大数值 `pcapa_en` 的 max absolute 大于1e-8就误判失败，也不能只看全局 max relative
忽略某个 leaf 超限。主分支已保留 forcing exact transport、五日block parity和 restart-block
测试（`tests/parity/test_driver_1961_orchestration.py:146,2277,2378`），本次回读而未重跑。

`docs/source_audits/current_performance_notes.md` 最后停在 July13 多日可行性失败；
`compiled_transition_state_ledger.md` 和 `fast_state.py` 开头也含历史 staging 描述。
`compiled_fast_path_production_readiness.md` 主要描述旧单日scan。
这些文档不应覆盖 July19 原始记录和当前源码。README/`docs/running.md` 当前七日默认
由公共 CLI 源码证实，但本轮没有把历史 acceptance 重新认证为 `5910ba9` 环境的新实测。
后续提交还包含 PFT identities 和物理 Teacher capture 的源码演进，进一步要求锁定当前基线。

跨点复用与每点独立进程建议不矛盾：前者证明某些结构兼容输入不会增加 executable，
后者隔离 XLA内存、故障和checkpoint；README记录过 Windows 多点顺次编译失败。
新的常驻串行/批量任务是探索 Linux/当前运行时的复用路线，需要测RSS、cache增长和稳定性，
不能仅由“cache一项”推出669点长任务稳定。

## 首轮可复现诊断合同与完成范围

以下保留立项合同。本轮本地已完成阶段0、1及阶段2的CPU部分；远端阶段0通过，
首个GPU诊断在scalar编译预算边界停止；第二轮stage协议保全四份独立reference及B4逐lane
parity；第三轮新B4完整parity/置换/隔离、热计时、scalar比较和scalar B1基线全部通过。
阶段3、4和Fortran同硬件比较尚未执行。

先冻结 `5910ba9` 基线及候选提交；保存命令、Python/JAX/jaxlib/CUDA版本、设备型号、
FP64设置、XLA选项、CPU核数/亲和性、OMP/BLAS线程、实际node/load、GPU进程、输入配置与
选点顺序。输出只放本分支 `outputs/research/global_throughput/<run-id>/`。
reuse 已有 forcing、restart 和已完成reference；不要为每点每日至每变量生成独立文件。
FP64在现行process模块显式开启（例如 `sechiba/hydrol.py:16`），实验仍须记录运行时dtype，
避免环境覆盖。JAX `0.10.1` 是项目声明依赖（`pyproject.toml:15-16`），不是历史CPU报告版本。

| 阶段 | 相同工作量及对照 | 通过条件/输出 |
| --- | --- | --- |
| 0：绑定 | 已知三点001/069/319加至少一个其余真实点；每点自身materialized run.def、forcing、state/templates | static兼容审计及input identity，禁止把第一点连续参数广播给其他点 |
| 1：B=1 | 提取纯transition前后；固定D=7、完整48步+daily chain；同一Day1末态、Days2–8，再连续下一块 | 全部输出与完整末态/masks/schema parity；core warm与host成本分离 |
| 2：异质B=4 | 同四真实点单点常驻串行 vs CPU batch；同批再测单GPU；固定2块=14 later-days | 逐点全部末态+日modelout、置换batch顺序、替换一条lane只影响该lane；路径/缺失标记一致 |
| 3：B=16/64 | 原合同不变；足够多真实异质兼容点；逐级扩大，内存/编译越界则记录并停止扩大 | 同工作量point-days/s、编译/执行峰值内存；重复点只能另列device-load诊断 |
| 4：代表性年/跨年 | 在已胜出的规模上跑真实1961末态→1962，冷/重启入口及independent NetCDF roundtrip | 12 annual history fields、四paper outputs、full state、direct/split-restart一致；随后才扩669/50年 |

首轮性能主窗为**从同一冻结初态重置**的两个完整7日块；另有连续不同块检验，防止重复
相同forcing/state偶然通过。冷实验使用新进程和独立空cache/禁cache；记录import/setup、
input preparation、lower、compile、首个同步执行。磁盘cache重开进程单列；warm实验预先
device_put并同步输入，至少5次重置起态测量，计时结束对完整末态与所有输出
`jax.block_until_ready`，如有effects则barrier。报告中位数、min/max/波动，首个call单列，
不得把异步dispatch enqueue当runtime。核心计时之外再测包含forcing/transfer/output的同工作量
wrapper。默认测试7日；只有正确性和编译预算通过才追加28日，不先构造整年巨型图。

短窗公开入口基线的可复现命令形态（仅供后续执行，当前未运行）：

```powershell
Set-Location 'E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput'
$env:ORCHIDEE_REPO_ROOT = 'E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput'
$env:ORCHIDEE_DATA_ROOT = 'E:/ORCHIDEE-MAN-JAX/data'
$env:ORCHIDEE_REFERENCE_ROOT = 'E:/ORCHIDEE-MAN-JAX/reference'
$env:ORCHIDEE_OUTPUT_ROOT = 'E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput/outputs/research/global_throughput/baseline'
$env:ORCHJAX_DISABLE_COMPILATION_CACHE = '1'
conda run -n ORCJAX python -m jax_orchidee.cli.main run --landpoint-id 001.0-071.0 --start-year 1961 --years 1 --days-per-year 15 --initial-state cold-start --compiled-sechiba-day on --compiled-day-block-size 7 --output outputs/research/global_throughput/baseline/point001-cold15.json
```

此命令测Day1+两个later-day块，**不提供常驻warm测量或batch测量**；公共年度聚合只是
15日窗的结果，不能拿它与Fortran365日annual truth作科学年度验收。批量诊断现由
`scripts/dev/benchmark_global_batch.py`提供，复现命令见本目录README；公共生产batch CLI
尚未提供。正式年度gates沿用现有semantic/acceptance harness。

资源账本同时给出 `valid_point_days / seconds` 和真实完整年的 `point_years / seconds`；
CPU allocated core-hours=`allocated cores*wall_seconds/3600`，GPU card-hours=
`allocated cards*wall_seconds/3600`，CPU辅助资源另列。若只有短窗，把年/669点/50年时间
明确标为projection；startup、编译、IO、tail/padding、checkpoint须计入实际总任务cost。
多卡提升单独记录卡数与效率，不能当作单卡算法收益。

Fortran公平基线使用相同论文源码版本、真实点/run.def、forcing、初态、时步与FP64，
同节点/资源固定的可比CPU测试，固定优化编译flags和线程数；分别记录kernel/无诊断输出
吞吐与两侧相同科学输出/checkpoint的端到端耗时。比较serial/常驻和多任务并行需同资源
预算。JAX scalar到batch的收益先与JAX同点串行比，再与Fortran比，不能把旧strict JAX的
21.03x当Fortran speedup。若只能取不同硬件历史Fortran时间，单列描述，不发布已验证加速比。

端到端通过后也不把当前sample当全分支覆盖；Fortran科学门槛沿用
[paper_scientific_acceptance_policy.md](../../source_audits/paper_scientific_acceptance_policy.md)：
annual四指标逐点年 `1e-5+1e-3*abs(reference)`，series normalized bias/NRMSE≤1e-3，
缺失/非有限/停止视失败。性能路径内部A/B继续使用更严的现有fast-path gate，不调宽容差。

## AD 边界：有能力基础，未证明公开端到端可微

动态state/forcing/连续参数和`lax.scan`为在数值层定义JVP/VJP提供了结构基础；
当前公开runner含文件读写、Python配置、NumPy、float与pickle转换，不是可直接`grad`
的端到端函数。AOT executable也不是AD变换入口；应在编译前对原transition定义scalar loss，
再变换/编译。四个STOMATE vectors和DIFFUCO/HYDROL inputs的动态传入，不证明所有待校准
参数都拥有梯度路径（closure值尤其需审计）。

main已经有自定义精确primal AD primitives：`ad_primitives.py:14-49` ratio在极小分母下
屏蔽不可表示tangent；`:57-86` sqrt在0取零subgradient；有batching规则和局部gradient测试
`tests/unit/test_ad_primitives.py:13,33`。这类约定不是“全部物理分支数学上光滑”的证明。
离散逻辑、阈值、mask和min/max切换需定义可微范围；full scan梯度是否有限、正确、长期稳定
及50年内存成本仍需单独loss/参数合同和方向导数证据。本审计没有执行whole-model AD测试，
也没有把日尺度研究的梯度验证转用于main公开物理runner。

后续首轮只测forward，不为AD额外保存反向轨迹。参数校准/同化的梯度收益与纯forward
吞吐是两个实验问题，不因使用JAX而自动改善；不应启动长期反向图作为batch可行性的前置条件。

## 本轮核查记录

第四轮scale执行边界：`benchmark_global_scale.py`对所有16/64个唯一点复核两块完整输入，
共享B4的前提是所有lane的科学static signature与schema/dtype/None严格兼容。
本地完整64输入及新60点冷状态已保存；B16独立输入文件保留。首次Linux入场及单点
诊断已执行，冻结参数修复后的完整64点生产reader入场已完成。实际B16生产reader曾在旧anchor
`001.0-071.0/group5/contfrac`的原有float32 leaf触发原逐leaf全FP64输入拒绝；没有把
错误归因于新点。gate现严格要求四条已知原生float32路径的确切schema，其余既有
浮点、离散、static、身份及非有限值检查不变。
原四点numerical reference只读导入；当前常驻B4对其完整输出和各点独立reference重验，
其余每组运行自身输入，不新增逐点scalar编译。B16/B64输出逐lane对齐本轮实际B4组输出，
计时baseline也对齐同一组冻结输入，不使用历史四点scalar的比例外推。
完成checkpoint包含完整输出、原始samples、gates、执行及测量origin；恢复保留完成B16，
未完成B64明确标为stopped/failed。full-input/science/runtime身份包含batch stage名，
不包含requested-size子集，因此先16后16/64可以复用已完成B16。

本地旧CPU reference导入显式选择`--reference-format local-cpu --anchor-prepared-input`，
核对记录runtime/compiler环境、原source Git科学差异、全部原输入canonical identity及每份
legacy reference hash，不重新编译scalar；旧记录缺少backend version及执行时科学hash时
明确披露，不把当前环境反填为历史证据。GPU默认staged格式要求原manifest完整身份/checksum。
portable reader只恢复pathlib metadata到当前host，不随意改路径或数值；声明config与context
config bytes、磁盘run.def与context values必须匹配，再重新计算当前static signature。

Linux准备脚本消费local audit、已存Day1的host NumPy数值状态及完整冻结BlockArguments，
核对v2资产、config、
state和两块完整数值argument digest，允许已声明VEGETATION_FILE路径映射；历史路径若已
不存在，只作为归档provenance，实际local文件hash必须等于canonical remote目标文件hash，
且目标字符串必须与remote四anchor run.def相同。remote完整64输入实际经生产reader再审计，
这仍不是GPU数值gate。首次112.716639s入场和30.850098s单点诊断已保全；215的两块
536-leaf输入逐路径比较有16个真实FP64低位差异，cold state/layout相同。完整数值输入
直接运输并逐字节检查，避免跨运行环境重新生成forcing/查找表，未修改任何过程或容差。
首个运输后入场窗口118.289638s在NEW60与旧remote4的state字段排列兼容性处失败；
此前首点原始frozen argument digest与自身Linux context检查均通过。仅93字段的
`sechiba_finalize_state`排列不同，不能只换字段标签而保留错误的位置关联。后续使用已有
值保持布局迁移及具名字段一一映射，只排列NEW60的state argument组；先核对源digest，
再记录绑定digest及每个字段dtype/shape/bytes等价证据。其余八组输入及原remote4保持原样。
后续入场内部上限1385s，外部window至多1435s；加上三次已用向上取整263s和100s预留
housekeeping，最多1798s，仍受原1800s/4-affinity免费CPU总额度约束。physical GPU window由原独立
3600card-seconds/最多两次账本管理，准备成本单列。

已直接阅读上列主代码边界、CLI/runner默认、forcing/parameter/state adapters、static dispatch、
AD primitives及相关测试实现；核查七日块三份dynamic gate、三点block2复用、多日/组件性能
原始JSON；检查base HEAD。初始来源审计未重跑历史昂贵gate；后续新增了四点独立输入审计、
本地B1/B4完整短窗状态/输出与隔离门槛、同工作量CPU实测，证据及复用来源见本文件开头。
没有做跨点全源码exhaustive audit。第四轮较大batch的有界实测见下；代表性
年度、长期restart、669点/50年和AD仍需各自证据。

## 第四轮最终测量核查

第四轮最终状态：B16完成，B64不完整；不发布B64吞吐。64个唯一真实点，嵌套16点前缀；每点连续两个7日块，
每天48个半小时时步，内层`npts=1`。同设备baseline实际运行每组各自forcing、参数和
冻结Day1状态，分别比较4/16组常驻B4，不外推旧四点scalar速度。每个已接受测量
保留至少五次同步raw samples，并完成完整两块逐lane parity、跨组置换、替换隔离及实际
计时B4组检查；汇总保留完整checkpoint SHA、原始执行origin及gate。

| 同工作量，两块各7日 | 同步热core中位 | 已准备wrapper中位 | 实际常驻B4组串行中位 | core点日/s | core相对B4组 |
| --- | --- | --- | --- | --- | --- |
| 本地CPU B16，224点日 | 9.068345 s | 9.012244 s | 12.447431 s | 24.701309 | 1.372624× |
| GPU B16，4组B4，224点日 | 4.095351 s | 4.278145 s | 12.208924 s | 54.696163 | 2.981166× |
| GPU B64，16组B4 | 未完成，吞吐不发布 | — | — | — | — |

GPU最终report为`over_budget`，129项已记录gate通过、0项失败；B64 manifest原stage
仍为`running`，没有complete SHA-verified snapshot或warm samples，原始状态保留。
window2监督记录`over_budget`、exit -15，解释为有界停止，不能当作数值失败。

本地B64未完成，没有warm吞吐；129项已记录gate通过，原runner的`running`状态不改写，
另以`bounded_stop_incomplete_curve`解释外部停止。累计reported supervisor wall为
1190.948/1200s，但最后一次超过所要求safe deadline **20s**，两项事实同时披露。

GPU使用`ibc13b02n04`上的1张V100-SXM2 32GB（GPU0），模型CPU affinity4–7，
JAX/jaxlib0.4.38，冻结科学core`a33b82d`、部署`ac34b83`。两次免费window总wall
3525.762176s，charge **3527/3600 card-seconds（0.979722 card-hours）**，余额73；最多两次、每次1800s的原账本没有重置，无付费任务，余额不授权第三个window。
B16来自window1，window2只读复用已保存测量，保留原PID/window与timing origin。

| GPU window | 状态 | 完整supervisor wall | charge卡秒 | sampled group RSS峰值 |
| --- | --- | --- | --- | --- |
| 1 | completed | 1730.367190 s | 1731 | 13965389824 bytes |
| 2 | over_budget | 1795.394986 s | 1796 | 15331934208 bytes |

两个GPU window的 sampled process-group CPU下界4187.180000s（1.163106 core-hours）；
reaped children user+system合计2023.383607s，监督进程另计460.079307s。
reaped计数范围不完整：强制停止的window2只记录0.003820s，不代表该窗口的总model CPU。
这些是CPU实耗观测，不是Slurm allocated core-hours；sampled group值只作观测下界。

| 编译阶段（与热计时分开） | lower | compile/load | 首次同步双块 |
| --- | --- | --- | --- |
| GPU常驻B4，window1 | 85.769640 s | 624.033640 s | 17.264705 s |
| GPU B16，window1 | 42.538340 s | 702.747832 s | 4.170296 s |
| GPU常驻B4，window2 | 60.211150 s | 623.854428 s | 7.130623 s |
| GPU B64，window2 | 39.568088 s | 780.176860 s | 5.682454 s |

window1 B4：MISS/write，key前缀`abee48fabcc6bc04`（完整key见原始compilations）。
window1 B16（window2复用数值checkpoint）：MISS/write，key前缀`29ad2fdee95b8546`（完整key见原始compilations）。
window2 B4：MISS/write，key前缀`8275826b83ecd379`（完整key见原始compilations）。
window2 B64：MISS/write，key前缀`8f4bc5f35be96c78`（完整key见原始compilations）。
window1/window2常驻B4的StableHLO SHA前缀分别为`812f440a8ce6c1a2`与`d7f98a20caef6587`，
实际key与IR不同，原因未解析；不归因于callsite metadata或科学过程变化。

B16_scale checkpoint 6928259 bytes，serialization 0.013086s，
atomic write 0.016157s；排除manifest/report/event写入。

物理GPU0逐秒采样：window1共1731样本、峰值1503MiB；window2共1795样本、峰值2077MiB。
这是sampled peak，不是瞬时VRAM上界，也不与host RSS混用。

window1 B4/B16为MISS/write，不把624.033640s与702.747832s编译称作cache load。
数值checkpoint复用不等于executable cache命中；actual memory analysis与checkpoint IO
见专用汇总。模型affinity4–7与监督进程
affinity0–55分别记录；共享免费节点没有Slurm独占资源分配。

Linux完整64点生产reader入场通过：report elapsed858.259652s，完整监督window
865.465266s；含此前112.716639/30.850098/118.289638s窗口，实测累计1127.321642s。
100s housekeeping是保守预留，不冒充实测；263+100+1435=1798s预算包络仍在原1800s内。
新60点冻结120块原始digest保留；93字段按具名映射到未改动remote4布局，42120条记录
核对dtype/shape/bytes与冷状态关联，其余八组参数原样保留。最终prepared SHA为
`be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`。

开启JAX x64，保留基线原生dtype；已知地理payload和solar子计算含FP32。
core不含host stack/transfer/materialization；wrapper包含这些已准备输入操作，
均不含新Day1、年度聚合或生产checkpoint。lower、compile/cache、首调用、checkpoint
IO与完整window分别记录，完整window不当作热吞吐。RSS不是VRAM；内存preflight是
保守入场筛查，不能作为无OOM保证。证据见
[第四轮专用汇总](../../../outputs/research/global_throughput/round4-measurement-summary.json)，
完整资源账本与原始samples保留在该汇总及链接的window/report中。
运行后[旧资产保全检查](../../../outputs/research/global_throughput/gpu-round4/post-window-preservation-check.txt)
21项hash均OK；[checkout/进程/GPU检查](../../../outputs/research/global_throughput/gpu-round4/post-window-checks.txt)
确认remote HEAD为`ac34b83`且干净、两次窗口自身PID/PGID均不在、GPU0已释放。
首次manifest检查因Windows CRLF文件名未规范化而在读取前失败，规范化后上述检查通过；
这不是hash不匹配或模型数值失败，原尝试日志保留。
本轮不认证Fortran加速、年度/669点/50年、AD、训练、10×目标或跨CPU/GPU因果加速。
