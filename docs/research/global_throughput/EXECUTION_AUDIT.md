# 主分支物理模型执行审计与首轮基准设计

更新：2026-10-05。审计起点：`main@5910ba9af857a91e565d13a25e3de2e3019080d1`，
工作分支：`research/global-throughput`。本文件只审计 source-backed 正向模型；不依赖日尺度神经替代。
本轮保持科学过程不变，已实现编译前的外层 batch；本地 B1/B4 有界数值与计时通过。
服务器诊断由协调主任务独立记录，本实现任务未启动远端或付费任务。

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

协调主任务完成 Explore1000 免费有界诊断，并传回本分支证据：
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

## 结论及证据边界

现有代码已经提供完整 later-day 多日纯数值扫描的实现基础，最小批处理落点是
`jax_orchidee/driver/orchestration.py:11231` 的内部 `transition`，而不是公共 CLI。
本轮已在 `.lower().compile()` 之前暴露这段函数，保留单点内部维度，以外层 `vmap`
装入异质网格。本地B1/B4短窗数值门槛和CPU吞吐诊断已通过；GPU在编译预算内未完成
batch验收。2.271倍只指本地同工作量JAX串行对照，不能外推为Fortran或GPU加速。

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
阶段2的GPU部分在scalar编译预算边界停止。阶段3、4和Fortran同硬件比较尚未执行。

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

已直接阅读上列主代码边界、CLI/runner默认、forcing/parameter/state adapters、static dispatch、
AD primitives及相关测试实现；核查七日块三份dynamic gate、三点block2复用、多日/组件性能
原始JSON；检查base HEAD。初始来源审计未重跑历史昂贵gate；后续新增了四点独立输入审计、
本地B1/B4完整短窗状态/输出与隔离门槛、同工作量CPU实测，证据及复用来源见本文件开头。
没有做跨点全源码exhaustive audit。GPU编译成本是当前远端诊断限制；更大batch、代表性
年度、长期restart、669点/50年和AD仍需各自证据。
