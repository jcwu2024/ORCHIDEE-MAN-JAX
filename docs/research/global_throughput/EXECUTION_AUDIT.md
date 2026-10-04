# 主分支物理模型执行审计与首轮基准设计

日期：2026-10-04。审计起点：`main@5910ba9af857a91e565d13a25e3de2e3019080d1`，
工作分支：`research/global-throughput`。本文件只审计 source-backed 正向模型；不依赖日尺度神经替代。
本轮未改科学过程、未执行模型长编译、未访问服务器、未提交计算任务。

## 结论及证据边界

现有代码已经提供完整 later-day 多日纯数值扫描的实现基础，最小批处理落点是
`jax_orchidee/driver/orchestration.py:11231` 的内部 `transition`，而不是公共 CLI。
应先在 `.lower().compile()` 之前暴露这段函数，保留单点内部维度，以外层 `vmap`
装入异质网格。该方案尚未实现或测量，GPU/批处理速度提升仍是假设。

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

现行工厂返回 `jax.jit(transition).lower(...).compile()` 的 AOT executable（11324-11349），
不暴露原 `transition`；应在编译前提取/复用原函数工厂。直接变换 AOT executable
不是本研究可依赖的路线。先做 `B=1` 输出/完整 state parity，再做不同点 `B=4`，才有理由
测量更大 B。直接把 `npts` 改成 B 是另一条路线，需要额外审计所有 reduction、grid、static
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

## 首轮可复现诊断合同（尚未执行）

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
15日窗的结果，不能拿它与Fortran365日annual truth作科学年度验收。批量诊断代码需后续实现，
当前没有虚构可用的batch CLI。正式年度gates沿用现有semantic/acceptance harness。

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
原始JSON；检查base HEAD。没有重跑昂贵数值gate、没有产生新速度实测、没有做跨点全源码
exhaustive audit。后续须补当前环境性能、异质B1/B4等价、batch缓存兼容与长期restart证据。
