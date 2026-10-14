# CPU48 v5 保存数组复核与执行边界

2026-10-14；只分析既有数组、执行 pin 源码和已保存遥测。**CPU internal strict gate 仍为 FAIL；scientific_acceptance=false。** 新主线按最新派工改为将成熟独立单点 worker 调度到 48 核，停止 B64 vmap 后续准备；不把有限末位差异的深层原因闭合作为吞吐研究前置。本文件不批准或执行任何新作业，不修改科学计算、容差或既有硬门。

## 输入与方法

证据目录为 `outputs/research/global_throughput/cpu48-b64-execution-v5-20261014/`，以下文件均保持原字节：

| 输入 | 字节 | SHA256 |
|---|---:|---|
| `raw_B64_batch_before_parity.pkl` | 27,562,300 | `a135f9436afa868176bf8cc76f28134218e510c65b75909075ba24a8964810ce` |
| `source-scalar.pkl` | 27,604,466 | `1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa` |
| `../gpu-round4/model/stage_B4_group_references.pkl` | 27,587,892 | `a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6` |

取树分别为 raw 的 `outputs`、scalar 的 `value['outputs']`、GPU 的 `value`。GPU 数值来源是 gpu-round4：执行 head `ac34b833784cd93222b7dd82e98c7d47277084ae`、节点 `ibc13b02n04`；**不是**热吞吐比较作业 15579869，也没有使用 `local-round4` 的同名文件。

本地临时 Python 脚本用 `conda run --no-capture-output -n ORCJAX python -B -X utf8` 执行，外层 `subprocess.run(..., timeout=300, check=True)`。在反序列化前通过 Windows `GetProcessAffinityMask/SetProcessAffinityMask` 设置并回读实际 affinity mask `1`（仅 CPU0）；`OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS/NUMEXPR_NUM_THREADS` 全为 `1`。NumPy 2.4.6。最终完整审计及元数据读取的进程内墙钟为 3.4235681 秒；这是保存数组分析耗时，不是物理性能测量。pickle 类导入可间接导入 JAX，但没有设备查询、物理调用、lower、compile 或缓存编译。一次最初的 ctypes affinity 签名失败发生在读取数组前，修正后执行；没有由此运行模型。

审计独立递归 dict（排序键）、tuple/list（逐项，包含 NamedTuple）；另比较容器类型、键和层级。没有调用原 harness 的 `compare`。每叶检查 path、shape、原生 dtype；整数/布尔使用精确相等；浮点用原规则 `abs(candidate-reference) > 1e-8 + 1e-10*abs(reference)`。只豁免以下四条路径中、shape `[64,1,14]` 的 `[...,0]` **双方同时为 NaN** 的 bare-soil roughheight 哨兵：

`[0][0][0][13]`、`[0][0][5][18]`、`[1][0][0][13]`、`[1][0][5][18]`。

其余任何 NaN/Inf 均作为问题计数；没有扩大 mask。合同来自执行 pin `scripts/dev/benchmark_global_batch.py::roughheight_sentinel_contract`（572–588 行），该函数检查 bare_soil/Fortran PFT1、按原 spec 查找字段，并约束 inner shape。

## 完整树结果及三方数值

候选与 scalar 的 **760 native leaves / 3,447,168 元素**全部遍历。容器结构、路径、shape、dtype 完全一致；744 个 float64 叶、4 个 int32、8 个 bool、4 个 int64。16 个离散叶共 7,680 元素，精确差异为零。四条允许路径各 64 个 matched NaN，共 256 个；非有限 mask 不一致数为零，其它非有限元素数为零。

有限超差恰为 **3 个元素 / 3 个叶比较 / 2 个点**，没有额外 schema、dtype、离散值或非有限问题。下表相对量均用 `abs(candidate-scalar)/abs(scalar)`，bound ratio 用原 scalar 参考界限。

| block / path / 坐标 | candidate | scalar | GPU exact reference | CPU 内部绝对差 / 相对量 / bound ratio |
|---|---:|---:|---:|---|
| Days2–8；`[0][0][0][7]`；`[6,0,13]` | 592.8152488486185 | 592.8152487724788 | 592.8152488575903 | `7.613971320097335e-8` / `1.284375079059338e-10` / `1.098990146880813` |
| Days9–15；`[1][0][5][48]`；`[37,0]` | 29.926931253713647 | 29.926931273231105 | 29.926931237973783 | `1.951745787209802e-8` / `6.521703710248401e-10` / `1.5021872433093637` |
| Days9–15；`[1][0][6][0]`；`[37,0]` | 29.926931253713647 | 29.926931273231105 | 29.926931237973783 | 同上 |

lane6 为 `299.0-115.0`，lane37 为 `313.0-079.0`。同坐标 candidate−GPU 分别为 `-8.971824172476772e-9`、`+1.5739864522856806e-8`（后者出现两次）；scalar−GPU 为 `-8.511153737345012e-8`、`+3.5257322394954826e-8`（两次）。不能因 candidate 更接近某一参考就撤销原 strict FAIL。

另将同样 760 叶独立与 GPU 比较，结构/dtype/离散值/非有限检查均一致通过：

| 比较 | 超差点及元素 | 有限失败叶比较数 | 全树 max_abs |
|---|---|---:|---:|
| candidate vs scalar | lane6：Day8 rveget；lane37：Day15 fluxsens 两个状态槽 | 3 | `8.463580161333084e-5` |
| candidate vs GPU | lane45 `283.0-091.0`：Day8 rveget；lane11 `285.0-087.0`：Day15 rveget；lane37：Day15 fluxsens 两槽 | 4 | `1.7935875803232193e-5` |
| scalar vs GPU | lane6：Day8 rveget；lane11：Day15 rveget；lane37：Day15 fluxsens 两槽 | 4 | `6.669992581009865e-5` |

candidate/GPU 的四处绝对差依次为 `8.665494988235878e-8`、`5.92048081671237e-8`、`1.5739864522856806e-8`（两次）；scalar/GPU 为 `8.511153737345012e-8`、`7.072463859003619e-8`、`3.5257322394954826e-8`（两次）。两者均为 61/64 点通过，但失败点集合不同。**全树 max_abs 不是表中失败位置误差**：原界限随参考量级变化，绝对误差较大的其它位置可仍通过。

## 坐标到物理量：执行 pin 的结构证据

Python 来源均读取 `git show 0100c8b09be5137b822173e0a0be66cb64939727:<path>`，没有将研究树 `62e905...` 当成执行物理树 `75d8b1e9...`。图项目 `E-ORCHIDEE-MAN-JAX`、generation `2026-10-14T11:04:08Z`；Verify 查询及一次覆盖检查显示工作树和所需 Fortran 目录均 excluded/not_tracked，因此精确 pin 源码与目标 Fortran 文本为实际证据，未作图的完整性声明。

执行 canonical 准备收据 `outputs/research/global_throughput/round4-inputs-remote-canonical/remote-preparation-audit.json` 的 `new_point_records` 中，两个目标点的 `positional_state_representation_migration.bound_layout` 都明确给出：component0=`diffuco_previous_step_state`、field7=`rveget`；component5=`sechiba_finalize_state`、field48=`fluxsens`；component6=`driver_previous_step_state`、field0=`fluxsens`。其 prepared pickle SHA `be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7` 与 v5 目标 free report 声明一致。原 anchor pickle 本地 SHA `da6cd17ed7c9531d997888e6bbd3167c34751bb7c80b001cfd72b7a98dde7bfe` 与 canonical 收据一致；只读反序列化检查同样字段顺序。Linux PosixPath 只映射为 PurePosixPath 读取元数据，不作路径 IO。local prepared 的 SECHIBA 字段顺序不同，未使用它解释执行数组。

结构连接为：`fast_state.py::fast_state_from_previous_packet`（约142–168 行）按 packet 原插入顺序构造 spec 和 values；`prepare_global_frozen_arguments.py::canonicalize_state_arguments`（181–242 行）显式映射为原 remote anchor 的字段顺序并核对 named bytes/non-state groups；`orchestration.py::_paper_later_day_block_transition_factory`（11207、11272–11307 行）以 initial spec 重建最终 `next_values`，每 block 输出 `(final_values, daily_outputs)`。原 `run_two` 再把第一块最终状态传入第二块；故这里是 Day8、Day15 的最后半小时末状态，非日均或七日累积量。

| 失败位置 | 物理含义和单位 | 空间/PFT |
|---|---|---|
| `[0][0][0][7]` | `rveget`，整个冠层的气孔阻力，`s m^-1` | lane6、inner landpoint0、PFT 索引13 = Fortran PFT14 / `mangrove_pft14` |
| `[1][0][5][48]` | SECHIBA finalize 保存的 `fluxsens`，感热通量，`W m^-2` | lane37、inner landpoint0；此量无 PFT 维 |
| `[1][0][6][0]` | driver 保存的同一 `fluxsens` | 同上 |

单位的局部 Fortran 源真值：`E:/ORCHIDEE-MAN-JAX/fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main` 第402行及 `diffuco_trans` 第1801行直接声明 rveget 单位；`enerbil.f90::enerbil_main` 第468行、`enerbil_flux` 第1401行声明 fluxsens 单位，第1533行给出感热计算。此为针对字段/单位的来源核对，不是完整 Fortran 分支审计。

**fluxsens 的两个失败比较是同一个物理值的两份状态表示。** 这由构造关系证明：执行 pin `orchestration.py::_add_enerbil_local_to_previous_fields` 第2845、2857–2859行把 `enerbil_local.payload['fluxsens']` 写入 driver；`_sechiba_finalize_state_from_components` 第3074、3083、3096行把同一 Enerbil local payload 写入 finalize。day-end 只对 slowproc 声明字段更新 finalize，见3180起和9239–9242行；最后按相同 spec 返回。该结论不依赖两个浮点数恰好相等。rveget 则由 `_add_diffuco_local_to_previous_fields` 第2876–2879行写回 DIFFUCO 状态。

## 遥测能说明什么

既有 v5 `model-report.json` 的 compile=`256.66296943090856s`、first-call=`131.93829891085625s`。phase 记录包含 phase 名、elapsed wall、RSS；有 lower_start、compile_start、compiled、raw_output_saved。post-compile 记录179个 TID及其允许 CPU mask，48 affinity 核准入通过。**这些没有提供逐阶段 CPU user/system 增量、每线程运行时间、利用率采样或硬件性能计数器**，不能把179线程/48 affinity解释为48个有效并行核，也不能定位编译或 first-call 的实际核利用率。

报告 process user+system=`1294.22s`、benchmark wall=`759.1614878531545s` 均覆盖多阶段；`batch.log` 的 `/usr/bin/time` 另保存总进程 CPU百分比160%、wall13:31.20、总context switches（voluntary3,390,410/involuntary5,078）、缺页及IO计数。它们是聚合数据，口径也不完全相同，不能移植为 compile/first-call 的阶段指标。没有由此计算 cold/warm 加速比。

本次执行模型为 **一个 JAX CPU进程、一个 CPU device、`vmap`64点、48核 affinity**；`global_batch.py::make_batch_transition` 第322–325行明确为 scalar transition 的 vmap。这不是48个成熟 single workers。permutation/replacement 物理调用确实执行，两条记录 gate 都通过；但失败阶段未封存，不能声称已独立用 sealed arrays 复算 isolation。CPU internal gate 抛出后，reset gate、5 core+5 wrapper warm样本和 validation-array seal 均未到达；现存 raw output不能补出这些证据。

原唯一付费 job15857765 为 FAILED1:0/819s，已关闭；819×48=10.92核时，未核实费用情景 CNY0.546/0.7644。旧 scalar job15820467 的2197s和29.293333333核时另计一次，CNY1.464666667/2.050533333；不重新生成或重复收费。免费collector15864638 COMPLETED0:0/6s仅完成失败记账，不构成有效吞吐阶段。

## 下一主线的边界

最新派工优先复用成熟 single worker，尽快形成实际 CPU/GPU 同量比较。修改前执行 pin `0100c8b09be5137b822173e0a0be66cb64939727` 的 `benchmark_global_server_cpu.py::SingleRunner`（186–227行）编译一个B1可执行后串行遍历点；`pool_worker`（230–289行）已支持单核/线程变量1、每worker独立编译一次并持久复用；`PoolRunner`（292–350行）已用spawn、轮转任务和完整唯一点汇总。原CLI只开放4核pool，原实现每worker读取全64输入并保留assigned输出，不能把它直接称为已接受48worker方案。

原 scalar 完整report（`outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar/report.json`，SHA `81bd4ceb413c079eb8f1fa7c00c61f4a8d74ddb75278d2a49bd1e5fd82db4da0`）保留 lower55.8773/compile684.2656/B64串行first137.3622秒，core warm中位127.074984秒（7.050955 point-days/s），wrapper121.38245秒（7.381627 point-days/s）。postcompile RSS5,180,694,528B、warm5,565,206,528B、HWM21,406,240,768B（约19.94GiB）包含编译/整组输出，不能线性承诺48进程能放下或已有扩展效率。新worker方案与必要harness另见[当前方案](CPU_SCALAR_POOL_PLAN.md)；本分析本身不运行模型。

科学 strict FAIL 与以后**明确标记的 exploratory timing**可作为分开报告的候选政策：保留完整误差，不要求先解释每个有限末位差异；schema/dtype、非有限、isolation/reset仍为硬门。该候选必须由主管作出明确科学/门策略决定，再落实到获准的新worker路线。当前保存证据缺 reset和独立sealed isolation，不能直接拿 v5 当作已满足条件的热吞吐样本。禁止把“误差小”替换成科学接受，也不放宽原容差。

深层溯因到此停止。现有端点数组不能归属舍入、融合、归约顺序或上游传播，也不能据此断言科学可忽略；保留这个证据边界即可推进新的CPU worker工作。
