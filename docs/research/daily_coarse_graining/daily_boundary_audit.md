# 日尺度边界审计与 replay speed-ceiling 方案（v0 历史记录）

> 本文的 pre-daily-STOMATE 边界已被生产数据契约 `daily_markov_contract_v3`
> 取代。本文仅保留旧边界和 speed-ceiling 实验的审计依据。新 Teacher shard 与
> surrogate 接口必须使用 `S[d] + native forcing[d] + P -> B_fast[d]`，再由保留的
> daily STOMATE 生成 `S[d+1]`，不得按
> 下文的 `forcing_48` 或 retained-STOMATE v0 接口新增数据。

状态：初版审计与 30 日 canonical-minimal replay 已完成，Teacher commit
`7333b46`，2026-07-21。

本文件只定义研究边界和最小实验方案，不改变 `jax_orchidee/` 的科学语义、参数或
默认生产路径。首版粗化模型只能称为 `teacher-matched` 或 `coarse`。

## 1. 结论

生产 later-day 路径的实际顺序是：

```text
Day-N complete packet
  -> bind forcing/static/day metadata
  -> 48 x SECHIBA transition, emitting 48 STOMATE entry payloads
  -> daily accumulator + maintenance fold
  -> 48 x OK_LEAK using same-step SECHIBA fields and maintenance respiration
  -> overlay 13 common OK_LEAK live fields + conditional deepC_peat
  -> season/STOMATE input bundles
  -> STOMATE daily carbon
  -> modelout diagnostics
  -> daily accumulator reset + slowproc surface update + season writeback
  -> merge complete Day-N+1 packet
  -> optional year rebase / restart serialization
```

关键代码入口是：

- `jax_orchidee.driver.orchestration::_paper_1961_later_day_half_hour_transition`
  产生 48 个 SECHIBA entry 和最后一个半小时物理状态；
- `_paper_later_day_daily_process_from_completed_entries` 产生日累计量和 maintenance；
- `_paper_half_hour_ok_leak_fold_from_entries` 按 48 步推进 OK_LEAK；
- `_paper_day_stomate_daily_carbon_from_bundles` 执行日碳过程；
- `_paper_day_first_stomate_state_packet` 和 `_paper_day_end_state_packet` 完成日末写回；
- `_paper_compiled_later_day_block_executable` 将上述完整日转移放入跨日 `lax.scan`。

因此，`compiled complete-day carry` 是完整 Teacher 日转移的优化载体，不是现成的
粗化注入边界。可复用的是 `DriverFastStateBundle` 的固定 PyTree 布局和
`DriverPreviousStepStatePacket` 的无损适配契约。经边界复核，首版 coarse operator
应覆盖所有亚日尺度 fast-day 过程：48 次 SECHIBA、日累计/maintenance 和 48 次
OK_LEAK；输出直接进入原始 season/STOMATE 日碳链。

## 2. 边界复核与决策

### 2.1 OK_LEAK 不是纯日末过程

旧版开发规范把 OK_LEAK 放在 coarse operator 之后，但当前实现中的 OK_LEAK 依赖
48 个半小时 entry，而不是只依赖日末物理状态。它逐步更新：

```text
litter_above, litter_below,
lignin_struc_above, lignin_struc_below,
litterpart, dead_leaves,
fuel_1hr, fuel_10hr, fuel_100hr, fuel_1000hr,
carbon_32l, DOC, interception_storage
```

同时每步消费 `soil_mc`、水文通量、`temp_sol`、`tdeep`、`hsdeep`、
`shumdiag_peat` 和 maintenance respiration。因此，若把 OK_LEAK 留在 coarse operator
之外，运行时仍必须保留 48 步 bridge，不是真正的一次日尺度转移。

### 2.2 最终采用 pre-daily-STOMATE 边界

首版边界现统一为：

```text
日初完整状态 + forcing_48
  -> coarse fast-day operator
  -> 日末 SECHIBA 状态 + 日累计/maintenance + OK_LEAK 日末状态
  -> Teacher season/STOMATE 日碳 -> modelout -> 写回 -> restart
```

`pre_daily_stomate_replay` 是政策一致的主 speed-ceiling。`sechiba_only_replay` 仍可
作为瓶颈拆分诊断，用于量化 daily fold + maintenance + OK_LEAK 的成本，但不是最终
coarse 架构。

### 2.3 当前 packet 不是可直接学习的 state schema

已存在的 1961 年末生产 checkpoint（PFT14 单点）包含 7 个 component、334 个叶子：

| Component | 实例字段数 | 粗化边界处置 |
| --- | ---: | --- |
| `driver_previous_step_state` | 14 | 物理 carry；部分是下一步耦合系数 |
| `diffuco_previous_step_state` | 36 | 物理状态与诊断混合 |
| `enerbil_previous_step_state` | 13 | 能量状态与诊断混合 |
| `hydrol_previous_step_state` | 48 | 水文状态、通量和离散 soil class 混合 |
| `thermosoil_previous_step_state` | 17 | 土壤热状态与诊断混合 |
| `slowproc_stomate_previous_step_state` | 113 | 碳慢状态、季节记忆、离散状态和日累计混合 |
| `sechiba_finalize_state` | 93 | restart/finalize 镜像、静态量和诊断混合 |

这些数字是当前 PFT14 单点实例，不是所有条件分支的固定分母。现有
`sechiba_state_field_contract.yaml` 声明 5 个运行 component 的 123 个字段；
`restart_state_lifecycle.yaml` 覆盖 STOMATE restart/packet 生命周期。粗化 ledger 应
引用这两个真源，不应另抄一份会漂移的 334-field 清单。

但 123-field contract 当前尚未闭合：基线测试识别出生产 owner 已写入
`thermosoil_previous_step_state.e_soil_lat`，实际 checkpoint 也包含该字段，而 YAML
contract 没有对应条目。因此该 YAML 只能作为待修复的 source ledger，不能作为已
冻结的完整 coarse schema。本次审计把 `e_soil_lat` 作为显式 provisional supplement，
不修改 Teacher 或原 source-audit 文件。

尤其不能让网络独立预测重复镜像，例如 `temp_sol`、snow state、surface vegetation
和 `sechiba_finalize_state` 中的副本。正式 adapter 必须选择一个 canonical owner，
再解析生成镜像并检查一致性。

## 3. 字段总账方案

机器可读草案位于：

`manifests/coarse_graining/daily_boundary_v0_draft.yaml`

### 3.1 分类规则

- `prognostic`：跨半小时/跨日影响后续物理转移的 canonical 动态库存。粗化转移
  只更新 SECHIBA 所有的这一子集。
- `diagnostic`：从 canonical state、forcing 或 Teacher 输出解析产生；不得作为
  独立自由预测目标。`sechiba_finalize_state` 默认属于镜像/诊断层，除非现有
  lifecycle ledger 明确证明某字段具有独立 carry 语义。
- `accumulator`：17 个当前 active-paper 日累计/maintenance 字段。政策一致 replay
  直接回放这些 Teacher target；未来 coarse model 直接输出或解析计算它们，并保持
  Teacher 的单位、归一化和 reset 语义。
- `static`：`Paper1961PreparedDriverContext` 中的网格、参数、开关和 PFT 属性，只读。
- `discrete`：`tstep/day/year`、`njsc`、布尔状态、mask、PFT present/phenology flags
  等。首版中由规则或保留的 Teacher 日过程更新，不由连续网络拟合。

### 3.2 首版输入、输出和原样携带

首版 coarse 输入：

- 日初完整 `DriverFastStateBundle`；
- 当天 48 步 forcing：`zlev`, `zlevuv`, `u`, `v`, `qair`, `temp_air`, `pb`,
  `precip_rain`, `precip_snow`, `lwdown`, `swdown`, `ccanopy`, `salinity`,
  `tide_height`；
- 已冻结的 static config 和日历/landpoint metadata。

首版 coarse 输出分为四层：

1. canonical 日末 SECHIBA prognostic state；
2. STOMATE 消费的日累计与 maintenance 接口；
3. 13 个通用 OK_LEAK 日末 litter/soil-carbon/DOC 状态，以及 `PERMA_PEAT` 路径的
   条件 `deepC_peat`；
4. 可由上述 canonical 输出解析重建的 diagnostic/restart 镜像。

`slowproc_stomate_previous_step_state` 中未被 OK_LEAK 更新的慢状态在 fast-day 阶段
原样携带；13 个通用 OK_LEAK 字段和条件 `deepC_peat` 由 coarse operator 更新；
其余 season、植被碳和长期记忆字段继续由 Teacher 日过程更新。

### 3.3 Teacher target 提取用 48 步 bridge

当前 compiled SECHIBA entry stack 有 46 个字段。按 active paper path 的实际
consumer，Teacher target 提取和 `sechiba_only_replay` 诊断的最小候选是 30 个字段：

```text
precip_rain, precip_snow, veget, veget_max, totfrac_nobio, gpp,
humrel, litterhumdiag, t2m, temp_sol, stempdiag, shumdiag,
t2m_min, t2m_max, wspeed, snow, tmc_topgrass,
fwet_new, liqwt_ratio,
soil_mc, wat_flux, runoff_per_soil, drainage_per_soil, runoff2peat,
canopy2ground, precip2ground, precip2canopy,
shumdiag_peat, tdeep, hsdeep
```

其中 `fwet_new`/`liqwt_ratio` 由 `PEAT_OCCUR` 条件控制，`shumdiag_peat` 由
`PERMA_PEAT` 条件控制。v0 target 提取先保留三者，避免用当前点的开关裁剪 schema。

这 30×48 个字段不是正式 coarse 运行时接口。正式模型必须直接生成日末状态、
日累计接口和 OK_LEAK 日末状态，否则没有完成时间粗化。

必须用 consumer audit 证明其余 16 个 compiled entry 字段不影响三个 target family
的生成，再允许从 target-extraction payload 中删除。不能仅凭单点数值相等删除。

### 3.4 17 个 active-paper 日累计字段

一次 1962 Day 1 的只读运行审计确认当前 daily fold 实际输出：

```text
humrel_daily, litterhum_daily, t2m_daily, tsurf_daily,
tsoil_daily, soilhum_daily, precip_daily, gpp_daily, wspeed_daily,
snowfall_daily, snowmass_daily, tmc_topgrass_daily,
t2m_min_daily, t2m_max_daily,
resp_maint_part, resp_maint_radia, flood_root_radia
```

这些字段属于 pre-STOMATE 接口。日末 packet 中的 `daily_accumulators` 是经过
`do_slow` reset 的下一日初状态，不是当天的训练 target；两者必须使用不同字段 ID。

## 4. 水、能量和碳接口

### 4.1 水

canonical 库存来自 HYDROL/雪/冠层状态，包括 `mc`, `mcl`, `water2infilt`,
`qsintveg`, snow states, `flood_res`, `wt_ab`, `wt_ab_tide` 等。日 bridge 提供降水、
土壤水分、垂向水通量、runoff/drainage、canopy routing 和 peat diagnostics。

当前缺失项：还没有一个冻结的“日初库存 + 外部输入 - 日末库存 - 输出 = residual”
水预算 API；`sechiba_finalize_state` 中的 `tot_watsoil_beg`/`tot_watveg_beg` 也未被
证明足以构成完整日预算。Gate 2 前必须新增 source-backed budget ledger。

### 4.2 能量

canonical 热状态包括 `temp_sol`, `ptn`, snow thermal state、`cgrnd/dgrnd`,
`soilcap/soilflx`, `tdeep/hsdeep`。`stempdiag`, `temp_sol`, `tdeep`, `hsdeep` 进入
STOMATE/OK_LEAK bridge。

当前缺失项：没有冻结的日能量闭合 target，也没有明确哪些瞬时能量通量需要日积分。
因此 v0 可以做状态 replay ceiling，但不能声称已有能量守恒训练接口。

### 4.3 碳

粗化阶段更新 OK_LEAK 所有的亚日尺度 litter、32 层土壤碳、DOC 和冠层截留状态，
并生成 `gpp_daily` 与 maintenance 接口；但不近似 season、biomass allocation、NPP、
turnover 或长期碳记忆。后者继续由 Teacher STOMATE 更新。

当前仍缺一个冻结的 pre-STOMATE 碳预算 residual 定义。`carb_mass_total` 是后续
modelout 诊断，不能反向充当粗化边界的独立 target；OK_LEAK 碳状态必须通过显式
库存/通量约束或守恒投影验收，不能只依赖逐字段拟合误差。

## 5. Producer / consumer 总账

| 边界对象 | Teacher producer | 保留 consumer | 粗化处置 |
| --- | --- | --- | --- |
| 日初完整 state | 上一日 `_paper_day_end_state_packet`；跨年由 rebase/restart | 当日 transition、daily process | 固定 PyTree 输入 |
| 48 步 forcing | driver forcing completion/interpolation | SECHIBA coarse transition | 只读输入 |
| 日末 SECHIBA state | `_paper_1961_later_day_half_hour_transition` | 下一日 SECHIBA、day-end merge、restart | coarse 主要输出 |
| 30-field bridge | 每步 compact SECHIBA entry producer | daily fold、maintenance、OK_LEAK | 仅 Teacher target 提取和诊断，不进入正式 coarse runtime |
| 17-field daily fold | `stomate_daily_process_fold_from_entries` | season、STOMATE daily carbon、reset/modelout | coarse 输出/解析 target |
| 13-field common OK_LEAK carry + conditional `deepC_peat` | `_paper_half_hour_ok_leak_fold_from_entries` | STOMATE bundle、outputs、day-end packet/restart | coarse prognostic 输出 target |
| STOMATE daily state | daily carbon + season + OK_LEAK | modelout、slowproc update、restart | Teacher 保留 |
| reset accumulator state | `_paper_day_daily_reset_fields` | 下一日 daily fold | Teacher 保留，不作当天 target |
| complete day-end packet | `_paper_day_end_state_packet` | next day / year handoff / restart writer | Teacher merge 保留 |

## 6. 纯内存 replay speed-ceiling

### 6.1 最小实现边界

不修改 Teacher 代码的第一轮可以在 `research/daily_coarse_graining/` 中建立实验
adapter，调用现有私有边界函数；确认继续路线后，再提一个单一、测试覆盖的正式
daily-transition 注入接口。

准备阶段只运行一次 Teacher，收集连续日的内存 PyTree：

```text
ReplayRecord = {
  day_start_spec_hash,
  final_sechiba_values,
  daily_fold_fields,
  ok_leak_interface,
  first_step_metadata,
  optional_bridge_48_for_diagnosis,
  expected_day_end_values,
  expected_modelout,
}
```

计时阶段预先把所有数组 materialize 到内存，并在计时前完成 device placement。
不得逐日读 pickle/NetCDF/Zarr，不得把 Teacher 数据准备时间混入热运行。

### 6.2 测量矩阵

对同一 landpoint、forcing、初态和进程，依次测量：

| 模式 | 跳过内容 | 保留内容 | 用途 |
| --- | --- | --- | --- |
| `teacher_compiled` | 无 | 完整生产路径 | 基线 |
| `pre_daily_stomate_replay` | SECHIBA + fold + maintenance + OK_LEAK | season、日碳、modelout、写回 | 首版政策一致 ceiling |
| `sechiba_only_replay` | 48 次 SECHIBA | fold、maintenance、OK_LEAK、日碳、modelout、写回 | 瓶颈拆分诊断 |
| `state_copy_only` | 全部科学过程 | 仅内存读取/复制/scan | replay 搬运成本下限，不是模型结果 |

分别报告：数据准备、首次编译、首次调用、同步后的热运行、device/host 转移和可选 IO。
主指标是秒/landpoint-year 和 50 年外推；同时报告 1 年实测。先用 7 日 warmup +
30 日重复测量检查实现，再做 365 日 ceiling。50 年只根据稳定的 1 年结果外推，首轮
不运行 50 年 Teacher 数据生成。

所有 replay 模式必须逐字段比较其保留后续阶段的 modelout 和完整 day-end packet。
浮点使用现有 production gate `atol=1e-8`, `rtol=1e-10`；离散、schema、mask 和
adapter roundtrip 精确相等。`pre_daily_stomate_replay` 必须在相同边界输入下做到
后续阶段相等，以证明注入位置正确。

### 6.3 现有性能给出的预期

当前生产 compiled 年运行约为 79--94 秒/landpoint-year，即约 66--78 分钟/50 年，
不是概念文档早期假设的“十几分钟”。30 日 profile 中 daily fold、day preparation、
OK_LEAK 和 compiled SECHIBA 都是可见成本，而且累计时间互相重叠。

所以不能在实测前承诺 1--2 分钟/50 年。主 go/no-go 指标是
`pre_daily_stomate_replay`；它直接测量完整 fast-day operator 被一次日尺度预测替代后
的端到端上限。`sechiba_only_replay` 只用于解释 fast-day 内部成本组成。

30 日 canonical-minimal 动态 replay 已实测为 `0.01865 s/day`，单 executable 跨
4 个完整 7 日 block 复用；本轮 Teacher/replay 中位数为 `3.173/0.559 s`，即
`5.67x`。结合上一轮完整动态对象图实验的 `4.80x`，该 dynamic Teacher replay
实现测得 `4.8--5.7x`，而固定常量边界为 `14.18x`。短窗 50 年外推约 5.7 分钟，
尚未计入 coarse operator 推理，也不是任意 learned operator 的架构绝对上限。

当前最小动态 transport 为 343 个叶子、266426 bytes/day：310-leaf fixed-spec
SECHIBA packet、2 个最后一步诊断量、17 个 daily 字段和 14 个 active-path
OK_LEAK 字段。它已删除 pre-step boundary、fold scaffold、重复 update dict 和只作
shape 用的 `resp_hetero_soil`。容器精简后绝对 replay 时间与完整动态图基本相同，
说明 retained Teacher tail 才是主要成本。

## 7. 缺失与 Gate 0/1 退出条件

当前未闭合项：

- 669-landpoint Teacher 最终验收未完成，所有研究数据只能标记
  `provisional_teacher`；
- 123-field SECHIBA contract 的现有测试因缺少
  `thermosoil_previous_step_state.e_soil_lat` 而失败；它与实例 packet 的其他
  conditional/packet-only 增量尚未形成一个单一 daily schema；
- target extractor 的 46 -> 30 bridge 裁剪尚缺静态 consumer 自动审计；
- canonical state 与 duplicate/finalize mirrors 的 materialization 规则尚未逐字段
  冻结；
- 水、能量、pre-STOMATE 碳预算 residual 尚无 source-backed 字段公式；
- 当前 runner 没有正式的 pre-daily-STOMATE 注入接口；
- cold-start Day 1、later day 和 restart-year Day 1 的 boundary adapter 需要分别验收；
- JAX 首次编译很重，本次只读 Day 1 检查耗时约 109 秒并出现 XLA algebraic
  simplifier 循环警告，ceiling 计时必须明确区分编译与热运行。

只有以下条件同时满足，才进入数据/schema 冻结：

1. `pre_daily_stomate_replay` 在 3 日、跨年 Day 1--2 和 restart roundtrip 上通过完整
   packet 验收；
2. `pre_daily_stomate_replay` 的 365 日热运行显示足以改变研究可行性的收益；
3. Teacher target 提取能够完整产生日末物理状态、17-field daily interface、
   13-field common OK_LEAK 状态和条件 `deepC_peat`；
4. budget ledger 明确哪些量可硬约束，哪些只能作为诊断；
5. Teacher commit、config hash、run.def hash、state spec hash 全部记录。

## 8. 预计工作量和下一步

必须区分“快速性能判断”和“正式边界硬化”。以单人、现有代码熟悉度为前提：

| 工作包 | 预计工作量 |
| --- | ---: |
| 单点 later-day 最小内存 replay + 30 日重复计时 | 已完成 |
| 无神经日尺度 baseline + 单日/7 日 parity harness | 1--2 天 |
| 正式 target extractor、manifest 校验和 consumer audit | 1--2 天 |
| 跨年/restart 门禁和 365 日实测 ceiling | 1--2 天 |
| 水/能量/碳预算 source audit 与 ledger v1 | 3--5 天 |

速度路线判定为“有条件继续”：存在约 5 倍零推理成本空间，但没有数量级余量。
下一步先做无神经日尺度 baseline 和最小 parity harness，同时启动预算 ledger；
其真实推理加入后必须重新执行 go/no-go。暂不创建神经网络代码、不冻结训练集、
不生成大规模数据，也不提交 GPU 长任务。
