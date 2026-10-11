# main 物理正确性检查入口

核查日期：2026-10-11。**本轮没有发现已确认的 main 生产物理实现 bug；
第三阶段已复现有当前 producer 身份的 319 cold-start 首年 NPP 科学超限。**
已有通过结果保留；旧结果的身份关联、当前科学失败及未核定的原因分别记录。
本页记录第一阶段有界盘点、第二阶段 319 历史身份调查及第三阶段当前首年对照，
不是全模型重新验收，也不改既有科学合同。

## 固定基线和支持范围

- cwd / Git root：`E:/ORCHIDEE-MAN-JAX`；分支 `main`。第一阶段基线
  `ee42eb14058a028c697147a5651912d9b8eb75e3`，已接受于
  `cd9c6b8022061a6e6e480de2e825ab28e72eac68`；第二阶段从该提交开始。
  前两阶段开始时工作区均干净；旧数值产出不赋予这些审查提交的 producer 身份。
- 第三阶段使用第二阶段接受提交
  `09968cf86551e8dfc039c1740b94fb3d42cf3f12`；开始时工作区干净，
  实际 runner/source、输入和执行身份见
  [identity.json](../../outputs/reviews/main-physics-20261011/319-current/identity.json)。
  这是新执行的 producer 身份，不是旧 A/B 产出的 producer 身份。
- 实际导入：`E:/ORCHIDEE-MAN-JAX/jax_orchidee/__init__.py`；ORCJAX、
  Python 3.11.15、JAX 0.10.1。本轮 Python 均使用 `-B`。
- 范围由 [README](../../README.md)、[交付合同](pft14_equivalence_delivery_contract.md)
  和 [paper 配置](../../configs/orchidee_man_250919.yaml) 确定：PFT14、
  `paper_250919_legacy14`、独立单点 paper 生产路径、noleap、48 半小时/日。
  单点 paper 拼接不是 `nbp_glo > 1` 的河网运行；其他 PFT/未支持配置不在等价承诺内。
- C1 `db4dd8f1cc86a33dc8e343725fda2803f63fb358`、C2
  `e317fa81b97b8b8c959f2e7dfdfba525bb54dec8`、C3
  `2a3e303845154576c6f4b5c0b8155106e5bcc70b` 已接受。
  [共享核心报告](shared_core_branch_review.md) 的 14 / 12 / 20 项回归、
  C2 极小分母导数策略和 C3 已记录的 FP64 舍入差异继续有效，本轮没有重复运行。

## 现有覆盖与证据地图

以下地图保留第一阶段结果，区分“档案已证明的范围”与“第一阶段核实的范围”；旧产出不重新标记为当前基线产出。
完整路径、账本条目和历史源码变更在
[盘点记录](../../outputs/reviews/main-physics-20261011/evidence-inventory.json)。

| 层次 | 命名证据与已有结论 | 证明范围 / 本轮边界 |
| --- | --- | --- |
| Fortran 源分支 | [reachable ledger](pft14_reachable_ledger.yaml) 为 52 个过程条目：37 active、11 conditional、4 inactive；[evidence scope](pft14_evidence_scope.yaml) 将全部 48 个可达条目列为 production_verified | 52 是过程条目数，不是 IF/WHERE 分母。保存的 [certificate](../../outputs/reference_mode/pft14_equivalence_certificate.json) 记录 5,698 sites、11,085 arms 已分类、4,580 scientific arms 有证据、260 owner regions 通过。本轮读取记录，未重新扫描整个 Fortran 分母。 |
| 独立数值 Oracle | [micro-oracle index](../../outputs/reference_mode/micro_oracles/index.json) 记录 73 个原始过程抽取、26 个 family；scientific arms 的证据路由分别含 1,243 oracle+production、2,020 owner transition、513 state IO、804 source proof | 原源码抽取本身不是执行证据；source proof、状态/IO 合同也不能泛称为每个输入的数值 Oracle。HYDROL 本轮核对了一个原始 Fortran 文件身份、命名比较和边界源码；其他 family 复用已有记录。 |
| 生产过程/状态 | [HYDROL/THERMOSOIL](../../outputs/reference_mode/production/hydrol_thermosoil/comparison.json)、[STOMATE](../../outputs/reference_mode/production/stomate/comparison.json)、[lifecycle](../../outputs/reference_mode/production/lifecycle/comparison.json) 均记录 passed / production_execution | 有独立 Oracle 接入、同一步输入/输出及状态写回证据；HYDROL runtime witness 为 tstep0→tstep1，lifecycle 为 1961 Day1/Day2 与 1962 Day1，不能单独替代整年/669 点验收。 |
| restart / 年接续 | [stage5](../../outputs/reference_mode/stage5_lifecycle_acceptance.json) 的 19 个 checks 通过；[restart lifecycle](../../outputs/reference_mode/restart_state_lifecycle.json) 181 字段闭合；[三点矩阵](../../outputs/performance/compiled_fast_path_acceptance/final_3point_annual_compiled.json) 覆盖 1961 全年、完整末日状态、1962 直接及三文件 split restart | 三点为 001、069、319；严格/compiled 比较与 Fortran 科学比较是不同证据。矩阵记三点 annual_scientific_tolerance_pass=true，但不证明三点各自 50 年序列均通过。 |
| 完整七日块 / executable reuse | [14-day cold / restart、365-day 记录](../../outputs/reviews/main-physics-20261011/block-gates.json) 均 passed；[跨点 reuse](../../outputs/performance/compiled_fast_path_acceptance/reuse_3points_current.json) 全部 strict parity 通过且后续点/重复运行不新增 executable | cold 对 strict source order；restart/annual 对 compiled daily no-block，浮点 `1e-8 + 1e-10*abs(ref)`、离散/模式精确。公共 [CLI](../../jax_orchidee/cli/main.py) 注入 on / block=7；底层 multiyear 的 block=0 默认不能据此判为公共 CLI 未启用。 |
| 七个新点长期科学证据 | [七点复核记录](../../outputs/reviews/main-physics-20261011/longrun-evidence.json) 指向七个原始 `compiled_1961_2010.json` | 103.0-095.0、225.0-101.0、281.0-093.0、287.0-111.0、293.0-113.0、295.0-081.0、331.0-057.0；每点 50 年、每年 365 日、年度 truth 完整、cold-start、block=7、NumPy accumulator off。用当前 acceptance reader 读取保存值，七点均通过原年度及序列科学门槛；没有重新运行模型。 |

七点 2010 最大 **原报告 `annual_reference_delta.rel_error` 指标**为
`8.498008041180061e-5`。其定义经
[`multiyear._delta`](../../jax_orchidee/runners/multiyear.py) 151–159 核实为
`abs_error / max(abs(reference), 1.0)`，不是无下限的标准相对误差。
例如 331 点 NPP 的 reference 为 `0.7658335566520691`，abs_error 和所记
rel_error 均为 `8.498008041180061e-5`；`abs_error/abs(reference)` 约为
`1.109641640453847e-4`。七点 28 个终年记录均符合前述报告定义。
[科学政策](paper_scientific_acceptance_policy.md) 仍用
`abs_error <= 1e-5 + 1e-3*abs(reference)` 以及原 normalized bias / RMSE
`<=1e-3`；本轮没有用报告 rel_error 替换门槛。

## 有界核实：HYDROL 动态根与新执行年

选此对象是因为 [7 月 11 日多点报告](multipoint_validation_20260711.md)
曾将 dry/snow 高风险结果与根状态关联，且根同时影响 HYDROL 水分胁迫与 STOMATE
维持呼吸。先核对条件、数组索引和状态时刻，再判断外观上的疑点。
实际源码摘录见 [bounded-source.txt](../../outputs/reviews/main-physics-20261011/bounded-source.txt)，
数值/身份记录见 [hydrol-evidence.json](../../outputs/reviews/main-physics-20261011/hydrol-evidence.json)。

| 状态 | 核实结果 |
| --- | --- |
| 已排除：动态根未实现 / 各 soil tile 使用独立根而丢失 SAVE 顺序 | Fortran `hydrol.f90` 283 为共享 SAVE nroot，`hydrol_soil` 6584–6635 在各 tile 上更新；JAX `hydrol_module_diagnostics` 7856–7916 逐 tile 将 `stress.nroot` 写回 `nroot_work`。该模块函数 AST 与 7333b46、deba027、5910ba9 均一致。 |
| 已排除：树的 tile6 阈值不同就是错误 | Fortran 6594–6597 的树根更新仅对 tile4/5 用 peat 阈值，而 6660–6666 的旧 water stress 包括 tile6；JAX 7113–7140 保留这个区别。JAX tile 为零基循环、传入 Fortran 编号 `jst+1`；nroot 是无量纲归一化层份额。 |
| 已排除：所有微小根候选都应受 min_sechiba 门槛，或 new_watstress 应改变根更新公式 | Fortran 6601/6621 为 `SUM(nroot_tmp)>0`；6648 才选择 new_watstress。JAX 7145–7154 对任何正和归一化，零和保留旧根；相应三项边界回归定义在 `tests/unit/test_hydrol_diagnostics.py` 181–293，本轮未重跑。 |
| 已排除：跨年删除 nroot 本身就是丢失 restart 状态 | 本轮完整核对 `hydrol_finalize` 1682–1797：写 `us` 等但没有 nroot；`hydrol_var_init` 4097–4137 重建根并按 ALT 掩码归一化。paper `Job0_bio` 313–342 每年重新执行并滚动三文件。JAX `rebase_driver_state_for_year_start` 2721–2755 丢弃年内 nroot，11517 的实际新年调用使用该 rebase；这与年内保留共享根的合同不同。 |
| 已复核的独立通过证据 | `hydrol_soil_owner_cwrr_water_root_peat_alt` 的 [502 项比较](../../outputs/reference_mode/micro_oracles/hydrol_soil_owner_cwrr_water_root_peat_alt/comparison.json) 全部通过；其中 nroot/humrel/vegstress 共 105 项、最大绝对误差 `2.220446049250313e-16`。其 [coverage](../../outputs/reference_mode/micro_oracles/hydrol_soil_owner_cwrr_water_root_peat_alt/branch_coverage.json) 记录该 family 所要求的 109/109 arms。Fortran 文件 SHA256 与本地一致：`39fa6d4bf4385c89a4370d75d314b07adf43fc1462b0c003d61d4c4dc3f20577`。 |
| 证据待补 / 尚未判断 | 当前 `hydrol_water_stress_diagnostics` AST 与 5910ba9 一致，与 7333b46 不同。[精确差异](../../outputs/reviews/main-physics-20261011/hydrol-forward-guard-delta.patch) 中，根归一化新增 safe denominator：正和仍是源商，非正和仍保留旧根；不能把 AST 不同等同于前向物理错误。但旧 Oracle JSON 未内嵌 JAX producer commit，本轮没有据此宣称该 JSON 是 ee42eb1 的新执行证据，也没有新验收全部梯度或 root 状态输入域。 |
| 确认实现不符 | **无。** 本轮没有得到一个经来源、可达输入和现有容差确认的当前 main 物理反例。 |

## 第二阶段：319 两条历史 50 年链的身份和后续证据

本阶段只调查 `319.0-057.0` 的精确命名结果、相邻命令/日志/checkpoint、
验收目录内同点 JSON、命名 owner/state 记录及必要的 Git 源片段。
详见 [身份与比较记录](../../outputs/reviews/main-physics-20261011/319-history/receipts.json)
及 [源码摘录](../../outputs/reviews/main-physics-20261011/319-history/source-excerpts.txt)。
第一阶段 HYDROL/年边界证据继续复用，未重跑 Oracle、回归或模型。

| 证据项 | A：wetdiaglong_effective_319 | B：12point_wetdiaglong_fix |
| --- | --- | --- |
| 原结果 / 实际窗口 | [compiled_1961_2010.json](../../outputs/acceptance/wetdiaglong_effective_319_1961_1981/319.0-057.0/compiled_1961_2010.json)；1961–2010，每年 365 日 | [compiled_1961_2010.json](../../outputs/acceptance/compiled_1961_2010_12point_wetdiaglong_fix/319.0-057.0/compiled_1961_2010.json)；1961–2010，每年 365 日 |
| 首个原科学超限 | 1961 NPP；JAX `0.3860042396860273`、reference `0.38518497347831726`，abs_error `8.192662077100565e-4` > 原 limit `3.951849734783173e-4` | 1963 NPP；JAX `-0.013051838239321791`、reference `-0.013333331793546677`，abs_error `2.8149355422488545e-4` > 原 limit `2.333333179354668e-5` |
| 全链原科学门槛 | 失败；48 个年度变量值超限，涉及 30 年；NPP normalized RMSE `0.0020904160262748147` | 失败；128 个年度变量值超限，涉及 45 年；四变量序列统计均未过，NPP normalized RMSE `0.016239404396783952` |
| 实际接续 / 命令证据 | 明确记录从 1981 年末 checkpoint 恢复；50 年文件前 21 个 year records 与 [21 年文件](../../outputs/acceptance/wetdiaglong_effective_319_1961_1981/319.0-057.0/compiled_1961_1981.json) 及 [1981 summary](../../outputs/acceptance/wetdiaglong_effective_319_1961_1981/319.0-057.0/compiled_checkpoints/multiyear_partial_through_1981.json) 逐记录完全相同，随后列出 29 个新 checkpoint。相邻 worker stdout/stderr 为空，monitor stderr 只有 CLIXML 标头；未找到完整原 argv。 | [workers.json](../../outputs/acceptance/compiled_1961_2010_12point_wetdiaglong_fix/workers.json) 留存四分片请求：cold-start、1961 起 50 年、`--strict-on-failure off --force` 和匹配输出目录；319 在 selection 的零基位置 7，按所核对 runner 的取模规则属于 shard 3。这是相邻命令记录，尚非文件级 PID/source receipt。 |
| producer / 有效版本结论 | **精确 producer 未核定**；prefix 与 continuation 均无 Git/source 身份，不能据恢复运行推断源码改变或不变。 | **精确 producer 未核定**；命令未记录源码 SHA、源码 overlay 或 Python 环境，不能把 worker PID 当成生产版本。 |

两条 `compiled.stdout.log` 的末年记录均与各自原 JSON 完全一致；ready 只证明
窗口完成，不是科学通过。它们共享记录的 config 和 materialized `run.def` 路径，
点位为 `I5/S26_69.834_0.0019_0.3980_97.918`，CSV target 为 2005。
两条保存的 50 年 reference modelout 值全部相同；本阶段另读当前本地点位
1961、1963、2005、2010 NetCDF，四变量均与保存 reference 精确一致，当前
reference PFT index 为 13。当前 materialized 内容与当前纯物化结果相同，
`OK_WETDIAGLONG=FALSE`、`dyn_nroot_larix=TRUE`、`NEW_WATSTRESS=FALSE`。
这些是当前输入/参考的核对结果：旧 JSON 只保存路径，未冻结当时 forcing、
配置内容或环境身份，不能由目录中的 `fix/effective` 名称确认当时有效开关。

Git 的初始快照 `7333b46` 已无条件输出 `compiled_day_block_size`，当前 runner
还输出稳定 PFT 元数据；两条旧 JSON 均缺这些字段。它们的格式不吻合所核对的
这两个未修改 runner，不能直接归到任一提交，也没有足够本地记录给出一个确定
的候选提交集合。已核对源码中 `_evaluate`、`_delta`、物化 helper 和 acceptance
命令构造从 `7333b46` 到第二阶段基线 AST 相同；annual reference reader 的
稳定 PFT 选择改动有明确源提交 `87f1f50`，当前仍解析到历史 PFT14 的 index 13。
这些源提交身份不是旧产出的 producer 身份，源码相同或年度数字相同也不能代替它。

有界目录调查共找到三条同点、cold-start、1961–2010 的完整候选链；除 A/B 外的
[local_50year_pilot](../../outputs/acceptance/local_50year_pilot_20260713/319.0-057.0/319.0-057.0/compiled_1961_2010.json)
也失败，127 个年度变量值超限，首个为 1961 NPP。日期目录不能证明执行先后。
在列明的本地证据集合内，**未找到可确认替代 A 或 B 的同点、同科学合同、完整
50 年实际通过结果**；这不是对全部项目、服务器或用户档案的不存在断言。

反证和范围限制一并保留：

- [wetdiaglong_fix_319 的 1961 单年](../../outputs/acceptance/wetdiaglong_fix_319/319.0-057.0/compiled_1961_1961.json)
  及同名 12 点单年结果通过，且年度记录与 B 的 1961 相同；B 的全链仍在 1963
  首次超限。单年改进不能关闭长链，也不能凭相同数字认定生产源码相同。
- [2010 Fortran restart 单年](../../outputs/acceptance/pilot_2010_restart_2points/319.0-057.0/compiled_2010_2010.json)
  通过，但它从下载的 Fortran state 起跑；三点矩阵的 1961 全年及 1962
  两日 direct/split restart 通过也保留。它们均未证明 cold-start 50 年链。
- 旧 stage6 / local 1961 的 strict 与 compiled 均科学失败，且 compiled 年度记录
  与 A 的 1961 相同；这是保存单年 counterevidence，不能扩写为两条 50 年链的
  strict 分类。B 的相邻命令明确关闭 strict-on-failure，未找到匹配全窗口 strict 结果。
- [2009 state 比较](../../outputs/reference_mode/diagnose_069_319_2009_checkpoint_vs_fortran_start.json)
  直接命名 B 的 2009 checkpoint；旧 maintenance 记录在 `1e-10` / `1e-6`
  诊断容差下分别报 day 173 / 200 首偏差，但没有连到 A/B 的源码身份。
  这些 owner/state 线索不是当前 main 反例或完整修复证据；七个新点也不包含 319。

第二阶段没有确认当前 main 物理 bug，也没有确认这两条旧失败已被有效替代。
最小后续步骤是取得命名原执行的 source/manifest；若无法恢复，才以明确当前 SHA
和输入身份做 319 cold-start 首年、必要时 carried 1961–1963 的有界比较，复用
对应 owner/state 线索且保留原门槛。最终声称长链关闭仍需有身份的 1961–2010
完整实际通过记录。第二阶段没有启动这些运行；第三阶段实际执行范围见下节。

## 第三阶段：有身份的当前 319 cold-start 首年

新产出独立保存在
[319-current](../../outputs/reviews/main-physics-20261011/319-current/)。
点位为 `319.0-057.0/I5/S26_69.834_0.0019_0.3980_97.918`，
稳定 layout 为 `paper_250919_legacy14`、mangrove index 13；1961、365 日、
48 半小时/日、cold-start，无旧 checkpoint 输入。
`used_run.def` 为当前 materialized 文件的逐字节副本；配置也保存了内容副本，
实际仍从 main 的原配置路径读取，以保留 source reader 的根目录语义。
manifest 保存 36 个配置候选输入/参考文件的 SHA256、大小和时间，合计
3,305,915,411 bytes；该集合不声称每个 static asset 都被实际消费。
环境为本地 ORCJAX / Python 3.11.15 / JAX 0.10.1 / CPU，实际导入路径如上。

compiled 使用 on / block=7 / static daily-carbon JIT on，
NumPy accumulator 和 single-pass fold off，module/local DIFFUCO JIT on，
prebuild day payloads on；显式请求全部 15 个 science/diagnostic fields。
同身份 strict 使用 compiled off / block=0 / static daily-carbon JIT off，
其余输入、冷初始化和窗口相同，module/local DIFFUCO JIT 继续 on，
与既有 failure-classification 路径一致。
实际 argv、源码/输入运行后检查和结果见
[compiled receipt](../../outputs/reviews/main-physics-20261011/319-current/compiled-1961.receipt.json)、
[strict receipt](../../outputs/reviews/main-physics-20261011/319-current/strict-1961.receipt.json)；
完整命令与前置 helper 失败见
[命令记录](../../outputs/reviews/main-physics-20261011/319-current/commands.md)。

| 1961 年度变量 | compiled JAX | strict JAX | Fortran reference | compiled abs_error / 原 limit |
| --- | ---: | ---: | ---: | ---: |
| AGB_model | 4.591235924144104 | 4.591235924144387 | 4.590915203094482 | 0.00032072104962121983 / 0.004600915203094482 |
| BGB_model | 4.120666870505465 | 4.120666870505813 | 4.120461463928223 | 0.00020540657724232858 / 0.004130461463928223 |
| GPP_model | 0.7776216017113179 | 0.777621601711987 | 0.7776592373847961 | 0.00003763567347825347 / 0.0007876592373847962 |
| NPP_model | **0.3860042397133881** | **0.3860042397139073** | **0.38518497347831726** | **0.0008192662350708368 / 0.0003951849734783173** |

compiled 完成 365 日、15 字段 count 均为 365、末步 17519，
无 missing/stopped component，JSON 所列科学/诊断数值均有限。
runner 年度耗时 275.2018044000142 s，wrapper 含结果/状态核验
276.4121996999893 s。只有 NPP 超过原年度科学门槛；
单年 normalized bias / RMSE 均为 `0.0021269423562208477`，
同样超过 `1e-3`。更严的 numerical diagnostic gate 也失败，单独保留。
strict 同样完整完成 365 日、15 字段，所列数值有限，runner 年度耗时
956.6834202000173 s，wrapper 含核验 957.554574199981 s；
NPP abs_error=`0.0008192662355900326`，normalized bias / RMSE
均为 `0.0021269423575687608`，只有 NPP 未过年度科学门槛。
两次 code=0，运行后 source/input metadata/config/run.def unchanged 均为 True，
当前 live 1961 reference 与保存 reference 精确相同。
四个年度 JAX 值的 strict/compiled 最大差约 `6.692e-13`（GPP），
NPP 差约 `5.192e-13`。按原政策分类为
**`shared_semantic_input_or_reference_mismatch`：共享语义、输入绑定、汇总或参考差异待定位**。
这不是 compiled-only 退化，也尚未证明共享物理过程实现必有 bug；
当前科学超限本身没有定位物理函数缺陷。完整数值核验见
[analysis.json](../../outputs/reviews/main-physics-20261011/319-current/analysis.json)。

**完整状态有限性与科学输出有限性分别记录。** 两份 checkpoint 的原始全树扫描各
为 351 个 numeric leaves、23,992 个元素，发现两个 NaN，故 raw state finite=false：
`diffuco_previous_step_state.roughheight_pft` 与
`sechiba_finalize_state.roughheight_pft`，均为 shape `(1,14)` 的 `[0,0]`。
该位置是 Fortran PFT1 / bare_soil，layout 的 active_mask **确为 True**、
prescribed fraction 为 0，不能按全局 inactive mask 排除。
实际 `IMPOSE_AZE=FALSE`、`ROUGH_DYN=TRUE`；动态 producer
`condveg.py` 364/366/417–419 保留 bare slot NaN，只写后续 PFT，
对应 `condveg.f90::condveg_z0cdrag_dyn` 1707–1718 的 `DO jv=2,nvm`。
Fortran `diffuco_aero` 1067–1069、1107–1118、1128–1134 同样从 PFT2 开始，
PFT 特定 drag 还受 `ok_LAIdev` 控制；本地 14 个 `OK_LAIDEV` 均为 False。
JAX `diffuco.py` 4168–4177 在运算前用 grid roughheight 替换未消费坐标，
4227–4233 选择 grid drag；orchestration 2936–2937/3131 保留该 sidecar。
因此这两个特定未定义坐标是当前有界 producer/consumer 路径的合法哨兵，
不是已确认的 NaN 传播 bug；原始 checkpoint 和 receipt 保留不改。

原 receipt 的 `partial_exactly_matches_cached_summaries=false` 也保留。
pickle 的 `paper_modelout_csv_delta` 是 tuple，JSON 是 list；按结构和数值语义
比较后相同，partial summary 与结果 year prefix 相同。
这不修改原序列化格式，也不把 Python 容器类型差异当作 year-state 缺失。
两份 checkpoint 都只有上述两个坐标非有限，其余 23,990 个 numeric 元素有限。
精确坐标和 tuple/list 语义比较见 analysis；对应
[source excerpts](../../outputs/reviews/main-physics-20261011/319-current/source-excerpts.txt)
保留本阶段 producer/consumer 和 Fortran 条件。

**PEAT_HYDRO 仍有 319 配置/语义边界待核定。** raw run.def 值为 `TRUE`，
当前 JAX `orchestration.py` 68–75/630–631 则区分参数开关与 HYDROL soil 的
`PAPER_1961_HYDROL_SOIL_PEAT_HYDRO=False`，6381–6382/6987–6988
传入 soil/root/water-stress 路径。Fortran `hydrol.f90` 95 声明局部
`LOGICAL,SAVE :: peat_hydro`，该声明没有初始化器；
`pft_parameters.f90` 3626–3627 对参数开关读取 run.def。
`hydrol_main` 1278 调用 `hydrol_soil`，tile loop 的 5858/5891 调用
`hydrol_soil_coef`，8259/8290 的实际条件是 **HYDROL 局部变量为 True 且
Fortran soil tile 为 4/5/6**，不能由 raw 字符串直接决定。
现有 [hydrol v9 audit](hydrol_1961_v9_audit.md) 保存 tile4/bottom11 的 1200 个
coefficient rows，均为 `peat_hydro=F/branch_peat=F`，以及 144 个 tile4 step rows。
其配套 [设计入口](../porting/pft14_extensible_design.md) 9–10 关联
`reference/case_001_071`；CSV 没有点位/producer 字段，未建立对应 319 Fortran
执行的身份桥。第一阶段 ledger 的 False 只保留其原审计范围，不能仅凭 case001
trace 外推 319 也应 False；无初始化器的声明也不是全运行 False 的来源证明。
CSV 核验、内容身份和点位限制见
[peat-trace-scope.json](../../outputs/reviews/main-physics-20261011/319-current/peat-trace-scope.json)。
当前 pinning 的 319 有效性尚需核定，既未排除也未确认为本次 NPP 原因，
不声称所有其他 peat 相关消费者 inactive。

1961 已失败，按批准的条件与主控确认，**未继续到 1963**，也未执行 50 年或
669 点积分。现有 A/B、单年 restart、七个其他点和已接受 C1–C3 的结论继续保留；
当前数值接近历史 A 的 1961 残差不构成 producer 或因果同一性。
最小下一步只核对 **1961 NPP 的来源与有限局部证据**：先核对 daily NPP、
maintenance/growth respiration 与 annual 汇总的观测/单位/时刻，再复用
[day173@1e-10](../../outputs/reference_mode/diagnose_319_maintenance_first_divergence.json)
和 [day200@1e-6](../../outputs/reference_mode/diagnose_319_maintenance_material_divergence.json)
的命名线索，保留其 **maintenance 累积后、非 turnover 后 biomass** 的观测边界，
沿温度/长期温度、根状态和维持呼吸核对命名输入与写回；必要的执行证据限定
到相应局部过程和有来源的输入，不扩大到新年度窗口或 50 年。
旧记录没有本次 producer 身份，不能称为当前首偏差日；并先核定上述局部开关
的点位适用性。本阶段不自动改生产函数或容差，也不启动新的模型/trace。

## 已确认限制与优先检查清单

完整 669 点验收未完成。保存 certificate 的 `equivalent=false` 同时引用旧
stage6 2/3 代表点结果，不能把该旧 false 直接当作当前实现失败；反过来也不能把
七点通过扩写为 669 点通过。`process_span_unmapped=4732` 是诊断性 span 计数，
证书已有另一套 arm 分类/owner 路由，不是 4,732 个未实现过程。
HYDROL peat hydraulic coefficient 条目虽然命名含 conditional，但实际 reachability
为 inactive，并给出了 audited runtime `peat_hydro=F` 的来源理由；其 regression-only
不能脱离有效配置解释成可达缺口。该 False 的点位范围须保留：第三阶段仅确认
所关联 case001 trace；对 319 的有效 Fortran 局部值尚未建立身份桥，见上节。
更一般的潜在输入域仍受原交付合同约束。

下列四项按下一步价值排序；它们是检查对象，非四个待修 bug。

1. **证据到当前物理版本的关联（证据待补）。**
   七点、source/owner 和 state 档案已有很强的通过证据，但若干原始 JSON 未内嵌
   JAX producer SHA。入口为本轮 inventory 的 production_change_history、
   `scripts/dev/audit_pft14_equivalence_certificate.py::build_certificate`、
   `deba027` / PFT identity 与 restart 提交 / `5910ba9` 以及已接受 C1–C3。
   本轮以 HYDROL 示例核对原 Fortran身份、当前函数 AST 和具体 guards。
   最小后续办法是关联命名原执行记录与 Git 源片段，对有科学影响的差异选择既有
   小型 Oracle/回归；相同代码复用证据。尚未核定这些档案的完整当前版本桥，
   这不撤销原通过结果，也不要求首轮穷尽 producer 历史或重跑长序列。

2. **319 高风险历史长序列及当前首年科学偏差（历史身份仍待补；第三阶段已取得当前首年反例）。**
   [第一阶段失败值](../../outputs/reviews/main-physics-20261011/named-high-risk-history.json)
   和 [第二阶段 receipts](../../outputs/reviews/main-physics-20261011/319-history/receipts.json)
   保留第一阶段 069 保存 50 年链通过的结论，以及两个 319 失败和反证：
   A 为已验证的 21+29 年恢复链、首超限 1961 NPP；
   B 有匹配的相邻四分片命令、首超限 1963 NPP。两条 producer 均未核定，
   本地三条完整同点候选均未过原门槛，未建立有效替代；单年通过及七个其他点
   不能关闭它们。第三阶段的当前 identified cold-start 1961 NPP 同样超限；
   strict 分类及最小 owner/state 后续范围见第三阶段，不从科学失败直接认定物理函数 bug。
   当前 HYDROL/年边界已核实的来源和反证继续有效，不从旧失败直接修改 root 或门槛。

3. **当前三文件 restart 与年接续的完整状态合同（有通过证据，待版本关联）。**
   入口为 `_paper_day_end_state_packet`、`paper_1961_driver_restart_year_start_day_result`、
   [restart lifecycle 合同](restart_state_lifecycle.yaml) 和 stage5 的 181 字段记录；
   Fortran writer 来源为 ledger 所记 `stomate_io.f90::writerestart` 2200–2948，
   HYDROL writer / 年初始化范围如上。已有 001/069/319 全年直接/split restart
   与离散/模式检查通过，nroot 删除疑点已排除。
   最小办法对照 PFT identity/restart 接入后的 reader/writer/字段元数据，复用已有
   年边界与序列化检查；只针对实际变更核对 axes、mask、时间归属和重算字段。
   尚未独立重审全部 181 字段在 ee42eb1 的读写链，不能据此说状态缺失。

4. **七日 compiled 块跨点绑定与年输出（有通过证据，待当前入口/全点接入核对）。**
   入口为 `_paper_compiled_later_day_block_executable` 11185–11355、
   `_paper_run_compiled_later_day_blocks` 11358–11471、
   `read_forcing_model_step_cached`、`audit_paper_landpoint_binding`、
   `compute_modelout_from_fields`。账本给出 Fortran `sechiba_main` 997–1224、
   `forcing_read_interpol` 660–1687/1899–1968；本轮已核对 paper postprocessor
   48–76/115–118 的 PFT14 选择、碳库相加及 AGB/BGB `0.02` 转换。
   已有 14/365 日 A/B、三点 executable reuse、七点 block=7 的完整 50 年科学通过。
   当前 block 的 forcing/science day、参数、landpoint payload 与 restart template
   是显式动态参数，`year=1961` 的静态 metadata 本身不是锁定所有年 forcing 的证据。
   最小办法核对缓存绑定和公共 CLI→runner 的实际参数流，复用已有 forcing transport /
   block parity / landpoint binding 检查；669 点数据绑定审计与 669 点长期科学验收分别记录。
   尚未完成 ee42eb1 的完整跨点消费者审计及 669 点积分验收。

## 本轮执行记录

[第一阶段命令和结果](../../outputs/reviews/main-physics-20261011/commands.md) 记录实际本地读取、
保存值科学门槛复核、Git/AST/Fortran 身份核对以及辅助脚本首次导入失败后的修正。
第一阶段新增本页和 `outputs/reviews/main-physics-20261011/` 辅助证据，并更新
`docs/RESEARCH_WORKFLOW.md` 导航。未修改生产源码、测试、科学阈值或旧产出。
未执行 pytest/GPU/长模型运行、训练、服务器作业、Teacher/prepared 生成或数据重标。

第一阶段图工具恢复可用：项目 `E-ORCHIDEE-MAN-JAX`、generation
`2026-10-10T08:46:50Z`、12,100 nodes / 72,107 edges。
[图记录](../../outputs/reviews/main-physics-20261011/graph-receipt.json) 和
[补充覆盖](../../outputs/reviews/main-physics-20261011/graph-additional.json) 明确显示
目标源码 metadata_changed，Fortran 在 excluded subtree；关键结论直接读当前源码，
不把旧图当当前完整性证明。采用 Verify 层级，不作全项目不存在/完整覆盖断言。
最终 `git diff --check`、新报告空白/相对链接检查及证据引用存在性检查见
[verification.json](../../outputs/reviews/main-physics-20261011/verification.json)。

第二阶段只更新本页和 `outputs/reviews/main-physics-20261011/319-history/`。
[第二阶段命令](../../outputs/reviews/main-physics-20261011/319-history/commands.md)
记录保存值读取和辅助脚本的 scalar/list checkpoint 天数兼容修正；
[图记录](../../outputs/reviews/main-physics-20261011/319-history/graph-receipt.json)
仍为 Verify，generation 相同、源码 metadata_changed、产出目录 excluded，
关键身份结论直接核对当前源码和 Git 片段。
[第二阶段 verification](../../outputs/reviews/main-physics-20261011/319-history/verification.json)
记录链接、证据引用、原产出身份、基线、修改范围及 `git diff --check`。
未改生产源码、测试、科学阈值或原产出，未执行长模型/GPU/训练/服务器工作。

第三阶段只更新本页和 `outputs/reviews/main-physics-20261011/319-current/`。
实际执行两个当前有身份的 1961 单年模型；未续跑 1962/1963、50 年或 669 点。
[命令](../../outputs/reviews/main-physics-20261011/319-current/commands.md) 及原始 receipt
分别记录执行完成与科学失败；辅助 analysis 只读取现有产物并写新 sidecar。
[图记录](../../outputs/reviews/main-physics-20261011/319-current/graph-receipt.json)
保留所提供的 Verify generation，但本阶段 index_status/trace_path/coverage 的
本地调用为 `Transport closed`，不声称 fresh MCP 或全模型完整性；关键结论直接
核对当前有界源片段。新
[verification](../../outputs/reviews/main-physics-20261011/319-current/verification.json)
记录实际 input/result/checkpoint 身份、原政策复核、科学输出与状态有限性分离、
语义相等、链接与修改范围及 `git diff --check`。未改原 JSON/pickle/receipt、生产
源码、测试、科学阈值或原通过结论；未执行 pytest/GPU/训练/SSH/服务器作业。
提交与推送由主控在实际产物验收后执行。
