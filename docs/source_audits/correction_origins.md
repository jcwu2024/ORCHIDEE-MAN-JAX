# C1–C44修正来源登记（2026-11-14）

当前累计44项修正、24个代码提交。原C1–C40的40项复查前来源审计完整保留于下节；C41／C43及C44的cold默认遗漏也在复查前`7cae3e02`存在，C42则是本轮`fd0fe51495626b39ce42d50dec29b3a6cef744b3`新增restart准入helper的默认边界遗漏。当前合计43项有复查前JAX证据、1项属复查期新helper遗漏，不能称44项全是旧翻译错误；也未由C42推出标准算例数值回归。

C41–C43共同修复提交为`8a772de6f4977018a875093d535b6088317716c8`，C44为`9c15c042a5ade9c91d91ac91a8c0ea39ed87c8f2`；新增接受限定在源码／适配器／缓存语义。详见[累计登记](shared_core_branch_review.md#累计修改位置与提交索引)、[C41–C43接受记录](../../outputs/reviews/main-physics-20261113/parameter-repairs-calendar-design/controller-acceptance.json)和[C44接受记录](../../outputs/reviews/main-physics-20261113/clock-authority-cold-inputs/controller-acceptance.json)。日历接线及两个clock helper候选仍未修复，不进入44项分母。

## 原C1–C40来源审计快照

以明确的复查前边界 `7cae3e02adad23674767a590fdf4d27f239ce065` 为准，**40项的错误表达式或源合同遗漏均能直接在该基线中定位**。这支持“这40项所修的问题在本轮共同核心复查前已存在”，但不支持“40条独立、当时均激活的物理公式bug”。没有在这40项中证实复查期间新引入的回归；未审计所有未接纳候选或每个中间实现。

审计冻结已接纳登记HEAD为 `765c75fc51af981b1f687f8e36b05f8dd382dc77`。先读[累计登记](shared_core_branch_review.md#累计修改位置与提交索引)，再逐项核对基线源码及22个准确修复提交；没有切换版本、运行旧测试/旧模型或追溯到最初移植提交。下表“首次已知”统一为基线 `7cae3e02`，真实引入提交未知且至迟在该边界已存在；不能把修复日期当成引入日期。

来源标签计数：`baseline-defect-present`=40；`review-era-new-or-extended-interface-omission`=0；`confirmed-regression-introduced-during-review`=0；`insufficient-origin-evidence`=0。基线存在性的置信度为高，准确引入提交仍未知，零unknown不代表已找出40个最初引入提交。

性质按一个主维度互斥计数：源过程计算/常数/单位/顺序13项；默认/参数/状态/restart接口20项；显式诊断API4项；执行缓存/AD稳定性3项。源过程计算组也包含条件分支，不等同13项都在纸面案例自然激活。C2/C3尤其不应称为两条新增物理公式错误。40个概念修正对应22个代码提交，多字段、多caller或一次修复尝试不重复计数。

机器证据与可复现脚本：[origin-audit.json](../../outputs/reviews/main-physics-20261113/final-gas-runtime-and-fix-origins/origin-audit.json)、[build-origin-audit.py](../../outputs/reviews/main-physics-20261113/final-gas-runtime-and-fix-origins/build-origin-audit.py)。JSON逐项保存完整revision、函数跨度/身份及精确基线摘录；表内行号都是**基线版本**，可用 `git show 7cae3e02adad23674767a590fdf4d27f239ce065:<path>` spot-check，不能直接当作当前工作树行号。修复列保留完整SHA；`git show <repair> -- <path>`可复核改动。

| ID | 修正性质 | 基线直接证据（路径省略`jax_orchidee/`） | 首次已知/引入边界 | 准确修复提交 | 来源标签/置信度 |
| --- | --- | --- | --- | --- | --- |
| C1 | 执行缓存 | `sechiba/diffuco.py:3472–3473` `_diffuco_pft_column_mask`：host lru_cache直接保存JAX数组；可保存tracer | `7cae3e02`；最初引入未知、至迟基线 | `db4dd8f1cc86a33dc8e343725fda2803f63fb358` | `baseline-defect-present` / 高 |
| C2 | AD稳定性 | `stomate/carbon_kernels.py:4000–4013` `npp_closed_update`：维持呼吸份额仍用普通除法，尚未用稳定比值原语 | `7cae3e02`；最初引入未知、至迟基线 | `e317fa81b97b8b8c959f2e7dfdfba525bb54dec8` | `baseline-defect-present` / 高 |
| C3 | AD/份额条件化 | `stomate/carbon_kernels.py:4563–4576` `turnover_leaf_age_fall`：叶龄份额从带共同叶质量的分子/分母求商，尚无relative质量路径 | `7cae3e02`；最初引入未知、至迟基线 | `2a3e303845154576c6f4b5c0b8155106e5bcc70b` | `baseline-defect-present` / 高 |
| C4 | restart单位/默认 | `stomate/restart_io.py:790–803` `read_stomate_readstart_states_from_template`：tau缺值上限把系数3当作天数 | `7cae3e02`；最初引入未知、至迟基线 | `4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99` | `baseline-defect-present` / 高 |
| C5 | 调用参数 | `sechiba/thermosoil.py:3153–3171` `thermosoil_explicit_step`：getdiff调用缺原水文nslm；同函数较早profile调用已传nslm，不能混淆两处 | `7cae3e02`；最初引入未知、至迟基线 | `a117cd5afc92ea55c9aeb528eec6f724de780111` | `baseline-defect-present` / 高 |
| C6 | 过程时钟单位 | `driver/orchestration.py:7933–7946` `_paper_ok_leak_boundary_from_sources`：TF_DOC实际producer传整日dt而非半小时dt | `7cae3e02`；最初引入未知、至迟基线 | `27ff14b38aafc7df91ce52b8393642ca08952395` | `baseline-defect-present` / 高 |
| C7 | 过程率/质量单位 | `stomate/integration.py:3995–4011` `stomate_ok_leak_explicit`：DIC聚合直接接收四项litter/soil率，缺步长转换 | `7cae3e02`；最初引入未知、至迟基线 | `27ff14b38aafc7df91ce52b8393642ca08952395` | `baseline-defect-present` / 高 |
| C8 | 过程共享输入顺序 | `stomate/integration.py:3958–3971` `stomate_ok_leak_explicit`：soilcarbon接收原bulk_dens，缺litter及soil两次共享赋值 | `7cae3e02`；最初引入未知、至迟基线 | `27ff14b38aafc7df91ce52b8393642ca08952395` | `baseline-defect-present` / 高 |
| C9 | 默认/装配参数 | `driver/orchestration.py:8325–8338` `_paper_compiled_ok_leak_fold`：独立源默认cutoff错误地绑定水文层数 | `7cae3e02`；最初引入未知、至迟基线 | `27ff14b38aafc7df91ce52b8393642ca08952395` | `baseline-defect-present` / 高 |
| C10 | 过程赋值顺序 | `stomate/soilcarbon_kernels.py:3547–3560` `soilcarbon_leak_doc_diffusion`：第二峰值通量仍用旧f0构成的共同分母 | `7cae3e02`；最初引入未知、至迟基线 | `225cb95a6a44334390d1656d85f66fd54d342a19` | `baseline-defect-present` / 高 |
| C11 | 过程常数 | `sechiba/hydrol.py:2445–2458` `snow3lheat_explicit`：雪融化潜热默认333600而非333700；与土壤冻结lhf分开 | `7cae3e02`；最初引入未知、至迟基线 | `dd6d6335ba5011fcd86285984136a6799f6916ac` | `baseline-defect-present` / 高 |
| C12 | 过程分支阈值 | `sechiba/hydrol.py:10576–10615` `hydrol_soil_all_tiles_explicit_step`：统一peat阈值传入各阶段，丢失tile4/5和4/5/6差别 | `7cae3e02`；最初引入未知、至迟基线 | `d48b923e891946d3c9db77ba9b684666cb511b1d` | `baseline-defect-present` / 高 |
| C13 | 过程重构顺序 | `sechiba/hydrol.py:11774–11787` `hydrol_alt_residual_solve_step`：dummy只从原mc重构一次，缺冻结条件首赋值后再读更新mc | `7cae3e02`；最初引入未知、至迟基线 | `d48b923e891946d3c9db77ba9b684666cb511b1d` | `baseline-defect-present` / 高 |
| C14 | 显式诊断API | `sechiba/hydrol.py:7583–7596` `hydrol_peat_water_table_diagnostics`：暖层总数代替地表连续暖层前缀 | `7cae3e02`；最初引入未知、至迟基线 | `d48b923e891946d3c9db77ba9b684666cb511b1d` | `baseline-defect-present` / 高 |
| C15 | 参数传递/缓存 | `sechiba/condveg.py:1094–1107` `condveg_main_minimal`：main签名/roughness调用不转发已有Z0_BARE | `7cae3e02`；最初引入未知、至迟基线 | `bbcdbf7a661cb11f5295883e54c3b1ea9d596d39` | `baseline-defect-present` / 高 |
| C16 | 状态/调用传递 | `sechiba/hydrol.py:8019–8038` `hydrol_module_diagnostics`：诊断读取split旧ae而非surface更新值；driver也沿用旧split | `7cae3e02`；最初引入未知、至迟基线 | `2f36dc7f7945d52387a2848aa124c043118ed45f` | `baseline-defect-present` / 高 |
| C17 | 过程末态刷新 | `sechiba/hydrol.py:10709–10733` `hydrol_soil_all_tiles_explicit_step`：normal末态后直接进入dummy，无final-mc冻结profile刷新 | `7cae3e02`；最初引入未知、至迟基线 | `ad0d322cc4951d4af158f4e8cbb6d306bdf374f3` | `baseline-defect-present` / 高 |
| C18 | 过程整数/浮点边界 | `sechiba/hydrol.py:6441–6454` `hydrol_soil_coef_mineral_profile_from_tables`：非冻结表达式先加imin再INT；peat同类表达式同在基线 | `7cae3e02`；最初引入未知、至迟基线 | `ad0d322cc4951d4af158f4e8cbb6d306bdf374f3` | `baseline-defect-present` / 高 |
| C19 | 显式诊断API | `sechiba/hydrol.py:7592–7605` `hydrol_peat_water_table_diagnostics`：潮汐水库只对当前tile6选择，非按cell的tile6面积选择全部tile | `7cae3e02`；最初引入未知、至迟基线 | `ad0d322cc4951d4af158f4e8cbb6d306bdf374f3` | `baseline-defect-present` / 高 |
| C20 | 门控参数/缓存 | `sechiba/condveg.py:1131–1144` `condveg_main_minimal`：main没有snowcri_alb入口/实参，固定消费helper默认 | `7cae3e02`；最初引入未知、至迟基线 | `f8d36e1f8c03046e12e146aed2abb8bd286403de` | `baseline-defect-present` / 高 |
| C21 | 参数传递/缓存 | `sechiba/condveg.py:1153–1170` `condveg_main_minimal`：动态roughness调用缺MIN_WIND | `7cae3e02`；最初引入未知、至迟基线 | `c214e43da57c21b68002186fbf71c67d45835bae` | `baseline-defect-present` / 高 |
| C22 | 参数传递/缓存 | `sechiba/condveg.py:1153–1170` `condveg_main_minimal`：roughness调用缺HEIGHT_DISPLACEMENT | `7cae3e02`；最初引入未知、至迟基线 | `c214e43da57c21b68002186fbf71c67d45835bae` | `baseline-defect-present` / 高 |
| C23 | 参数传递/缓存 | `sechiba/condveg.py:1153–1170` `condveg_main_minimal`：roughness调用缺Z0_ICE | `7cae3e02`；最初引入未知、至迟基线 | `c214e43da57c21b68002186fbf71c67d45835bae` | `baseline-defect-present` / 高 |
| C24 | 参数传递/缓存 | `sechiba/condveg.py:1131–1144` `condveg_main_minimal`：质量雪fraction调用缺SNOW_DENSITY | `7cae3e02`；最初引入未知、至迟基线 | `c214e43da57c21b68002186fbf71c67d45835bae` | `baseline-defect-present` / 高 |
| C25 | 状态/雪相位传递 | `driver/orchestration.py:2932–2944` `_add_condveg_to_previous_fields`：packet仅保存四项roughness，未保存snow fractions；restart首CONDVEG仍接旧雪 | `7cae3e02`；最初引入未知、至迟基线 | `c214e43da57c21b68002186fbf71c67d45835bae` | `baseline-defect-present` / 高 |
| C26 | 显式诊断默认 | `sechiba/hydrol.py:8429–8442` `hydrol_tmc_update`：可选水量检查默认零容差而非源2e-8 | `7cae3e02`；最初引入未知、至迟基线 | `84b5d425418a6d1b18fa6dbe080348c2faa72e13` | `baseline-defect-present` / 高 |
| C27 | 过程跨PFT顺序 | `stomate/soilcarbon_kernels.py:3738–3751` `soilcarbon_leak_doc_export`：每个PFT内部把peat累计buffer置零，并只回加本PFT loss | `7cae3e02`；最初引入未知、至迟基线 | `9fc87c011a4e991f1637223b0f8fb2a224000595` | `baseline-defect-present` / 高 |
| C28 | 显式诊断默认 | `stomate/soilcarbon_kernels.py:2706–2719` `soilcarbon_perma_peat_redistribute`：peat_OLT诊断默认0而非源1e-8 | `7cae3e02`；最初引入未知、至迟基线 | `9fc87c011a4e991f1637223b0f8fb2a224000595` | `baseline-defect-present` / 高 |
| C29 | 过程根归一化 | `sechiba/hydrol.py:2048–2060` `hydrol_nroot_from_humcste`：ALT0时对全部原根求和触发归一化，缺源znh<ALT累计条件 | `7cae3e02`；最初引入未知、至迟基线 | `ea86d3fbe97b3a1ecf5c5d8dda931aea5ea2ae69` | `baseline-defect-present` / 高 |
| C30 | 过程网格几何 | `sechiba/hydrol.py:1922–1934` `hydrol_first_step_profil_froz_from_arrays`：冻结profile从相邻节点差构造dh而非真实dlh*1000 | `7cae3e02`；最初引入未知、至迟基线 | `ea86d3fbe97b3a1ecf5c5d8dda931aea5ea2ae69` | `baseline-defect-present` / 高 |
| C31 | 过程末态温度 | `sechiba/thermosoil.py:3579–3592` `run_thermosoil_first_step_module`：结束直接返回downstream，缺显雪grid温度clamp写回 | `7cae3e02`；最初引入未知、至迟基线 | `cb4f76817acd852cde8cd8d4195e1e52bfd8c770` | `baseline-defect-present` / 高 |
| C32 | restart packet所有权 | `driver/restart_bundle.py:94–105` `paper_restart_bundle_state_from_day_end_packet`：gas/remainder直接沿用base，未用当前packet；standalone同类 | `7cae3e02`；最初引入未知、至迟基线 | `77c3ac7d30ddf466a3cc2b4d6824824713a632b1` | `baseline-defect-present` / 高 |
| C33 | restart派生量 | `driver/restart_export.py:88–101` `stomate_restart_states_from_day_end_packet`：合并pool后缺失total仍留旧base，不从实际pool重建 | `7cae3e02`；最初引入未知、至迟基线 | `77c3ac7d30ddf466a3cc2b4d6824824713a632b1` | `baseline-defect-present` / 高 |
| C34 | restart初始化/持久状态 | `driver/orchestration.py:11513–11533` `paper_1961_driver_restart_year_start_day_result`：直接rebase进入main，缺非持久热初始化；file路径同缺且首packet漏e_soil_lat | `7cae3e02`；最初引入未知、至迟基线 | `fd0fe51495626b39ce42d50dec29b3a6cef744b3` | `baseline-defect-present` / 高 |
| C35 | restart目录传递 | `driver/orchestration.py:7109–7122` `paper_1961_driver_timestep_scaffold`：STOMATE boundary调用未传first_step_restart_state.run_dir | `7cae3e02`；最初引入未知、至迟基线 | `fd0fe51495626b39ce42d50dec29b3a6cef744b3` | `baseline-defect-present` / 高 |
| C36 | restart active初始化/保持 | `driver/orchestration.py:5390–5404` `_paper_first_step_restart_entry_sources`：active STOMATE也用inactive derivvar覆盖；缺恢复assim/height及月温初始化growth | `7cae3e02`；最初引入未知、至迟基线 | `fd0fe51495626b39ce42d50dec29b3a6cef744b3` | `baseline-defect-present` / 高 |
| C37 | restart整数接口 | `sechiba/restart_io.py:162–166` `sechiba_finalize_source_state_from_restart`：NC REAL映射直接返回，未恢复njsc/veget_year的INTEGER owner类型 | `7cae3e02`；最初引入未知、至迟基线 | `fd0fe51495626b39ce42d50dec29b3a6cef744b3` | `baseline-defect-present` / 高 |
| C38 | compiled forcing绑定 | `driver/orchestration.py:9934–9947` `_paper_1961_later_day_transition_inputs`：静态模板按year/tstep48取天气；start0盐/潮位用day_start_payload而非实际series0 | `7cae3e02`；最初引入未知、至迟基线 | `45654fea50aec6edf7774ac0f0b90170eea617db` | `baseline-defect-present` / 高 |
| C39 | gas缺值默认 | `stomate/restart_io.py:795–808` `read_stomate_readstart_states_from_template`：O2/CH4缺值默认都为0，而非源298.3/0.001267 | `7cae3e02`；最初引入未知、至迟基线 | `76e406e0e0e75402d221bb281908fa4260bc5d66` | `baseline-defect-present` / 高 |
| C40 | gas创建/状态carry | `driver/orchestration.py:3451–3464` `driver_previous_step_state_from_next_step_hydrol_scaffold`：首日后继scaffold白名单末尾无四gas；cold producer/extractor也无创建/携带 | `7cae3e02`；最初引入未知、至迟基线 | `76e406e0e0e75402d221bb281908fa4260bc5d66` | `baseline-defect-present` / 高 |

C40的三个边界需分清：基线已没有cold gas producer/extractor，且首日后继`driver_previous_step_state_from_next_step_hydrol_scaffold`的carry清单缺四gas；基线的`STOMATE_FIXED_PATH_CARRY_FIELDS`及初始restart提取却已有gas。因此“某处已有gas”不能证明完整首日生命周期。初版修复没有闭合该既有遗漏，是不完整修复，不自动成为复查引入回归或新C编号。本审计不改C40已接纳的`76e406e0e0e75402d221bb281908fa4260bc5d66`身份，也不把本轮后续运行候选预先接纳。

C34非持久热初始化、C35显式目录、C36 active初始化及growth保持、C37 INTEGER解码、C38静态模板/实际forcing绑定，均在基线已有对应入口且缺少所修合同；不是仅凭2026-11修复时间判作新增接口遗漏。C32/C33的bundle/standalone packet合并和派生total接口也直接存在于基线。C15/C20–C24新增参数接线是在修补基线已有component/helper与caller的断接；不能因修复增加签名或resolver就反推缺陷由复查新增接口造成。

此表只审计来源和性质；数值、运行覆盖、活动gate、历史参考producer及研究分支同步状态仍以原登记和[主报告](main_physics_review.md)为准。没有由源码存在性推出年度影响、所有分支通过或gas动力学激活。

## C41–C43及未修日历候选的有界来源核查

本节新增来源核查与上方原40项快照分开。主控已将P-CTX-OKPC、P-REFSOC-DEFAULT、P-COMPILED-CLOSURE分别接纳为C41、C42、C43；FIELD-CANDIDATE-01仍未修复。完整JAX提交边界、精确源码跨度、当前未跟踪Fortran文件SHA256及限制见交付时的[机器记录](../../outputs/reviews/main-physics-20261113/parameter-repairs-calendar-design/candidate-origins.json)和[说明](../../outputs/reviews/main-physics-20261113/parameter-repairs-calendar-design/candidate-origins.md)，原交付候选ID保留以便追溯。

| 候选ID | 有界来源分类 | 结论与限制 |
| --- | --- | --- |
| `P-CTX-OKPC` | 复查前已存在的JAX参数接口遗漏 | `7cae3e02`仍把raw `OK_PC`复制进context；初始仓库提交`7333b46c`已出现该表达式。当前本地Fortran `control.f90:255–278`要求`OK_LEAK`有效时将`ok_pc`置false，但该源文件不受Git跟踪，不能声称它在复查前边界的Git历史中存在。 |
| `P-COMPILED-CLOSURE` | 复查前已存在的JAX缓存／闭包绑定遗漏 | `7cae3e02`已有跨context复用compiled constructor缓存的路径，缓存闭包可捕获首个context；初始仓库提交`7333b46c`已包含该路径。已接受反例替换了编译与物理delegate，不是实际compiled物理数值失败。 |
| `FIELD-CANDIDATE-01` | 复查前已存在的选定driver日历owner接线遗漏 | `7cae3e02`已有JAX `slowproc_main_pft14_step`日历owner，但选定orchestration未调用；day-end adapter也未接入Jan-1 reset、Jan-2 `peatC_ok`写入及普通日末`growth_day`增量。当前本地Fortran `slowproc.f90:556–594`给出对应顺序，文件未跟踪，不能据此恢复Fortran的Git历史。普通日末部分patch仍不完整，未应用。 |
| `P-REFSOC-DEFAULT` | 复查期间新helper缺少源默认 | helper及缺键`FALSE`回退由`fd0fe51495626b39ce42d50dec29b3a6cef744b3`于2026-11-12引入。当前本地Fortran `thermosoil.f90:337–374`在可选override前将`use_refSOC`初始化为true；该源文件未跟踪，历史限制同上。此项是新增restart准入边界默认遗漏，不证明热初始化数值错误或标准算例回归。 |

这四组不应统一描述为“旧翻译遗留”或标准案例回归。前三组只确认JAX行为已存在于复查前Git边界；准确初始作者提交在可用历史中未知。第四组属于新加入helper的默认遗漏。原C1–C40的40项／22提交接受范围保持不变；新增C41–C43共享上述一个代码提交。所有生产修复的准确提交、验证范围和研究同步状态以累计登记为准。

## C44及两个未修clock候选的来源

本节的COLD-THERMAL-INPUT已接纳为C44；两个clock设计候选仍未修。原C1–C43的来源审计不重做。详细边界和哈希见交付时的[机器记录](../../outputs/reviews/main-physics-20261113/clock-authority-cold-inputs/cold-candidate-origins.json)及[说明](../../outputs/reviews/main-physics-20261113/clock-authority-cold-inputs/cold-candidate-origins.md)，候选ID保留以供追溯。

| 草案ID | 来源分类 | 有界结论 |
| --- | --- | --- |
| `COLD-THERMAL-INPUT` | 复查前已存在的cold source-default/static-input adapter遗漏 | 两个cold `use_refSOC`缺键`FALSE`分别用于静态读取门及cold initializer实参，均由`git blame`追到初始JAX基线`7333b46c`，且存在于`7cae3e02`。新restart helper的C42是另一个review-era问题，已在C41–C43接受提交中修复，不在本候选重复计数。当前Fortran source未跟踪；无标准case数值回归结论。 |
| `CLOCK-YEAR-LENGTH-DESIGN` | 复查前JAX表达式用day单位`one_year`除以second单位`dt_sechiba`，与恢复IOIPSL duration语义不同，活动影响未定 | JAX用`rint(one_year / dt_sechiba)`；恢复的IOIPSL `tlen2itau('1Y',dt,date0,...)`按实际日期转换。noleap、Jan1、dt=1800的源算术为17520步，JAX表达式为0；这是公式推导，不是运行观察。历史001的date0/calendar仍缺。 |
| `CLOCK-GREGORIAN-EPOCH-DESIGN` | 复查前JAX Gregorian helper与恢复IOIPSL日序epoch不同，活动影响未定 | JAX用astronomical Julian split；恢复的IOIPSL Gregorian日期路径使用Lilian日数及`2299160`偏置。历史001的date0及source-to-archive桥仍缺。 |

上述cold项现为C44，按一组cold控制／静态输入接口修复登记，相关satsoil传参和多caller不重复计数。clock两项仍为未修候选，没有生产日历修复或原输出失败结论；既有C1–C43的接受范围保持不变。
