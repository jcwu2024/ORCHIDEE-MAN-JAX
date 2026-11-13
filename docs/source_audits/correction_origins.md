# C1–C40修正来源审计（2026-11-13）

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
