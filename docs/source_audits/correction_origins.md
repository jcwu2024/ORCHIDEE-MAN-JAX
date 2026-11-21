# C1–C81 修正与支持补齐来源登记（2026-11-21）

## C81 叶片内部 CO₂ 的逐层重启状态

累计 **81 项修正或支持补齐、42 个生产提交**。其中 79 项有复查前来源证据，C42 和 C57 为复查期间遗漏。C81 的生产、测试与 CI 提交为 `64d13e4d71bce938f4d0ebac80d83744eb143020`。

原 `diffuco_trans_co2` 先清零整个 `leaf_ci`。裸地保留零，未同化植被点的各层取该点大气 CO₂ 浓度 `Ca`，同化点上实际参与求解的叶层写入叶内浓度。`diffuco_finalize` 在启用 CO₂ 时保存全字段。复查前 `7cae3e02adad23674767a590fdf4d27f239ce065` 的 `diffuco.py` 第 2903 行给同化点未计算层也填了 `Ca`，`driver/orchestration.py` 第 3055 行仅更新选中 PFT，使裸地和已知缺席的植被槽沿用旧值。两处一起决定实际重启字段，登记一项。

36 组同输入，旧例有 319 个 PFT14 坐标和 9,360 个裸地及缺席背景坐标偏离原值。修后 eager 和 JIT 各 10,080 个重启坐标与原程序匹配。319 处都在本次冠层平均值的读取掩码之外，同批 11 个所测物理输出均通过。其他实际存在的植被类型需要各自的求解证据，原 IOIPSL 后端及历史、年度 GPP/NPP/DOC 影响仍无本包验证。[主控验收](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase10-canopy-energy-and-consumers/controller-acceptance.json)

## C80 论文构建的分解活动下限

2026-11-21，第九包验收通过。累计 **80 项修正或支持补齐、41 个生产提交**。78 项在复查前已有相应问题或支持缺口，C42 和 C57 为复查过程中的遗漏。C80 的生产与回归提交为 `889024a213e1e665d3141668a0be5caf174814ea`。

原 `microactem` 在温度和泥炭湿度下限使用 `EPSILON(0.)`，论文 `-r8` 构建取 binary64。只读检查复查前 `7cae3e02adad23674767a590fdf4d27f239ce065` 可见，论文预计算入口及编译折叠调用分解控制时没有传入精度，两处分解计算都继承了 `microactem` 的 R4 默认。两个非编译驱动也沿耦合入口继承该默认。首次引入日期未知。

修复将论文所需的 R8 值传到两次计算，保留通用接口的 R4 默认与显式选择。暖干 PFT14 首个活动率从约 `5.77e-12` 降至原参考的 `1.08e-20` 日⁻¹。只更换三个活动率输入即可复现旧组件结果。凋落物和土壤碳对照及实际状态写回已通过原门限。暖干原土壤碳局部收支警告仍出现。本包没有年度 DOC 或 NPP 影响估计。[主控验收](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase09-carbon-activity-and-consumers/controller-acceptance.json)

## 第八包接受的 C78 和 C79

2026-11-21，主控接受第八包修订，生产与回归提交 `9591652668df1afb4955d7282a6091b96d844e4b`。累计 **79 项修正或支持补齐，40 个生产提交**。其中 77 项在复查前已有相应错误或支持缺口，C42 和 C57 为复查期间遗漏。C78 和 C79 的旧表达式均经只读核对存在于 `7cae3e02adad23674767a590fdf4d27f239ce065`。首次引入时间和历史作业受影响情况未知。

| 编号 | 源规则与修复位置 | 同输入反例及验证 |
| --- | --- | --- |
| C78 | `hydrol.f90::hydrol_soil_coef` 8255–8280 先计算实际可用水分，再除以饱和与残余含水量之差并取整选系数。共享 `jax_orchidee/sechiba/hydrol.py::_frozen_cwrr_coordinates` 为矿物土、泥炭土及公共选择函数保留各步浮点结果。 | 原冻结输入在批量计算时，案例 8 顶层应取第 25 档却取 26，案例 12 应取第 26 档却取 25。修订后两例的普通与批量 JIT 均选中原档位，实际含水量写回通过原容差。180 个原生输入在五种执行形式下得到 1,080 对相同的可用水分和箱号。 |
| C79 | `stomate_permafrost_soilcarbon.f90::microactem` 2644、2660 用湿度除以 0.02，取整并限于 1–45 档。`jax_orchidee/stomate/permafrost.py` 按 binary32、binary64 分别计算能进入下一档的最小可表示湿度，供共享查表函数使用。 | 两个原输入在批量计算时从原第 17、20 档误入第 18、21 档，使两个分解率字段共 46 个坐标超限。修订后各档边界及两侧通过原参考，5 种案例排列的 80 项活动字段比较通过。暖干条件下的活动下限是独立待办。 |

以后集中同步时按这两个共享过程、两份 Phase08 测试数据、普通及批量回归和 CI 配置核对。C78 的三个调用位置共用一种选表规则，C79 的两个下游字段共用同一查表函数。第八包首次修补和批量修订按这两个原因计数。当前证据限于本地 CPU，整模型运行成本、部署 GPU 和年度影响仍待验证。[修前来源](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase08-hydrol-snow-and-consumers/candidates-and-origins.json) · [主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase08-hydrol-snow-and-consumers/controller-acceptance.json)

## 第六包接受的 C73–C77

2026-11-20，第六包经主控接受，五项修补登记C73–C77，生产／测试／CI提交`f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b`。累计**77项修正／支持补齐、39个相关生产提交**；75项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏，不是77项物理公式错误。本轮五项均有复查前`7cae3e02adad23674767a590fdf4d27f239ce065`只读源码证据，首次引入日期未断言，未恢复运行历史代码。

| 编号 | 修正 | 原源与原因 | 位置、验证及边界 |
| --- | --- | --- | --- |
| C73 | allocation源默认上限 | `constantes_var.f90:1050`及`constantes.f90:1217–1223`默认0.5；公共allocation遗漏参数时旧值0.6。恢复0.5，显式0.6仍支持。 | `stomate/carbon_kernels.py::allocation_step`；同14元组120/2611失败→2611通过。driver显式读配置，不能推断全部历史paper受影响；Phase05明确0.6参考由runner显式保留。 |
| C74 | post-soil凋落物写回 | littercalc后soilcarbon会修改共享litter_below；三处旧消费者取了较早littercalc返回，丢失非零冻融更新。 | `driver/orchestration.py`半小时／日packet及`stomate/integration.py`LPJ诊断统一取soilcarbon返回；原3-call、实际下次调用及日末carry。原littercalc阶段对象不改。 |
| C75 | 短forcing间隔SWdown | `readdim2.f90:1585–1618`在dt_force≤3600走线性插值或NETRAD当前样本，旧通用路径仍走太阳角处理。 | `driver/domain.py::_forcing_model_step_from_cached_point`；hourly同输入旧两向量失败、修后378字段／向量比较通过；WATCHOUT及长间隔原分支保留。 |
| C76 | 年度标量负值预处理 | `slowproc.f90:6387–6396`先用原值>-9000判有效，再把负值截为0；旧reader漏掉此顺序。 | `driver/static.py::read_slowproc_annual_bbox_field`；bbox／nearest同输入原参考与实际salinity控制。所测最终salinity受0.5下限保护，没有据此声称同化影响。 |
| C77 | 时间字段首槽mask | `slowproc.f90:6115–6118`仅首时隙决定整地点有效性，后续各时隙按同mask传输。 | `driver/static.py::read_slowproc_time_bbox_field`；原IO反例3/-4530/2与旧4/40/1区分，sentinel仅用于IO见证，物理消费者另用合法元组。 |

[年代记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-origin-check.json)保留精确旧函数，原候选／历史回执不改。按五个独立原因计数，C74三处writer不拆成三个修复；本包reference reader偏移／测试工厂／可移植schema复制等交付过程问题不另记历史生产bug。

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-acceptance.json)核对46份产品、交付与独审封存身份，并复核配对原源、实际写回及历史源码。保存的隔离CI按精确节点合并为454通过、9项科学strict xfail、10项外部身份skip、0失败／错误；这不是一次不间断全通过运行。首次漏复制已跟踪restart schema的9个setup错误保留，修正复制后仅重测受影响的10节点，未重测此前通过项。配置内Ruff通过，生产lint 62前／62后无新增；hosted CI未运行。主控解析JUnit并复用owner／独审验证，未再积分；12份JSON仅Git换行规范化、语义相同。[发布记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-publication.json)保留最终版本。

保存的嵌套JIT整返回差异现有有界递推取整与表面通量抵消证据；13/14 soilflx_pft及其别名仍严格失败，不声明唯一HLO／机器指令原因。两个新增合成相等浓度DOC元组捕获到入口约1e-17取整变化，实测入口重放能复现严格相等分支变化及4／5个失败坐标；没有定位最早上游算术，也不外推旧年度轨迹。新8项JIT完整系数通量门／60坐标仍未归因。旧HYDROL／热／雪／低光／DOC／NPP／grain严格门各守原范围，不同配置DGVM诊断不计新修复。

52父过程的整体路线等级仍为source_mapped／open；4,059词法候选、565源审查组和187跨模块／外部调用组不等于数值覆盖或缺陷数。新有限证据包括14生长元组2611比较、扩散实际arm、非零冻融carry、冠层／capacity前驱、forcing／静态输入／modelout／routing和6份科学restart文件往返（1402检查，含147独立字段）。真实IOIPSL／native producer、maintained完整调度、自然历史、私有peat／PBL、原未定义合同与完整源分母仍有缺证。C64限npts1／nstm6；年度DOC真实边界／残差／异养呼吸均为null，原mass warning与负中间量保留。

下一包优先实质审阅[565源组和187调用组](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/denominator/remaining.json)：解析真实callee与目标域，复用准确旧证据，明确排除依据，对确有缺口补独立原参考和实际compiled／写回。并行归因新8项完整系数通量失败，不重开本包已闭合的嵌套与DOC有界机制。停止标准仍见[阶段计划](physics_branch_closeout_plan.md)；本地可做项完成后再汇总真正外部缺项，不以扩大扫描清单替代验证。整模型综合验证尚未启动，`numerical_acceptance=false`；参数／状态90＋1＋1／92及93／126／181口径保持，研究树同步暂停。

## 已接受快照：C67–C72与第五包范围

2026-11-20，主控接受第五包六项独立修正／支持补齐，登记C67–C72，生产／测试／CI提交`b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12`。累计**72项修正／支持补齐、38个相关生产提交**；70项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏，不是72项物理公式错误。本轮六项均经只读`git show`确认存在于复查前`7cae3e02adad23674767a590fdf4d27f239ce065`；首次引入日期未作断言，未恢复或运行旧模型。

| 编号 | 修正 | 原源与原因 | 位置与验证范围 |
| --- | --- | --- | --- |
| C67 | NPP裸土SLA更新范围 | `stomate_npp.f90:594–642`的SLA更新限j=2..nvm；旧通用clamp改写PFT1，现保留其入口SLA。 | `stomate/carbon_kernels.py::npp_leaf_age_sla_age_update`；原参考、裸土carry及clamp回归 |
| C68 | 日碳源控制量静态编译合同 | prescribe的firstcall／stomate_restart_none及constraints的agriculture／undef从动态叶提升为静态配置；保持源默认值与显式值。 | `stomate/integration.py`静态与outer编译适配器；真实外层JIT回归，调用者需保持配置静态 |
| C69 | 分配后LAI／slai相位 | 原LPJ分配后只更新slai，NPP／turnover仍消费入口lai，末尾才更新lai；修正三条组合入口。 | `stomate/integration.py`三条日碳链；字面原caller960比较通过，原adapter参考已降为诊断 |
| C70 | 凋落物源小量判断常数 | 源min_stomate=1e-8漏入组合核，旧generic默认0；现向pool混合和两类fuel同步传原常数。 | `stomate/carbon_kernels.py::littercalc_leak_core_with_controls`；低于／等于／高于边界、真实writer再入 |
| C71 | 季节日期先后顺序和写回 | 原太阳最低值日期先更新再供NGD；现由共同前驱接收latitude／calendar，并防止旧packet覆盖新日期。 | `coupled.py`共同season前驱与`driver/orchestration.py`调用／写回；实际前驱、packet及下一实参 |
| C72 | 季节小量归零精度 | 源EPSILON(zero)的zero为REAL(r_std)，参考r_std64；旧代码用float32 epsilon，现恢复float64精度。 | `stomate/season.py`三处归零helper；保留微小非零值的原参考。permafrost常量未改 |

[复查前源码证据](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-origin-check.json)逐项保存原函数及来源身份。按原因登记，不按多个caller重复计数；候选开发期间修正的测试输入位置、旧adapter相位参考和元数据兼容问题不新增历史bug编号。

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-acceptance.json)核对30份当前产品及原交付／独审封存身份，复用owner与独审证据。最终隔离六文件回归260通过、4个外部原源／历史资产身份门跳过；实际边界13通过、来源门2通过，相邻62通过分别保留，不相加为科学覆盖。新11份Python lint通过，生产旧诊断64前／64后未增加；hosted CI未执行。主控只解析回执、读配对源码及实际caller，没有重复积分或数值套件。5份新JSON的Git规范化仅为换行，解析值相同；发布状态见[记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-publication.json)。

本包接受有限组件、源条件与真实写回／下一实参；不宣布PB各组、完整源分母或物理阶段完成。保留原HYDROL／热／雪／低光／DOC严格失败及已接受浮点结论的原范围。嵌套JIT完整返回13/14项失败、最大绝对差5.34106093e-12仍未归因。fixedsnow原未定义值、私有peat／外部PBL、native partial、自然联合／雪能量前驱及DOC原始通量／floor观测等缺证继续具名保留；年度DOC真实残差与异养呼吸仍未知，不由端点推算。C64仍限npts1／nstm6。

下一包按[具名剩余动作](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/remaining.json)并行核对目标源条件分母、生长／扩散等剩余算术分支、嵌套JIT整返回差异和PB-05跨组数值接口；可在现有本地组件边界内补实际前驱，不能把合成元组称为自然轨迹。复用已接受参考，不重开已闭合浮点调查。整模型综合验证尚未启动，`numerical_acceptance=false`。参数／状态90＋1＋1／92及93／126／181口径保持；研究分支同步继续暂停。

## 已接受快照：C65／C66与第四包范围

2026-11-20，主控接受C65（编译热过程保留原冻结profile中止检查）和C66（初始化先判整组重算，再校验保留字段），共同生产／测试／CI提交`d2d48e6d5007079493cf91cca21c978eb79fa24a`。累计**66项修正／支持补齐、37个相关生产提交**，其中64项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏；不是66项物理公式错误。两项源行为均见于最早跟踪7333b46c及复查前7cae3e02，不能追称导入前的首次引入日期。见[来源登记](correction_origins.md)及[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)。

| 编号 | 原源、旧行为与修复 | 证据与边界 |
| --- | --- | --- |
| C65 | thermosoil_getdiff:2802–2809在PFT冻结profile不一致时中止；旧Tracer路径跳过检查。共享functional check经真实module／scan／block保留，host消费前拒绝 | 原源保存反例、修前JIT错误返回成功、修后拒绝；合法输入NSLM原断言保留。反例含合成新表温，不能称自然轨迹失败；不改变计算公式 |
| C66 | thermosoil_initialize:665–740任一系数全缺失会重算九字段；旧代码逐字段拒绝mixed marker早于组决定。现先判是否整组覆盖，仅保留组继续严格校验 | 原源43输出、实际profile／energy消费者通过；仅公共API安全覆盖元组，native partial producer／完整driver可达未证；形状、有限性及ptn／refSOC政策不变 |

同一C65下的module／scan／block接入与本次NSLM测试迁移不按调用点新增编号；NSLM遗漏属于尚未接受候选的集成收尾，不是第三项复查前漏洞。原两行为在`7333b46c0b38650fb6c9250582876137831657b8`与`7cae3e02adad23674767a590fdf4d27f239ce065`均有直接源码证据；导入前真实首次日期未知。[年代证据](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/integration/history-check.json)与原交付保持。

[主控验收](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-acceptance.json)复用原包修前2失败／修后隔离13通过＋1原源身份门skip及本地源门，并核对[集成修订](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-repair/README.md)：NSLM原compiled回归的输入、断言和容差保持，新增真实内层module JIT／外层checkified窄owner的合法材料字段及非法拒绝，两环境各3通过、无skip。11份当前及冻结产品、原／新封存身份匹配；主控本次未重跑数值。原源码与科学封包保留，Git对两份新JSON仅规范换行，解析值相同。发布见[记录](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-publication.json)。

C65改变纯explicit热helper的编译合同：checkify后JIT，返回Error／result并在消费前check_error；module／scan／block与维护测试成组接入。C66仅接受有任一整字段缺失、因此九系数整组覆盖的公共初始化输入，native partial restart及完整driver可达尚未证明，保留组的混合marker仍拒绝。旧C64源等价仍限npts=1、nstm=6。原HYDROL100 eager＋93 JIT、固定热／雪21和不同输入handoff2严格次数不变。新增完整返回的嵌套JIT比较另有13/14项失败、最大绝对差5.34106093e-12，成因未定；窄接口通过不等于整状态等价。

未来集中同步须成组评审`sechiba/thermosoil.py`、`driver/orchestration.py`、两份新增原参考fixture、source_conditions／NSLM／scan回归和CI。私有explicit JIT返回由结果改成Error／result；研究自有外层编译入口须接入相同错误合同，不能整文件覆盖。以新的不可变部署重建受影响编译缓存，单独验证CUDA／AD／训练；本包未认证这些模式，也未读取、修改或同步研究树。旧Teacher数据与运行中部署保持原身份。

水文normal／ALT残差与独立雪groundflux现有完整同输入、内部首差及传播链证据，接受有界浮点归因，不改物理公式、求和或科学容差。dry／joint热、NPP／DOC／grain的旧结论保持各自范围。新嵌套JIT完整返回差异不能自动套用上述归因。私有peat历史build、自然初始雪包／第二energy容量及native partial restart等缺证仍在；PB-03、完整源分母及整个物理阶段尚未完成，numerical_acceptance=false。

## 已接受快照：C64冷启动逐土壤块强制水位

2026-11-19，主控接受`HYDROL-ZWT-FORCE-TRANSPORT-01`，登记C64，共同生产／测试／CI提交`e275a33fd5864957e9b241138359c1f6001a5756`。累计**64项修正／支持补齐、36个相关生产提交**；62项有复查前错误或支持缺口证据，C42／C57仍为复查期间已接受代码的遗漏。

原`hydrol_init:2685–2688,2834–2850`及`sechiba_io_p::r21setvar_p:389–444`在整数组未恢复时读取完整ZWT_FORCE向量。旧真实cold入口仅取__00001，构造器再广播首值；完整[.25,.5,.75,1,1.5,1e20]变成六个.25，首块未强制时还可能丢掉其他块有限水位和zforce激活。两个丢失表达式均见于复查前7cae3e02与最早tracked导入7333b46c，外部导入前的首次日期未知；同一原因跨两入口只计一项。

修复在共享soil resolver、真实cold入口和构造器间保留完整逐块值；完整索引或恰好nstm个字面值可用，重述1e20默认的压缩日志保持等价，低层scalar API保持。部分非默认、混合形式、singleton raw、无效索引／非有限值明确拒绝；原IOIPSL词法与优先序没有被认证。**源等价验收仅限独立单点npts=1、nstm=6**：原setvar先判断第一轴，npts==nstm==6会走不同SPREAD，不能由helper接收shape外推多点等价。restart／部分恢复不被cold覆盖。

同核心旧检查3失败／1通过，最终无原Fortran隔离uv19通过／0skip，64个项目导入均来自隔离副本；包括真实cold reader／constructor和eager／窄JIT饱和消费者。原typed setvar三例、默认兼容的中间失败及修正保留。主控核对9份产品／冻结副本、配对原源及独审，复用已有验证，未再运行物理；注释后改动AST相同。详见[源和年代](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase03-hydrol-thermal-closeout/source-conditions/report.md)、[主控验收](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase03-hydrol-thermal-closeout/controller-acceptance.json)与[发布](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase03-hydrol-thermal-closeout/controller-publication.json)。

同包dry／joint热通量已接受为有界内部浮点计算和相消放大，**不增加热模型修复**，16个热严格坐标仍失败；雪groundflux和水文首差／5个ALT坐标未解，不能共用此结论。新冻结／最终热钳制消费者证据按各自边界接受，PB-03、整个阶段和numerical_acceptance仍false。未来同步C64时须保留C48／C49／C55／C62／C63的时钟、初始化、参数和控制合同，以及单点验收边界；两研究分支未同步。

## 已接受快照：C62／C63水文潮汐与有效检查开关

2026-11-19，主控接受基线`dfc67a8a94f4a750eb25251d1435ce856efa573c`上的PB-03组件包及生产集成修订，共同生产／测试／CI提交`7b35196462fbfd53f531fd1dc751699092f80104`。正式累计**63项修正／支持补齐、35个相关生产提交**；61项有复查前错误或支持缺口证据，C42／C57仍是复查期间已接受代码的遗漏。原土柱求和顺序候选已撤回，不算修复。

| 登记 | 必要性与年代 | 接受范围及限制 |
| --- | --- | --- |
| C62：HYDROL-TIDE-TRANSPORT-01 | 原`dim2_driver.f90:147–151`打开两文件，`hydrol.f90:5541–5542,5645–5655,6276–6288`逐次读取并使用独立水位及变化量。复查前`7cae3e02adad23674767a590fdf4d27f239ce065`与最早tracked导入`7333b46c0b38650fb6c9250582876137831657b8`的wrapper丢弃已提供潮汐值且driver缺自然producer。旧falling tuple的tmc误差约0.24，water2infilt／wt_ab_tide误差0.24；首次库外引入未知 | 补齐两份F10.3原单位文本、明确native-run记录原点、source clock、实际cold／ordinary／first-main／compact／compiled消费者和wrapper；新native运行从首记录打开，continuous保留原点。历史TIDES=false日志不算活动潮汐轨迹，接口替身不算整模型验证，缺文件／冲突／EOF明确失败 |
| C63：HYDROL-CHECK-CWRR-01 | 原`constantes_soil.f90:463–473`默认false并读取override；`hydrol.f90:5662,6335–6367,7363`门控qflux及wat_flux。上述两个旧Git快照无条件计算诊断通量且没有完整有效开关传递；旧mineral_rain_canopy的qflux误差0.5451314626606469、refsoc_mixed_freeze为0.13113306982923245 | false源零通量合同及override贯通准备、owner、缓存查询和返回callable。完整true还缺源守恒／ET检查，明确拒绝，不能只计算qflux就称支持；低层独立诊断API的历史默认保留。未证明年度DOC影响 |

原缺输入拒绝是复查期间的合同变化，与更早丢值／缺producer分开；本修订开发中的接线补漏不另加旧bug。旧generic harness的soilc_max与peat albedo常量错误只按已激活输入限定影响；经核查不撤销C13／C17／C18／C55已接受范围。新water-depth采样相位修正的17个保存值恰好相同，不虚构数值反例。详见[年代及旧参考影响](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/provenance/report.md)。

主控核对22份当前产品及冻结副本、独审／manifest／patch绑定和求和函数原文恢复，复用已有验证：隔离本地uv 135通过、195精确坐标strict xfail、4身份门跳过；另55输入／实际adapter、87受影响邻接检查通过。四份JSON夹具仅经Git换行规范化且值相同。严格失败仍失败，不以数量计算bug，未重跑模型或原参考。完整true、私有peat producer、热／雪及其他物理条件仍在剩余表；`numerical_acceptance=false`。见[主控接受](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/controller-acceptance.json)与[发布记录](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/controller-publication.json)。

未来同步需按共同提交、source-clock／状态与C43／C48／C55／C60既有依赖成组核对，不能仅复制tides helper或整文件覆盖研究改动；两研究分支仍未同步。

## 已接受快照：C60／C61与参数状态收尾

2026-11-18，基线`a3551dac4dbb2f53bee2ef47c80b4995eb57af82`上的WATCHOUT原包、R1–R5修正与R2公开历史续包合并接受，代码／测试／CI提交`aa14d04210ab371aae6e959da193561f37f55d0e`。正式累计61项修正／支持补齐、34个相关生产提交；59项有复查前错误或支持缺项证据（含C60），C42／C57为复查期间已接受代码的遗漏。不能把61解释成61条当时均激活的物理公式bug。

| 登记 | 必要性、年代与证据 | 修复和接受边界 |
| --- | --- | --- |
| C60：WATCHOUT-TRANSPORT-01 | 完整复查前`7cae3e02adad23674767a590fdf4d27f239ce065`与a355快照中高层forcing／cache缺字段、driver拒绝该模式，低层参考不能证明高层已接通；这是旧支持缺口，不虚称旧可执行路径曾产生数值失败。准确最初引入未知 | 文件schema／geometry／高度／cadence与有效控制→插值／bundle→DIM2／intersurf→DIFFUCO／ENERBIL→compiled动态tuple和source条件driver9／13读写。actual first-substep初始化、once-only SWnet、static drag及standard／cold／native公开context交接一体接受；RELAXATION原fatal保留 |
| C61：FORCING-FIRST-READ-01 | 7cae `forcing_interpolation.py:264`与a355都有Python `round(86400/dt_force)`；原`readdim2.f90:1435–1452`要求NINT及完整首日。19200秒得到4.5时旧选record4、原源选5；四条20000秒虽index合法却不足源完整首日。新cadence接线暴露旧表达式边界，首次引入未知 | 正数NINT半值远离零，源完整首日guard恢复；实际public bundle及file输入验证。原插值公式、参考与容差保持；默认长度／TIME_LENGTH的新接线归C60，不按调用点增加C号 |

来源：[原运输缺口](../../outputs/reviews/main-physics-20261118/watchout-parameter-transport/origin-binding.json)、[R1–R5年代](../../outputs/reviews/main-physics-20261118/watchout-parameter-transport-repair/origin-binding.json)、[R2交接](../../outputs/reviews/main-physics-20261118/watchout-context-handoff/origin-binding.json)。本地Fortran未被旧Git提交跟踪，不伪称恢复其历史producer。

C60开发期间发现的SWnet assembler、compiled首tuple、NETRAD／SPRED入口、writer变量／模式、raw预读误当初始化、局部context丢失、old count与new split混用、非WATCHOUT override实参、continuous缺历史fallback和reference漏返回，均是**本轮未接受支持实现中的集成遗漏**，合并在同一链并保留修前证据，不倒写为7cae已有，不追加独立旧bug。夹具、缓存身份和lint修正另外记录。

最终公开交接同输入16失败／19通过→35通过，另既有7项通过；之前99＋1与原113各保留当时版本。完整stage反例与终审保留，主控复核未重跑数值；提交时3测试仅移除空EOF，AST不变。见[最终验收](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-acceptance.json)。参数／状态有限工作面已完成，完整科学接受仍false；未来两研究分支按共同commit及基线依赖集中核对，尚未同步。

## 已接受快照：C55／C43扩展与C56–C59参数状态交接

[交付包](../../outputs/reviews/main-physics-20261118/parameter-state-closeout/README.md)基线`db8dd72eea30e3c2dc30dd3b7368a664d5a56058`，共同代码／测试／CI提交`5fe049ffa249f2c53daf669d576fe590ecbbb999`。按概念修复登记为C55与C43扩展及四个新编号，不能把六候选组机械算成六个新bug。该包接受时累计59项修正／33个相关生产提交，其中57项有复查前证据，C42与C57为复查期间引入。原精确来源、失败和限制见`origin-binding.json`及[主控验收](../../outputs/reviews/main-physics-20261118/parameter-state-closeout/controller-acceptance.json)；没有执行旧整个模块或旧模型。

| 登记 | 年代及旧反例 | 修复与接受边界 |
| --- | --- | --- |
| C55扩展：HD-DEPTH-PARAM-01 | 旧初始化接受深度覆盖，runtime重建却回默认，bottom kfact的0.07243975703425146回到0.1；同一参数传递链的扩展，精确首次引入未知 | 七活动标量＋一inactive metadata独立合同，贯通表／缓存／系数／driver；十向量合同不变；保留源AKS_POWER检查nk_rel及注释分支 |
| C56：HD-REFSOC-LIFECYCLE-01 | 复查前7cae exporter强制false、IO未识别条件字段；db8的[42500.125] tag丢失或被拒；首次引入未知 | 条件读取／ALLmarker初始化、双镜像carry、finalize和真实小文件运输；新resolver不是历史已有API |
| C57：COLD-CLOCK-CONTEXT-01 | **复查期间**`782535b4130f2ddc8cd30a251e0493709e1ab26a`（2026-11-14）引入ready出口未定义context；原局部AST两分支NameError | 已有prepared identity保留，否则从cold bundle建clock；上游完整物理ready未验证，不称全部冷启动已通过 |
| C43扩展：SCAN-CONTEXT-01 | 7cae的scan cache key漏捕获context identity，真实窄JIT的A/B串值；原C43已修later-day block，本次扩到scan；精确首次引入未知 | identity入key并由closure持有；同形状动态参数仍正确复用，无整模型数值声明 |
| C58：BLOCK-DISPATCH-01 | 7cae已有state派生phenology mask而block key未含它；同context／layout、不同mask复用旧分支；首次引入未知 | immutable dispatch同时入key和closure快照；真实窄compiled block使用物理witness |
| C59：STATE-TEMP-GROWTH-01 | 7cae已漏half-hour carry；782535b4后来新增必需calendar consumer，两历史事实分开；实际原clock入口可复现丢值 | 中央carry补字段，源slowproc588的pre-STOMATE读取和stomate5049日更新顺序不改；不声称旧标准例当时已失败 |

独立终审另发现**本包新集成**的disabled初始化仍carry旧refSOC，已保留两条真实红反例并在文件首writer／new_native双镜像清除。此子问题归C56同一链，不单列新C号，也不倒写成7cae已有。受影响36节点最终按35通过＋一项夹具修正后定向通过；生产仅改两个初始化owner，连续路径和strict exporter不改。

四项IO run.def夹具依赖、LRU对象identity断言、NamedTuple误用dataclass属性及空HYDROL组件访问均是验证设施问题，原失败保留，不计作模型缺陷。主控仅另删测试末尾空行，执行AST不变，无数值复跑。当前94节点最终通过是组合证据，不是单次全绿或hosted CI。

WATCHOUT尚需本地source→driver运输补齐；原封存包将外部producer合同视为必须先解决的阻塞，由补充源审查修正。真实外部producer身份未知继续保留，不登记为已证实的模型bug。研究同步暂停，准确文件／依赖见[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)。

## C53–C55 已接受来源

2026-11-18主控接受C55=`HD-SOIL-PARAM-01`，[参数链包](../../outputs/reviews/main-physics-20261117/hydrol-soil-parameter-closure/README.md)代码／测试／CI提交`474f52088be77a543ebf181b232acfc8a02ddc02`。Fortran hydrol_init2069–2252设置同一组矿物参数，var_init4071–4088派生有效点阈值，soil6395–6741／diag_soil9089–9284持续消费它们。旧初始化可用自定义值，模板／wrapper／诊断却回到USDA默认；原case1／5的drysoil_frac和vegstress四失败归一项参数传递遗漏。五例eager／真实JIT各191／191通过，原输入／参考／容差保持。复查前7cae3e02已有相关函数与硬编码默认，首次引入未知、旧整模型未重跑；见[来源核查](../../outputs/reviews/main-physics-20261117/hydrol-soil-parameter-closure/controller-origin-check.json)与[主控验收](../../outputs/reviews/main-physics-20261117/hydrol-soil-parameter-closure/controller-acceptance.json)。本轮也补齐同一十向量的refSOC、缓存及实际调用支持，不能声称全部配置或论文自然轨迹原来失败。

2026-11-17主控接受[水文／冠层包](../../outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/README.md)的两项有限接口／执行修复，登记C53=`HD-FWET-01`、C54=`HD-JIT-01`，共同代码／测试／CI提交 `139e1ba7ba2db46d6f1a3b00f0592335bf40cafa`。C53按hydrol_soil:7307–7342保留当前TOPMODEL湿地诊断，仅在没有新值时沿用旧值；原ordinary／runtime／fast布局曾将0.412613023955767覆盖为0.123。C54按diffuco_trans_co2:2214–2234在显式参数缺光照表时提前调用既有静态布局helper，修复真实JIT层数转换异常。两项在`7cae3e02adad23674767a590fdf4d27f239ce065`已有精确源码证据，首次引入未知；新参考设施不伪称旧版已有。前者当前wrapper TOPMODEL=True仍未贯通，后者论文默认getter本来预供表，故不声称论文自然路径曾触发两项失败。主控复用独立原源、19项交接和10项隔离回归／Ruff证据，无数值重跑；见[验收](../../outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/controller-acceptance.json)及[发布](../../outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/controller-publication.json)。当时水文四项自定义参数差异另列；现由C55同一参数链闭合。

C55接受时累计55项修正、32个相关生产代码提交，其中54项有复查前证据、1项C42属复查新增helper的默认遗漏；当前累计以上方C1–C66登记为准。C53为状态选择、C54为有限编译入口修复，计数不等同活动物理公式错误数。原C1–C52来源和接受范围保留；C51／C52共同提交 `9a856c6cf859437d2436f4f3748d4c9e770838a1`。C49初次提交 `ea75bde31012cf519d4b0fa5c689d865c9d9c2b7` 与US_INIT扩展 `d6bad452932d1a7179ff5ab5fd80c771da89e336` 仍算同一初始化链。

C45–C47共同提交3f7b482765b7b41339ee56a818b1f164490a5d1e，具有配对源、保存旧反例和新identified native参考；不恢复历史001 producer。详见[累计登记](shared_core_branch_review.md#累计修改位置与提交索引)和[helper接受记录](../../outputs/reviews/main-physics-20261114/native-clock-helper/controller-acceptance.json)。旧C41–C43提交8a772de6f4977018a875093d535b6088317716c8、C44提交9c15c042a5ade9c91d91ac91a8c0ea39ed87c8f2保留。生产接线C48见下节，不并入这三个已接受helper修复。

## C51–C52：完整热组件及driver蒸发单位边界已接受

基线main `82d622722feea3cfe5034d8b15f98cda78e04872` 的[整包](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/README.md)分别保存两个复查前来源。主控于2026-11-17接受为C51、C52，共同代码／Oracle／fixture／CI提交 `9a856c6cf859437d2436f4f3748d4c9e770838a1`。见[主控验收](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/controller-acceptance.json)及[发布记录](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/controller-publication.json)。交付时pending回执保留其原始身份。

| 编号 | 复查前证据与最小修复 | 验证／边界 |
| --- | --- | --- |
| C51 thermal final coef裸地权重 | `7cae3e02`两个full-step final coef实参均为raw veget_max；源thermosoil_main1002–1009传veget_max_bg。仅两个实参改现有helper，C50 energy调用保持 | [来源](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/thermal/origin-binding.json)；修前52失败字段，修后原8参考eager／JIT552字段过；不同于C50的energy消费者，两个coef调用点合计一项 |
| C52 driver vevapp amount→rate | `7cae3e02`共享writer直接复制ENERBIL每步amount；源intersurf743–749除xrdt后才返DIM2。新增纯除法helper、实际dt元数据、共同writer一次转换 | [来源](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/integration/origin-binding.json)；非零原源与实际writer／单点IO通过；SECHIBA evapora仍amount，peq外部producer gap不因此关闭 |

只执行保存旧writer的单一函数AST负控制，未import／执行整个旧模块。两项的精确首次引入提交均未知；当前新建参考基础设施和dt metadata接线不伪称旧版已有。积雪没有新确认翻译错，O0／完整JIT grain严格失败独立保留，不能用strict xfail增加通过数或新C号。主控将该xfail限制到已知case的有限grain差异，其他字段先正常断言；定向1项仍为xfail且Ruff通过，属于测试保护，不增加模型修正数。没有整模型运行、研究树同步或NPP／DOC改善结论。

## C50：thermal energy消费者的裸地权重已接受

2026-11-16，[独立thermal包](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/README.md)从原`thermosoil.f90:602–603,913,2442–2455`确认energy必须使用补足裸地的`veget_max_bg`。复查前`7cae3e02adad23674767a590fdf4d27f239ce065`及本包基线`551cc8cdbc4f16fa37ed2c89f260c36b57c596a8`的`thermosoil_explicit_step`、`thermosoil_no_explicit_snow_step`均传raw `veget_max`；kernel直接按入参加权，未做补足。已保存两个完整文件及实参AST，[来源绑定](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/origin-binding.json)证明复查前已存在，精确引入提交未调查；旧整个模块／模型未执行。

独立本地原Fortran稀疏案例`veget_max[0,[0,1,13]]=[.05,.125,.25]`，补足后bare=.625：原源`surfheat=216899.3529053959, coldcont=0`，旧JAX eager／窄JIT反置，4个energy字段失败已封存。生产仅改两个调用实参为`thermosoil_veget_max_bg(veget_max)`，kernel接口和其余执行AST不变；不是C34初始化helper修复，也不按调用点分别计两项。修复后288字段对照、真实隔离thermal组件／carry及95通过／2旧控制skip见[验证](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/verification-results.json)。没有据此声称PFT14标准案例、NPP或DOC改善。

主控于2026-11-16接受为**C50**，代码／Oracle／fixture提交 `6a5d4a7f2092f32ef9de064e61ee3224c5b48b01`，见[验收](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/controller-acceptance.json)与[发布](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/controller-publication.json)。累计50项／29个相关生产代码提交，其中49项有复查前证据。新脚本与fixture是本轮建立的独立参考基础设施，不能倒写成旧版已有；验收另补脚本的已有目录日志保护，2项CLI检查通过，不另算模型修正。审查者早期误读daily同名文件后在main独立重核，未写入／同步研究树；范围纠正保存在本包审查记录。组件证据不改变`numerical_acceptance=false`或93／126／181分母，旧交付回执仍保留当时pending状态并由主控接受记录更新结论。

## 此前已接受：C49上游US_INIT支持域与实际入口补齐

基线main `512d2a1134b96586d41528aea0376e81fe42e72e`，见[当前包](../../outputs/reviews/main-physics-20261115/us-init-production-closure/README.md)及[来源绑定](../../outputs/reviews/main-physics-20261115/us-init-production-closure/origin-binding.json)。只读git show保存复查前`7cae3e02adad23674767a590fdf4d27f239ce065`的完整HYDROL owner、cold helper、USDA配置helper及raw_run_def通道：旧完整owner直接`jnp.sum(us,axis=3)`，这三个有界函数没有US_INIT setvar，证明该上游缺项在复查前已存在；精确首次引入仍未知，不外推为全项目负面结论，也未执行旧模块。

C49接受的是已物化us子域，并明确拒绝合法但尚未实现的整层marker分支。该guard是当时已接受的显式边界，不能改称今天新发现的旧版隐藏保证或新的C49失败。本包是源支持域扩展，加上新的scalar adapter、真实参数通道、完整state写回及年度consumer镜像接线；调用点不分别计旧bug，也不把今天新增adapter／接线伪称旧版已有。

主控于2026-11-16接受声明域内的上游补齐，代码／测试提交 `d6bad452932d1a7179ff5ab5fd80c771da89e336`，见[验收记录](../../outputs/reviews/main-physics-20261115/us-init-production-closure/controller-acceptance.json)及[发布回执](../../outputs/reviews/main-physics-20261115/us-init-production-closure/controller-publication.json)。复用唯一组合[84／84通过](../../outputs/reviews/main-physics-20261115/us-init-production-closure/verification-results.json)、独立终审和执行AST绑定，核对当前13个交付文件，无重跑。正式49项／28个相关生产代码提交，仍48项有复查前证据／C42新helper遗漏；本次作为C49已声明边界的支持域扩展，不另编号。账本90＋1＋1／92和93／126／181、numerical_acceptance=false保持。getincom本体、源NONE、历史producer／构建、DOC／thermal／maxmass／peqAcoef与整模型门不变，研究同步暂停。原C49、此前源／通过／失败工件95个原字节保留，以下C49接受快照不回写。

## C49：vegstress源初始化与实际入口接线已接受

2026-11-15，本包基线main `194d91350bf375f2c9e49360401b807e07d8a02e`；[上一包](../../outputs/reviews/main-physics-20261115/vegstress-read-path-closure/README.md)取得同树lookup／gather源码，闭合源whole-local ALL999999原值加S，并保存旧真实owner失败和21项检查。[来源绑定](../../outputs/reviews/main-physics-20261115/vegstress-read-path-closure/origin-binding.json)证明pre-review `7cae3e02adad23674767a590fdf4d27f239ce065`与BASE完整owner均遗漏显式all999999分支；真实首次引入提交仍未知。

[主控反馈](../../outputs/reviews/main-physics-20261115/vegstress-read-path-closure/controller-review.json)指出该包仅完成局部owner／reader，实际论文入口仍raw、dispatcher默认cold helper且结果未展开。本次是**同一修复组的集成补齐**：[交付](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/README.md)把共享规则接入真实paper file-init与new-native年入口，默认dispatcher改完整owner并以as_payload展开科学state。既有raw codec／strict93／writer保持；continuous／同年内存继续／checkpoint解码不重放。新adapter、接线与as_payload是本次实现，不将其伪称成旧版已有代码；局部规则与入口集成合并登记为C49，不按调用处重复计数。

[本轮证据](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/verification-results.json)累计32不同检查有通过证据（29＋2＋1，总监督19.358秒），首轮两项新guard夹具的归约1 ULP失败及原输入源保留。只将该guard控制改为精确整数加零，原精确门及生产算法不变，生产源码各轮一致。四真实NC论文入口、默认owner选择／后续callback写回、l_first重复拒绝、新进程与连续相位／pickle及US_INIT前置域均有有界入口证据；旧21按原绑定保留复用。

主控已在声明的源码及materialized-us适配器域接受C49，代码／测试提交 `ea75bde31012cf519d4b0fa5c689d865c9d9c2b7`，见[验收记录](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/controller-acceptance.json)。正式为**49项／27生产修复提交**（48项有复查前证据、1项C42属新helper默认遗漏）。账本90＋1＋1／92、93／126／181及`numerical_acceptance=false`不变。us未完成整层US_INIT是Fortran合法但当前materialized-us子域未实现的上游边界，显式拒绝；源NONE、native数值／历史身份、DOC／thermal／maxmass／整模型门保留。没有新模型步／完整物理／Fortran／GPU／训练／调度／付费／环境重建，研究同步仍暂停；原失败和成功工件保留。提交及远端核对另记[发布回执](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/controller-publication.json)。

## C48（原 FIELD-CANDIDATE-01）生产接线已接受

2026-11-14 的[接线包](../../outputs/reviews/main-physics-20261114/calendar-driver-wiring/README.md)完成共享 calendar、真实 metadata／DIM2、实际 caller／compiled／年与 IO 接线。来源沿用复查前 `7cae3e02` 已存在的 driver owner 调用遗漏。本包保存 b5 `b5eff879e00154c936a1709d5ab91aae65f158d6` 中完整 `_runtime_next_state_from_components` 函数，在相同物理替身下调用其旧 signature，复现 Jan1／Jan2 丢失／stale 日历值，再执行当前完整 builder 验证双镜像修复；fixture、原函数 SHA 与跨度见[来源绑定](../../outputs/reviews/main-physics-20261114/calendar-driver-wiring/baseline-replay-identity.json)。没有执行旧模块／旧模型，也不追认未受 Git 跟踪 Fortran 的历史身份。

当前 `slowproc.f90:556–610` 的 pre-STOMATE 顺序、Jan1／Jan2／午夜、全部PC分支及严格边界为源依据，C45–C47 primitive/native证据复用。120个唯一检查最终通过：首次112通过／7夹具失败、修夹具后定向7通过，再补1个实际host factory；原回执保留、生产SHA未改。[C48主控接受](../../outputs/reviews/main-physics-20261114/calendar-driver-wiring/controller-acceptance.json)限于接线语义，不宣称标准案例数值回归或其排除，不恢复原001／旧archive构建／命名实际重启producer。正式累计48／26，研究同步暂停。下方旧包表格保留交付当时状态，当前C48状态以本节为准。

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

## C44及2026-11-13包交付时的clock候选来源

本节记录2026-11-13包交付时的状态：COLD-THERMAL-INPUT已接纳为C44，两个clock设计候选当时仍未修。原C1–C43的来源审计不重做。详细边界和哈希见交付时的[机器记录](../../outputs/reviews/main-physics-20261113/clock-authority-cold-inputs/cold-candidate-origins.json)及[说明](../../outputs/reviews/main-physics-20261113/clock-authority-cold-inputs/cold-candidate-origins.md)，候选ID保留以供追溯。

| 草案ID | 来源分类 | 有界结论 |
| --- | --- | --- |
| `COLD-THERMAL-INPUT` | 复查前已存在的cold source-default/static-input adapter遗漏 | 两个cold `use_refSOC`缺键`FALSE`分别用于静态读取门及cold initializer实参，均由`git blame`追到初始JAX基线`7333b46c`，且存在于`7cae3e02`。新restart helper的C42是另一个review-era问题，已在C41–C43接受提交中修复，不在本候选重复计数。当前Fortran source未跟踪；无标准case数值回归结论。 |
| `CLOCK-YEAR-LENGTH-DESIGN` | 复查前JAX表达式用day单位`one_year`除以second单位`dt_sechiba`，与恢复IOIPSL duration语义不同，活动影响未定 | JAX用`rint(one_year / dt_sechiba)`；恢复的IOIPSL `tlen2itau('1Y',dt,date0,...)`按实际日期转换。noleap、Jan1、dt=1800的源算术为17520步，JAX表达式为0；这是公式推导，不是运行观察。历史001的date0/calendar仍缺。 |
| `CLOCK-GREGORIAN-EPOCH-DESIGN` | 复查前JAX Gregorian helper与恢复IOIPSL日序epoch不同，活动影响未定 | JAX用astronomical Julian split；恢复的IOIPSL Gregorian日期路径使用Lilian日数及`2299160`偏置。历史001的date0及source-to-archive桥仍缺。 |

上述cold项现为C44，按一组cold控制／静态输入接口修复登记，相关satsoil传参和多caller不重复计数。按2026-11-13包交付时的边界，clock两项仍未修；其时没有生产日历修复或原输出失败结论，既有C1–C43接受范围保持不变。后续本地helper修复状态见下一节。

## C45–C47时钟helper来源（2026-11-14已接受）

`CLOCK-YEAR-LENGTH-DESIGN`与`CLOCK-GREGORIAN-EPOCH-DESIGN`仍分别保留其既有来源分类：二者的旧JAX表达式均在初始仓库提交`7333b46c`及复查前边界`7cae3e02`中，且不是本轮新引入的生产错误。新增识别的IOIPSL参考在41条clock owner/primitive记录和4个合成restart元数据case上与配对源码一致；这支持有界helper合同，不代表原始`001`、source-to-archive构建身份、生产driver日历接线或年度物理验收。完整证据、来源跨度与hash见[native helper来源记录](../../outputs/reviews/main-physics-20261114/native-clock-helper/helper-origins.md)和[机器记录](../../outputs/reviews/main-physics-20261114/native-clock-helper/helper-origins.json)。

C47 fractional-second修复有独立baseline证据，正式与C45年长、C46 Gregorian起点分开登记：`7333b46c`、`7cae3e02`及`a5748f6a`中的noleap callback/state将秒标为整数；唯一已确认运行时截断发生在`intersurf_time`构造返回状态时的`second=int(second)`。year-offset helper直接使用传入值，其`int`注解本身不执行截断。组合验证用native `nlfrac` owner值执行保存的旧源码，旧helper返回状态为900秒而IOIPSL owner返回900.2499952912331秒。其值来自请求900.25秒序列化后的`date0=715765.0104195601`，故也记录标量精度与owner量区分；不推出历史001曾使用fractional date0。候选审查中发现并修正的`floor(x+0.5)`近半值NINT边界属于新候选实现审查，不归因旧生产源码，也不另计正式修正。既有synthetic 2001 stub仍是独立旧fixture，native参考不验证其身份或历史值。组合local receipt为80项PASS、无error/skip，baseline控制3项。
