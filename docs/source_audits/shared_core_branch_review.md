# 共享物理核心与研究分支核查

## C104 暖档日湿量的日末复位

主线 `d97f618f15de419b73d058da80e9389902e229ce` 让普通和编译日累积入口在 `PEAT_OCCUR=FALSE` 时也保存 `fwet_daily`、`liqwt_daily`，使日末统一复位能取得两项当前值。日尺度 `f898282a` 和全球效率 `ffe77a00` 的只读快照都有原先漏键的入口。集中同步时核对各自的日内累积、日末复位和重启读写，连同 C79 的开关为真时的累积，以及本轮 C98 六叶活动层和 C103 日末字段的审计回归。两个非零暖启动标记可用于检查同步后的实际行为。[来源证据](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-wetness-carbon-lifecycle/validation/source-origins.txt) · [回归与接收](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-wetness-carbon-lifecycle/README.md)

## C103 禁火配置与日末凋落物状态

主线 `6c1dbe7806841fc4489663c82dbd20df88d3359f` 将火灾碳排放、火险指数和不可用凋落物接入日末更新、文件和暖启动读取。只读检查确认，日尺度 `f898282a`、全球效率 `ffe77a00` 的源码都还把这三项列为固定沿用旧值。集中同步时一起核对普通与编译日末、C95/C98/C101 的状态传递、PFT 草地管理参数，以及 LEAK 分解前和 LPJ 写回时使用的两份凋落物库。三个非零暖启动标记与五个继承对照可直接用于验证。[修正与依赖](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-stomate-restart-state/README.md) · [主控记录](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-stomate-restart-state-acceptance/controller-acceptance.json)

## C102 冷启动整日的读入顺序

主线 `b2fef009a2b80809ef41098bc9e166c3d31dfc05` 先验证源时钟是否从完整日的第一步开始，再用第一笔气象初始化 WATCHOUT 拖曳选择。日尺度 `f898282a` 和全球效率 `ffe77a00` 的已查快照均缺少引入这一顺序的 `aa14d042`，也缺少先前的源日检查 `782535b4`。集中同步时按各自的源时钟、冷启动入口和 WATCHOUT 所有权合并，复核完整日与中途起步、连续年与新原生年、C100 气象续读及 C101 编译日末。日历测试现覆盖上述入口的步进读取时机。[分支祖先](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-k144-compiled-calendar/calendar/ancestry.json) · [冷启动和跨年用例](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-k144-compiled-calendar/calendar/README.md)

## C101 完整源日的多日编译

生产提交 `d2ccda8e511b87b70e02db94c62e5264ac1ada36` 补齐 `782535b4` 接入源时钟后外层日扫描到 STOMATE 日累积器的调用。日尺度 `f898282a` 和全球效率 `ffe77a00` 的已查快照仍传静态局部步号。集中同步时，配套核对源时钟绑定、完整日行的步数和起止时刻、实际编译入口的午夜条件及普通逐步路径，带上日累积、输入行长度回归和发布测试。001 两个编译日的证据适用于此处的 CPU x64 路线。[来源与分支依赖](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-k95-k97-observation/candidates.md) · [主控验收](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-k95-k97-observation-acceptance/controller-acceptance.json)

## C100 文件续跑与 C98 编译调用

两研究分支后续集中同步以 `8adf774a55b9f5f4c413cb5e7ca28f97b0032f3e` 为依据。C100 需结合 `782535b4` 的源时钟和 C93 气象选轴，核对本分支的逐步读器、编译预读、同一气象文件暖启和显式局部步号起点。C98 需连同 `f44033d5` 的整个活动层修正，确保私有状态经过每个半小时并只将所需实参传入土碳过程。携带普通与编译气象读者、土碳签名和年界临时配置的回归。已查日尺度 `f898282a` 和全球效率 `ffe77a00` 都有相关共享入口。[同步依赖和来源](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-compiled-and-restart/candidates.md) · [主控验收](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-compiled-and-restart-acceptance/controller-acceptance.json)

## C96–C99 与 C95 可选配置的同步依赖

日尺度与全球效率分支后续集中同步时，以生产提交 `f44033d5d7e849dbc479770a306de8ac6de7030d` 为依据。此次核对的研究分支快照为日尺度 `f898282a`、全球效率 `ffe77a00`。按各分支保留的实际计算、编译和重启入口核对以下依赖。[主控验收](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-endpoint-state-acceptance/controller-acceptance.json) · [来源和分支范围](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-endpoint-state-followup/README.md)

- C96、C97 配合 C71 的日内更新次序、C88 的日平均量和 C90 的年界信号。核对冷启哨兵、纬度与季节状态的精度，以及最低辐照日期和积温的日末写回。
- C98 配合 C29 的暖启根系初始化。从土碳半小时步保留公共深度、私有层号和上一年量，核对普通与编译两条通路。新原生运行从文件的公共深度重建私有状态，连续运行携带上一时步状态。
- C99 保留布局中第一槽的裸地温度，核对冷暖文件、地表能量返回以及其余植被槽的更新。C95 的可选配置还涉及 STOMATE 直建冷启、文件读入和新年入口。对应的回归与发布测试随本提交同步。

## C93–C95 的共享依赖

集中同步时以 `ad3375ec29be675635dda2aa4b99d7df106776ac` 为依据。所查日尺度快照 `f898282a`、全球效率快照 `ffe77a00` 均早于主线时钟接入及 C8 的当步密度补齐。按各分支实际调用的入口移植、核对相关回归。[主控验收](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-two-day-local-acceptance/controller-acceptance.json)

- C93 核对 `driver/source_clock.py` 的气象选轴与原点，连同源时钟接入 `782535b4` 和原 CRUNCEP 并列坐标用例。重启轴沿用独立读器。
- C94 核对 C8 `27ff14b3`、普通和编译半小时循环的容重、日末写出及暖文件优先读取。在日尺度替代流程或全球批量流程中，追到实际使用该状态的下一处计算。
- C95 同步 STOMATE 直建冷启默认、模板读入与暖启生产载荷的取值。保留对缺变量、全哨兵和已存有限值的回归。[重启对照](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-two-day-local-closeout/restart/README.md)

## 1961 水文追踪读取工具

集中同步日尺度和全球效率分支时，按 `17db2adc5263a042163f0d7292ea9a0b93c42749` 更新共享追踪模式、重启年比较脚本、读取回归及其 CI 选择，并带上 `server_1961_trace_contract.md`。原 `post_tile` 第四个数据值由 `resolv(ji)` 写出。旧报告可按同一原文件和时间步、格点、土块重新解读。独立 v9 系数文件保留其 `branch_peat` 与 `peat_hydro` 定义。已在本地初始提交 `7333b46c` 和复查前快照 `7cae3e02` 核实旧列名。此项计入追踪工具维护，模型修正沿用 C1–C92。[主控验收](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-trace-reader-and-reference-preflight/review/controller-acceptance.json) · [两分支依赖与复查前记录](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-trace-reader-and-reference-preflight/trace-reader/comparison.json)

## C91、C92 的共享依赖

两项修正共用 `43fca734fbfcd15e500b55c1519daa5b820d7da1`。日尺度和全球效率分支待集中同步时，C91 核对 `sechiba/hydrol.py::hydrol_routing_soil_water_split` 的格点面积、正需求和作物土类条件，包括微小正面积土块。C92 核对同文件 `hydrol_soil_coef_peat_profile_from_tables` 无冻结分支里的除法和选档边界，配合已有 C78 冻结回归。随带 `test_hydrol_surface_water.py`、`test_hydrol_peat_coef.py` 及对应发布测试。两研究分支所核对的历史提交和文件散列见候选清单。[主控验收](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase16-soil-thermal-and-hydrology/controller-acceptance.json) · [同步依赖](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase16-soil-thermal-and-hydrology/candidates.json)

## C90 的共享依赖

日尺度和全球效率分支集中同步时，依照 `ad161b28c75f7f1e38d2bbf1260ec2f9eac5887b`，在 C88 的日平均量传递基础上，把主时钟的 `end_of_year` 传给 `stomate/main.py` 中的记忆和年统计两处。随带 `tests/unit/test_stomate_main.py` 的普通日／年末回归，按各分支实际使用的季节入口检查结转。测试启动及输入夹具另见 `768b311494d4c857df63f2ffb65922b055d93b86`。[主控验收](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase15-season-thermal-doc-and-ci/controller-acceptance.json)

## C87–C89 的共享依赖

三项共用 `c34b2bfb382adc432cdab96446f7817b970541c9`，承接 C86 的 LPJ 最终输出。日尺度和全球效率分支集中同步时，C87 核对 `stomate/integration.py` 的入口比例与 `driver/orchestration.py`、`sechiba/slowproc.py` 的写回。C88 核对 `stomate/main.py` 把当天累积量传给三个季节过程的调用。C89 核对 `coupled.py` 两个公开入口到 `stomate/integration.py` 的成对实参及唯一一次缩放。随带本提交的四份回归和 CI 入口，并按研究分支实际使用的流程验收。[主控验收](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase14-water-heat-carbon-receivers/controller-acceptance.json) · [逐项文件与依赖](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase14-water-heat-carbon-receivers/candidates.json)

## C86 的共享依赖

日尺度和全球效率分支集中同步时，核对 `0720d1bab0251cd460d9bcbf4bb3dcd1fa452701` 中 `stomate/integration.py` 的两个 LPJ 输出入口。独立输出按建植、树冠、覆盖与收获的实际结果组装字段，组合入口在覆盖后写出 GPP。同步时按各分支调用情况核实覆盖已执行与跳过的情况，随带 `test_phase13_lpj_output_receivers.py`、`test_stomate_integration.py` 的公开入口回归和 CI 配置。[主控验收](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase13-vegetation-carbon-receivers/controller-acceptance.json) · [来源登记](correction_origins.md)

## C83 扩展、C84 和 C85 的共享依赖

第十二包的共享提交为 `87698036a37ba4e87cf0d4af10dc9c4cc8aba18b`，承接 C83 的 `a77a1508ada77e2ee9a9093c77d11e839fd095a8`。日尺度和全球效率分支集中同步时，按实际调用核对 `jax_orchidee/sechiba/hydrol.py` 的两条根系路径、入渗的二维轴和静态 JIT 参数，以及 `jax_orchidee/sechiba/sechiba_step.py` 的分层诊断配置传递。逐点有效阈值和显式矿质背景可以分别指定。随带 `test_phase12_root_factor.py`、`test_hydrol_infilt.py`、`test_sechiba_step.py` 及对应 CI 入口。[主控验收](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase12-root-water-and-vegetation-carbon/controller-acceptance.json) · [来源登记](correction_origins.md)

## C82 和 C83 的共享依赖

2026-11-22，累计 83 项修正或支持补齐、43 个生产提交。日后集中同步时按 `a77a1508ada77e2ee9a9093c77d11e839fd095a8` 核对。C82 涉及 `sechiba/thermosoil.py` 及重启补算回归。C83 涉及 `sechiba/hydrol.py` 的土类表、逐点阈值及两种导水率接收形式，以及 `driver/orchestration.py`、`driver/restart_state.py` 的冷启动水分传递。随带已验证的水文、参数调用、热土壤、重启测试与 CI。两研究分支的实际接入继续留待集中办理。[来源登记](correction_origins.md)

## C81 叶层 CO₂ 重启写回

第十包接受 C81，累计 81 项修正或支持补齐、42 个生产提交。未来集中同步时以 `64d13e4d71bce938f4d0ebac80d83744eb143020` 核对 `sechiba/diffuco.py` 的叶层初始化和 `driver/orchestration.py` 的全字段写回，两处须一起检查。随带 Phase10 原参考及回归、重启测试输入调整和 CI。独立日尺度、全球效率工作树的实际调用待集中核对。[来源登记](correction_origins.md)

## C80 论文分解活动下限

2026-11-21，第九包接受 C80，累计 80 项修正或支持补齐、41 个生产提交。日后集中同步时，以 `889024a213e1e665d3141668a0be5caf174814ea` 为单位核对 `stomate/permafrost.py` 的精度选择、`stomate/main.py` 和 `driver/orchestration.py` 的论文入口、`coupled.py` 的传参，以及 Phase09 测试、原参考 NPZ 和 CI。日尺度与全球效率分支仍待集中核对。[来源登记](correction_origins.md)

## 第八包接受的 C78 和 C79

2026-11-21，主控接受冻结土壤水力系数分箱和 STOMATE 泥炭湿度分箱两处修复，提交 `9591652668df1afb4955d7282a6091b96d844e4b`。累计 79 项修正或支持补齐，40 个生产提交。未来日尺度和全球效率分支按各自实际调用核对两个共享模块、两份 Phase08 原参考数据、普通及批量回归与 CI 配置。研究分支同步仍按用户决定留待集中办理。适用精度、源规则和代表案例见[来源登记](correction_origins.md)。

## 第六包接受的 C73–C77

2026-11-20，第六包经主控接受，五项修补登记C73–C77，生产／测试／CI提交`f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b`。累计**77项修正／支持补齐、39个相关生产提交**；75项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏，不是77项物理公式错误。本轮五项均有复查前`7cae3e02adad23674767a590fdf4d27f239ce065`只读源码证据，首次引入日期未断言，未恢复运行历史代码。

未来集中同步须成组核对五个生产模块、旧Phase05显式参数helper、新参考／回归及CI。特别保留post-soil三处状态所有权，区分真实source默认值与显式实验配置；新的不可变部署应重建受影响输入／编译缓存。旧C65 checkify与C68静态控制合同继续适用；不得整文件覆盖研究树，本包未验证CUDA／AD／训练。

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-acceptance.json)核对46份产品、交付与独审封存身份，并复核配对原源、实际写回及历史源码。保存的隔离CI按精确节点合并为454通过、9项科学strict xfail、10项外部身份skip、0失败／错误；这不是一次不间断全通过运行。首次漏复制已跟踪restart schema的9个setup错误保留，修正复制后仅重测受影响的10节点，未重测此前通过项。配置内Ruff通过，生产lint 62前／62后无新增；hosted CI未运行。主控解析JUnit并复用owner／独审验证，未再积分；12份JSON仅Git换行规范化、语义相同。[发布记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-publication.json)保留最终版本。

保存的嵌套JIT整返回差异现有有界递推取整与表面通量抵消证据；13/14 soilflx_pft及其别名仍严格失败，不声明唯一HLO／机器指令原因。两个新增合成相等浓度DOC元组捕获到入口约1e-17取整变化，实测入口重放能复现严格相等分支变化及4／5个失败坐标；没有定位最早上游算术，也不外推旧年度轨迹。新8项JIT完整系数通量门／60坐标仍未归因。旧HYDROL／热／雪／低光／DOC／NPP／grain严格门各守原范围，不同配置DGVM诊断不计新修复。

52父过程的整体路线等级仍为source_mapped／open；4,059词法候选、565源审查组和187跨模块／外部调用组不等于数值覆盖或缺陷数。新有限证据包括14生长元组2611比较、扩散实际arm、非零冻融carry、冠层／capacity前驱、forcing／静态输入／modelout／routing和6份科学restart文件往返（1402检查，含147独立字段）。真实IOIPSL／native producer、maintained完整调度、自然历史、私有peat／PBL、原未定义合同与完整源分母仍有缺证。C64限npts1／nstm6；年度DOC真实边界／残差／异养呼吸均为null，原mass warning与负中间量保留。

下一包优先实质审阅[565源组和187调用组](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/denominator/remaining.json)：解析真实callee与目标域，复用准确旧证据，明确排除依据，对确有缺口补独立原参考和实际compiled／写回。并行归因新8项完整系数通量失败，不重开本包已闭合的嵌套与DOC有界机制。停止标准仍见[阶段计划](physics_branch_closeout_plan.md)；本地可做项完成后再汇总真正外部缺项，不以扩大扫描清单替代验证。整模型综合验证尚未启动，`numerical_acceptance=false`；参数／状态90＋1＋1／92及93／126／181口径保持，研究树同步暂停。

## 已接受快照：C67–C72与第五包范围

2026-11-20，主控接受第五包六项独立修正／支持补齐，登记C67–C72，生产／测试／CI提交`b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12`。累计**72项修正／支持补齐、38个相关生产提交**；70项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏，不是72项物理公式错误。本轮六项均经只读`git show`确认存在于复查前`7cae3e02adad23674767a590fdf4d27f239ce065`；首次引入日期未作断言，未恢复或运行旧模型。

未来同步须成组评审本轮五个生产模块、原参考fixture、六个回归文件和CI。季节日期需由共同前驱先生成并经packet保留；静态控制量需在研究自有外层JIT固定，重建受影响的不可变编译缓存。C65的Error／result合同仍需一起保留；不得整文件覆盖研究树，本次不认证CUDA／AD／训练。

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-acceptance.json)核对30份当前产品及原交付／独审封存身份，复用owner与独审证据。最终隔离六文件回归260通过、4个外部原源／历史资产身份门跳过；实际边界13通过、来源门2通过，相邻62通过分别保留，不相加为科学覆盖。新11份Python lint通过，生产旧诊断64前／64后未增加；hosted CI未执行。主控只解析回执、读配对源码及实际caller，没有重复积分或数值套件。5份新JSON的Git规范化仅为换行，解析值相同；发布状态见[记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-publication.json)。

本包接受有限组件、源条件与真实写回／下一实参；不宣布PB各组、完整源分母或物理阶段完成。保留原HYDROL／热／雪／低光／DOC严格失败及已接受浮点结论的原范围。嵌套JIT完整返回13/14项失败、最大绝对差5.34106093e-12仍未归因。fixedsnow原未定义值、私有peat／外部PBL、native partial、自然联合／雪能量前驱及DOC原始通量／floor观测等缺证继续具名保留；年度DOC真实残差与异养呼吸仍未知，不由端点推算。C64仍限npts1／nstm6。

下一包按[具名剩余动作](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/remaining.json)并行核对目标源条件分母、生长／扩散等剩余算术分支、嵌套JIT整返回差异和PB-05跨组数值接口；可在现有本地组件边界内补实际前驱，不能把合成元组称为自然轨迹。复用已接受参考，不重开已闭合浮点调查。整模型综合验证尚未启动，`numerical_acceptance=false`。参数／状态90＋1＋1／92及93／126／181口径保持；研究分支同步继续暂停。

## 已接受快照：C65／C66与第四包范围

2026-11-20，主控接受C65（编译热过程保留原冻结profile中止检查）和C66（初始化先判整组重算，再校验保留字段），共同生产／测试／CI提交`d2d48e6d5007079493cf91cca21c978eb79fa24a`。累计**66项修正／支持补齐、37个相关生产提交**，其中64项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏；不是66项物理公式错误。两项源行为均见于最早跟踪7333b46c及复查前7cae3e02，不能追称导入前的首次引入日期。见[来源登记](correction_origins.md)及[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)。

未来集中同步须成组评审`sechiba/thermosoil.py`、`driver/orchestration.py`、两份新增原参考fixture、source_conditions／NSLM／scan回归和CI。私有explicit JIT返回由结果改成Error／result；研究自有外层编译入口须接入相同错误合同，不能整文件覆盖。以新的不可变部署重建受影响编译缓存，单独验证CUDA／AD／训练；本包未认证这些模式，也未读取、修改或同步研究树。旧Teacher数据与运行中部署保持原身份。

[主控验收](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-acceptance.json)复用原包修前2失败／修后隔离13通过＋1原源身份门skip及本地源门，并核对[集成修订](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-repair/README.md)：NSLM原compiled回归的输入、断言和容差保持，新增真实内层module JIT／外层checkified窄owner的合法材料字段及非法拒绝，两环境各3通过、无skip。11份当前及冻结产品、原／新封存身份匹配；主控本次未重跑数值。原源码与科学封包保留，Git对两份新JSON仅规范换行，解析值相同。发布见[记录](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-publication.json)。

C65改变纯explicit热helper的编译合同：checkify后JIT，返回Error／result并在消费前check_error；module／scan／block与维护测试成组接入。C66仅接受有任一整字段缺失、因此九系数整组覆盖的公共初始化输入，native partial restart及完整driver可达尚未证明，保留组的混合marker仍拒绝。旧C64源等价仍限npts=1、nstm=6。原HYDROL100 eager＋93 JIT、固定热／雪21和不同输入handoff2严格次数不变。新增完整返回的嵌套JIT比较另有13/14项失败、最大绝对差5.34106093e-12，成因未定；窄接口通过不等于整状态等价。

下一包并行推进PB-01／PB-02剩余冠层和能量源条件，以及PB-04植被生长、分配、周转、凋落物与碳过程的当前源／同输入／真实消费者核对。按有限源条件补缺，复用已接受参考，不重开已闭合浮点调查。PB-05跨组收口和后续完整模型综合验证仍需完成。参数／状态90＋1＋1／92及93／126／181等口径保持，研究分支同步继续暂停。

## 已接受快照：C62／C63；物理分支检查

2026-11-19，PB-03潮汐producer／消费者及有效CHECK_CWRR修订通过主控有限验收，共同生产／测试／CI提交`7b35196462fbfd53f531fd1dc751699092f80104`。正式累计**63项修正／支持补齐、35个相关生产提交**，其中61项有复查前证据、C42／C57为复查期间遗漏。C62接通两份独立潮汐流及native-run记录原点；C63恢复false源零通量并拒绝尚未支持的完整true语义。仅改变土柱求和顺序的候选已撤回，不计新修复。依据、旧反例和年代见[来源登记](correction_origins.md)。

未来集中同步须以dfc67及此前已接受依赖为基线，成组核对`driver/tides.py`、`driver/orchestration.py`、`parameters/soil.py`和`sechiba/hydrol.py`，连同四个Oracle脚本、四份fixture、六份回归和CI。保留C43缓存所有权、C48时钟、C55参数、C60实际forcing／context合同，不用整文件覆盖研究分支。当前未操作或同步daily／global树。

[主控接受](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/controller-acceptance.json)复用隔离135通过／195精确strict xfail／4身份skip及55、87两批接口回归，另核对当前22份产品、冻结身份及求和原文。严格失败、true支持边界、私有peat／自然雪前驱及其他组缺项未升级为通过；`numerical_acceptance=false`。准确发布身份见[发布记录](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/controller-publication.json)。接下来按[物理阶段计划](physics_branch_closeout_plan.md)继续PB-03有界因果及源分支收口，未启动综合模型运行。下方日期段落保留各自历史口径。

## 已接受快照：C60／C61；参数状态接力结束

2026-11-18，接受基线`a3551dac4dbb2f53bee2ef47c80b4995eb57af82`上的WATCHOUT完整本地运输、初始化／连续历史与首读取整修正，共同代码／测试／CI提交`aa14d04210ab371aae6e959da193561f37f55d0e`。登记C60为旧支持缺口补齐，C61为旧NINT／首日准入错误；本包新接线的多处修正归C60，不机械增加旧bug数。正式61项修正／支持补齐、34个相关生产提交；[来源登记](correction_origins.md)保留精确年代、源规则、反例和新集成问题。

未来集中同步必须按**基线a355及已接受累计修复＋本提交**审查依赖：driver下`bundle.py`、`dim2.py`、`domain.py`、`forcing_interpolation.py`、`orchestration.py`、`restart_bundle.py`、`restart_file_io.py`、`sechiba_boundary.py`，另`sechiba/diffuco.py`和`stomate/restart_io.py`。同时带入两audit caller、八WATCHOUT回归文件、固定reference fixture、已有接口测试适配、CI和字段YAML。尤其保留C43/C48/C49/C52/C55–C59的缓存、时钟、初始化、单位与状态合同，不能仅搬某个helper或覆盖研究整文件。

支持固定annual file-start及有效state/context对的continuous-history；normalized单点14／2PFT、default93／canonical126、条件refSOC1及WATCHOUT driver9／默认13的各自分母不相加。真实外部producer身份、一般IOIPSL／多点scatter、fresh-midyear／TIME_SKIP与完整科学门仍限制外推。数值接受仍false。此次未读取、修改或同步daily／global研究树。

[主控验收](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-acceptance.json)复核实际代码和六条最终交接边，复用分版本99＋1及最终35／另7证据，无新模型运行；[发布回执](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-publication.json)记录准确远端SHA与工作区。本部分接力已停止，其他模块不自动启动。下方有日期段落保留历史接受范围，当前状态以本节和累计索引为准。

## 已接受快照：C55／C43扩展与C56–C59（当时WATCHOUT待补）

2026-11-18，本轮代码／测试／CI提交`5fe049ffa249f2c53daf669d576fe590ecbbb999`。深度参数归C55既有初始化到运行参数链扩展，scan context归C43已有缓存所有权扩展；条件refSOC、cold出口、block dispatch与temp_growth分别为C56–C59。正式59项／33个相关生产提交，57项有复查前证据、C42／C57为复查期间遗漏。C57准确引入为C48提交782535b4；本包stale refSOC集成遗漏归C56并单独保留年代。见[主控验收](../../outputs/reviews/main-physics-20261118/parameter-state-closeout/controller-acceptance.json)和[来源登记](correction_origins.md)。

未来集中同步须把七生产文件作为一个依赖组评审：`sechiba/hydrol.py`、`restart_io.py`、`restart_export.py`、`restart_lifecycle.py`及`driver/orchestration.py`、`restart_state.py`、`restart_bundle.py`；随同三个audit caller／validator、新四个接口测试、既有caller／cache测试、CI及字段YAML。保留C43 context ownership、C48日历相位、C49初始化、C55十向量和C51–C54边界。default93／canonical126不膨胀，refSOC条件1显式登记；flag、presence、双镜像与三文件需一起同步，不能只搬helper。

HydrolDepthParameters单列七活动标量与一inactive metadata，原HydrolSoilParameters十叶次序不动；旧runtime两／三参数构造和pickle有有限兼容证据，精确tuple解包需改字段访问。静态参数不写入科学carry／NetCDF。scan closure持有context，block静态dispatch同时入key／快照；两者不能用字段名相同推断不同参数可共享编译缓存。

本轮94不同节点最终通过是组合接口证据，完整cold／compiled物理及模型积分未执行。主控无数值重跑。下一本地包闭合WATCHOUT读取／插值／driver参数传递，真实外部producer身份独立未知。没有访问、写入或同步daily/global；全部共享修复待后续明确集中核对，不直接覆盖研究文件。`numerical_acceptance=false`和原科学门保持。

## 已接受基线：C53／C54水文写回与冠层编译入口，研究同步继续暂停

main基线 `a6a99f8c612813ad8ba3520d0ccb8f83dd059420` 的[本包](../../outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/README.md)于2026-11-17获主控有限接受，登记C53=`HD-FWET-01`、C54=`HD-JIT-01`，共同代码／测试／CI提交 `139e1ba7ba2db46d6f1a3b00f0592335bf40cafa`。生产只修改`driver/orchestration.py`两处fwet fallback和`sechiba/diffuco.py`无光照分层表时的源静态表接线；正式54项／31生产提交。两处均有复查前源码证据，不代表当前论文默认路径已观测到对应失败。见[主控验收](../../outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/controller-acceptance.json)及[发布](../../outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/controller-publication.json)。

未来集中同步须按依赖组核对这两个生产文件、新回归测试与两份小参考fixture、组件适配脚本和CI节点。保留已有C48日历、C49 us初始化、C51热系数及C52蒸发量／率边界；不可将七层组件us直接塞入十一层文件，也不可把独立TOPMODEL诊断链当作wrapper配置已准入。当前水文187／191保留四项自定义参数贯通缺证；冠层10例eager／真实JIT原源通过，交接19项通过，详细范围见[当前剩余工作](main_equivalence_remaining_work.md)。

本轮没有访问、写入或同步daily／global工作树，没有整模型运行或部署；已接受共享修复按以后明确授权集中评估同步。`numerical_acceptance=false`及原状态／分支分母保持。下一优先为自定义土壤参数共同schema／支持域核准，不能用当前有界数据交接结果代替研究线科学验证。

## 已接受基线：C51／C52热过程与蒸发单位共享依赖组，同步继续暂停

main `82d622722feea3cfe5034d8b15f98cda78e04872` 的[并行整包](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/README.md)于2026-11-17获主控接受，包含两个有复查前证据的修复：C51 THERMOSOIL两个final coef使用补足裸地权重；C52 ENERBIL实际执行dt随local payload传给共同writer，按源intersurf只在driver vevapp处除dt。共同代码／测试／CI提交 `9a856c6cf859437d2436f4f3748d4c9e770838a1`。完整thermal8例eager／实际JIT对照552字段过，根有限carry／writer／单点IO15项过。HYDROL和DIM2生产代码未改。

未来集中核对必须包含三个生产文件`sechiba/thermosoil.py`、`sechiba/enerbil.py`、`driver/orchestration.py`，两项新oracle脚本、A／B／C及联合测试和三份原源数值fixture、C50 CI测试及workflow。特别是dt元数据、共同writer、SECHIBA amount与driver rate的命名空间须作为同一依赖组评审，codec不再二次除dt。CI只纳入可移植节点；源identity节点仍需本地只读Fortran。

积雪17 helper／4完整组件的635／636字段通过，1项grain完整JIT严格失败和自然PFT14跨层`domain-unproven`保留；非零peq外部producer合同仍缺。本包只证明声明组件域和有限接口，不授权模型积分或部署。正式52项／30相关生产提交，51项复查前证据＋C42新增helper遗漏；90＋1＋1／92、93／126／181及`numerical_acceptance=false`保持。主控将雪测试的strict xfail收窄到已知case的有限grain差异，定向1项xfail及Ruff通过，不增加模型修正数。没有访问、写入或同步daily／global工作树；此前C50审查误读研究树的范围纠正仍保留在原包。[主控验收](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/controller-acceptance.json)和[发布记录](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/controller-publication.json)补充原交付pending回执。

## 已接受基线：C50 thermal energy权重修复，同步继续暂停

main `551cc8cdbc4f16fa37ed2c89f260c36b57c596a8` 上的[本地独立thermal包](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/README.md)已获主控接受，登记C50，代码／Oracle／fixture提交 `6a5d4a7f2092f32ef9de064e61ee3224c5b48b01`。原源`thermosoil_main:913`传`veget_max_bg`，本轮将`sechiba/thermosoil.py`中explicit及no-explicit两个energy调用实参补足裸地，其余生产执行AST不变。稀疏植被原Fortran反例证实旧两路均可把surfheat错分到coldcont；17标准案例通过后，修复后18保存案例两caller eager／窄JIT的288字段通过，真实隔离explicit owner／carry单次通过，pytest95通过／2旧控制skip。全no-explicit分支及完整compiled热路径未认证，详见[主控接受](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/controller-acceptance.json)。

未来集中核对的新增依赖组为`sechiba/thermosoil.py`、`scripts/dev/oracle_thermosoil_initialization.py`、`tests/unit/test_thermal_first_consumer.py`及其原Fortran数值fixture，加本包相位／来源／验证合同及字段YAML注释。已有C34 `driver/restart_state.py`、cold owner及`driver/orchestration.py`输入／写回在本轮被核验但未改；原Fortran、公共比较器与科学容差保持。本包不能作为只移动两行而遗漏fixture来源／支持域的研究部署回执，也不授权现在同步。

研究树未写入或同步；审查者早期误读daily同名源后，已纠正并在main重核，不能把本轮描述成全程未读取研究树。正式为50项／29个相关生产代码提交，49项有复查前证据、C42为新增helper遗漏；90＋1＋1／92及93／126／181、`numerical_acceptance=false`保持。独立初始化／首消费者已有本地有限证据，其他thermal分支、历史身份、DOC、maxmass、peqAcoef与整模型门继续保留。主控复用数值验证，另补Oracle的已有目录保护并完成2项CLI拒绝检查；不涉及生产物理。发布和远端核对见[记录](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/controller-publication.json)。

## 已接受基线：C49上游US_INIT扩展，同步暂停

[当前包](../../outputs/reviews/main-physics-20261115/us-init-production-closure/README.md)以main `512d2a1134b96586d41528aea0376e81fe42e72e` 为基线，补齐C49已声明缺失的逐层US_INIT物化，并传递真实run.def参数、保留initialized us及年度consumer镜像。主控于2026-11-16接受，代码／测试提交 `d6bad452932d1a7179ff5ab5fd80c771da89e336`；见[验收记录](../../outputs/reviews/main-physics-20261115/us-init-production-closure/controller-acceptance.json)和[发布回执](../../outputs/reviews/main-physics-20261115/us-init-production-closure/controller-publication.json)。源条件、配置有限语法和相位见[合同](../../outputs/reviews/main-physics-20261115/us-init-production-closure/phase-contract.md)。复用唯一组合84／84通过及独立终审，无重跑；大型scaffold未执行，其实参只有源码证据。旧失败／成功回执保持原身份。

未来集中核对需作为一个依赖组处理六生产文件：`sechiba/hydrol.py`、`sechiba/io_contracts.py`、`sechiba/restart_io.py`、`sechiba/initialize.py`、`driver/restart_state.py`、`driver/orchestration.py`；配套`test_us_init_production.py`和`test_vegstress_read_path.py`、字段YAML的us/vegstress合同及本包phase/config/state-flow说明。复用未修改的`driver/init.py` raw_run_def、strict93／writer和multiyear pickle合同；不能只搬shared helper而漏掉参数、科学snapshot、实际caller或年度两镜像。

未检查或改动daily/global工作树；同步仍暂停，不生成交接执行或新部署。正式49项／28个相关生产代码提交，本次作为C49初始化链支持域扩展登记，不增加C50。48项复查前证据／C42新增helper遗漏、90＋1＋1／92、93／126／181和numerical_acceptance=false保持。getincom本体／完整语法、源NONE、历史producer／构建、DOC严格state／真实碳通量、thermal、maxmass、peqAcoef单位及整模型限制保留。另行用户hostname连接检查不构成科学运行授权。

以下保留C49接受时快照；上游US_INIT边界的当前实现以新包为准。

## 已接受快照：C49 vegstress共享规则与实际初始化入口

2026-11-15，本包基线main `194d91350bf375f2c9e49360401b807e07d8a02e`；[本次接线包](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/README.md)沿用[上一包](../../outputs/reviews/main-physics-20261115/vegstress-read-path-closure/README.md)的源合同与21项owner检查，并完成[主控指出的入口／相位缺项](../../outputs/reviews/main-physics-20261115/vegstress-read-path-closure/controller-review.json)。合法已有文件缺vegstress或whole-local ALL999999按源在原值上加所选S，ordinary／partial保留；raw complete93 codec的函数体不变。

论文聚合入口自动选file-init reader，默认SECHIBA dispatcher改选完整HYDROL owner并实际展开科学state；年度`new_native_run`应用同一规则一次，`continuous`／同年内存继续／完整checkpoint解码保留状态。us的US_INIT上游层须先物化；未物化的整层marker在file/fullowner/new-native路径显式拒绝，partial保持。源NONE未定义边界与局部cold API约定分开。相位证据、实际caller及子域限制见[合同](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/phase-contract.md)。

[本地验证](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/verification-results.json)累计32不同检查有通过证据：首轮29通过／2个guard fixture精确比较失败，保留原回执与夹具；用精确整数加零隔离guard后仅2项定向通过，再加实际内存入口1项，总监督19.358秒。生产源码各轮未变，原数值门保留。四真实NC论文入口直接经strict93／所有finalizer／writer；默认owner、callback顶层写回、l_first、年度相位、checkpoint及上游拒绝均有实际入口检查。四物理委托和后续runtime被替换，没有物理模型执行；原21和4／19／120／80／41继续按各自绑定复用。

主控已将局部源规则与实际接线作为同一组修复接受为C49，代码／测试提交 `ea75bde31012cf519d4b0fa5c689d865c9d9c2b7`，见[验收记录](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/controller-acceptance.json)。未声称NPP／DOC或标准算例改善；[来源](../../outputs/reviews/main-physics-20261115/vegstress-read-path-closure/origin-binding.json)只证明复查前7cae3e02已有owner遗漏，精确引入未知。正式**C1–C49／49项／27生产修复提交**。账本90＋1＋1／92、93／126／181、`numerical_acceptance=false`与DOC／thermal／maxmass／整模型／历史身份门不变；研究树未同步，下一阶段未派发，优先补齐US_INIT上游物化。旧失败和成功工件逐字保留；本轮另按用户询问只读复核曙光连接成功，未启动远端科学运行／作业。提交与远端核对见[发布回执](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/controller-publication.json)。

## 此前 C45–C47 原生 helper 接受快照

2026-11-14，主控已接受[原生clock／helper修复](../../outputs/reviews/main-physics-20261114/native-clock-helper/README.md)：正式编号C45–C47，代码／测试／fixture提交3f7b482765b7b41339ee56a818b1f164490a5d1e；累计**47项／25个代码提交**。研究分支同步继续暂停。共同生产改动仅`sechiba/lifecycle_completion.py`和新纯`sechiba/ioipsl_calendar.py`，纠正首次date-aware年长、Gregorian Lilian日数及非Gregorian小数秒；同一run跨年保留首次常量。显式日期约定、独立startdate锁及previous binding限定适配器支持域，旧seconds／astronomical／无绑定payload与未测calendar提前拒绝。两个直接unit测试迁移，旧2001合成oracle保留原TEMPLATE／CASES，仅当前比较／coverage入口在编译前拒绝；相关接口和fixture须随本次准确提交交接；[主控接受记录](../../outputs/reviews/main-physics-20261114/native-clock-helper/controller-acceptance.json)复用原80项通过及独立终审，未重跑。

新标量参考job `127957445`使用既有IOIPSL archive，1 CPU／1 GiB／300秒硬限，在compute node实际14秒完成11个进程、41条clock记录（metadata另含1个预期exit1）；源预期全部一致，原`1e-10/rel0`门不变。实际compiler／四module／动态库／两新executable与各自link map、前后SHA及原始退出均保存；首传输超时、明确absence恢复和中文locale解析误报完整保留，实际只有1个job，无原生重跑。唯一本地相关组合80／80通过，含41native对照、6项纯DIM2 reset映射、3个旧基线反例；本地无模型、物理、JIT、Fortran或remote调用，服务器标量Fortran的非零计账单列。旧50＋11、C44专项50与gas／DOC／92字段证据复用，未重复执行。

没有接入生产SLOWPROC日历、compiled carry／fast schema或year-restart IO，不修改物理kernel、公式／容差、主控计划／合同或daily/global工作树。历史001 producer、旧source/object-to-archive manifest及真实命名年重启tuple仍缺；新参考不是历史producer的替代身份。DOC严格状态／真实守恒残差、独立热真值、maxmass条件域与87归一＋5缺证／92及93字段分母保持，仍不满足整模型等价合同。详见[当前剩余工作](main_equivalence_remaining_work.md)、[helper合同](../../outputs/reviews/main-physics-20261114/native-clock-helper/helper-repairs.json)和[有限来源](../../outputs/reviews/main-physics-20261114/native-clock-helper/helper-origins.json)。

2026-11-14主控已接受C44冷启动热参数与静态碳输入绑定，代码／测试提交`9c15c042a5ade9c91d91ac91a8c0ea39ed87c8f2`，累计**44项修正／24个代码提交**。源默认、控制门、静态读取及cold／首步参数统一，尚未绑定变体提前返回明确缺项；C42及物理kernel不改。50项新纯适配器检查一次通过，主控核对实际文件、源条件和回执并复用独立终审。IOIPSL源／库／link及独立current-reference cold输入取证获有界接受，原001／原生clock等缺项保持；两clock候选及生产日历未集成。详见[主控记录](../../outputs/reviews/main-physics-20261113/clock-authority-cold-inputs/controller-acceptance.json)及下方C44条目。本包无模型／物理／Fortran或服务器计算；daily/global工作树未动且同步继续暂停，DOC／热真值／maxmass／历史身份限制保留。

2026-11-13 主控已接受C41有效OK_PC、C42 restart use_refSOC缺省、C43 compiled context缓存隔离／强引用；代码及测试提交`8a772de6f4977018a875093d535b6088317716c8`，累计**43项修正／23个代码提交**。C43同时更新开发快照读者及三项旧cache测试；等价独立context可能额外编译。同实例复用和8项FIFO保持。50项相关pytest与11项纯日历检查一次组合通过；主控核对实际文件和回执，未重复执行。接受仅及源参数／适配器／缓存语义，未运行物理kernel、真实JIT、模型或Fortran。详见[主控接受记录](../../outputs/reviews/main-physics-20261113/parameter-repairs-calendar-design/controller-acceptance.json)及下方累计索引；研究分支同步继续暂停。

完整日历设计已覆盖source clock、Jan1／Jan2分支、pre-STOMATE顺序、once-only、双镜像及compiled／IO接线需求；本地缺IOIPSL原noleap起点与重启约定的可验证桥，故只接受设计和精确缺证，未改生产日历或应用旧日末部分补丁，不分配C44。[来源登记](correction_origins.md)区分C41／C43的复查前遗漏与C42由`fd0fe514`新增的helper默认边界遗漏，不把它们全称旧翻译问题或标准算例回归。独立cold refSOC gate／IO、热真值、DOC严格状态及守恒缺证均保留；旧producer接受不自动认证新生产路径。下一包核实原时钟来源及cold初始化合同，不启动未审完的模型运行方案。

2026-11-13 主控已验收DOC／碳核算离线分析：**不能把累计诊断差0.029237968847称作碳损失。** 旧producer `e2a9d9f2d5aabca2e2d391400f14559f0dbf8c09`的319/1962 memory−files审计独立库终态差为**0.00286422847186 gC/m²陆地**，其中DOC为0.00286422845561，其余约1.6251e-11；共同年初/A年末/B年末总量为527.245664208/529.312346246/529.309482018。详细[库存与分项](../../outputs/reviews/main-physics-20261113/doc-carbon-accounting/accounting-results.md)、[源单位/相位合同](../../outputs/reviews/main-physics-20261113/doc-carbon-accounting/source-accounting.md)及[独立审核](../../outputs/reviews/main-physics-20261113/doc-carbon-accounting/independent-review.md)属于本次离线包，不是新模型运行。

DOC按点/PFT/32层/两相/7来源池/碳元素统计，保存值已是逐层面积库存，乘实际`veget_max`一次，不再乘层厚或水文tile面积。冠层`interception_storage`是独立DOC碳库而非水，producer已含面积权重；deepC_peat/soilc_total及替代分支量不重复计入。实际PFT14=1、产品全年零均有caller/carry/更新源路径证据，不只依靠端点推断。原DOC78坐标/maxabs2.234685861e-4及累计诊断原限0.0003646537522的严格失败保持。

源`carb_mass_total`在旧SAVE/restart值上跨日累加`11*活体+前11层litter/POC/DOC深度比项+154*产品`，无dt；其variation也非真实库存变化。相同终态公式的差0.000110307646694275不是已观测末日增量，末日D_old未保存。全年可重建活体诊断贡献差−1.39637836891e-7，产品贡献零；剩余0.0292381084849925由未保存的逐日litter/POC/DOC及运算舍入构成，不能唯一归因DOC，不是质量残差。

对齐365日及面积后，GPP/NPP/maintenance/growth的A总量为86.4620357003/−18.2809084295/94.5023231438/10.2406209860 gC/m²，对应A−B约−1.03935e-10/−6.05936e-11/−1.56513e-11/−2.76887e-11。DOC差/GPP为3.3127007e-5，仅供量级比较；缺真实边界DOC、大气/DIC呼吸分区和floor/约束碳增减，年度异养呼吸与真正质量残差均null。主输出通过、库存差及守恒缺证必须分开，不排除真实通量或长期影响。

本包未确证新的源语义遗漏或生产代码缺陷；既有近零水/排水分支的局部浮点敏感性证据保留，尚未解释的全年余项仍列未知。当前累计仍**40项已接受修正/22个代码提交**；无新编号、生产/参数/容差变更、commit/push或daily/global同步。新增模型积分/模型日、远程、Fortran执行、GPU、训练及付费均0。

三文件核心数组、独立标量/轴/掩码/非单位权重及非均匀冠层控制已完成；独立审核没有重复重算全部非DOC库。必要的[捕获准备](../../outputs/reviews/main-physics-20261113/doc-carbon-accounting/capture-plan.md)仅离线检验：拟普通CPU首日A/B各1日、须过原第1日4+26字段严格准入，与历史compiled scan不同；缺晚1962完整起点，不默认全年回放。硬杀只保留最后完成的host checkpoint，不含进行中的调用；本包不执行该方案。最终交付见[离线包入口](../../outputs/reviews/main-physics-20261113/doc-carbon-accounting/README.md)。

主控接纳范围与保留限制见[验收记录](../../outputs/reviews/main-physics-20261113/doc-carbon-accounting/controller-acceptance.json)。只发布上述两份审查报告，不新增修复；下一包转入有限源码／状态覆盖对账，准备中的DOC捕获不执行。

## 此前已接受的gas运行与来源审计

上一包（2026-11-13）主控已验收两条8日运行和C1–C40来源审计，**最终gas运行缺项已补齐；40项所修问题均有复查前基线证据**。该包无生产源码或测试修改，累计仍为**40项已接受修正/22个代码提交**，daily/global同步继续暂停。
验收范围与保留限制见[主控接纳记录](../../outputs/reviews/main-physics-20261113/final-gas-runtime-and-fix-origins/controller-acceptance.json)；复用执行方验证与独立终审，主控未重复模型、测试或全套来源调查。

本轮另以明确复查前边界`7cae3e02adad23674767a590fdf4d27f239ce065`逐项核对[C1–C40来源](correction_origins.md)：40项所修的错误表达式或源合同遗漏均已存在于该基线；未在这40项中证实复查期间引入的新回归，最初引入提交仍未知。性质分别为13项源过程计算/常数/单位/顺序、20项默认/参数/状态/restart接口、4项显式诊断API和3项执行缓存/AD稳定性；不将40项等同40条独立且活动的物理公式bug。C40初版未闭合既有首日carry遗漏属于不完整修复，不新增编号或倒推为新回归。

两项共同代码提交为 `76e406e0e0e75402d221bb281908fa4260bc5d66`；此前验收复用16项定向回归、真实三文件写读及独立终审，主控当时未重复模型运行。
此前gas修复包基线为main `488424df2618f209bb4d0d1bf3d8d9cba23602d1`；最终源码overlay以[修复版冻结](../../outputs/reviews/main-physics-20261113/gas-carry-and-initialization/gas-freeze-v2.json)登记。先前发布的319年度结果保持原`e2a9d9f2…` producer。

本轮直接使用已发布main `765c75fc51af981b1f687f8e36b05f8dd382dc77`，完成**319-cold8和001-restart-nondefault8**，无生产源码或测试改动。每路与已保存前8日日输出比较均为289行/264数值叶、0失败、最大绝对差0；原严格门`1e-8+1e-10*abs(reference)`及shape/dtype/mask/discrete/clock精确合同保持。实际host见证均记录首日tstep47、`jax._src.stages.Compiled`完整days2–8调用/返回、末tstep383与四gas逐坐标精确保持；实际日末packet经当前三文件writer→reader通过，独立base的gas故意置错仍由packet正确覆盖。

001仅复制原三份NC，且只改四gas为事先固定的有限、正、非默认、坐标可区分数组；其余schema/属性/值/mask及原件身份均核验保持。实际首STOMATE reader读取新复制目录已见证。这是2010归档边界在1961forcing下的人工接口夹具，不认证1961历史初始化或年度科学真值。新包共**2启动/16模型日/393.869341监督秒**，各900秒硬限、串行CPU/x64、备用启动未用；任务TEMP清理，原16项回归与001cold365证据复用。详见[本轮已审交付](../../outputs/reviews/main-physics-20261113/final-gas-runtime-and-fix-origins/README.md)及[实际运行证据](../../outputs/reviews/main-physics-20261113/final-gas-runtime-and-fix-origins/runtime-results.json)。

初版真实319cold2和001paper-start2均完成且各66个daily leaf通过原严格门，最大日差分别7.66053886991358e-15和8.881784197001252e-16，clock/domain/352既有必需路径通过；**两路均缺四gas，生命周期失败**。
真实首日`driver_previous_step_state_from_next_step_hydrol_scaffold`有独立carry清单，遗漏四字段；仅在compact runtime检查carry不足以证明该入口。现只补`STOMATE_GAS_FIELDS`，不扩展其它carry或动力学。
两份原run/receipt/freeze与可重建源码patch完整保留，不把初版数值通过称为最终版本回归。直接非默认cold/restart反例在旧源码均因gas缺项失败，修复后16项定向测试通过，覆盖该真实入口、普通/FastState接口、日末及底值故意置错的真实三文件写读。

Fortran `constantes_var.f90:1467–1468`及`stomate_io::readstart:1177–1199`规定O2=298.3、CH4=0.001267的全数组sentinel回退；已有有效或部分sentinel数组优先，显式API参数只覆盖缺字段/全sentinel回退。
四字段`O2_soil/CH4_soil/O2_snow/CH4_snow`现从cold/restart初始化、首提取、首日专用跨步、普通/compiled布局保持到日末与写盘接通；旧缺项pickle/NC未补填。OK_PC/CH4_CALCUL活动gas动力学仍关闭，不将gas修复归因DOC或001主指标差异。

原gas包额度内剩余唯一001cold1961/365日使用修复版独立冻结身份，当时未重试两个短窗口。完整年度对象SHA `77198fddc77be777240dd30a4337dbc0c343ddfe3d27f77060be93b4aa25e361`；最终gas门**通过**，原整体science/normalized/diagnostic为**通过/通过/通过**。完整daily及末态保留在原gas包`001-cold365/run.pkl`；四主指标如下：

| 指标 | 新001cold年度 | 原年度参考 | 原annual门 |
| --- | --- | --- | --- |
| AGB | 4.656111406026477 | 4.656111240386963 | 通过 |
| BGB | 4.131135638281122 | 4.1311354637146 | 通过 |
| GPP | 1.5009596523801692 | 1.5009596347808838 | 通过 |
| NPP | 0.5990213772418084 | 0.5990213751792908 | 通过 |

此前gas包实际3启动/369模型日，监督570.064392秒，各900秒硬限、串行CPU/x64、0技术重试；其记录冻结，本轮另按上述2启动/16日独立计账。两包远端/新Fortran/GPU/训练/付费/环境重建均0，任务system TEMP已清理。每路源码/运行器/input身份独立且不混称同一producer。
原annual `1e-5+1e-3*abs(F)`、normalized `1e-3`、diagnostic `2e-6+1e-7*abs(F)`及严格比较`1e-8+1e-10*abs(A)`均未改变；shape/dtype/mask/discrete/clock仍精确。

001通用及点专用Job0脚本在i=0/1961明确三输入NONE，i=1才切换刚产生的start文件；本地归档start时钟858480/date17885属于2010边界，不是脚本规定的1961输入。新cold协议在年度前已按源固定。
实际`slowproc(ok_stomate)→stomate_initialize→readstart`链及biomass/deepC/gas条件回退已取证；外部IOIPSL restini/restget实现不在本地，不能据caller推证NONE时的实际默认数值。历史001/1961 used_run/实际输入producer仍缺，年度数值比较不补称历史身份真值；旧paper-start四指标失败完整保留。

C34有界相位审查未发现已准入USDA/default-knob、显雪/refSOC、完整系数路径的新接线遗漏。三字段在首次main ENERBIL前初始化，但ENERBIL直接消费soilcap/soilflx；stempdiag随后供HYDROL，pcapa_en/temp_sol_beg供THERMOSOIL能量检查。
完整系数重启最终var_init的薄雪状态与cold/缺系数随后coef覆盖相位分开。既有319A/B数组相同仅为JAX producer证据，独立Fortran初始化/首次真实消费者数值真值仍缺；本包未改热物理或新增trace。

剩余优先边界为DOC/carb_mass_total严格状态及DOC通量、上述独立热真值、显雪maxmass条件域、001历史实际输入与IOIPSL外部语义。旧DOC78坐标maxabs2.234685861e-4及carb_mass_total差0.02923796885超原限0.0003646537522继续保留，不接受完整重启连续性，不把原始库存坐标和充作质量/通量。
原gas包未补最终版319及非默认restart的实际编译运行；本轮两条8日路线已补该缺项和直接compiled调用/返回见证。原FastState单元段仍仅为Python接口证据，旧年度只证明其001cold路径。旧保存物没有第8日完整packet参考，因此本轮不宣称该完整状态逐值相等或关闭DOC严格失败；两点及选定窗口不外推669点、全分支、AD或长期接受。

此前gas修复详见[冻结交付](../../outputs/reviews/main-physics-20261113/gas-carry-and-initialization/README.md)、[机器摘要](../../outputs/reviews/main-physics-20261113/gas-carry-and-initialization/summary.json)、[001协议](../../outputs/reviews/main-physics-20261113/gas-carry-and-initialization/initial-protocol.md)及[热相位审查](../../outputs/reviews/main-physics-20261113/gas-carry-and-initialization/thermal-phase-review.md)。此前四年度包及C1–C38登记保持原身份；冻结交付仍记录接纳前的38项/21提交，C39/C40正式编号、范围与限制见[主控验收](../../outputs/reviews/main-physics-20261113/gas-carry-and-initialization/controller-acceptance.json)，准确代码提交见[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)。本轮仅发布三份审查/来源报告，无新编号、生产代码提交或研究分支同步；冻结owner交付仍保留其主控接纳前身份。

同步安排（2026-10-27用户决定）：**暂缓与两个研究分支对齐，优先稳固main**。
本阶段不安排分支同步或交接包工作；当前累计40项已接受修正/22个代码提交，登记与适用范围持续维护，
待基础分支本轮加固后集中评估交接。C4–C40仍待维护方集中核对，
并不表示本次核查过研究分支。**C31已由主控验收；319点1961年度四指标通过原限，原NPP超限在该算例闭合**。
Fortran job126077710正常完成，620,494条native记录、实际相位和年度/共同边界保持检查通过；
六项预设shape笔误与原失败独立保留。本地原cold和唯一追加cold分别因序列化核对、首七日块
观察器时钟失败，均已计账并保留原失败。用户再明确授权的一次900秒attempt02以355.197170秒
到end204，真实end197 packet/两套spec/九家族与Day199全48步已捕获，实际callback和原限
同轨迹准入通过。限定C29/C30生产集成已完成，最终代码4项定向测试/6次组件调用及静态接口检查
通过；C29/C30已由主控接受，准确代码提交为`ea86d3fbe97b3a1ecf5c5d8dda931aea5ea2ae69`。
原42行/28ID登记保持，上一批增C29/C30两行后为30项/17代码提交；本轮C31另增一行。
见[上一批交付](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/README.md)。
[完整surface/split/蒸发producer调查](../../outputs/reviews/main-physics-20261029/npp-surface-flux-origin/README.md)
已完成原生捕获、七日准入和8个完整同输入对照；累计24组件调用额度已用完。
该取证批次新增冻结轨迹显雪grid温度clamp遗漏的源证据、实际反例和隔离候选补丁，当时尚未改生产或接受新修复。
当时局部蒸发至top-RHS传播有界闭合，clamp至drag及年度NPP改善未验证。
此前C29/C30接受时为30项/17代码提交，既有证据与分支同步暂停安排保持。
2026-11-04主控已接受上述有界取证，直接源码核对确认当前main同样缺温度clamp调用；
本轮[最小修复与验证](../../outputs/reviews/main-physics-20261104/npp-snow-temperature-clamp/README.md)
已在共享THERMOSOIL返回产生一次grid温度clamp，四个调用端补齐effective显雪开关、当前post-HYDROL雪状态，
各writer/finalize消费该闭合值，PFT与较早ENERBIL返回不改。31项最终定向通过；旧driver五个packet路径及两个调用端控制均预期失败。
冻结dh-only＋仅此修复的唯一七日回放完成，实际9529→9530温度写回逐位正确，
目标蒸发/flux_top差降至约1.17e-13，9529–9531所测四核完整tuple通过原限。
本轮唯一当前C29/C30＋修复cold365已完成（418.937秒worker/434.560秒监督），AGB/BGB/GPP/NPP原science、normalized及诊断均通过。
NPP误差由8.18156e-4降至1.81954e-9，原限3.9518497e-4不变；原七个PFT14分支不一致日现为空。
当前年度288个实际Day199求解tuple确认9530 top-RHS残差8.8996e-12和求解后MC差7.1782e-11通过；
深层两坐标RHS仍超严格限，必需末态数值schema通过不等于92个额外字段或所有物理域准入，旧carry失败保留。
主控已[正式接受C31](../../outputs/reviews/main-physics-20261104/npp-snow-temperature-clamp/controller-acceptance.json)，代码提交
`cb4f76817acd852cde8cd8d4195e1e52bfd8c770`；累计31项/18代码提交。原44行保持，新增后为45行/31ID。
研究分支同步继续暂停；下一优先项是真实NetCDF重启与跨年连续性的有界验证。

| 编号 | 源合同与改动 | 验证与适用范围 | 接受及同步状态 |
| --- | --- | --- | --- |
| C31 | `thermosoil_main:1041–1050`；共享THERMOSOIL结束时计算显雪grid温度clamp，四个caller传有效gate/当前HYDROL雪，writer/finalize消费闭合值 | 31项定向回归、隔离七日传播及319/1961原cold365通过；PFT/早期ENERBIL相位保留；不认证完整bucket/restart生命周期或全点/长期，深层RHS两严格残差保留 | 已接受；`cb4f76817acd852cde8cd8d4195e1e52bfd8c770`；生产文件为`driver/orchestration.py`和`sechiba/thermosoil.py`，配套`tests/unit/test_driver_snow_temperature_clamp.py`；两研究分支待集中核对 |

以下首次分歧节保留前批已接受的有界证据。
实际ALT0初根归一化条件及preprofile层厚来源差异已确认；单独纠正层厚使Day130 MC/MCL回到原限，
年度原生NPP差却只减少约5.294e-8，原七个分支不一致日及年度科学失败保留。
纠正轨迹下一未闭合边界为Day199 step25/kjit9530实际土壤更新内部，入口水差7.16e-11、出口3.751e-4；
前批缺少的实际纠正冻结/系数/求解tuple已在本批补齐，RHS完整同输入核对通过，起因继续向surface/split上游定位。见
[本批交付](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/README.md)及下文当前节。
本批运行次数已耗尽；主控已[接受有界源差异及干预证据](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/controller-acceptance.json)，
并[安排单独计账续作](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/controller-followup-decision.json)：
补齐Day199内部链、落实两处修正与回归；当前捕获/准入门已通过，唯一年度检查在49.085176秒
因观察器dtype读取的tracer→NumPy错误失败，1/1已计账、无retry；只存首core/stress七条，
无年度对象/完整末态。隔离恢复v3已冻结并通过最终零模型预检及独立审核，
见[方案](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/jax/annual-recovery-proposal01/recovery-proposal-v3.json)
与[审核](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/annual-recovery-independent-review.json)。
两份静态拒绝草案保留；用户随后明确授权一次独立计账的本地CPU/x64年度验证，硬限900秒，
已按[新授权记录](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/jax/annual-recovery-proposal01/recovery-execution-release.json)完成。
监督504.738770秒/exit0、TEMP清理，365日/end17519、52七日块及预期native计数完整。
AGB/BGB/GPP通过原科学限，NPP新值`0.38600312976573087`、误差`0.0008181562874136072`
仍大于原限`0.0003951849734783173`，综合科学接受及严格诊断仍false。
见[四指标判定](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/jax/annual-recovery-proposal01/run/annual-modelout-policy.json)。
原年度1/0历史计账保持；新增代码修复的接受不改变NPP科学失败。
全年五类NPP mask对旧original/dh-only的14PFT不变；current F详细NPP仅Day199，presence14、
negative14×12及13个有效class精确，PFT1 class缺失。历史365日F参考仅作已标来源的描述比较，
Day199四项原生通量各有一个超严格限的PFT14差异；原七个不一致日337/338/339/341/342/359/364保留，
不替代current全年详细tuple。
上一批Day338局部跨阈值机制已由主控
[接受有限机制与同输入证据](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/controller-acceptance.json)，
其440项raw失败、文本负零及NPP/完整C3缺项保持历史身份，原失败/门槛不变。
上一批已[接受有限生产层证据](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/controller-acceptance.json)，
其原生累加器缺失、完整捕获失败及58份清单保持历史状态。
上批365日捕获、当前背景局部分支放大及年度输出核算已由项目主控
[验收](../../outputs/reviews/main-physics-20261027/npp-current-capture/controller-acceptance.json)，其64份清单及回执冻结。
该接受不作为本批验收；上游起因与年度因果未知。此前两次历史输入交换的
[主控验收](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/controller-acceptance.json)保留原身份。
上一批新增接受C29/C30；该批历史结果保留，原始NetCDF重启与跨年验证仍待独立裁定。

上一批接受修正（2026-10-29，未含C31）：以下两项已在main实现、通过定向回归和有界年度检查，并由主控接受。
两项共用代码提交`ea86d3fbe97b3a1ecf5c5d8dda931aea5ea2ae69`；NPP仍失败，不能将代码接受视作年度科学接受。
代码、实际测试日志及补丁绑定见[最终回执](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/isolated-c29-c30/receipt-final.json)和
[执行方审查](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/production-integration-review.json)与
[主控验收](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/controller-acceptance.json)。验收另纠正了旧spacing helper将节点间距误称冻结dh权重的文档，未改其运算。

| ID | Fortran来源与触发 | main实现与验证 | 提交/接受状态 |
| --- | --- | --- | --- |
| C29 | `hydrol.f90::hydrol_var_init:4126–4141`；`ok_pc OR ok_leak`下只累加`znh<ALT`原始根，ALT0保留原剖面，仅正normalizer触发归一化 | `hydrol_nroot_from_humcste`；dnh几何的ALT0/正ALT/零和、多点14PFT及既有ok_pc回归通过 | 已接受；`ea86d3fbe97b3a1ecf5c5d8dda931aea5ea2ae69`；综合年度NPP仍失败，单独归因未建立 |
| C30 | `hydrol_var_init:4065–4070`及`hydrol_soil_froz:8332–8460`；preprofile冻土入口必须使用真实`dlh*1000` | helper/restart/assembler显式`dh_mm`及四个driver调用；restart和窄JIT通过，两个既有dz消费者保持 | 已接受；`ea86d3fbe97b3a1ecf5c5d8dda931aea5ea2ae69`；综合年度NPP仍失败 |

本批四项最终测试对应6次组件调用，17.431秒；此前隔离4次/12.115秒仍独立计账。Day199 RHS另有
unchanged-actual及完整Fortran输入两次控制，完整返回通过原限；该结果仅排除所测RHS实现为本次
输入差异的来源。地表通量与上游evapot已不同，尚未证明surface/ENERBIL内部缺陷或年度NPP起因。
全日最早严格差异在kjit9505的TMC及底层RHS，不能称9530之前完整输入全通过；目标9530的
主要放大为surface flux差`1.1868225924e-3`→首层RHS差`1.1868226011e-3`→MC/MCL约`3.751e-4`。
4,032组动态根三类分支和under-MCR标志逐项一致，不等于所有连续值一致。F保存的AE/precisol_ns
属于pre-split，尚缺完整surface/split/ENERBIL同相位输入；详见
[独立源码审查](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/fortran/flux-top-source-review.md)。
最小surface-only[本地准备回执](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/fortran/next-surface-preparation/dry-run-receipt.json)
已通过零模型检查；252条/13,032字节仅为当时的拟捕获格式，无新增实际采集。
主控已[安排续作](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/controller-surface-followup.json)，
要求先补齐surface/split及直接上游蒸发计算的完整合同，再在同一次有界捕获中取证；不把surface-only准备冒称整条链完备。

失败年度调用的[首core局部根配对](../../outputs/reviews/main-physics-20261028/npp-day199-soil-update/jax/partial-first-core-pair.json)
覆盖14×11坐标并通过原限，最大差`1.942890293094024e-16`。该结果只映射保留的JAX输入`nroot`
与Fortran首soil-entry根；中间JAX模块不接收该输入，不称整组entry同相位或独立初始化callback。
同相位Fortran `dh`缺失，首冻结剖面仍未判定。新增完整末态的required schema/来源/alias/finite
检查通过，不等于全物理域通过；对savedV02/original/dh-only的原限比较有1,099/1,109/69项失败。
全年selected water对旧dh-only原限通过；entry mc/mcl对F soil_entry跨preprofile相位，
只作描述比较，同相位F post-prefreeze输入仍缺，不据合并干预作root-only/freeze-only年度归因。

Stage26历史结果：四包边界及集中审查S26-R1已主控验收，最终132项定向通过，**该轮有限源码审查收尾**。
见[Stage26交付入口](../../outputs/reviews/main-physics-20261026/final-source-boundaries/README.md)及下方接受分组。
新增C21–C28；Stage25的C17–C20及93项原证据保留。
最新修复登记：2026-10-26。累计28项已接受修复、16个代码提交，仍未建立319年度归因。
**V01已主控验收通过**：001点1961年前9日的编译/普通路径与动态状态恢复通过原数值容差。
[最终交付](../../outputs/validation/v01-after-stage26/domain-v3/README.md)及
[验收回执](../../outputs/validation/v01-after-stage26/primary-v01-acceptance.json)保留完整证据。
两种day9恢复结果最大差异0；累计19模型日/4调用/3进程。原NaN及来源记录两次失败保留，
经源码核验的定义域/路线合同另版通过。本次没有新增物理修复，28项/16代码提交不变。
**V02交付已由主控验收，年度科学验证仍未通过**：固定319点1961冷启动全年运行的
新NPP`0.3860031827055333`，参考`0.38518497347831726`，绝对误差
`0.000818209227216038 > 0.0003951849734783173`；AGB/BGB/GPP年度门槛通过。
见[本次交付](../../outputs/validation/v02-after-stage26/319-1961/README.md)、
[主控验收](../../outputs/validation/v02-after-stage26/primary-v02-acceptance.json)及下方V02结果。
实际1模型进程/1调用/365模型日；原年度失败、raw重启及跨点/长期限制保留。
**V03a数值执行检查已由项目主控验收**：同进程独立cold A8→B8→A8，
再独立ordinary B8，4调用/32模型日。真实outer复用及点位绑定通过；B两路线沿用V01数值状态
合同的完整比较在原限内通过。本次额外载体检查触发44项失败，其历史false保留，不称V01原科学门槛失败。
见[合同与交付收尾](../../outputs/validation/v03a-after-stage26/contract-closure-v1/README.md)及下方V03a结果。
此前primary-v03a-acceptance保留为执行聊天内部审查；真正的
[项目主控最终验收](../../outputs/validation/v03a-after-stage26/controller-final-acceptance.json)另存独立回执。
无新增生产修复，28项/16代码提交和研究分支待核对同步状态不变；不是669点或长期科学验收。

此前Stage23的C11–C16六项HYDROL/雪/CONDVEG修复已主控验收，最终定向组合72项通过；
四组准确提交及适用域见下文Stage23。此前C6–C10五项OK_LEAK修复均已完成主控验收，最终focused组合49项通过。
C6–C9提交为`27ff14b38aafc7df91ce52b8393642ca08952395`；C10独立提交为
`225cb95a6a44334390d1656d85f66fd54d342a19`。实施基线`493e830`不是修复提交。
C10源顺序修复单独分组，明确保留可能产生负剩余库存的源算法局限。C5（THERMOSOIL原水文层数传递）已完成主控验收，独立修复提交为
`a117cd5afc92ea55c9aeb528eec6f724de780111`。C4（restart tau天数回退与配置传递）已在main独立提交为
`4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99`，供日尺度和全球效率分支后续核对同步。
本次未检查或修改两个研究工作树；其C4–C28同步状态均为**待维护线程核对**，不代表已同步。
C1–C3的分支同步记录截至下述2026-10-11核查，不能当作两个研究分支的最新状态。

Stage23六项修复登记为**已接受C11–C16**，原S22 IDs保留用于证据追溯。
四组代码已独立提交，不代表两个研究分支已同步。

累计口径：**28项已接受修复，16个代码提交**；同一提交可以包含多项修复，
测试数量也不等于独立物理过程或完整模型覆盖率。以下总表、各项源码/回归依据及
精确SHA共同构成修复登记。Stage24四项候选经Stage25实现与补项验收，现计入C17–C20；
其历史处置和本批接受范围分别见下文。

## 当前首次土壤水分歧：两项实际源差异，年度起因尚未闭合（2026-10-28）

固定main `06166005e580f918f21df7e3a3ca67553e76bdb8`、319/1961原配置，实际ORCJAX导入本checkout；
无生产源码、输入、参数、容差或研究工作树修改，无commit/push。本批未新增C ID，
累计28项已接受修复/16个代码提交及42行/28ID登记原文保持。主控已接受本批有界证据，
两处源差异的生产实现与回归交回原Sol续作，尚不计入已接受修复。见
[机器摘要](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/summary.json)及
[源/数据索引](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/source-data-index.json)。

初始MC/MCL逐位相同，根最大差8.671324769338185e-11；不能声称全部初始化相同。
实际全14PFT ALT0/OK_LEAK=True/OK_PC=False，正ALT重启模板未选用。Fortran
`hydrol_var_init`（hydrol.f90:4097–4143）仅在znh<ALT分母正时归一化；实际分母0而保留raw根。
JAX `hydrol_nroot_from_humcste`（1998–2069）仍归一化全profile。完整控制154坐标逐位相同，
孤立Fguard组件对F最大差1.9429e-16；第二F捕获直接确认实际零分母/raw保留。
统一缩放可在us/humrel抵消，年度影响未验证；annual-root-guard仅准备/未执行，不登记为已接受修复。见
[初根分析](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/jax/initial-root-normalization-analysis.json)。

Day130 step35/kjit6228是所选MC/MCL首个原限事件，位置point1/layer1/tile4，非layer4或全部water首次。
更早Day1 ULP、Day4 vevapnu、Day9 transpir、Day10 sm/smt事件保留；步flux是mm/1800s（转日率×48），
sm/smt是kg/m²，MC/MCL是m³/m³，不能跨量纲按数字容差/先后归因。完整F首日求解tuple经原JAX求解器
MCL逐位复现，不支持最初事件的求解器公式缺陷。Fortran `vertical_soil_init`348–360及
`hydrol_var_init`4067–4069区分dnh/dlh与dz/dh；JAX preprofile误用节点间距作为冻结dh，
helper另两处dz消费者正确。诊断只将该输入换为已有cwrr_grid.dlh*1000，未全局改helper、根或硬编码网格。
F6228原生顶层dh=.977517107mm、纠正profile1.4939311468363007e-5，洁净J为1.4939311594241563e-5；
Day130全部48×6×11 MC/MCL最大差6.926953455e-11通过，原最大约4.27862e-7/8.61284e-7。见
[实际阶段链](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/jax/block128-134-grid-clean/phase-effect-chain.json)。
第二块漏嵌套观察器的绑定失败保留，第三块直接48次11层dh及回调结构匹配；
洁净RHS仍最大3.07515e-8超原限，完整carry差异也保留。错相位映射均只离线修正，未隐性增加科学调用。

年度唯一干预只纠正该dh来源。原生产NPP0.3860031827055333对存储参考0.38518497347831726，
误差0.000818209227216038>原限0.0003951849734783173继续冻结；干预NPP0.3860031297652873，
误差0.0008181562869700176仍失败。GPP科学通过，呼吸字段保持原诊断角色且严格诊断失败。
原生float64年均NPP J–F仅由+0.0008182072511646561变为+0.0008181543109196312；
全365日/14PFT NPP原生mask不变，337/338/339/341/342/359/364七个跨侧不一致日全保留。
纠正年Days1–198年均NPP差贡献1.177667674e-11、Days199–365贡献0.0008181542991；
floor会计不等于因果。见
[原科学合同](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/jax/annual-preprofile-grid/four-flux-original-science-metrics.json)及
[原生年度会计](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/jax/annual-preprofile-grid/native-npp-accounting.json)。

下一边界为纠正Day199 step25/kjit9530：6×11 MC/MCL入口最大差7.160808057626866e-11，
出口MC差0.00037509996692523806。两侧所查窗口同时初现顶层冰不能证明冻结起因；F under-MCR false、
根零mask与源投影萎蔫谓词匹配不能替代完整内部tuple。纠正温度/冻结、coef/bin、主/虚拟/残差求解输入
未保存，原J轨迹不能替代，也没有纠正end197完整packet。唯一下一建议是两侧kjit9529–9531全相位原生捕获，
全6tile/11层/14根mask并保留真实纠正窗前packet；该
[具体方案](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/jax/annual-preprofile-grid/day199-localization-and-next-capture.json)
在执行方交付时**未授权/未执行**；主控现按上述续作决定另立预算，原批次次数仍耗尽、
旧回执不回写。原起因目标明确未完成。

原完整年carry69项失败（68 DOC+carb_mass_total），干预完整配对carry也失败；365 daily/原生NPP和
有限水域通过、独立必需数值schema/有限性/clock通过不能升级为整年准入；可选额外字段完整源schema仍未证明。
有界DOC/carbon源路径不反馈实际
HYDROL/THERMOSOIL/GPP/NPP，不撤销carry失败。图Verify2代次2026-10-28T06:36:27Z，
关键变更/排除路径按精确源回退，观察器融合影响及单点/年/配置覆盖限制保留。
早期审查中的pending/未运行描述保留其审查时点，最终状态以本批实际完成收据为准。

实际本地年2/2、七日块3/3、组件8/8，组件science5.123s、保守监督24.32301855s；
Fortran三次提交、两次实际allocation/原cold365，共1136s/2.5244444444 allocated CPUh。
126050930 pending取消零allocation，两个实际job均COMPLETED0:0；第二捕获366字段/317748记录逐字节往返、
399项producer检查、365×14四通量及147匹配原生字段逐位保持通过。jcwu私有producer写入/zzhao只读。
JAX/F raw分别1,982,247,819/934,383,864B，JAX保守全binary2,055,868,062B均<2GiB；
406文件超200建议目标已记录。TEMP清理、无本包cache；余秒不恢复次数，金额未知。见
[总资源账](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/resource-ledger.json)及
[静态核验](../../outputs/reviews/main-physics-20261028/npp-first-state-divergence/final-verification.json)。
此前待命由明确用户指令“等待主控指挥”触发，后由主控继续指令解除，预算未重置。
历史失败、358份旧manifest工件及原主控接受回执冻结；本批未发主控聊天消息，也不自记主控验收。
研究分支同步/交接仍暂停；两项诊断差异未计入已接受修复。

## 上一批humrel起源与独立Fortran配对（2026-10-28）

本批固定main `33ab88d8c636262033cd921bd58c670e8e281f7f`及`E:/ORCHIDEE-MAN-JAX`实际包，
ORCJAX CPU/x64。复用原end337 packet/动态pcapa_en/context/九家族，唯一compiled7完成Day338–344，
worker168.719秒、监督172.67159175872803秒/900秒硬限、exit0。四次prepare均0科学，TEMP已清；
无本地annual/cold/baseline/warmup/科学重试。

| 当前证据域 | 结果与限制 |
|---|---|
| 本地原生捕获 | mandatory完整，20类×48=960条，另七日完整NPP；含六tile、51参creator、动态根及真实generic累加器 |
| 七日完整daily/NPP | 对原年度maxabs7.587966781058726e-16、对上批compiled7最大1.4113055444079075e-15，原限通过；四NPP分支精确 |
| 不受限raw候选比较 | **FAIL 440项**：200离散/240浮点，全部在assimilate=false且无active叶的Yin/canopy域，原总false不改 |
| 实际生产域 | 全48 bool门/defined C3返回/entry及唯一活动offset5全记录精确；Day338 GPP仍4.564204140244641e-7 |
| 独立Fortran自身 | 49边界/六tile/14 PFT的native-bit域通过；全文本往返784记录/980标量signed-zero失败保留 |
| 两次原水组件 | JAX控制与Fortran cross的完整6返回、全部相关根分支/locals均PASS/maxabs0；仅单tile stress域 |

本地raw先存再准入，15 overlay逆AST通过，显式clock跨原独立JIT，append-only journal与合并结果精确。
actual PEAT_OCCUR=false的generic ordered accumulator此次完整保存48步，最终原生值与NPP入口精确；
旧批缺失及derived会计不回写。Day340/342/343 tiny GPP与原年度分别相隔90,326/3,583,316/1,013,089 ULP，
绝对差均小于8e-16；不能称七日bitwise一致。新day344 packet无旧包可比较。
原atol1e-8/rtol1e-10及结构/shape/dtype/离散/来源合同保持，连续activity额外bitwise检查与实际bool门区分。
见[原准入/失败分类](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/admission-classification.json)。

offset4 HYDROL writer/tstep16180经真实state carry到offset5 DIFFUCO/tstep16181；
实际Fortran对应kjit16181→16182。tile4/PFT14第6层JAX sm3.0827056142277756，
Fortran sm3.079323123859772；共同smw_tmp3.0826979486351997、smf_tmp5.7149560143648、pcent0.8。
两侧根入/出均仅该层为1、presence1、under-MCR=false，NEW_WATSTRESS=false、DYN_NROOT_LARIX=true。
本地factor/denom/humrelv3.6402170609775793e-6，F为0，下一步各自读取同值。
本地正margin7.665592575811786e-6，F在阈值下方；根正denom归一化/零denom保留分支不同而最终根相同。
浅层无根support，aggregate humrel不额外乘tilefrac；STOMATE接收的名为humrel参数实际为HYDROL vegstress。
保存MC/MCL/DZ按原第6层积分式的代数会计与native分列，未执行内部土壤solver；更早根support形成未知。
Fortran全48步writer/reader humrel及生产GPP均0。本步sm差0.0033824903680037544，
由零基节点4/5/6的保存MCL差，经共同原权重贡献0.0029008678151902713/0.0004816225528136591/0，
派生和与原生sm差残差1.7607443281164592e-16；F wilt margin为−0.0033748247754279426。
实际end337 tile4 MC/MCL已不同（maxabs0.0016603309705315594），PFT14根精确相同；
当前writer诊断解释不建立该更早状态差的首次起因。见
[跨侧原生/派生会计](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/cross-state-analysis.json)。
详见[源码/phase契约](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/source-contract.md)
和[主报告](main_physics_review.md)本批节。

独立producer为Cancon jcwu目录下`orchidee_jax_319_humrel_current_20261027`，从冻结源隔离构建，
唯一job125893716/a14r1n19，1节点8CPU16GiB/最大2h；实际allocation500秒=1.111111 allocated CPU-hours，
RSS约3.173GiB。编译376秒、原cold365轨迹93秒，模型exit0；事后旧send文件名检查令wrapper4/调度FAILED，
错误回执及archive失败保留，仅取回现有文件离线验收。32项来源核验通过，实际PP源/binary/worker/输入/冻结文件绑定；
run.def/used_run.def与冻结版相同。实际numout及有效配置1000/.5/.5、48个firstcall=false，
仅支持旧源`den_man/weight_ven_a/b`与仓库对应literal的本case等价，原静态pending审计身份保留。
实际四通量各365×14 float32存储值与已接受Stage12精确相同，maxabs0；Day338/PFT14 GPP0、
NPP9.99999993922529e-9、MAINT_RESP−9.99999993922529e-9、GROWTH_RESP0。
完整Day338原生NPP九字段guard仍337，C3虽65输入在而完整10输出缺证据；两缺项保留。
直接TRANSFER位解码独立域与失败decimal域分列，不改文本/容差。来源、资源和捕获边界见
[Fortran入口](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/fortran/README.md)。

组件域独立准入后，PID3924唯一300秒worker实际2调用，合计1.281秒、监督4.872981071472168秒，TEMP清。
先不变JAX完整记录214叶/4109元素及整个返回通过；再原`hydrol_water_stress_diagnostics`接实际F17参，
完整6字段/348元素和101叶/814元素相关分支/locals均maxabs0。under-MCR增量是native前后counter之差的
派生接口映射，部分谓词由native操作数按源IF组合并有provenance；C3整对跳过，调用0。
见[组件域准入](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/component-domain-admission.json)及
[实际比较](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/component-admission.json)。
所测单tile胁迫域未复现翻译差，输入层水量差已存在；该证据不覆盖完整51参creator、内部soil solver或全年因果。
下一最小事实是形成该end337 tile4水状态的更早实际soil-update完整入参与原生输出对应，
现有post-solver及前entry记录不足以代替；本批不追加轨迹。

原年度NPP误差0.000818209227216038仍超原限0.0003951849734783173；28修复/16代码提交、
42行/28ID登记及原支持域不变，无新Stage/修复、生产物理/参数/阈值/容差修改。
两研究工作树未访问/修改，对齐/交接继续暂停；无GPU/训练/环境重建、commit/push、自动化或主控聊天消息。
图当前Transport closed，历史代次仅作历史记录，使用精确源回退。交付时待主控状态保留，
旧58/64清单和主控回执冻结；最终工件/两报告绑定见
[静态核验](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/final-verification.json)。

2026-10-28主控核对236份实际清单工件、报告SHA及上述机制/组件/参考证据，复用执行方检查，零科学/统计重跑。
[独立主控验收](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/controller-acceptance.json)
只接受当前局部水状态跨阈值机制和所测单tile组件的同输入一致；全部raw/文本/wrapper失败与参考缺项保留。
用户进一步明确未解决问题仍需继续；主控撤回收尾安排，改为从初始状态与连续共同边界定位首次有意义分歧，
再核对对应soil-update完整输入/输出及传播。已完成批次不等于起因定位完成，上游浮点/实现原因和全年归因仍未知。
历史主控验收/发布回执不回写，其收尾安排由
[后续目标更正](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/controller-followup-decision.json)替代；有限证据仍有效。
年度NPP原失败及门槛保持，没有新增生产修复或容差调整；28项/16代码提交及暂停研究对齐的决定保持。

## 上一批Day338 GPP：生产层定位与完整捕获缺口（2026-10-27）

固定`E:/ORCHIDEE-MAN-JAX`、main `7acead83618e64483095e69621f11e70137a9658`及该checkout实际包，
ORCJAX/JAX0.10.1 CPU/x64、Python-B/无持久缓存。复用上批真实end-Day337/tstep16175的packet、
动态pcapa_en、context及原九家族，只运行一次原factory的compiled7块Day338–344；
真实science_day_numbers跨原scan/JIT传入观察clock，原静态placeholder和独立JIT保留。
worker PID65264，调用及保存114.391秒、监督118.074427秒/900秒硬限、exit0，TEMP已清理。
两次prepare均0科学；年度/cold/额外组件/baseline/warmup/重试为0，唯一科学预算已用尽。

| 实际证据 | 准入与限制 |
|---|---|
| 七日完整daily，231数值叶/2961元素 | 原atol1e-8/rtol1e-10通过，maxabs2.17010222251378e-15，0失败 |
| 七日NPP完整九参/locals/六返回，196叶/13251元素 | 同原限通过，maxabs同上，0失败；enough/over_tax/no_alloc/negative七日精确不变 |
| 结构及定义域 | 容器/键/shape/dtype/离散/来源精确，数值载体类不要求相同；两处源未定义NaN合同保留 |
| Day338原生亚日记录 | Yin/canopy/C3/entry各48条、合计192；此前无同轨迹亚日内部支路可逐步比较 |
| 完整mandatory capture | **FAIL**，缺`('accumulator',338,0)`；可用轨迹与保存NPP支路PASS单列 |
| 新day344末packet | 已真实保存；旧day344末packet不存在，不伪造末状态新旧数值比较 |

实际`PEAT_OCCUR=FALSE`、compiled runtime开启、numpy runtime关闭，走`daily.py:1432–1499`
generic ordered loop（仍在outer compiled图内）。观察器仅挂到peat-static JIT，guard不成立而未调用，
没有原生逐步gpp_d及累加前后记录。初次离线准入KeyError、原工具版本和错误均保留，只修离线分析，无科学重试。
见[准入](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/admission.json)和
[实际路线/缺口](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/accumulator-record-gap.json)。

Day338/339/340/341/344 tiny GPP与上批0ULP，零/正保持；Day342差2.17010222251378e-15、
相对差1.4355773044550261e-9、10,248,018ULP，Day343差3.809255394568358e-16、
相对差1.9638870613581155e-10、899,435ULP。因此不称七日bitwise一致；观察/重建可能影响fusion，
关键保存NPP支路不变及数值原限通过仅形成有界轨迹准入。

Day338唯一活动/正GPP为offset5、第6个半小时、JAX零基tstep16181；Fortran kjit16182仅为时钟惯例映射。
实际humrel3.6402170609775793e-6，距1e-8阈值3.6302170609775794e-6；
swdown888.8902888737543、LAI1.2954020484751498>0.01、veget0.7262122337533891、veget_max1，
temp_growth19.24361600534081在(-4,55)内，全部活动门true。其余47步humrel0、实际活动false/GPP0。
Yin20层raw候选可正，实际8个active叶层参与积分，raw/active fallback均0；inactive raw候选不算生产正GPP。
canopy原生assimtot7.028563100383167e-5，经salinity0.5及inundation0.6012772289745216，
controlled_assimtot2.1130574723354817e-5，再按原式乘12e-6、veget_max和dt1800，
得到mesh单步GPP4.564204140244641e-7。entry逐值相同，日NPP入口也为该值且与上批0ULP。
单步单位gC m⁻²(mesh) timestep⁻¹，已经包含dt；日入口为归一化PFT日速率。
来源为`diffuco.f90::diffuco_trans_co2`2338–2384/2884–2899及`diffuco.py`2296–2372/2648–3349。
该步humrel并非仅差1ULP贴阈值，但更早扰动经分支放大仍未排除，其上游创建不在本批观测内。

真实保存end337初始累加0、48条entry和veget_cov_max1，按源代数离线推导offset5的gpp_d
为2.1908179873174275e-5，加权中间量0.0394347237717137，最终除86400得到4.564204140244641e-7，
与实测NPP入口一致。`derived-accumulator-accounting.pkl`明确仅DERIVED，不能补作原生逐步累加或宣称该链完整观测闭合。
来源`stomate.f90::stomate_main`2921–2931/3020–3032/3207–3208及`stomate_accu_*`9341–9411，
对应JAX `daily_inputs.py:171–220`、`daily.py:1432–1499`；细节见
[源码说明](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/accumulator-source-note.md)。

命名历史证据中Stage8 Day338只有下游NPP，旧JAX GPP1.2241381168386465e-6并非当前值；
Stage10完整65参DIFFUCO、Stage12水文carry主要为Day337，不能冒充当前Day338配对参考。
当前绑定stomate_history_1961.nc只有time_counter1/float32年度聚合，没有该日零/正GPP锚。
所以本批确认正值在光合生产层/entry已存在，仍不能判定翻译错误、浮点起因或年度因果。
后续最小独立参考需Fortran Day338 kjit16177..16224、全部65输入、activity/叶层/controls/GPP/累加/交付，
float64或17位roundtrip及精确离散，绑定end337/prefix及实际source/binary/输入。
历史入口和非执行命令模板见[历史证据](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/historical-evidence.md)，
Explore1000的arch/编译器/MPI/XIOS/NetCDF、输入与prefix、节点/资源/时限/费用仍未绑定，无已准入短重启；
旧qhcess资源/性能不迁作新平台事实。方案见
[下一最小事实](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/next-minimal-facts.json)，本批未执行任何远端动作。

年度NPP仍0.3860031827055333对参考0.38518497347831726，误差0.000818209227216038>
原限0.0003951849734783173；上批局部分支放大/floor0/年度会计保持其独立接受身份。
Verify图代次2026-10-26T20:35:28Z/full/complete/best_effort，JAX元数据变化、Fortran/outputs排除，精确源回退不等于全分支覆盖。
无生产/参数/容差修改、Fortran/SSH/远端/GPU/训练/付费/环境重建、commit/push、自动化或主控聊天消息。
28修复/16代码提交、42行/28ID表及支持域限制不变，无新Stage/修复ID；研究分支对齐、交接继续暂停，本批无water-origin新捕获。
交付时实际工件/两报告绑定见
[最终静态核验](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/final-verification.json)，上批manifest/回执保持冻结。
2026-10-27主控核对58份清单工件、交付报告SHA及实际结果，复用执行方验证，零科学/统计重跑。
[独立主控验收](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/controller-acceptance.json)
接受有限生产层/唯一活动步/humrel输入与有界七日轨迹证据；完整捕获失败、未观测累加器及上游/年度未知保持。
下一重点是humrel创建和独立同phase参考，不为单独补日志重跑全年；本次仅发布报告，无新生产修复。

## 当前轨迹NPP：原生机制与年度输出会计（2026-10-27）

本批固定main `84e1d8d71dd2110f35d36b525881e70a74a68f61`、`E:/ORCHIDEE-MAN-JAX`实际导入，
ORCJAX/JAX0.10.1 CPU/x64、Python-B/无持久缓存。原319点1961输入/15字段/全部开关复用，
run.def未重新物化，仍cold1+52×7；新年度worker PID57056，区别于V02 PID53212。
实际**1整模型调用/365日、2原NPP组件调用**；年度调用及保存245.062秒，监督261.016796秒/900秒硬限，
独立组件worker两调用合计监督3.257107秒/120秒硬限；系统TEMP已清理，无额外baseline/warmup。
完整原始年度返回先保存，再按V01接受的定义域与原`atol1e-8/rtol1e-10`作独立准入。

| 原限内自准入范围 | 最大绝对差 | 失败数 |
|---|---:|---:|
| 完整年度返回 / 全365日完整daily对象 | 2.7755575615628914e-17 | 0 |
| day365完整末packet | 0 | 0 |
| 当前Day338不变输入原组件六返回 / 546元素 | 1.9852334701272664e-23 | 0 |

容器/键/shape/dtype/离散及来源仍精确，数值载体类不要求相同；两处命名roughheight未定义NaN合同、
260独立源码字段分类/92额外字段覆盖缺口保留。观察可能改变fusion，本结果只证明原限内轨迹一致。
可逆源码overlay保存365日完整九参、原生分支/pump/reserve/pre-floor/negative/bm_create、
原floor操作数及六返回，算术AST逆变换精确相等。动态scan日号进入callback，准确1..365，
year1961、日末tstep=day×48−1；原static_daily_carbon请求未变，later-block自身原outer dispatch路线未改。
原executable不经代理，实际end-Day330/337完整packet、九家族、context及pcapa_en已保存，
day365完整packet在年度对象中。新tuple保留新producer身份，不回填为旧V02原捕获。

当前Day338走over_tax：GPP4.564204140244641e-7，原生pre/post-maintenance均0.3155353646470735，
growth2.5559543185369993e-8、NPP−0.3155349337862027；**原生floor修正0**。
pre-floor最小9.998478064767094e-9仍为正，源只修正负值。完整控制通过后仅gpp[0,13]置0，
另八参和其余GPP坐标逐值/shape/dtype冻结；NPP变2.7977620220553945e-13、
maintenance变−2.7977620220553945e-13，NPP有符号响应+0.31553493378648245。
当前完整背景的局部放大已证；反事实内部floor未直接捕获，不能搬用原生floor0。

全年enough302日、over_tax51日、no_alloc12日；11日floor非零且均在no_alloc。
原生`NPP=GPP−growth−pre_maintenance+原floor-rate`残差最大2.220446049250313e-16，
原生NPP对每日输出最大差0。**年度指标为365个日速率的算术平均，单位g C m⁻² day⁻¹**。
floor之和3.4274549371626826、日均0.009390287499075842；固定原操作数扣除floor项的
NPP日均0.37661289520645747仅为代数会计，不是关闭floor的年度反事实，也不是相对参考误差。

输出层Δ取当前JAX−既有绑定参考：GPP−0.00003886349333881434，维护−0.0008469903396264022，
生长−0.000010082380928505552；两呼吸合计−0.0008570727205549078，NPP+0.000818209227216038。
`ΔNPP=ΔGPP−Δmaintenance−Δgrowth`闭合残差−5.551115123125783e-17。
因此年度NPP偏高在报告通量层对应较低维护/生长呼吸，部分被较低GPP抵消；
没有全年配对Fortran原生floor/分支记录，尚不能向floor差额、上游浮点起因或某fix分配因果。
新四指标与V02相同，AGB/BGB/GPP过原限；NPP0.3860031827055333对参考0.38518497347831726，
误差0.000818209227216038>原限0.0003951849734783173，年度科学失败及严格诊断false保留。

详见[当前README](../../outputs/reviews/main-physics-20261027/npp-current-capture/README.md)、
[年度输出差额](../../outputs/reviews/main-physics-20261027/npp-current-capture/annual-output-accounting.json)、
[剩余最小事实](../../outputs/reviews/main-physics-20261027/npp-current-capture/next-minimal-facts.json)
及[main报告当前末节](main_physics_review.md)。若后续继续，end-Day337是实际可复用边界，
但仍需独立参考producer的同phase/单位/足够精度的上游输入输出；历史tuple不自动证明当前起因。
本批科学预算已用尽，未启动后续。工具分阶段版本和首次静态失败保留，不改原始receipt。
Verify图2026-10-26T20:35:28Z为best_effort，JAX元数据变化、Fortran/outputs排除，精确源回退不等于全分支覆盖。
无新生产/参数/容差修改、Fortran/远端/GPU/训练/付费/环境重建，执行方未commit/push。
28修复/16代码提交、42行/28ID表及原支持域不变，无新Stage27或修复ID；
water-origin新捕获、两研究分支对齐和交接继续暂停。实际工件/两报告/固定HEAD/工作区绑定见
[最终静态核验](../../outputs/reviews/main-physics-20261027/npp-current-capture/final-verification.json)，保留交付时的待主控状态。
2026-10-27项目主控已核对64份清单工件、交付报告SHA及实际结果，复用执行方验证，零模型/组件/统计重跑。
[独立主控验收](../../outputs/reviews/main-physics-20261027/npp-current-capture/controller-acceptance.json)
接受本批有限机制证据和输出差额核算，不将其提升为上游浮点或年度因果结论；年度失败与28项/16提交不变。
本轮版本化发布仅含两份报告，原清单和回执保持冻结。

## NPP历史局部分支机制：已由主控验收，年度起因保留（2026-10-27）

本节保留上批验收时的缺项和344日原建议；其中“未保存/未实现/未执行”是该批历史状态。
后续365日捕获及当前Day338结果见上一节，不能与历史Stage8输入或原回执合并。

本轮从固定main `2c15d4b0d76af6333084c0a8e93a33369da4b8a6`复用原计划和完整历史载荷，
按主控明确授权完成恰好两次`npp_closed_update`。这是Stage8 Day338的两个实际完整背景，
仅交换PFT14 `gpp[0,13]`的0与1.2241381168386465e-6；另八参及其余GPP坐标逐值/shape/dtype冻结。
两个历史producer、时钟F16224/J零基16223与当前V02分别保留，不能把干预称当前轨迹或新Fortran参考。
六真实返回及完整输入见[本轮交付](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/README.md)，
连续数值/残差/原限见[机器结果](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/response-summary.json)；
上批细表和当时四日缺项见[main报告的NPP历史节](main_physics_review.md)。项目主控已接受该批局部分支敏感性证据，
保留上游浮点起因和当前年度归因未知，见[最终验收](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/controller-acceptance.json)。

**小绝对GPP变化触发约0.315的局部响应已证；上游差异来自浮点舍入尚未证明。**
下表为PFT14、单位`g C m⁻² day⁻¹`；残差取干预减另一实际背景原值。

| 冻结背景及干预 | NPP原值→干预值 / 有符号Δ | 维护呼吸干预值 | 对方原值残差 / 原限 | 两主要输出严格复现 |
|---|---|---|---|---|
| F：GPP0→1.2241381168386465e-6 | 9.999999994736442e-9→−0.31504082549082607 / −0.31504083549082607 | 0.3150419810772084 | NPP +0.0004938172603840552 / 1.0031553464275122e-8；维护 −0.0004938172603840552 / 1.003155357983376e-8 | 否，已大幅响应但仍有背景剩余 |
| J：GPP1.2241381168386465e-6→0 | −0.3155346427512101→2.7977620220553945e-13 / +0.3155346427514899 | −2.7977620220553945e-13 | NPP −9.999720218534236e-9、维护 +9.999720218534236e-9 / 各1.0000000001e-8 | 是，接近原限；非exact或六返回全相同 |

规则沿用`atol1e-8/rtol1e-10`，不以“更近”代替复现，不设置新阈值。F原维护−9.999999994736442e-9，
J原维护0.31553579833759243；维护有符号Δ分别+0.31504199107720837、−0.3155357983378722。
两次大响应与只一侧严格复现并存；两侧六返回全坐标比较均未全部通过，见[完整分析](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/mechanism-analysis.json)。
不能否认分支放大，也不能把全部原生差额归给GPP；无年度解释比例。

源锚为`stomate_npp.f90::npp_calc`314–387/449–531与`carbon_kernels.py::npp_closed_update`3954–4057。
由实际完整输入和源码谓词确定，零GPP走`no_alloc`储备扣除，正GPP走`over_tax`的tax/多组织pump，
`enough`均false。源码随后对负库存补至min并从维护呼吸扣创建量，因此零/正扣库路径可放大最终净通量。
直接新增观测仅六返回，反事实pre-floor/negative/bm_create未直接输出；不从post-maint反推为实测floor。
整个carbon_kernels.py的SHA在Stage8、V02与当前均为
`56516c8690ab9f656f940c106871a50beb5531f5ea881cc8d1280d0b67a5c1e1`，原18/18同输入基线复用，未重放。

| 证据类别 | 有界结论与不能跨越的限制 |
|---|---|
| 绝对量级 / 小分母 | 当前年度reference约.385且绝对误差超原限，不是近零分母假失败；小GPP绝对量不证明其为舍入噪声 |
| 输出采样 / 聚合 / 精度 | Stage8所选core→send36/36及同输入18/18，Stage10原NetCDF高精度恢复34/34；三项float32严格诊断失败保留。不能据此排除内部数值敏感性 |
| 内部阈值 / 输入状态 | 新两次干预证明局部分支敏感；旧液水跨阈值/创建读取48桥及DIFFUCO同输入通过只约束旧窗口，未确定当前上游误差起点 |
| 已确认源问题 / 因果 | 28修复/16代码提交及接受域保留；没有新NPP同输入源级反例，也未把任何修复或局部通过认定为年度原因 |

当前V02 Day338实际GPP4.564204140244641e-7、NPP−0.3155349337862027，区别于上述历史J输入。
此前仅读取337/338/339/343四个保存日的通量，零/正脉冲形态仍在；只有day365/tstep17519终态，
缺day336完整状态、逐日pre-NPP九参/floor/native水文tuple。当前年度NPP0.3860031827055333对参考
0.38518497347831726，误差0.000818209227216038>原限0.0003951849734783173，科学与严格诊断失败不变。
当前年度最多将分支敏感性登记为有局部因果证据的候选机制，尚不能定类为已证浮点数值起因，也不预设有bug或无bug。

下一方案仅交主控：使用与V02相同生产源码/输入并记录新producer，在Day337–339直接保存完整九参/六返回/clock及所需源锚定floor局部；
须对保存V02同窗输出按原限自准入，不能把新生成tuple改称V02原捕获。
无合格day336状态则须另获准从cold-start演化；V02的337是块尾、338–344同一7日块，公开ndays339会切到tail。
因此最短保持原7块路线的是prefix344、仍仅观测337–339；块内安全停止及新capture runner尚未实现/准入。
不使用day365或历史旧tuple代替。[最小合同](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/next-observation-plan.json)
列实际API/字段/源码锚，未来预算提案1次本地CPU prefix344/900秒、0额外baseline/warmup；没有现成捕获命令或本轮新授权。
配对参考须核定输入/控制/producer；
上游舍入起因另需完整相同原生输入及独立源输出/误差事实，旧不同液水入口不能替代。
年度定因还需按原单位/聚合闭合该机制解释的当前差额及其余全年残差，本轮不重算旧表。未启动新捕获，water-origin仍暂停。
本次一个science子进程加监督进程，120秒硬限内exit0、CPU/x64/Python-B/持久缓存关闭、系统TEMP清理；
0baseline/warmup/整模型/新参考调用，无显式JIT wrapper或整模型图，仅两调用固有JAX算子。
原计划目录保留“未执行、组件0”的历史回执，新授权执行另目录保存；无生产/参数/容差修改、远端/GPU/训练/付费/环境重建。
Verify图代次2026-10-26T20:35:28Z，JAX元数据变化、Fortran排除，机制依赖精确源回退，不声称全分支覆盖。
本轮有限交付到两调用/真实结果/机制边界/缺项与后续条件/两报告核验齐全为止；raw NetCDF重启容量和跨年连续性
仍待独立验证，不展开669点或长期矩阵。28修复/16代码提交、42行/28ID表保持，无新修复编号；本次仅发布报告。
C4–C28同步仍是历史待核对登记，两研究分支对齐及交接包继续暂停；
最终工件/两报告/工作区与修复登记绑定见[静态核验](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/static-verification.json)。
主控核对21份新工件、实际调用及分析，复用既有验证，新增组件/统计重跑为0。上述344日提案保留为执行方
原建议；后续派发改为一次原365日观测，以同时保存全年NPP机制并与V02年末完整状态核对。
最多一次本地CPU积分及两次当前输入小组件干预，不改生产或容差、不扩上游水文/Fortran/远端捕获。

## Stage26 接受修复与接入建议（C21–C28）

基线main `61c6b3170419bb2c5d22eab80f98bfe314e9fa74`；2026-10-26主控已核对
S26-R1实际修改、独立修前及最终132项JUnit/log、20个被测文件与11个交付文件身份。
复用既有A/B/C/D审查及验证，无主控数值重跑。以下8项加入C21–C28；R1属于C25，
不重复编号。代码/测试按三组提交，报告接受登记单独提交，未修改两研究工作树。
C21–C25为`c214e43da57c21b68002186fbf71c67d45835bae`，C26为
`84b5d425418a6d1b18fa6dbe080348c2faa72e13`，C27–C28为
`9fc87c011a4e991f1637223b0f8fb2a224000595`。

| 分组 / 临时发现 | 接入依赖及证据边界 |
|---|---|
| C21–C24：A-MIN_WIND / A-HEIGHT_DISPLACEMENT / A-Z0_ICE / A-SNOW_DENSITY（按此顺序） | 各源owner/gate/default已独立核对；condveg有效API/main/init/wrapper、orchestration cold调用及两cache。基线已有C15/C20。A新60通过，其中实际组件/调用表达式与构造stub区分，不扩原reject/init模式或全bucket生命周期 |
| C25：S26-A-FRAC-CARRY（含S26-R1） | 与参数组一起接入writer→mapper→assembler及cold fraction；S26-R1补齐restart scaffold从成功HYDROL的post雪生成fraction，避免旧fraction与新雪混合。必需restart变量不变。18新项=原7+补项11，含实际call/真实组件及writer/packet；完整compiled/restart跨点仍V01 |
| C26：S26-B-TMC-DEFAULT / P4-17 | hydrol_tmc_update只恢复源allowed_err2e-8及说明，显式check API默认修复，未改算法/override/check=false。B9新+14旧通过；P4-02/03/04未支持check/fatal owner，P4-05条件不变量与静态guard缺口有界保留，无新支持入口 |
| C27：S26-C-PEAT-ACCUM | 既有显式组件的连续buffer/非peat reset/累积回加；不开放多peat生产。当前生产仍单PFT14 |
| C28：S26-C-PEAT-OLT | helper/core源默认1e-8，仅诊断；显式override与库存/转移保留，其他min参数及C10不变。C8新+4直接旧通过，另选C10源顺序8控制通过；两C项共用新测试文件并一组提交 |

[提交分组/hunk依赖](../../outputs/reviews/main-physics-20261026/final-source-boundaries/submission-suggestions.md)
和[当前工件核验](../../outputs/reviews/main-physics-20261026/final-source-boundaries/verification/delivery-verification.json)
绑定五个生产文件、四个新测试及两报告。补项后最终单次132通过，0fail/error/skip；
早期错误fraction参考的117pass/3fixturefail、正确参考下原源码重放、120pass中间版本
及writer-before有效失败完整保留，不混作共同baseline或最终writer证据。
原121及其manifest/被测源与测试快照保留；R1独立before为4有效fail/7pass，
最终132保留原121全部ID，18个旧被测文件不变，仅公共orchestration/carry测试更新。
该补项只改实际restart CONDVEG实参的六雪字段覆盖，缺state/显式几何边界保留；
是原carry接入闭合，不另计独立修复或接受C编号。
Stage25整93不重跑；测试数不代表全模型/全分支覆盖。

raw-restart容量以唯一源初始化pcapa_en/producer身份缺项停止，不称源未定义；
雪349true/355false/360true以入口质量不变量缺项停止，360false局部逻辑保留，无新雪补丁。
[最终M01–M12/V01–V03矩阵](../../outputs/reviews/main-physics-20261026/final-source-boundaries/final-matrix.md)
与本次主控验收一起结束有限源码审查；不把有限未知改成全配置已验证。

原[V01准备方案](../../outputs/reviews/main-physics-20261026/final-source-boundaries/next-validation-plan.md)
保留历史未执行身份。主控发布后已另行授权实际V01，生产者固定
`d5d487f101ac49c2b933c91f1cb08d42bb473bfd`，仅本地ORCJAX CPU/001点1961。
[实际汇总](../../outputs/validation/v01-after-stage26/001-api-9day/summary.json)为一次模型调用、
8个模型日：cold8/block7完整返回，8个48步日界、无missing/stopped、末tstep383通过，
每日输出264个数值叶/3384个元素全部有限。末packet扫描353个数值叶/23994个元素，
diffuco_previous_step_state及sechiba_finalize_state的`float64[1,14] roughheight_pft`
各有一个`[0,0]=NaN`，无Inf；相同坐标/值的组件条目不计为两项独立缺陷，不作mask豁免。
**原始尝试的V01独立有限值门槛未通过**；当时第9日续跑/恢复、普通cold9均未运行，连续性及普通/编译
误差尚未得到，原atol1e-8/rtol1e-10未改。后续模型停止，交主控判断，不自动启动V02/V03。

首段205.9292175秒含编译/同步，失败尝试208.6074725秒含导入/预检至失败，不能称完成
19个模型日或稳态吞吐。完整结果、daily、packet/provenance、实际有限容量和原traceback
已保存；离线0/0存储最大绝对误差0不能覆盖有限值失败。运行前后生产/输入/runner身份
一致，系统TEMP已清理，无生产修改或模型重跑。raw NetCDF容量缺项、原319年度失败、
TOPMODEL/fwet及原支持域限制、研究同步待核对均保留；没有年度改善/归因或raw重启准入声明。

后续调查未新增整模型积分或生产修改。实际NaN为001点/PFT1 bare ground，保存末状态
height/veget/veget_max/lai均0；冻结IMPOSE_AZE=false、ROUGH_DYN=true、
HEIGHT_DISPLACEMENT=.66、全部OK_LAIDEV=false。Fortran condveg.f90静态1467–1472/
动态1705–1717只赋值PFT2..nvm，INTENT(out)的PFT1未定义；JAX动态/静态初始化NaN、
循环从索引1开始，故此处保留显式sentinel，不能称源内存必为NaN。该写法来自仓库根
基线`7333b46c0b38650fb6c9250582876137831657b8`，Stage26前已存在；消费端safe mask来自
`5910ba9af857a91e565d13a25e3de2e3019080d1`。末边界保存值不能确定首次NaN日/步。

[调查入口及复用建议](../../outputs/validation/v01-after-stage26/nonfinite-investigation/README.md)
核对producer/writer/两packet及ordinary、fast-layout、恢复mapper；DIFFUCO的
`OK_LAIDEV & (index>=1)`在log/乘除/sqrt之前选择安全输入，grid标量roughheight仍有效。
11次局部函数调用（含窄CPU JIT封装2次调用）中，三条运输/消费路线NaN0与finite123
对照11叶/63元素均有限、maxabs0；反事实启用PFT14的守卫控制仍通过，启用列NaN
阳性控制使3个输出数组非有限。后者是构造组件控制，不是冻结配置的科学准入；没有
完整ordinary/compiled/restore、所有配置或AD结论，也未确定原历史producer实参。

主控正式接受的合同初始仅限本次001/1961/d5d487f、legacy14/NVM14、
IMPOSE_AZE=false/ROUGH_DYN=true的实际动态重算producer；静态控制仅机制证据。
只检查两处命名roughheight_pft的[0,0]未定义域；保留预期raw NaN、不豁免Inf，
全部已定义坐标及其他字段仍严格finite。imposed分支
定义全部PFT，不可通用豁免PFT1或inactive字段。未证明新生产缺陷；本批已另建验证器，
32窄控制及首段离线重评通过，随后按授权开始余下阶段，未重跑source8。纯验证合同改变
用另版门槛对未变物理的旧首段结果生成派生回执，保留
旧producer和原failed；若改物理，受影响状态/输出须新producer重生成，旧packet仅诊断。
原17工件/索引SHA保持，新调查独立manifest、TEMP已清理；图服务本次Transport closed，
结论依赖精确源码回退。两条第9日路线都在原进程退出后加载，须分别称完整结果重新加载续跑
与独立packet文件恢复续跑，不宣称同进程8+1边界。C1–C28仍28项/16提交，不新增C编号。

domain-v2当时只完成完整结果重新加载后的day9：tstep384–431/48步，104.8217936秒含编译/同步，
完整结果/daily/packet/动态容量先同步保存，随后packet schema/provenance门槛失败。
[只读差异](../../outputs/validation/v01-after-stage26/domain-v2/schema-failure-inspection.json)
显示数值字段shape/dtype无差异，daily33叶/423元素严格finite，packet353叶/23994元素
仅保留原两NaN；唯一结构差异为slowproc_stomate_previous_step_state的provenance从7条变6条，
缺`stomate.f90 lines 3288-3489 half-hour OK_LEAK state writeback`，其余6条完全相同。
生产记录生成位置为orchestration.py:2775的2条base、8213–8239的1条half-hour、9275–9333
的4条day-end；尚不能由记录差异推断物理漏过程或科学数值错误。当时domain-v2按seed精确比较而显式失败，
故保留[失败回执](../../outputs/validation/v01-after-stage26/domain-v2/execution-001/failure-48208.json)，
未改验证器、生产或容差放行。在该历史停止点，独立packet恢复day9、ordinary cold9及两组路径比较未运行。
domain-v2新增1模型进程/1调用/1日；当时连同旧source8累计2模型进程/2调用/9日，回归/离线不计模型。

主控现已验收[domain-v3最终结果](../../outputs/validation/v01-after-stage26/domain-v3/execution-001/summary.json)。
13个半小时字段经源顺序选择并覆盖写回，17组实际组装控制/221字段断言和8路线检查通过；
这些是标记值组装证据，不是221个独立科学试验。编译块7条与later/ordinary6条来源分别由
真实helper生成并精确核验，未全局忽略来源；12项新增拒绝控制通过，原32控制按身份复用。
原两处roughheight未定义坐标单列，其余状态/每日输出有限，原数值容差不变。

实际复用source8及旧重载day9，只新增独立packet恢复day9和ordinary cold9，共10模型日；
累计19模型日/4调用/3进程。恢复day9的完整packet含来源和每日输出最大差异0；普通/编译
1–9日及最终状态通过atol1e-8/rtol1e-10。每日输出maxabs2.3125945602942011e-13，
全状态maxabs2.5588087737560272e-7位于大尺度pcapa_en且在原相对容差内；maxrel
3.9964615879614042e-12。两种恢复都经过跨进程加载，不认证未执行的同进程8+1或raw NetCDF。
主控核对实际源码/验证器/回执及46新工件、13运行源/7取证源身份，无数值重跑。
两次历史失败、旧producer及存储原值保留，无生产代码/参数修改、无新修复ID。
V01只接受本次短窗路径与动态恢复；V01批次未执行V02固定319全年或V03跨点/长期。
随后另获授权的V02实际结果见下节，不回写V01历史回执。

参数及fraction carry影响非默认值和运行state字段；今后相应维护线程需按拟支持用途
评估compiled/restart/Teacher消费者兼容性，不能用文本合并代替数值准入或整文件覆盖。
C4–C28同步仍待维护线程核对，既有Teacher/prepared/restart/reference producer与执行版本
分别保留。本批没有同步授权/操作。319仍failed，water-origin暂停；V01–V03、C8 tiny/
负值fatal/nonfinite、C10源负库存、TOPMODEL/fwet未准入边界保留，无年度归因。

### V02实际结果：319点全年完成，NPP科学门槛仍失败

V01验收发布后另获本批授权，固定main `876ac3bb8c49469ae9d9a9e0aff5982a1e4ebd64`
（相对V01仅报告变化，生产源码不变），原319.0-057.0/I5/S26_69.834_0.0019_0.3980_97.918、
1961 cold-start365日、compiled block7和原15字段/开关完整执行一次。
[V02交付](../../outputs/validation/v02-after-stage26/319-1961/README.md)、
[执行回执](../../outputs/validation/v02-after-stage26/319-1961/execution-receipt.json)及
[四年度指标/完整离线统计](../../outputs/validation/v02-after-stage26/319-1961/summary.json)记录真实结果；
旧next-validation-plan仍为历史方案，未回写成执行回执。

实际ORCJAX/本工作区包/CPU进程PID53212，Python -B、持久编译缓存关闭；原CLI入口、
config SHA3e3ec5c4518ebf4d470c60eda3e9bd6ddfd22e5c6dee473694f2409f17e619af及
319-current/inputs/used_run.def SHA5ecb4d2615dec4d54a88f2db5444ac2820d8327ec6c10c6af16945b4ff56374a保留。
仅改输出/checkpoint目录，未重新物化run.def。1模型进程/1调用/365模型日，无年度重试；
365个48步日界、末tstep17519、无missing/stopped、15字段各365个有限float64(1,14)输出通过。
runner264.5704782秒含编译/同步，CLI进程281.3368429秒另含导入/完整对象保存/汇总；不作纯吞吐对照。
两条XLA simplifier警告保留，实际exit0。年度完整run精确回读0/0、原年汇总离线exact replay、
checkpoint完整packet含来源0/0一致。pickle tuple/JSON list仅规范化元数据，packet门槛不放宽。

实际319 legacy14/NVM14及动态/non-imposed配置支持两处roughheight_pft[0,0]的原未定义NaN；
其余定义坐标/每日输出/capacity有限。365=1+52×7的真实完整块路线由源码helper生成来源合同并精确比较，
未机械照搬001点元数据。源码必需合同及260数值字段shape/dtype通过；另92个额外字段的
独立源码shape合同未完整覆盖，逐项列出并保持有限域/完整存储检查，不声称全部packet结构认证。

AGB/BGB/GPP的新JAX依次为4.591235866253038、4.120666562048175、0.7776203738914573，
年度误差依次0.000320663158555945、0.000205098119952396、0.0000388634933388143，
均通过原`1e-5+1e-3*abs(reference)`及normalized bias/RMSE限1e-3。
**NPP=0.3860031827055333，参考0.38518497347831726，绝对误差0.000818209227216038
仍大于原限0.0003951849734783173**；normalized bias/RMSE均0.0021241981997049434，失败。
因此全年完整但科学接受false；单年度统计只含一对值，不代表逐日表现。诊断atol2e-6/rtol1e-7另列且false，
不代替科学门槛，全部字段/诊断已完成统计。

原319-current producer09968cf8及同一参考/输入身份保留；原NPP失败
`0.0008192662350707258 > 0.0003951849734783173`和旧payload直读误差末位差另列，未回写。
新NPP比旧JAX低约1.05701e-6，仍未闭合年度偏差。旧CLI只有摘要；辅助逐日比较使用有独立身份的
Stage6 JAX producer c7110e03，四通量最大日变化分别为GPP2.63996e-4/day218、NPP1.89971e-4/day218、
MAINT_RESP6.23768e-7/day364和GROWTH_RESP7.38774e-5/day218，原字段单位不变。
其窗口统计/相关性不构成修复因果，历史Fortran未补称本次producer。科学失败后继续完成离线交付，未第二次积分。

本次无生产修复/参数/容差修改，28项/16代码提交及C4–C28待同步状态不变；
两份报告更新。执行前后有限源/config/run.def/输入及历史身份一致，
系统TEMP清理；图Transport closed下用精确源回退。V03/跨点/长期、raw NetCDF容量、TOPMODEL/fwet/
支持域和water-origin暂停限制保留，未检查或修改两个研究工作树。
本批[最终静态核验](../../outputs/validation/v02-after-stage26/319-1961/final-verification.json)
记录保存工件/身份/报告链接与42行、28个修复ID原登记；无模型或统计重算。
2026-10-27主控验收本批交付，复核33份新工件与15份当前源码身份，复用13项离线检查，
未重复运行或重算统计；[主控验收](../../outputs/validation/v02-after-stage26/primary-v02-acceptance.json)
保留年度科学接受false、严格诊断false及260/92字段合同边界。该验收没有新增已接受修复，
也没有解除319科学失败、raw重启、跨点/长期或研究分支待同步限制；原交付回执保持历史身份。

### V03a实际结果：跨点复用及继承数值合同通过，额外载体检查失败保留

V02验收发布后另获授权执行本地ORCJAX/CPU批次，固定main
`a317c77750f7f2f42db21edaf05e6fa7f7a0c3a2`；生产树与V01 `d5d487f`、V02 `876ac3bb`相同。
一次预检核定真实root包/ORCJAX Python/cpu:0及47个已有输入元数据路径，原manifest和seed保持身份。
原config SHA `3e3ec5c4518ebf4d470c60eda3e9bd6ddfd22e5c6dee473694f2409f17e619af`不变；
A为001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658，used-file SHA
`f75d64dcbe321a0934368a415f1820687e00d5e6e0829f2c808bd2313e5343c2`，实际W/E/S/N −180/−178/−20/−18；
B为原319/I5/S26_69.834_0.0019_0.3980_97.918，used-file SHA
`5ecb4d2615dec4d54a88f2db5444ac2820d8327ec6c10c6af16945b4ff56374a`，实际138/140/−34/−32。
实际域、PFT14五校准值及所有legacy14/NVM14/NSTM6/CONDVEG开关见
[完整交付](../../outputs/validation/v03a-after-stage26/README.md)及[边界观测](../../outputs/validation/v03a-after-stage26/shared/cache-observations.json)，
未用YAML示例域，未重新物化run.def。

公共cold-start API的真实进程PID58044顺序运行独立cold A8→B8→A8，每次cold day1+真实block7；
原进程中不清cache、不reload。随后独立PID59312执行ordinary B8，compiled off/block0，
其他common flags含static daily carbon true与接受的V01普通路线一致。两进程顺序而非并发。
四次调用全部完成8个48步日界、末tstep383、无missing/stopped；累计4调用/32模型日，
完整run/packet/实际pcapa_en先保存并精确回读再进入下一模型调用，没有重试。
四调用计时176.7734517/24.1930783/12.1478658/107.1586996秒含编译、同步及边界保存；
外部进程286.9264433/156.1406413秒另含导入、验证及写盘，不作稳态吞吐声明。

两个A各自对接受的V01 source8及互相比对均完整exact0；B-compiled与V02前8个完整daily差0，
B-ordinary与同一daily最大绝对差3.5271785492341223e-13、相对差8.150827422816209e-13，均通过。
V02没有独立day8末packet，本批没有补造。两条新B末packet则均已真实保存。
本次B compiled/ordinary额外载体检查拒绝44个叶类型差异：43个JAX ArrayImpl对NumPy ndarray、
1个ArrayImpl对Python float；普通模型已返回且保存，exit1来自保存后的比较断言。
原[summary](../../outputs/validation/v03a-after-stage26/summary.json)、
[gate/failure](../../outputs/validation/v03a-after-stage26/ordinary/failure.json)和冻结工具均不改写，
**本次额外strict carrier false永久保留**，不伪造模型重跑通过。它不是接受的V01原科学门槛。

V01实际合同由`domain-v3/domain_verifier.py::schema`的23–36行与
`run_remaining_v01.py::execute`的118–133行实际调用的`ComparisonAccumulator.compare`（113–187行）组成：schema将数值叶用
np.asarray记录原shape/dtype，再作数值parity，不要求数值载体类相同。历史比较器自身的float64
误差算术不能独立证明dtype精确；本次仍保留完整键/容器、shape/dtype、离散精确及原浮点限。
V03a冻结`validation.py::compare`的188行在数值前增加无条件type相同条件，才产生上述44个载体失败。
公共packet/daily/lite协议的payload为object，未固定数值载体类；原pickle保留实际对象。
compiled直接重建与ordinary合并/float(dt_days)的解释仅限已核对边界，不证明44叶完整上游历史或新的物理bug。

已有[完整数值比较](../../outputs/validation/v03a-after-stage26/numeric-diagnostic-v1.json)未指定dtype地
用np.asarray读取数值叶，不改变保存数组或其dtype；全部620数值叶/27381元素及原44个跳过叶
均在原atol1e-8/rtol1e-10组合限内，数值超限0，6项负控制通过。因此按已接受V01数值状态合同，
本批跨路比较结果为通过；额外载体条件仍false。仅等价这43个ArrayImpl/ndarray和1个ArrayImpl/float
实际数值关系，不扩展为其他类型问题通用豁免，不通用忽略metadata或放宽浮点限。
最大绝对差为pcapa_en[0,2,0]的6.644986569881439e-7：compiled1127009.884473888对ordinary
1127009.8844732235，原限0.00011271098844732235；最大相对差为wat_flux[0,7,3]的
2.404596709685098e-10，原绝对组合限1.0000060882007604e-8。原值、参考、字段/坐标和allowance
逐项保存在回执。本次合同收尾复用已有完整结果、诊断和控制，无新模型/组件调用或数值重算，
额外类型检查的原失败及全部manifest身份保持不变。

每个实际对象先独立核定定义域；仅两个命名float64[1,14] roughheight_pft的精确[0,0]保留
源未定义NaN，其余定义值/daily/动态capacity有限。同路/存储来源精确；compiled末slowproc7条与
ordinary6条分别按精确源路线核验，不把合法来源差异当物理漏过程或全面忽略metadata。
源码shape/dtype合同仍为260字段已分类/92额外字段未完整独立覆盖；派生成对全字段shape/dtype
检查不消除此来源合同缺口，更不认证raw NetCDF capacity。

真实prepared LRU为A miss、B miss、回A hit；A context首/回相同、B独立。实际outer key相同，
同一executable id2183757175440由A首次创建，B和回A均按`is`命中原cache对象、cache增量0；
三个边界保存真实caller的九类动态输入及实际域/校准值并0/0回读。原函数实参、返回对象与数值图
保持原样。仅A-first出现一次inner factory边界，另两次外层命中可跳过它；不据此声称新的inner命中
或XLA编译/重跟踪次数，也不把历史Stage16替身869断言升级为真实XLA证据。

此前[primary-v03a-acceptance](../../outputs/validation/v03a-after-stage26/primary-v03a-acceptance.json)
由本执行聊天生成，是执行方内部审查，生成时项目主控尚未最终验收；该文件和身份保留。
2026-10-27项目主控已接受本次有限缓存复用、绑定和沿用V01数值状态合同的完整比较。见
[项目主控最终验收](../../outputs/validation/v03a-after-stage26/controller-final-acceptance.json)、
[合同与交付收尾](../../outputs/validation/v03a-after-stage26/contract-closure-v1/README.md)及
[主控补项审查](../../outputs/validation/v03a-after-stage26/controller-review-20261027.json)。主控复用已完成
73份工件/24份源码身份核验及执行方检查，核对新增收尾和合同来源，不重跑模型或数值统计。
44项额外载体检查原失败永久保留；原
[静态核验](../../outputs/validation/v03a-after-stage26/final-verification.json)保留当时报告身份，新核验单独保存。
系统TEMP已清理，图Transport closed使用精确源回退。本批只发布两份报告，无新增已确认生产问题/修复、新C编号或两个研究工作树同步，
42个有界登记行/28个ID、28项接受修复/16个代码提交不变。
V01原失败、V02年度NPP/诊断失败、raw重启、TOPMODEL/fwet/支持域、water-origin暂停及
C4–C28研究待同步均保留；本次两点1961八日不能宣称669点、年度/长期科学通过或修复因果已建立。

初始核查日期：2026-10-11。当时结论：共同物理模型只需维护一套；研究中的共享代码
差异可以拆成三项通用修复、一个可选 Teacher 接口和两项批处理运行能力。
初始核查完成分类、证据复核与轻量测试。C1、C2、C3 已分别以独立提交
`db4dd8f`、`e317fa8`、`2a3e303` 接入主分支并通过针对性回归。日尺度已有
这三项同等实现；全球效率线已有 C1，仍需由其维护线程同步 C2、C3。
可选接口和批处理能力继续按下述范围分别评估。

## 初始核查基线

| 工作线 | 核查提交 | 固定目录 |
| --- | --- | --- |
| main | `7cae3e02adad23674767a590fdf4d27f239ce065` | `E:/ORCHIDEE-MAN-JAX` |
| 日尺度 | `7463642eb3ff20f2352bc5732e9904b5181f11ff` | 根目录下 `runtime/worktrees/daily-coarse-graining` |
| 全球效率 | `52326bd5e67629269a4a4a1c39885276eefd5ec8` | 根目录下 `runtime/worktrees/global-throughput` |

以三个提交的文件内容直接比较，`jax_orchidee/` 的差异只有以下四个路径。
初始 main 基线最近的物理源码提交为 `5910ba9`，其后至该基线的提交是管理文档更新。
研究工作树中其他任务的未提交修改不属于本核查基线。

| 路径 | 日尺度相对 main | 全球效率相对 main |
| --- | --- | --- |
| `jax_orchidee/driver/orchestration.py` | +4 行：可选通量捕捉与回调 | +63 / -23 行：提取可转换的日块函数 |
| `jax_orchidee/stomate/carbon_kernels.py` | +36 / -1 行：两项数值 / 梯度修复 | 相同 |
| `jax_orchidee/sechiba/diffuco.py` | 相同 | +7 / -1 行：缓存修复 |
| `jax_orchidee/driver/global_batch.py` | 不存在 | 新增 325 行：批处理研究适配器 |

其他差异包括研究脚本、测试和研究数据合同。日尺度还增加 SciPy 依赖、研究
配置 / manifests，并更新 manifests 导航；这些不等于共同物理公式发生变化。
既有生产配置和数据合同文件、`uv.lock` 没有出现在此次差异中。
提交、逐文件统计和配置差异记录在
[inventory.json](../../outputs/reviews/shared-core-20261011/inventory.json)。

## 逐项结论与接入顺序

| 编号 | 改动 | 应归属 | 本轮结论 |
| --- | --- | --- | --- |
| C1 | DIFFUCO PFT 掩码改为缓存不可变 NumPy 数组 | 通用执行正确性修复 | main `db4dd8f`；日尺度 `120c637` 已同步且 14 项通过；全球已有同等实现 |
| C2 | NPP 维持呼吸份额使用已有稳定比值原语 | 通用物理核 AD 修复 | main `e317fa8`，12 项通过；保留极小分母导数策略；日尺度已有，全球待同步 |
| C3 | 叶龄周转份额消去共同叶质量因子 | 通用物理核数值修复 | main `2a3e303`，20 项通过；保留原测试、适用范围和源身份；日尺度已有，全球待同步 |
| C4 | restart tau回退复用天数上限，并传递当前季节配置 | 通用状态读取正确性修复 | main `4b92d3d`，31项相关测试及1项默认入口补查通过；日尺度/全球均待核对同步（2026-10-23） |
| C5 | THERMOSOIL显式步向getdiff转发原水文nslm | 通用物理装配正确性修复 | main `a117cd5`，新wrapper回归9项、既有窄检查2项通过，主控已验收；日尺度/全球均待维护线程核对（2026-10-24） |
| C6 | TF_DOC实际producer使用半小时干沉降dt | 通用调用单位修复 | main `27ff14b`，主控已验收；实际装配/小组件carry，完整compiled模型未证；两研究分支待核对 |
| C7 | DIC聚合仅将四个litter/soil率转步质量 | 通用过程接口修复 | main `27ff14b`，主控已验收；独立及真实混合三出口/no-flow，维护/root不重复乘dt；两研究分支待核对 |
| C8 | 共享bulk_dens按litter→soil源时刻两次赋值 | 通用输入/装配修复 | main `27ff14b`，主控已验收；raw/normalized/边界及marker回归，tiny跨步/负值fatal未闭合；两研究分支待核对 |
| C9 | sro_bottom默认5独立于nslm11/ndeep32 | 通用参数装配修复 | main `27ff14b`，主控已验收；普通/compiled/water/core/litter统一，已有API覆盖保留；两研究分支待核对 |
| C10 | DOC峰值扩散第二赋值读取已更新左通量 | 通用源行为恢复，独立分组 | main `225cb95`，主控已验收；四源算例×eager/JIT，源负库存局限保留；两研究分支待核对 |
| C11 | 雪融化潜热及两个harness恢复源值333700 | 通用源常数修复 | main `dd6d633`；独立土壤冻结lhf333600保留；旧参考不重标，两研究分支待核对 |
| C12 | HYDROL不同阶段保留tile4/5与4/5/6阈值差别 | 通用条件分支修复 | main `d48b923`；不全局取消tile6泥炭行为；两研究分支待核对 |
| C13 | 冻结dummy按源顺序执行两次含水量重构 | 通用源赋值顺序修复 | main `d48b923`；只证明固定profile片段，末态profile刷新另列Stage24；两研究分支待核对 |
| C14 | TOPMODEL解冻深度使用地表连续暖层前缀 | 已有诊断API修复 | main `d48b923`；未开放生产TOPMODEL；两研究分支待核对 |
| C15 | Z0_BARE贯通调用层并隔离compiled缓存 | 通用参数传递修复 | main `bbcdbf7`；真实组件、AST装配和缓存stub证据分列；两研究分支待核对 |
| C16 | 当步surface的ae进入诊断、carry和下步split | 通用状态传递修复 | main `2f36dc7`；真实组件组合通过，不重复计算总蒸发残差；两研究分支待核对 |
| C17 | normal末态mc刷新冻结profile，并传至dummy、输出和后续状态 | 通用源顺序/状态传递修复 | main `ad0d322`；保留normal mc/mcl/tmc，真实closure→writer→fallback及有限冻结控制通过；两研究分支待核对 |
| C18 | 矿物/泥炭非冻结查表先INT再整数加imin | 通用浮点边界表达式修复 | main `ad0d322`；局部ratio屏障、真实表及CPU窄JIT通过，冻结表达式保留；两研究分支待核对 |
| C19 | TOPMODEL诊断按cell tile6面积为所有tile选择潮汐水库 | 已有诊断API修复 | main `ad0d322`；严格阈值/多cell/控制项通过，生产TOPMODEL和fwet仍未准入；两研究分支待核对 |
| C20 | 有效SNOWCRI_ALB贯通main/init/wrapper/cold及缓存 | 通用参数门控/传递修复 | main `f8d36e1`；源默认/gate、严格布尔输入及真实组件通过，整模型compiled未证；两研究分支待核对 |
| C21 | 有效MIN_WIND贯通粗糙度入口及缓存 | 通用参数传递修复 | main `c214e43`；低风速组件源公式及门控控制通过；两研究分支待核对 |
| C22 | 有效HEIGHT_DISPLACEMENT贯通入口及缓存 | 通用参数传递修复 | main `c214e43`；保留源默认.66及非默认高度公式；两研究分支待核对 |
| C23 | 有效Z0_ICE贯通入口及缓存 | 通用参数传递修复 | main `c214e43`；冰区动量/热粗糙度组件通过；两研究分支待核对 |
| C24 | 有效SNOW_DENSITY贯通CONDVEG质量雪入口及缓存 | 已支持组件参数传递修复 | main `c214e43`；独立soil配置gate/正值检查，非全部密度consumer或bucket生命周期准入；两研究分支待核对 |
| C25 | 保存雪覆盖率并修正restart入口的post-HYDROL雪时序 | 通用状态/时序传递修复 | main `c214e43`；实际call→组件→writer/packet→DIFFUCO的18项通过，原raw缺字段回退保留；两研究分支待核对 |
| C26 | tmc可选诊断恢复源allowed_err默认2e-8 | 显式诊断API合同修复 | main `84b5d42`；接受真实迁移roundoff并保留超限停止、check关闭和显式override，不改科学门槛；两研究分支待核对 |
| C27 | 连续peat PFT间累积DOC_RUN_2_PEAT并按源回加 | 已有显式组件源行为修复 | main `9fc87c0`；连续/隔断/单PFT控制通过，未开放多peat生产；两研究分支待核对 |
| C28 | peat_OLT诊断采用源min_stomate默认1e-8 | 诊断默认修复 | main `9fc87c0`；阈值两侧/相等和实际core通过，库存/转移不变；两研究分支待核对 |
| I1 | 日块可选捕捉 `daily_flux_labels` | 共同 Teacher / 数据导出接口 | 先处理回调兼容性，再与日块函数提取协调接入 |
| R1 | 提取 `_paper_later_day_block_transition_factory` | 共同 CPU / GPU 运行接口候选 | 保持原标量路径，按既定前向合同核查提取后的行为 |
| R2 | `global_batch.py` 的 lane 组装与 `vmap` | 全球运行能力候选 | 继续留在全球研究线；满足支持范围、生命周期和运行接口要求后可进入 main |

C1–C3 已接入 main；I1、R1、R2 仍是接入候选。三个通用修复均按具体源码和
回归单独整理，没有整体合并研究检查点；C3 原提交 `5fb886c` 包含大量其他研究
改动，C1 原提交 `ad6c0e1` 也包含容量实验准备材料，均不应整体用于同步修复。

### 累计修改位置与提交索引

登记维护约定（用户2026-11-12重申）：后续须将适用的已接受修复同步到日尺度和全球效率分支；
当前索引覆盖 C1–C92，共 92 项已接受修正或支持补齐、48 个生产提交。C91 修正灌溉作用域和需求检查，C92 修正无冻结泥炭选档的除法边界。C90 统一日计数和季节统计的年末标志。C87 修正叶龄比例的取样时刻，C88 修正日平均值进入季节计算的传递，C89 为通用凋落物接口补齐前次日量。C86 修正 LPJ 最终诊断与历史输出的取值时序。C84 修正根系导水率公式，C85 补齐二维入渗参数的轴，C83 已包括通用耦合诊断入口。C82 修正基岩热容量开关，C83 修正土类和点位等长时的水文参数传递。C81 修复叶内 CO₂ 逐层状态和重启写回。C80 修正论文构建的活动下限。C78 和 C79 为冻结水文与泥炭分解活动的分箱修复。C73–C77为第六包五个独立原因；C67–C72为第五包六个独立原因，见当前来源登记；C65为compiled源中止检查、C66为初始化整组覆盖顺序，NSLM迁移补齐不另编号；C64是cold逐土壤块强制水位丢失，单点源等价边界需随修复保留；C62为潮汐输入链、C63为有效CHECK_CWRR合同，求和顺序候选不计修复；C60整合WATCHOUT支持链与本轮接线修补，C61单列旧首读取整／准入；C55的十向量与深度扩展仍属同一参数链，C43的scan扩展仍属context缓存所有权，C56–C59按四个独立遗漏登记；C49上游US_INIT扩展仍随原项登记，C50的两个热能调用点按一个权重遗漏计数，C51另计末段系数消费者的同类遗漏，C52计driver单位换算边界遗漏。C53的两个fallback按一个状态选择遗漏计数，C54为显式参数编译入口修复。
此前九条重启候选合并为C32–C38；本批gas默认值与状态丢失分别登记C39、C40，四数组/多处接线不重复计数。未来新增候选继续单列，正式接纳前不计入已接受总数。
每项持续保留修改原因与Fortran依据、准确生产/测试文件和提交、验证证据、触发范围、依赖及已知限制；
同一缺陷的多处接线合并登记，接口完善与数值算法修正分别说明。
同步时逐项核对目标分支当前实现及依赖，保留其研究改动；分别登记两个分支的接入提交与验证结果，
完成实际核验后才标记已同步。C1–C3的历史同步记录保留原核查日期，C4–C89仍待集中核对，
不从旧记录推断任一研究分支的当前状态，也不把单点/单年通过推广为全域或长期通过。

下列路径均相对`jax_orchidee/`，测试及必要harness随对应提交一起同步。
各项详细段落和[main审查报告](main_physics_review.md)保留修改必要性、源位置、
修复前后证据与适用范围；这里用实际Git提交复核修改位置，不重复历史数值实验。

| 编号 | 主要生产文件 | 准确代码提交 |
| --- | --- | --- |
| C1 | `sechiba/diffuco.py` | `db4dd8f1cc86a33dc8e343725fda2803f63fb358` |
| C2 | `stomate/carbon_kernels.py` | `e317fa81b97b8b8c959f2e7dfdfba525bb54dec8` |
| C3 | `stomate/carbon_kernels.py` | `2a3e303845154576c6f4b5c0b8155106e5bcc70b` |
| C4 | `coupled.py`、`driver/restart_state.py`、`stomate/restart_io.py` | `4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99` |
| C5 | `sechiba/thermosoil.py` | `a117cd5afc92ea55c9aeb528eec6f724de780111` |
| C6–C9 | `coupled.py`、`driver/orchestration.py`、`stomate/integration.py`、`stomate/soil_properties.py` | `27ff14b38aafc7df91ce52b8393642ca08952395` |
| C10 | `stomate/soilcarbon_kernels.py` | `225cb95a6a44334390d1656d85f66fd54d342a19` |
| C11 | `sechiba/enerbil.py`、`sechiba/hydrol.py`；另有两个雪harness模板 | `dd6d6335ba5011fcd86285984136a6799f6916ac` |
| C12–C14 | `sechiba/hydrol.py` | `d48b923e891946d3c9db77ba9b684666cb511b1d` |
| C15 | `sechiba/condveg.py`、`driver/orchestration.py` | `bbcdbf7a661cb11f5295883e54c3b1ea9d596d39` |
| C16 | `sechiba/hydrol.py`、`driver/orchestration.py` | `2f36dc7f7945d52387a2848aa124c043118ed45f` |
| C17–C19 | `sechiba/hydrol.py`、`driver/orchestration.py` | `ad0d322cc4951d4af158f4e8cbb6d306bdf374f3` |
| C20 | `sechiba/condveg.py`、`driver/orchestration.py` | `f8d36e1f8c03046e12e146aed2abb8bd286403de` |
| C21–C25 | `sechiba/condveg.py`、`sechiba/diffuco.py`、`driver/orchestration.py` | `c214e43da57c21b68002186fbf71c67d45835bae` |
| C26 | `sechiba/hydrol.py` | `84b5d425418a6d1b18fa6dbe080348c2faa72e13` |
| C27–C28 | `stomate/soilcarbon_kernels.py` | `9fc87c011a4e991f1637223b0f8fb2a224000595` |
| C29–C30 | `sechiba/hydrol.py`、`driver/orchestration.py` | `ea86d3fbe97b3a1ecf5c5d8dda931aea5ea2ae69` |
| C31 | `sechiba/thermosoil.py`、`driver/orchestration.py` | `cb4f76817acd852cde8cd8d4195e1e52bfd8c770` |
| C32–C33 | `driver/restart_bundle.py`、`driver/restart_export.py` | `77c3ac7d30ddf466a3cc2b4d6824824713a632b1` |
| C34–C37 | `driver/restart_state.py`、`driver/orchestration.py`、`sechiba/restart_io.py` | `fd0fe51495626b39ce42d50dec29b3a6cef744b3` |
| C38 | `driver/orchestration.py` | `45654fea50aec6edf7774ac0f0b90170eea617db` |
| C39 | `stomate/restart_io.py` | `76e406e0e0e75402d221bb281908fa4260bc5d66` |
| C40 | `stomate/restart_io.py`、`driver/orchestration.py` | `76e406e0e0e75402d221bb281908fa4260bc5d66` |
| C41 | `driver/orchestration.py` | `8a772de6f4977018a875093d535b6088317716c8` |
| C42 | `driver/restart_state.py` | `8a772de6f4977018a875093d535b6088317716c8` |
| C43 | `driver/orchestration.py`；开发cache读者同步适配；本轮扩到scan context ownership | `8a772de6f4977018a875093d535b6088317716c8`；`5fe049ffa249f2c53daf669d576fe590ecbbb999`（scan扩展） |
| C44 | `driver/orchestration.py` | `9c15c042a5ade9c91d91ac91a8c0ea39ed87c8f2` |
| C45 | `sechiba/lifecycle_completion.py`、`sechiba/ioipsl_calendar.py` | 3f7b482765b7b41339ee56a818b1f164490a5d1e |
| C46 | `sechiba/lifecycle_completion.py`、`sechiba/ioipsl_calendar.py` | 3f7b482765b7b41339ee56a818b1f164490a5d1e |
| C47 | `sechiba/lifecycle_completion.py`、`sechiba/ioipsl_calendar.py` | 3f7b482765b7b41339ee56a818b1f164490a5d1e |
| C48 | `driver/source_calendar.py`、`source_clock.py`、`orchestration.py`、`fast_state.py`、`dim2.py`、forcing／restart适配器及`sechiba/slowproc.py`、`stomate/restart_io.py` | 782535b4130f2ddc8cd30a251e0493709e1ab26a |
| C49 | `sechiba/hydrol.py`、`io_contracts.py`、`restart_io.py`、`initialize.py`及`driver/restart_state.py`、`orchestration.py` | `ea75bde31012cf519d4b0fa5c689d865c9d9c2b7`（初次vegstress）；`d6bad452932d1a7179ff5ab5fd80c771da89e336`（US_INIT上游补齐） |
| C50 | `sechiba/thermosoil.py` 两个energy调用；同提交包含原源Oracle及回归fixture | `6a5d4a7f2092f32ef9de064e61ee3224c5b48b01` |
| C51 | `sechiba/thermosoil.py` 两个final coef调用；同提交包含完整组件Oracle及回归fixture | `9a856c6cf859437d2436f4f3748d4c9e770838a1` |
| C52 | `sechiba/enerbil.py`实际dt元数据／amount→rate helper、`driver/orchestration.py`共享writer；联合测试及CI随同 | `9a856c6cf859437d2436f4f3748d4c9e770838a1` |
| C53 | `driver/orchestration.py`普通及compact状态构建器两处fwet fallback；实际交接与小fixture／CI回归随同 | `139e1ba7ba2db46d6f1a3b00f0592335bf40cafa` |
| C54 | `sechiba/diffuco.py`显式参数缺表时的静态光照布局接线；原Fortran冠层fixture／真实组件JIT测试随同 | `139e1ba7ba2db46d6f1a3b00f0592335bf40cafa` |
| C55 | `sechiba/hydrol.py`、`driver/orchestration.py`、`driver/restart_state.py`；同一soil/depth合同、caller及CI | `474f52088be77a543ebf181b232acfc8a02ddc02`（十向量）；`5fe049ffa249f2c53daf669d576fe590ecbbb999`（深度扩展） |
| C56 | HYDROL条件refSOC：`sechiba/hydrol.py`及restart IO/export/lifecycle，`driver/restart_state.py`、`restart_bundle.py`、`orchestration.py`；audit／合同／测试随同 | `5fe049ffa249f2c53daf669d576fe590ecbbb999` |
| C57 | `driver/orchestration.py` cold ready出口context；prepared／unprepared局部回归 | `5fe049ffa249f2c53daf669d576fe590ecbbb999` |
| C58 | `driver/orchestration.py` block state-derived静态dispatch缓存分区／快照；真实窄JIT回归 | `5fe049ffa249f2c53daf669d576fe590ecbbb999` |
| C59 | `driver/orchestration.py`中央STOMATE_HALF_HOUR_CARRY_FIELDS的temp_growth；实际clock交接测试 | `5fe049ffa249f2c53daf669d576fe590ecbbb999` |
| C60 | driver forcing／bundle／DIM2／boundary／orchestration与restart链，`sechiba/diffuco.py`、`stomate/restart_io.py`；公开continuous历史、fixture／八测试／CI随同 | `aa14d04210ab371aae6e959da193561f37f55d0e` |
| C61 | `driver/forcing_interpolation.py`首普通读取NINT和完整首日准入；实际文件／public bundle回归随同 | `aa14d04210ab371aae6e959da193561f37f55d0e` |
| C62 | `driver/tides.py`、`driver/orchestration.py`、`sechiba/hydrol.py`潮汐文件／原点／动态传递；独立参考、实际入口回归及CI随同 | `7b35196462fbfd53f531fd1dc751699092f80104` |
| C63 | `parameters/soil.py`、`sechiba/hydrol.py`、`driver/orchestration.py`有效CHECK_CWRR false语义及完整true明确拒绝；配置／缓存／数值回归随同 | `7b35196462fbfd53f531fd1dc751699092f80104` |
| C64 | `parameters/soil.py`完整ZWT_FORCE resolver、`sechiba/hydrol.py`cold逐块构造、`driver/orchestration.py`真实cold入口；两回归／CI随同，源等价仅npts1/nstm6 | `e275a33fd5864957e9b241138359c1f6001a5756` |
| C65 | `sechiba/thermosoil.py`原冻结profile检查及private Error/result、`driver/orchestration.py`scan/block错误传播；source_conditions／NSLM／scan回归、fixture及CI成组 | `d2d48e6d5007079493cf91cca21c978eb79fa24a` |
| C66 | `sechiba/thermosoil.py::thermosoil_initialize`九系数先判整组缺失再校验保留组，原参考及实际profile／energy消费者回归随同；native producer仍未证 | `d2d48e6d5007079493cf91cca21c978eb79fa24a` |
| C67 | NPP裸土SLA更新范围；`stomate/carbon_kernels.py::npp_leaf_age_sla_age_update`；原参考、裸土carry及clamp回归 | `b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12` |
| C68 | 日碳源控制量静态编译合同；`stomate/integration.py`静态与outer编译适配器；真实外层JIT回归，调用者需保持配置静态 | `b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12` |
| C69 | 分配后LAI／slai相位；`stomate/integration.py`三条日碳链；字面原caller960比较通过，原adapter参考已降为诊断 | `b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12` |
| C70 | 凋落物源小量判断常数；`stomate/carbon_kernels.py::littercalc_leak_core_with_controls`；低于／等于／高于边界、真实writer再入 | `b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12` |
| C71 | 季节日期先后顺序和写回；`coupled.py`共同season前驱与`driver/orchestration.py`调用／写回；实际前驱、packet及下一实参 | `b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12` |
| C72 | 季节小量归零精度；`stomate/season.py`三处归零helper；保留微小非零值的原参考。permafrost常量未改 | `b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12` |
| C73 | allocation源默认上限；`stomate/carbon_kernels.py::allocation_step`；同14元组120/2611失败→2611通过。driver显式读配置，不能推断全部历史paper受影响；Phase05明确0.6参考由runner显式保留。 | `f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b` |
| C74 | post-soil凋落物写回；`driver/orchestration.py`半小时／日packet及`stomate/integration.py`LPJ诊断统一取soilcarbon返回；原3-call、实际下次调用及日末carry。原littercalc阶段对象不改。 | `f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b` |
| C75 | 短forcing间隔SWdown；`driver/domain.py::_forcing_model_step_from_cached_point`；hourly同输入旧两向量失败、修后378字段／向量比较通过；WATCHOUT及长间隔原分支保留。 | `f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b` |
| C76 | 年度标量负值预处理；`driver/static.py::read_slowproc_annual_bbox_field`；bbox／nearest同输入原参考与实际salinity控制。所测最终salinity受0.5下限保护，没有据此声称同化影响。 | `f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b` |
| C77 | 时间字段首槽mask；`driver/static.py::read_slowproc_time_bbox_field`；原IO反例3/-4530/2与旧4/40/1区分，sentinel仅用于IO见证，物理消费者另用合法元组。 | `f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b` |
| C78 | 冻结 CWRR 的可用水分和取整分箱，`sechiba/hydrol.py::_frozen_cwrr_coordinates` 及三个实际选择入口。同步含矿物土、泥炭土的普通和批量含水写回回归。 | `9591652668df1afb4955d7282a6091b96d844e4b` |
| C79 | `stomate/permafrost.py` 的 Moyano 湿度档位，使用与原除法舍入一致的 binary32、binary64 边界。同带原边界、批量分解率和代表输入数据。 | `9591652668df1afb4955d7282a6091b96d844e4b` |
| C80 | 两处分解计算使用论文默认实数精度。四个论文入口、通用接口、Phase09 原参考及回归同批核对。 | `889024a213e1e665d3141668a0be5caf174814ea` |
| C81 | 叶层计算的清零、未同化点的 `Ca` 和裸地及确定缺席背景的重启写回。两处生产修改及相应回归、原参考和 CI 同批核对。 | `64d13e4d71bce938f4d0ebac80d83744eb143020` |
| C82 | 显式 THERMOSOIL 的冻土与基岩热容量分支、重启初始化补算及相应回归。 | `a77a1508ada77e2ee9a9093c77d11e839fd095a8` |
| C83 | 水文入渗、分层湿度和胁迫、两条冷启动热入口的土类索引，保留逐点有效阈值。通用耦合入口逐字段补齐配置，保留显式诊断值。 | `a77a1508ada77e2ee9a9093c77d11e839fd095a8`、`87698036a37ba4e87cf0d4af10dc9c4cc8aba18b` |
| C84 | 两条根系导水率路径均使用固定 USDA 分子。随带配置表及泥炭接收回归。 | `87698036a37ba4e87cf0d4af10dc9c4cc8aba18b` |
| C85 | 入渗显式区分逐点深度剖面与土类表。随带维护入口绑定、静态 JIT 参数及旧调用回归。 | `87698036a37ba4e87cf0d4af10dc9c4cc8aba18b` |
| C86 | LPJ 独立输出和组合入口写出建植、覆盖或收获后的对应值。随带两个入口及三种覆盖用例的回归。 | `0720d1bab0251cd460d9bcbf4bb3dcd1fa452701` |
| C87 | LPJ 入口的叶龄比例及日末、SLOWPROC、重启组装的对应写回。 | `c34b2bfb382adc432cdab96446f7817b970541c9` |
| C88 | 固定 STOMATE 的日平均值进入三个季节过程。 | `c34b2bfb382adc432cdab96446f7817b970541c9` |
| C89 | 通用耦合入口向 OK_LEAK 成对传入前次日周转与死亡量，沿用已有缩放。 | `c34b2bfb382adc432cdab96446f7817b970541c9` |
| C90 | 固定 STOMATE 主时钟向季节记忆和年统计传入同一个年末标志。 | `ad161b28c75f7f1e38d2bbf1260ec2f9eac5887b` |
| C91 | HYDROL 按活动格点和严格正灌溉需求校验作物土类，并允许小正面积作物土块。 | `43fca734fbfcd15e500b55c1519daa5b820d7da1` |
| C92 | 无冻结泥炭按原除法和取整顺序选择土壤导水系数。 | `43fca734fbfcd15e500b55c1519daa5b820d7da1` |

### C53／C54 状态选择与显式参数编译入口（2026-11-17已接受）

C53按hydrol_soil:7307–7342保留新湿地诊断，无新诊断时才沿用旧值；两处旧fallback在普通／runtime／fast布局中将0.412613023955767覆盖为0.123。C54按diffuco_trans_co2:2214–2234复用现有静态光照分层helper，显式参数无表时在进入真实local JIT前建立20层布局；已有表不重建。两项均在7cae3e02存在，精确首次引入未知，不改方程或容差。

主控复用19项交接和10项隔离回归／Ruff、冠层10例原源eager／JIT证据，未重复数值运行。未来同步同时核对共同writer、实际参数适配器、开发fixture与回归依赖，保留C48／C49／C51／C52既有行为。TOPMODEL=True wrapper尚未准入、默认冠层getter本来供表，不能声称论文自然路径的已观测失败或修复后科学指标改善。水文自定义土壤参数四项差异、不同土层数IO及整模型门仍未关闭。

### C51／C52 热系数权重与driver蒸发单位（2026-11-17已接受）

C51按源thermosoil_main:1002–1009将补足裸地的veget_max_bg传给两个末段coef消费者，复用既有helper；它不重复登记C50的energy实参，也不改coef算法。旧52字段反例与修后552字段／112332标量通过均绑定原8例参考。C52按enerbil_flux:1505–1508及intersurf_main_2d:743–749区分每步kg m⁻²和driver kg m⁻² s⁻¹；实际执行dt随payload传到唯一共享writer后除一次，SECHIBA amount与driver codec不再转换。两个旧错在7cae3e02已存在，精确引入提交未知。

共同提交包含三个生产文件及原源fixture／开发Oracle／回归与CI；有限接口15项通过，隔离CI26项与Ruff证据分别复用attempt05／06。后续同步必须成组核对C50／C51权重依赖、实际dt和writer各caller、amount／rate命名空间及独立测试，不能只替换整文件。积雪strict xfail仅放行已知case的有限grain差异，其他字段先正常检查；这一测试保护不计新模型缺陷。自然雪域、外部PBL producer、真实reader／历史身份和整模型／NPP／DOC门仍未关闭。

### C50 热能诊断权重（2026-11-16已接受）

源thermosoil_main:913将裸地补足的权重交给energy，explicit及no-explicit两个JAX调用曾传raw权重。复查前7cae3e02已存在，两点按同一概念错误计一项。稀疏植被原源反例和修复后eager／窄JIT证明分支修复；完整no-explicit、完整compiled及整模型不随此接受。生产复用现有共享权重函数，仅改两个实参，原energy API不改。未来同步须携带同提交中的独立fixture／测试与当前相位合同，不能从一次组件通过外推其他科学结果。

### C49 HYDROL初始化链（2026-11-15初次接受，2026-11-16上游补齐）

源 `hydrol.f90:3039–3058` 对整个本地vegstress数组全999999的情形，在原值上加由已物化us求和并选出的土壤stress；普通与partial数组保持。已识别IOIPSL读取链支持有效已有文件缺变量时填999999。复查前7cae3e02完整owner已遗漏该分支，首次引入提交未知；源的特殊加法按语义保留，不作为新科学公式。共享规则、真实文件初始化reader、默认完整HYDROL owner及state展开、新年度`new_native_run`接线同属C49。连续年、同年内存继续和完整checkpoint解码不重放初始化，raw／strict93／writer合同保持。

初次32项不同检查具有通过证据，含4个真实NC入口、dispatcher后续consumer及finalizer写回、相位和US_INIT边界；首轮两项测试夹具的1 ULP失败和定向修复过程保留，未改算法或容差。物理与runtime委托被替换，没有执行完整物理初始化或模型积分。初次接受限于us已完成上游物化的声明域，当时整层缺失US_INIT显式拒绝；不追改该历史保证。

2026-11-16接受上游补充提交 `d6bad452932d1a7179ff5ab5fd80c771da89e336`：按hydrol2821–2832及sechiba_io_p485–508逐层完整local ALL条件执行scalar US_INIT，再保留物化us到实际file／owner／年度consumer双镜像。复用既有generic setvar，配置支持typed scalar和plain decimal、惰性读取；连续与解码不重放。唯一相关84项通过，实际aggregate＋assembler组合及写读回有证据，大型scaffold只核对源码实参，没有完整物理或模型积分。该合法支持域扩展与新参数／state接线随C49计账，不拆C50；原生／完整parser、NPP／DOC、历史身份及整模型数值接受仍未升级。两个准确生产提交、各自测试和合同须成组核对后再同步到研究分支。

### C48生产日历更新与状态接线（2026-11-14已接受）

源 `slowproc.f90:556–610` 在STOMATE之前执行Jan1首active步的growth_day清零、Jan2首active步的peatC_ok条件写入及午夜temp_growth>5的growth_day增量；复查前JAX已存在owner，但选定driver没有调用／保存这些更新。当前由单一共享owner连接cold／first／ordinary／compact／compiled和日年界，绑定实际forcing／restart时钟，游标不进入原93 science字段，三文件IO保存显式版本化metadata。完整旧b5 builder在相同替身下失败、当前builder通过；120项最终通过，其中初次7项夹具失败及定向修复保留。接受限于源与合成适配器／IO语义，未执行物理模型、不升级年度／DOC／热／maxmass接受。该项按一个共同接线遗漏登记，不拆成多个字段bug；C45–C47仍为独立primitive修复。研究同步待后续统一核验。

### C45–C47时钟基础接口（2026-11-14已接受）

| ID | 源依据与旧反例 | 已接受修复／实际范围 |
| --- | --- | --- |
| C45 | intersurf:1893–1894调用calendar::tlen2itau(1Y):358–364；旧helper将年长天数直接除以步长秒，结果0 | 按实际起点日期跨年取差，源NINT；noleap1961为17520步、Gregorian2000／2001为17568／17520，首次常量在同run后续保持 |
| C46 | calendar::ymds2ju/ju2ymds使用Lilian日数及2299160偏移；旧helper把同一值作astronomical JD，解出错误年份 | 恢复源整数运算／日期归一化，Gregorian2000-01-01=152385；显式IOIPSL origin、独立startdate锁和previous binding，禁止静默重释旧payload |
| C47 | intersurf非Gregorian分支保留REAL sec；旧返回状态int(second)截断 | 保留原生900.2499952912331秒，区分scalar／ymds startdate；Gregorian owner仍按源次序NINT，不把primitive秒与owner秒混为一谈 |

三项共享提交3f7b482765b7b41339ee56a818b1f164490a5d1e。两生产文件、两个unit文件、新native fixture及旧oracle归档guard须一起核对；后者保留原TEMPLATE／CASES，禁止用synthetic2001入口验证新helper。仅canonical Gregorian/noleap获得本次支持；旧无绑定callback／previous、seconds-valued年份及不明epoch明确拒绝。三个旧表达式均在7cae3e02及7333b46c存在，候选实现审查中的NINT near-half纠正不另计编号。

原生新参考为1作业／11进程／14调度秒；41条clock与配对源码及Python一致，4个metadata进程含1个预期缺轴错误。唯一80项本地检查包含这些41条比较、6个纯DIM2映射、3个旧源码反例和相关控制，不能加总为独立多套验证。主控复用原回执和独立终审，未重跑。代码接受不认证生产日历／compiled／year-restart接线、历史001身份、实际年重启连续性或整模型数值。原DOC／热／maxmass等限制保持，本次未同步研究分支。


### C44冷启动热参数与静态碳输入绑定（2026-11-14已接受）

源`thermosoil.f90:202–204,338–371,563–567`要求fresh-module默认与条件读取，两个cold caller原先各自缺省FALSE，会跳过应读取的refSOC并传错系数控制。一个不可变控制对象现统一有效soil-C／refSOC／freeze／satsoil／bedrock及静态输入，复用配置声明的reader，并将相同satsoil传给初始化和首步。形状／有限性检查及现有数据错误保持；未绑定的modeled/mineral/toporganic等消费者在物理初始化前返回精确缺项，未新增这些物理分支的支持。相关支持域详见[当前报告](main_equivalence_remaining_work.md)。

一组接口修复对应一个代码提交，多字段／两处默认／多个caller不重复计数。旧AST缺省反例为读取0／flag FALSE，修后读取1／TRUE，显式TRUE对照保持；`tests/unit/test_cold_thermal_inputs.py`50项一次通过，含真实early guard和实际caller／carry表达式。静态reader和物理消费者使用替身，不认证完整cold路径或独立热数值。主控未重复运行。cold默认遗漏有`7cae3e02`及`7333b46c`基线证据；依赖C41的有效OK_PC resolver，C42为独立restart helper并未改变。后续同步须包含本提交的控制resolver、reader、caller、支持域检查及测试，并保留研究线自己的改动。时钟候选不计入C44，本次研究分支未同步。

### C41–C43参数／缓存修正（2026-11-13已接受）

| ID | 源依据、必要性与触发边界 | 修复及验证范围 |
| --- | --- | --- |
| C41 | `control.f90:261–278`在OK_LEAK开启时覆盖OK_PC=false；raw两者为true时旧context及独立首步实参错误 | 共享有效flag resolver覆盖prepared、cold和restart入口；四布尔组合、context字段及四caller实际关键字表达式通过；AST表达式回放不等于物理caller执行 |
| C42 | `thermosoil.f90:362–371`在可选读取前初始化use_refSOC=true；新restart准入helper缺键时错误拒绝 | 源soil-C门内缺键TRUE，显式FALSE和原禁用门保持；缺键／T／F及门控通过。依赖C34的helper，独立cold默认与静态IO未因此闭合；非热数值真值接受 |
| C43 | compiled later-day闭包捕获prepared context，但旧key漏其身份；同config／layout、不可变FR_DT2／3的缓存B曾向spy转发A参数 | context实例分区、typed强引用entry及8项FIFO；同实例复用、不同实例隔离、fresh／cached、无闭包保活／clear／淘汰通过；等价独立实例额外编译是已接受代价 |

三项共享一个代码提交，相关回归为`tests/unit/test_parameter_context_repairs.py`、三个stage23／25／26 cache测试及既有restart gate测试；直接读者为`scripts/dev/audit_compiled_fast_path_acceptance.py`。一次50项pytest与11项日历原型检查通过（10正常＋1预期缺证）；旧红例及独立终审复用。C41／C43在`7cae3e02`已存在，C42是`fd0fe514`新增helper的缺省遗漏，不能统称复查前bug。日历未修、无新C号；物理／真实compiled／年度科学和原严格状态门均未升级。后续同步需同时核对caller、cache读者与测试依赖，保持研究分支自身改动；本次未同步。

### C1–C40来源年代与性质（2026-11-13）

[逐项来源审计](correction_origins.md)以复查前main `7cae3e02adad23674767a590fdf4d27f239ce065`为明确边界，逐项核对原累计登记、基线表达式/遗漏及22个准确修复提交。40项均有直接基线证据，来源标签为`baseline-defect-present`；另三类（复查期新扩接口遗漏、已证实复查引入回归、来源不足）本次均为0。首次已知有错版本是该基线，准确最初引入提交未追溯；不能用修复日期替代引入日期，也不能由此保证每个中间改动都无回归。

40项按主性质分为源过程计算/常数/单位/顺序13、默认/参数/状态/restart接口20、显式诊断API4、执行缓存/AD稳定性3，不等同40条独立且自然激活的物理公式bug。C40初版未完成既有首日carry遗漏，不自动算新回归或另增编号。逐项基线行号、函数、完整repair SHA、置信度及未知边界见新表和[机器证据](../../outputs/reviews/main-physics-20261113/final-gas-runtime-and-fix-origins/origin-audit.json)；本节不改历史接纳身份、运行覆盖或研究分支同步状态。

### C39–C40气体持久状态修正（2026-11-13已接受）

两项分别是错误的缺值默认与未传递的持久状态，属于源合同修复；不等同新增两条活动气体物理公式。两项共用一个代码提交，包含`tests/unit/test_stomate_gas_lifecycle.py`及`docs/source_audits/restart_state_lifecycle.yaml`。本组承接C32的packet优先writer合并及既有固定配置carry接口；以后同步须检查这些依赖，不能只复制整份文件。

| 编号 | 修改必要性与Fortran依据 | 验证与边界 |
| --- | --- | --- |
| C39 | reader缺字段/全数组sentinel曾回退为0；按`constantes_var.f90:1467–1468`和`stomate_io::readstart:1177–1199`统一为O2=298.3、CH4=0.001267 | 真实NC覆盖缺字段、全/部分sentinel、显式override及已存值优先；不据此证明缺失的外部IOIPSL在历史NONE运行中的实际行为 |
| C40 | cold producer/首提取未创建四gas，cold/restart共用的首日scaffold另有遗漏carry清单；补齐后接入既有普通/compiled布局、日末及写读链，保持`writerestart:2801–2812`持久合同 | 两条非默认状态反例修前均在真实首日carry处失败，修后16项相关回归通过，含故意置错writer base的真实三文件写读；最终001cold365四gas保持通过。接纳时非默认重启实际编译运行待补，本轮319cold8/001人工非默认restart8均已通过实际首日、完整七日编译调用/返回和三文件写读，见文首新包 |

初版319cold2、001paper-start2实际gas失败及各自源码冻结完整保留，不能充当最终版本回归。最终001cold365全部原年度门通过，只接受该点/协议/年度；gas动力学当前关闭，001初始化协议纠正不计为模型代码修复，也不归因gas改变。C34源相位核查无新增修正；DOC/碳总量严格状态失败、独立热真值、maxmass条件域和历史001输入身份仍开放。主控复用独立终审和owner验证，未重复积分；两研究分支本次未核查或同步。

### C32–C38重启修正登记（2026-11-12已接受）

以下为七项概念修正，不等同七个活跃物理公式错误；源码依据、修前反例、完整测试及原九条候选到正式编号的映射见文首主控接纳记录和两份owner审计。59项回归绑定三个提交的最终组合，中间提交未分别运行。两研究分支均待以后实际核对。

| 编号 | 必要改动与源合同 | 验证及适用范围 |
| --- | --- | --- |
| C32 | `stomate_io::writerestart`持久化字段由当前packet所有；bundle与standalone共用完整gas/remainder合并 | 真实NC修前失败及写读回归；缺字段保持独立声明base。旧C31缺四gas，历史gas连续性仍未证明 |
| C33 | `stomate_lpj:1545–1548`、`stomate_init:1586–1589`：缺失total由实际合并pool重建，显式不一致值交writer拒绝 | 三项旧merge→真实writer负例及当前正负控制；不自动修平显式不一致数据 |
| C34 | `thermosoil_initialize:337–374,572–744`：源gate下在main前重建非持久热状态、保留递推系数；同时携带持久`e_soil_lat`接口 | 限显雪/refSOC/soil-carbon-temp路径；三路实测热字段、A/B相同。alias不单列物理修复，readjust调用仍注释 |
| C35 | 首步STOMATE readstart使用显式恢复目录 | 真实调用表达式→boundary→NC reader正负控制，以及实际B路径绑定；未做逐文件deny审计 |
| C36 | `slowproc_initialize:275–302`、`stomate_init:2238–2261`及`stomate_main:5049`：active首步沿用恢复assim/height、正确重算deadleaf、月温初始化growth并保持到日更新 | active/inactive组件控制与B/001真实首步及后继growth消费；初始化和保持归一项，不扩大全部sentinel配置支持 |
| C37 | `njsc/veget_year`遵守Fortran INTEGER owner，NC REAL wire在运行时恢复有限精确int32 | 真93字段NC读取、非法值拒绝及A/B末态类型通过；driver只认证USDA12，veget_year不发明日历范围 |
| C38 | compiled静态模板与实际年forcing分离，start0盐度/潮位使用实际series首样本 | 完整binder start0/start48及禁止sentinel天气读取的组件控制；本轮三条单日路线未触发compiled firstcall，不称整年覆盖 |

七项接受不关闭DOC原严格失败，不认证当前版本的多点/年度/完整重启；001仅接受精确源sentinel实例分类及首日完成，原nonfinite=false和historical verifier限制保留。

### Stage24验收时的处置（历史；四项修复现见Stage25）

2026-10-25主控验看实际探针、保存值和成对源码，接受[有限审查证据](../../outputs/reviews/main-physics-20261025/main-source-closure/README.md)
的声明范围。下一批修复按下表处理；在实现和回归验收前，不分配新的已接受C编号。

| 发现ID | 已证实的范围 / 修改依据 | 下一批处置 |
| --- | --- | --- |
| S22-DUMMY-REFRESH | 源hydrol_soil6884–6886用normal末态mc再次刷新冻结profile；当前dummy和返回状态沿用旧profile，真实表组件对照已复现差异 | 修复刷新及状态传递；保留normal mc/mcl/tmc，补热力/线性冻结及直接调用组合回归 |
| S24-CWRR-NONFREEZE-BIN | 源8291–8301先INT再加imin，当前先加再INT；合法浮点边界选bin16/17 | 只对齐非冻结矿物/泥炭表达式；真实表、邻近边界与窄JIT补证，冻结表达式保留；现有value-tag不证明物理幅度 |
| S22-TOPMODEL-TIDE | 源7286–7293按cell的tile6面积为所有tile选水库；当前仅tile6选tide，显式参数组件差异已复现 | 修已有诊断API选择条件；保留生产TOPMODEL未准入，不扩展参数IO/生产支持 |
| S24-CONDVEG-SNOWCRI | 源已生效的非默认SNOWCRI_ALB在wrapper/main/init丢失；组件20给.5而源为1/3，默认10一致 | 先补实参入口与OK_SECHIBA/IMPOSE_PARAM有效配置门控证据，再修传递/缓存及相关调用；不能直接采纳门控关闭的raw覆盖值 |

fwet覆盖、其他四个参数接缝和五处检查/默认合同保持各自条件/静态证据域，
不自动计为生产bug或修复。雪maxmass三条普遍不可达证明撤回，登记为条件域未证明；
这次是证据登记修正，不是新的模型算法修复。319年度NPP原科学失败保留，
C1–C16及本批新差异均未建立该年度差异的原因或修复后改善。

### Stage25接受登记：C17–C20

基线main `7905b1165f6e812bf87514017765278f9df5da6d`，固定根及ORCJAX实际导入
均为`E:/ORCHIDEE-MAN-JAX`。Stage24上表保留验收当时的处置；四项已按后续授权
实施。2026-10-26主控已验看两补项实际代码/测试、修前失败、最终JUnit/log和八文件
交付身份，接受下表四项，按顺序对应C17–C20。**累计20项、13个代码提交**，
原C1–C16准确SHA完整保留。A-R1属C17状态闭合，D-R1属C20新helper合同，不另计缺陷。
C17–C19提交`ad0d322cc4951d4af158f4e8cbb6d306bdf374f3`，C20提交
`f8d36e1f8c03046e12e146aed2abb8bd286403de`；两研究分支同步仍待维护线程核对。

| 原发现ID | 必要性与修改位置 / 补丁组 | 输入条件、修前/修后及未证域 |
| --- | --- | --- |
| S22-DUMMY-REFRESH | 源hydrol6884–6905按normal末态mc更新profile、7165–7174只还原normal水状态。`hydrol.py`刷新/dummy/返回及module接口，`driver/orchestration.py`四调用传实际dh和五冻结控制；A-R1实际state writer改从soil结果持久化新profile；A组，依赖已有C13 | 冻结开启、normal mc改变；thermo/linear/module/实际closure四个修前数值失败，off通过；修后独立profile/真表/RHS/两次重构与normal mc/mcl/tmc还原、diag/下游组合、linear窄JIT和缺真实输入拒绝通过。A-R1真实零雪closure→writer/packet及无THERMOSOIL fallback在初版复现.000794945001109798误差，补齐后通过并保留normal状态；普通有THERMOSOIL下步重算，不能泛化逐步影响。土壤lhf333600和固定profile低层API保留；不认证normal solver、完整thermo JIT/refSOC/模型年 |
| S24-CWRR-NONFREEZE-BIN | 源8291–8301非冻结ratio→scale→INT→整数加imin；只修`hydrol.py`两个真实表consumer，B组。非冻结ratio局部屏障防编译重排，冻结8261/8275表达式保留 | 合法浮点边界.1682给源bin16、旧eager17；矿物/泥炭各16个边界/端点×eager/JIT及freeze控制均通过。整数顺序初版3pass1peatJITfail，局部屏障后4pass；原consumer AST隔离重放D差为4910.4347847302415/268.56503923288255mm²/day，不冒充共同修前/编译Fortran。无epsilon/生产nextafter/全局flags/容差调整；任意backend、逐位/性能未证 |
| S22-TOPMODEL-TIDE | 源7286–7293按cell tile6面积严格阈值为所有tile选择水库，`hydrol.py`已有诊断API，C组，依赖已有C14 | 3cell含无/等于/略高阈值，tidesoff/等水库/peatoff/既有5tile组件合同，eager/JIT及连续暖层旧回归通过；修前meanwt最大差.04m。生产TOPMODEL=true、四参数IO和fwet carry仍未准入/未修 |
| S24-CONDVEG-SNOWCRI | 源control455/495–503/550–555、intersurf2267/2302–2310/2330–2335和constantes_var129/697、constantes507–513核准effective gate/default。`condveg.py`resolver/main/init/wrapper，driver cold直接调用及两effective缓存键；D-R1补齐严格boolean输入合同；D组，依赖已有C15 | 已初始化SECHIBA且IMPOSE_PARAM=true才读raw，缺impose用源默认true，关闭gate不消费raw；explicit标量API表示已生效。有效20+雪10旧.5→源1/3；修前4fail/8controls。D-R1未知active gate改为ValueError、既有true/false及initialized-off不读raw保留，不计新物理缺陷。D50个唯一ID在新93组合通过，初版D44历史身份保留。实际组件/五call表达式×三gate情况/两个cache构造stub分列，不宣称完整driver或其他四参数/unsupported init准入 |

修前共同原状27项为17有效fail/10pass，保存三生产SHA、空diff/index和实际rootimport，
父级验看后才放行生产。初版选择61个新测试ID（保留原27，新增A2+D32）与25个直接相关
旧回归。实际组合84pass/2fixturefail，随后仅修directmain测试函数的合法无albedo输入，
所有断言/容差及生产保持，逆还全文件SHA证明无其他D测试修改，再仅重跑该2项为2pass。
**86个唯一ID均有pass证据，不是单次86pass**；原失败回执保留，不登记为物理反例。
[初版联合验证记录](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/verification-review.json)
及原失败回执保持历史身份，原入口工件另存`history/initial-delivery/`。

主控两项补充仍属Stage25。在与初版三生产SHA一致的dirty实现上先保存
[补项修前7项](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/revision-before.json)，
2有效fail/5pass/0error/0skip，无fixture/import/TypeError；父级验看后才改A writer和D gate。
三tests身份不变，按原12个selectors在最终生产一次完成
[93项全通过](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/revised-final.json)：
旧86 ID全部保留，新增A1+D6，仍为25个直接相关旧ID；
[补项联合验证](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/revised-verification-review.json)
分列命令、ID、JUnit/stdout、稳定身份、rootimport和TEMP清理。未重跑共同修前27或原调查。
B局部屏障另复用主控CPU JAX0.10.1两consumer非边界grad/vmap的8个有限观察：
与旧consumer AST body在当前namespace/真表中重放匹配，不是历史整module或一般AD准入。

三生产文件、三新测试和本两份报告组成可审修改；A–D生产hunk、联合测试和报告依赖组
分别打包，并在系统TEMP内从基线副本依次apply/check和逐文件重构核对。
[main报告](main_physics_review.md)的第二十五阶段及
[交付核验](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/delivery-verification.json)
保留具体源/当前函数与有限文件身份。执行方交付时未stage/commit/push，主控在验收后
按上述两组提交代码及测试，接受登记单独提交；不修改或同步其他研究工作树。
原manifest/patch保留交付时草案身份，主控接受更新不回写历史回执。

后续同步按C17–C19水文组和C20参数组核对。前者在含C13/C14的基线上建立，后者承接
C15接口/缓存；若研究线已有同等实现，先比较hunk与调用合同，不整文件替换。
三份Stage25测试和直接相关旧回归随组核对；93项CPU组件结果不代替研究线的GPU、
梯度、compiled运行验证。既有Teacher/prepared/restart保留原producer身份。

319年度NPP仍failed、water-origin暂停，V01–V03、C8/C10/Stage20/raw-restart及旧
Teacher/prepared/restart身份保留；本批未建立年度影响。无模型日年/完整模型编译、
Fortran/oracle/广泛suite/SSH/服务器/GPU/训练/付费/环境重建或自动化。
执行方复用既有Verify图generation2026-10-25T07:07:50Z及Transport closed后的精确源回退。
主控验收时服务恢复至generation2026-10-25T14:39:20Z，定向writer/parser/调用查询配合
coverage及当前源码核对；Python metadata_changed、Fortran excluded限制仍保留，
没有重建图或全域覆盖声明。剩余一批有限源码边界审查及集中收尾见main报告；
整模型科学验证另列V01–V03，不因本次修复登记而视为完成。

### C1：与 GPU 批处理无关的缓存缺陷

核查基线 main 的 `_diffuco_pft_column_mask` 在 `lru_cache` 内创建 JAX 数组。若首次调用
发生在追踪中，缓存会保留该次 `DynamicJaxprTracer`。全球分支仅改为创建相同
布尔值的只读 NumPy 数组，由消费处在各自追踪中转换；没有改变物理公式或掩码值。

本轮用从分支回归改编的小数组测试检查 14 个 PFT 和两种独立追踪形状。
main 在 `jax.checking_leaks` 下 14 项全部复现泄漏；在同一 main 模块内注入
精确候选实现后 14 项全部通过，包括列替换结果、dtype、掩码内容与只读属性。
这不是“所有正常主模型运行都会失败”的断言。

源码及正式回归已作为独立提交
`db4dd8f1cc86a33dc8e343725fda2803f63fb358` 接入 main。该提交只包含
`jax_orchidee/sechiba/diffuco.py` 和 `tests/unit/test_diffuco_mask_cache.py`，
便于日尺度分支单独同步。修复后的 DIFFUCO 文件与全球分支 `52326bd` 完全一致。

在实际修复后的 main 上执行
`conda run -n ORCJAX python -B -m pytest -q -p no:cacheprovider tests/unit/test_diffuco_mask_cache.py`，
14 项全部通过；测试文件 Ruff 检查和 `git diff --check` 通过。
测试使用 `PYTHONDONTWRITEBYTECODE=1`；关闭 cacheprovider 产生的 `cache_dir`
配置警告不影响结果。记录见
[主分支回归日志](../../outputs/reviews/shared-core-20261011/main-diffuco-fix-pytest.log)。
预审[补丁](../../outputs/reviews/shared-core-20261011/global/diffuco-mask-host-cache.patch)
和 [global-evidence.md](../../outputs/reviews/shared-core-20261011/global/global-evidence.md)
保留初始复现过程。

日尺度已由其维护线程同步为 `120c637`，保留来源记录并通过同一组 14 项回归；
核对 main / 日尺度的 DIFFUCO 文件无差异。全球分支已有同等修复。
三条开发线的这项修复已一致，历史实验执行版本仍保留原身份。

### C2、C3：共同修复，不依赖神经网络架构

C2 来自 `9023de2a3fbb3ddb49f4476d9bdac4b003502d1b`。它保持 NPP 前向源商、
泵出碳库及掩码，改用 main 已有的稳定比值原语。该原语在
`abs(denominator) < sqrt(finfo(dtype).tiny)` 时把商的切线设为零；FP64 阈值约
`1.49e-154`。这是一项明确的导数策略，不是任意小尺度下精确求导的证明。
独立方向探针中 `n=d/4, dn=d, dd=0` 的数学方向导数为 1，而实现于阈值下为 0、
阈值及以上为 1，前向值均为 0.25。本次接入保留原有阈值和上述导数策略。

C3 保留 Fortran 顺序的叶、根、果等 stock / turnover 更新，只在有效正叶质量
区域以相对质量计算叶龄份额，避免数学上会抵消的巨大梯度项。实数代数一致，
浮点份额的计算顺序有变化；零、负、阈值以下及退化情况保持原路径。

本轮复核了 Fortran `stomate_npp.f90::npp_calc` 299–386、447–531 和
`stomate_turnover.f90::turn` 569–665，以及命名报告和已保存的数值证据。
C3 当前函数源身份与已接受执行版本一致。四个实际边界的 stock / turnover
未变，份额前向最大差异不超过 `3.33e-16`；后续四项完整损失的 CPU JVP 与
CUDA 目标坐标 VJP 对照通过原容差。全参数及前向回归通过，但其他状态差异仍保留。
它们不证明所有 PFT、所有状态或完整神经训练通过。

初始本地复核通过 6 项碳过程回归及全部 8 项周转条件测试。当前神经合批的
梯度失败不应覆盖 C3 已独立通过的证据，也不能由 C3 的通过反推整个神经训练合格。
源版本、原始证据路径和测试命令见
[daily-evidence.md](../../outputs/reviews/shared-core-20261011/daily/daily-evidence.md)。

两项修复随后在实际 main 导入路径下分别接入并验证：

- C2：`e317fa81b97b8b8c959f2e7dfdfba525bb54dec8`，仅修改
  `jax_orchidee/stomate/carbon_kernels.py` 和
  `tests/unit/test_stomate_carbon_kernels.py`。12 项 NPP 源表达式、负碳库阈值、
  掩码、微小维持呼吸梯度及生产适配器回归通过。
- C3：`2a3e303845154576c6f4b5c0b8155106e5bcc70b`，仅修改同一碳过程源码并新增
  `tests/unit/test_turnover_fraction_conditioning.py`。20 项周转、临界叶龄、
  非活动掩码、微小叶质量及 turnover / gap / kill 适配器回归通过，其中包含
  日尺度已有的全部 8 项条件测试。

接入后的碳过程源码和原碳过程测试文件与日尺度 `120c637` 完全一致；新增测试仅做
格式及等价可调用语法整理，归一化 AST 与原测试一致。NPP 函数 AST 与来源提交一致；
周转函数 SHA256 为已接受的
`f6240cf5ce64a47b9bbe0647b5e5715305e9a9230ea77f2bd9f42981ae35a93c`，
去除新增份额计算后，原物理 stock / rate / mask / 更新顺序语句的 AST 未变。
原测试模块的 Ruff 检查未新增诊断，新增周转测试文件 Ruff 检查通过。

命令、运行环境、源身份及原始日志见
[接入证据](../../outputs/reviews/shared-core-20261011/carbon-integration/integration-evidence.md)
和[机器记录](../../outputs/reviews/shared-core-20261011/carbon-integration/integration-receipt.json)。
本次复用上述已接受的有界 CPU / CUDA 证据，没有重跑历史 GPU 或完整模型任务。

### C4：restart tau回退的单位与配置传递（2026-10-23）

独立修复提交：`4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99`。只包含三份生产文件
`jax_orchidee/stomate/restart_io.py`、`jax_orchidee/driver/restart_state.py`、
`jax_orchidee/coupled.py`（相关docstring）及两份直接回归
`tests/unit/test_stomate_restart_io.py`、`tests/parity/test_driver_1961_restart.py`。
审查报告另行提交，便于研究分支单独同步该修复。

触发条件是restart中长期温度有效，但`tau_longterm`为未初始化sentinel；partial helper
也允许tau字段缺失，而完整生产入口仍会先拒绝缺字段。原helper把系数3当成3天；
Fortran `stomate_data.f90`697–700和`stomate_io.f90`620–637要求系数乘年天数。
现在helper复用`SeasonTimeScales`的天数默认值（本配置1095天），生产reader传入
已有配置解析器给出的上限。显式天数覆盖保持原单位，自定义系数2.5对应912.5天；
有效tau原样保留，长期温度缺失/all-sentinel时仍按源优先初始化tau=2。

ORCJAX一次相关检查31/31通过，包括真实三文件生产读取/首次season选择；主控针对
新增配置解析依赖补跑原有无`run_def_path`入口测试，1项通过。验证仅涉及状态读取，
没有重跑模型日/年或服务器任务。命令及证据见
[Stage15入口](../../outputs/reviews/main-physics-20261023/restart-tau-fix/README.md)、
[31项检查](../../outputs/reviews/main-physics-20261023/restart-tau-fix/checks/attempt-01.json)及
[默认入口补查](../../outputs/reviews/main-physics-20261023/restart-tau-fix/checks/primary-default-entry.json)。

所核定的319/1961第三阶段输入本来含有效tau=1095，显式旧上限3/新上限1095读取
五组状态及report精确相同；因此不触发此缺陷，原年度NPP失败仍未关闭。
其他触发回退的输入可能改变后续模型结果；不得把旧Teacher、prepared数据或实验
producer改标为修复后版本，也不由这次读取检查宣称全部历史结果不变。

日尺度与全球效率维护线程后续可核对并使用`git cherry-pick -x 4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99`。
同步应同时保留helper默认值、生产配置传参和相关回归，按各分支实际接口解决冲突，
不整体覆盖共享文件或替换正在运行的checkout。核查各自有效配置和输入是否触发回退，
通过相关状态读取回归后，将本节同步状态更新为各分支实际提交与验证结果。
本次只登记修复，未向其他聊天发送消息或自动合并研究分支。

### C5：THERMOSOIL原水文层数与深层潜热容量截断（2026-10-24）

状态：**最小实现、本地回归及主控验收完成；独立修复提交为
`a117cd5afc92ea55c9aeb528eec6f724de780111`**。本轮基线为
`04441e7d70f8e396757d79aa6dd363b4fc4eb214`，不是修复提交。Stage18已验收
S18-THERM-NSLM的源码原因及独立Fortran保存值，详见
[Stage18入口](../../outputs/reviews/main-physics-20261024/core-process-sweep/README.md)。

Fortran `thermosoil.f90::thermosoil_getdiff` 2742–2743要求 `jg>nslm` 的
`pcappa_supp` 清零。JAX `thermosoil_explicit_step` 在humlev已扩到ngrnd后未传
原水文nslm，direct helper默认推断遂使用扩展层数。最小修复只在该调用转发
`nslm=_as_float64(shumdiag_perma).shape[1]`；direct公式、默认值、精度、阈值和
已经正确传参的cold初始化调用均保持原语义。回归必须经过实际wrapper，不能将
Stage18的显式nslm direct comparator通过当作修复后wrapper通过。

生产改动仅 `jax_orchidee/sechiba/thermosoil.py:3171` 一行。新增独立测试
`tests/unit/test_thermosoil_nslm.py` 及小fixture
`tests/unit/fixtures/thermosoil_nslm_fortran.json` 已随独立修复提交；运行只读取该fixture，
不依赖ignored outputs或其诊断脚本。fixture保存4层pcapa/supp的8个独立Fortran值，
由原104值确认自然槽1–13逐层相同后压缩；自然PFT14明确按源自然分支角色映射槽1，
原oracle的crop槽14不作自然PFT14参考。源文件、过程跨度、编译单元及CSV身份均留存。

同一新回归在旧wrapper实测2项失败：自然槽1–13为52/104不符，自然PFT14为4/8，
浅层先行断言通过。只补传参数后新模块9项通过，涵盖相同独立比较、两个含端点相变
边界、暖/冷区间外、长期湿度0、等水热层数及一次真实wrapper容量投影JIT。既有
direct getdiff与wrapper composition窄检查2项通过；JIT只验证所选容量投影和traced
原湿度形状，不升级为生产模块全部输出或完整块编译等价。原容差1e-14/1e-12保留。

三次检查均在ORCJAX、Python -B、禁用pytest cacheprovider及系统TEMP下运行；TEMP
均已清理。cacheprovider禁用后的既有cache_dir配置warning原样保留，没有据此修改
项目配置或重跑。没有新的Fortran编译、模型日/年、服务器或研究执行。实际命令、
旧失败/新通过日志及身份见
[Stage19入口](../../outputs/reviews/main-physics-20261024/thermosoil-nslm-fix/README.md)、
[实现与验证回执](../../outputs/reviews/main-physics-20261024/thermosoil-nslm-fix/implementation-and-validation.json)。

日尺度和全球效率分支的潜在同步范围是这一共享thermal wrapper及其独立组件回归/
小fixture。各维护线程可据上述独立修复提交，按实际差异核对或同步准确
提交，保留各自执行优化和训练接口，检查是否真实经过该wrapper或独立thermal副本；
不得整体覆盖共享文件或原地替换正在运行的checkout。本轮不操作两工作树或向其聊天
发消息，不将旧Teacher/prepared数据改标为修复后生成。

paper的nslm11/ngrnd32及冻结开启只证明静态层数条件；深层自然活动mask、正长期
湿度及含端点相变区间是否在实际轨迹同时出现、年度影响和319 NPP因果均未证。
原319失败及原阈值保留。raw-restart首步旧热容量回退仍是未证接口限制，不并入C5。

主控验收已核对实际新测试/fixture、执行器、旧失败和新通过日志、三个当前受测文件
身份及一行生产差异，复用执行方9+2检查，未重跑数值。原交付回执的待验收状态保留
其交付时点身份；本节和主审查报告记录最终验收及修复SHA。

### C6：TF_DOC实际producer的半小时干沉降时钟（2026-10-25）

状态：**主控已验收，修复提交`27ff14b38aafc7df91ce52b8393642ca08952395`**。`stomate.f90::stomate_main:3380–3388`
向soilcarbon传半小时dt，`stomate_soilcarbon.f90::soilcarbon_leak:1206–1233`只在干项
乘dt。初始化、普通fold及compiled wrapper三处改传`runtime.dt_sechiba/86400`，降水
定义保留。leaf100/cover.4独立预期干项`.0007666666666666667`，旧为`.0368`；湿项、
TF关闭、真实ground/storage两步carry有界通过。compiled参数/窄scan使用明确core seam，
不宣称完整compiled模型数值等价。C6旧3失败/3控制通过；与C9同15项修复后全通过。
实现位于`driver/orchestration.py`，回归`tests/parity/test_ok_leak_producer_contracts.py`。
见[Stage21说明](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/implementation-and-validation.md)。

### C7：DIC聚合的litter/soil率与即时maintenance/root质量（2026-10-25）

状态：**主控已验收，修复提交`27ff14b38aafc7df91ce52b8393642ca08952395`**。litter源2667–2670、2740–2750、
2778–2781及soilcarbon1671–1704返回日率，stomate3314–3315、3409–3410先乘半小时dt
再3442–3456聚合。resp297–298的即时maintenance已含dt，stomate3258的floodroot也是
步质量；日累计是另一消费者。`stomate_ok_leak_explicit`只将四个litter/soil聚合参数乘
`dt_days`，保留核心rate对象，maintenance/root不再乘。新独立fixture混合三出口为
`[.03529411764705882,.09895833333333333,.7058823529411764]`，另有真实混合组件和no-flow。
两项旧混合失败已修复，回归/小fixture在`tests/parity/test_ok_leak_dic_units.py`及同目录
fixtures；既有manual-chain仅同步该边界期望。未证年度效应。

### C8：bulk_dens两次源赋值、共享owner与prepared marker（2026-10-25）

状态：**主控已验收，修复提交`27ff14b38aafc7df91ce52b8393642ca08952395`**。litter1971–1973及soilcarbon1473–1478
各执行一次`<500 ×1000`。raw static/map/default保留raw，controls前执行litter赋值；
generic/same-step合并采用prepared值，adapter跳过已完成的litter后执行soil赋值，compiled
以tracer-safe`jnp.where`尊重marker，防止tiny值第三次转换。新增`soil_properties.py`
含精确Fortran provenance；`peat_bulk_density`分离，不在低层adsorption偷偷转换。
同原10项旧8失败/2通过→同10通过：4真实adapter数值、2controls实参，另2仅为新增marker
缺失的KeyError。当前20项含真实producer/controls/adapter、generic/peat及窄scan参数检查，
最后组合全部通过。raw1.65、1650、500、.5及tiny`.0001→.1→100`有界覆盖；tiny仅证
一次调用，未引入通用跨步bulk carry，负值源fatal/nonfinite及跨步回写仍未闭合。

### C9：独立于网格层数的runoff cutoff（2026-10-25）

状态：**主控已验收，修复提交`27ff14b38aafc7df91ce52b8393642ca08952395`**。源`constantes_var.f90:294`默认
`sro_bottom=5`，独立于水文11/深层32。初始化、普通和compiled装配统一水量及core/litter
cutoff；compiled独立static参数，排除重复packet键。已有内存API覆盖3仍可用，不新增
run.def能力，不改11/32形状。producer哨兵水量5层55、旧11层506、覆盖3为14仅证明
参数；真实decomposition消费捕获cutoff，独立标量layer5/6 DOC`.075/.1125`，旧compiled
为`.075/.075`。C9旧9失败，修复后与C6同15项全通过；不是新Fortran oracle或全年验证。

### C10：DOC峰值扩散顺序恢复及负剩余库存局限（2026-10-25）

状态：**主控已验收，独立修复提交`225cb95a6a44334390d1656d85f66fd54d342a19`**。主控已授权按源`stomate_soilcarbon.f90::
soilcarbon_leak:2002–2006`恢复先2005后2006；第二分母读取更新后的左通量。只改该
分母和来源注释，WHERE/循环/非峰值/冻结保留。默认半小时D、正薄层与`[1,3,1]`
给flux`[1.5,12/7]`，末库存`[2.5,-3/14,19/7]`。**负中央库存是源算法局限，未clamp或
改造limiter。** 新独立fixture四算例×eager/窄JIT8项及既有直接检查通过，旧峰值两项
失败真实保留；原simultaneous1.5/1.5期望按源改正。

独立[四文件patch](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/C10-source-order.patch)
和[分组身份](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/C10-group.json)
仅含soilcarbon扩散hunk、旧测试期望hunk、新source-order测试与fixture，不混C7/C8。
原交付只作正/逆向适用性检查；主控已将该四文件单独提交为上述C10修复。

C6–C10最终[组合49项](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/combined-focused.json)
exit0，绑定最后实际文件身份；[交付核验](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/verification.delivery.json)
只作元数据/链接/Git/补丁检查，不重跑数值。fixture前置错误与有效缺陷失败分开，重叠
检查不累加为独立覆盖率。图工具不可用，复用Stage20 Verify generation
`2026-10-24T15:26:49Z`（Python metadata_changed、Fortran excluded），确切源补足，
不宣称新图覆盖。两研究分支待维护线程按最终真实提交及各自接口核对，不整体覆文件、
不替换运行checkout、不重标历史Teacher/prepared。本批无研究工作树或跨聊天操作。
原319失败、C5实际轨迹/年度影响未证、raw-restart旧容量未知及暂停水分起源调查保留；
两个Stage20 candidate与完整SAVE生命周期不并入，科学门槛/预算不改。详见
[Stage21入口](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/README.md)。

主控验收已复核实际代码/测试/fixture、有效旧失败和最终49项日志；14份当前受测
文件身份一致，未重复数值运行。C6–C9共享一个九文件修复提交，C10使用另一个四文件
提交，审查文档另行提交。执行回执中“待主控验收/SHA待定”保留原交付时点身份，
本登记给出最终接受状态与准确提交。同步时按这两组真实差异处理，不能以整文件替换
覆盖研究分支的独立实现，也不能把历史Teacher/prepared改标为修复后产出。

### Stage23：六项共享修复登记（已接受C11–C16）

实施基线为main `c94da5b623da4f6da377520ce08436933b6ed1b7`，不是修复SHA；
Stage22历史数值producer保留`1229de7b41711a5177d572feef5cc95f4cbc2a32`。
下表不更改C4–C10的接受状态、准确提交或研究同步记录。详见
[Stage23实现/证据/补丁入口](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-fixes/README.md)和
[main审查Stage23](main_physics_review.md)。

| 修复编号 / 原发现ID | 已接受改动 | 依赖与接受边界 |
| --- | --- | --- |
| C11 / S22-SNOW-LF | 共享derived333700、8显式雪+1bucket默认及两个harness参数关系 | hydrol+enerbil共享常数，原雪test预期；独立土壤lhf333600保留。12新增/8既有通过；不重建或重标旧333600 oracle |
| C12 / S22-TILE6-THRESHOLDS | 保留mineral阈值，top/clip/部分residual仅4/5，deep/coef/liquid保留4/5/6 | hydrol各阶段hunks及旧routing fixture；BCD共用一个自含回归文件，不全局取消tile6 peat行为 |
| C13 / S22-FROZEN-DUMMY-ORDER | freeze分支第一次赋值后无条件第二次重构，真实caller传switch | hydrol dummy helper/caller；固定profile/retry/restore有界证据，不关闭DUMMY-REFRESH候选 |
| C14 / S22-TOPMODEL-PREFIX | 地表连续暖层前缀及numlayers边界 | hydrol diagnostic API；源>6/undef保留，不开放生产TOPMODEL或关闭潮汐/carry候选 |
| C15 / S22-CONDVEG-Z0 | wrapper/main/init和cold绑定Z0_BARE，两compiled cache隔离 | condveg+driver E hunks；13新增/21既有通过。物理wrapper与AST实参/cache stub证据分列，不称完整compiled模型验收 |
| C16 / S22-AE-STATE | 真实post-surface ae进入diag，shared writer carry与finalize相同diag阶段 | hydrol+driver F hunks；9项真实组件→writer→下次split通过；grid总vevapnu残差不重复加 |

A12失败、BCD17失败/13通过、E2失败/4通过及2 cache失败、F7失败/2通过的before均在
共同生产barrier前保存，无error/skip；之后原BCD30项通过并加4条numlayers边界。
最终组合以[focused回执](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-fixes/combined-focused.json)
记录的72项通过为准（exit0、无error/skip，TEMP清理）；重叠测试数不当作独立oracle或
完整覆盖。旧雪预期按源常数改；旧tile4综合routing
夹具改为source真正有阶段阈值差异的tile6，保留原数值断言及独立tile4检查，不放宽容差。

主控已验看全部生产/test改动和有效before/after，复用数值检查、没有重复积分或组件运行。
16份交付文件身份均匹配，最终三份生产文件与组合启动时SHA一致。代码分为四组提交：

| 编号 | 精确修复SHA |
| --- | --- |
| C11 | `dd6d6335ba5011fcd86285984136a6799f6916ac` |
| C12–C14 | `d48b923e891946d3c9db77ba9b684666cb511b1d` |
| C15 | `bbcdbf7a661cb11f5295883e54c3b1ea9d596d39` |
| C16 | `2f36dc7f7945d52387a2848aa124c043118ed45f` |

交付patch保留A/B/C/D/E/F问题hunk分组；B/C/D共用回归随C12–C14同一提交。
hydrol、driver的同文件hunk不能以整文件覆盖研究分支，不自动同步daily/global。
运行checkout、Teacher/prepared、旧Fortran输出及
原333600 harness结果继续保留各自producer身份。

三个S22候选继续开放；319年度失败、暂停water-origin、C10源负库存、C8 tiny跨步/
negative fatal/nonfinite、两项Stage20 peat候选及raw-restart旧容量未知均保留。
无年度改善/因果、全配置/完整compiled模型或新生产准入结论；无新Fortran、模型日年、
服务器/GPU/训练/付费、研究工作树、自动化或跨聊天操作。执行方交付后，由主控完成上述
接受与提交；交付时的pending/SHA null回执保留历史状态，以本登记和主控回执为后续接受依据。

### I1：四行改动包含回调接口变化

I1 来自 `2eb63842aefc5a685451e502815039bdc24b0ae1`，把日通量标签接入已有的
compiled block，并把捕捉开关加入缓存 key。底层标签类型与单日捕捉能力 main
已经存在。它是可选的 Teacher 导出能力，不需要搬入神经网络实现。

但日尺度版本无条件把 `training_output_projector` 从两个位置参数改为三个，
即使通量捕捉关闭也传入第三个 `None`。日尺度保留的
`scripts/dev/probe_ok_leak_outer_block_reentry.py::target_projector` 仍只有两个
参数；签名检查可确认三参数调用不兼容。这是日尺度旧诊断调用点的具体缺口，
该脚本不在 main，不能表述为 main 现有生产调用已经失败。

集成时需显式保持旧回调合同或采用可选 / 版本化的新接口，并检查相关调用点；
不要用捕获任意 `TypeError` 的方式掩盖回调内部错误。I1 与 R1 修改同一段日块
组织，应一起设计接口，但可以分别验证和提交。

### R1、R2：已有真实前向证据，尚不是完整生产入口

全球分支保持物理过程方程，把原日 scan 提成函数并在外层批量调用。
当前适配器只接受每点 `npts=1`、48 半小时 / 日、连续 7 日和兼容的静态配置，
拒绝 peat / TOPMODEL，且模块明确未提供生产 CLI 和持久可执行缓存。

单元测试使用模拟标量函数检查组装、拒绝条件与 lane 独立性；研究线另有真实
GPU 前向证据，不能把两者混为一谈：

- 固定 `bd9e4c04...` 的 job15662711：B512 / B1024 两阶段各 1027 门槛通过，
  保留 760 个原生输出叶和 1024 次 FULL64 对照。
- 固定 `8aa2e05a...` 的 job15671732：B1024 对照阶段通过 2051 门槛及 2048 次
  FULL64 对照；B2048 在第二连续块执行时 OOM，没有完整验收结果。

以上为 64 个真实点重复扩展的容量负载，不能替代真实 669 异质点、独立集合成员、
长期状态 / restart 和生产 IO 的对应验收。原证据分别见全球工作树的
[B1024 结果](../../runtime/worktrees/global-throughput/outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/CURRENT_RESULT.md)
和 [B2048 结果](../../runtime/worktrees/global-throughput/outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/CURRENT_RESULT.md)。

未来接入 main 应按拟支持的运行合同复用已有前向状态、输出、掩码、原生 dtype
证据并补实际缺口。全球当前研究正向吞吐，完整梯度验收不是本轮新加的接入
前置条件；若声明批量可微性，再按其明确用途补相应证据。

## 数据、执行版本与同步

固定 v6 Teacher 的生产报告记录 producer
`19087b171564f44b49eb96fa1217169deca5d57a`，其 `carbon_kernels.py` 与初始 main
核查基线 `7cae3e0` 相同，早于本次 C2、C3 接入。报告里的生产 / 哈希验收和 CPU smoke，不等于 main README
中仍待完成的 669 点长期 Fortran 等价验收；两种验收的目标不同。

Teacher 生成版本、prepared 对象构建版本、当前执行版本可以不同，必须分别保留
身份和消费者相容性。C2 保持源前向商，C3 允许上述有界浮点差异；这些事实本身
既不要求重新生成 Teacher，也不证明所有 prepared 包可无条件混用。本轮没有
核定当前活动 prepared 包的完整生产清单。既有 v6 / prepared 复用工作单、类和
模块序列化身份继续保留；不要改写历史 producer 标签。

同步状态：C1 在三条开发线上已有同等实现；C2、C3 已进入 main，日尺度原本已有
对应实现，不需盲目重复应用。全球效率线 `52326bd` 尚缺 C2、C3，由其维护线程在
完整阶段边界依次同步 `e317fa8`、`2a3e303`，并核查实际依赖的前向行为。正在执行、
排队或用于历史对照的代码和可执行文件继续保留原版本，不在原地替换。

主分支的这轮已知通用修复收尾完成，后续可开始按 Fortran 来源和分支覆盖账本
检查已有物理过程代码。I1 / R1 的兼容接口及 R2 的全球运行支持范围仍由相应
工作线推进，尚未自动纳入 main；本轮不能称为三个分支全部文件已经一致。

## 本轮验证与局限

本轮使用 ORCJAX / Python 3.11.15 / JAX 0.10.1 CPU。DIFFUCO 原 main 的 14 项
失败是有意复现；候选及接入 main 后均为 14 项通过，日尺度同步回归也为 14 项通过。
碳过程初始复核为 6 + 8 项通过；实际 main 接入后，C2 的 12 项、C3 的 20 项通过。
独立解析探针还核查了导数阈值、旧回调签名和已接受周转源身份。
保存证据链接位于上述分项报告和接入记录；
这些 `outputs/` 文件按项目规范不入 Git，本文保留可审查的结论、版本与复现入口。

知识图仅覆盖 main，generation `2026-10-10T08:46:50Z`；目标文件元数据已变更，
研究新增文件不在 main 图中，因此关键结论使用直接 Git 差异和源码回退核查。
本轮不是全模型 / 全分支科学验收，没有重新执行历史 GPU 任务、训练或生成数据。
本页随共同修复接入更新，不追加研究实验时间线。
