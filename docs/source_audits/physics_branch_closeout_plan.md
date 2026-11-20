# 物理计算与条件分支验证收尾

2026-11-18 启动。当前范围以 [PFT14等价交付合同](pft14_equivalence_delivery_contract.md)
和 [完成顺序及可维护性要求](pft14_completion_execution_plan.md) 为准。
本计划承接已接受的参数／状态传递工作，进入模块语义验证，不宣布整模型等价。

2026-11-20，第六包经主控接受，五项修补登记C73–C77，生产／测试／CI提交`f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b`。累计**77项修正／支持补齐、39个相关生产提交**；75项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏，不是77项物理公式错误。本轮五项均有复查前`7cae3e02adad23674767a590fdf4d27f239ce065`只读源码证据，首次引入日期未断言，未恢复运行历史代码。

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-acceptance.json)核对46份产品、交付与独审封存身份，并复核配对原源、实际写回及历史源码。保存的隔离CI按精确节点合并为454通过、9项科学strict xfail、10项外部身份skip、0失败／错误；这不是一次不间断全通过运行。首次漏复制已跟踪restart schema的9个setup错误保留，修正复制后仅重测受影响的10节点，未重测此前通过项。配置内Ruff通过，生产lint 62前／62后无新增；hosted CI未运行。主控解析JUnit并复用owner／独审验证，未再积分；12份JSON仅Git换行规范化、语义相同。[发布记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-publication.json)保留最终版本。

保存的嵌套JIT整返回差异现有有界递推取整与表面通量抵消证据；13/14 soilflx_pft及其别名仍严格失败，不声明唯一HLO／机器指令原因。两个新增合成相等浓度DOC元组捕获到入口约1e-17取整变化，实测入口重放能复现严格相等分支变化及4／5个失败坐标；没有定位最早上游算术，也不外推旧年度轨迹。新8项JIT完整系数通量门／60坐标仍未归因。旧HYDROL／热／雪／低光／DOC／NPP／grain严格门各守原范围，不同配置DGVM诊断不计新修复。

52父过程的整体路线等级仍为source_mapped／open；4,059词法候选、565源审查组和187跨模块／外部调用组不等于数值覆盖或缺陷数。新有限证据包括14生长元组2611比较、扩散实际arm、非零冻融carry、冠层／capacity前驱、forcing／静态输入／modelout／routing和6份科学restart文件往返（1402检查，含147独立字段）。真实IOIPSL／native producer、maintained完整调度、自然历史、私有peat／PBL、原未定义合同与完整源分母仍有缺证。C64限npts1／nstm6；年度DOC真实边界／残差／异养呼吸均为null，原mass warning与负中间量保留。

下一包优先实质审阅[565源组和187调用组](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/denominator/remaining.json)：解析真实callee与目标域，复用准确旧证据，明确排除依据，对确有缺口补独立原参考和实际compiled／写回。并行归因新8项完整系数通量失败，不重开本包已闭合的嵌套与DOC有界机制。停止标准仍见[阶段计划](physics_branch_closeout_plan.md)；本地可做项完成后再汇总真正外部缺项，不以扩大扫描清单替代验证。整模型综合验证尚未启动，`numerical_acceptance=false`；参数／状态90＋1＋1／92及93／126／181口径保持，研究树同步暂停。

## 授权、基线与接力

用户直接授权：“所以在全面综合审查之前。就只剩物理计算与条件分支验证了是吧？
你可以开启下一波的跟sol的互相接力。把这部分完成。”

- 主控聊天：梳理主分支检查工作，`01a128e9-fda8-7781-ad9b-3ab29323909d`，host `local`。
- 执行聊天：用Sol执行主分支检查工作，`01a1298c-60d9-7020-9086-abffe56c9ccb`，host `local`。
- 授权轮次：`01a1ef4a-34ae-7f21-affa-8c9b629fec62`；人类消息：
  `01a1ef4a-34fe-7063-9db9-a5aa86b2dec8`。这是新的直接用户授权，覆盖本阶段执行完成后的回传。
- 阶段启动时已接受 main 基线：`000d5307ede59df6b8f51d015d0ad5cf17f79845`；当时生产／测试／CI提交：
  `aa14d04210ab371aae6e959da193561f37f55d0e`。本计划的文档提交不改变生产基线。
- 启动时累计61／34；2026-11-20接受至C77后，当前正式为**77项修正／支持补齐、39个相关生产提交**。
  75项有复查前错误或支持缺口证据，C42／C57为复查期间已接受代码的遗漏。不是77项物理公式错误。
  最新已接受生产／测试／CI提交`f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b`，原启动基线与历史封包保留。

Sol完成一个连贯包、停止修改并封存后，只给主控发送一次消息，包含产物路径、
实际验证、待验收事项和剩余工作，然后停止。主控验收实际产物、处理科学判断、
发布已接受修改，并在本阶段范围内派下一包。执行期间主控不做平行重复调查。
不启用定时自动化或重复轮询；需要补齐的非紧急事项一次合并交回原执行者。
达到本阶段完成标准后双方停止接力，不自动开启整模型综合验证。

## 目标域

目标是既定PFT14论文工作流中，独立单点位置、forcing值／来源、连续参数和
支持状态变化时可达的物理计算与条件分支；不能缩成319点或默认参数下已走过的路径。
按原Fortran条件判断潮汐、盐度、水分、泥炭／冻结、碳循环等分支可达性，
不因某份旧报告标为inactive就直接排除。

需要改变非论文离散开关的更宽物理变体、非PFT14角色和多landpoint routing，
依交付合同单独登记源依据及支持边界。不能为结题把域内未实现分支改称域外。
一般原生IOIPSL、多点scatter、外部PBL producer、历史001身份和整模型科学门
分别保留，不把其缺证当作已经发现的逻辑错误，也不阻塞无关的本地同输入检查。

沿用当前过程／driver／runner架构与显式JAX状态设计。检查公式、常量、单位、
索引和取整、IF／WHERE／SELECT CASE／mask、顺序更新、重置与累加，以及实际
消费者和写回。必要JAX重构不要求逐行复刻Fortran，但必须保持这些语义。

## 有限工作清单

第一包建立当前覆盖总账，同时推进前两组实际验证；后续按总账缺口分成连贯包，
不按发现一个小问题就停止，也不承诺固定轮数。以下是工作分组，不是测试数量或轮数：

| 分组 | 范围和交付重点 |
| --- | --- |
| PB-01 冠层和光合 | DIFFUCO、CONDVEG、QSAT；辐射／温度／湿度、盐度／潮汐、水分限制与PFT mask；实际冠层到能量调用 |
| PB-02 能量交换 | ENERBIL、融雪／蒸发和局部coupling；单位、符号、能量／水量转换、实际生产组件及写回 |
| PB-03 水文、土壤热和积雪 | HYDROL、THERMOSOIL及snow；复用已接受土壤参数、热组件与已知浮点证据，只补剩余源条件和真实消费者证据 |
| PB-04 植被和碳过程 | PFT14相关STOMATE：生长／分配／周转、呼吸、凋落物、土壤碳／DOC等；顺序、状态门和日／年相位的组件级合同 |
| PB-05 跨组收口 | 补齐上述组间实际输入／输出／写回与相关driver／diagnostic数值合同，复核全表、独立终审和下一阶段准入缺项 |

复用2026-11-13 `equivalence-gap-reconciliation` 的稳定owner／branch ID、
来源和既有证据；创建当前覆盖层，不修改封存历史包。
52过程条目、308去重owner span、12扩展记录、26428 branch arms，以及旧
4217数值arms／824关联Oracle只是各自口径的历史清点，不能相加，不能宣称
它们已是完整目标分母，也不能把3393缺少关联pass的arms称为3393个bug。
手工核对扫描盲区和实际调用；同一案例可覆盖多arm，但每个被覆盖arm必须有明确映射。

每行至少包含：稳定源ID与文件／子程序／行段，源条件及域内可达依据，JAX owner／
真实caller，输入／输出／writeback相位，已有证据及其源码／输入身份，当前证据等级，
结果（pass／已证浮点敏感／未解差异／缺参考／域外），未完成项和下一最小动作。
分母变动必须记录新增、去重或源域修订理由；不因当前pass数改变目标分母。

证据等级保持 `inventoried`、`source_mapped`、`jax_regression`、
`fortran_oracle`、`production_verified` 的区别。旧“covered／verified”标签
只能在查清实际合同后映射；已有有界接受不因换总账而作废，也不自动升级。

## 验证与修复规则

1. 先核对原源条件、有效配置、完整同输入元组和更新顺序，再判断是否存在逻辑差异。
   为阈值两侧、零／非零和mask组合选合法案例；合成输入注明适用域，不冒充自然轨迹。
2. 优先复用身份明确且源码未受影响的原Fortran参考。缺项允许本地CPU小型原过程
   micro-driver或完整组件参考；记录原源／提取段、编译选项、输入和边界。
   JAX自身生成的预期值只算回归，不能充当Fortran真值；不为接通参考改写原过程语义。
3. 在适当的有限边界执行真实生产组件、实际adapter、eager及真实窄JIT路径，
   核对返回状态与下一消费者实参。替换外围过程可隔离接口，但必须明确哪些物理
   真正执行；delegate／mock或AST只能证明相应接口，不能认证被替换物理。
4. 确认缺陷后在同一包完成最小共享修复、旧失败／新通过、邻近回归和来源记录。
   保留架构与相位区别，不加点号特例、静默默认或全状态无条件复制。
   新候选C号待主控接受后正式计账；记录复查前／复查期间／本包新遗漏的真实年代、
   准确提交与将来集中同步依赖，不能按调用点重复计数。
5. 不修改原科学容差、epsilon、阈值或输入期望以通过检查。自然浮点扰动改变分支
   与翻译错误分开；复用已接受NPP／DOC／snow因果证据，不无理由重开追因。
   原DOC严格状态门、积雪grain严格失败和尚缺的真实碳通量继续保留。
   新矛盾才触发有界复核；不要求bitwise一致或无限追究唯一机器指令原因。
6. 以当前源码绑定复用已有验证，只对相关修改运行必要检查。新增轻量回归纳入适当CI；
   本地检查与hosted CI、独立参考与接口测试分别说明，不机械累加测试数量。

## 执行边界与所有权

工作仅在 `E:/ORCHIDEE-MAN-JAX` 的main。Sol沿用当前模型／努力设置，负责执行、
集成、检查与交付；允许在Sol／Terra／Luna家族内直接组织有意义的并行原生子智能体
及独立终审，明确文件所有权，子任务不再向下设协调链。所有执行者共享工作区，
不得覆盖或回退他人修改。共享orchestration及跨模块接口由Sol负责集成。

本阶段允许必要的本地CPU/x64原Fortran微参考、组件计算、窄JIT与有限接口调用。
前一参数／状态包的“零物理运行”限制只属于前一包，不阻止本阶段已授权验证。
不自动运行整模型step／day／year积分或669矩阵；不执行SSH／服务器／GPU／训练／
新付费任务，不重建环境，不操作daily／global研究树。无法本地隔离的参考应先完成
可行准备，给出准确缺项和最小命令／资源；只有真正超范围的行动才交主控处理。

Sol不stage／commit／push。主控在实际产物验收后按项目既有授权提交、普通推送并
核对远端SHA与工作区。已接受封包保持原样；新输出写入 `outputs/reviews/`。
首包目录为
`outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/`。

## 完成标准与停止边界

当前首包状态（2026-11-19）：[主控已有限接受冠层／能量及CI修订](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-acceptance.json)，
测试／参考／CI提交`ebd1141d5b214ab0b874d124eab2466eef8e0a77`。[当前覆盖表](physics_branch_coverage.md)保留新增原Fortran参考、eager／窄JIT和有限真实消费者证据。
未确证生产缺陷，正式61／34不变。低光cim严格失败及极低正LAI原Fortran陷阱显式保留；
首包不是PB-01／PB-02或整个阶段全面闭合。下一优先按机器表进入PB-03当前源／案例关联及首个未代表条件，
不重跑已有五土壤参数案例；后续PB-04与跨组缺项继续按具名最小动作接续。

本轮执行包为 `outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/`。
Sol并行组织HYDROL土壤／潮汐／冻结／泥炭与THERMOSOIL／snow两组，先把C13／C17／C18／C50–C55、
18个初始化案例及8个完整热组件案例按原身份关联当前源条件，再执行缺失合法条件的一组连贯组件检查。
HYDROL明确区分私有peat_hydro和同名参数；thermal／snow区分合成合法、源可消费和自然前驱可达，
不通过重复长跑碰触分支。Sol整合真实humrel／vegstress／wat_flux与热水分等下一消费者和写回，
为本组13个具名过程条目形成精确处置，不以helper通过替代生产组件证明。
该包可修复确证源级问题并补相关回归；保留原科学阈值、已证浮点结论及所有未解门。
PB-01／PB-02残余缺项继续在总账中，不因转入PB-03被删除。仍无整模型／远端任务授权，
每包封存回传、主控验收后再派下一包。

本轮新增完整同输入热参考中，恒温暖态的两类 JIT 热通量输出出现原严格门残差；
相邻两次雪组件的 JIT 还保留旧 grain 差异并出现 groundflux 残差。它们按实际失败坐标
保留独立 strict xfail，其余坐标正常断言；没有以改输入、expected 或容差获得通过，
也没有把未确证因果写成已接受浮点结论。HYDROL 私有 SAVE 泥炭开关与全局同名参数
的有效 producer、合成雪前驱与自然气候历史的区别仍是具名限制。当前机器表保留
52 条及七种历史稳定 ID，13 条 PB-03 处置不等于整 route 或全 arm 通过。
主控要求本轮先完成[生产集成与配置修订](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/README.md)，
原候选包及其科学证据保持原样。修订接通潮汐双文本到真实 driver／precall／编译消费者，
绑定有效 `CHECK_CWRR`，完整 owner 对缺少守恒／ET 语义的 true 明确拒绝，保留低层诊断 API。
土柱累加顺序候选按主控科学范围决定撤回，恢复已接受求和实现；原参考、输入、容差不变，
只重测受影响的现有严格门与真实接口。两个源级原因的年代和旧参考常量影响按明确输入限定。
2026-11-19修订获[主控接受](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/controller-acceptance.json)，登记C62／C63；共同生产提交`7b35196462fbfd53f531fd1dc751699092f80104`。主控核对22份产品并复用已有独审与验证，没有再积分。严格失败和完整true未支持保持明确。
当前`PB03-STRICT-THERMAL`及同组分支收口[已接受包](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase03-hydrol-thermal-closeout/README.md)完成有界调查与必要修补，见[主控验收](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase03-hydrol-thermal-closeout/controller-acceptance.json)。两个保存同输入原输出逐位不变；相同物性操作数的窄JIT在底层递推出现1ULP差异，已观测系数经原表面式传播到通量，支持有界内部浮点归因，严格门仍保留。独立雪groundflux链未混入此结论，水文首差mcl阶段／5个蒸发限值仍具名未解。

同包补原冻结阈值→实际液态水消费者、原最终热钳制→真实writeback，并形成13条紧凑源条件覆盖层。私有peat有效producer受原声明USE冲突和历史build身份缺项限制，不能按默认false排出目标域。冷启动ZWT_FORCE向量压缩为首值已接受为C64，有原源及复查前代码依据、旧反例和19项隔离输入／消费者通过（无原Fortran、无skip），提交`e275a33fd5864957e9b241138359c1f6001a5756`。源等价仅npts1/nstm6，等长轴／一般多点和原IOIPSL词法未获证。原NPP／DOC／snow grain结论、所有严格坐标、其他PB组及科学／外部限制继续保留，不追求bitwise或改求和顺序接近参考，不自动启动整模型验证。

[第四包与集成修订](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-repair/README.md)于2026-11-20获主控接受，登记C65／C66，共同提交`d2d48e6d5007079493cf91cca21c978eb79fa24a`。

[主控验收](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-acceptance.json)复用原包修前2失败／修后隔离13通过＋1原源身份门skip及本地源门，并核对[集成修订](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-repair/README.md)：NSLM原compiled回归的输入、断言和容差保持，新增真实内层module JIT／外层checkified窄owner的合法材料字段及非法拒绝，两环境各3通过、无skip。11份当前及冻结产品、原／新封存身份匹配；主控本次未重跑数值。原源码与科学封包保留，Git对两份新JSON仅规范换行，解析值相同。发布见[记录](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-publication.json)。

C65改变纯explicit热helper的编译合同：checkify后JIT，返回Error／result并在消费前check_error；module／scan／block与维护测试成组接入。C66仅接受有任一整字段缺失、因此九系数整组覆盖的公共初始化输入，native partial restart及完整driver可达尚未证明，保留组的混合marker仍拒绝。旧C64源等价仍限npts=1、nstm=6。原HYDROL100 eager＋93 JIT、固定热／雪21和不同输入handoff2严格次数不变。新增完整返回的嵌套JIT比较另有13/14项失败、最大绝对差5.34106093e-12，成因未定；窄接口通过不等于整状态等价。

水文normal／ALT残差与独立雪groundflux现有完整同输入、内部首差及传播链证据，接受有界浮点归因，不改物理公式、求和或科学容差。dry／joint热、NPP／DOC／grain的旧结论保持各自范围。新嵌套JIT完整返回差异不能自动套用上述归因。私有peat历史build、自然初始雪包／第二energy容量及native partial restart等缺证仍在；PB-03、完整源分母及整个物理阶段尚未完成，numerical_acceptance=false。

2026-11-20，主控接受第五包六项独立修正／支持补齐，登记C67–C72，生产／测试／CI提交`b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12`。累计**72项修正／支持补齐、38个相关生产提交**；70项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏，不是72项物理公式错误。本轮六项均经只读`git show`确认存在于复查前`7cae3e02adad23674767a590fdf4d27f239ce065`；首次引入日期未作断言，未恢复或运行旧模型。

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-acceptance.json)核对30份当前产品及原交付／独审封存身份，复用owner与独审证据。最终隔离六文件回归260通过、4个外部原源／历史资产身份门跳过；实际边界13通过、来源门2通过，相邻62通过分别保留，不相加为科学覆盖。新11份Python lint通过，生产旧诊断64前／64后未增加；hosted CI未执行。主控只解析回执、读配对源码及实际caller，没有重复积分或数值套件。5份新JSON的Git规范化仅为换行，解析值相同；发布状态见[记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-publication.json)。

本包接受有限组件、源条件与真实写回／下一实参；不宣布PB各组、完整源分母或物理阶段完成。保留原HYDROL／热／雪／低光／DOC严格失败及已接受浮点结论的原范围。嵌套JIT完整返回13/14项失败、最大绝对差5.34106093e-12仍未归因。fixedsnow原未定义值、私有peat／外部PBL、native partial、自然联合／雪能量前驱及DOC原始通量／floor观测等缺证继续具名保留；年度DOC真实残差与异养呼吸仍未知，不由端点推算。C64仍限npts1／nstm6。

下一包按[具名剩余动作](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/remaining.json)并行核对目标源条件分母、生长／扩散等剩余算术分支、嵌套JIT整返回差异和PB-05跨组数值接口；可在现有本地组件边界内补实际前驱，不能把合成元组称为自然轨迹。复用已接受参考，不重开已闭合浮点调查。整模型综合验证尚未启动，`numerical_acceptance=false`。参数／状态90＋1＋1／92及93／126／181口径保持；研究分支同步继续暂停。

第六包派发时要求先给每个目标条件明确归属、源定义、证据等级与最小待办，再并行补本地可完成的算术／扩散分支、编译差异和跨组数值桥；已关闭的有限writer动作不重复派发。外部身份或原定义缺失需有界证据与明确待决项，不能靠新增合成运行假称闭合。共享生产编辑由Sol主owner集成，必要原源反例和实际编译入口回归齐全后再交主控；完成后只回传一次并停止。

最终仍按下列标准判断阶段完成，不能把分包接受或差异登记称为全部验证。

- 所有目标模块及源条件已在有限当前总账中有明确处置，没有未归属的域内分支，
  没有靠扫描无结果或默认案例未触发而遗漏的条件。
- 每个域内条目具备相应原源、独立参考、当前生产caller／compiled组件／writeback证据，
  允许按合同复用未受修改影响的已接受证据；源级和同输入逻辑缺陷均完成修复与回归。
- 未解同输入差异、域内缺参考和生产边缺证不得写成完成；必要外部阻塞要明确标为部分
  交付并保留下一步，不能将“全部登记”称作“全部验证”。
- 已接受的浮点敏感性可作为有界语义结论保留，但其严格数值失败仍独立为失败。
  本阶段即使完成语义检查，也不改变整模型 `numerical_acceptance=false`。
- 独立终审检查当前证据与源覆盖、既有修复是否保持可维护性；主控发布本阶段验收，
  列明还需综合验证的代表域、当前严格失败与外部限制后停止接力。

最终不以修复数量、运行时长或“跑通”作为完成标准。整模型冷启动／跨日跨年／
重启／代表点和最终论文矩阵属于下一阶段，需后续安排。
