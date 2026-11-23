# 当前物理计算与条件分支覆盖

2026-11-23，第十四包已验收，登记 C87–C89。新证据涉及 18 个源码组中的 424 个原成员及 15 个调用组，其中 15 个源码组在继承余项内。本轮仍余 90 个源码组、33 个调用组，后者含 12 个外部缺口。[主控验收](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase14-water-heat-carbon-receivers/controller-acceptance.json) · [具名余项](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase14-water-heat-carbon-receivers/open-groups.json)

三套土壤碳输入在两种执行方式下共 100,440 个标量对照通过，四处径流导出或扣减选中库存。原排水导出和扣减使用不同层厚，样例的量比约为 1.5，六条原质量诊断保留。热写回到下一水文冻结和液态水计算共 24 字段、924 标量通过。排水受库存限制的分支、自然雪热前驱及年度影响仍待证。[土壤碳](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase14-water-heat-carbon-receivers/doc-inventory/README.md) · [水热](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase14-water-heat-carbon-receivers/water-heat/README.md)

2026-11-23，第十三包已验收，累计 **86 项修正或支持补齐、45 个生产提交**。C86 覆盖独立适配器和组合入口两条 LPJ 输出路径。单点实测覆盖后的 GPP 和两个历史输出均为 3.75，固定 PFT14 关闭动态覆盖。[主控验收](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase13-vegetation-carbon-receivers/controller-acceptance.json) · [输出复验](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase13-vegetation-carbon-receivers/controller-repair/README.md)

本包核对 17 个源码组的 629 个原成员。凋落物警告及 ALT 全真掩码在已声明范围内关闭，剩余 90 个源码组、33 个调用组，后者含 12 处外部缺口。原周转与 JAX 在 108 项同输入比较中通过，直接核对最终叶面积和 Vmax 的 90 项以及实际周转返回进入这两个过程的 180 项分别通过。土壤碳仍缺库存成为限制条件时的一致水文输入，全年 DOC 收支、异养呼吸、NPP 和 GPP 影响待验证。[叶片对照](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase13-vegetation-carbon-receivers/lai-vmax/README.md) · [具名余项](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase13-vegetation-carbon-receivers/open-groups.json)

2026-11-22，第十二包已验收，累计 **85 项修正或支持补齐、44 个生产提交**。根系公式、入渗轴及两个通用耦合入口有组件级对照。余下 92 个源码组和 33 个调用组，其中 12 处调用缺外部证据。[主控验收](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase12-root-water-and-vegetation-carbon/controller-acceptance.json) · [具名余项](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase12-root-water-and-vegetation-carbon/open-groups.json)

2026-11-20，当时正式累计**77项修正／支持补齐、39个相关生产提交**，含C62–C77。冠层／能量[首包与CI修订已获主控有限接受](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-acceptance.json)；没有宣布整个阶段或PB-01／PB-02完成。测试／参考／CI提交为`ebd1141d5b214ab0b874d124eab2466eef8e0a77`，当时不改变生产模型或61／34计数。范围与门槛遵循[当前计划](physics_branch_closeout_plan.md)和[PFT14交付合同](pft14_equivalence_delivery_contract.md)：独立单点、forcing值／来源、连续参数和支持状态下的源条件，不能缩成默认点或旧319轨迹。

执行资产以[CI交付修订](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-ci-repair/README.md)绑定：缺原源的隔离checkout仍执行全部数值／声明输入检查，仅两项source/span门跳过；实际本地uv三文件结果110 passed／2 source skipped／2 strict xfailed。有源静态门另2通过，hosted Ubuntu／uv sync未执行。主控核对实际修订与9份执行资产并复用检查；JSON夹具经Git换行规范化后值一致，当前管理文档另随验收更新。原科学证据和覆盖等级不升级，后续PB-03进度见下文。

| 工作组 | 当前可复用与新增证据 | 尚需的最小动作 |
| --- | --- | --- |
| PB-01 冠层／光合 | 复用 C53／C54 和旧 10 例；新增 DIFFUCO 20、盐潮 8、CONDVEG 8 例，当前 eager／组件 JIT；真实 water100 到能量消费者 | 低光 cim 严格失败保留，已有同系数浮点放大证据；极低正 LAI 原参考 trap 的受支持前驱／参考合同未证；根通气及部分 albedo 分支缺参考 |
| PB-02 能量／QSAT | 原完整能量过程与 QSAT 的 5×8 点、24 节点；420 字段／运行检查全部通过，实际 downstream／carry／量率边界单列 | 独立 beta tuple 不证明自然联合可达；tiny-Milly false 与部分雪消融 conjunct 需合法 tuple 或不变量证明；外部 PBL 身份单列 |
| PB-03 水文／土壤热／雪 | [第四包已接受证据](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/README.md)完成normal／ALT与独立雪groundflux有界归因；补合法root、精确canopy容量与入渗耗尽真实消费者；热false mask源不变量及雪carry单列；C65 compiled freeze abort和C66整组重算顺序已接受 | 原strict坐标全部保留；私有peat历史build、自然初始雪包／第二energy容量、native partial restart及其余源条件仍未证；C64限npts1/nstm6；13条处置不等于全arm闭合 |
| PB-04 植被／碳 | 原 daily／season／phenology／allocation／NPP／gap／turnover／litter／soilcarbon Oracle families，已有日历和状态工作保留 | 具体 owner 按日相位／增长周转／凋落物土壤碳分派，补当前完整同输入和顺序写回关联；DOC 严格状态及真实 flux 缺证仍在；OK_PC 完整 driver guard 与局部微参考分开 |
| PB-05 跨组／诊断 | 本包 6 项真实组件消费者与单位写回检查；旧 forcing／静态聚合／module order／modelout／normalized restart 证据保留 | 精确关联剩余 forcing 数值、日表面、科学 restart 消费者、modelout 单位和组间边；多点 routing、一般 native IO、历史 producer 身份独立登记 |

[当前机器总账](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/coverage/current-coverage.json)为 52 个历史过程条目保留具体 owner、源条件、JAX caller、相位、旧证据与下一动作；第二包仅增补 13 个 PB-03 条目的精确处置。7 种旧集合稳定 ID 原样保留；308 去重 span、12 扩展、26428 arms 与旧 4217 数值 arms／824 Oracle 关联不是同一分母，不能相加或称缺关联就是 bug。旧 HYDROL 109-arm 参考没有与这 13 条登记 span 的直接行号交集，不能广播为其逐 arm 通过。宽基础设施 arm 仍有目标域判定缺项，当前表不宣称分母完整。

证据等级 `inventoried`、`source_mapped`、`jax_regression`、`fortran_oracle`、`production_verified` 按实际边界分别记录。旧有界接受不因当前 overlay 默认 inventoried 而作废，也不广播到全文件；站点候选 arm 关联不等于逐 arm pass。物理组件、外围输入组装和严格数值门的结果分别保留，详见[分批验证](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/verification-results.json)。

[具名剩余动作](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/coverage/remaining.json)保留 PB-03 未解门和其余四组缺项。
主控要求先完成[生产集成与配置修订](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/README.md)：
潮汐两份原始文本按 native-run/source-istp 进入真实 driver／编译强迫／precall／owner，
`CHECK_CWRR` 默认 false 与完整 true 拒绝跨缓存绑定；土柱求和顺序候选已撤回至接受基线。
原覆盖表和科学回执不改写历史；修订重测受求和恢复影响的原 19 例并保留每个严格失败坐标，
不重跑已接受土壤参数五例、不重开 NPP／DOC／snow 既有因果。两个生产原因的年代及
C13／C17／C18／C55旧参考影响已有有界核查，未整体撤销原接受。修订现获[主控接受](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/controller-acceptance.json)，两项登记C62／C63，共同提交`7b35196462fbfd53f531fd1dc751699092f80104`；土柱求和候选不计修复。
当前隔离135通过／195精确strict xfail／4身份skip，19例水文100 eager／93 JIT失败坐标保持原限。旧机器表中的151坐标属于撤回求和候选的历史版本，以修订numeric回执为当前数值身份；不回写冻结表。
[当前紧凑覆盖层](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/coverage.json)保留13个PB03稳定过程ID，按源条件、真实caller／writeback、具名案例和剩余动作关联证据；52条父表和七种历史ID集合保持原身份，不复制巨型arm列表或更改分母。第三包dry／joint热浮点归因保持关闭；第四包复用342阶段／228恒等式，只观察三组保存失败输入，normal同profile的列积分取整、同原操作数窄JIT递推和ALT预算相消已有有界证据，五个蒸发限值包含两组保存peat别名。雪链先核完整同入口，再观测transfer一ULP热差→微量liquid→底层excess，gone／melt时groundflux仍零。没有求和重写、阈值或strict门修改。

第四包已接受的C65／C66分别保留原终止条件跨嵌套JIT／scan／block的纯错误状态，以及初始化原665–740的整组缺失先判／覆盖顺序；旧反例与同测试隔离13通过／1原源身份门skip有实际回执，旧输入和科学容差原样。公共partial初始化修补不证明native IOIPSL producer或当前完整driver路径已证；纯热helper的新编译消费者需checkify并在消费前检查错误。合法false mask由源生命周期排除，任意注入false不获支持。新增原根／legacy stress、canopy精确容量和入渗顶层耗尽只升级具名条件与真实边界，精确min等式的浮点不可达证明仅限保存USDA4 tuple，不广播全连续域。

当前[修订包已获主控接受](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-acceptance.json)，生产／测试／CI提交`d2d48e6d5007079493cf91cca21c978eb79fa24a`。NSLM原compiled caller已迁移，全部原输入／数值断言／容差保持；真实module内外JIT的材料字段与源中止检查、本地和隔离UV各3项通过，无skip。原封包、独审及修订增量审查均保留。首次新增整返回比较的13/14项失败、max abs5.34106093e-12单独保留，成因未定；不因窄接口通过而提升全状态数值接受，不新增第三项原因。

私有peat的一次本地补证确认hydrol与全局参数对象有不同COMMON存储符号；本地FCM身份与历史1961运行仍不一致，缺准确binary／module／ppsrc／compiler绑定。[外部请求](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/hydrol-conditions/external-evidence-request.json)保留true目标条件，不以trace false排除。C64及其`e275a33fd5864957e9b241138359c1f6001a5756`、npts1/nstm6和一般native grammar／多点未证限制保持接受范围。[具名剩余动作](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/remaining.json)保留自然初始雪包、第二energy容量、其余PB组与完整源分母；本包本地完成不能称PB03或物理阶段完成。
第四包接受时为66项／37相关生产提交；`numerical_acceptance=false`，整模型综合验证和669尚未开始。

第五包[机器覆盖层](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/coverage.json)为32个PB-01／PB-02／PB-04父过程保留原稳定ID、源身份、旧等级和当前有限处置；[条件对照](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/source-condition-reconciliation.json)逐项链接原条件证据、源排除或精确剩余动作。15条冠层增量、5组能量条件及124旧arm、101条season条件／2隐含cap、207条生长条件／68循环域、254条litter-soilcarbon条件／24算术界限是不同视图，不相加成通过分母，不把完整元组allclose称逐arm执行。原52过程和308／12／26428／4217／824历史口径不改。

本包有界原参考和真实调用／写回见[回执](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/validation-receipt.json)。六项已由主控接受为C67–C72，共同提交`b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12`，正式72／38；全生长链字面原caller960比较通过，最初adapter相位参考保留为诊断。原浮点结论、严格失败和未解嵌套JIT门各守原范围；source-mapped、原组件数值和实际transport证据分别列明，不升级整过程或自然轨迹。

fixedsnow原源未定义值、外部PBL、自然低LAI／联合／雪能量前驱、私有peat历史身份、native partial、DOC原始通量／floor和其余算术／diffusion条件及跨组IO继续按[具名动作](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/remaining.json)保留。本包已获主控有限接受，不是物理阶段或完整分母收口；numerical_acceptance=false，无整模型／669／远端运行。参数／状态90＋1＋1／92及93／126／181口径保持，研究树未读写、同步暂停。

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-acceptance.json)核对30份当前产品及原交付／独审封存身份，复用owner与独审证据。最终隔离六文件回归260通过、4个外部原源／历史资产身份门跳过；实际边界13通过、来源门2通过，相邻62通过分别保留，不相加为科学覆盖。新11份Python lint通过，生产旧诊断64前／64后未增加；hosted CI未执行。主控只解析回执、读配对源码及实际caller，没有重复积分或数值套件。5份新JSON的Git规范化仅为换行，解析值相同；发布状态见[记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-publication.json)。

2026-11-20，第六包经主控接受，五项修补登记C73–C77，生产／测试／CI提交`f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b`。累计**77项修正／支持补齐、39个相关生产提交**；75项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏，不是77项物理公式错误。本轮五项均有复查前`7cae3e02adad23674767a590fdf4d27f239ce065`只读源码证据，首次引入日期未断言，未恢复运行历史代码。

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-acceptance.json)核对46份产品、交付与独审封存身份，并复核配对原源、实际写回及历史源码。保存的隔离CI按精确节点合并为454通过、9项科学strict xfail、10项外部身份skip、0失败／错误；这不是一次不间断全通过运行。首次漏复制已跟踪restart schema的9个setup错误保留，修正复制后仅重测受影响的10节点，未重测此前通过项。配置内Ruff通过，生产lint 62前／62后无新增；hosted CI未运行。主控解析JUnit并复用owner／独审验证，未再积分；12份JSON仅Git换行规范化、语义相同。[发布记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-publication.json)保留最终版本。

保存的嵌套JIT整返回差异现有有界递推取整与表面通量抵消证据；13/14 soilflx_pft及其别名仍严格失败，不声明唯一HLO／机器指令原因。两个新增合成相等浓度DOC元组捕获到入口约1e-17取整变化，实测入口重放能复现严格相等分支变化及4／5个失败坐标；没有定位最早上游算术，也不外推旧年度轨迹。第六包验收时，8项新JIT完整系数通量门／60坐标留待第七包归因。旧HYDROL／热／雪／低光／DOC／NPP／grain严格门各守原范围，不同配置DGVM诊断不计新修复。

52父过程的整体路线等级仍为source_mapped／open；4,059词法候选、565源审查组和187跨模块／外部调用组不等于数值覆盖或缺陷数。新有限证据包括14生长元组2611比较、扩散实际arm、非零冻融carry、冠层／capacity前驱、forcing／静态输入／modelout／routing和6份科学restart文件往返（1402检查，含147独立字段）。真实IOIPSL／native producer、maintained完整调度、自然历史、私有peat／PBL、原未定义合同与完整源分母仍有缺证。C64限npts1／nstm6；年度DOC真实边界／残差／异养呼吸均为null，原mass warning与负中间量保留。

已验收的[第七包覆盖表](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase07-source-dispositions-and-energy/dispositions.json)保留565个源码组、187个调用组、4,059个条件ID和1,277个原调用点。源码组中161组在其声明范围内有支持证据、94组依原源排除、92组属基础设施、185组有部分证据、33组有明确本地缺口。调用组相应为14组有支持证据、61组依原源排除、76组属基础设施、24组有部分证据、12组缺外部参考。`stomatelpj` 会调用 `cover`，但目标配置的 `ok_dgvm=false` 跳过了其物理计算；采用总表中的判定，旧子条目的 `partial` 标签待在新覆盖层对齐。

第八包增量已验收。33 个本地缺口和 35 个相邻部分组的 445 个原成员 ID 已逐一关联。另有 16 组土壤热与 STOMATE 分支补证。共 84 组获得新增证据，其中 55 组在声明范围内得到支持。9 个部分调用组补充了真实消费者证据。仍有 163 个源码组、36 个调用组（含 12 个外部缺口），52 个父过程仍待整体验证。[覆盖差额](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase08-hydrol-snow-and-consumers/controller-repair/coverage-delta.json)

C78 和 C79 验证了所测 CPU 单点与批量形式下的分箱结果。第九包另行接受 C80，覆盖论文活动下限、原组件池量和通量及实际写回。局部原生残差和年度影响分别保留。[主控验收](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase09-carbon-activity-and-consumers/controller-acceptance.json)

第九包对 30 个碳源码组补证，其中 14 组有限支持、7 组按实际论文开关排除、9 组部分开放。实测 STOMATE 的两处碳调用获有界支持，LPJ 的 NPP 调用仍部分开放。剩余 142 个源码组、34 个调用组，外部缺口 12 个。本包还纠正第八包水文平滑合成网格的首层间距范围。原严格差异继续记录。 [覆盖差额](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase09-carbon-activity-and-consumers/coverage-delta.json) · [平滑证据](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase09-carbon-activity-and-consumers/hydrol-smoothing/README.md)

第十包 50 个冠层与能量组逐项对账，37 组获有界支持，13 组部分开放，另有 4 个相邻组获得有限组件证据。CONDVEG 的完整局部返回和下一次空气动力接收得到有界支持，DIFFUCO 和 ENERBIL 仍有前驱缺口。低风的 31 项气动 JIT 严格差异及原未赋值、外部、自然前驱的限制仍具名保留。DIFFUCO `leaf_ci` 在 36 组同输入下证实存在逐层重启写回错误，登记为 C81。 [第十包覆盖差额](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase10-canopy-energy-and-consumers/coverage-delta.json) · [主控验收](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase10-canopy-energy-and-consumers/controller-acceptance.json)

两组热系数输入中，底层递推的舍入差异经表层相消、PFT加权和雪土混合传到输出。原来8项严格比较的60个失败坐标全部保留。52个父过程的整体路线仍开放，`numerical_acceptance=false`。原始观测、编译计算、实际写回和外部运行身份的具体缺项见[剩余总账](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase07-source-dispositions-and-energy/open-groups.json)。

第十一包逐项关联 42 个热土壤组与 21 个水文积雪组的 707 个原条件。9 个源码组获得范围内支持。实际 ENERBIL、HYDROL、THERMOSOIL 的三个调用组补充了参数、相位和写回证据，仍有其他前驱待核对。剩余 92 个源码组、33 个调用组，其中 12 个缺外部证据。水文源网格平滑 7 项、旧人工网格 8 项、土柱新比较 16 项和气动 31 项严格失败分别保留。52 个父过程和七套历史身份继续沿用，年度质量余量、DOC、NPP 效应未知，`numerical_acceptance=false`。[第十一包覆盖差额](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase11-soil-water-heat-and-aerodynamics/coverage-delta.json) · [具名余项](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase11-soil-water-heat-and-aerodynamics/open-groups.json)

第十一包接受 C82 的基岩热容量开关和 C83 的土类索引。返修对照补齐了矿质土类索引、有效逐点阈值与后续湿度/胁迫的接收证据。层湿度、胁迫和冻土湿度的受影响源码组继续按具体余项标记为部分支持，原 9 组有限支持及 92／33／12 口径保留。气动、土柱和平滑严格失败及年度未知继续按原证据记录。 [主控验收](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase11-soil-water-heat-and-aerodynamics/controller-acceptance.json) · [修订覆盖差额](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase11-soil-water-heat-and-aerodynamics/controller-repair/coverage-delta.json) · [气动勘误](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase11-soil-water-heat-and-aerodynamics/controller-repair/aero-correction.md)

第十二包关联 9 个植被碳、8 个 STOMATE/LPJ 调度和 10 个土柱源组，共 1,026 个原成员。这 27 个组仍为部分支持。土柱 254/270 个字段比较通过，16 项严格差异保留。复用上游两套结果时，当前 JAX 接收方在第一个入口通过 30/30 项，第二个入口通过 24/30 项，6 项超限均在零起算第 4 个格点。原 Fortran 的周转、最终 LAI 和 Vmax 同输入对照待补。源码组和调用组分母为 565/187，词法候选 4,059 个，父过程 52 个，沿用七套历史身份。整模型 `numerical_acceptance=false`。[覆盖差额](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase12-root-water-and-vegetation-carbon/coverage-delta.json) · [接收方结果](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase12-root-water-and-vegetation-carbon/npp-lpj/README.md)
