# 主线进度

2026-11-20：**参数与状态传递已按声明范围收尾；物理计算与条件分支验证正在继续。** 尚未宣布完整模型等价，也未启动最终综合验证。

2026-11-20，主控接受C65（编译热过程保留原冻结profile中止检查）和C66（初始化先判整组重算，再校验保留字段），共同生产／测试／CI提交`d2d48e6d5007079493cf91cca21c978eb79fa24a`。累计**66项修正／支持补齐、37个相关生产提交**，其中64项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏；不是66项物理公式错误。两项源行为均见于最早跟踪7333b46c及复查前7cae3e02，不能追称导入前的首次引入日期。见[来源登记](correction_origins.md)及[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)。

| 工作组 | 当前结果 | 尚需工作 |
| --- | --- | --- |
| 参数／状态 | 已接受初始化、日／年交接、normalized restart、WATCHOUT及首插值准入的有限合同 | 一般native IO／外部producer及完整模型科学门另留 |
| PB-01冠层／光合 | 原参考及eager／JIT、实际能量消费者证据已接受 | 根通气、部分albedo、极低正LAI合法前驱与源条件收口 |
| PB-02能量／QSAT | 已有420字段／运行检查、3456数值及量率写回证据 | 合法联合beta、tiny-Milly／雪条件与外部PBL |
| PB-03水文／热／雪 | C62–C66已接受；normal／ALT、dry／joint热及雪groundflux有界归因完成 | 私有peat历史身份、自然雪／能量前驱、native partial与其他源条件，完整分母未闭合 |
| PB-04植被／碳 | 既有Oracle、NPP／DOC结论及日历状态证据复用 | 当前完整同输入、条件分支及真实碳通量／写回核对 |
| PB-05跨组 | 已有有限真实接口及单位合同 | 剩余跨组、diagnostic、restart／modelout数值边界与终审 |

[主控验收](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-acceptance.json)复用原包修前2失败／修后隔离13通过＋1原源身份门skip及本地源门，并核对[集成修订](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-repair/README.md)：NSLM原compiled回归的输入、断言和容差保持，新增真实内层module JIT／外层checkified窄owner的合法材料字段及非法拒绝，两环境各3通过、无skip。11份当前及冻结产品、原／新封存身份匹配；主控本次未重跑数值。原源码与科学封包保留，Git对两份新JSON仅规范换行，解析值相同。发布见[记录](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-publication.json)。

C65改变纯explicit热helper的编译合同：checkify后JIT，返回Error／result并在消费前check_error；module／scan／block与维护测试成组接入。C66仅接受有任一整字段缺失、因此九系数整组覆盖的公共初始化输入，native partial restart及完整driver可达尚未证明，保留组的混合marker仍拒绝。旧C64源等价仍限npts=1、nstm=6。原HYDROL100 eager＋93 JIT、固定热／雪21和不同输入handoff2严格次数不变。新增完整返回的嵌套JIT比较另有13/14项失败、最大绝对差5.34106093e-12，成因未定；窄接口通过不等于整状态等价。

水文normal／ALT残差与独立雪groundflux现有完整同输入、内部首差及传播链证据，接受有界浮点归因，不改物理公式、求和或科学容差。dry／joint热、NPP／DOC／grain的旧结论保持各自范围。新嵌套JIT完整返回差异不能自动套用上述归因。私有peat历史build、自然初始雪包／第二energy容量及native partial restart等缺证仍在；PB-03、完整源分母及整个物理阶段尚未完成，numerical_acceptance=false。

下一包并行推进PB-01／PB-02剩余冠层和能量源条件，以及PB-04植被生长、分配、周转、凋落物与碳过程的当前源／同输入／真实消费者核对。按有限源条件补缺，复用已接受参考，不重开已闭合浮点调查。PB-05跨组收口和后续完整模型综合验证仍需完成。参数／状态90＋1＋1／92及93／126／181等口径保持，研究分支同步继续暂停。

原参数／状态收尾生产提交`aa14d04210ab371aae6e959da193561f37f55d0e`及[验收](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-acceptance.json)保留。原冠层／能量[首包与CI验收](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-acceptance.json)及C62／C63、C64历史封包均保持原身份，不把各版本的测试数量相加或称最终字节全套重跑。当前证据见[覆盖表](physics_branch_coverage.md)，接力与停止标准见[阶段计划](physics_branch_closeout_plan.md)。Sol完成一包只回传一次，主控集中验收发布后派下一包，不启用定时轮询。
