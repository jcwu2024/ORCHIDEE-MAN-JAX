# 主线进度

2026-11-20：**参数与状态传递已按声明范围收尾；物理计算与条件分支验证正在继续。** 尚未宣布完整模型等价，也未启动最终综合验证。

2026-11-20，主控接受第五包六项独立修正／支持补齐，登记C67–C72，生产／测试／CI提交`b73a12e5fb8a8b7ab6bdfa7e7a03c20c89808e12`。累计**72项修正／支持补齐、38个相关生产提交**；70项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏，不是72项物理公式错误。本轮六项均经只读`git show`确认存在于复查前`7cae3e02adad23674767a590fdf4d27f239ce065`；首次引入日期未作断言，未恢复或运行旧模型。

| 工作组 | 当前结果 | 尚需工作 |
| --- | --- | --- |
| 参数／状态 | 初始化、日／年交接、normalized restart、WATCHOUT及首插值准入的有限合同已接受 | 一般native IO／外部producer及完整模型科学门另留 |
| PB-01冠层／光合 | 第五包7组新增原参考和有限实际写回／消费者已接受 | fixedsnow原未定义值、自然低LAI／重启前驱、低光cim严格门与外部PBL |
| PB-02能量／QSAT | 第五包198比较／3176值、联合beta和量率写回已接受 | 自然联合状态、非正capacity producer、tiny-Milly广域准入与外部PBL |
| PB-03水文／热／雪 | C62–C66及normal／ALT、dry／joint热和雪groundflux有界归因已接受 | 未解嵌套JIT整返回、私有peat、native partial及自然雪／能量前驱 |
| PB-04植被／碳 | 第五包season、生长、litter-soilcarbon完整元组与真实返回写回已接受；C67–C72已登记 | 其余算术／diffusion条件、自然前驱、DOC原始通量／floor观测 |
| PB-05跨组 | 当前冠层／能量／日碳返回到packet及下一实参的有限合同已接受 | 剩余forcing／diagnostic／restart／modelout数值边界与阶段终审 |

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-acceptance.json)核对30份当前产品及原交付／独审封存身份，复用owner与独审证据。最终隔离六文件回归260通过、4个外部原源／历史资产身份门跳过；实际边界13通过、来源门2通过，相邻62通过分别保留，不相加为科学覆盖。新11份Python lint通过，生产旧诊断64前／64后未增加；hosted CI未执行。主控只解析回执、读配对源码及实际caller，没有重复积分或数值套件。5份新JSON的Git规范化仅为换行，解析值相同；发布状态见[记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/controller-publication.json)。

本包接受有限组件、源条件与真实写回／下一实参；不宣布PB各组、完整源分母或物理阶段完成。保留原HYDROL／热／雪／低光／DOC严格失败及已接受浮点结论的原范围。嵌套JIT完整返回13/14项失败、最大绝对差5.34106093e-12仍未归因。fixedsnow原未定义值、私有peat／外部PBL、native partial、自然联合／雪能量前驱及DOC原始通量／floor观测等缺证继续具名保留；年度DOC真实残差与异养呼吸仍未知，不由端点推算。C64仍限npts1／nstm6。

下一包按[具名剩余动作](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/remaining.json)并行核对目标源条件分母、生长／扩散等剩余算术分支、嵌套JIT整返回差异和PB-05跨组数值接口；可在现有本地组件边界内补实际前驱，不能把合成元组称为自然轨迹。复用已接受参考，不重开已闭合浮点调查。整模型综合验证尚未启动，`numerical_acceptance=false`。参数／状态90＋1＋1／92及93／126／181口径保持；研究分支同步继续暂停。

当前[覆盖表](physics_branch_coverage.md)保留32条本包父过程、52条历史过程及各条件视图，均不相加为全源码分母。修复位置、提交和将来同步依赖见[来源登记](correction_origins.md)与[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)。Sol完成一包只回传一次，主控集中验收发布后派下一包，不启用定时轮询。最终按[阶段计划](physics_branch_closeout_plan.md)逐条完成标准判断，不以修复数或运行时长替代验收。
