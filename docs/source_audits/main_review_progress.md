# 主线进度

2026-11-21，**物理检查第八包已验收，累计 79 项修正或支持补齐、40 个生产提交。** 参数与状态传递已在既定范围收尾，物理分支验证仍在进行。

2026-11-20，第六包经主控接受，五项修补登记C73–C77，生产／测试／CI提交`f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b`。累计**77项修正／支持补齐、39个相关生产提交**；75项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏，不是77项物理公式错误。本轮五项均有复查前`7cae3e02adad23674767a590fdf4d27f239ce065`只读源码证据，首次引入日期未断言，未恢复运行历史代码。

| 工作组 | 当前结果 | 尚需工作 |
| --- | --- | --- |
| 参数／状态 | 既有有限合同已接受；新增6份科学restart文件到实际reader／下一组件的有限往返 | 真实IOIPSL／native producer与自然年际历史等范围另留 |
| PB-01冠层／光合 | 新增restart／held／维护LAI前驱与有限消费者 | 自然低LAI历史、外部PBL及fixedsnow原未定义合同 |
| PB-02能量／QSAT | 正capacity与QSAT域有条件证据，有限原组件输入已绑定 | 新8项JIT完整系数通量失败待归因；超出证明前提的源域与外部PBL |
| PB-03水文／热／雪 | 既有有界归因及新增嵌套整返回浮点机制已接受，严格门仍失败 | 私有peat身份、真实native partial、自然初始雪／第二capacity与其余源码条件 |
| PB-04植被／碳 | C73默认值、C74状态所有权已修；生长、扩散、冻融有限同输入链通过 | 剩余源条件、自然联合前驱、年度DOC真实通量；相等DOC严格失败保留 |
| PB-05跨组 | C75–C77输入处理已修，forcing／静态控制／modelout／routing与restart有限接口验证 | maintained完整调度、跨模块源调用归属及阶段终审 |

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-acceptance.json)核对46份产品、交付与独审封存身份，并复核配对原源、实际写回及历史源码。保存的隔离CI按精确节点合并为454通过、9项科学strict xfail、10项外部身份skip、0失败／错误；这不是一次不间断全通过运行。首次漏复制已跟踪restart schema的9个setup错误保留，修正复制后仅重测受影响的10节点，未重测此前通过项。配置内Ruff通过，生产lint 62前／62后无新增；hosted CI未运行。主控解析JUnit并复用owner／独审验证，未再积分；12份JSON仅Git换行规范化、语义相同。[发布记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-publication.json)保留最终版本。

保存的嵌套JIT整返回差异现有有界递推取整与表面通量抵消证据；13/14 soilflx_pft及其别名仍严格失败，不声明唯一HLO／机器指令原因。两个新增合成相等浓度DOC元组捕获到入口约1e-17取整变化，实测入口重放能复现严格相等分支变化及4／5个失败坐标；没有定位最早上游算术，也不外推旧年度轨迹。第六包验收时，8项新JIT完整系数通量门／60坐标留待第七包归因。旧HYDROL／热／雪／低光／DOC／NPP／grain严格门各守原范围，不同配置DGVM诊断不计新修复。

52父过程的整体路线等级仍为source_mapped／open；4,059词法候选、565源审查组和187跨模块／外部调用组不等于数值覆盖或缺陷数。新有限证据包括14生长元组2611比较、扩散实际arm、非零冻融carry、冠层／capacity前驱、forcing／静态输入／modelout／routing和6份科学restart文件往返（1402检查，含147独立字段）。真实IOIPSL／native producer、maintained完整调度、自然历史、私有peat／PBL、原未定义合同与完整源分母仍有缺证。C64限npts1／nstm6；年度DOC真实边界／残差／异养呼吸均为null，原mass warning与负中间量保留。

第七包已通过[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase07-source-dispositions-and-energy/controller-acceptance.json)。两组相同输入下，土壤底层递推系数在二进制浮点的最后一位出现差异，传到表层后，两项接近的量相减使热通量出现残差。保存的操作数、两个输出别名及雪土加权结果支持这一解释。8项比较的60个坐标仍按原容差记录为失败。该结论限于这两组输入。新增3项隔离回归通过，外部原源身份检查在隔离环境跳过、在本地通过，配置的Ruff检查通过。本轮只增加测试、原参考和CI配置，累计仍为77项修正或支持补齐、39个生产提交。

第八包登记 C78 冻结水文分箱、C79 泥炭湿度分箱，代码提交 `9591652668df1afb4955d7282a6091b96d844e4b`。上次批量复核有 90 个失败坐标记录（含重复单点），修订后的同一脚本 18 行结果全部通过。原生边界核对、实际含水量写回和 STOMATE 分解率均使用原参考。隔离测试 18 项通过，1 项外部原源身份检查跳过，该项在本地通过。[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase08-hydrol-snow-and-consumers/controller-acceptance.json) · [累计修复及同步依赖](correction_origins.md)

本包将 55 个源码组推进到其声明范围内有支持证据，仍有 163 个源码组和 36 个调用组要补证。下一包优先处理暖干 PFT14 的活动下限。已有同输入对照在两个分解率字段共观察到 64 个超限坐标，差异传至凋落物和土壤碳的活动系数。下一步要核对原精度在实际调用中的传递，并用相同输入检查碳库更新。年度 DOC 边界通量、质量残差和异养呼吸仍未知，`numerical_acceptance=false`。[剩余清单](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase08-hydrol-snow-and-consumers/open-groups.json)

修复位置与未来同步依赖见[来源登记](correction_origins.md)、[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)和[当前覆盖表](physics_branch_coverage.md)。Sol每包封存后回传一次，主控集中验收发布后继续派发，不定时轮询。565／187是源码审阅分组，不是752个已知bug，也不是要求752次模型运行。
