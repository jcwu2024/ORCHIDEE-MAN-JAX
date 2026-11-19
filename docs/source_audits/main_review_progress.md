# 主线进度

2026-11-18，**参数和状态传递已验收完成；用户新授权的物理计算与条件分支验证接力现已启动。** 范围遵循既定PFT14等价合同；还没有宣称整个模型与Fortran全面等价。

| 工作面 | 这次完成意味着什么 |
| --- | --- |
| 参数读入和覆盖 | 选定参数能从配置送到实际消费者，缓存不会把另一份设置混进来 |
| 初始化与继续运行 | 区分新运行、重启初始化和连续运行；沿用前次确定的设置，缺少必需历史时明确报错 |
| 日／编译块与年界交接 | 受检状态和初始化模式跨接口保留；空运行或失败不伪造新历史 |
| 保存与读回 | 已支持的14／2PFT normalized单点文件链及条件refSOC字段按源合同处理 |
| WATCHOUT输入 | 文件识别、读取／插值、driver／消费者、条件重启与连续历史交接在声明域内接通 |

当前累计**63项修正／支持补齐、35个相关生产提交**。参数／状态收尾接受的C60／C61之后，2026-11-19新增接受C62潮汐输入链和C63有效CHECK_CWRR合同，共同提交`7b35196462fbfd53f531fd1dc751699092f80104`。61项有复查前证据，C42／C57为复查期间遗漏；这不是63项物理公式错误。[来源登记](correction_origins.md)和[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)保留准确年代、证据与同步依赖。研究分支尚未同步。

参数／状态包代码／测试／CI提交：`aa14d04210ab371aae6e959da193561f37f55d0e`。上一包99项隔离检查＋另1项静态检查，本次最终35项与另7项检查分别通过；各批按各自源码版本记录，不机械相加。主控复用这些验证并核对实际代码和交接边，没有重跑整模型。原参考与科学阈值不变。

2026-11-19，主控已接受[冠层／能量首包及CI修订](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-acceptance.json)的有限增量证据。原Fortran、eager／真实窄JIT及实际消费者交接已按边界记录，[当前覆盖总账](physics_branch_coverage.md)保留未完成分支。没有生产模型修改或新修复编号；低光cim严格失败与极低正LAI前驱／参考问题保留。测试／参考／CI提交为`ebd1141d5b214ab0b874d124eab2466eef8e0a77`，不增加34个生产提交的计数。

本轮为[PB-03组件增量包](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/README.md)：关联已接受土壤／热证据，补有限水文／冻结／refSOC、完整热和相邻雪组件原参考，以及真实水分输入、writer／packet、下一DIFFUCO选择与STOMATE入口adapter。后者没有执行碳过程或完整下一步。热通量／雪groundflux新严格差异、旧grain门、私有泥炭开关producer和自然雪域限制分别保留；不因这些数值差异增加修复计数或宣称PB-03完成。实际检查、独审和主控状态以新包为准，PB-01／PB-02剩余项继续保留。每包封存后回传一次，主控验收接续；不设定时消息，不自动启动综合模型运行。

主控对原 PB-03 候选要求补齐生产集成和配置合同，当前修订见
[controller-repair](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/README.md)。
本修订接通两份原潮汐文本到真实 HYDROL 入口，绑定有效 `CHECK_CWRR`，完整 owner 明确拒绝尚缺
守恒／ET 语义的 true；低层诊断 API 的历史默认保留。土柱求和恢复已接受基线，原顺序候选只保留为诊断。
原包、参考、容差和未解严格坐标保留；修订已获[主控接受](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/controller-acceptance.json)，[发布记录](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/controller-publication.json)单列准确身份。
隔离uv 135通过／195精确strict xfail／4身份门skip，另55输入／adapter及87邻接回归；主控复用结果并核对22份实际产品，未重跑模型。
正式63／35；下一包为`PB03-STRICT-THERMAL`及同组有界源分支收口。私有peat／自然雪前驱和其他PB组仍有缺项，整个物理阶段没有完成。

之后仍需完整模型集成，以及部分外部参考／原生IO身份和科学限制的处理。原有DOC与积雪严格门仍保留，`numerical_acceptance=false`。这些缺证和限制不等于已经确认的新代码bug，已接受浮点成因也不把严格失败改写为通过。

已接受的[CI修订](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-ci-repair/README.md)解决干净检出缺原Fortran文件的问题。完整三文件在tracked＋明确overlay的隔离环境实际本地uv运行：110通过、2来源门跳过、2严格预期失败；声明输入与数值检查仍必跑。本地有源时另有2项静态来源门通过。主控复用结果，核对9份当前执行资产和依赖审计，没有重跑数值；hosted Ubuntu／uv sync未执行。原候选及科学回执保留，发布身份见[发布记录](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-publication.json)。

[参数／状态完成标准](parameter_state_closeout_plan.md) · [主控验收](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-acceptance.json) · [发布核对](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-publication.json) · [新阶段范围及停止标准](physics_branch_closeout_plan.md)。
