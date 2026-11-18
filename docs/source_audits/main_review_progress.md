# 主线进度

2026-11-18，**参数和状态传递已验收完成；用户新授权的物理计算与条件分支验证接力现已启动。** 范围遵循既定PFT14等价合同；还没有宣称整个模型与Fortran全面等价。

| 工作面 | 这次完成意味着什么 |
| --- | --- |
| 参数读入和覆盖 | 选定参数能从配置送到实际消费者，缓存不会把另一份设置混进来 |
| 初始化与继续运行 | 区分新运行、重启初始化和连续运行；沿用前次确定的设置，缺少必需历史时明确报错 |
| 日／编译块与年界交接 | 受检状态和初始化模式跨接口保留；空运行或失败不伪造新历史 |
| 保存与读回 | 已支持的14／2PFT normalized单点文件链及条件refSOC字段按源合同处理 |
| WATCHOUT输入 | 文件识别、读取／插值、driver／消费者、条件重启与连续历史交接在声明域内接通 |

本批登记两项：**C60**是既有WATCHOUT支持缺口的补齐，**C61**是旧取整及完整首日检查问题的修复。它们不是两项新发现的物理公式错误。累计**61项修正／支持补齐、34个相关生产提交**；修复依据、年代、准确提交和未来同步依赖均已记入[来源登记](correction_origins.md)和[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)。研究分支尚未同步。

代码／测试／CI提交：`aa14d04210ab371aae6e959da193561f37f55d0e`。上一包99项隔离检查＋另1项静态检查，本次最终35项与另7项检查分别通过；各批按各自源码版本记录，不机械相加。主控复用这些验证并核对实际代码和交接边，没有重跑整模型。原参考与科学阈值不变。

2026-11-19，主控已接受[冠层／能量首包及CI修订](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-acceptance.json)的有限增量证据。原Fortran、eager／真实窄JIT及实际消费者交接已按边界记录，[当前覆盖总账](physics_branch_coverage.md)保留未完成分支。没有生产模型修改或新修复编号；低光cim严格失败与极低正LAI前驱／参考问题保留。测试／参考／CI提交为`ebd1141d5b214ab0b874d124eab2466eef8e0a77`，不增加34个生产提交的计数。

接力下一包为PB-03：并行核对水文／冻结／泥炭和土壤热／积雪，先关联已接受案例，再补未验证的合法条件及真实下一消费者。后续接植被／碳与跨组边，PB-01／PB-02剩余项也继续保留。每包封存后回传一次，主控验收接续；不设定时消息，不自动启动综合模型运行。

之后仍需完整模型集成，以及部分外部参考／原生IO身份和科学限制的处理。原有DOC与积雪严格门仍保留，`numerical_acceptance=false`。这些缺证和限制不等于已经确认的新代码bug，已接受浮点成因也不把严格失败改写为通过。

已接受的[CI修订](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-ci-repair/README.md)解决干净检出缺原Fortran文件的问题。完整三文件在tracked＋明确overlay的隔离环境实际本地uv运行：110通过、2来源门跳过、2严格预期失败；声明输入与数值检查仍必跑。本地有源时另有2项静态来源门通过。主控复用结果，核对9份当前执行资产和依赖审计，没有重跑数值；hosted Ubuntu／uv sync未执行。原候选及科学回执保留，发布身份见[发布记录](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-publication.json)。

[参数／状态完成标准](parameter_state_closeout_plan.md) · [主控验收](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-acceptance.json) · [发布核对](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-publication.json) · [新阶段范围及停止标准](physics_branch_closeout_plan.md)。
