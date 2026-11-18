# 参数和状态传递收尾

**2026-11-18：声明支持域内已通过主控验收，本部分接力结束。** 最终代码／测试／CI提交`aa14d04210ab371aae6e959da193561f37f55d0e`。这是参数与状态接口的有限收尾，不是全模型数值等价验收，也不启动下一大阶段。

## 支持范围与完成项

范围为既定PFT14论文工作流、选定参数／forcing变化、normalized本地单点14／2PFT restart、条件refSOC，固定annual file-start及持有前次有效state/context的continuous-history。RELAXATION保持源fatal；未实现的其他域不因本次通过而自动准入。

| 工作面 | 当前接受结果 |
| --- | --- |
| 有效参数与缓存 | C55十向量和七活动深度标量；C43 scan/later-day context ownership与C58 block静态dispatch；WATCHOUT实际file cadence和有效控制到消费者 |
| 创建／恢复相位 | C57 cold出口；C56条件refSOC file-init／new-native规则；actual首子步选择drag、first-main只执行一次，continuous不重初始化 |
| 运行中交接 | 条件refSOC镜像、C59 temp_growth及默认／compact／窄compiled接口；standard／cold／native返回真实已初始化context，两公开continuous入口缺历史即拒绝 |
| 保存／读回 | default93＋条件refSOC的真实小型三文件14／2PFT链；WATCHOUT四系数由forcing供应，不误从driver restart读写 |
| forcing支持链 | C60 schema／读取／插值／driver／消费者／编译tuple／IO贯通；C61正NINT及完整首日guard；真实文件默认长度和显式TIME_LENGTH复用源owner |

## 验收依据与限制

有限清单关联92额外字段、93科学字段、126 canonical、181生命周期、选定参数179及soil/depth子视图，沿用原row身份、源owner、单位／形状／掩码、默认／覆盖、相位、入口／出口。各视图重叠，不相加；90＋1＋1／92及条件refSOC1不变。本支持域未发现遗留必需运输边或未分类交接边。主控核对[最终六边](../../outputs/reviews/main-physics-20261118/watchout-context-handoff/ledger/required-edges.json)及继承账本后接受；这是可审计的有限结论，不是所有未知分支不存在错误的证明。

证据依次为已接受parameter-state-closeout、WATCHOUT原包、R1–R5 repair和R2公开handoff。前包94不同节点的组合证据与C55各191项复用；上一repair99项隔离＋另1静态，最终R2同输入16失败／19通过→35通过及另7项既有接口检查按各自版本单列。独立终审与主控实际源码核对完成；162＋2最终封包校验通过。旧113、66／30不相加，不冒称在最终源码重跑。主控只去除三个测试空EOF并证明AST一致，未新运行数值或整模型。

正式61项修正／支持补齐、34个相关生产提交；C60为旧支持缺口补齐，C61为旧首读取整问题，开发中各次接线遗漏另记年代但不追加C号。[验收](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-acceptance.json)、[来源](correction_origins.md)、[同步依赖](shared_core_branch_review.md#累计修改位置与提交索引)及[发布](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-publication.json)保存精确记录。两个旧WATCHOUT包、R2包及此前主控退回记录原字节保留，当前决定由新的主控记录补充。

`numerical_acceptance=false`。完整物理／compiled数值覆盖和整模型门、DOC真实碳通量／严格状态、积雪严格grain／自然域、真实外部producer／build身份、一般native IOIPSL／多点scatter、任意fresh-midyear／TIME_SKIP仍分别开放。这些不属于本次运输收尾的额外任务；已接受NPP／DOC／积雪浮点归因不重开。无服务器、GPU、训练、付费、669或研究树同步。

## 接力终态

本部分已完成，Sol保持停止；不再发送本部分续包或安排自动提醒。下方保留原人类授权以供追溯，其“完成后停止”条件现已满足。其他工作仍需按用户后续指示处理。

## 授权、交付与停止

主控聊天为“梳理主分支检查工作”，`01a128e9-fda8-7781-ad9b-3ab29323909d`，host=`local`。执行聊天为“用Sol执行主分支检查工作”，`01a1298c-60d9-7020-9086-abffe56c9ccb`，host=`local`。

用户在主控turn `01a1ea4b-f758-70c3-85ac-7dde94bb92fe`、用户消息`01a1ea4b-f7eb-7911-a29f-7c9f86f591a6`明确要求：

> 做完后停下来之前要给你“梳理主分支检查工作”发消息激活你，然后你可以继续给他派下一个……一直到参数和状态传递这一部分做完。

这是本部分的直接人类授权，覆盖执行者向指定主控发一次交付唤醒消息，以及主控验收后继续派本范围下一包；覆盖此前仅手动查看的偏好。需要核实授权时可读取该主控turn。每包完成且停止修改后，执行者发送一次简短交付消息（包路径、实际验证、剩余项／阻塞），随后结束本轮；不另设自动提醒，不发送无变化进度消息，不向其他聊天发送消息。主控只在验收期间工作，复用验证，不与执行者重复调查。

既有本地ORCJAX CPU/x64、有界组件／窄JIT、必要小Fortran参考和小型真实文件IO继续适用。无整模型步／日／年积分、669矩阵、SSH／服务器／GPU／训练／付费作业、环境大重建或研究树同步。真实外部阻塞由主控向用户说明；不重复失败动作或自动扩大资源范围。

执行者不stage／commit／push；主控验收后记录每项修复的依据、年代、反例、验证、准确提交及未来同步依赖，提交并普通推送可审阅阶段。整部分通过后停止接力并报告剩余其他工作，不自动进入下一大阶段。用户新的停止或改写指示始终优先。
