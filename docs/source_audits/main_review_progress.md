# 主线进度

2026-11-18，主控已接受本轮参数和状态交接修复，代码／测试／CI提交 `5fe049ffa249f2c53daf669d576fe590ecbbb999`。参数覆盖、条件状态的创建和保存读回、跨步保留及缓存隔离，在下表所列接口范围内已有证据。**这部分还差WATCHOUT输入传递链的本地收尾；整个模型的数值等价仍未验收。**

累计为**59项修正／33个相关生产提交**。其中57项有复查前源码证据；C42与本轮C57是复查期间引入的遗漏。计数包括接口、默认值和缓存问题，不等于59条活动物理公式错误，也不是完成百分比。

| 本轮接受 | 实际结果与边界 |
| --- | --- |
| 土壤深度参数（C55扩展） | 七个活动标量从配置／初始化传到表、系数消费者和编译参数；一项原源注释分支的metadata保持不激活。原十向量合同不变 |
| 条件refSOC状态（C56） | 按源开关读取、缺失／全marker初始化、双镜像交接及真实三文件读回；默认93科学字段／126 canonical字段不变，条件字段单列 |
| 冷启动出口（C57） | 修正日历接线时引入的未定义context；保持已有prepared对象或从cold bundle建clock。只验证该就绪出口，不宣称完整冷启动物理链已通过 |
| 编译缓存（C43扩展、C58） | scan隔离捕获的context；block隔离由状态决定的phenology分支。真实窄JIT使用明确的物理替身 |
| 跨步温度状态（C59） | 补temp_growth半小时保留，实际clock入口可读取；旧carry遗漏与后来新增必需消费者分开记账 |

主控核对了实际差异、源规则、425项交付绑定、独立审查及保存JUnit中94个不同节点的最终通过证据：59个可移植接口、9个深度driver、6个旧缓存构造、20个本地静态合同检查。它们是分批修复后的组合证据，**不是单次全套94全绿，也不是hosted Ubuntu执行**。C55原五例eager／JIT各191项证据直接复用，主控没有重跑数值或模型。

独立终审还抓到本包新接入refSOC时“关闭开关仍保留旧条件值”的遗漏，已在首writer和新年度双镜像处修正，原两条失败证据保留，归入C56同一链。夹具错误单列；主控提交前只去掉一个测试文件尾部空行，执行AST不变。

下一包核对**WATCHOUT文件→时间插值→driver／intersurf的参数运输**。原Fortran按forcing中的小写levels识别这条路径，本地源和文档已给出系数约定，足以继续；不能把缺实际大气producer身份当作当前必须停工的理由。RELAXATION=true仍按原源拒绝，不再次乘dt、不清零文件提供的A系数。实际外部producer版本和输出时间身份继续单列未知。

92／93／126／181及179参数投影等账本视图相互重叠，不能相加。原NPP／DOC／积雪浮点结论不重开；DOC严格状态与真实碳通量、其他物理条件覆盖和整模型门继续保留，`numerical_acceptance=false`。daily/global研究树仍未同步。

用户已授权逐包接力：Sol封存并停止修改后给本主控发一次交付消息；主控验收后再派本范围下一包。这部分完成后停止，不自动进入其他阶段。范围和授权见[收尾计划](parameter_state_closeout_plan.md)。

[主控验收](../../outputs/reviews/main-physics-20261118/parameter-state-closeout/controller-acceptance.json) · [当前总账](../../outputs/reviews/main-physics-20261118/parameter-state-closeout/controller-progress-ledger.json) · [来源登记](correction_origins.md) · [发布回执](../../outputs/reviews/main-physics-20261118/parameter-state-closeout/controller-publication.json)
