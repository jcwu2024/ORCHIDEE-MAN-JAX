# main 物理正确性检查入口

工作顺序（2026-10-27用户决定）：**先稳固main，暂不与日尺度和全球效率分支对齐**。
已接受修复继续登记准确提交、触发范围、证据及限制，待基础分支这一轮加固后集中评估交接，
避免反复同步。**当前humrel起源与独立Fortran配对已执行**：本地原生捕获补齐960条亚日记录，
正humrel由Day338 offset4水文写入、carry到offset5 DIFFUCO；唯一贡献为tile4/PFT14/第6层。
同相位Fortran层水量在共同阈值下方、humrel为0；原JAX单tile水分胁迫组件在两侧实际tuple上
分别复现各自完整返回，控制与跨侧比较maxabs均0。所测域未复现翻译差，差已在层水量输入存在。
七日daily/NPP原限内通过；**全raw候选比较440项失败仍保留**，独立Fortran文本负零及
Day338原生NPP/完整C3输出缺项也保留。更早水状态起因与年度因果未定，年度NPP科学失败不变。
见[本批交付](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/README.md)及本页末节；
主控已[接受有限机制与同输入证据](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/controller-acceptance.json)，
**本轮NPP专项诊断在此收尾，不自动追加更早历史捕获**。局部水状态跨阈值机制已证实，
首次状态分歧和年度偏差保留为未解决科学问题；不认定纯浮点起因、不新增修复、不放宽原门槛。上一批已由主控
[接受有限生产层证据](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/controller-acceptance.json)，
其原生累加器缺失及58份清单保持原历史状态。此前365日捕获、局部分支放大及年度输出会计已由
项目主控[验收](../../outputs/reviews/main-physics-20261027/npp-current-capture/controller-acceptance.json)，
该批64份清单及回执保持冻结，不能把上批接受状态移作本批验收。
此前两次历史背景交换的[主控验收](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/controller-acceptance.json)
保留原身份。原始NetCDF重启与跨年连续性验证仍待独立裁定；当前科学门槛、支持域和未解决问题保持不变。

当前Stage26：四包边界及补项S26-R1已通过主控验收，最终132项定向通过，**本轮有限源码审查收尾**。
见本页第二十六阶段及[交付入口](../../outputs/reviews/main-physics-20261026/final-source-boundaries/README.md)。
新增C21–C28，累计28项已接受修复、16个代码提交；原接受证据与精确SHA保留。
**V01已主控验收通过**：001点1961年前9日，编译/普通路径与动态packet恢复均通过原数值容差。
见[最终交付](../../outputs/validation/v01-after-stage26/domain-v3/README.md)及
[验收回执](../../outputs/validation/v01-after-stage26/primary-v01-acceptance.json)。累计19模型日、4次调用、
3个模型进程；第9日两种恢复结果最大差异为0，逐日输出最大绝对差异约2.313e-13。
仅对两处源未定义roughheight槽位及实际路线来源记录采用经源码核验的合同；原两次失败保留。
**V02交付已由主控验收，年度科学验证仍未通过**：一次319点1961冷启动全年运行的新NPP
`0.3860031827055333`，参考`0.38518497347831726`，绝对误差`0.000818209227216038`
仍大于原限`0.0003951849734783173`；AGB/BGB/GPP年度门槛通过。
见[本次交付](../../outputs/validation/v02-after-stage26/319-1961/README.md)、
[主控验收](../../outputs/validation/v02-after-stage26/primary-v02-acceptance.json)及下方V02结果。
仅1模型进程/1调用/365模型日，完整年度对象及checkpoint已保存，原年度失败保留。
本次没有新增生产代码修复，累计28项/16代码提交不变；年度偏差因果仍未建立，
raw NetCDF重启与跨点/长期验证限制保留。
**V03a数值执行检查已由项目主控验收**：同一进程独立cold A8→B8→A8，再独立进程ordinary B8，
共4调用/32模型日。实际跨点复用同一个compiled外层可执行对象；A回返、旧A8及B旧逐日对照通过。
最终B两路线按已接受V01数值状态合同通过原限，最大绝对差`6.644986569881439e-7`小于对应限
`0.00011271098844732235`。本次额外检查Python数值载体类相同而触发44项失败，历史false保留；
该新增条件不是V01原科学门槛。见[合同与交付收尾](../../outputs/validation/v03a-after-stage26/contract-closure-v1/README.md)
及[项目主控最终验收](../../outputs/validation/v03a-after-stage26/controller-final-acceptance.json)。
此前primary-v03a-acceptance保留为执行聊天内审；真正最终接受范围见本次独立回执。
无新增生产问题或修复，28项/16代码提交不变；669点/年度/长期科学验收仍未完成。

核查日期：2026-10-28。**本轮尚未确认可解释319年度偏差的生产物理过程bug；
第三阶段已复现有当前 producer 身份的 319 cold-start 首年 NPP 科学超限，
第四阶段已定位年度主要分量；第五阶段确认保存观测不足以继续分配晚年负差；
第六阶段完成唯一获准的全年日值捕获并将暂定负差定位到交替脉冲窗口，
第七阶段恢复候选日志字段/位置并核对零 GPP 分支；第八阶段在恢复新身份认证、
命名历史 producer 和实际源码后，已完成一次真实 Fortran 参考年、一次原本地
CPU JAX 年及 Day338/339/343 三个完整同输入 NPP 回放，18/18 返回数组比较
通过原严格限。两条原轨迹进入 NPP 时已有 GPP、biomass、allocation 和维持呼吸
输入差异；这些回放没有复现 NPP 核心不匹配，不能据此归因具体上游函数或 floor。
指定9天直接 core→send 的36项比较及三份旧日志的27项完整 tuple 比较通过。
三个 float32 存储→send 严格 double 诊断超限保留，单次 float32 cast 兼容性另记。
第八阶段计算分配31分36秒、4.2133 allocated CPU-hours；第十阶段七次作业累计
1458秒，合计55分54秒、7.4533 allocated CPU-hours，在原单节点8核/16GiB/
累计2小时范围内，余额3846秒；货币费用未知。原年度 NPP 科学失败仍保留。
第九阶段已核对实际 producer 的 GPP 掩码、控制、单位、日累计和清零路径，
复用保存值分析 Day337–343 与359/364；选定窗口最早可见的共同边界仍是
Day337 的NPP入口，GPP与生物量/分配/维持呼吸输入均已不同。第十阶段已完成
JAX Day337 的48步真实捕获及自身原轨迹17项严格比较。当天48步的humrel均为0，
低于1e−8活动阈值，光合活动全为FALSE、半小时GPP全为0；归一化/有序累计及
NPP/send/reset核对通过。Fortran第七次job124444469已完成原cold-start365日模型，
最终调度FAILED来自运行后的重复观测解析拒绝。独立恢复核验已将48步和两侧日界
逐项对回原始文本；96对重复值、数字文本及来源行号完全一致。原始自身比较33/34的
唯一年度GPP失败确认为默认文本导出精度问题；现存NetCDF高精度读回确认四通量日值相同，
原检查器和严格限下34/34通过。48步跨侧分析已完成，唯一活动/GPP分支差在
JAX16136/Fortran16137；该步humrel的门槛距离分别为−1e-8和7.673888101763685e-6。
去重后的1个时刻完成两次完整65参数真实输入重放，均通过；Fortran输入的8个有效
叶层及两类根候选严格通过，GPP最大误差2.9869266926163215e-15。
父级工件核验47/47通过；最后报告复用已完成journal，新增调用0。
第十一阶段已用保存值及成对源码追溯该链：窗口内最早可见的上游应力差异在
offset7（JAX16135/Fortran16136）的STOMATE入口vegstress；实际TRANS humrel首次
差异在下一步offset8。入口字段虽名humrel，实际是HYDROL vegstress，不能混用。
两步的非零值之比符合有条件的0.8源关系，但关键步的原生液水、根、mask及写读桥
未保存，尚不能确认具体根因。下一唯一优先观测是offset7 HYDROL创建/写入→offset8
DIFFUCO读取，保留48步邻域；第十一阶段已准备字段、插入点、命令和资源方案。
第十二阶段两侧原cold-start365日捕获均完成。JAX直接Stage10自身准入11990项、
另行绑定Stage6的365日四通量1460项、原生链614项，以及Fortran自身准入7152项均通过；
266项配对核对通过，分析新增模型/核调用0。offset7/tile4/layer6中Fortran原生液水略高于
共同萎蔫阈值，创建humrel=7.683888101763685e-6，JAX仍低于阈值并创建0；下一步TRANS
精确读取各自前一步值。两侧进入土壤更新前液态水已不同，剩余边界在创建器上游状态/历史，
不能据此认定同输入土壤求解器缺陷或年度首次分歧。Fortran三次分配共1247秒、余额2599秒，
无pending作业；原年度NPP科学失败继续开放。
第十三阶段只读既有保存值和成对源码，新增118项离线profile核对通过；两侧各自
offset6→7、7→8的`mc/mcl`均精确连续，JAX前一日末与下一日初状态也精确一致。
offset7第6层积分液水差主要来自相邻第5节点；选定观测的`mc=mcl`且冻结profile为0。
该结果约束这些边界上的净状态丢失及仅液水/冰水解释，尚未定位更早水分历史差异的起点。
本阶段新增模型、核/Oracle调用、编译、SSH调查和分配均0，余额仍为2599秒。
第十四阶段完成日末状态、三文件restart和内存跨年的计时、日累计及季节记忆有界审查。
正常完整restart未发现所选状态丢失；确认tau读取helper的条件性默认值不符：
有效长期温度下，未初始化tau回退3天而非配置的1095天。缺字段先被严格reader拒绝，
sentinel字段则可进入该回退；正常writer产生的有效tau不受此回退影响。未修改生产源码，
未执行新模型或第十三阶段下一捕获；该条件问题不作为319年度NPP偏差的原因。
第十五阶段已修正tau helper的天数默认值，生产入口复用既有季节尺度配置解析并将
天数上限传入该helper；缺长期温度时tau=2的源优先级、有效tau和显式上限保持原语义。
相关reader检查及真实三文件入口/首次状态选择回归31/31通过；未执行模型积分，
原年度NPP失败继续保留，具体输入触发范围见第十五阶段。
第十六阶段审查公共点/年入口、forcing/context、compiled消费者及输出归属；
当前有界证据未形成新的绑定反例。一次替身接口捕获42个日界、869项断言通过，
真实模型/编译/科学核调用均0；当时实际used-file全部静态key的跨点分母尚未验证。
第十七阶段已补齐原669点实际used/case/effective三方全键比较：并集3308键，
仅四个域边界和五个既定校准键点间变化，均进入已审查的动态消费者路径；未确认
新的静态复用缺陷。旧完整块/scan/年度/七点长期结果的成功producer与输入收据
仍缺失，保留原通过范围而不升级为当前整链认证。基础输入/缓存/证据排查可在本批
停止；完整块真实XLA数值、代表点科学失败及669点整体科学门槛仍分别保留。
第十八阶段完成DIFFUCO、ENERBIL、THERMOSOIL、allocation、phenology、turnover
六包及实际生产跨过程联查，确认THERMOSOIL显式步漏传原水文层数的组件缺陷；
只交付建议补丁，未修改生产。paper的11/32层形状条件存在，实际深层相变触发和
年度影响未证，不能据此解释原319年度NPP失败。
第十九阶段已按验收结论只补传原nslm，新增独立wrapper回归先复现旧失败，修复后
9项新回归及2项既有窄检查通过，主控验收完成，独立修复提交为 `a117cd5`，登记共享修复C5。
第二十阶段完成其余碳过程六包及生产装配审查，主控确认TF干沉降时钟、DIC土壤呼吸
单位、raw密度归一、runoff cutoff装配和扩散峰值赋值顺序五项有界差异，该审查阶段未改生产。
扩散项按原Fortran执行顺序判定，源算法在合成条件下的负库存局限单列；实际轨迹和
年度影响仍未证。第二十一阶段已完成五项最小修复及49项相关检查，主控已验收：
C6–C9提交`27ff14b`，C10独立提交`225cb95`，共享修复登记保留适用域和待同步状态。
局部同输入闭合不能确定全年NPP因果。
第二十二阶段已完成HYDROL、显式雪、CONDVEG及真实普通/compiled装配六包有界审查，
汇总六项源/接口差异和三个条件候选，主控已验收其有界证据；本阶段生产、测试和配置未改。
雪潜热常数333700与当前实现/旧harness的333600差异已由真实雪组件复现，其余发现
分别受tile、冻结、TOPMODEL配置、非默认参数或状态选择条件限制；均不建立年度归因。
完整main有限剩余矩阵与独立XLA/319/669科学门槛分列，详见第二十二阶段。
第二十三阶段六项最小修复已完成主控验收，登记为C11–C16，四组代码提交见该节末尾；
最终定向组合72项通过。三个S22候选、年度NPP失败和研究分支待同步状态保留。
第二十四阶段有限剩余六包审查已获主控有界验收：非冻结bin索引、末态冻结
refresh、TOPMODEL潮汐选择和非默认SNOWCRI参数四项取得有界组件数值证据；fwet覆盖
保留为生产触发未准入的条件状态观察，五处检查/默认值差异仅按静态契约登记。
旧雪maxmass跨层普遍不可达证明撤回为条件域未证明，未确认新的maxmass算术缺陷。
本阶段没有生产修复；主控已补齐C1–C16累计登记，并修正三条雪分支的不可达登记。
下一批按源码依据实施最小修复，SNOWCRI先补有效配置门控；未重跑Stage23，
不据此宣称年度归因或完整main通过。
命名日值 trace 的具体历史文件身份已取得；两条长期链 producer 和年度因果
仍未全部核定，尚未确认该年度差异的源级原因。**
已有通过结果保留；旧结果的身份关联、当前科学失败及未核定的原因分别记录。
本页记录有界盘点、319 历史身份、当前首年对照、局部来源及晚年保存证据核查，
不是全模型重新验收，也不改既有科学合同。

## 固定基线和支持范围

- cwd / Git root：`E:/ORCHIDEE-MAN-JAX`；分支 `main`。第一阶段基线
  `ee42eb14058a028c697147a5651912d9b8eb75e3`，已接受于
  `cd9c6b8022061a6e6e480de2e825ab28e72eac68`；第二阶段从该提交开始。
  前两阶段开始时工作区均干净；旧数值产出不赋予这些审查提交的 producer 身份。
- 第三阶段使用第二阶段接受提交
  `09968cf86551e8dfc039c1740b94fb3d42cf3f12`；开始时工作区干净，
  实际 runner/source、输入和执行身份见
  [identity.json](../../outputs/reviews/main-physics-20261011/319-current/identity.json)。
  这是新执行的 producer 身份，不是旧 A/B 产出的 producer 身份。
- 第四阶段从第三阶段接受提交
  `a8e92e6646a1d6d08f7bcd2beeee080ab88e5a4c` 的干净 main 开始；生产源码与
  第三阶段数值 producer 相同。新的 cold-start day198 前缀与同年 day199 回放
  使用该当前身份；旧 day198 checkpoint 的原 producer 仍未知。
- 第五阶段从第四阶段接受提交
  `b7167b72169d39269b43ce8139abb345043817b0` 的干净 main 开始；只读保存值，
  生产源码仍与第三、四阶段相同，未执行新模型或 trace。
- 第六阶段数值捕获 producer 为
  `c7110e030bae21512538b1ae45add8d0317ce0d7`；第七阶段从已接受报告提交
  `103ce03887924eec7ae8e5500b454353016df95a` 的干净 main 开始。
  两提交之间仅报告改变；本阶段只分析保存值和源码，未生成新模型轨迹。
- 第八阶段从第七阶段接受提交
  `eb8659e2c1bf3b8244a6a31a78d996d826c2a312` 的干净 main 开始；实际导入仍在
  此 checkout。四份生产源码 SHA、36 份冻结输入 size/mtime 和 run.def SHA
  均与第六阶段匹配，见
  [jax-plan.json](../../outputs/reviews/main-physics-20261012/319-matched-capture/jax-plan.json)。
  本阶段最初准备有两组人工 kernel 预检；该准备快照没有真实模型轨迹。
- 本次恢复执行从已接受的本地准备/认证阻塞进度提交
  `9c6ef25a1d4c9179541398108815190f44033a55` 的干净 main 开始；cwd、分支和
  实际 module 导入路径再次确认。本轮新增认证后的有界只读取证及实际源码
  日志副本；未重跑原16项准备、9项收尾核验或第4–7阶段调查/模型。
- 本次真实匹配执行从已接受的 `e05369c9e70b4c982f34545e37af7c1782d8142d`
  续作，仍在固定 main；HEAD 未改变。Fortran/JAX 实际执行身份、完整捕获和
  同输入比较见第八阶段；旧准备回执保留原历史计数。
- 第九阶段从已接受并推送的
  `85ace286931eef30551cb670f980fd383090820c` 的干净main开始；实际cwd、分支和
  module导入路径确认。相关生产树与第八阶段数值producer相同；本阶段只读取
  命名实际Fortran来源和现有轨迹，不生成新模型年、prefix或科学回放。
- 第十阶段从已接受并推送的
  `638253db350314becce8eeb8089f3e5c8b68ae6c` 的干净main开始；cwd、分支和实际
  module导入路径确认。仅点319、1961、PFT14、Day337观测，保留原cold-start
  365日调度及JAX block=7。执行/来源身份见
  [执行合同](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/execution-contract.json)和
  [实际JAX命令](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/jax/jax-run-attempt-3/command.json)。
- 第十阶段Fortran恢复从JAX结果/本地恢复准备的接受提交
  `3f906c627c1649bf1bc677ca5977acdce93bf42f`的干净main继续；实际module导入仍在
  本checkout。相对`638253db`仅本页报告改变，历史JAX数值身份不重标；
  [恢复回执](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/access-restored-20261014.json)
  保存已有授权、初始剩余5304秒及图覆盖限制。
- 第十一阶段从第十阶段已接受并推送的
  `6e9a60311d994f6fef1098d9f10b1ba461da1477` 的干净main继续；cwd、分支及实际
  module导入路径确认。相对数值producer `638253db` 仅本页报告变化；只分析保存值、
  当前精确源码和命名producer的只读来源，未新增积分、生产函数重放、构建或分配。
  新证据集中在[Stage11入口](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/README.md)，
  第十阶段目录保持只读。
- 第十二阶段从已接受的`91a1a79b3bf021f7fcae38e668c62d88f52c8d6a`的干净main开始；
  cwd、分支及实际module导入再次确认。后续授权覆盖观测器实施及两侧原365日轨迹捕获，
  第十一阶段的准备态限制不用于禁止这次执行。全部新实现/产物限于
  [Stage12目录](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/README.md)，
  仅本页允许版本化修改；旧第十/十一阶段工件只读，不暂存、提交或推送。
- 第十三阶段从已接受的`6f9d9f9340995376c7f79779072ee00b66485bb8`的干净main开始；
  cwd、分支及实际module导入路径确认。只读第十二阶段保存值和本地成对源码，所有
  新工件位于[Stage13目录](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/README.md)，
  仅本页允许版本化修改；不新增积分、生产核重放、构建、SSH调查或分配。
- 第十四阶段从第十三阶段已验收并推送的`847a5f35797613e9b6778810566187c7e4ccb9f7`
  的干净main开始；cwd/分支/实际导入路径再次确认。新工件限于
  [Stage14目录](../../outputs/reviews/main-physics-20261018/restart-year-state/README.md)，
  仅本页允许版本化修改；旧Stage12/13只读，不暂存、提交、推送或跨聊天发消息。
- 第十五阶段从第十四阶段已独立提交的`e5032c3054dede6ed27275badd069a61ebd6c27c`
  的干净main开始；cwd/分支/实际导入路径确认。新工件限于
  [Stage15目录](../../outputs/reviews/main-physics-20261023/restart-tau-fix/README.md)，
  版本化修改限于tau reader、生产restart配置传递、相关过时docstring、两份直接回归
  及本页；旧Stage14与之前工件只读，不暂存、提交、推送或跨聊天发消息。
- 第十六阶段从C4及文档登记已接受的`6f20117badb229dc4e871b82d7cefd586d3dd258`
  的干净main开始；cwd/分支/ORCJAX实际导入路径确认。新工件限于
  [Stage16目录](../../outputs/reviews/main-physics-20261023/compiled-input-binding/README.md)，
  仅本页允许版本化修改；只审查公共入口、输入绑定、compiled块和输出归属，
  旧Stage14/15工件只读，水分起源捕获继续暂缓。
- 第十七阶段从第十六阶段主控验收后的`dd32b283188f650a3f7012fc1ee637a5be3a2487`
  干净main继续；cwd/分支/实际ORCJAX导入路径已确认。连续完成实际输入分母、
  旧证据适用范围及消费者/缓存对应三包；新工件限于
  [Stage17入口](../../outputs/reviews/main-physics-20261024/input-and-evidence-closure/README.md)，
  仅本页版本化修改，不修改生产源码、测试、配置、容差或其他工作树。
- 实际导入：`E:/ORCHIDEE-MAN-JAX/jax_orchidee/__init__.py`；ORCJAX、
  Python 3.11.15、JAX 0.10.1。本轮 Python 均使用 `-B`。
- 范围由 [README](../../README.md)、[交付合同](pft14_equivalence_delivery_contract.md)
  和 [paper 配置](../../configs/orchidee_man_250919.yaml) 确定：PFT14、
  `paper_250919_legacy14`、独立单点 paper 生产路径、noleap、48 半小时/日。
  单点 paper 拼接不是 `nbp_glo > 1` 的河网运行；其他 PFT/未支持配置不在等价承诺内。
- C1 `db4dd8f1cc86a33dc8e343725fda2803f63fb358`、C2
  `e317fa81b97b8b8c959f2e7dfdfba525bb54dec8`、C3
  `2a3e303845154576c6f4b5c0b8155106e5bcc70b` 已接受。
  [共享核心报告](shared_core_branch_review.md) 的 14 / 12 / 20 项回归、
  C2 极小分母导数策略和 C3 已记录的 FP64 舍入差异继续有效，本轮没有重复运行。

## 现有覆盖与证据地图

以下地图保留第一阶段结果，区分“档案已证明的范围”与“第一阶段核实的范围”；旧产出不重新标记为当前基线产出。
完整路径、账本条目和历史源码变更在
[盘点记录](../../outputs/reviews/main-physics-20261011/evidence-inventory.json)。

| 层次 | 命名证据与已有结论 | 证明范围 / 本轮边界 |
| --- | --- | --- |
| Fortran 源分支 | [reachable ledger](pft14_reachable_ledger.yaml) 为 52 个过程条目：37 active、11 conditional、4 inactive；[evidence scope](pft14_evidence_scope.yaml) 将全部 48 个可达条目列为 production_verified | 52 是过程条目数，不是 IF/WHERE 分母。保存的 [certificate](../../outputs/reference_mode/pft14_equivalence_certificate.json) 记录 5,698 sites、11,085 arms 已分类、4,580 scientific arms 有证据、260 owner regions 通过。本轮读取记录，未重新扫描整个 Fortran 分母。 |
| 独立数值 Oracle | [micro-oracle index](../../outputs/reference_mode/micro_oracles/index.json) 记录 73 个原始过程抽取、26 个 family；scientific arms 的证据路由分别含 1,243 oracle+production、2,020 owner transition、513 state IO、804 source proof | 原源码抽取本身不是执行证据；source proof、状态/IO 合同也不能泛称为每个输入的数值 Oracle。HYDROL 本轮核对了一个原始 Fortran 文件身份、命名比较和边界源码；其他 family 复用已有记录。 |
| 生产过程/状态 | [HYDROL/THERMOSOIL](../../outputs/reference_mode/production/hydrol_thermosoil/comparison.json)、[STOMATE](../../outputs/reference_mode/production/stomate/comparison.json)、[lifecycle](../../outputs/reference_mode/production/lifecycle/comparison.json) 均记录 passed / production_execution | 有独立 Oracle 接入、同一步输入/输出及状态写回证据；HYDROL runtime witness 为 tstep0→tstep1，lifecycle 为 1961 Day1/Day2 与 1962 Day1，不能单独替代整年/669 点验收。 |
| restart / 年接续 | [stage5](../../outputs/reference_mode/stage5_lifecycle_acceptance.json) 的 19 个 checks 通过；[restart lifecycle](../../outputs/reference_mode/restart_state_lifecycle.json) 181 字段闭合；[三点矩阵](../../outputs/performance/compiled_fast_path_acceptance/final_3point_annual_compiled.json) 覆盖 1961 全年、完整末日状态、1962 直接及三文件 split restart | 三点为 001、069、319；严格/compiled 比较与 Fortran 科学比较是不同证据。矩阵记三点 annual_scientific_tolerance_pass=true，但不证明三点各自 50 年序列均通过。 |
| 完整七日块 / SECHIBA scan复用 | [14-day cold / restart、365-day 记录](../../outputs/reviews/main-physics-20261011/block-gates.json) 均 passed；[跨点 reuse](../../outputs/performance/compiled_fast_path_acceptance/reuse_3points_current.json) 的三点SECHIBA scan strict parity与callable/executable复用通过 | cold 对 strict source order；restart/annual 对 compiled daily no-block，浮点 `1e-8 + 1e-10*abs(ref)`、离散/模式精确。三点scan复用不是当前完整日块factory跨点XLA复用的直接记录。公共 [CLI](../../jax_orchidee/cli/main.py) 注入 on / block=7；底层 multiyear 的 block=0 默认不能据此判为公共 CLI 未启用。 |
| 七个新点长期科学证据 | [七点复核记录](../../outputs/reviews/main-physics-20261011/longrun-evidence.json) 指向七个原始 `compiled_1961_2010.json` | 103.0-095.0、225.0-101.0、281.0-093.0、287.0-111.0、293.0-113.0、295.0-081.0、331.0-057.0；每点 50 年、每年 365 日、年度 truth 完整、cold-start、block=7、NumPy accumulator off。用当前 acceptance reader 读取保存值，七点均通过原年度及序列科学门槛；没有重新运行模型。 |

七点 2010 最大 **原报告 `annual_reference_delta.rel_error` 指标**为
`8.498008041180061e-5`。其定义经
[`multiyear._delta`](../../jax_orchidee/runners/multiyear.py) 151–159 核实为
`abs_error / max(abs(reference), 1.0)`，不是无下限的标准相对误差。
例如 331 点 NPP 的 reference 为 `0.7658335566520691`，abs_error 和所记
rel_error 均为 `8.498008041180061e-5`；`abs_error/abs(reference)` 约为
`1.109641640453847e-4`。七点 28 个终年记录均符合前述报告定义。
[科学政策](paper_scientific_acceptance_policy.md) 仍用
`abs_error <= 1e-5 + 1e-3*abs(reference)` 以及原 normalized bias / RMSE
`<=1e-3`；本轮没有用报告 rel_error 替换门槛。

## 有界核实：HYDROL 动态根与新执行年

选此对象是因为 [7 月 11 日多点报告](multipoint_validation_20260711.md)
曾将 dry/snow 高风险结果与根状态关联，且根同时影响 HYDROL 水分胁迫与 STOMATE
维持呼吸。先核对条件、数组索引和状态时刻，再判断外观上的疑点。
实际源码摘录见 [bounded-source.txt](../../outputs/reviews/main-physics-20261011/bounded-source.txt)，
数值/身份记录见 [hydrol-evidence.json](../../outputs/reviews/main-physics-20261011/hydrol-evidence.json)。

| 状态 | 核实结果 |
| --- | --- |
| 已排除：动态根未实现 / 各 soil tile 使用独立根而丢失 SAVE 顺序 | Fortran `hydrol.f90` 283 为共享 SAVE nroot，`hydrol_soil` 6584–6635 在各 tile 上更新；JAX `hydrol_module_diagnostics` 7856–7916 逐 tile 将 `stress.nroot` 写回 `nroot_work`。该模块函数 AST 与 7333b46、deba027、5910ba9 均一致。 |
| 已排除：树的 tile6 阈值不同就是错误 | Fortran 6594–6597 的树根更新仅对 tile4/5 用 peat 阈值，而 6660–6666 的旧 water stress 包括 tile6；JAX 7113–7140 保留这个区别。JAX tile 为零基循环、传入 Fortran 编号 `jst+1`；nroot 是无量纲归一化层份额。 |
| 已排除：所有微小根候选都应受 min_sechiba 门槛，或 new_watstress 应改变根更新公式 | Fortran 6601/6621 为 `SUM(nroot_tmp)>0`；6648 才选择 new_watstress。JAX 7145–7154 对任何正和归一化，零和保留旧根；相应三项边界回归定义在 `tests/unit/test_hydrol_diagnostics.py` 181–293，本轮未重跑。 |
| 已排除：跨年删除 nroot 本身就是丢失 restart 状态 | 本轮完整核对 `hydrol_finalize` 1682–1797：写 `us` 等但没有 nroot；`hydrol_var_init` 4097–4137 重建根并按 ALT 掩码归一化。paper `Job0_bio` 313–342 每年重新执行并滚动三文件。JAX `rebase_driver_state_for_year_start` 2721–2755 丢弃年内 nroot，11517 的实际新年调用使用该 rebase；这与年内保留共享根的合同不同。 |
| 已复核的独立通过证据 | `hydrol_soil_owner_cwrr_water_root_peat_alt` 的 [502 项比较](../../outputs/reference_mode/micro_oracles/hydrol_soil_owner_cwrr_water_root_peat_alt/comparison.json) 全部通过；其中 nroot/humrel/vegstress 共 105 项、最大绝对误差 `2.220446049250313e-16`。其 [coverage](../../outputs/reference_mode/micro_oracles/hydrol_soil_owner_cwrr_water_root_peat_alt/branch_coverage.json) 记录该 family 所要求的 109/109 arms。Fortran 文件 SHA256 与本地一致：`39fa6d4bf4385c89a4370d75d314b07adf43fc1462b0c003d61d4c4dc3f20577`。 |
| 证据待补 / 尚未判断 | 当前 `hydrol_water_stress_diagnostics` AST 与 5910ba9 一致，与 7333b46 不同。[精确差异](../../outputs/reviews/main-physics-20261011/hydrol-forward-guard-delta.patch) 中，根归一化新增 safe denominator：正和仍是源商，非正和仍保留旧根；不能把 AST 不同等同于前向物理错误。但旧 Oracle JSON 未内嵌 JAX producer commit，本轮没有据此宣称该 JSON 是 ee42eb1 的新执行证据，也没有新验收全部梯度或 root 状态输入域。 |
| 确认实现不符 | **无。** 本轮没有得到一个经来源、可达输入和现有容差确认的当前 main 物理反例。 |

## 第二阶段：319 两条历史 50 年链的身份和后续证据

本阶段只调查 `319.0-057.0` 的精确命名结果、相邻命令/日志/checkpoint、
验收目录内同点 JSON、命名 owner/state 记录及必要的 Git 源片段。
详见 [身份与比较记录](../../outputs/reviews/main-physics-20261011/319-history/receipts.json)
及 [源码摘录](../../outputs/reviews/main-physics-20261011/319-history/source-excerpts.txt)。
第一阶段 HYDROL/年边界证据继续复用，未重跑 Oracle、回归或模型。

| 证据项 | A：wetdiaglong_effective_319 | B：12point_wetdiaglong_fix |
| --- | --- | --- |
| 原结果 / 实际窗口 | [compiled_1961_2010.json](../../outputs/acceptance/wetdiaglong_effective_319_1961_1981/319.0-057.0/compiled_1961_2010.json)；1961–2010，每年 365 日 | [compiled_1961_2010.json](../../outputs/acceptance/compiled_1961_2010_12point_wetdiaglong_fix/319.0-057.0/compiled_1961_2010.json)；1961–2010，每年 365 日 |
| 首个原科学超限 | 1961 NPP；JAX `0.3860042396860273`、reference `0.38518497347831726`，abs_error `8.192662077100565e-4` > 原 limit `3.951849734783173e-4` | 1963 NPP；JAX `-0.013051838239321791`、reference `-0.013333331793546677`，abs_error `2.8149355422488545e-4` > 原 limit `2.333333179354668e-5` |
| 全链原科学门槛 | 失败；48 个年度变量值超限，涉及 30 年；NPP normalized RMSE `0.0020904160262748147` | 失败；128 个年度变量值超限，涉及 45 年；四变量序列统计均未过，NPP normalized RMSE `0.016239404396783952` |
| 实际接续 / 命令证据 | 明确记录从 1981 年末 checkpoint 恢复；50 年文件前 21 个 year records 与 [21 年文件](../../outputs/acceptance/wetdiaglong_effective_319_1961_1981/319.0-057.0/compiled_1961_1981.json) 及 [1981 summary](../../outputs/acceptance/wetdiaglong_effective_319_1961_1981/319.0-057.0/compiled_checkpoints/multiyear_partial_through_1981.json) 逐记录完全相同，随后列出 29 个新 checkpoint。相邻 worker stdout/stderr 为空，monitor stderr 只有 CLIXML 标头；未找到完整原 argv。 | [workers.json](../../outputs/acceptance/compiled_1961_2010_12point_wetdiaglong_fix/workers.json) 留存四分片请求：cold-start、1961 起 50 年、`--strict-on-failure off --force` 和匹配输出目录；319 在 selection 的零基位置 7，按所核对 runner 的取模规则属于 shard 3。这是相邻命令记录，尚非文件级 PID/source receipt。 |
| producer / 有效版本结论 | **精确 producer 未核定**；prefix 与 continuation 均无 Git/source 身份，不能据恢复运行推断源码改变或不变。 | **精确 producer 未核定**；命令未记录源码 SHA、源码 overlay 或 Python 环境，不能把 worker PID 当成生产版本。 |

两条 `compiled.stdout.log` 的末年记录均与各自原 JSON 完全一致；ready 只证明
窗口完成，不是科学通过。它们共享记录的 config 和 materialized `run.def` 路径，
点位为 `I5/S26_69.834_0.0019_0.3980_97.918`，CSV target 为 2005。
两条保存的 50 年 reference modelout 值全部相同；本阶段另读当前本地点位
1961、1963、2005、2010 NetCDF，四变量均与保存 reference 精确一致，当前
reference PFT index 为 13。当前 materialized 内容与当前纯物化结果相同，
`OK_WETDIAGLONG=FALSE`、`dyn_nroot_larix=TRUE`、`NEW_WATSTRESS=FALSE`。
这些是当前输入/参考的核对结果：旧 JSON 只保存路径，未冻结当时 forcing、
配置内容或环境身份，不能由目录中的 `fix/effective` 名称确认当时有效开关。

Git 的初始快照 `7333b46` 已无条件输出 `compiled_day_block_size`，当前 runner
还输出稳定 PFT 元数据；两条旧 JSON 均缺这些字段。它们的格式不吻合所核对的
这两个未修改 runner，不能直接归到任一提交，也没有足够本地记录给出一个确定
的候选提交集合。已核对源码中 `_evaluate`、`_delta`、物化 helper 和 acceptance
命令构造从 `7333b46` 到第二阶段基线 AST 相同；annual reference reader 的
稳定 PFT 选择改动有明确源提交 `87f1f50`，当前仍解析到历史 PFT14 的 index 13。
这些源提交身份不是旧产出的 producer 身份，源码相同或年度数字相同也不能代替它。

有界目录调查共找到三条同点、cold-start、1961–2010 的完整候选链；除 A/B 外的
[local_50year_pilot](../../outputs/acceptance/local_50year_pilot_20260713/319.0-057.0/319.0-057.0/compiled_1961_2010.json)
也失败，127 个年度变量值超限，首个为 1961 NPP。日期目录不能证明执行先后。
在列明的本地证据集合内，**未找到可确认替代 A 或 B 的同点、同科学合同、完整
50 年实际通过结果**；这不是对全部项目、服务器或用户档案的不存在断言。

反证和范围限制一并保留：

- [wetdiaglong_fix_319 的 1961 单年](../../outputs/acceptance/wetdiaglong_fix_319/319.0-057.0/compiled_1961_1961.json)
  及同名 12 点单年结果通过，且年度记录与 B 的 1961 相同；B 的全链仍在 1963
  首次超限。单年改进不能关闭长链，也不能凭相同数字认定生产源码相同。
- [2010 Fortran restart 单年](../../outputs/acceptance/pilot_2010_restart_2points/319.0-057.0/compiled_2010_2010.json)
  通过，但它从下载的 Fortran state 起跑；三点矩阵的 1961 全年及 1962
  两日 direct/split restart 通过也保留。它们均未证明 cold-start 50 年链。
- 旧 stage6 / local 1961 的 strict 与 compiled 均科学失败，且 compiled 年度记录
  与 A 的 1961 相同；这是保存单年 counterevidence，不能扩写为两条 50 年链的
  strict 分类。B 的相邻命令明确关闭 strict-on-failure，未找到匹配全窗口 strict 结果。
- [2009 state 比较](../../outputs/reference_mode/diagnose_069_319_2009_checkpoint_vs_fortran_start.json)
  直接命名 B 的 2009 checkpoint；旧 maintenance 记录在 `1e-10` / `1e-6`
  诊断容差下分别报 day 173 / 200 首偏差，但没有连到 A/B 的源码身份。
  这些 owner/state 线索不是当前 main 反例或完整修复证据；七个新点也不包含 319。

第二阶段没有确认当前 main 物理 bug，也没有确认这两条旧失败已被有效替代。
最小后续步骤是取得命名原执行的 source/manifest；若无法恢复，才以明确当前 SHA
和输入身份做 319 cold-start 首年、必要时 carried 1961–1963 的有界比较，复用
对应 owner/state 线索且保留原门槛。最终声称长链关闭仍需有身份的 1961–2010
完整实际通过记录。第二阶段没有启动这些运行；第三阶段实际执行范围见下节。

## 第三阶段：有身份的当前 319 cold-start 首年

新产出独立保存在
[319-current](../../outputs/reviews/main-physics-20261011/319-current/)。
点位为 `319.0-057.0/I5/S26_69.834_0.0019_0.3980_97.918`，
稳定 layout 为 `paper_250919_legacy14`、mangrove index 13；1961、365 日、
48 半小时/日、cold-start，无旧 checkpoint 输入。
`used_run.def` 为当前 materialized 文件的逐字节副本；配置也保存了内容副本，
实际仍从 main 的原配置路径读取，以保留 source reader 的根目录语义。
manifest 保存 36 个配置候选输入/参考文件的 SHA256、大小和时间，合计
3,305,915,411 bytes；该集合不声称每个 static asset 都被实际消费。
环境为本地 ORCJAX / Python 3.11.15 / JAX 0.10.1 / CPU，实际导入路径如上。

compiled 使用 on / block=7 / static daily-carbon JIT on，
NumPy accumulator 和 single-pass fold off，module/local DIFFUCO JIT on，
prebuild day payloads on；显式请求全部 15 个 science/diagnostic fields。
同身份 strict 使用 compiled off / block=0 / static daily-carbon JIT off，
其余输入、冷初始化和窗口相同，module/local DIFFUCO JIT 继续 on，
与既有 failure-classification 路径一致。
实际 argv、源码/输入运行后检查和结果见
[compiled receipt](../../outputs/reviews/main-physics-20261011/319-current/compiled-1961.receipt.json)、
[strict receipt](../../outputs/reviews/main-physics-20261011/319-current/strict-1961.receipt.json)；
完整命令与前置 helper 失败见
[命令记录](../../outputs/reviews/main-physics-20261011/319-current/commands.md)。

| 1961 年度变量 | compiled JAX | strict JAX | Fortran reference | compiled abs_error / 原 limit |
| --- | ---: | ---: | ---: | ---: |
| AGB_model | 4.591235924144104 | 4.591235924144387 | 4.590915203094482 | 0.00032072104962121983 / 0.004600915203094482 |
| BGB_model | 4.120666870505465 | 4.120666870505813 | 4.120461463928223 | 0.00020540657724232858 / 0.004130461463928223 |
| GPP_model | 0.7776216017113179 | 0.777621601711987 | 0.7776592373847961 | 0.00003763567347825347 / 0.0007876592373847962 |
| NPP_model | **0.3860042397133881** | **0.3860042397139073** | **0.38518497347831726** | **0.0008192662350708368 / 0.0003951849734783173** |

compiled 完成 365 日、15 字段 count 均为 365、末步 17519，
无 missing/stopped component，JSON 所列科学/诊断数值均有限。
runner 年度耗时 275.2018044000142 s，wrapper 含结果/状态核验
276.4121996999893 s。只有 NPP 超过原年度科学门槛；
单年 normalized bias / RMSE 均为 `0.0021269423562208477`，
同样超过 `1e-3`。更严的 numerical diagnostic gate 也失败，单独保留。
strict 同样完整完成 365 日、15 字段，所列数值有限，runner 年度耗时
956.6834202000173 s，wrapper 含核验 957.554574199981 s；
NPP abs_error=`0.0008192662355900326`，normalized bias / RMSE
均为 `0.0021269423575687608`，只有 NPP 未过年度科学门槛。
两次 code=0，运行后 source/input metadata/config/run.def unchanged 均为 True，
当前 live 1961 reference 与保存 reference 精确相同。
四个年度 JAX 值的 strict/compiled 最大差约 `6.692e-13`（GPP），
NPP 差约 `5.192e-13`。按原政策分类为
**`shared_semantic_input_or_reference_mismatch`：共享语义、输入绑定、汇总或参考差异待定位**。
这不是 compiled-only 退化，也尚未证明共享物理过程实现必有 bug；
当前科学超限本身没有定位物理函数缺陷。完整数值核验见
[analysis.json](../../outputs/reviews/main-physics-20261011/319-current/analysis.json)。

**完整状态有限性与科学输出有限性分别记录。** 两份 checkpoint 的原始全树扫描各
为 351 个 numeric leaves、23,992 个元素，发现两个 NaN，故 raw state finite=false：
`diffuco_previous_step_state.roughheight_pft` 与
`sechiba_finalize_state.roughheight_pft`，均为 shape `(1,14)` 的 `[0,0]`。
该位置是 Fortran PFT1 / bare_soil，layout 的 active_mask **确为 True**、
prescribed fraction 为 0，不能按全局 inactive mask 排除。
实际 `IMPOSE_AZE=FALSE`、`ROUGH_DYN=TRUE`；动态 producer
`condveg.py` 364/366/417–419 保留 bare slot NaN，只写后续 PFT，
对应 `condveg.f90::condveg_z0cdrag_dyn` 1707–1718 的 `DO jv=2,nvm`。
Fortran `diffuco_aero` 1067–1069、1107–1118、1128–1134 同样从 PFT2 开始，
PFT 特定 drag 还受 `ok_LAIdev` 控制；本地 14 个 `OK_LAIDEV` 均为 False。
JAX `diffuco.py` 4168–4177 在运算前用 grid roughheight 替换未消费坐标，
4227–4233 选择 grid drag；orchestration 2936–2937/3131 保留该 sidecar。
因此这两个特定未定义坐标是当前有界 producer/consumer 路径的合法哨兵，
不是已确认的 NaN 传播 bug；原始 checkpoint 和 receipt 保留不改。

原 receipt 的 `partial_exactly_matches_cached_summaries=false` 也保留。
pickle 的 `paper_modelout_csv_delta` 是 tuple，JSON 是 list；按结构和数值语义
比较后相同，partial summary 与结果 year prefix 相同。
这不修改原序列化格式，也不把 Python 容器类型差异当作 year-state 缺失。
两份 checkpoint 都只有上述两个坐标非有限，其余 23,990 个 numeric 元素有限。
精确坐标和 tuple/list 语义比较见 analysis；对应
[source excerpts](../../outputs/reviews/main-physics-20261011/319-current/source-excerpts.txt)
保留本阶段 producer/consumer 和 Fortran 条件。

**PEAT_HYDRO 仍有 319 配置/语义边界待核定。** raw run.def 值为 `TRUE`，
当前 JAX `orchestration.py` 68–75/630–631 则区分参数开关与 HYDROL soil 的
`PAPER_1961_HYDROL_SOIL_PEAT_HYDRO=False`，6381–6382/6987–6988
传入 soil/root/water-stress 路径。Fortran `hydrol.f90` 95 声明局部
`LOGICAL,SAVE :: peat_hydro`，该声明没有初始化器；
`pft_parameters.f90` 3626–3627 对参数开关读取 run.def。
`hydrol_main` 1278 调用 `hydrol_soil`，tile loop 的 5858/5891 调用
`hydrol_soil_coef`，8259/8290 的实际条件是 **HYDROL 局部变量为 True 且
Fortran soil tile 为 4/5/6**，不能由 raw 字符串直接决定。
现有 [hydrol v9 audit](hydrol_1961_v9_audit.md) 保存 tile4/bottom11 的 1200 个
coefficient rows，均为 `peat_hydro=F/branch_peat=F`，以及 144 个 tile4 step rows。
其配套 [设计入口](../porting/pft14_extensible_design.md) 9–10 关联
`reference/case_001_071`；CSV 没有点位/producer 字段，未建立对应 319 Fortran
执行的身份桥。第一阶段 ledger 的 False 只保留其原审计范围，不能仅凭 case001
trace 外推 319 也应 False；无初始化器的声明也不是全运行 False 的来源证明。
CSV 核验、内容身份和点位限制见
[peat-trace-scope.json](../../outputs/reviews/main-physics-20261011/319-current/peat-trace-scope.json)。
当前 pinning 的 319 有效性尚需核定，既未排除也未确认为本次 NPP 原因，
不声称所有其他 peat 相关消费者 inactive。

1961 已失败，按批准的条件与主控确认，**未继续到 1963**，也未执行 50 年或
669 点积分。现有 A/B、单年 restart、七个其他点和已接受 C1–C3 的结论继续保留；
当前数值接近历史 A 的 1961 残差不构成 producer 或因果同一性。
第三阶段提出的有限 NPP/呼吸来源核查已在第四阶段执行，见下节。
旧 [day173@1e-10](../../outputs/reference_mode/diagnose_319_maintenance_first_divergence.json)
和 [day200@1e-6](../../outputs/reference_mode/diagnose_319_maintenance_material_divergence.json)
仍仅为命名历史线索，不能称为当前首偏差日；当前新证据明确区分观测边界和状态来源。

## 第四阶段：年度分量与当前 day199 土温输入

全部新证据位于 [319-npp-local](../../outputs/reviews/main-physics-20261011/319-npp-local/)。
复用第三阶段两份全年产物，未重复全年两模式。对实际 319 NetCDF 读取的
GPP/NPP/maintenance/growth 均为 float32、`average`、`1800 s` / `1 yr`，
选择 `[0,13,0,0]`。paper extractor 直接读 GPP/NPP；`0.02` 仅用于 AGB/BGB，
GPP/NPP 的 `[0,1]` normalizer 不改值。Fortran `npp_calc` 最终为
`gpp - resp_growth - resp_maint`，后者含 biomass-floor 修正，不能混同未经修正的
日累计 root 分量。精确输出/采样源码见
[sampling audit](../../outputs/reviews/main-physics-20261011/319-npp-local/sampling-scope/audit.json)。

| 对 compiled 年均 NPP 有符号残差的贡献 | gC/m²/day | 占净残差 |
| --- | ---: | ---: |
| ΔGPP | −0.0000376356734781 | −4.594% |
| −Δmaintenance | +0.0008469708371584 | 103.382% |
| −Δgrowth | +0.0000099310713905 | 1.212% |

[分解](../../outputs/reviews/main-physics-20261011/319-npp-local/npp-component-decomposition.json)
在浮点舍入内闭合，strict 同样以 maintenance 为主要分量。它解释年度残差的
**数值构成**，不是 maintenance 公式错误的因果证明。原 NPP 超限和门槛不变。
compiled 的年度 `MAINT_RESP`（JAX − reference）为 **`−0.0008469708371584`**，
即维持呼吸偏低使 NPP 偏高；表中正值是其对 NPP 的贡献，不能误读为呼吸偏高。
命名历史 Fortran LPJ trace 恰有 365 条，四通量日均值转为 float32 后均等于当前
319 年度 NetCDF；[记录](../../outputs/reviews/main-physics-20261011/319-npp-local/saved-trace-analysis.json)
提供数值桥，仍不能补出原 binary/source 身份或 XIOS 实际发送覆盖。

先对 SHA256 `918d17fb720f95a6a22bc26ebe8289a5c6d44ca8b0c55624571de62cb1ed7ddb`
的历史 day198 状态做当前一天回放；原 producer 未记录，其重新生成的 48 次
maintenance 输入与旧 Oracle 的 binary digest 精确相同。随后按主控指令使用
**第三阶段冻结的同点同配置输入**，在当前 SHA 上执行一次既有 compiled cold-start
前缀：198 日、block=7、末步 9503，runner 375.927 s。
[receipt](../../outputs/reviews/main-physics-20261011/319-npp-local/current-prefix/receipt.json)
记录实际 argv、输入/源码身份及新 checkpoint。虽通用文件名含 `year_end`，
该状态只到 day198；下一天直接调用 `year=1961/start_tstep=9504` 的 later-day
诊断，未 rebase 或走新年 restart。day199 回放耗时 105.640 s，总计 109.257 s。
不把 198 日均值与全年 reference 比较，也不赋予旧 checkpoint 当前 producer 身份。

仅用保存值进一步做[分段诊断](../../outputs/reviews/main-physics-20261011/319-npp-local/maintenance-period-decomposition.json)。
[口径补核](../../outputs/reviews/main-physics-20261011/319-npp-local/period-definition-scope/audit.json)
确认两份 JAX `MAINT_RESP` 都取 post-NPP 的 `resp_maint`，含 biomass-floor 修正，
单位为 gC/m²/day，198/365 个日值均含首日零值；输入、模式及生产源码相同。
历史 LPJ 日 trace 的总量定义仍属暂定匹配，365 日 NPP 恒等式最大残差 `6.00e-15`，
四通量均值转 float32 均匹配 NC；这些支持数值诊断，仍不恢复原运行身份/发送覆盖。
前段 JAX 总量为前缀均值×198，后段为全年均值×365 减前段，Fortran 两段直接加日值：

| 区间 | JAX 累计，gC/m² | 历史 Fortran trace 累计，gC/m² | JAX − trace，gC/m² |
| --- | ---: | ---: | ---: |
| Day1–198，独立前缀 | 37.622801792519 | 37.622801571907 | `+2.206119376e-7` |
| Day199–365，**差分推得** | 46.304479981856 | 46.613624362209 | `−0.309144380353` |

全年相对 trace 的累计差为 `−0.309144159742`；后段占净负差约 **100.0000714%**，
前段微小正差抵消一部分。后段不是新增逐日观测。维持呼吸 trace 均值减 float32 NC
为 `−5.364967082e-10 gC/m²/day`，乘365后的累计差为 `−1.958213005e-7 gC/m²`；
加入此舍入差即得到原 NC 基准的全年累计残差 `−0.309144355563`。
`fsum` 与既有 NumPy trace 均值差仅 `5.55e-17 gC/m²/day`。

两份运行的 Day1 单独执行，Day2–197 为28个相同分块；前缀 Day198 走单日 runtime，
全年则在 Day198–204 七日 compiled 块内。令 ε 为全年原轨迹前198日总量减独立前缀
总量，则所报前段差为原前段差−ε，所报后段差为原后段差+ε。保存的全年摘要没有
Day198 日值/前198日总量，**无法给 ε 数值上界**；旧历史 checkpoint 比较不能充当它。
均值除法再乘天数及后段减法本身的 float64 总量恢复舍入上界约 `2.21e-14 gC/m²`，
不包含 ε、原累加或 trace 输出精度。故此为带上述限制的时间段线索，非精确轨迹切片或因果闭合。

[当前回放](../../outputs/reviews/main-physics-20261011/319-npp-local/current-maintenance-replay.json)
与历史 day198 的相关状态差异：biomass 最大 `8.34e-12`、stempdiag `1.08e-12`、
动态 nroot `1.10e-13`；较大量纲能量系数 pcapa_en 最大绝对差 `3.15e-7`，
其按参考量级缩放的描述性比较另列于 verification，不作全状态等价断言。
day199 的 air forcing、长期温度、SLA、rprof、系数和开关
精确相同；48 次土温最大差 `4.15e-11`。这些是有界字段比较，不是旧版本身份恢复。
当前和历史状态回放各自的十项同输入 Fortran Oracle 检查均通过：原始过程抽取、
core/JIT/cache、48 次有序累积与写回；per-part 最大误差 `6.51e-19`，日累积/写回精确。
这排除了该输入上的公式/缓存数值不等价，不能扩大为所有分支或输入域通过。

[时序审计](../../outputs/reviews/main-physics-20261011/319-npp-local/maintenance-timing-scope/audit.json)
确认源码顺序为每步 maintenance 计算/累积先于 `IF(do_slow)` 中的日过程，
随后消费并清零累计量。所查 OK_LEAK 路径虽将 biomass 声明为 `INTENT(inout)`，
过程体只有读取；PFT14 的 `ok_laidev=false` 用当前叶 biomass/SLA 求 LAI。
这支持所查路径的日内静态缓存，NPP 可在日边界改变 biomass，不能跨日沿用。
冷启动/初始季节时序及原 Fortran 实际调用/发送覆盖仍未核定。

当前命名窗口 **day199 最后半小时**（JAX tstep9551 / Fortran itime9552），
只换历史 Fortran 同时刻的 `stempdiag`，保留 JAX biomass/SLA/rprof/系数：

| 同一步量 | 原当前输入 | 换参考土温后 |
| --- | ---: | ---: |
| 根温度 K | 282.7275886171008 | 282.7199175916129 |
| 总 maintenance radia 对 Fortran 的有符号残差，gC/m²/半小时 | `2.68757852284e-8` | `2.91463591109e-10` |

[局部归因](../../outputs/reviews/main-physics-20261011/319-npp-local/local-attribution.json)
给出约 **98.9155% 的该半小时残差**来自土温输入效应；root 和 sapbelow 分量的
变化分别为 `−2.478e-8`、`−1.805e-9`，再换 air/longterm 几乎无额外效应。
根温度使用 `rprof=1.25 m` 的源卷积，**不是 HYDROL 动态 nroot**。
本日累计 pre-NPP 根呼吸残差为 `2.10241929047e-7`；仅换最后一步后仍为
`1.85462369448e-7`，只解释 **11.7862% 的全天根分量残差**。其余 47 次 Fortran
土温输入未保存在所用 day-end trace 中，不能以末步或 daily mean 代替。
更关键的是，**该半小时总呼吸及本日 pre-NPP 根呼吸残差均为正，与年度总
maintenance 的负残差方向相反**。98.9% 是局部输入效应，不能解释年度主要偏差，
也不支持据此修改 THERMOSOIL；全天或年度因果均未闭合，当前首偏差日未确定。

[PEAT 审计](../../outputs/reviews/main-physics-20261011/319-npp-local/peat-scope/audit.json)
保留 raw 配置与 HYDROL 局部 SAVE 的区别；该阶段尚未核定319有效Fortran局部值，
未据case001 False trace改开关。第十二阶段已直接观察选定重样本的局部值False，范围见下节。
**本阶段没有确认可达源级物理错误或提出生产修复。**
分段诊断把优先范围进一步收窄至 **Day199–365 的年度维持呼吸负差贡献**。
保存的晚年日值/分量已在第五阶段盘点；观测缺口和具体后续捕获方案见下节。
现有同输入 Oracle 与当前前缀状态可复用。

## 第五阶段：晚年保存观测与 NPP 总量边界

**保存证据尚不能把 Day199–365 的 `−0.309144380353 gC/m²` 差分残差
继续分配给某个当前时段或部件。** 该值仍含第四阶段未量化的 ε，历史 LPJ
定义/producer/XIOS 发送身份仍属暂定；本阶段没有获得新轨迹或确认物理 bug。
[保存值分析](../../outputs/reviews/main-physics-20261011/319-late-year/saved-late-year-analysis.json)
和[字段盘点](../../outputs/reviews/main-physics-20261011/319-late-year/schema-inventory.json)
限定到两份当前全年结果/partial、当前 day199 回放及命名历史 trace/诊断文件：

- 当前全年 JSON 及 partial 只保存 count/首尾/均值/min/max。首尾日有总量，
  但没有完整逐日向量或 min/max 所在日。当前 Day365 的 post-NPP 总量为
  `−1.80427894847e-12 gC/m²/day`，历史 trace 为 `−2.79776202206e-13`，
  该日有符号差 `−1.52450274626e-12`。独立前缀 Day198 的日差为
  `+1.54456549217e-8`，仍不能替代原全年 compiled Day198 或给 ε 上界。
- 当前 day199 回放 JSON 和三份 NPZ 保存维持呼吸输入、48 步 radia/parts
  与日累积 parts，未保存 post-NPP 总量。五份旧部件诊断只有 Day1–173 或
  Day1–200；其中 Day199–200 pre-NPP 根差之和 `+1.42136608222e-6 gC/m²`，
  叶差之和 `+2.00533436800e-8`，符号仍与年度主负差相反。
- 两份旧 JAX 状态 JSON 均有365行，但没有维持呼吸/GPP/NPP 日通量；bound
  与 plain 是不同绑定/轨迹，producer 未核定，未合并或重新赋予当前身份。

[边界源码核对](../../outputs/reviews/main-physics-20261011/319-late-year/source-boundary/audit.json)
明确 Fortran `stomate_npp.f90::npp_calc` 314–322 先按 `PFTpresent` 求部件总和，
511–515 再扣除 biomass-floor 创建量/dt，530 计算 NPP；JAX
`npp_closed_update` 与 modelout 总量映射遵循这些边界。叶/根两项之和没有覆盖
所有部件、PFT 掩码或 floor 修正，不能当作最终 `MAINT_RESP`。
旧诊断的 `gpp_daily` 也不能直接补成同日总量：Day200 该字段为0，而 NPP
`0.511728221984`、growth `0.199005419661` 均为正；这是不同时间边界的保存字段，
不得用它代数构造一条“已测”post-NPP 曲线。

按后续边界诊断价值，最强的**历史候选窗口为 Day336–344**：LPJ trace 的首批
晚年精确零 GPP 日为 Day338/342/344，对应 post-NPP 总量分别约
`−1.0e-8`/`−4.82e-12`/`−1.79e-12`，修正前叶/根仍为正。
窗口内历史 post-NPP 总量累计 `1.88287272475 gC/m²`，仅已观测叶/根两项
累计 `2.72447291853`；两者之差没有独立测到各部件/floor，不能归为某一修正。
Day336 的旧 bound 日末土温最大差为 `+0.287143260 K`（第1层）。另一输入候选
Day199–218 包含 Day215 的 `+2.966994035 K`（第4层）峰值，但仅有日末温度，
且已测 Day199/200 呼吸差为正。两窗口都是观测优先线索，尚无当前逐日负差
证明其主贡献；Day336–365 的历史总量 `6.23211106221` 也不是当前残差分段。

直接缺少的观测是**原完整 compiled 轨迹的365个 post-NPP 总量及原前198日
总和**。进一步归因时才需要定位日的完整 pre-NPP parts/PFT mask、floor 创建量/dt
与同边界公共 GPP/NPP/growth；若指向输入，再取得48步输入，保留历史 producer 缺口。
唯一优先后续动作是[最小捕获方案](../../outputs/reviews/main-physics-20261011/319-late-year/future-diagnostic/plan.json)：
同输入、cold-start 1961、完整365日和原 compiled block=7 路径，在既有 Python
年度摘要入口保存日 modelout 总量向量，先核对复现的年度摘要，再按日/分段比较
历史 LPJ，并测量该新完整轨迹与已有独立前缀的 ε；不声称恢复旧运行未保存的日值。
脚本只捕获已返回的公共输出，不改变物理计算；本阶段只验证 dry-run，未执行模型。

## 第六阶段：当前完整日值与历史 trace 对照

仅执行获准的一次本地 ORCJAX 捕获：319.0-057.0、1961 cold-start、365 日、
compiled block=7。runner code=0，实际设备 `cpu:0`，耗时 `353.10 s`；
捕获日值有序且完整，四字段均有限。输入签名和生产源码树运行后未变，
未自动重试或启动第二条轨迹。执行身份为
`c7110e030bae21512538b1ae45add8d0317ce0d7`，Python 3.11.15、JAX 0.10.1，
实际导入 `E:/ORCHIDEE-MAN-JAX/jax_orchidee/__init__.py`。
逐日捕获运行目录为
`outputs/reviews/main-physics-20261011/319-late-year/future-diagnostic/runs/capture-20261012T012052Z/`；
本阶段核验材料、命令和日志摘要在
[分析目录](../../outputs/reviews/main-physics-20261012/319-daily-capture/)。

四字段新日向量的算术均值复现 runner 年度摘要，最大差约 `1.11e-16`；
与已保存年度均值比较的 `365*mean` 差最多 `2.84e-14 gC/m²`（NPP）。
新运行 Day1–198 与独立前缀按其保存均值构造的积分差为 GPP/NPP/维持呼吸
`0`、growth respiration `−7.11e-15 gC/m²`。这只是本次运行和序列化旧均值的
比较，不是旧全年执行未保存的 endpoint ε 上界。

NPP 科学合同仍失败：新 JAX `0.386004239713388`，reference
`0.38518497347831726 gC/m²/day`，绝对差 `0.0008192662350707`，高于原限值
`0.0003951849734783`；近零分母解释不适用。年度维持呼吸 JAX−历史 trace
净差为 `−0.3091441597415 gC/m²`。新日值显示 Day1–198 为 `+2.2061e-7`，
Day199–365 为 `−0.3091443803535 gC/m²`。负向贡献集中在交替大脉冲：
Day337、339、341、359 的维持呼吸差分别约为 `−0.316556`、`−0.314204`、
`−0.312361`、`−0.309689 gC/m²/day`；这些日 JAX GPP 为零，历史 trace GPP
为小正值。Day338、342、364 出现相反方向的大值，其中 Day338/342 是 JAX 有小
正 GPP 而 trace 为零，Day364 则是 JAX 大负 NPP/正维持呼吸而 trace 接近零。
Day337–343 净差 `−0.3145820828 gC/m²`，Day339–343 净差约
`−0.3135617041 gC/m²`；Day359–365 净差仅 `+0.000294197`。覆盖全年80%负向
绝对贡献的最短连续窗为 Day337–359。这些交替值只是 producer、分支可达性或
日/发送边界的核查线索，不能单独判定错位。该窗描述的是与命名历史 trace 的数值差，
trace 的 producer、变量定义和 XIOS 发送日边界仍未核定，因此逐日定位仍属暂定，
不能据此判断错位或确认 Fortran/JAX 源码错误。

维持呼吸日值采用不同 binary64 求和方式时，年度累计最大差为
`7.11e-14 gC/m²`；日值转 float32
后再以 binary64 求和与原 binary64 日值求和差约 `4.23e-8 gC/m²`，若转 float32
后继续以 float32 累加，差约 `2.47e-6 gC/m²`。trace 与 float32 NetCDF 的累计
差约 `1.96e-7 gC/m²`。这些远小于约 `0.309 gC/m²` 的日值净差，但只描述
日末归约和序列化，不是模型内部轨迹浮点误差上界。历史近零维持呼吸日没有
实质负向贡献，合计净差反而为正。

stderr 有一条 persistent compilation cache 写入 `FileNotFoundError` 警告及两条
XLA simplifier 循环提示；没有致命错误，runner 正常结束并核验全部365日，故未重跑。
唯一下一步是围绕 Day339–343 主净差窗及 Day337–338、Day359/364 抵消边界，
核查已有历史 trace 的 producer 身份、发送日边界和相关分支可达性；本阶段证据
不授权新的实验或轨迹，也不支持物理代码修改。

## 第七阶段：脉冲窗口的日志日界和成对源分支

只对 Day339–343 及 Day337–338、Day359/364 做离线核对，复用第六阶段完整
捕获和舍入结论，不重跑年度统计、模型或 Oracle。实际分析见
[analysis.json](../../outputs/reviews/main-physics-20261012/319-pulse-boundaries/analysis.json)，
来源审计见[provenance-audit.json](../../outputs/reviews/main-physics-20261012/319-pulse-boundaries/provenance-audit.json)，
成对出处及观测强度见[分支账本](../../outputs/reviews/main-physics-20261012/319-pulse-boundaries/branch-ledger.json)；
所有本地源码行号有[字面摘录](../../outputs/reviews/main-physics-20261012/319-pulse-boundaries/source-excerpts.txt)。
年度 NPP 原科学失败保持不变：绝对差 `0.0008192662350707` 大于原限值
`0.0003951849734783 gC/m²/day`。

**恢复的是候选日志定义及生成配方，尚非实际运行身份。**
`outputs/server_trace_patch/apply_319_dayend_state_trace.py` 91–106 明确写出
`itime,GPP,NPP,LAI,leaf,root,resp_maint,resp_growth,PFTpresent,age`；锚点为
`stomate_lpj.f90` 1133–1135，紧接 1118–1131 的 `npp_calc`，位于后续
kill/gap/turnover 之前。63–89 的另两组日志分别在 GPP 日累积及部件呼吸累积之后，
以 `MOD(kjit,48)==0` 采样。候选配方
`outputs/server_trace_patch/run_319_dayend_state_trace.sh` 25–42 指定复制
`modipsl_250919_dev`、应用补丁、Intel2020 编译并复制 paper XML；47–61
指定 1961、1Y、cold-start、CO2 317.27。但命名本地 trace 目录保留的8个文件中
没有实际 patch/build/run 日志、源码 revision/hash、可执行文件身份或当次
`used_run.def`/有效 XML。目录名及配方不能补成这些身份。调查止于命名记录和
相关本地字面引用，没有追查无关源码档案或访问服务器。

9天的原始三组日志步号均为 `48*day`，对应当前捕获的零基 `end_tstep+1`；
`do_slow` 全为 T，累积日志 GPP 与同一步 LPJ GPP 全部一致，LPJ 四通量也与
第六阶段保存的历史字段一致。这支持候选定义下的**名义时钟及 pre/post-LPJ
GPP 对齐**，不能证明实际 XIOS 日界。候选 paper XML 的
`field_def_orchidee.xml` 6 默认 `prec=4, operation=average, freq_op=1ts`，
697–698 定义维持/生长呼吸；`file_def_orchidee.xml` 726 为1日输出。
它们没有当次有效 XML 的身份桥。并且日志与发送本来就不在同一源边界：
`stomate_lpj.f90` 1359–1380 调用 cover/peat cover，3081 的
`delta_veg>min_stomate` 可在3094–3097修改通量；GPP/NPP 到1677–1678、
维持/生长呼吸到1702–1703才发送。实际 cover guards/发送值未保存，不能用
年度匹配或通量恒等式宣称 logger 就是 XIOS 当日输入。

下表都是已保存值，单位 `gC/m²/day`；历史身份仍为暂定。

| Day | JAX GPP | 历史 GPP | JAX 维持呼吸 | 历史维持呼吸 | JAX−历史维持呼吸 |
| --- | --- | --- | --- | --- | --- |
| 337 | 0 | `7.40095e-7` | `−1.00000e-8` | `0.3165561771` | `−0.3165561871` |
| 338 | `1.22414e-6` | 0 | `0.3155357983` | `−1.00000e-8` | `+0.3155358083` |
| 339 | 0 | `3.32747e-5` | `−1.80167e-12` | `0.3142041233` | `−0.3142041233` |
| 340 | `2.85359e-5` | `8.17118e-6` | `0.3129171364` | `0.3124278545` | `+0.0004892819` |
| 341 | 0 | `3.56189e-6` | `−1.78796e-12` | `0.3123610020` | `−0.3123610020` |
| 342 | `1.95550e-7` | 0 | `0.3120243992` | `−4.81520e-12` | `+0.3120243992` |
| 343 | `3.44648e-5` | `2.10580e-6` | `0.3119514698` | `0.3114617296` | `+0.0004897401` |
| 359 | 0 | `9.60326e-7` | `−1.80467e-12` | `0.3096890049` | `−0.3096890049` |
| 364 | `1.69141e-6` | 0 | `0.3099832018` | `−2.79776e-13` | `+0.3099832018` |

主窗维持呼吸净差为 `−0.3135617040582 gC/m²`。历史零 GPP 的338/342/364日，
pre-NPP 叶、根两项均为正，和分别为 `0.3042648657`、`0.3003972277`、
`0.2985995740 gC/m²/day`；这些是直接观测的两项，不能代替全部件/PFT掩码
下的总量。当前 lite 捕获没有完整 pre-NPP 部件、掩码、储备或 floor 创建量。

| 核对条件 | Fortran 出处 | JAX 出处 | 当前可支持的结论 |
| --- | --- | --- | --- |
| PFT 掩码和严格正 GPP / tax | `stomate_npp.f90` 307–357 | `carbon_kernels.py` 3984–4001 | 同样排除 bare、按 present 求和、用 `GPP*dt>0` 和严格 tax 比较。保存 GPP 支持名义 sign 路径；没有完整输入或 tax 决策观测。 |
| 超 tax 泵取 / 非正 GPP 扣储备 | `stomate_npp.f90` 359–387 | `carbon_kernels.py` 96–105、4001–4014 | 两侧都是相同8部件比例泵取；非正 GPP 则只从碳水储备扣除、令可分配碳为零。实际泵取和储备未知。 |
| 非 crop 分配 / 正分配才有生长呼吸 | `stomate_npp.f90` 397–491 | `integration.py` 3339–3355、`carbon_kernels.py` 4016–4031 | 当前配置 PFT14 `ok_LAIdev=FALSE`、growth fraction `.28`，按 upstream allocation 传入；具体 f_alloc 未保存。没有扩大为 crop 等价证明。 |
| 负碳池创建 / 维持呼吸修正 / NPP | `stomate_npp.f90` 505–530 | `carbon_kernels.py` 4033–4049 | 两侧创建 `min_stomate−negative_pool` 并从维持呼吸扣除创建量/dt。JAX精确钉住floor，Fortran顺序加回；没有 bitwise 或所有输入域等价声明。 |
| 参数绑定和输出映射 | `constantes_var.f90` 177、1263；`stomate_lpj.f90` 1118–1131、1677–1703 | `coupled.py` 39、3013–3080；`integration.py` 2425–2443、3510–3557、5130–5143 | 当前 paper bundle 显式传 `min_stomate=1e-8`，tax默认`.8`与当前saved run.def一致；底层min默认0不是此生产绑定。历史实际参数及发送身份仍未知。 |

**源码支持一种条件机制，尚未证明本窗经过了该floor路径。** 对非crop、
有限正 pre-maintenance `M`、`dt>0`、GPP=0，若输入储备为`R`且
`R−M*dt<0`，又没有其他负池/分配，扣储备再做floor会得到 corrected maintenance
`(R−min_stomate)/dt`；相同上游状态的微小正GPP若走超tax、各部件库存充足，
可通过分散泵取保留接近`M`的总量。两份源码本身都允许这一零值边界不连续。
这是带假设的代数分析，没有代入不存在的`R`、完整`M`或创建量观测，
没有运行合成kernel实验。因而必须分开三件事：源码确有该条件；日值与名义
sign路径相符；内部浮点/状态扰动是否实际触发了tax/floor及该量级差异。
第三件事现有证据未测到。第六阶段输出舍入排除结论不界定内部轨迹敏感性。

本阶段**没有确认具体翻译错误**，也没有确定历史参考日界错误或内部浮点成因。
唯一优先下一步为[最小匹配边界捕获需求](../../outputs/reviews/main-physics-20261012/319-pulse-boundaries/minimal-future-capture.json)：
取得可识别的Fortran/JAX producer、有效输入/日界，在主窗和对照日保存同边界
GPP、完整parts/PFTmask、allocation后biomass/f_alloc、tax/floor参数及决策/创建量，
并保存 post-NPP 与真实发送/当前摘要边界四通量及时间；若差异先进入GPP，才对
首个受影响日补48步assimilation/vegetation门槛/累积记录。新的参考必须声明自己的
producer，不能补称旧历史producer。本轮只写需求，未执行新捕获或申请作业。

## 第八阶段：真实匹配捕获完成，同输入 NPP 核心回放通过

本阶段已获准为 319/1961/PFT14 cold-start 做实际匹配边界捕获，覆盖
Day337–343、Day359/364。第七阶段的“只写需求”限制不再阻止本阶段执行；
前一个进度 checkpoint 的阻塞是本地 ACL 修复后服务端拒绝旧身份，见历史
[访问回执](../../outputs/reviews/main-physics-20261012/319-matched-capture/access-receipt.json)。
最初原身份因权限过宽被 OpenSSH 忽略，专用身份本地读取报 Permission denied，
这些是修复前历史。临时复制 key、收紧副本 ACL 并连接的命令曾被自动审批拒绝，
返回仅为 `blocked by policy`，该命令未执行，也未创建临时 key。

主控随后成功在原 key 上直接收紧 ACL，关闭继承，仅保留当前账户
`DESKTOP-84EB591/admin`、Administrators 和 SYSTEM；当前账户与原 owner 相同，
无可用 SSH agent。原/新 SDDL 及 applied 状态保存在主控的
[ACL 回执](../../outputs/reviews/main-physics-20261012/319-matched-capture/ssh-acl-repair.json)。
**原文件权限已修改，本地原 key 的 ACL 阻塞已修复**；凭据内容未读出、复制、
修改或替换。主控原 key 直接只读 hostname/id 检查及一次认证诊断均 exit255，
见[登录日志](../../outputs/reviews/main-physics-20261012/319-matched-capture/ssh-direct-login-check.txt)及
[认证日志](../../outputs/reviews/main-physics-20261012/319-matched-capture/ssh-auth-diagnostic.txt)。
连接建立、既有 ED25519 主机指纹匹配，server-sig-algs 支持 rsa-sha2-256/512；
Trying private key 后服务端继续要求认证，不再出现本地 bad-permissions/load-key
错误。**旧身份被服务端拒绝，原因未由日志确定**，不能仅凭文件名断言到期。
该旧身份阶段没有认证成功、远端数据读取、构建或作业；上述回执保留为历史。

用户随后明确授权新的本机身份（文件名日期为2026-12-24）。首次直连仍被本地
OpenSSH ACL 拦住；确认 owner 为当前账户后，按授权仅在原文件上收紧 ACL，
保留原/新 SDDL，未复制、读出或替换密钥。**新身份实际认证成功，连接至 login01。**
新旧结果分别记录在[新访问回执](../../outputs/reviews/main-physics-20261012/319-matched-capture/access-new-identity.json)与旧回执中。
原只读 wrapper 主体完成后因末尾 Windows stdin CR 返回127，认证并未失败；
后续 LF 字节输入的[详细检查](../../outputs/reviews/main-physics-20261012/319-matched-capture/legacy-319-details-new-identity.txt.receipt.json)
完整 exit0。只读入口补充新日志名及末尾换行处理，所有失败/成功原日志保留。

[身份分析](../../outputs/reviews/main-physics-20261012/319-matched-capture/restored-identity-analysis.json)及
[取回回执](../../outputs/reviews/main-physics-20261012/319-matched-capture/identified-319-provenance-receipt.json)
记录实际命名根 `orchidee_jax_319_dayend_20260718` 的31份有界文件（tar 2,170,880 bytes）：

- model/run 两处 `orchidee_ol` 同为55,659,600 bytes，SHA256
  `c65dc5ecb24c1544cf3aecfb65e30af416f03a34ee7d9c8778649c87346ada58`。
  最初 build.log 是失败记录；实际后续 `build_v3_link.log` 记录成功链接并在
  2026-07-18 11:38:34 结束，与二进制 mtime 一致。不能以最初失败日志否定后续构建。
- 实际 NPP 源 SHA 为 `74d1a77b…03aef`，与本地来源一致；实际 LPJ 为
  `e4f0ad12…648045`，与先前本地准备底本不同，实际 stomate 为 `1caf1e4a…41be22e`。
  完整 SHA、路径和出处均在身份分析中；这不是全源码 immutable snapshot 的证明。
- 实际 used_run.def 为1961/1Y、CO2 317.27、NVM14、三个 restart 输入 NONE、
  DT_SECHIBA1800/DT_STOMATE86400、PFT14 ok_LAIdev FALSE、tax .8。
  实际 `PEAT_HYDRO=TRUE`，不能把 case001 的 False 直接扩写为319；全局开关值
  本身仍不能证明具体 peat mask 支路活跃。本轮不扩展为新的水文调查或 bug 声明。
- 取回的 LPJ/accumulator/维持呼吸三份原始日值与此前本地分析文件**字节完全相同**。
  先前已完成的 logger clock 和脉冲定位可通过该文件桥复用。实际有效 XIOS
  context/field/file XML 已识别：stomate1 启用、1d，四字段 average/1ts/prec4。
  这补足具体配置/文件身份，仍没有完整 NPP entry 或真正 send 四通量/时钟 tuple。

原 Fortran logger 没有新增捕获所需的完整内部值，因此仍需新的参考 producer。
已对**取回的实际源码**生成
[实际来源 patch 审计](../../outputs/reviews/main-physics-20261012/319-matched-capture/actual-source-observers/patch-audit.json)：
剥离日志恢复实际 npp/lpj 原字节，保留旧 logger 和原算术次序；不会用本地不同
版本 LPJ 覆盖远端。该审计是本地准备快照；后续隔离构建状态见下文。

本地完成的可审查准备见
[README](../../outputs/reviews/main-physics-20261012/319-matched-capture/README.md)：

- [JAX 捕获脚本](../../outputs/reviews/main-physics-20261012/319-matched-capture/capture_jax_npp.py)
  复用第六阶段冻结输入、参数和 block=7 协议，在内存中观测原 NPP locals：
  完整输入、实际 sign/tax masks、泵取、floor 前 biomass、负池 mask/创建量、
  原 floor sum/dt 操作数和 post-NPP 输出；动态模型日由原 day wrapper 传入，
  仅保留指定9天。[AST 审计](../../outputs/reviews/main-physics-20261012/319-matched-capture/jax-overlay-audit.json)
  确认撤去观测及还原提取的操作数后，原算术 AST/运算次序精确恢复。
  [预检](../../outputs/reviews/main-physics-20261012/319-matched-capture/jax-preflight.json)
  仅用 GPP=0 和 `1e-6` 两组人工非 crop tuple，在 CPU 收到两次 callback；
  返回值最大差为0及 `6.617444900424222e-24`。人工 payload 已清空，
  不作为实际窗口数据。观测可能影响编译器融合；以上不是全年轨迹等价证明。
- [Fortran 准备脚本](../../outputs/reviews/main-physics-20261012/319-matched-capture/prepare_fortran_observers.py)
  只在新输出目录生成 npp/lpj 两份隔离副本，观测完整 entry、实际分支、泵取、
  floor 前值、原 WHERE 创建量、逐部件维持呼吸更新、post-NPP 和真实 XIOS
  send 前四通量/时钟，并记录 pre-cover flags/vegetation。
  [patch 审计](../../outputs/reviews/main-physics-20261012/319-matched-capture/isolated-fortran-source/patch-audit.json)
  确认剥离所有日志块后恢复原文件字节及算术顺序，原顺序 floor 未改。
  ES25.17E3 输出带 shape，展开数组按 Fortran 顺序解释。后续从创建量算出的
  sum/dt 必须标为离线派生，不能冒称原 reduction 的直接观测。
- [Fortran 日志语法检查](../../outputs/reviews/main-physics-20261012/319-matched-capture/fortran-observer-syntax.json)
  已在有类型的 dummy context 用既有 gfortran 16.1 通过；新增行最长102。
  这不是 ORCHIDEE 完整构建，也不是远端 Intel 编译或真实参考执行。

初始准备时真实 Fortran/JAX 全年轨迹均为0；缺 identified、完整 Fortran 回执时，
JAX `--execute` 已实际先于创建 run 目录拒绝。该计数属于历史准备快照。
本次在原 Fortran 捕获 SHA 和五项准入条件均核实、`jax-run` 尚不存在后，
使用[实际准入回执](../../outputs/reviews/main-physics-20261012/319-matched-capture/fortran-readiness-postbuild1-resolved.json)
启动唯一一次原本地 CPU JAX 全年，runner exit0、耗时303.89955秒。
[执行身份](../../outputs/reviews/main-physics-20261012/319-matched-capture/jax-run/command.json)及
[9天完整观测](../../outputs/reviews/main-physics-20261012/319-matched-capture/jax-run/selected-npp-capture.json)
保留实际输入、日值和摘要；年度及指定日 GPP/NPP/MAINT_RESP/GROWTH_RESP 均通过
第六阶段原严格限 `1e-8 + 1e-10*abs(ref)`，四项年度 delta 均为0。
这是有界四通量观测一致性，不是所有状态或逐位全年轨迹等价。
运行 stderr 的两次 XLA algebraic simplifier “50 runs” 警告见
[原日志](../../outputs/reviews/main-physics-20261012/319-matched-capture/matched-comparison-20261012/jax-execution.stderr.txt)；
成功返回和数值结果均保留，未因警告重跑。

输入准入曾因有效配置差异暂缓；原
[待核实回执](../../outputs/reviews/main-physics-20261012/319-matched-capture/fortran-readiness-postbuild1.json)
未覆盖。其后[输入核实](../../outputs/reviews/main-physics-20261012/319-matched-capture/postbuild1-input-identity-resolution.json)
确认原 JAX first-step/compiled-day 使用年度 CO2 查表，1961实际317.27ppm与
Fortran相同，见[无轨迹 reader 检查](../../outputs/reviews/main-physics-20261012/319-matched-capture/matched-comparison-20261012/effective-co2-input.json)。
soilclass `test.nc` 和 `reftemp.nc` 的实际 producer size/SHA 与冻结输入一致；
NVM14、PFT_TO_MTC14=2、dt1800、VEGET_YEAR0及cold-start/year控制保留。
Fortran原 run.def、有效 used_run.def 与 JAX物化 run.def 的三个不同 SHA 保持
独立身份，未修改配置来制造一致性。ALMA_OUTPUT TRUE/FALSE 是实际来源支持的
history输出格式差异；这里不宣称序列化 history 全面等价。单位来源对应真实大写
日发送 ID：`NPP_STOMATE/npp_daily`、`GPP/gpp_daily`、`MAINT_RESP/resp_maint`、
`GROWTH_RESP/resp_growth`；XML中的小写 rate-field 定义不替代这些发送字段。

[完整同输入比较](../../outputs/reviews/main-physics-20261012/319-matched-capture/matched-comparison-20261012/comparison.json)
在 Day338/339/343 将完整直接观测 Fortran entry tuple 送入未修改的原 float64 CPU
JIT NPP 核心；每次比较 `biomass`、`bm_alloc`、`resp_maint`、`resp_growth`、`npp`、
`biomass_before_alloc` 的完整返回数组。18/18通过同一严格限，outside-limit count
全部为0；Day338 biomass最大差 `5.2635586871892785e-18`，其余17项为0。
Fortran精度来源见[已核实执行/精度回执](../../outputs/reviews/main-physics-20261012/319-matched-capture/postbuild1-execution-verification-resolved.json)：
实际producer IOIPSL `defprec.f90:19–20` 定义 `r_8=SELECTED_REAL_KIND(15,307)`、
`r_std=r_8`；实际成功构建的 `mpiifort -r8 -fp-model precise` 配置支持8字节/64位
real解释。这是源码和实际构建选项证据，未新增compiler probe、构建或模型运行。
两条真实轨迹原生 biomass shape 均为 `(1,14,12,1)`；实际 producer
`constantes_var.f90:232–233` 定义 nelements=1、icarbon=1，JAX carbon索引为0。
没有丢弃、补零或填充 element。先前误把人工两元素预检当作真实轨迹形状的
比较尝试在 Day337 的形状检查、首次 kernel 调用前停止，完成回放数为0；
[原错误日志](../../outputs/reviews/main-physics-20261012/319-matched-capture/matched-comparison-20261012/replay-execution.stderr.txt)
保留，修正 helper 后才执行上述三次回放；没有追加任何全年积分。

三回放日的两条原轨迹 entry biomass/GPP/f_alloc/resp_maint_part 已超严格限，
pft_present、frac_growthresp、dt、tax、min_stomate 完全相同。以下为直接观测的
PFT14值，F/J分别指原 Fortran/JAX 轨迹；reserve创建量为该部件直接观测值。

| 日 | entry GPP F / J | reserve floor创建量 F / J | post NPP F / J |
|---|---|---|---|
| 338 | 0 / 1.2241381168386465e-6 | 0.31504199107720837 / 0 | 9.999999994736442e-9 / −0.3155346427512101 |
| 339 | 3.3274738014818624e-5 / 0 | 0 / 0.31418535863012853 | −0.31417271195702 / 1.801669924361704e-12 |
| 343 | 2.1057950331453215e-6 / 3.4464809264695845e-5 | 0 / 0 | −0.31145974177337743 / −0.3119189349729448 |

9天窗内 Fortran floor 触发日为338/342/364，JAX为337/339/341/359。实际分支、
逐部件创建量、Fortran顺序维持呼吸更新及JAX原correction操作数均已保存；
Fortran创建量sum/dt只标为离线派生。证据支持原轨迹进入 NPP 前已有差异，并未
复现这三个相同输入 tuple 的 NPP 核心不匹配；不能进一步认定具体上游函数错误、
floor的独立因果、其他分支覆盖或年度科学通过。

[日界比较](../../outputs/reviews/main-physics-20261012/319-matched-capture/matched-comparison-20261012/fortran-boundary-comparison.json)
将实际 direct post-NPP→send 四通量在9天逐一核对，36/36通过；三份旧日志与新
logger的所选完整tuple比较27/27通过。NetCDF为365个noleap日记录、veget1–14；
日d真实bounds为 `[(d−1)*86400,d*86400]`，中心为 `(d−0.5)*86400` 秒，
send时钟为d；PFT14 cover=1及原NPP面积约定保留，不扩写为其他面积换算结论。
四字段各5110有限值、float32存储；36项与一次float32 cast兼容，作为独立离线
表示诊断。保持原double严格限时仍有三项 storage→send 诊断超限：Day340 NPP
`1.0932610039482427e-8`，Day343 MAINT_RESP `1.264388871380362e-8`，Day359
MAINT_RESP `1.0134268757511222e-8`。这些结果保留，不放宽门槛或混作直接double比较。

第六阶段原 NPP 科学失败保留：JAX 年均 `0.386004239713388`，参考
`0.38518497347831726`，绝对差 `0.0008192662350707258` 超过原限
`0.0003951849734783173`；本阶段没有新科学通过结论，没有确认翻译 bug 或
floor 因果，也没有重做已完成的舍入排除或 Day199 Oracle。

**账户覆盖决策已解决：用户确认“已有可用额度，正常使用即可”。**
从已接受 `e05369c9` 续作的三次有界只读查询见
[计算条件记录](../../outputs/reviews/main-physics-20261012/319-matched-capture/compute-admission-20261012/compute-admission.json)。
kshcnormal 为 UP，首次快照有723个 idle 节点；候选 a05r1n09 的独立节点记录
为 IDLE、CPUAlloc=0、负载0.05、32核、RealMemory126500M，快照不构成预留。
qhcess/user_qhcess2 关联及近期其他已完成8核构建记录可见，构建耗时449–628秒；
这些不是本次参考 producer 或费用证据。本次近期作业列表还含另一既有 RUNNING
作业，不能继续把较早的“未见运行作业”快照当作当前账户状态。
billing TRES、QoS UsageFactor=1及空白资源限额都不能单独证明新增收费、免费
资格或货币额度，既有/缺失作业也不能单独证明收费或执行权限。
相同 SSH 入口先后落在 login09、login03、login01：首次命令名盘点含
mybalance/gbalance，后两节点及 `bash -l` 环境均未获得可用余额查询；没有余额、
账户单价或结算规则返回。费用调查到此停止，不推测最坏货币费用。随后用户针对
单节点8核/16GiB/最多2小时确认已有可用额度，见
[用户覆盖确认](../../outputs/reviews/main-physics-20261012/319-matched-capture/compute-admission-20261012/user-allocation-coverage.json)；
此确认解决执行覆盖决策，不证明货币费率为0，不要求另行调查余额或重复询问。
隔离构建/捕获的具体资源、目录和命令保存在
[执行方案](../../outputs/reviews/main-physics-20261012/319-matched-capture/prepared-compute-capture-20261012/README.md)。
用户确认后按原捕获范围执行；只有确需新增付费或扩大资源/时限才准备具体变更。
首次分配124281665在a05r1n09运行，sacct为8核/16GiB、23分43秒、FAILED 127:0；
wrapper为24分05秒。新副本构建耗时340秒，但编译输入仍读取旧未加观测的
预处理缓存；启动器因缺libnetcdf.so.13/libnetcdff.so.6/libfabric.so.1停在动态
加载阶段，没有完整模型积分。历史原NPP、LPJ、二进制及所读取缓存的哈希和
mtime保持原记录；此为有界身份检查，不声称旧树全部文件均已审计。必要修复
只在新副本清理/重建缓存及配置既有运行库，不重复制模型，不改变物理源码。
第二次分配124285780在a05r1n10占用372秒（6分12秒），同1节点8核/16GiB；
新隔离来源重建357秒并成功链接，NPP/LPJ预处理源码和编译路径均在新根。
两种观测标记实际存在；`strings | grep -Fq` 在pipefail下返回141，导致假阴性
exit7并释放分配，模型没有启动。原脚本及失败日志保留。直接二进制检查通过，
加载原Intel/MPI环境后ldd全部依赖解析。两次实际累计1795秒（29分55秒），
原7200秒范围剩余5405秒。post-build分配124287267于12:15:04Z在a05r1n10获配，
亲和性0–7、初始负载0.05；复用该二进制，最多3600秒，模型timeout3000秒，
不重新构建或复制模型。该资源上限仍在原2小时/16 CPU-hours内。作业已COMPLETED 0:0，
实际101秒（step96秒）；模型成功标记、NPP/send观测及历史三份日值logger均已生成。
累计1896秒（31分36秒）及4.2133 allocated CPU-hours；完整数组、实际日边界与身份
已离线核实，见[已核实参考执行](../../outputs/reviews/main-physics-20261012/319-matched-capture/postbuild1-execution-verification-resolved.json)。
[原执行快照](../../outputs/reviews/main-physics-20261012/319-matched-capture/postbuild1-execution-verification.json)
保留准入时 input/JAX-pending 状态及资源账；当前配对状态见上述新比较产物。
实际构建路径、二进制标记及141退出码见
[第二次分配取证](../../outputs/reviews/main-physics-20261012/319-matched-capture/matched-execution-20261012/audit-retry1-after-guard.txt)；
Slurm实际用量与完整运行库解析见
[运行环境回执](../../outputs/reviews/main-physics-20261012/319-matched-capture/matched-execution-20261012/validate-retry1-runtime.txt)。
未在任何login节点编译/运行模型，不自动改用其他平台。
现有 Intel 原环境已恢复，ifort 19.1 20200306 的 version 查询 exit0；最初未加载
环境的 license 错误和 vendor setup 的 unset-variable 失败均保留，不能当作正确
环境下的当前编译失败。该版本查询仍不替代实际计算节点上的完整编译 admission。
1961 forcing 实际存在（851,746,768 bytes），只读 size/mtime，未重算整个文件哈希。
新参考只在 jcwu 新隔离根进行；实际成功 Fortran/JAX 年各1次，同输入回放3次。
原16项和9项记录仍是历史资料；本次新增身份/文件桥/实际源补丁/引用核验见
[恢复执行核验](../../outputs/reviews/main-physics-20261012/319-matched-capture/verification.restored-access.json)。
没有重跑旧人工 callback、Fortran 语法/算术 AST、Day199 Oracle 或原16/9/12项；
本次新增产物验收见[最终执行核验](../../outputs/reviews/main-physics-20261012/319-matched-capture/matched-comparison-20261012/verification.final.json)。
真实捕获及有界比较已完成，原年度科学失败和未决因果保留。

## 第九阶段：GPP上游源路径与选定窗口，下一观测限于Day337

本阶段从第八阶段接受提交续作。完整入口为
[Stage9 README](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/README.md)，
[实际Fortran源图](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/fortran-producer/fortran-gpp-path.md)、
[实际JAX路径/声明参数](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/jax-path/jax-gpp-path.json)与
[保存值分析](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/saved-window/analysis.json)。
只读取得命名实际producer的4份compiler-inputs及2份run.def成员，共
1,454,080字节；[身份](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/fortran-producer/source-identity.json)
和[传输回执](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/fortran-producer/compiler-inputs-transfer.log.receipt.json)
记录原根、归档/成员身份与无远端写入。未重新询问认证、ACL或额度；已有额度
确认保留，本阶段无新构建/积分/分配是任务边界。

本次JAX沿原实际argv的48步日编译、block=7、NumPy accumulator off、
single-pass fold off、outer compiled daily carbon走生产路径；PFT14 local
beta/C3链创建的payload `gpp`直接进入STOMATE entry，不能以未证明生产入口的
`pft14_gpp_from_assimilation`便利函数替代它。五份当前JAX源码及相关函数的
字面摘录/身份保留。选取的48项实际声明参数一致，run.def哈希仍分别记录；
这不证明48步动态状态或掩码相等。

| 边界 | 实际Fortran producer来源 | 当前JAX来源与核定范围 |
| --- | --- | --- |
| 活动C3入口/掩码 | `diffuco_main` 673–679选择`diffuco_trans_co2`；STOMATE启用且`stomate.f90` 6337–6339拒绝`ok_co2=FALSE`，接受执行满足TRUE；`IS_C4__00014=FALSE`。`diffuco.f90` 2352–2384检查LAI、veget/veget_max、swdown、humrel和temp_growth | `diffuco_trans_co2_c3_pft_explicit` 3095–3341实际被local链调用；`diffuco_trans_co2_activity` 2288–2364。`OK_LAIDEV14=FALSE`取LAI>0.01；crop阈值0.001分支不用于本配置。动态门槛值未保存 |
| 叶层、冠层和控制 | `diffuco.f90` 2566–2774的C3求解/fallback，2825–2828按层积分assimi/Rd；2839–2849乘PFT14盐度/淹水控制 | `diffuco.py` 2640–2941矢量叶层和`jnp.sum`；1611–1635/3298应用控制。两源都把失败/低叶层解落在−Rd，随后总GPP没有额外MAX(0)或+Rdtot；不同求解/归约执行的实际数值效应未核定 |
| 控制生存期 | `diffuco.f90` 445–627只在`firstcall_diffuco`计算并保存control_salinity/control_inudate | `orchestration.py` 3661–3698的后续日helper丢弃当前salinity/tide参数，复用previous-state保存控制；3701–3753的firstcall helper才重算。不能从“day static cache”名字认定错误的每日重算 |
| 半小时GPP/单位 | `diffuco.f90` 2902：`assimtot*12e-6*veget_max*dt_sechiba`，μmol CO2/m²PFT/s转为gC/m²mesh/timestep；未活动点保留初始化0 | `diffuco.py` 3059使用同一受assimilate控制的式子；`orchestration.py` 5821–5910传入真实local payload，非日志别名 |
| PFT面积归一化 | `stomate.f90` 2921–2932用`veget_max/(1−totfrac_nobio)`形成veget_cov_max，3021–3032再形成`gpp_d=gpp/veget_cov_max*86400/dt_sechiba`；有源阈值/裸土掩码 | `daily_inputs.py` 199–220；本次compiled runtime从first entry复用shared vegetation normalization，未保存48步实际权重，不能据此证明或否定复用条件 |
| 日累计与NPP入口 | `stomate.f90` 3208/9377–9399按步加`gpp_d*dt_sechiba`，do_slow除dt_stomate；4309传给LPJ，实际`stomate_lpj.f90` 1121进入NPP | `daily.py` 1212–1508通用有序fold。实际PEAT_OCCUR=FALSE不选择peat-only static-JIT路径；`orchestration.py` 7559–7639与`integration.py` 1660–1750/3249–3372/3412–3482/3510–3557把同一daily field传入NPP，f_alloc/biomass来自实际alloc输出 |
| 发送与reset | NPP后才有cover过程和1738日GPP发送；`stomate.f90` 4968清零 | `orchestration.py` 8778–8800清零并写回day-end state。名义日d对应Fortran步48d、JAX tstep48d−1；不混同半小时累计量、日率和存储float32值 |

下表均是已保存的两条原轨迹在NPP入口的直接观测；有符号差和最大绝对差是
离线归约。它们不是上游同输入回放，也不证明全年首次分歧。native biomass
仍为`(1,14,12,1)`，f_alloc/maintenance为`(1,14,12)`；完整值、索引、shape和
原严格限见上述analysis。

| 日 | JAX GPP入口 | Fortran GPP入口 | JAX−Fortran | biomass最大差 | f_alloc最大差 | maint_part最大差 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 337 | 0 | 7.40095013e−7 | −7.40095013e−7 | 0.00953 | 1.17e−6 | 1.85e−5 |
| 338 | 1.22413812e−6 | 0 | +1.22413812e−6 | 0.168 | 1.05e−4 | 2.75e−4 |
| 339 | 0 | 3.32747380e−5 | −3.32747380e−5 | 0.00955 | 5.48e−7 | 1.73e−5 |
| 340 | 2.85358715e−5 | 8.17117891e−6 | +2.03646926e−5 | 0.167 | 1.04e−4 | 2.71e−4 |
| 341 | 0 | 3.56188524e−6 | −3.56188524e−6 | 0.167 | 1.04e−4 | 2.71e−4 |
| 342 | 1.95549742e−7 | 0 | +1.95549742e−7 | 0.334 | 2.08e−4 | 5.41e−4 |
| 343 | 3.44648093e−5 | 2.10579503e−6 | +3.23590143e−5 | 0.168 | 1.04e−4 | 2.72e−4 |
| 359 | 0 | 9.60325572e−7 | −9.60325572e−7 | 0.165 | 1.01e−4 | 2.74e−4 |
| 364 | 1.69140837e−6 | 0 | +1.69140837e−6 | 0.331 | 2.03e−4 | 5.48e−4 |

所选9天PFT-present精确一致，但该mask不是assimilate、layer_active、GPP面积
归一化或maintenance的part vegetmask。指定窗口最早的共同观测边界为Day337
日末NPP入口；实际Fortran当天after-accum日志、NPP入口和post-NPP/send GPP
均为`7.400950133835139e−7`，JAX入口为0。因此已观测的后续发送位置不能解释
该入口差；最早不等的半小时/叶层/前一日状态仍未观测。Day338差值反向，不能
仅凭这一模式确定掩码、根fallback或floor的年度因果。

保存schema给出九个完整NPP核心入口元组及365日日值，没有所需48步DIFFUCO
参数、gate/leaf值、归一化权重和累计前后状态。有效XML中STOMATE GPP按日
写入；保留header虽operation为1800s，但write为1d且仅365记录，不能当48步流。
实际DIFFUCO源码每步向numout写pre/post-control assimtot，保留stdout尾部确有
未标步号的零值；它们不能被赋予Day337时间/PFT身份，亦不足组成完整元组。
不从日末总量反构缺失48点，不把未观察过的门槛值当直接证据。

已把候选与反证保存在
[candidate register](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/candidate-register.json)：
活动门槛/叶层mask、C3求解和冠层归约、归一化/累计/reset，以及已观测的
biomass/allocation/maintenance差异继续分开核查。前几项是源机制/执行区别，
不是已确认原因；后几项是真实原轨迹输入差异，也未追溯到最早创建点。原三日
18/18 NPP同输入回放、36/36 core→send、27/27旧日志tuple复用，未重跑；原
三项存储double诊断超限和年度NPP失败不变。

唯一下一观测已经明确为
[Day337合同](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/next-observation/contract.json)和
[命令方案](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/next-observation/command-plan.md)：
Fortran步16129–16176、JAX tstep16128–16175，记录48个真实源样本及NPP/reset
边界；完整生产函数58个必需参数、可选参数、控制/leaf/权重、原生9输入元组
和相关状态都列入合同。已检查的原执行资料无匹配Day336起始快照；年末restart
不能重标为该起点。后续采集需复现未改算术的冷启动前缀；方案保留原365日
调度/原block=7、只打开Day337观测，明确不是现成单日积分。现有binary不能
输出新增源字段，隔离观测实现、目标shell展开dry-run、构建和执行均为后续工作。
本阶段只生成/核验合同与参数模板，没有启动模型、prefix或新作业。

## 第十阶段：Day337两侧捕获、严格准入与有界同输入比较完成

入口为[当前README](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/README.md)、[命令记录](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/commands.md)和
[当前续作合同](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/CURRENT_CONTINUATION.md)。点319.0-057.0、1961、PFT14；
Fortran kjit16129–16176、JAX tstep16128–16175，前日界16128和日末NPP/send/reset16176。
两侧均保留原cold-start365日轨迹。既有JAX捕获继续复用，本轮没有新积分、重建或计算分配。

[Fortran第七次终态](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/fortran/137_stage10_job_status_124444469_1.txt)记录job124444469在
a15r4n12、单节点8CPU/16GiB，实际495秒。模型构建369秒、原365日运行89秒已完成；
调度FAILED/ExitCode1:0来自运行后的parser，不是模型启动或积分失败。
[实际producer receipt](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/fortran/retrieved/124444469/source-receipt-124444469.json)绑定
构建/运行命令、三份raw及三份PP源码、独立binary、run.def/used_run.def、XML、forcing和静态输入。
执行lpjfix3包SHA为`472c2967d19804c04e7478b6adf5548631c77bc5f44956a97e95f33bc90a84a7`，
schema为`942fab0da90b0b29f01a44f4120b97e1438757ab2edece3b21d4c210d18d1518`。
原producer保护快照前后相同，diff为空；不改变旧源、binary或科学数据。

### 原始捕获与有来源的解析恢复

[原始capture](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/fortran/retrieved/124444469/stage10_gpp_subdaily.txt)为496815字节，SHA
`d53c061b00358ba8e1d28ed05b26e267b6933804cb6cd177448906a3f05b0f40`。
原parser因同一步重复`min_sechiba`拒绝。独立salvage仅接纳`min_sechiba`、`dt_sechiba`
在48步各出现两次且值相同；原parser、失败回执和全部raw字节均保留。
[独立恢复核验](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/capture-recovery-verification.json)的9项检查通过：96对数字文本/值/行号
完全一致，对应实际编译源DIFFUCO与STOMATE的四处WRITE；全部raw记录与48个sample、
Day337 NPP及两侧边界共51组逐字段往返一致，九个原生NPP入口的完整元素数通过。
[解析结果](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/fortran/retrieved/124444469/parsed-capture-124444469.json)没有通过修改status来代替上述证据。

保存的time dump实际含365个日中点及完整0–31536000秒边界，noleap、1961起点。
原receipt的`time_counter=UNLIMITED`是误抓dimensions的读取缺陷，原receipt保持原样；
恢复核验从data段读取实际365坐标，未把该字符串当作时钟通过证据。

### 自身准入与年度GPP读数问题

[原始34项自身比较](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/fortran/retrieved/124444469/stage8-own-trajectory-parity-124444469.json)
为33/34通过（Fortran16/17、JAX17/17），严格限保持`1e-8 + 1e-10*abs(reference)`，
shape和离散mask精确核对。唯一失败是Fortran年度GPP：reference `0.777659237560746`、
observed `0.7776592241724974`、abs_error `1.3388248509826894e-8`，超过
`1.0077765923756075e-8`。该原失败保留为`stage8_own_trajectory_changed`，未凭差小归因浮点或生产bug。

[独立文本精度复核](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/annual-cdl-precision-review.json)确认两侧变量均为
float32存储、相同time/PFT14/单位/daily average定义；Stage8使用`ncdump -p9,17`，Stage10
默认导出最多7位。将Stage8的365个GPP日值格式化为7位，逐项复现Stage10全部文本及失败年均。
默认7位文本再cast到float32仍有227个日值不同，不能据此作为原存储值恢复或准入。
[现存NetCDF高精度核查](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/fortran/analysis/fortran-restored/stage10-gpp-high-precision-comparison.json)
绑定两份原文件的路径/哈希及实际只读`ncdump -p 9,17`命令、回执、log和新CDL。
两侧365个PFT14 GPP的高精度文本及恢复的float32值逐项一致，四通量的365个PFT14文本日值
和年平均也逐项一致；重新导出的Stage8四数组与已接受参考精确一致。
默认文本年平均差恰为原失败的`−1.3388248509826894e-8`，因此该项是分析导出精度问题，
没有实际年度GPP存储值改变的证据。native float32值以float64求均为`0.7776592376006434`，
原9位文本求均为`0.777659237560746`；二者定义分别记录，未改原科学数据。

[修正输入后的严格34项](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/stage8-own-trajectory-parity-high-precision.json)
全部通过（Fortran17/17、JAX17/17）。[真实命令回执](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/corrected-parity-command.json)
证明仍使用原检查器、原数学和原阈值，仅换用有身份的高精度Stage10导出输入。
原默认精度CDL、33/34失败、原producer receipt均保持原样；本轮没有新增积分或分配。

[首次阻塞准入](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/blocked-original-CDL/analysis-admission.json)
在原自身失败时停止，调用计数为0；原command/status/README及调用历史保留。
严格34项确实通过后完成下述有界比较，当前[准入记录](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/analysis-admission.json)
与[结果说明](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/README.md)绑定实际分析和append-only journal。

### 48步跨侧结果与完整真实tuple重放

[完整分析](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/analysis.json)和[逐步表](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/steps.csv)
包含48个实际时刻。窗口首个输入/状态差在JAX16128/Fortran16129已存在，不能称作全年首差。
唯一活动/零正分支差、首个有效canopy/GPP差及最大GPP差均选中同一时刻：
JAX16136/Fortran16137、offset8；去重后只选1个完整真实tuple，没有补足到三个。
该步Fortran的humrel−min_sechiba为`7.673888101763685e-6`，JAX为`−1e-8`（humrel=0）；
其他活动门槛的有符号距离均为正。Fortran活动且有8个有效叶层，JAX不活动。
两侧actual cover均1；Fortran捕获的`gpp_mesh_step=7.400950133835139e-7`，JAX为0。
这给出该步水分活动门控差异的直接证据，但wind、LAI、veget等输入也已不同，不能确定humrel
创建差的原因或将全年NPP归为单一GPP原因。

实际调用当前生产函数共2次，分别使用该时刻完整65参数的Fortran和JAX真实输入。
Fortran输入重放的8个有效leaf/root_vc与8个root_j全部通过原严格限，离散mask精确匹配；
leaf.assimi及root_vc最大绝对误差`1.7053025658242404e-12`，root_j为
`2.0605739337042905e-13`，GPP为`2.9869266926163215e-15`。
JAX真实输入重放也与原函数观测一致。每次调用有且仅有1个实际root callback。
[调用journal](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/replay-call-journal.jsonl)SHA为
`8935246cd100ff048c2ecc01a478f313861c64245a86d0742362ad3d182e624a`，保存全部参数、返回值和callback。
随后native shape/report适配失败及修复保留于[调用历史](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/analysis-invocations.jsonl)；
最终报告按selection、side及全部65个参数精确匹配，恢复已完成调用，新增调用0。
完整计数为started2/completed2/failed0，不能将最终[command](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/command.json)
的0次新增调用误读为从未重放。

跨侧叶/根数值比较必须取两侧calculate/leaf及相关root-validity交集；JAX不活动使交集计数0，
这类项标记无有效值，不能当作数值通过。上述活动数值证据来自独立Fortran真实输入重放。
九个原生NPP入口中biomass、GPP、f_alloc、resp_maint_part有差，其他5项通过；
原生NPP-GPP→send和两侧精确reset的3项通过。Fortran有序累计末值`0.0639442091563356`
经86400秒平均为`7.400950133835139e-7`，未以任意求和替代实际累计过程。
JAX entry紧凑payload没有LAI，同调用点辅助比较明确为unavailable；完整生产kwargs.lai仍直接取自真实记录，
未借用另一调用点的LAI。不同输入轨迹不构成翻译反例，局部同输入闭合也不是全DIFFUCO分支证明。

唯一优先后续方向是沿JAX16136/Fortran16137的实际humrel创建与传递链向上游追溯，
核对两侧对应输入和状态；需要新采样的范围另由主控安排。本次只完成现有证据分析，
原年度NPP科学失败及既有存储诊断保持开放。

### 已接受JAX证据与科学边界

[JAX attempt3命令](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/jax/jax-run-attempt-3/command.json)保留CPU、block7、365/365日及252.189秒；
[真实capture](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/jax/jax-run-attempt-3/selected-trajectory.json)SHA为
`7669cd740bfef93176fd82789c59e21d1cc5aa39617973d8e740575ad39e22fb`。
四个事件流各48条，九个完整原生NPP入口及前日界/reset保留；自身17/17严格比较通过，
287/287内部链和3/3 NPP/send/reset通过，195条journal与七种事件流逐样本一致。
实际48步humrel=0、min_sechiba=1e-8、有符号距离−1e-8，活动门槛全FALSE、GPP全0、cover均1。
同身份calculate和raw root validity交集的有效叶/根计数0，单独JAX窗口没有增加活动根精度证据；
本次Fortran输入重放的活动证据另列。

原JAX三次年度启动wall分别28.688/250.751/252.189秒，共531.628秒，完整年2、完整capture1；
另三次1/1/2日route diagnostic共241.900秒。[运行账](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/jax/attempt-history.json)与原失败回执保留。
原[24项JAX验收](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/verification.final.json)、Stage8的18/18 NPP、36/36 core→send、27/27旧tuple、
Stage9已接受核验和三个float32 storage→send严格double诊断超限继续保留，未因本次读数问题改写。

### 实际分配与保留的失败链

| 作业 | 实际终态/失败位置 | primary分配秒 |
| --- | --- | ---: |
| 124403380，a12r1n00 | FAILED；MPI setup拼写，未构建/积分 | 8 |
| 124408135，a08r3n12 | FAILED；复制后LPJ hash guard，未构建/积分 | 607 |
| 124431192，h20r1n02 | FAILED；观测patch多余END DO，未积分 | 128 |
| 124435002，a10r2n19 | FAILED；解包顺序检查，未构建/积分 | 21 |
| 124438871，h16r2n11 | FAILED；LPJ observer引用未声明npp，未积分 | 190 |
| 124441509，k20r2n04 | FAILED；继承binary检查路径不一致，未构建/积分 | 9 |
| 124444469，a15r4n12 | FAILED；365日模型完成后parser重复记录拒绝 | 495 |
| Stage10合计 | 完整模型年1、回收并恢复的真实capture1 | 1458 |

Stage8已用1896秒，合计3354秒（55分54秒）、7.4533 allocated CPU-hours，原7200秒余额
3846秒（64分06秒）。最后TimeLimit72分钟是单次上限，不是当前余额。本次收尾不使用新分配，货币费用未知。
[恢复记录](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/fortran/STAGE10_RECOVERY.md)保留每个调度回执和构建失败，不重置预算。
[producer证据索引](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/producer-evidence-index.json)绑定6份compiler输入、14份XML、时钟和保护证据；
[资源账](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/resource-restored.json)逐项绑定七个primary调度行。
观测修复仅移除重复END DO、将LPJ观测绑定实际npp_daily及统一继承binary准备检查；
[单行修复](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/fortran/96_stage10_lpj_npp_observer_repair.diff)和[静态复核](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/lpj-observer-binding-review.md)保留来源。
Stage8 LPJ实际raw/PP是`38201e…`/`e0a548…`；移除六段旧观测日志后的`e4f0ad…`
只是审计patch基线，不能标成Stage8实际compiler输入，见[源码对账](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/fortran/50_stage10_source_reconciliation.json)。

[父级工件核验](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/verification.restored.json)47/47通过，含独立保存值数值重算、实际调用/journal恢复、
原34项准入、来源/资源/链接及7份已接受工件不变检查；复核积分0、重放0。
原[24项JAX验收](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/verification.final.json)保持原样。
图项目`E-ORCHIDEE-MAN-JAX`、generation2026-10-14T11:04:08Z保持ready；[本次覆盖回执](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/graph-closeout-coverage.json)
记录outputs明确excluded/not_tracked，
采用精确源码和保存产物补足，未重索引或声称全覆盖。仅版本化更新本页，其他新产物均在原Stage10 outputs根，
执行方未stage/commit/push、改生产物理/配置/容差/其他研究树、发跨聊天消息或启用自动化。

## 第十一阶段：humrel上游链及最小下一观测

本阶段复用已接受的第十阶段48步捕获、34/34严格自身准入、9项解析恢复、47项工件
核验及两个完整65参数真实调用，不重跑上述检查或模型。离线结果见
[保存值分析](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/analysis/results.md)、
[48步CSV](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/data/humrel-48steps.csv)及
[候选与反证矩阵](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/analysis/candidate-evidence.md)。

### 实际参数、创建、写回和下步读取

[JAX成对链](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/jax/humrel-state-chain.md)
核对实际compact/module-JIT/local-JIT路径；四份生产文件Git blob与数值producer相同，
观测器字节与接受的执行身份一致。
[Fortran链](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/fortran/humrel-chain.md)及
[取证回执](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/fortran/retrieval-receipt.json)
记录一次严格主机验证的SSH只读检索：只取第十阶段命名producer树的HYDROL/SECHIBA
raw和PP四文件，无远程写入。旧编译日志确认这些PP路径被编译、对应对象被链接；
旧producer回执未记录这四份文件的构建时内容hash，因此当前命名树到历史二进制的
字节闭合仍有限。下表Fortran行号为该取回raw，PP对应行号见链表，非自动以本地源码
代替实际执行来源。

| 边界 | JAX精确来源与行为 | Fortran取回来源与行为 |
| --- | --- | --- |
| TRANS实参 | `orchestration.py:6084–6160/3485–3514`从前一步state逐字段按diffuco→hydrol→slowproc选取；`diffuco.py:3888–3906`取PFT14列13，humrel无缺省零填充 | `sechiba.f90:998–1006`向DIFFUCO传入模块humrel，在本步HYDROL之前 |
| 原生创建输入 | `hydrol.py:10391–10756/6931–7020`以最终修正的mc重新构造mcl，再形成液水sm及阈值层剖面 | `hydrol.f90:6474–6562`形成水分诊断；`6565–6672`计算分层应力/根加权 |
| clip/zero/求和 | `hydrol.py:7023–7184/7772–7933`保留旧/新应力、动态根、PFT缺席、under-mcr与tile分支，humrel与vegstress分别归约 | `hydrol.f90:6658–6668`为旧应力分支，`6688–6714`缺席/under-mcr清零，`9006–9044/9103–9110`作tile掩码、求和及非负floor |
| 写回→下一步 | `orchestration.py:5913–5939/2947–2966`把HYDROL humrel写入hydrol与diffuco carry；下一compact读取这两个副本 | `sechiba.f90:1027–1056`在DIFFUCO后调用CWRR HYDROL，其inout humrel供下步DIFFUCO使用 |
| 同步入口的另一量 | `orchestration.py:5879–5885`把当前HYDROL vegstress以humrel字段送STOMATE | `sechiba.f90:1184–1187`传vegstress，经`slowproc.f90:346/399/973–976`的humrel形参送STOMATE；`stomate.f90:2483/3032`为形参及实际观测写出 |

此改名链的slowproc为有界本地源码回退，SECHIBA/slowproc构建时字节未绑定；
STOMATE观测写出的raw/PP hash则与job124444469实际compiler-input回执相同。
原生输入语义的源关联与各文件的历史字节证明分开记录，不扩大取回四文件的证明范围。

实际compiled外层代表metadata `day_index=2/start_tstep=48`、内层`tstep=47/48`
不是Day337物理时间。接受的观测器使用动态science_day/offset，记录16128–16175；
未来插入点必须沿用这些tokens。日缓存`_compiled_diffuco_day_inputs`不含humrel，
而各compact步有前一步restart读取。DIFFUCO COMB虽可在TRANS之后按`qsatt<qair`
清零humrel，之后HYDROL仍写回carry；两处源码均反对直接把本次零值归为冻结日缓存
或COMB残留，但未观测的内部写读桥仍不能据此完全排除。

### 首个可见上游信号及保存状态边界

| Day337 offset | JAX tstep / Fortran kjit | 实际TRANS humrel：JAX / Fortran | 同步STOMATE入口vegstress：JAX / Fortran |
| --- | --- | --- | --- |
| 0–6 | 16128–16134 / 16129–16135 | 0 / 0 | 0 / 0 |
| 7 | 16135 / 16136 | 0 / 0 | 0 / `6.1471104814109476e-6` |
| 8 | 16136 / 16137 | 0 / `7.683888101763685e-6` | 0 / 0 |
| 9–47 | 16137–16175 / 16138–16176 | 0 / 0 | 0 / 0 |

这是选定窗口中实际TRANS参数首次exact及strict差异；窗口其他输入/状态在offset0
已经不同，不能称为年度首次分叉。`entry_payload.fields.humrel`是vegstress的改名，
不是当步DIFFUCO humrel。47个相邻入口→下步TRANS对的零/正号均相容；Fortran唯一
非零对的比值为`0.7999999999999999`，`vegstress−0.8*next_humrel`为
`−8.470329472543003e-22`。相关非peat旧应力分支令
`A=(sm-smw_tmp)*(smf-smw)/(smf_tmp-smw_tmp)^2`，则相关分支为
`us=clip(A/pcent)*nroot`、`vegstressv=sum(clip(A)*nroot)`；同层、根、mask一致且
两式均不饱和时才有`vegstress=pcent*humrel`。保存配置12个pcent均为0.8。
这是有条件的离线桥，支持正常一步流向，不能当作原生humrel观测或证明全部前提。

保存的真实JAX日初/末七组件state含humrel、humrelv、us、nroot和原生mc/mcl；
两边界的humrelv/us均零，PFT14的JAX nroot仅在Fortran编号第6层为1，
不能称根不存在或推断实际Fortran根。边界别名/归约8项精确相等，只证明这些边界，
不覆盖offset7内部。state.tstep未序列化，逻辑日界依实际observer位置确定。
入口soil_mc是聚合总水量，不能代替原生tile液水mcl。历史case001相关hydrol-v9的
144条tile行及1200条系数行仅作命名辅助，不提升为319当前轨迹证据。

有效配置有`NEW_WATSTRESS=FALSE`、`dyn_nroot_larix=TRUE`、`OK_FREEZE_CWRR=TRUE`。
参数`PEAT_HYDRO=TRUE`与HYDROL的private、无初始化器的`LOGICAL,SAVE peat_hydro`
是不同作用域，取回PP仍保留其运行期分支；JAX soil-local显式False、历史case001
trace为False均不能证明319本次Fortran局部值。第十一阶段该值仍缺失，尚未确认配置不匹配；
第十二阶段随后直接观察选定重样本的局部值False及创建操作数。最终比较器两侧
min均为1e-8，Fortran正margin为`7.673888101763685e-6`，不是末端ulp量级差；
更上游相消/阈值敏感性仍待观测，不调整dtype、mask、公式或容差。

### 唯一下一步：原生HYDROL创建→carry→TRANS读取

[最小观测方案](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/plans/minimal-observation.md)、
[字段/维度/准入合同](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/plans/minimal-observation.json)、
[插入点草稿](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/plans/observer-hooks.draft.txt)及
[具体argv](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/plans/commands.prepared.json)
已准备，当前并非可运行的新捕获器。轻量记录覆盖全部48步的humrel/vegstress、tile
输出和写读副本；重记录仅offset6–8及前一Day336日界，核心桥为offset7 HYDROL
最终液水/层阈值/根更新/mask→原生humrel及写入副本→offset8实际读取。记录运行期
局部开关、native dtype/shape、tile/PFT/层号和生产完成/下步消费时间；未命名的
inline clip前表达式只能由保存操作数离线推导，不能标为原生观测。

默认保留两侧原1961 cold-start前缀及365日调度。JAX完整日初state仅是未验收的
短restart候选；Fortran无已核定Day336 restart，年末restart不能倒置使用。
后续须在新隔离来源中加入纯观测、绑定未来raw/PP HYDROL和SECHIBA构建字节，
沿用原strict数值限`1e-8+1e-10*abs(ref)`及精确离散/shape，比较原365日四通量、
9个NPP输入、48组65参数/分支/输出、日边界及累计/send/reset，证明轨迹保留。

资源仅为未来提案：原成功数值job构建369秒、模型89秒、分配495秒；新增观测开销
未知，估计500–900秒，拟硬上限1200秒、单节点8CPU/16GiB，最多2.6667 requested
CPU-hours，若用满则余额由3846降为2646秒。JAX历史本地252.189秒另记，拟250–600秒、
本地cap900秒；货币费率未知。本阶段实际新增分配/构建/模型/生产重放均0，未提交
这些命令，未重跑旧验收。原年度NPP科学失败、存储诊断和年度因果问题继续开放，
无已确认生产bug。交付主控手动审阅此聚焦观测方案后决定后续执行。

## 第十二阶段：原生HYDROL捕获与配对诊断

本阶段实施第十一阶段最小观测合同，保留319.0-057.0、1961、PFT14、cold-start365日
及JAX block7/半小时scan/local-JIT。轻量原生记录覆盖Day337全部48步和前一Day336
offset47边界；重记录仅该边界与Day337 offsets6–8。新捕获器与记录见
[Stage12入口](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/README.md)和
[命令记录](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/commands.md)。
原年度NPP科学失败、原存储诊断和年度因果问题继续开放。

### JAX实际执行与自身准入

[预检](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/jax/jax-preflight.json)
16项通过；两次实际2日production-route预检均完成模型路线，保留后处理/namespace
绑定失败记录。修正读出复用已保存journal及有界真实packet/JIT检查，没有第三次模型预检。
[唯一年度启动](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/jax/launch.json)
完成365日，实际258.600996秒，低于900秒进程树上限，无自动年度重试。
四份新增overlay剥除精确observer与身份参数后，与原生产AST相等；原scan、算术及JIT
边界保留。其磁盘工件核验见
[22项核验](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/verification/jax-artifacts.json)。

[直接Stage10自身准入](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/analysis/jax-own-parity.json)
11990项通过，最大绝对误差0，严格限仍为`1e-8+1e-10*abs(reference)`。
范围含48组65参数/活动/输出/叶层候选、入口、9个原生NPP输入、保存日界、
有序累计/send/reset及365日统计；离散/shape/dtype精确一致。非finite候选仅验证原生
坐标/符号与mask保留，不把它们当成有效物理解。

Stage10未保存完整365日日值向量，因此另以有独立身份的Stage6直接post-NPP日值为
参照，[365×4逐日比较](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/analysis/jax-daily-stage6-parity.json)
1460项全部通过、最大误差0。11项身份检查核对生产源码/config无diff、实际数值argv、
JAX/import、run.def及共享36输入manifest：该小清单hash等于Stage6绑定hash，各输入
当前size/mtime签名逐项一致，Stage8已接受来源桥相同。没有为此重跑基准年或重新hash
大数据文件；不把Stage6日值改称Stage10直接观测。

### JAX原生创建与写读

[原生capture](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/jax/native-capture.json)
有227条：reader、HYDROL输出、state_write各49条（48选定步+1前日边界），
soil输入/最终各4条，三个逐tile重阶段各24条。
[保存值核对](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/analysis/jax-native-analysis.json)
614项通过，48对HYDROL输出→两份carry/finalize→下一步reader精确一致，同步STOMATE
入口等于本步vegstress。offset7/tile4/layer6的原生`sm=3.0802668141645855`、
`smw_tmp=3.0826979486351997`；离线差为`−0.0024311344706142712`。
原生old factor与root denominator均0，根fallback保留该层权重1，`us/humrel/vegstress`
均0，下一步TRANS读0。该结果闭合JAX选定桥，配对Fortran证据见下节。

JAX未单独emit实际`mask_soiltile`局部量；实际absence、under-mcr、active-layer及
root-support mask已捕获。由保存tile fraction/vegetmax恢复的tile mask，以及未命名的
inline preclip表达式，均仅可标为离线推导。本阶段上述分析和工件核验新增生产函数调用0。

### Fortran实际执行与严格自身准入

原cold-start365日Fortran模型已在第三次实际分配job124575036完成，调度状态
COMPLETED/0:0，Slurm ElapsedRaw417秒；内部build296秒、run95秒、script403秒不另加。
实际native capture解析得到1259条非空记录、47个不同label，全部schema label均存在；
原生KIND记录为real8/integer4/logical4，这些是kind编号，不笼统解释为字节数。
source reversal及1259条合成记录解析预检只证明工具行为，真实准入另见下述回执。
父级源码复核已纠正观测器的`us(1,:,jst,:)`切片，并补齐tile入口mc/mcl及原生
`mask_soiltile`。原HYDROL-private未初始化`peat_hydro`只观察，不初始化或修复。

真实解析首先拒绝schema将`znt`声明为11项：原始24条heavy记录各有32项。
原schema、首次失败及raw均保留，修订契约完整保留32项，实际47个label的shape/count已核对。
高精度文本读出还暴露目标`ncdump -p17,17`对float32的格式问题：
四通量各5110个空槽、veget14个空槽，而365个time及730个bounds有实际值，见
[导出诊断](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/analysis/fortran-export-format-diagnostic.json)。
同一SHA的NetCDF原生实读确认veget为float32、14项1–14、无mask；相同目标ncdump的
`-p9,17`可正常显示。[完整7变量侧文件](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/fortran/retrieved/attempt-3b/annual-fluxes-p9-17-124575036.cdl)
按实际float32语义解码，并由相邻receipt绑定真实argv、原CDL/source receipt、
同一NetCDF/ncdump及独立reader身份。9位有效数字支持float32往返，double读出仍17位；
原文件、生产dtype及严格数值限保持原样。该读出修复没有新增模型年。

[Fortran自身准入](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/analysis/fortran-own-parity.json)
7152项通过、失败0，严格限仍为`1e-8+1e-10*abs(reference)`。准入绑定原Stage10
raw/parsed/schema/source/高精度CDL/验收身份，核对365×14×4原生日值、48组65参数与
活动/输出、9个NPP原生输入、原始记录顺序/重复、mask/nonfinite/root9999 sentinel，
以及有序累计/send/reset。新增Stage10 raw与已接受原raw逐字节相同。
原source receipt的time_counter="UNLIMITED"是header文字，准入使用实际坐标；
source-receipt402秒、worker403秒与Slurm417秒分别保留，不混用。

三次分配均为单节点8CPU/16GiB/最多1200秒，按实际Slurm整项计时一次：

| job | 节点 | 终态 / ExitCode | 分配秒数 | 实际结果 |
|---|---|---|---:|---|
|124550126|h12r2n14|FAILED /124:0|814|720秒编译子进程超时，模型未启动|
|124558362|g03r4n17|FAILED /1:0|16|FCM既存build lock退出，模型未启动|
|124575036|a02r1n11|COMPLETED /0:0|417|增量编译及原365日模型完成|

累计Stage12分配1247秒；原7200秒减去前阶段3354秒及本阶段1247秒，余额2599秒，
无pending分配。本地JAX另计，货币费率未知。实际回执与身份以
[Fortran账本](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/fortran/resources.json)
及[父级资源汇总](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/resources.json)绑定。
两次失败、各自冻结worker及原部分构建保留；空FCM锁在确认作业/相关进程结束后，按原
bytes/mode/uid/gid/size/mtime归档核对并移除，成功读回确认原锁不存在。

attempt3只staged，两个提交路线在自设idle/empty-queue门槛前退出，未sbatch或消费marker；
不能据此归因于资源不可用。独立attempt3b复用原source/flags/部分构建，实际选中
MIXED节点a02r1n11，准入快照为18可用CPU、77500MiBSlurm内存余量、62972MiBFreeMem、
归一化负载0.38。冻结脚本实际使用负载上限1.0及首个合格IDLE/MIXED，未实现方案中的
0.5/idle优先；所选节点实际负载满足0.5，该源码差异保留，不以之后快照补称当时无idle。
第二次本地launcher源码曾被误改、原bytes未恢复，当前文件不充作历史原件；该作业身份
由原冻结worker、submission/consumed/admission/job及终态回执绑定。

### 配对创建阈值、根与下一步读取

[配对保存值分析](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/analysis/paired-native-analysis.json)
266项通过、失败0、新增模型/核调用0；详细解释、49条轻样本及24条重样本见
[原生结果](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/analysis/native-results.md)。
offset7/tile4/Fortran编号layer6的直接原生值为：

| 量 | Fortran | JAX |
|---|---:|---:|
|积分液水`sm`|3.0827141294163454|3.0802668141645855|
|萎蔫阈值`smw_tmp`|3.0826979486351997|3.0826979486351997|
|`sm-smw_tmp`（离线推导）|+1.6180781145624934e-5|−0.0024311344706142712|
|原生`us`/创建humrel|7.683888101763685e-6|0|
|原生创建vegstress|6.1471104814109476e-6|0|
|原生最终layer6根权重|1|1|

Fortran旧clip factor及根归一化分母是离线推导；JAX同名factor/denominator为已保存原生
局部量且均0。JAX真实零分母fallback仍保留根权重1，不能称无根。offset8两侧新创建值
均0，但Fortran实际TRANS reader精确收到offset7的正humrel。两侧各48对相邻
producer→next-reader均通过；选定Day337内最早创建/vegstress差在offset7，最早TRANS
读取差在offset8。原STOMATE入口名为humrel的字段是本步HYDROL vegstress，不混作TRANS值。

offset7进入土壤更新前，两侧tile4的`mcl`最大差已为0.004301499804477013；前一Day336
边界已有0.006137724021737706差。因此本轮解释的是不同液水状态越过阈值及一步carry，
下一待查边界在观测创建器上游的pre-soil状态/历史，不能认定同输入土壤求解器错误、
年度首次分歧或年度唯一原因。全部24条重样本的冻结水profile最大值及`mc-mcl`最大值均0；
有效peat=false、new_watstress=false、dyn_nroot_larix=true、soilclass3、pcent0.8。
Fortran实际HYDROL-private未初始化peat_hydro在原位置被观察，未初始化/修复；WRITE中
PFT存在/缺席条件是对源操作数的条件求值，不称为既有mask局部量。

本轮有界捕获、自身准入及配对诊断已完成，原年度NPP失败仍为
`0.0008192662350707258 > 0.0003951849734783173`。没有生产修复或新的生产bug确认。
下一步缩小液态水状态/历史差异需要另行指定源依据与观测范围，不在本次自动续跑。

## 第十三阶段：进入土壤前水分状态的上游边界

本阶段限于点319、1961、PFT14原cold-start365日轨迹的既有本地工件与成对源码。
[Stage13入口](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/README.md)、
[保存值结果](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/analysis/results.md)
及[118项计算](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/analysis/saved-profile-analysis.json)
给出新证据；第十二阶段的11990/1460/614/7152/266/49项已接受结果继续复用，没有重跑。
本阶段不把不同输入的状态更新当作同输入物理过程反例。

### 状态定义、相位和实际来源

`mc`为原生节点总水，`mcl`为原生节点液水，均按landpoint×node×tile排列；JAX捕获
形状为`(1,11,6)`。积分层液水`sm`与STOMATE诊断`soil_mc`各有独立定义，不能替代
原生`mcl`。`soil_mc`从总水诊断`mc_layh_s`导出；本轮未以该名字推定液水状态。

[JAX源链](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/jax-source/source-chain.md)
核对cold-start的`mcl=mc.copy()`、restart namespace的两次transpose、precall、显式
HYDROL、输出写回和下一步读取。三份生产源码在Stage12 producer`91a1a79b`与当前
`6f9d9f9`间blob一致。七日compiled块的实际observer记录保留动态日/offset身份，
不从代表性scan metadata推定日界。cold-start源公式不等于已有初始状态实测。

实际Fortran producer候选raw源码位于Stage12目录并与job124575036的构建后receipt
哈希相符；不是仅凭本地同名文件推断。hydrol/sechiba去除指定观测块及`kjit`传参后，
非空源行在忽略行尾空白时与本地源码一致，见
[映射核验](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/source/fortran-source-map-verification.json)。
预处理源码只有已绑定哈希，文本未保存在本地；本阶段没有SSH补取。

Fortran的`hydrol_tmc_update`在soil之前可按guard改变`mc`，没有`mcl`写入；初始冻结
profile准备也不写`mcl`。候选源码5774–5781的tile-entry观测在本步首次液水重建
5920–5935之前。JAX的precall保留前一步`mc/mcl`并准备冻结profile，soil_input在
显式模块前；模块外层将收到的`mc/mcl`直接传入all-tiles。因此这两个控制点的
**carried液水字段**可以对应，不能据控制点不同泛称液水相位不等，也不能把所有
`mc`、冻结、温度、通量操作数一并认作相等。

### 真正的主求解输入和写回边界

[成对inner-HYDROL源审查](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/source/fortran-inner-hydrol-order.md)
核对系数、入渗、液水重建、三对角求解、质量/上下界校正及写回顺序。所选主求解路径
在读取液水前，已经从入渗后的`mc`及冻结profile重建`mcl`；JAX同样重建`mcl_before`，
没有用旧入口`mcl`作为独立求解输入。因此入口液水差是状态历史线索；本步真正消费的
总水、当前冻结profile、通量和系数仍需单独核对。

|边界|成对源顺序|本次实际证据与缺项|
|---|---|---|
|系数和入渗|入渗前计算系数→入渗更新总水→重算系数；矿质冻结分支用`mc`、`mcr`、冻结比例、查表bin及根因子|所选Fortran记录soilclass3、peatfalse、freeze true；bin、系数、入渗通量及独立的温度guard操作数未捕获|
|主扩散|重建液水→构造上边界/内部/下边界RHS→按`resolv`三对角求解|两侧所选冻结profile为0且`mc=mcl`；没有配对完整RHS、根汇、上下边界通量、系数或求解结果；JAX实际mask未输出|
|求解后|排水/残差校正→重建总水→过饱和路由→负径流校正→强制水位→低于残余水处理|所选under-mcr记录false；其他动态guard和中间结果缺失，源顺序对应不等于这些分支已实测或同输入通过|
|最终状态|Fortran从校正后总水重建液水，并在dummy蒸发积分前保存、之后恢复`mc/mcl/tmc`；JAX保存`mc_final/final_mcl`到下一步|源码和下面8条已保存字段桥约束所选过渡的净dummy水分泄漏；不扩大为所有中间字段或全年状态均已验证|

没有发现经完整实际操作数确认的同定义源级反例。上述映射不是完整数值等价证明；
尤其不能因最终冻结profile为0，就推定更早入渗的独立subzero-temperature guard未触发。

### 保存状态连续性和完整profile

新分析覆盖Fortran归一化步号16128、16135、16136、16137，即前一日末及Day337
offset6、7、8；每时刻6tile×11node，CSV保留264行。两侧stress-input至下一步entry
的`mc/mcl`在offset6→7、7→8均逐值精确一致，共8条字段桥、最大误差0。
JAX Day336末原生`mc/mcl`也与保存的Day337初previous state精确一致。
这些是新核对的数组桥，不是重跑原48条humrel桥。

该日界packet实际含afterDay336/beforeDay337及afterDay337/beforeDay338两份状态；
没有额外Day336初状态或Day338积分。状态未序列化原生tstep，日归属来自实际wrapper
保存位置。采样连续性排除了这些桥上的净水分状态丢失，不证明全年每一步的状态传递。

以下差均为Fortran−JAX，tile4，节点编号采用Fortran编号：

|量|Day336末|Day337 offset6|offset7|offset8|
|---|---:|---:|---:|---:|
|entry液水profile最大绝对差|0.006137724021737706|0.00434869089765362|0.004301499804477013|0.004183630262115337|
|该最大差节点|2|3|3|3|
|第6层entry线性投影差|0.002482858856223075|0.0023797805997185016|0.002415638674865858|0.0024473152517605038|
|第6层stress实际积分差|0.0024538207086944996|0.002415638674865858|0.0024473152517605038|−0.0034619675367310965|

第6层积分使用节点5/6/7，保存的共同`dz`给出权重
`3.910068428 / 35.190615852 / 7.820136856`。offset7的stress节点差分别为
`+0.0006299974419507748 / −4.551740668407023e-7 / 0`，积分贡献分别为
`+0.0024633331074924874 / −0.000016017855731983723 / 0`。
所以越过第6层阈值的积分差主要来自相邻第5节点；不能称为第6节点有很大的正液水差。
完整profile的最大差在节点3，也不是第6节点。

offset7的entry投影差`0.002415638674865858`，加两侧不同输入更新量之差
`+3.167657689464564e-5`，得到stress积分差`0.0024473152517605038`。
这只是保存值的线性恒等分解；entry投影不是观测到的早期`sm`，也没有执行模型回放，
不能把这两个量解释成同输入求解器效应或年度因果百分比。

24条重样本中，两侧entry/stress的`mc-mcl`最大值及已捕获冻结profile最大值均0；
两份JAX日界状态同样`mc=mcl`且冻结profile为0。故选定差异同时存在于总水与液水，
仅液水/冰水差解释不符合这些保存点。中间重建操作数、未采样时刻及全年冻结状态
仍未直接观察，Fortran peatfalse仍只限所选24条实际记录；JAX局部tile mask未捕获。

### 当前结论和剩余缺项

已保存的逐步和日界状态桥闭合，较早水分profile差异已存在。没有完整匹配动态输入
元组，不能从终端残差反推缺失通量、系数、mask或温度/冻结操作数；也不能据当前
不同输入更新量之差确认物理实现bug、年度首次分歧或年度唯一原因。
原年度NPP科学失败仍为`0.0008192662350707258 > 0.0003951849734783173`。
本阶段只完成本地源码与保存值调查，不授权后续新运行或生产修改。

唯一下一观测为**前一步恢复后的总水状态→完整主更新输入/输出**配对边界；本阶段已准备
[字段/相位合同](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/analysis/next-observation-contract.json)：
以最早已有native观测Fortran16128/JAX16127为锚，保存其前一步Fortran16127/JAX16126
最终恢复后的`mc`及冻结状态，再保存tile4入渗后`mc`、重建液水、土壤/水力参数、
系数/矩阵、上下边界通量、根汇、mask/resolv、`dt/dz`，以及RHS、求解和各校正结果
直至最终`mc/mcl`。精确源位置和字段见成对源审查的单一下一观测段；它可区分已由
前一步继承的profile差与本次主更新边界新增差，并为之后同输入测试提供真实元组。
若前一步已有差，年度起点仍未确定。该合同不要求再观察已闭合的offset6→7/7→8
carry，也不把移动到更早一步称为找到起源；本阶段未实现观测器或执行该捕获。

## 第十四阶段：日末、三文件restart与跨年记忆

本阶段从`847a5f35797613e9b6778810566187c7e4ccb9f7`的干净main开始，
仅审查固定paper PFT14路径的三组状态；完整字段、条件、写读/首消费者和来源见
[计时与日累计表](../../outputs/reviews/main-physics-20261018/restart-year-state/temporal/findings.md)及
[17字段季节记忆表](../../outputs/reviews/main-physics-20261018/restart-year-state/seasonal/README.md)。
既有181字段lifecycle闭合、`known_global_gaps: []`及schema合同继续保留；
本阶段没有重新验收全部181字段，也不把缺少旧执行SHA解释为未实现过程。

| 选定状态组 | 成对源码与写读/首消费者结论 | 条件和证据边界 |
| --- | --- | --- |
| `dt_days`、STOMATE `date`、`do_slow`、`EndOfYear`、driver步号 | Fortran `slowproc`586–614/653–663按秒钟日界设置事件并推进SAVE date；`stomate_io`501–506/2187–2195读写date。JAX日fold完成后写packet；新年driver rebase为−1，再从0执行，STOMATE date继续保留。EndOfYear是实际消费者事件，不由文件年份标签推定。 | 当前三份本地materialized配置均为86400秒STOMATE步，paper为365日/年、48步/日；在此配置下核对事件顺序。旧`Job0_bio`11/40从仓库外RUNDEF复制运行配置，未读取该外部文件，不能用tracked模板缺省证明旧实际步长或peat开关。 |
| 13项序列化日累计量及3项运行时日累计量 | Fortran先使用完成日均值/极值，再于`stomate`4943–5026清零，随后writer保存。JAX显式`daily_accumulators`也是reset后状态；`humrel_daily`的NetCDF别名为`moiavail_daily`，`gpp_daily`重复副本必须一致；温度极值重设为±large_value。 | `snowfall_daily/snowmass_daily/tmc_topgrass_daily`不写此restart合同，但必须通过精确零验证，不能据缺变量称为丢失。当前本地PEAT_OCCUR=false，wet/liqwt累积分支inactive；其他配置为conditional。crop/permafrost附加日字段在此选定合同外，不因此声称相应物理过程inactive。 |
| 17项季节/年度及logical状态 | 长期/月/周温度和tau正常保留；firstcall仅在源条件满足时初始化。年度GDD/降水先thisyear→lastyear再清零；Tseason临时和春季时钟按源EOY规则重算/清零。begin_leaves/生长时钟先由season使用，再由phenology更新；senescence仍供allocation读取。bool按≥0.5读回。 | tau单位为天，配置上限3×365=1095；当前PFT14为NATURAL、MTC2/pheno1、onset模型none。非自然GDD与若干春季/落叶分支对活动PFT14 inactive，其他表示槽保留原mask。分支分类来自源和当前配置，不冒充本轮执行了各分支。 |

### 条件性tau默认值与文档反证

`restart_io.py::read_stomate_readstart_states_from_template`794的
`tau_longterm_max=3.0`默认值没有换算为配置的1095天；Fortran
`stomate_io.f90`615–637在长期温度有效且tau缺失/为val_exp时使用配置上限。
一次有明确缺项依据的
[微型状态检查](../../outputs/reviews/main-physics-20261018/restart-year-state/seasonal/tau-partial-state-check.json)
直接调用实际reader：长期温度279K时，missing/sentinel tau默认均为3，显式1095时均为1095；
已有tau=123时两者均保留123。该检查只构造1land/2PFT的NumPy/NetCDF状态，未积分模型。

缺字段与sentinel须分开：生产入口`restart_state.py`140–164先执行严格season reader，
因此缺tau在调用helper前失败；严格reader不拒绝存在的999999 sentinel，后续helper可走3天回退。
`restart_state.py`203返回该helper状态，`orchestration.py`1192/1522–1525选择它供restart消费者使用。
这是helper默认值的同输入反例及条件性生产读路由，不是完整sentinel三文件回放；
正常full writer写出的有效tau绕过该回退，既有正常完整restart证据不被此结果撤销。
没有修复生产代码，也没有证据将此条件归因319 cold-start年度NPP失败。

另有文档疑点：`coupled.py`2226–2228称tau未在reader中；实际dataclass、映射、reader/writer
及Fortran均读写tau。该注释已过时，不能当作缺失季节记忆；本阶段仅记录，未修改生产文件。

### 既有证据到当前源码的桥

[版本与配置桥](../../outputs/reviews/main-physics-20261018/restart-year-state/evidence/README.md)
核对实际producer脚本/输出合同、23个选定函数AST、11项别名/bool/时间尺度常量及四个Git快照。
memory/firstcall、尺度adapter、season packet、日末与新年入口从`7333b46`到当前一致；
annual完整函数从`deba027`一致，较早差异仅涉及herbivory安全分母，所选GDD/降水copy/reset未变；
readstart/fullwriter及Stage5 run与`7eb41be`一致。该桥不代表所有传递依赖相同。
当前YAML与`7eb41be`一致，tracked run.def自初始快照未变；三份receipt命名的本地
`used_run.def`当前字节/配置另行绑定，不能倒推旧运行当时字节或外部RUNDEF。

保留Stage5的19项通过、13 driver/93 SECHIBA/174 STOMATE精确roundtrip及1962 split/memory
状态/输出相同；复用001/069/319矩阵的年度末态、直接跨年、三文件split及离散/模式/schema证据。
181是lifecycle字段数，不是NetCDF变量数。旧receipt未内嵌执行SHA/完整命令/config digest，
compiled producer当前另有block参数而旧receipt未记block；恢复的命令形式不是历史命令证明。
这些具体限制保留，既有通过结果不重跑、不重标为当前执行，也不扩展为Fortran全年科学通过。
原319年度NPP失败仍为`0.0008192662350707258 > 0.0003951849734783173`。

## 第十五阶段：tau读取上限的天数与配置修复

本阶段只修复第十四阶段已确认的条件性读取缺陷。Fortran
`stomate_data.f90`697–700将`coeff_tau_longterm * one_year`存为天数，
`stomate_io.f90::readstart`620–637先判断长期温度是否全部为`val_exp`：
缺失/全sentinel温度时以输入温度初始化并设置tau=2；长期温度有效而tau
缺失/全sentinel时才使用配置上限。有效tau直接保留。该源优先级未改。

`restart_io.py::read_stomate_readstart_states_from_template`的默认值改为既有
`SeasonTimeScales().tau_longterm_max`，默认系数3和365天得到1095天，
显式参数仍直接表示天数。生产`reference_case_first_step_restart_state`复用
`paper_case_season_time_scales(config_path, used_run_def=run_def_path)`，将同一
季节配置换算后的天数传给helper，不将1095写成通用配置常量。显式`run_def_path`
转交同一解析器；未给出时沿用季节尺度解析器已有默认路径，不声称它与其他reader
各自的默认run.def路径相同，也未重构这些默认选路。
`coupled.py`仅更正tau已被restart读写及其单位/回退条件的过时docstring；
科学公式、dtype、mask和容差均未改。

[实际检查回执](../../outputs/reviews/main-physics-20261023/restart-tau-fix/checks/attempt-01.json)
记录一次ORCJAX本地执行：完整相关unit文件30项及新增生产入口回归1项，
31/31通过，pytest用时40.09秒。12项新helper参数案例分别检查有效长期温度下
tau缺失/sentinel/123与默认/显式42天上限，以及长期温度缺失/全sentinel时
tau缺失/sentinel/123均优先初始化为2。长期温度值和字段状态报告同时核对。

真实生产入口回归在系统TEMP复制原三文件和有效run.def；默认与自定义系数2.5
分别使sentinel tau恢复为1095和912.5天。正常模板的有效tau=1095在两配置下
保留，entry/season/daily/gas/remainder全部读入字段及report精确相同。
缺tau字段仍在helper之前由严格reader抛出KeyError，不扩大本次入口支持范围。
实际首次restart状态选择器选中返回的helper season对象，包含912.5天和原长期温度；
只在下游科学计算前捕获参数并停止，没有执行模型日或科学核。

### 原319/1961年度NPP运行是否触发

若长期温度有效而tau缺失/sentinel，修复会改变季节记忆的时间尺度，可能影响后续
NPP；不能概括为对所有运行都无影响。对同一319/1961反例在第三阶段的身份绑定
复现运行，
[身份与路由证据](../../outputs/reviews/main-physics-20261023/restart-tau-fix/P15/driver/319-association.md)
则给出具体反证：已保存manifest与实际argv绑定producer
`09968cf86551e8dfc039c1740b94fb3d42cf3f12`、319/1961及`--initial-state cold-start`；
三份restart和本次`used_run.def`当前字节均匹配该manifest，不是另一个同名模板。
compiled与strict回执具有相同producer、manifest路径/哈希和所选输入；已直接读取
该producer的Git源片段确认runner、context、helper、cold初始化及后日overlay路由。
相关runner/orchestration/reference当前源与该producer相同，helper默认值和生产
cap传递的修复差异单独列明，不从当前源码倒推历史。
STOMATE模板来自所选319的`reference_run_dir/stomate_start.nc`，长期温度为
`290.1180918236078 K`、tau为`1095`天，配置`COEFF_TAU_LONGTERM=3`也对应1095天。
tau状态报告为`restart`，并非由回退生成。

不能仅凭cold-start标签排除helper：准备context确实调用restart aggregator及helper。
首日科学路由随后从独立cold-start season构造器取得tau=2，后日overlay则携带模型
产生的tau。该区别约束初始读取与首日消费者，不把后续tau作用解释为零。
[轻量读取对照](../../outputs/reviews/main-physics-20261023/restart-tau-fix/P15/driver/readstart-cap-comparison.json)
在同一身份绑定模板上实际调用当前helper两次，显式3天模拟旧默认与配置1095天
比较：entry/season/daily/gas/remainder五组及report全部精确相同、无差异字段，
两次均保留tau=1095。没有运行旧Git源码或模型日。
因此这次补丁没有改变已检查的身份绑定运行初始读入状态，该输入不触发已修复分支；
不支持将该条件缺陷作为原319年度NPP失败的解释。
此身份桥明确覆盖第三阶段两次复现：compiled误差为`0.0008192662350708368`，
strict为`0.0008192662355900326`，两者均超过原门槛；不把这个producer身份替代
其他保留运行，也不用这些相近数值替换第六阶段审查保留的原年度失败值。

原319年度NPP失败继续保留为`0.0008192662350707258 > 0.0003951849734783173`；
本阶段没有重跑年度轨迹，也不以这些读取回归宣称年度差异已解决。
第十四阶段的3/1095天反例是修复前历史证据，旧源码/AST身份不重标为修复后身份。
水分起源调查和第十三阶段下一捕获继续暂缓。

## 第十六阶段：公共入口到compiled块的输入绑定

本阶段限定为公共CLI/runner的点与年，经reference/config/forcing选路、半小时插值和
日组装，进入compiled块及最终输出归属的一条链。以精确源码及既有记录为主，
没有重新执行模型日/年、真实XLA编译或科学核。详细来源、静态/动态表和原证据范围见
[Stage16入口](../../outputs/reviews/main-physics-20261023/compiled-input-binding/README.md)。

### 公共入口、context及forcing身份

[上游来源表](../../outputs/reviews/main-physics-20261023/compiled-input-binding/upstream/upstream-binding.md)及
[19份源/既有证据身份回执](../../outputs/reviews/main-physics-20261023/compiled-input-binding/upstream/source-receipt.json)
明确了各个缓存与归属边界：

| 边界 | 已核对的身份与限制 |
| --- | --- |
| 公共CLI选点 | 单次invocation选择一个landpoint，默认用其归档`used_run.def`物化点位专属run.def，并选择其reference/restart目录。显式`--run-def`优先于默认物化，但reference和输出点标签仍来自所选landpoint；标签不能单独证明显式文件与该归档case相符。 |
| prepared context | `lru_cache(maxsize=8)`按解析后的config路径、run.def路径、reference目录或null缓存；point/year不是独立key。默认路径/目录间接携带点身份，实际年由外层year loop及逐步forcing准备传入。该key表示路径身份，不是内容哈希或显式输入配对验证。 |
| forcing到日组装 | forcing缓存包含解析config、实际year及全局零基land indices；局部Fortran `kindex`转换为相应二维位置。`model_tstep // split + 1`及`model_tstep % split + 1`分别给出forcing记录和记录内split；日/block组装使用当前year和绝对start tstep。 |
| Fortran时间合同 | paper来源是`dim2_driver → forcing_READ → readdim2::forcing_read_interpol`的6小时CRUNCEP/1800秒路径。`orchideedriver → forcing_getvalues → forcing_tools`仅列为另一套interval API旁证，不拼接成paper实际调用链。 |
| 输出点/年 | 外层runner保留所选point/reference及实际year，重建每日day/tstep标签并写入逐年summary和最终JSON。块数组本身没有文本point ID，其归属必须连同上述外层绑定判断。 |

旧669包审计保存669/669通过和九个varying keys：四个LIMIT边界及PFT14的
VCMAX25、ARJV、maintenance slope、allocation minimum、residence time。
LIMIT经domain进入forcing/landpoint数据，五个校准坐标经STOMATE四向量及DIFFUCO
参数进入当前消费者；materializer固定本paper路径的PFT14/vegetation结构协议。
但varying-key分母只是669份case `run.def`全部key，并非全部实际`used_run.def`的
静态值比较；摘要没有source commit，只作原有旁证，没有重新运行该审计。
这是真实输入分母未验证的证据缺口，不直接判成bug或排除为不受支持。

### compiled消费者与连续块

完整块factory的key包括config解析路径、block长度、矿质土索引界、状态component/field
布局、边界捕获开关及projector key；它确实保留首次context和daily-carbon dispatch。
点/年没有出现在key中本身既不能证明错误，也不能证明所有配置变化均安全。
[当前消费者表](../../outputs/reviews/main-physics-20261023/compiled-input-binding/compiled/binding-review.md)
逐项追踪以下九组运行时实参，而非仅核对函数签名：

| 实参族 | 实际消费及写回 |
| --- | --- |
| 前一日状态、逐日forcing、science day | 进入完整日/半小时scan、日累计及季节/年末门控；返回当日状态与输出。 |
| STOMATE四组参数、HYDROL十组表数组、landpoint payload | 当前case的连续参数、土表及空间/陆面值覆盖相应静态模板并传至实际消费者。 |
| restart entry模板、season值、DIFFUCO参数 | 与要求的前一日carried字段overlay；DIFFUCO连续参数重建后进入首步及后续scan。 |

`science_day_numbers`控制实际日号、季节推进和年末判断；逐日forcing已经包含实际年
CO2及气象。内部`year=1961/day_index=2/start=48/tstep=47`作为局部源顺序metadata，
不能据此认定后年气象被锁定到1961。每块结束从`final_values`重建previous packet，
下一块从该状态开始；返回日标签及绝对tstep使用当前块索引。

九组动态值没有覆盖全部context。时间/shape、PFT角色和结构控制、固定数值表、
非override的参数与季节尺度等仍被闭包特化；必须结合实际paper输入不变量判断，
不能将任何允许的case参数变化一概解释为任意配置改写。未核对的配置变化仍保留为
证据限制，不作无条件跨配置复用认证。

[一次接口捕获](../../outputs/reviews/main-physics-20261023/compiled-input-binding/compiled/transport-capture.json)
实际调用当前factory/runner，在调用前替换jit/lower/compile、scan及首个模型边界。
A1961、B1961、B1962共享一个替身callable，各处理2–8与9–15两个七日块；
42个日界、869项断言通过，检查当前参数/forcing、day标签及第8→9日carry。
真实模型步、真实编译和科学核调用均0。该结果证明Python接口传递，
不作为真实XLA可执行文件的跨点数值等价证据。

历史14日cold/restart及365日完整块A/B继续复用；旧三点记录直接证明的是
SECHIBA scan复用和相应日/年及restart比较，不能改称本阶段完整块factory的实测。
七点50年通过及其producer限制保留，不重跑或重标。原319年度NPP失败仍为
`0.0008192662350707258 > 0.0003951849734783173`，水分起源捕获继续暂缓。
本阶段未取得新的绑定bug最小反例，未提出或自动实施生产代码修复。

## 已确认限制与优先检查清单

完整 669 点验收未完成。保存 certificate 的 `equivalent=false` 同时引用旧
stage6 2/3 代表点结果，不能把该旧 false 直接当作当前实现失败；反过来也不能把
七点通过扩写为 669 点通过。`process_span_unmapped=4732` 是诊断性 span 计数，
证书已有另一套 arm 分类/owner 路由，不是 4,732 个未实现过程。
HYDROL peat hydraulic coefficient 条目虽然命名含 conditional，但实际 reachability
为 inactive，并给出了 audited runtime `peat_hydro=F` 的来源理由；其 regression-only
不能脱离有效配置解释成可达缺口。该False的范围须保留：第三阶段仅确认关联case001 trace；
第十二阶段直接确认319本次选定24条重样本的HYDROL-private值False，不扩展为全年分支证明。
更一般的潜在输入域仍受原交付合同约束。

下列四项保留审查对象及本阶段后的状态；它们不是四个待修bug，也不自动派发下一轮检查。

1. **证据到当前物理版本的关联（证据待补）。**
   七点、source/owner 和 state 档案已有很强的通过证据，但若干原始 JSON 未内嵌
   JAX producer SHA。入口为本轮 inventory 的 production_change_history、
   `scripts/dev/audit_pft14_equivalence_certificate.py::build_certificate`、
   `deba027` / PFT identity 与 restart 提交 / `5910ba9` 以及已接受 C1–C3。
   本轮以 HYDROL 示例核对原 Fortran身份、当前函数 AST 和具体 guards；第十四阶段进一步
   完成选定restart/season字段的producer、23函数/11常量及当前配置桥，见上述版本桥。
   最小后续办法是关联命名原执行记录与 Git 源片段，对有科学影响的差异选择既有
   小型 Oracle/回归；相同代码复用证据。尚未核定这些档案的完整当前版本桥，
   这不撤销原通过结果，也不要求首轮穷尽 producer 历史或重跑长序列。

2. **319 高风险历史长序列及当前首年科学偏差（历史身份仍待补；第三阶段已取得当前首年反例）。**
   [第一阶段失败值](../../outputs/reviews/main-physics-20261011/named-high-risk-history.json)
   和 [第二阶段 receipts](../../outputs/reviews/main-physics-20261011/319-history/receipts.json)
   保留第一阶段 069 保存 50 年链通过的结论，以及两个 319 失败和反证：
   A 为已验证的 21+29 年恢复链、首超限 1961 NPP；
   B 有匹配的相邻四分片命令、首超限 1963 NPP。两条 producer 均未核定，
   本地三条完整同点候选均未过原门槛，未建立有效替代；单年通过及七个其他点
   不能关闭它们。第三阶段的当前 identified cold-start 1961 NPP 同样超限；
   第四阶段定位 maintenance 年度负残差及当前 day199 土温输入效应；同输入 Oracle
   通过，局部效应与年度主残差异号。第六阶段已捕获当前完整 compiled 日总量；
   相对历史 trace 的全年净差约 `−0.30914416 gC/m²`，Day337–343 为主脉冲窗，
   当时 trace producer、定义及发送日边界未核定，逐日差是暂定定位。日末归约/输出
   舍入远不足以解释该差，但没有界定内部轨迹误差。年度因果仍未闭合。
   第七阶段恢复候选logger字段/位置，9天原始累积与LPJ GPP/名义步号一致；
   严格零GPP、泵取和floor的成对源条件已核对；第八阶段恢复命名历史源码、
   二进制、有效XML及三份保存trace文件桥，在用户覆盖范围完成真实Fortran/JAX
   年各1次和9天完整entry/创建量/post-NPP/send捕获。Day338/339/343完整同输入
   NPP回放18/18通过；三日原轨迹在entry已有GPP/biomass/allocation/维持呼吸差异。
   direct core→send36/36、旧日志所选tuple27/27通过；三项float32 storage→send
   double诊断超限仍保留。原年度NPP失败没有关闭，具体上游来源和年度因果未定。
   第九阶段已核对实际GPP生成/累计/reset源路径与48项声明参数，完成指定9天
   保存值分析；缺少48步源元组，仍未确认掩码/叶层/累计或相关状态创建原因。
   第十阶段两侧Day337的48步真实捕获均完成；Fortran第七次模型年完成后因parser拒绝而
   调度FAILED，独立解析恢复保留全部raw及96对同值重复来源。原33/34的年度GPP差
   已由两份现存NetCDF的高精度导出独立解释，原严格检查器34/34通过。
   48步中唯一活动/GPP分支差在JAX16136/Fortran16137；JAX humrel=0使其活动门槛关闭，
   Fortran该步跨过门槛。去重后的1个真实时刻、2次完整65参数生产重放均通过，
   Fortran输入的8个有效叶层及两类根候选提供局部同输入证据；跨侧有效交集仍为0。
   父级47项工件核验通过。第十一阶段保存值进一步定位到前一步offset7的vegstress差异，
   成对源链支持HYDROL创建→下步TRANS。第十二阶段已实施offset6–8原生HYDROL/carry
   及48步轻量观测，两侧全年捕获和严格自身准入均完成。配对记录直接显示offset7
   Fortran液水略高于共同阈值、JAX低于阈值；两侧根权重均1，下一步精确读取各自创建值。
   进入土壤更新前液态水已不同，优先边界转为创建器上游状态/历史；该有界解释与原NPP
   核心回放均不扩展为全年因果或全分支正确性证明。
   第十三阶段118项保存profile核对进一步闭合所选两次相邻`mc/mcl`桥及JAX日界桥；
   offset7第6层积分差主要来自节点5，选定点的总水和液水相等、冻结profile为0。
   当前差异已进入carried总水历史；未取得完整同输入土壤元组，尚不能确定更早起点。
   第十三阶段为最早已保存步附近的前一步恢复总水→完整主更新元组/输出准备了
   字段/相位合同，实际捕获和同输入测试均未执行。第十四阶段按后续授权转查
   restart/跨年状态合同，暂缓沿该水分起源继续追加运行。
   不从科学失败或最后半小时解释比例直接认定物理函数 bug。
   当前 HYDROL/年边界已核实的来源和反证继续有效，不从旧失败直接修改 root 或门槛。

3. **当前三文件 restart 与年接续状态合同（选定计时/日累计/季节记忆已审查）。**
   入口为 `_paper_day_end_state_packet`、`paper_1961_driver_restart_year_start_day_result`、
   [restart lifecycle 合同](restart_state_lifecycle.yaml) 和 stage5 的 181 字段记录；
   Fortran writer 来源为 ledger 所记 `stomate_io.f90::writerestart` 2200–2948，
   HYDROL writer / 年初始化范围如上。已有 001/069/319 全年直接/split restart
   与离散/模式检查通过，nroot 删除疑点已排除。
   第十四阶段已核对上述三组状态的axes、mask、时间归属、重算和首次消费者，补上选定
   reader/writer/函数及配置版本桥；正常完整链未发现所选状态丢失。tau helper缺失/sentinel
   输入的3/1095天回退差异已用微型状态反例确认，源路由区分缺字段拒绝和sentinel可达。
   第十五阶段已修复helper天数默认值与生产配置传递，相关31项读取/选择检查通过；
   正常writer有效tau不触发该条件，缺长期温度仍优先tau=2，显式上限仍按天使用。
   这些结果不关闭319年度NPP失败或其年度因果。
   全部181字段仍复用原闭合结果，非本阶段逐项重审；旧运行SHA/外部RUNDEF关联限制保留。

4. **七日 compiled 块跨点绑定与年输出（当前有界源链及接口传递已查，完整静态输入分母未认证）。**
   入口为 `_paper_compiled_later_day_block_executable` 11185–11355、
   `_paper_run_compiled_later_day_blocks` 11358–11471、
   `read_forcing_model_step_cached`、`audit_paper_landpoint_binding`、
   `compute_modelout_from_fields`。账本给出 Fortran `sechiba_main` 997–1224、
   `forcing_read_interpol` 660–1687/1899–1968；本轮已核对 paper postprocessor
   48–76/115–118 的 PFT14 选择、碳库相加及 AGB/BGB `0.02` 转换。
   已有 14/365 日完整块A/B、三点SECHIBA scan reuse、七点block=7的完整50年科学通过。
   当前 block 的 forcing/science day、参数、landpoint payload 与 restart template
   是显式动态参数，`year=1961` 的静态 metadata 本身不是锁定所有年 forcing 的证据。
   第十六阶段补上公共CLI→runner→forcing→完整块/半小时消费者→输出的有界来源链，
   一次替身边界捕获通过；不将其作为真实完整块XLA复用或数值等价的新执行。
   旧669包绑定审计的varying-key分母是case `run.def`全部key，并未逐键比较全部
   实际`used_run.def`的静态值；这一真实输入证据缺口不等于已确认bug，也不能直接
   排除为不受支持。完整跨点静态语义安全、当前全分支及669点积分验收均未认证。

## 本轮基础审查的有限收尾与停止条件

本轮已覆盖的是证据地图及命名高风险边界的盘点/有界审查：Fortran分支/Oracle/生产
档案的层次与范围，HYDROL/root及年界，三文件restart、计时/日累计/季节记忆，
C1–C4修复/回归档案，以及本阶段公共输入到compiled消费者和输出的绑定链。
这些记录不等同于本轮重审全部源分支或重新执行所有Oracle。

本轮收尾只有下列四项必查。表中的验收条件限定本次交付，不扩展成新的调查任务：

| 必查项 | 已有证据与具体剩余缺口 | 本轮验收条件 | 可否独立并行 |
| --- | --- | --- | --- |
| 公共入口、prepared context及forcing/输出归属 | 既有669包绑定摘要、当前CLI/runner/domain及Fortran时序源；已补context缓存身份和显式路径优先级及19份源/证据身份，旧producer和used-file分母缺口登记。 | 点/年/config/reference/run.def与forcing索引到输出的路由可逐段追溯；旧摘要无producer及未查used-file静态分母明确记录。 | 可，与下一项独立；已由两位原owner并行。 |
| 完整块闭包、九组动态消费者及连续块carry | 当前43个精确锚点、14/365日历史A/B及一次869断言捕获；已将上游九个实际变化坐标与静态假设合并，未认证静态变化明确保留。 | 逐项说明动态值及实际消费者、捕获的静态值和未认证变化；不混同旧SECHIBA复用与当前完整块，不从缺point/year key直接判bug。 | 可，与上一项独立；最终合并依赖上游表。 |
| 证据范围、未决项及本页结论一致 | 原319失败、历史producer限制、C4及Stage14/15记录、上述两张表。 | 无相互矛盾或扩大通过范围；若发现具体反例，只交最小反例、影响和修复计划，不自动改生产代码。 | 合并验收，依赖前两项；不需新worker。 |
| 最终工件与工作区绑定 | baseline、源/旧证据身份、报告及delivery/index。 | 固定root/main/HEAD；仅本页未暂存修改、空index、链接/身份通过、旧45份stat及6个anchor未变，排除执行为0。结果以Stage16交付JSON为准。 | 串行，所有文档冻结后一次绑定。 |

两路主表交付时，收尾估计为context/机器回执补交、报告合并及最终绑定，
1–2轮、约10–25分钟工作时间；这是原owner按当时实际进度给出的粗估，不含主控人工
等待。身份/链接不一致可能增加一轮返修；不为该估计启动实验，也不据此估算未来科学
验收时长。补交实际约10–15分钟且已完成；两路来源表及报告内容已验收，最终绑定结果
由交付JSON记录。交付与核验完成后，本执行聊天没有自动续行任务，等待主控一次审阅。

以下事项**不阻塞本轮有界基础审查收尾**，仍限制更强的等价/发布结论：

- 原319年度NPP科学失败`0.0008192662350707258 > 0.0003951849734783173`、
  两条历史长链的producer缺口和年度因果继续开放；Stage13水分起源捕获继续暂缓。
- 未查全部实际`used_run.def`静态key、未完成全部旧证据到当前物理版本的关联、
  未重新审查全部181字段/所有分支，不能宣布完整跨点缓存语义或全分支认证。
- 七点长期通过不替代669点最终科学门槛；绑定摘要、源码追踪、接口stub与独立Fortran
  数值Oracle分别保留层次。C4研究分支同步由对应维护线程负责，不作为本聊天续行项。

建议停止在上述四项满足、未决项如实登记且最终工件绑定通过的边界。
不以“所有输出完全相同”或追出319全部上游历史作为本轮退出条件；后续科学结论仍须
满足既有交付合同，本报告不降低其门槛。

## 本轮执行记录

[第一阶段命令和结果](../../outputs/reviews/main-physics-20261011/commands.md) 记录实际本地读取、
保存值科学门槛复核、Git/AST/Fortran 身份核对以及辅助脚本首次导入失败后的修正。
第一阶段新增本页和 `outputs/reviews/main-physics-20261011/` 辅助证据，并更新
`docs/RESEARCH_WORKFLOW.md` 导航。未修改生产源码、测试、科学阈值或旧产出。
未执行 pytest/GPU/长模型运行、训练、服务器作业、Teacher/prepared 生成或数据重标。

第一阶段图工具恢复可用：项目 `E-ORCHIDEE-MAN-JAX`、generation
`2026-10-10T08:46:50Z`、12,100 nodes / 72,107 edges。
[图记录](../../outputs/reviews/main-physics-20261011/graph-receipt.json) 和
[补充覆盖](../../outputs/reviews/main-physics-20261011/graph-additional.json) 明确显示
目标源码 metadata_changed，Fortran 在 excluded subtree；关键结论直接读当前源码，
不把旧图当当前完整性证明。采用 Verify 层级，不作全项目不存在/完整覆盖断言。
最终 `git diff --check`、新报告空白/相对链接检查及证据引用存在性检查见
[verification.json](../../outputs/reviews/main-physics-20261011/verification.json)。

第二阶段只更新本页和 `outputs/reviews/main-physics-20261011/319-history/`。
[第二阶段命令](../../outputs/reviews/main-physics-20261011/319-history/commands.md)
记录保存值读取和辅助脚本的 scalar/list checkpoint 天数兼容修正；
[图记录](../../outputs/reviews/main-physics-20261011/319-history/graph-receipt.json)
仍为 Verify，generation 相同、源码 metadata_changed、产出目录 excluded，
关键身份结论直接核对当前源码和 Git 片段。
[第二阶段 verification](../../outputs/reviews/main-physics-20261011/319-history/verification.json)
记录链接、证据引用、原产出身份、基线、修改范围及 `git diff --check`。
未改生产源码、测试、科学阈值或原产出，未执行长模型/GPU/训练/服务器工作。

第三阶段只更新本页和 `outputs/reviews/main-physics-20261011/319-current/`。
实际执行两个当前有身份的 1961 单年模型；未续跑 1962/1963、50 年或 669 点。
[命令](../../outputs/reviews/main-physics-20261011/319-current/commands.md) 及原始 receipt
分别记录执行完成与科学失败；辅助 analysis 只读取现有产物并写新 sidecar。
[图记录](../../outputs/reviews/main-physics-20261011/319-current/graph-receipt.json)
保留所提供的 Verify generation，但本阶段 index_status/trace_path/coverage 的
本地调用为 `Transport closed`，不声称 fresh MCP 或全模型完整性；关键结论直接
核对当前有界源片段。新
[verification](../../outputs/reviews/main-physics-20261011/319-current/verification.json)
记录实际 input/result/checkpoint 身份、原政策复核、科学输出与状态有限性分离、
语义相等、链接与修改范围及 `git diff --check`。未改原 JSON/pickle/receipt、生产
源码、测试、科学阈值或原通过结论；未执行 pytest/GPU/训练/SSH/服务器作业。
提交与推送由主控在实际产物验收后执行。

第四阶段只更新本页和 `outputs/reviews/main-physics-20261011/319-npp-local/`。
[命令](../../outputs/reviews/main-physics-20261011/319-npp-local/commands.md) 记录保存值分析、
历史状态一天回放、当前 cold-start 198 日前缀和同年 day199 回放；未重复全年模型。
[verification](../../outputs/reviews/main-physics-20261011/319-npp-local/verification.json)
核对原产物/冻结输入保留、当前前缀身份、两组各十项原 Oracle 检查、年度/局部异号、
三份审计的字面源码、引用和修改范围及 `git diff --check`。
[图记录](../../outputs/reviews/main-physics-20261011/319-npp-local/graph-receipt.json)
沿用所提供的 Verify generation；本地工具仍为 `Transport closed`，采用有界源码回退。
未改生产源码、测试、科学阈值或旧产出，未执行 GPU/训练/SSH/服务器作业；
提交与推送留由主控验收后执行。

提交前补充仅分析已有198/365日摘要及365条 LPJ 日值，没有新模型执行。
[补充 verification](../../outputs/reviews/main-physics-20261011/319-npp-local/verification.with-periods.json)
保留原九组终验并增加分段口径/输入身份、差分来源、舍入与 ε 限制核验；
原 verification 和日志保留，原科学阈值及年度/局部异号结论不变。

第五阶段只更新本页和 `outputs/reviews/main-physics-20261011/319-late-year/`。
[命令](../../outputs/reviews/main-physics-20261011/319-late-year/commands.md)
记录命名保存值读取、字段盘点、晚年历史窗口与源码边界核对；没有新模型积分或 trace。
复用 Luna/high owner 完成来源摘录和未来捕获脚本；首轮
[verification](../../outputs/reviews/main-physics-20261011/319-late-year/verification.json)
八组通过，核对21份读取原证据、303行字面源码和82个当时的报告链接。
未来脚本补齐源码版本守卫和新旧摘要/ε 记录后，只增加针对性 dry-run/纯计算检查及
终版链接、范围检查，见
[终验](../../outputs/reviews/main-physics-20261011/319-late-year/verification.final.json)。
[图记录](../../outputs/reviews/main-physics-20261011/319-late-year/graph-receipt.json)
保留 Transport closed 与新增已知来源的 coverage 回退；不声称 fresh/full 图覆盖。
stage4 的分段/Oracle 通过证据未重跑；生产、测试、科学阈值及所有旧输入/产物保留。
未执行 GPU/训练/SSH/服务器作业、暂存、提交或推送；交付主控验收后待命。

第六阶段只更新本页；一次获准的本地全年捕获和逐日分析记录在
[执行回执](../../outputs/reviews/main-physics-20261012/319-daily-capture/execution-receipt.json)、
[分析](../../outputs/reviews/main-physics-20261012/319-daily-capture/analysis.json)
及同目录命令、preflight 和 stdout/stderr。可用[分析脚本](../../outputs/reviews/main-physics-20261012/319-daily-capture/analyze_daily_capture.py)
对已有运行目录复核，不会重新积分模型：
`conda run -n ORCJAX python -B outputs/reviews/main-physics-20261012/319-daily-capture/analyze_daily_capture.py --run-dir outputs/reviews/main-physics-20261011/319-late-year/future-diagnostic/runs/capture-20261012T012052Z`。
终验记录见[verification.final.json](../../outputs/reviews/main-physics-20261012/319-daily-capture/verification.final.json)。
捕获脚本与物理源码未改；不重跑模型，不暂存、提交或推送。

第七阶段只更新本页和
`outputs/reviews/main-physics-20261012/319-pulse-boundaries/`。
[命令记录](../../outputs/reviews/main-physics-20261012/319-pulse-boundaries/commands.md)和
[verification.json](../../outputs/reviews/main-physics-20261012/319-pulse-boundaries/verification.json)
记录实际9天边界核对、逐行源摘录/账本/链接检查、原输入保留及修改范围。
离线分析可用
`conda run -n ORCJAX python -B outputs/reviews/main-physics-20261012/319-pulse-boundaries/analyze_pulse_boundaries.py`
复核。[图记录](../../outputs/reviews/main-physics-20261012/319-pulse-boundaries/graph-receipt.json)
为可用MCP的Verify：generation仍为`2026-10-10T08:46:50Z`，相关JAX
metadata_changed、Fortran/历史记录excluded、新产物not_tracked；均以当前精确源码
和直接记录补足，不声称fresh/full图覆盖。没有重跑第4–6阶段验证，没有生产/测试/
容差修改、模型/Oracle/Fortran新trace、SSH/服务器/GPU/训练/付费工作；执行线程
交付时未暂存、提交或推送。主控已核对新增报告、实际产物和关键成对源码，
复用上述17项通过的轻量核验。本轮有界证据验收通过，不代表年度科学验收
通过或因果归属已经确认；下一阶段尚未启动。

第八阶段最初准备检查点只更新本页；新脚本、隔离 Fortran 副本和回执均在
`outputs/reviews/main-physics-20261012/319-matched-capture/`。
[命令记录](../../outputs/reviews/main-physics-20261012/319-matched-capture/commands.md)、
[访问回执](../../outputs/reviews/main-physics-20261012/319-matched-capture/access-receipt.json)与
[verification.json](../../outputs/reviews/main-physics-20261012/319-matched-capture/verification.json)
记录实际本地预检、观测撤除/原字节恢复、冻结输入/生产源码保留及最初访问阻塞；
原16项资料核验未重跑。主控新增 ACL 修复和两次 exit255 认证日志保持原样，
新增访问事实、文档一致性及路径的9项轻量核验通过；旧身份服务端认证拒绝记录另见
[收尾核验](../../outputs/reviews/main-physics-20261012/319-matched-capture/verification.access-closeout.json)。
[图记录](../../outputs/reviews/main-physics-20261012/319-matched-capture/graph-receipt.json)
采用 Verify，沿用同一 generation；JAX metadata_changed、Fortran excluded、
新产物 not_tracked，以当前精确源码和直接产物检查补足，不作 fresh/full 图声明。
该历史检查点尚未执行真实匹配采集及全年等价/科学验收，没有生产源码、测试、
容差修改或新年/点位、GPU、训练、付费工作；当时交付未暂存、提交或推送。
主控已集中验收本地准备及当前认证阻塞的报告和回执，复用原16项准备核验
和9项收尾核验，作为进度检查点保存。当时完整第八阶段捕获仍未完成，等待可用
连接信息；该验收不增加科学通过或因果归属结论。

随后用户提供新身份，历史恢复检查点从 `9c6ef25a` 开始。新认证成功后的实际只读
身份恢复、原环境 version admission 和实际源码日志副本见
[本次命令](../../outputs/reviews/main-physics-20261012/319-matched-capture/commands.new-identity.md)、
[身份分析](../../outputs/reviews/main-physics-20261012/319-matched-capture/restored-identity-analysis.json)与
[恢复执行核验](../../outputs/reviews/main-physics-20261012/319-matched-capture/verification.restored-access.json)。
新增12项证据/文档核验通过；[本次图记录](../../outputs/reviews/main-physics-20261012/319-matched-capture/graph-receipt.new-identity.json)
仍为同一 generation 的 Verify，相关产物 not_tracked，以直接来源/产物补足。
旧记录均保留；该检查点的可用计算分配及计费/额度条件尚待核实，尚未提交作业、
构建/运行真实 Fortran 或新 JAX 年，没有科学通过/bug 归属。主控已核对当时报告与新增
身份恢复证据，复用上述12项核验，接受历史来源恢复及本地准备的进度检查点。
该次验收仅涉及这些资料，当时真实匹配捕获仍未完成；不赋予后续执行身份。

本次再从已接受并推送的 `e05369c9e70b4c982f34545e37af7c1782d8142d`
续作，先完成有界计算/结算核实及可审阅命令准备，详见上述计算条件记录与执行方案。
已纠正构建 cwd、绝对 FCM 链接、副本写路径和 NetCDF 后处理边界；复制模型估计
21,145条目、965,885,992字节。方案是单节点8核/16GiB/2小时上限，编译最多30分钟，
单核模型最多60分钟；16 CPU-hours 只是按8核预留时间计算的资源上限，不是货币
费用。查询本身未取得结算信息；用户后续已明确确认已有额度覆盖该资源包，
可以继续原捕获。不能给出最坏货币费用或免费结论。新增九项检查通过，目标
Bash 4.2.46 无写入 dry-run exit0，历史 builder 第511行确认输出位置；见
[本次交付核验](../../outputs/reviews/main-physics-20261012/319-matched-capture/compute-admission-20261012/verification.final.json)。
该准备检查点未重跑原16/9/12项、旧Fortran语法/AST/callback、Oracle或模型；
其后用户明确授权继续，创建新参考根并在原资源范围完成隔离来源/运行库修复、
一次成功 Fortran 年、一次原本地 CPU JAX 年和三次完整同输入回放。前两次分配
都在模型启动前停止，修复没有新增全年积分。18项完整返回数组、36项直接发送
边界和27项旧日志所选tuple比较通过；三个存储double诊断超限和年度科学失败保留。
新[最终执行核验](../../outputs/reviews/main-physics-20261012/319-matched-capture/matched-comparison-20261012/verification.final.json)
仅检查本次实际产物、身份、严格比较、文档链接及修改范围，不重跑旧检查或模型；
[本次最终图记录](../../outputs/reviews/main-physics-20261012/319-matched-capture/matched-comparison-20261012/graph-receipt.final-execution.json)
沿用同一 generation 的 Verify，以精确来源和实际捕获补足 metadata_changed /
excluded / not_tracked，未重新索引或声称完整覆盖。本次有界捕获和比较已完成，
原 NPP 科学门槛仍未通过，未新增生产 bug 归属。主控已集中核对本轮报告差异、
实际执行/输入身份、完整同输入比较、原JAX年度输出及直接发送/存储边界记录，
复用执行线程16项通过的产物核验，接受本轮有界捕获和比较。未重复积分或原有
验证；三项存储诊断超限、原年度科学失败及未决上游原因均保留。本轮只提交
本页报告，通信保持全手动，下一阶段尚未启动。

第九阶段从`85ace286`续作，只更新本页和
`outputs/reviews/main-physics-20261013/319-gpp-upstream/`。复用两位Luna/high执行者
完成命名producer源图/库存核对与离线窗口分析；父级集中核对实际产物、当前
生产调用路径、源级反证、唯一下一观测合同和报告。收尾记录见
[commands](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/commands.md)与
[verification.final.json](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/verification.final.json)。
[本次图回执](../../outputs/reviews/main-physics-20261013/319-gpp-upstream/graph-receipt.json)
为同一generation的Verify：JAX metadata_changed、源/记录not_tracked，均以
精确当前来源和直接保存产物补足，不重索引或声称完整覆盖。没有生产/配置/
测试/门槛改动，未执行新模型年/prefix、Oracle、NPP回放、构建、计算分配、
GPU或训练，未重复旧验收。本次只读命名producer取证没有服务器写入；没有
自动消息，执行线程交付时未暂存、提交或推送。主控已集中核对实际来源身份、
保存窗口分析、候选与反证、下一观测合同及报告差异，复用上述20项通过的产物
核验，接受本轮有界检查。该验收不代表差异根因或年度科学失败已解决，也不
代表新观测器已实现或可直接运行；下一步仍须实现Day337的48步观测并验证其
不改变原轨迹。本轮仅提交本页报告，保持手动跟进，未启动下一阶段。

第十阶段原JAX进度checkpoint从`638253db`续作，保存3次年度启动、2次365日完成、
1次成功48步capture及3次route diagnostics。2026-10-14主控接受该JAX成果和当时的
Fortran本地恢复准备；[原24项验收](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/verification.final.json)
及捕获、287项链路/3项日末检查、195条journal保持原样。该历史接受点仅覆盖当时JAX部分。

本次从`3f906c62`继续，恢复后的Fortran第七次job124444469完成原365日模型；
调度FAILED来自后处理parser，独立salvage与9项raw恢复核验完成。
默认CDL精度引起的唯一自身年度GPP失败，经两份现存NetCDF只读高精度导出确认，
原检查器34/34严格准入通过。原分析执行者完成48步跨侧分析、1个去重时刻的2次
完整真实输入生产函数调用，全部有效检查通过；后续报告错误修复复用完成journal，新增调用0。
[实际命令](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/commands.md)、
[完整结果](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/analysis/fortran-restored/README.md)
及[父级47项核验](../../outputs/reviews/main-physics-20261013/319-gpp-subdaily-capture/verification.restored.json)
绑定实际产物、同输入数值、调用历史、来源与资源证据，复核积分/重放均0。
Stage10累计1458秒，加Stage8共3354秒、剩余3846秒；本次收尾新增计算分配0。
执行方只版本化更新本页，其他新工件在原outputs目录；交付时未暂存、提交或推送，保持主控手动验收。
原年度NPP科学失败及既有存储诊断保留；唯一优先后续方向是该时刻humrel创建/传递链的上游追溯，
本次没有启动新的模型采样或修改生产物理实现。

主控于2026-10-15集中核对实际高精度导出的文件身份、四通量比较、原严格34项准入、
48步分析及选定时刻的完整65参数重放记录，复用执行方47项工件核验，接受第十阶段交付。
默认文本精度造成的额外GPP检查失败已解释；同输入光合作用局部比较通过，仍未确认生产
物理代码错误。原年度NPP失败及humrel上游来源继续开放，本次验收未新增模型积分或重放。
本轮仅提交本页报告，保持手动跟进；该次第十阶段验收时，上游追溯尚未启动。

第十一阶段从`6e9a60311d994f6fef1098d9f10b1ba461da1477`继续，复用原执行者完成
JAX/Fortran创建与写读源链、48步保存值/原生日边界分析、候选反证矩阵及准备态
下一观测方案；父级核对实际工件及新报告。命令与范围见
[Stage11命令记录](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/commands.md)，
29项离线数据/来源检查通过，见
[离线核验](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/verification/verification.json)；
最终链接、报告身份、Git范围、只读来源及工件索引见
[交付核验](../../outputs/reviews/main-physics-20261015/319-humrel-upstream/verification/delivery.json)。
采用Verify图层，generation为`2026-10-14T11:04:08Z`；Python metadata_changed及
Fortran/outputs排除范围以精确源/工件补足，未重索引或声称全覆盖。一次命名只读取证
无远程写入，原Stage10证据及7份受保护handoff文件保留，原34/9/47和真实调用不重跑。
本阶段新增积分/生产重放/构建/分配均0，余额仍3846秒，未改生产物理/配置/容差、
其他研究树或原产物。执行方未暂存、提交、推送、跨聊天发消息或启用自动化；交付主控
手动审阅，下一次原生观测尚未实现或执行。原年度NPP失败及存储诊断继续开放。

主控于2026-10-16核对报告差异、48步实际保存值、来源身份限制及下一观测合同，
复用29项离线和17项交付检查，接受第十一阶段有界调查。offset7入口vegstress与
offset8实际TRANS humrel分别记录；条件性的0.8关系不替代原生HYDROL写读观测，
也不证明根因。该验收时下一方案仍为准备态，须实现观测器并验证自身轨迹后才有新科学证据。
本次验收未新增积分、重放、构建或分配；仅提交本页报告，保持手动跟进和原科学失败。

第十二阶段从`91a1a79b3bf021f7fcae38e668c62d88f52c8d6a`继续，复用原执行者完成两侧
原365日轨迹的原生HYDROL/carry/TRANS观测。JAX唯一全年运行258.600996秒，
11990项直接Stage10、另行身份绑定的1460项Stage6日值、614项原生链核对及22项工件
核验均通过；Fortran三次分配814/16/417秒，前两次无模型，第三次成功完成增量构建和
原365日模型，实际1259条记录、47个label、7152项严格自身准入及24项工件核验通过。
配对266项保存值检查通过，新增生产函数/核调用0；父级读取实际准入、配对、来源、
编译/PP/binary、读出修复及终态回执，不重复模型或原有昂贵核验。

实际命令、读出失败/修复和资源结算见
[Stage12命令](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/commands.md)，
实际Fortran来源/终态绑定见
[Fortran工件核验](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/fortran/retrieved/attempt-3b/fortran-artifact-verification-124575036.json)，
最终文档、旧15份工件、HEAD/main、仅本页未暂存差异、资源/链接及联合索引见
[Stage12交付核验](../../outputs/reviews/main-physics-20261016/319-humrel-native-capture/verification/delivery.json)。
图层为Verify，已取generation`2026-10-16T01:53:45Z`、12152nodes/72230edges；
后续刷新为Transport closed，Python metadata_changed及Fortran/outputs排除范围以
已读精确源码/实际工件补足，不重索引或声称fresh/full覆盖。

Stage12累计分配1247秒，原额度7200秒减前3354秒后余额2599秒，无pending分配；
本地JAX另计，货币费率未知。只更新本页和Stage12新目录，旧产物及生产源码/配置/
精度/mask/容差保留，交付未暂存、提交、推送、跨聊天发消息或启用自动化。
所选阈值/一步读取桥已解释，但进入土壤更新前液水已不同，原年度科学失败及上游
状态/历史原因继续开放。本阶段交付不授权新的观测、计算或生产修复。

主控于2026-10-16核对本轮报告差异、原生选定步记录、双方自身轨迹准入、来源/读出
身份及实际资源账本，复用执行者已完成的49项交付核验，接受第十二阶段有界捕获与
配对解释。Fortran的offset7创建值与offset8实际reader直接一致，JAX选定链均为0；
原生液水位于共同阈值两侧，但进入土壤更新前的状态已经不同，不能把该比较当作
同输入求解器反例，也尚不能区分更早的浮点累积、状态历史或实现差异。JAX未单独
捕获的tile mask仍为明确限制，不据此追加全年捕获。下一唯一优先方向为利用现有
保存状态与成对源码，界定pre-soil液态水差异的上游来源和最小缺项。本次验收未新增
模型、核重放、构建或分配，仅提交本页报告；原年度NPP失败及手动跟进方式保留。

第十三阶段从`6f9d9f9340995376c7f79779072ee00b66485bb8`的干净main继续，复用原执行者
分工完成成对内层源审查、JAX外层状态源链及保存profile分析。父级核对实际118项新计算、
264行完整profile、JAX静态源guard回执、Fortran候选raw映射及精确重建/求解/恢复摘录；
没有重跑原数值准入。分析首次因旧相对路径的回执规范化失败，在固定root解析路径后
完成；未改变运算、保存数据或容差，失败/修复保留于
[分析执行记录](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/analysis/execution-record.json)。
源码核验确认重建前的调用区间及入渗/系数/setup三份helper均无`mcl`代码引用，重建后
先有`mclint/tmci`读取，再进入RHS；不把“先重建”误述为此前没有任何冻结或总水处理。

命令与实际边界见[Stage13命令](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/commands.md)，
图证据沿用Verify、generation`2026-10-16T04:15:46Z`、12156nodes/72234edges；本轮刷新
仍ready。metadata_changed的JAX及排除的Fortran/outputs由精确源/工件读取补足，未重索引
或宣称全面覆盖。最终文档/源/数据身份、旧Stage12十份哈希及全部185份文件stat签名、
HEAD/main/root、仅本页未暂存差异、空index和实际链接由
[Stage13交付核验](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/verification/delivery.json)与
[证据索引](../../outputs/reviews/main-physics-20261016/319-mcl-upstream/analysis/evidence-index.json)绑定。
stat签名是size/mtime变化检测，不是对185份文件新增内容哈希。

本阶段新增积分、生产核/Oracle调用、编译、SSH调查和分配均0；历史累计4601/7200秒，
余额2599秒、pending0。只更新本页和Stage13新目录；不暂存、提交、推送、跨聊天发消息
或设置自动化。已解释所选液水相位、状态桥与节点积分关系，但较早总水差的起源、
浮点累积/状态敏感性及年度因果仍开放。下一观测合同只作准备，未自动启动捕获或修复。

主控第十三阶段验收（2026-10-16）：已读取实际报告差异、成对源链、保存profile分析、
下一观测合同与交付回执，核对执行者交付时的文档、分析及证据索引身份；复用118项
离线检查和34项交付核验，没有重复模型或过程核验证。接受本轮有界调查结论，
不作年度科学通过或全分支正确性验收。旧入口`mcl`在当前主求解前重建，故该字段差
只能结合`mc`等实际输入理解；节点5的积分贡献也是数值分解，不能据此认定差异起源。
目前仍无新确认的生产bug，原年度NPP失败保留。下一观测合同具备源位置和字段依据，
但其前一步可能仍已带有差异，不能承诺解决年度起源；本次不自动执行新捕获。
主控建议将该差异保留为未决项、暂缓追加运行，后续优先回到源码覆盖清单，选择尚未
充分验证的过程开展有界检查；这是后续工作建议，尚未派发或执行。本次仅提交本页。

第十四阶段从`847a5f35`继续，复用既有两位owner完成计时/日累计与季节记忆、
历史producer/配置桥；父级读取实际表、源摘录、版本/配置工件和tau微型反例，
核清外部RUNDEF及missing/sentinel条件，完成本页及Stage14入口收尾。
一次ORCJAX/`-B`状态检查直接读1land/2PFT微型NetCDF，6次helper调用覆盖三种tau状态
和默认/显式配置，未调用科学过程或积分模型；实际命令和结果见
[状态检查执行回执](../../outputs/reviews/main-physics-20261018/restart-year-state/seasonal/execution-receipt.md)。
其余为源、AST、JSON、Git及工件身份检查，没有重跑Stage4–13数值套件。

图层Verify再次确认`2026-10-16T04:15:46Z`、12156nodes/72234edges；Python metadata_changed
及排除的Fortran/outputs以精确当前源/工件补足，未重索引或声称全覆盖。
[Stage14交付核验](../../outputs/reviews/main-physics-20261018/restart-year-state/verification/delivery.json)与
[证据索引](../../outputs/reviews/main-physics-20261018/restart-year-state/verification/evidence-index.json)
绑定最终报告、实际源/配置/AST/回执、固定HEAD/root/main、仅本页未暂存差异、空index及链接；
保护旧Stage12/13的210份size/mtime签名与3份anchor内容哈希，stat不是新增bulk内容哈希。
本阶段新增模型、科学核/Oracle重放、编译、SSH调查、服务器/调度分配、GPU/训练及付费工作均0；
历史累计4601/7200秒、余额2599秒、pending0。生产源码、配置、精度、mask和容差未改，
未暂存、提交、推送、跨聊天发消息或启用自动化；未执行Stage13下一捕获。
正常完整restart选定状态链已有证据继续有效；tau条件性默认值差异尚未修复，
原年度NPP科学失败及因果问题仍开放。本阶段交付本聊天手动审阅，不自动启动后续工作。

主控第十四阶段验收（2026-10-23）：已审阅实际报告差异、计时/季节字段表、版本桥、
微型NetCDF检查脚本与结果，并核对交付报告和关键工件身份。复用122项交付核验，
没有重跑模型或原验证套件。当前源码确认长期温度有效时helper的缺失/sentinel回退
消费`tau_longterm_max`，默认3.0与配置天数1095不符；生产入口先严格读取，再返回
该helper状态，缺字段与sentinel的可达性须分别判断。接受这一条件性读取缺陷，
不扩大为正常完整restart丢失状态或319年度NPP差异的原因。
接续工作限定为修正tau读取默认值及必要的配置传递、补上相关轻量回归和过时注释，
重点验证sentinel、有效值保留、长期温度缺失时的源初始化及显式上限；完整生产入口
的sentinel状态检查仍需补齐。第十四阶段报告先独立提交，修复由原Sol另行交付验收。

第十五阶段执行（2026-10-23）：从上述`e5032c3`干净main继续，实际root、branch
和ORCJAX导入路径已确认。复用codebase-memory Verify证据并再次确认项目ready；
generation仍为`2026-10-17T20:47:05Z`、12,165 nodes/72,243 edges，改动元数据
及排除的Fortran/outputs依靠精确源/工件补足，不重索引或声称全覆盖。
生产修改限于tau helper默认值、restart入口配置参数传递和相关docstring；两份
直接回归及本页一并交付。一次组合检查31/31通过，回执保留实际argv、完整输出、
五份被测源码身份及系统TEMP清理。pytest cacheprovider有意禁用，现有cache_dir
配置产生一条warning；没有失败或skip，不为此修改项目配置或重跑旧套件。
[交付核验](../../outputs/reviews/main-physics-20261023/restart-tau-fix/delivery.json)及
[证据索引](../../outputs/reviews/main-physics-20261023/restart-tau-fix/evidence-index.json)
绑定最终报告、实际源码/测试/补丁、回执和新证据，核对固定HEAD/root/main、六条
未暂存修改与空index，并保护旧Stage14的26份size/mtime记录和3份anchor哈希。
新增模型积分、科学核/Oracle重放、大型JIT/构建、SSH/服务器/调度分配、GPU/训练
及付费工作均0；历史累计4601/7200秒、余额2599秒、pending0。未暂存、提交、
推送、跨聊天发送或启用自动化。原年度NPP失败和暂停的水分起源工作保持开放；
本阶段交付本聊天审阅，不自动启动后续工作。

主控第十五阶段验收（2026-10-23）：已核对实际补丁、Fortran单位/回退优先级、
配置传参、真实入口回归及319身份/读取对照。执行者31项测试的五份被测文件身份
与交付代码一致；报告身份亦匹配交付回执。针对新增配置解析依赖，额外执行原有
无`run_def_path`生产入口回归1项并通过，见
[默认入口补查](../../outputs/reviews/main-physics-20261023/restart-tau-fix/checks/primary-default-entry.json)；
未重复已有31项或任何模型积分。接受该有界修复，原319年度NPP失败继续开放。
代码和两份测试独立提交为`4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99`；
跨分支同步范围、验证和待核对状态登记为
[共享修复C4](shared_core_branch_review.md#c4restart-tau回退的单位与配置传递2026-10-23)。
本次未修改其他研究工作树，其同步须由对应维护线程核对。

第十六阶段执行（2026-10-23）：从`6f20117badb229dc4e871b82d7cefd586d3dd258`
干净main继续，实际root、branch和ORCJAX导入路径确认。复用原两位owner分别
核对上游绑定与compiled闭包/消费者，父级审阅实际表、捕获脚本及JSON并维护本页。
一次ORCJAX/`-B`接口捕获通过；其余为源码/AST、既有JSON和工件身份读取。
既有forcing/landpoint/block测试只读来源，没有执行原套件。

图层Verify确认项目`E-ORCHIDEE-MAN-JAX`、generation`2026-10-23T07:15:27Z`、
12,173 nodes/72,343 edges；目标Python的metadata变化及排除的Fortran/outputs依靠
精确源码/工件补足，未重索引或作全项目完整性断言。
[最终交付核验](../../outputs/reviews/main-physics-20261023/compiled-input-binding/delivery.json)及
[证据索引](../../outputs/reviews/main-physics-20261023/compiled-input-binding/evidence-index.json)
绑定本页、实际源/旧证据/捕获身份与当前工作区；旧Stage14/15的45份size/mtime和
6份anchor内容哈希受保护，stat不解释为新增bulk内容哈希。
新增模型日/年、真实XLA编译、科学核/Oracle重放、broad pytest、SSH/服务器/调度、
GPU/训练/付费工作及环境重建均0；历史累计4601/7200秒、余额2599秒、pending0。
只修改本页及Stage16证据目录，不暂存、提交、推送、跨聊天发送或启用自动化；
未改生产代码、配置、科学门槛或其他研究工作树。原319失败及已登记证据缺口保留，
交付本聊天审阅后停止，不自动开始下一阶段。

第十六阶段主控验收（2026-10-24）：已审阅实际上游/compiled两张来源表、接口捕获
脚本与执行回执、交付JSON及本页差异；冻结报告SHA256与实文件一致，复用owner的
869项接口断言、436项交付核验和301个链接检查，未重复执行捕获或模型。
接受本阶段声明的有界源码/接口检查；未确认新绑定缺陷，也不将替身接口通过升级为
完整块XLA数值等价或669点科学验收。实际used-file静态值分母、旧证据的当前版本
适用范围及原319年度NPP失败继续保留。用户授权下一批连续检查，优先补前两项的
本地证据，不自动重启319水分起源捕获。此次仅追加主控验收；owner冻结回执仍指向
追加前的报告内容，保留原身份，不重新生成历史交付文件。

## 第十七阶段：实际输入分母、历史证据与缓存消费者闭合

本批连续完成A/B/C后统一交付。未确认新的生产缺陷，因此没有生产补丁或需要
实施的修复回归；第十五阶段C4修复仍是既有成果。入口、机器汇总和实际检查命令见
[Stage17 README](../../outputs/reviews/main-physics-20261024/input-and-evidence-closure/README.md)、
[合并结果](../../outputs/reviews/main-physics-20261024/input-and-evidence-closure/integration/review-results.json)。

**A：原669点输入分母已补齐。** 使用公共runner相同的inventory/resolver选点，
669/669均选择raw布局的实际`z1/used_run.def`及case `run.def`；没有缺失、默认回退、
重复key行或同布局I/S歧义。另有4点存在normalized/legacy并存目录；resolver仍按
raw优先选定，备用case内容一致，备用布局没有used文件，未把它们隐去或重复计点。
全程只读解析及内存字典materialization，没有生成669份运行文件或prepared context。

| 比较对象 | 实际结果 |
| --- | --- |
| raw used / case / effective键数及并集 | 3250 / 202 / 3266；并集3308，缺失不作为默认值 |
| 源间存在模式raw/case/effective | 42个0/1/0、16个0/1/1、3106个1/0/1、144个1/1/1；各模式在669点内一致 |
| 点间变化 | 仅LIMIT_WEST/EAST/SOUTH/NORTH及PFT14 VCMAX25、ARJV、MAINT_RESP_SLOPE_C、ALLOC_MIN、RESIDENCE_TIME；没有新增静态或结构键 |
| materializer语义 | 669点均`base_is_fortran_used_truth=True`，不重新套case值；25项JOB_PROTOCOL_OVERRIDES最后应用，逐键存在/变化计数另存 |
| 年度/路径/输出 | 完整有效映射保留五个年度键，当前静态比较无点间变化；不能把完整映射误称为已剔除年度项的static projection |

数值/逻辑只作保守等值分组，列表、路径、键大小写及缺失保持原区别。
源间缺失或统一protocol变化不等于点间物理差异。全键表、点级解析路径/输入身份、
并存布局及实际命令见
[A输入审查](../../outputs/reviews/main-physics-20261024/input-and-evidence-closure/input/input_audit.md)。

**C：九个真实变化均有动态消费者路径。** 四个LIMIT先选择当前点domain，再进入
forcing和landpoint动态payload；四个STOMATE校准坐标进入完整参数override，
ARJV进入DIFFUCO连续参数。其余prior-state、science-day、HYDROL数组、restart及
season家族由当前runner供应。prepared-context键区分config/run.def/reference路径；
外层块键只含config路径、块长、矿物边界、状态布局及capture/projector，命中后保留
首个静态closure。内层SECHIBA键不能在外层命中后补救静态捕获，安全性不能只靠
“内层有key”解释。本批实际完整有效映射未发现新变化的静态消费者，故没有必要
构造额外替身探针或人工反例。详细九键表和三个缓存的命中条件见
[C消费者审查](../../outputs/reviews/main-physics-20261024/input-and-evidence-closure/integration/consumer-review.md)。

默认CLI每个进程一个点，不产生跨进程的内存缓存复用；同一点可跨年命中。
公共Python cold-start/restart-year入口则支持同进程依次传入不同实际paper点：
同config路径/块长/状态布局/矿物边界等条件可形成外层二次命中，当前点的九类值
仍由runner传入。此为源码可达及数据路由结论，未执行真实XLA跨点复用；不外推到
任意配置、同路径文件修改、所有Fortran分支或669点数值等价。

**B：保留旧通过，明确当前适用边界。** 四组结果见
[B证据边界](../../outputs/reviews/main-physics-20261024/input-and-evidence-closure/evidence/evidence-boundary.md)。
14/365日完整块A/B、三点SECHIBA scan复用和年度/split restart、七点50年保存科学结果，
均未在20份原件及有界相邻记录中找到成功运行的producer/argv/输入内容身份收据。
目录时间、相似数值和最近提交不能补成生产身份；三点scan仍不当作完整块factory复用。
source/owner/state总表保留原声明和certificate整体false，未重建或重审全部family。

本批只读源码桥确认37/37 Fortran文件pin、43/43 Stage16保存的def-to-end范围相同；
后者双方均不含decorator。首次42/43来自保存范围未含一个lru_cache decorator，
按原范围规范化后匹配，未改变科学结果。另有C1–C4七个接受函数、31个相关常量/
helper/注册范围、4组import绑定及两份既有config/template身份一致；复用原针对性
回归，不重新执行。C2极小阈值按dtype计算；C4默认上限仍为3×365=1095天，
只影响对应缺失/sentinel tau分支。Stage15已绑定319初始有效tau1095，不触发C4；
不能据此否定所有旧结果，也不能绑定身份不明的旧长期轨迹。

原319首年NPP失败`0.0008192662350707258 > 0.0003951849734783173`继续保留。
独立科学门槛仍是当前身份的完整块/同进程跨点真实数值证明、代表点科学接受及
669点整体接受；当前来源/接口检查不替代这些门槛或定位年度因果。水分起源调查
继续暂停。本批新模型日/年、真实编译、科学核/Oracle、SSH/服务器/调度、GPU/
训练/付费分配均0；历史预算4601/7200秒、余额2599秒、pending0不变。

建议以本批结束基础排查，不继续追加同类清单或无目标的旧历史追索。后续最多三项：

1. 主控审阅本页、A/B/C实际工件与合并结果后封存本批，约15–30分钟本地审阅；
   本聊天交付后停止，不暂存、提交、推送或自动开启下一阶段。
2. 若需要当前版本的compiled整链声明，另立有身份的最小完整块/跨点/restart数值
   复核方案，约半天本地设计；旧成功收据缺失不以继续无限搜索替代。新执行资源及
   用时须按实际编译/运行另定，本批不启动。
3. 若恢复科学准确性调查，另立319上游保存状态最小缺项方案，约半天至一天本地
   准备；可与第2项方案并行，但669点科学扩展须在代表点失败及当前数值准入得到
   处理后再定。以上时间仅估本地准备，不承诺模型或付费运行时长。

图证据为Verify、generation `2026-10-23T16:51:45Z`，12178nodes/72348edges；
Python/docs的metadata_changed及排除的Fortran/outputs均以精确源/实际工件补足，
未重索引或声称全覆盖。未重跑Stage16的869捕获、旧45-stat/6-anchor保护、旧测试
或科学证据。Stage16冻结回执保留原报告身份，不随本页新编辑重标。合并检查只核对
本批数据/路由一致性、实际工件和Git修改范围；仅本页未暂存修改，交付本聊天后停止。

第十七阶段主控验收（2026-10-24）：已审阅A/B/C实际报告、输入比较脚本的解析/
缺失/分组与物化逻辑、合并结果及本页差异；报告实文件与owner冻结SHA256一致，
复用本批九项合并核验，不重新解析669份输入或重跑模型。接受实际669点配置分母
及九个变化键的有界消费者结论；旧producer缺失、真实XLA数值和科学门槛保留。
本批配置/历史证据收尾已完成，不继续重复同类身份搜索。用户要求更大的连续任务，
下一批转入明确过程组的Fortran/JAX源码及接口审查；既有明确通过边界继续复用，
原319失败和水分起源暂停状态不变。此次仅追加主控验收，owner原冻结身份不重标。

## 第十八阶段：六包核心过程与实际生产跨过程联查

从干净 `main@1b32f66510cb00ac2d844cbf49120840bafbcb90` 继续，实际Git root、
分支及ORCJAX模块导入确认。复用两位既有owner完成DIFFUCO、ENERBIL、THERMOSOIL
及allocation、phenology、turnover六包；父级验看实际源比较表、有限账本、保存oracle、
窄诊断及补丁，并联查compact/compiled生产时钟、单位和状态写回。六包结论和准确
分母见[Stage18入口](../../outputs/reviews/main-physics-20261024/core-process-sweep/README.md)、
[合并机器表](../../outputs/reviews/main-physics-20261024/core-process-sweep/integration/review-results.json)。
surface为13个reachable IDs、22条比较行、30个owner/completion records及274个
branch records/544个decision entries；carbon为3个reachable IDs，allocation、
phenology入口、turnover分别29/28/61个ledger sites及56/42/111个decisions。
不同账本重叠，不相加作Fortran分支总数或通过率；phenology替代onset helper另列。

**确认组件缺陷S18-THERM-NSLM。** Fortran `thermosoil.f90:2742–2743` 要求原水文
层数以下的潜热补充容量清零；JAX显式步在humlev扩到热力层数后调用getdiff时漏传
原nslm，遂从扩展数组推断成ngrnd。冷初始化调用已正确传参。原nslm2/ngrnd4自然槽
诊断保留104项/当前52失败/显式nslm对照0失败；活动自然PFT14按自然角色映射保存槽1
补充8项/4失败/0失败，原oracle槽14是crop，未混同paper自然PFT14。七个direct边界
符合源预期，当前Fortran文件、过程跨度及重建编译单元provenance为4/4匹配。
[逐字段和边界说明](../../outputs/reviews/main-physics-20261024/core-process-sweep/surface/fixture-and-boundaries.md)
明确可比较范围；[建议补丁](../../outputs/reviews/main-physics-20261024/core-process-sweep/surface/proposed-nslm-fix.patch)
已通过普通应用检查，尚未应用或运行修复后的wrapper。

触发须有ngrnd>nslm、冻结开启、深层活动mask、正有效长期湿度及相变温度；FR_DT=2
时区间为含端点的272.15–274.15K。复用Stage17汇总并只构建一个代表配置的生产几何，
得到nslm11/ngrnd32；只证明paper形状及开关条件，未测实际温湿轨迹、年度影响或669点
触发。其余五包未确认新差错，不外推为全部PFT/selector/状态或当前整模型数值通过。

[跨过程联查](../../outputs/reviews/main-physics-20261024/core-process-sweep/integration/cross-process-review.md)
记录半小时新旧温度/热容量、同日GPP与维持呼吸不同累计、phenology→allocation→NPP→
gap→turnover、日末reset和次日/块carry。旧热容量传递保证限后续loop及assembled-cold；
raw-restart首步回退新容量的等价性未证，保留接口限制，未登记第二项数值缺陷。
Fortran local SAVE ok_zimov未定义初始化的源合同及显式替代路径等限制也保留。
未来修复/同步范围仅共享thermal显式wrapper和相关组件回归，研究分支另按固定工作树
核对，不在本轮改动。既有C1–C4和旧冻结回执身份保留，不重新生成历史验收工件。

图层Verify为generation `2026-10-24T03:49:00Z`、12179nodes/72349edges；
metadata_changed的Python和排除的Fortran以当前精确源补足，不重索引或宣称全覆盖。
新数值工作仅两个微型CPU thermal wrapper诊断及direct对照/七个边界，独立Fortran
CSV复用；父级另执行一次纯几何helper。未重复原诊断、Stage17输入遍历或旧大套件。
[命令记录](../../outputs/reviews/main-physics-20261024/core-process-sweep/surface/commands-and-results.md)
保留初次补丁检查失败和修复后通过；
[交付检查](../../outputs/reviews/main-physics-20261024/core-process-sweep/integration/delivery.json)
核对JSON、实际链接、补丁及Git范围。无模型日/年、多点积分、整模型编译、
Fortran重编译/运行、SSH/服务器/调度、GPU/训练/付费工作或环境重建。
历史预算仍4601/7200秒、余额2599秒、pending0。原319 NPP失败
`0.0008192662350707258 > 0.0003951849734783173`、旧producer限制、当前整链及669点
科学门槛继续保留；水分起源调查暂停。唯一tracked修改为本页，未暂存、提交、推送、
跨聊天发送或自动化。六包及联查工作完成，交付本聊天审阅后停止，不自动实施补丁或
开始下一研究阶段；本段不是主控验收记录。

第十八阶段主控验收（2026-10-24）：已审阅六包来源表、实际生产接口联查、逐字段
fixture说明、保存Fortran来源回执、窄诊断脚本/结果及建议补丁，并直接核对Fortran
2742–2743与当前getdiff/explicit-step参数传递。接受S18-THERM-NSLM为组件装配
缺陷：原水文层数在湿度映射后丢失，深热层相变时的潜热补充容量未按源截断。
自然PFT角色映射及比较字段限制成立；复用owner的数值和9项交付核验，未重复运行。
其余包仅接受所声明的有界源/接口范围，不升级为全分支或整模型数值验收。
paper11/32层及冻结开关只证明静态触发条件，实际温湿轨迹、319因果及年度影响仍未证。
后续实施限该漏传参数及直接回归，独立登记共享修复和分支同步范围；暂不处理raw-restart
旧容量接口未知项，不改变物理公式、精度或科学门槛。主控追加验收不重写旧诊断身份。

## 第十九阶段：THERMOSOIL原水文层数最小修复与C5登记

从干净main `04441e7d70f8e396757d79aa6dd363b4fc4eb214` 继续；该SHA是已验收
Stage18报告基线，不是修复提交。**本轮实现、本地验证和主控验收完成；独立修复提交为
`a117cd5afc92ea55c9aeb528eec6f724de780111`**。生产仅在 `thermosoil_explicit_step` 的getdiff调用补
`nslm=_as_float64(shumdiag_perma).shape[1]`（`thermosoil.py:3171`），使湿度扩层后
仍按原水文层数执行Fortran2742–2743的深层潜热容量截断；direct公式、默认值、
精度/阈值、正确的cold初始化及无关API未改。见
[Stage19入口](../../outputs/reviews/main-physics-20261024/thermosoil-nslm-fix/README.md)、
[共享修复C5](shared_core_branch_review.md#c5thermosoil原水文层数与深层潜热容量截断2026-10-24)。

新增 `tests/unit/test_thermosoil_nslm.py` 和小型JSON fixture，运行不读取ignored
outputs或导入其诊断脚本。保存的独立Fortran参考值、源文件/过程跨度/编译单元及
CSV身份随fixture保留；原自然槽1–13逐层相同的104值压缩为8值，角色映射的自然
PFT14不冒充原oracle的crop槽14。比较仅限pcapa/pcappa_supp，原容差仍
atol1e-14/rtol1e-12，没有用JAX公式生成预期。

同一新回归在旧wrapper真实失败：自然槽1–13有52/104不符，自然PFT14角色映射
4/8不符，浅层先行断言通过；补传参数后新模块9项全部通过，其中包含两个原比较、
两个含端点相变温度、暖/冷区间外、正mc但长期湿度0、等水热层数及一次wrapper
容量投影JIT。另复用并执行既有direct getdiff及wrapper composition检查2项通过。
这是修复后真实wrapper证据，区别于Stage18的显式nslm direct comparator。
当前wrapper未暴露关闭冻结模式，未为测试扩展API或断言Fortran未定义的supp。

父级验看实际新测试、fixture、旧失败断言、执行脚本及三份回执，复用已完成的9+2
通过检查，没有重复数值运行。main图getdiff四个入站caller及wrapper三个入站入口
已核对，模块入口的普通/JIT分支包装同一wrapper。JIT证据限所选容量投影，不宣称
生产模块全输出、完整块或跨点XLA通过。图层由ready12180/72350、coverage generation
`2026-10-24T10:29:39Z`开始；新增测试/fixture覆盖回执为
`2026-10-24T10:41:54Z`，no_recorded_issue但metadata_changed，当前精确源/工件补足。
Fortran排除的源真值及Stage18四项匹配provenance直接复用，不重新调查旧oracle。

实际命令、exit、计时、身份和日志见
[实现与验证](../../outputs/reviews/main-physics-20261024/thermosoil-nslm-fix/implementation-and-validation.md)、
[机器回执](../../outputs/reviews/main-physics-20261024/thermosoil-nslm-fix/implementation-and-validation.json)。
ORCJAX/Python -B、pytest禁用cacheprovider，三个系统TEMP均已清理；因此产生的
既有cache_dir配置warning保留。最终范围/身份/链接检查见
[交付核验](../../outputs/reviews/main-physics-20261024/thermosoil-nslm-fix/delivery.json)。
Sol交付仅改一份生产文件、新增测试/fixture及两份审查文档，交付时未暂存、提交或推送。
主控已将生产改动、测试及fixture单独提交为上述C5修复，审查文档另行提交，以便研究
分支准确核对和同步。本轮不操作日尺度/全球工作树或向其聊天发送消息。

paper11/32层且冻结开启仍只证明静态条件。实际深层活动mask、正长期湿度及相变
温度是否同时出现、年度影响和319 NPP因果均未证。原319首年失败
`0.0008192662350707258 > 0.0003951849734783173` 保留，水分起源调查继续暂停。
raw-restart首步旧容量回退仍为未证接口限制，不顺带修复。无模型日/年、多点积分、
整模型编译、Fortran重编译/运行、广泛测试、SSH/服务器/调度、GPU/训练/付费工作，
历史预算4601/7200秒、余额2599秒、pending0不变。交付后停止，不自动开始后续研究。

### 第十九阶段主控验收（2026-10-24）

主控已检查实际生产差异、新回归全文、独立fixture、检查启动器、旧失败/新通过日志及
三份执行回执；当前三个代码/fixture文件身份均与修复后受测文件一致。真实旧wrapper
失败和同一测试修复后通过成立，9项新回归及2项既有检查证据足够支持本次参数转发修复，
主控未重复数值运行。独立修复提交只包含这三个文件，C5登记准确SHA及两个研究分支
待核对同步状态。图coverage generation `2026-10-24T10:41:54Z` 的相关路径均标注
metadata_changed，验收依据当前实际差异/文件和既有成对源证据，不作完整图覆盖声明。
接受范围仍限深层潜热容量截断的组件修复，不改变上文实际轨迹、年度影响和319因果限制。

## 第二十阶段：碳过程六包与实际OK_LEAK装配审查

2026-10-24启动、2026-10-25完成。基线main
`21fc7985d8554f120fcb149ffcc0d0c59a08145c`，工作区为
`E:/ORCHIDEE-MAN-JAX`，ORCJAX实际导入来自该checkout。
**本批六包审查及整合交付完成，等待主控手动验收；未实施生产修复。**
发现五项限定条件下的源码/接口差异，不能据此宣称实际日轨迹、年度影响或319因果已证。
总入口、逐项证据和最终范围核验见
[Stage20 README](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/README.md)、
[合并结果](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/review-results.json)、
[实际工件验收](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/integration/acceptance.md)、
[交付核验](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/delivery.json)。

| 包 / 文档审查单元 | 结论与限制 |
| --- | --- |
| [1维护呼吸](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/maintenance/maintenance-review.md)，14行 | 11有界接受、2未覆盖/输入限制、1源未定义诊断；静态cache/stack、即时量/日累计及NPP消费者分开核对 |
| [2gap死亡/turnover](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/gap-litter/package2-gap-ledger.md)，7行 | 6有界接受、1替代路径限制；当前自然tree14常数死亡路径，不包括DGVM/fire/增长效率等分支 |
| [3litter](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/gap-litter/package3-litter-ledger.md)，10行 | 7有界接受、1未覆盖、2跨包确证接口；sro和density引用包5同一发现，不增唯一缺陷数 |
| [4soil POC/DOC](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/soil-doc/package4-ledger.md)，7单元 | 相同输入下活动因子、分解、池转移及peat重分布有界接受；TF时钟/cutoff差异和SAVE/阈值边界单列 |
| [5DOC吸附/输运/扩散/export](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/soil-doc/package5-ledger.md)，8单元 | 五项唯一确证差异、2候选及明确未覆盖范围；helper成功不清除生产装配问题 |
| [6真实生产装配/carry](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/integration/package6-ledger.md)，12行 | 6有界接受、4跨包确证引用、2未覆盖；源时序、13个池字段路由、日LPJ输出绑定已查，完整生命周期未证 |

合计58仅为有重叠的文档行/单元数，不是58个独立oracle或Fortran分支，不给合并通过率。
包4/5每个单元可同时包含正确局部代数、错误接口及未覆盖边界。paper-selected arms、
owner区域、oracle输出行与标量值分母分别保留，不相加；未重做669配置盘点或旧全链检查。

五项确证均有当前原Fortran、实际caller/输入链和限定数值证据。精确源跨度、触发条件、
最小修复/回归建议见
[有界发现](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/soil-doc/findings.md)及
[边界数值](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/soil-doc/boundary-results.json)：

| 唯一差异 | 源与实际装配 | 证据及适用限制 |
| --- | --- | --- |
| TF_DOC干沉降dt | stomate.f90:3385传1/48日，soilcarbon:1225乘该dt；普通fold8582及compiled8367/8688传日量1 | leaf100/cover.5受控贡献当前.046，源.0009583333333333334，48倍；降水项不受此dt差异支配，实际canopy轨迹未测 |
| DIC soil呼吸率/步质量接口 | stomate:3409–3410先乘dt再3442–3456聚合；integration:3996–4011直接传soil/floodsoil率 | 隔离soil项的runoff/flood/drain三出口均48倍；没有据此认证全部litter/maintenance项或批准统一乘dt |
| raw bulk density归一缺失 | litter:1971–1973、soilcarbon:1473–1478将<500乘1000；raw static/default1.65沿真实输入链原样进入吸附 | 独立标量Kd/平衡公式给归一1650首层吸附.5440147884，当前raw为.0011916319；已归一trace/restart及全部669输入不作错误推断 |
| sro_bottom与nslm混淆 | constantes_var:294固定5；普通water用11/core默认5，compiled water/core/litter均11 | 全producer限定范围未提供不同cutoff；layer6及均匀水层受控对照显示参数影响。同helper对照不是独立Fortran oracle；无routing的零flood/fastr可抑制实际效应 |
| DOC峰值扩散赋值顺序 | soilcarbon:2005先改左flux，2006读取新左值；JAX3548–3555两者共用旧分母 | 默认半小时D、正薄层合成[1,3,1]给源flux[1.5,1.7142857143]、JAX[1.5,1.5]；真实paper几何/轨迹触发未证。源负剩余池是其确定算术，不以理想守恒改写源真值 |

真实时序为前一日日turnover/bm_to_litter先供48次半小时更新，各乘1/48，再执行日LPJ，
生成次日通量；Fortran3288–3293先于4297调用，当前完整entry的cold/restart/later-day
路径源顺序相同。不能因旧“same-day litter”文字判bug。维护呼吸按每步t2m/stempdiag算即时
parts并逐步累积给NPP，实际维护温度为11水文层、z_soil含零共12边界；ndeep32是另一维。
PFT14既natural/tree又peat，PERMA_PEAT开启，不把peat重分布或runoff2peat误列为inactive。
13个指定半小时carry字段和新日通量的源路由已核对，但旧presence测试不证明数值持久性。

日LPJ的MAINT_RESP/AGR绑定对应stomate_lpj:1702–1707、2283–2294，JAX保留日NPP
resp_maint及日累计parts。stomate:5034–5037另有半小时grid-area的radia*veget_cov_max输出，
本轮未端到端验收。两个候选仍是连续多个peat PFT的runoff buffer和极小正库存peat_OLT
阈值0对源1e-8；未声称已证库存/实际轨迹错误。active-layer/SAVE输入重建与carry保持
未覆盖/输入限制，未重开Stage14–17生命周期。旧altcalc mask下INTENT(out)及bare维护
温度/系数诊断保留源未定义，bare呼吸强制零的明确契约不变。缺完整entry的legacy
post-NPP fallback、crop/农业/DGVM/fire/其他死亡分支、完整firstcall/cryoturbation/restart、
任意非有限/无效数组、多元素及新梯度亦未获本批准入。

数值工作只包括原owner的短CPU组件：维护一个独立保存fixture为28LAI+336parts=364值，
两entry共12比较行全部通过，最大Fortran差1.734723475976807e-18；第二entry是内部
cache/stack/fold一致性，不是第二个独立源oracle。gap4、turnover9、litter75输出行和
soil-owner216行保存结果复用，相关当前源hash匹配；未重跑这些结果。soil/DOC三次短
诊断最终含5组：4组独立标量源预期、1组同helper cutoff效果对照；exit0仅表示诊断完成，
不把已展示差异称为parity通过。父级验看实际脚本/JSON/ledger后复用数值，原owner完成
11/32层、最终grid-area范围、L6/L10 disposition及扩散精确行号收口，没有重复运行。

图证据沿用Verify generation `2026-10-24T15:26:49Z`、12242nodes/73190edges。
Python的metadata_changed、排除的Fortran及JIT/嵌套scan漏边由当前确切源补足。
恢复会话MCP图工具不可用，明确记录0新图调用/0重建，不捏造新freshness或因缺工具停工。
[精简命令记录](../../outputs/reviews/main-physics-20261024/carbon-process-sweep/commands-and-results.md)
保留实际环境、执行范围及非数值lookup更正；系统TEMP已清理。新工件只写Stage20目录，
本轮唯一tracked修改为本页；不暂存、commit、push、改Git/代理配置、操作其他工作树、
跨聊天回传或建立自动化。无新Fortran运行、模型日/年、整模型编译、广泛测试、
SSH/服务器/调度、GPU/训练/付费或环境重建。C5接受提交仍为
`a117cd5afc92ea55c9aeb528eec6f724de780111`，实际轨迹/年度影响未证；原319失败
`0.0008192662350707258 > 0.0003951849734783173`、raw-restart旧容量未知、暂停水分
起源调查及科学门槛全部保留。历史服务器预算4601/7200秒、余额2599秒、pending0不变。

**唯一优先下一步：主控审阅本批五项差异，决定最小共享OK_LEAK调用单位/cutoff/raw输入
修复及直接回归批；扩散顺序须明确按原Fortran真值处理。** 本轮不自动实施、登记新共享
修复编号、运行模型或进入科学阶段；整批交付后正常结束，等待手动验收。本段不是主控验收记录。

### 第二十阶段主控验收（2026-10-25）

主控已检查六包实际ledger、整合记录、诊断脚本及保存结果、执行方16项交付核验，
并直接核对五项发现的当前Fortran语句和JAX生产参数链。图工具当前不可用，复用旧图
回执并读取确切源码，不宣称新增图覆盖。未重复组件数值、模型或旧oracle运行。
接受上述五项**有界源/接口差异**及两项候选、其余输入/生命周期限制的区分；58个
文档单元和16项交付检查均不代表模型全分支或科学准确率。

下一实施批限四项调用/输入修复及单独的扩散源顺序修复。TF使用实际半小时dt，
DIC保持soil核心返回率的契约，只在需要步质量的聚合边界转换；实施前还须补齐
litter和maintenance各项单位的同入口核对，不能将所有项统一乘dt。raw密度修复须
核对Fortran共享bulk_dens的就地更新和调用时刻，保留已归一值，不能假定该阈值转换
对任意数值都幂等。sro_bottom按源独立于nslm/ndeep，在普通和compiled实参中保持一致。

扩散的源2005赋值后，2006读取更新后的左通量；主控决定按本项目原Fortran真值合同
恢复此顺序，独立记录/提交，保留其在特定正输入合成例中产生负剩余库存的源算法局限。
这不授权另造守恒限制器或修订原模型。各修复须有实际入口的旧失败/新通过回归及
可独立运行的小型参考，完成后才登记最终修复SHA并验收。原319年度失败、C5年度影响
未知、raw-restart旧容量未知及暂停水分起源调查均不因本次源码验收改变。

## 第二十一阶段：五项OK_LEAK最小修复与实际入口回归

2026-10-25，从干净main `493e830702d71b2595e972739eab89c33fa4eaee` 接续已验收
Stage20的五项实施授权；固定工作区 `E:/ORCHIDEE-MAN-JAX`，ORCJAX实际导入该checkout。
**C6–C10均已完成主控验收。C6–C9修复提交为
`27ff14b38aafc7df91ce52b8393642ca08952395`；C10独立修复提交为
`225cb95a6a44334390d1656d85f66fd54d342a19`。** 实现、同测旧失败/新通过和最终组合
均已完成；原执行交付时未暂存/提交/推送，其回执保留交付时点身份。基线SHA不是修复SHA。
见[Stage21入口](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/README.md)、
[逐项实现/验证](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/implementation-and-validation.md)、
[机器记录](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/implementation-and-validation.json)和
[共享登记C6–C10](shared_core_branch_review.md)。

| 修复 | 最小实现与源依据 | 有界证据及未证范围 |
| --- | --- | --- |
| C6 TF_DOC干沉降时钟 | stomate:3380–3388、soilcarbon:1206–1233；初始化/普通/compiled三处producer转发`dt_sechiba/86400`，降水定义保留 | 正leaf/cover及TF关闭、真实TF/ground/storage两步carry；参数seam和窄scan不代表完整compiled模型 |
| C7 DIC率/步质量 | litter:2667–2670,2740–2750,2778–2781、soilcarbon:1671–1704及stomate:3314–3315,3409–3410；仅聚合处四个litter/soil率乘dt | maintenance源resp:297–298已含dt，floodroot源stomate:3258是步质量，不再乘；独立混合三出口、真实混合组件/no-flow，核心rate返回不改 |
| C8共享bulk density | litter:1971–1973和soilcarbon:1473–1478各执行一次`<500 ×1000`；raw static保留，litter controls前准备并合并，adapter/compiled尊重prepared marker再执行soil赋值 | raw1.65/default、1650、500、.5、tiny及真实controls/adapter、窄scan；peat密度分离。tiny`.0001→.1→100`只证一次调用，跨步回写/negative fatal/nonfinite未闭合 |
| C9独立runoff cutoff | constantes_var:294默认5；初始化/普通/compiled水量、core/litter统一，独立于nslm11/ndeep32；已有显式API覆盖保留 | producer合成水量5层55、旧11层506及override3为14；真实decomposition消费捕获cutoff，layer5/6独立预期`.075/.1125`；没有新增run.def能力 |
| C10峰值扩散源顺序 | soilcarbon:2002–2006；第二分母读取已更新第一通量，WHERE/循环/非峰值/冻结分支保留 | 正薄层`[1,3,1]`给flux`[1.5,12/7]`、库存`[2.5,-3/14,19/7]`；四源算例×eager/窄JIT。明确源负库存局限，无clamp/新limiter |

同一新回归的有效旧/新证据：C6/C9同15项为12失败/3通过→15通过；C8同原10项为
8失败/2通过→同10通过，其中4为真实adapter数值、2为controls实参、另2为新增marker
缺失的KeyError，后两项不冒充旧密度数值错误；C7/C10同11项为4失败/7通过，修复后
该11项及3既有直接检查共14通过。C6/C9两次fixture/seam前置失败、C7初次z_soil
形状错误单列保留，均不作缺陷证据。owner早期检查不认证稍后C8最终marker修改。

最后协调后的[组合focused检查](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/combined-focused.json)
exit0、49项通过（15 producer+20 density+3 DIC+8 diffusion+3既有直接检查），pytest39.40秒、
进程43.62秒；绑定14份相关生产/测试/fixture身份，系统TEMP已清理。重叠检查不加总成
独立覆盖率；关闭cacheprovider后的既有cache_dir warning保留。新测试只依赖自含合成
输入或tests内小fixture，不依赖ignored outputs。实际源/测试/fixture、有效旧断言与新日志
均经[工件验收](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/acceptance.md)。

C10交付独立[四文件patch](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/C10-source-order.patch)
和[分组身份](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/C10-group.json)，
不含C7/C8；逆向检查及系统TEMP中基线文件的正向适用性检查均仅`--check`，未apply/stage。
[命令记录](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/commands-and-results.md)和
[交付核验](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/verification.delivery.json)
记录实际范围；最后核验仅元数据、链接、Git及工件身份，不重跑数值。

MCP图工具不可用，沿用Stage20 Verify generation `2026-10-24T15:26:49Z`、12242nodes/
73190edges；Python metadata_changed、Fortran excluded及嵌套scan/JIT漏边由当前确切源
补足，0新图调用/0重建，不宣称完整覆盖。两项Stage20 candidate、完整active-layer/SAVE
生命周期与raw-restart旧容量未知不并入。本轮无新Fortran编译/运行、模型日/年、整模型
编译、广泛suite/73oracle、SSH/服务器/调度、GPU/训练/付费分配或环境重建；配置、
精度、mask、科学门槛和容差未放宽。两个研究工作树未操作，各修复同步均待维护线程
核对实际接口/提交，旧Teacher/prepared不重标；没有跨聊天消息或自动化。

原319失败`0.0008192662350707258 > 0.0003951849734783173`继续开放；五项修复不解释
其因果或证明年度改善。C5接受提交仍为`a117cd5afc92ea55c9aeb528eec6f724de780111`，
实际轨迹/年度影响未证；水分起源调查暂停。历史服务器4601/7200秒、余额2599秒、
pending0不变。执行方交付后的主控验收与真实修复SHA见本节开头和下述记录。

### 第二十一阶段主控验收（2026-10-25）

主控已审阅五项实际生产差异、新helper及全部新增回归/fixture、原测试期望修改、
有效旧失败与最终49项通过日志、执行脚本和34项交付核验。当前14份受测文件身份
均与最终组合回执一致；主控未重跑数值。接受参数捕获、真实组件数值和窄JIT各自
声明的证据范围，不把替身入口或metadata检查当完整compiled模型认证。

C6–C9的九个代码/测试文件合为一个共享接口修复提交；C10四文件另作独立源行为
恢复提交，均未混入审查文档。C7完整核对后只转换四个litter/soil率，maintenance/root
保留步质量；C8严格保留litter→soil两次字面赋值及prepared marker，tiny跨步回写仍
不在接受域；C9保留独立cutoff和既有显式接口。C10的源负库存局限继续单列，没有
用非负截断改写Fortran行为。共享表已登记准确SHA，两个研究分支仍待其维护线程核对。
本次接受组件/接口修复，不建立实际年度影响、319因果或历史数据重新生成的结论。

## 第二十二阶段：HYDROL、显式雪、CONDVEG与真实状态联查

Stage22执行方已完成六包、有具体疑点的轻量CPU核验、实际工件验看和整合，
**主控已验收其有界源/接口差异，修复及回归另批实施**。从已接受Stage21后的干净main
`1229de7b41711a5177d572feef5cc95f4cbc2a32`开始，cwd/Git root及ORCJAX实际导入
仍为`E:/ORCHIDEE-MAN-JAX`。生产、tests、配置只读，唯一tracked修改为本页；
新工件限于[Stage22入口](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/README.md)。
执行方交付时未暂存、提交、推送或跨聊天发送；主控验收见本节末尾，以下发现尚非修复登记。

| 包 | 实际来源/状态台账 | 有界范围 |
| --- | --- | --- |
| P1 CWRR | [27行](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/hydrol/P1-ledger.md) | 节点/边界、液水冰、系数/RHS/三对角、post-solve顺序、dummy及mask |
| P2 水通道 | [29行](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/hydrol/P2-ledger.md) | 入渗、蒸散/根吸水/胁迫、径流排水、dt/面积/PFT/tile与配置条件 |
| P3 雪结构 | [7边界](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/snow/p3-explicit-snow-structure.md) | snowfall、分层/再分配、粒径/压实、生灭及局部返回状态 |
| P4 雪热/相变 | [9边界](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/snow/p4-heat-phase-flux.md) | 热库存、profile、融再冻、升华、maxmass、液热及HYDROL交接 |
| P5 CONDVEG | [20行](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/condveg/condveg-review.md) | 雪覆盖/反照率/发射率/粗糙度、面积和PFT、配置/初始化接口 |
| P6 生产/状态 | [20边界](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/integration/P6-ledger.md) | 实际普通/compiled body、当前雪到CONDVEG/THERMOSOIL、选定carry/init/finalize |

这些行/边界是文档单元，不与数值字段、oracle cases或source arms相加成覆盖率。
每包JSON保留源文件/过程/跨度、条件、读写状态、当前owner、证据及未核对范围；
旧`.missing/.partial`后缀不作当前代码缺失结论。

### 六项有界差异与实际可达性

完整条件、独立小例、作用范围、建议最小修复与回归见
[findings.json](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/findings.json)。
跨包同一问题只计一次；本批不实施修复、不新分配C编号。

| ID / 置信范围 | 源与当前差异 | 已得证据 / 限制 |
| --- | --- | --- |
| S22-SNOW-LF，高 | `constantes_var.f90:374–376`定义`2.8345e6−2.5008e6=333700 J/kg`；当前雪热/相变相关默认及旧独立harness均333600，真实snow-main无覆盖参数 | 实际HYDROL雪adapter中1kg/m²新雪热库存约−333600，源应−333700，差+100J/m²；10kg/layer eager/窄JIT差+1000J。moderate maxmass cap源1kg、当前1.00029976019kg。当前真实调用链使用该默认，但实际forcing触发量/年度影响未测 |
| S22-TILE6-THRESHOLDS，源/实参高 | `hydrol.f90:7488–7564,7982–7993,7665–7769`的top/部分阈值只peat4/5，深层系数等用4/5/6；JAX将tile6阈值先统一换peat值 | 独立top fill .5mm对2mm，真实all_tiles seam将mineral mcs/mcr .6/.1换.9/.15。直接tile控制例后续clip抵消暂态差，未证最终runoff错误。当前paper HYDROL-local peat=False且PFT14 override4，不触发该组合 |
| S22-FROZEN-DUMMY-ORDER，固定profile高 | `hydrol.f90:7046–7058,7078–7108`在冻结dummy中第二次重构读取已更新mc；当前helper只一次 | 固定冻结比例.5，源top .3972278481再到.3958417722，JAX停首值；dummy TMC27.990对27.985mm。非冻结控制一致，dummy随后还原；完整freeze refresh和实际温度轨迹未认证 |
| S22-TOPMODEL-PREFIX，diagnostic API高 | `hydrol.f90:7245–7267,7308–7342`逐层止于首个冻结节点；JAX统计所有暖层 | 11层首/第4层冷、其余暖时，源连续暖层0/3≤6，fwet0；实际helper为.4875026035157896，全暖控制一致。selected TOPMODEL_NEW=False；paper普通/JIT wrapper未转发开关/四参数，true无diag时runtime entry明确停止，不能称已达paper偏差 |
| S22-CONDVEG-Z0，实际wrapper高 | `constantes.f90:483–489`读Z0_BARE；动态bare-only源闭式为该值，当前main/wrapper未绑定此配置 | 实际eager wrapper传.01/.02均ok且有albedo，z0m均.010000000000000002；selected .01控制一致。非默认参数接口差异已证，任意配置/完整compiled/年度效果未证 |
| S22-AE-STATE，源/状态路由高 | `hydrol.f90:5843`先ae+=subsink，9014 mask、9066乘bare fraction；当前diag8023读pre-surface split，driver2954又carry split，而finalize3102读diag | 实际surface 2+.2=2.2符合独立源；真实shared writer在split .20/diag .25 seam中，两表示均选.20。ordinary/compiled共用该writer；组合物理轨迹及年度未测。当步water2extract正确，grid总vevapnu另加subsink*vegtot，不称同一步总蒸发质量遗漏 |

paper `PEAT_HYDRO=TRUE`不覆盖已接受的HYDROL-local SAVE=False判定；
`orchestration.py:67–75,6317,6352,6382–6383`明确传local值，原paper job:141/162
仍将PFT14设tile4。此处只界定可达性，不重开全SAVE。实际source顺序是
DIFFUCO→ENERBIL→HYDROL→CONDVEG→THERMOSOIL，fusion仅非显式雪；
compact:6406使用当前HYDROL雪，6444使用当前moisture/snow/CONDVEG。
compiled nested transition:10404调用同compact并共享writer；10506–10522限定完整48步、
JIT/forcing/cache/template/tables及not-localpeat/no-trace等条件。
这是实际body的源级核对，没有新完整编译或compiled数值认证。

### 候选、源行为与证据域

三个候选单列：S22-FWET-CARRY在真实writer中观察新diag .8被旧.2覆盖，但创建器仅
TOPMODEL_NEW、当前paper producer不准入，不能升级为已达生产错误；
S22-DUMMY-REFRESH是源6883–6900在dummy前重算冻结profile与当前显式输入的组合缺口；
S22-TOPMODEL-TIDE是源7281–7298按cell tile6条件向所有tile加潮汐水库与JAX只选tile6
的source-only条件差异，尚无独立数值。这些与六项已复现/源状态路由差异分开。

零雪雪温疑点已排除，见
[完整序列补证](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/snow/zero-state-boundary.md)：
source:222–227、JAX:4038–4042先把精确零深度雪温重置273.15，profile虽只在有雪时写
温度，不能推出保留任意入口雪温。实际ENERBIL:1571–1577也使零深度temp_sol_add=0；
单独API非零add与tiny容差域保持限制。该疑点只做必要静态补证，未重复数值。
源tiny-depth gone先清depth再读mass、局部rain=0、部分清理不改所有字段等按原顺序
保留，不另造守恒/物理修订。CONDVEG fixed-snow的fraction_veg、imposed部分临时量及
bare roughheight有源未定义范围；selected配置不走对应支路，不能制造源数值oracle。

保存雪oracle实际是10cases、17字段比较、39/39arms；旧YAML的3point/44 prose滞后。
源procedure SHA匹配仍不能排除harness的333600依赖参数错误；原通过值保留其条件，
不重标为实际源333700等价。旧maxmass跨层disposition以density≤750为硬界的证明不足：
750仅是压实更新入口gate，当前source/JAX都能从749.999999更新到750以上且质量守恒。
这只否定该硬界前提，未证paper跨层overflow已发生，需包含后续状态/restart的域证明。
保存CONDVEG为3points、15比较行/69标量值；`mean_*`表示第二次mean-soil双波段调用，
不是光谱平均。branch receipt仅六个具体IDs，owner的main6/6、static5/5另有分母；
均不等于完整CONDVEG分支覆盖。

### main有限剩余矩阵与交付

[有限矩阵](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/remaining-main-matrix.md)
给出12个范围条目及单列的三个数值/科学门槛，避免继续无限寻找旧证据或泛称main完成：

| 范围 | 已完成/复用 | 明确剩余或不准入 |
| --- | --- | --- |
| 输入/上下文/669配置分母 | Stage16/17有界结论复用 | 旧完整块/长期成功producer缺项与真实XLA数值另列，不重搜/重跑 |
| 能量/热/主要生长碳过程 | Stage18/19及C5；Stage20/21及C6–C10复用 | 不扩大到所有配置/梯度/年度因果 |
| 本批HYDROL/雪/CONDVEG/真实装配 | 六包各声明的source/状态/小参考范围 | CWRR表构造、完整冻结dummy refresh/系数组合、TOPMODEL准入/潮汐/carry、非默认CONDVEG参数与独立producer bridges仍需限定审查 |
| 可选检查/生命周期与证明域 | 已核对的mask、source quirks和配置gate | check/fatal callbacks、canopy未赋值/SAVE、完整active-layer/peat潮汐、雪harness参数/跨层不可达证明未全面闭合 |
| 当前配置外或显式未支持 | localpeatFalse/TOPMODEL_NEWFalse及独立单点CWRR边界 | Choisnel/全河网/multi-nobio/部分init reject与源未定义分支分开记录，不默许支持 |
| 独立数值与科学门槛 | 历史通过保留原适用域 | 当前完整XLA/restart准入、319失败/水分起源、669点/长期科学门槛，均不混入源码行数 |

新数值仅owner的65条HYDROL组件记录、雪adapter/小helper与两处窄JIT、两次真实CONDVEG
wrapper，以及父级shared writer两种表示各三场景。后者验证17个选定水/雪字段转发，
不作17个独立科学oracle或完整scan通过。各包独立参考、实际脚本/收据均已验看，
原owner数值未重复执行；9066实际赋值行号纠正仅重生成metadata。
[命令与结果](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/commands-and-results.md)和
[最终交付核验](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/delivery/verification-results.json)
保留JSON/链接/Git范围/TEMP检查；metadata通过不代表科学通过。

图工具启动时不可用、中途恢复，Verify project `E-ORCHIDEE-MAN-JAX`，generation
`2026-10-25T06:02:38Z`、12323nodes/73746edges；恢复后再次确认ready并核对相关coverage。
Python metadata_changed、Fortran excluded及nested scan漏边用当前确切源补足，
不重索引、不宣称完整性。系统TEMP清理记录保留，无新Fortran、模型日/年、整模型编译、
广泛suite/73oracle、SSH/服务器/调度、GPU/训练/付费、环境重建或研究工作树操作。

原319年度NPP仍failed：`0.0008192662350707258 > 0.0003951849734783173`。
C5–C10年度影响/归因未证；本批六项与该失败的因果关系也未证明。C10源负中层库存保留，
C8 tiny跨步/negative fatal/nonfinite未闭合；两项Stage20 peat候选、完整SAVE/active-layer、
raw-restart旧容量未知不扩大。water-origin investigation paused；旧Teacher/prepared身份
不重标，C4–C10研究同步仍由各维护线程处理。历史服务器4601/7200秒、余额2599秒、
pending0不变。

### 主控验收与后续修复范围（2026-10-25）

主控已检查六项发现的实际源码边界、独立小参考脚本与保存的数值、参数/状态转发记录，
接受其所声明的局部不一致。复用owner已完成的检查，新增数值执行为0；交付核验
186/186是JSON、链接、范围及TEMP等metadata检查，不是186项科学验证。
当前main导入路径和基线身份已确认，除本页外没有tracked修改。主控图依据为
Verify generation `2026-10-25T07:07:50Z`；Python metadata_changed、Fortran excluded
及template not_tracked均用确切源读取补足，未宣称全图或全配置覆盖。

六项均进入下一批最小修复与直接回归：雪常数及相关harness参数、tile6分阶段阈值、
冻结dummy重复赋值、TOPMODEL连续解冻前缀、Z0_BARE传递、ae_ns更新到诊断及跨步carry。
其中ae_ns必须补真实组件到writer的组合回归，分别验证状态加权和当步总蒸发，
避免重复加入subsinksoil；当前分段观测不冒充完整物理轨迹。冻结修复只接受已证实的
赋值顺序，完整profile refresh仍未闭合。TOPMODEL前缀修复不自动开放生产配置。
三个条件候选继续单列，不作为已接受修复；原319年度失败和所有历史证据身份保留。
详见[主控验收回执](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-sweep/primary-acceptance.json)。

**唯一优先下一步：原Sol在main完成上述六项最小修复、相关组件/接口回归和共享修复登记草案，
交主控验收后再提交。** 本次审查checkpoint不宣称修复完成、全年通过或研究分支已同步。

## 第二十三阶段：六项最小修复与组合回归（已验收）

2026-10-25，在固定main根工作区从干净基线
`c94da5b623da4f6da377520ce08436933b6ed1b7`实施上一节已授权的六项修复。
Git root/branch及ORCJAX实际导入均已确认。**主控已验收实现与有界回归，登记为C11–C16，
四组准确修复提交见本节末尾。** 执行方交付时未提交，原交付回执保留该时点状态。Stage22数值
producer仍为`1229de7b41711a5177d572feef5cc95f4cbc2a32`，不得以本基线重标。
入口为[Stage23交付](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-fixes/README.md)，
逐项原始命令、日志、JUnit和TEMP记录均保留在该目录。

| 原S22 ID | 最小修复及源依据 | 有界验证与限制 |
| --- | --- | --- |
| S22-SNOW-LF | `constantes_var:374–376`的共享`CHALFU0=CHALSU0−CHALEV0=333700`；8个显式雪及1个bucket默认值，两harness模板先声明依赖再求差 | 真实1kg snowfall adapter、独立热库存/部分融再冻/液热/非生物融雪、moderate/extreme cap及窄JIT；独立土壤`lhf:569=333600`和显式参数override保留；模板仅静态检查，无新Fortran |
| S22-TILE6-THRESHOLDS | all_tiles保留mineral mcs/mcr；top/over-mcs2及部分residual用4/5，深层/系数/液水重构保留4/5/6（hydrol:7488–7564,7982–7993,7665–7769） | 真实all_tiles→tile路径，1/4/5/6、面积/无tile及localpeat=False控制；没有全局从peat移除tile6，未开放paper localpeat |
| S22-FROZEN-DUMMY-ORDER | hydrol:7044–7058先按freeze开关重构或mc=mcl，再按7080–7092无条件重构，第二次读取第一次结果；真实caller转发开关 | 固定profile0/.5、冻结开/关、retry/未retry、resolv和原prognostic不变；未实现候选6883–6900完整profile refresh |
| S22-TOPMODEL-PREFIX | hydrol:7245–7267只数地表连续暖层，保留numlayers上界及>6 gate、7308–7342 undef/湿地公式 | 首/第4/6/7冻层、全暖、undef和numlayers6/7/2/15；15仅验证既有JAX clamp，非越界Fortran等价；eager/窄JIT；不自动开放生产TOPMODEL |
| S22-CONDVEG-Z0 | `constantes:483–489`的Z0_BARE经wrapper/main/init缺值回退及cold实参；两捕获context的compiled cache键含该值 | missing/.01/.02真实wrapper eager/窄JIT、main/init/restart保留；ordinary/compact/cold源call表达式只证实参组装，两实际cache构造stub只证缓存隔离，不是完整driver/compiled模型认证 |
| S22-AE-STATE | hydrol:5843地表ae+=subsink后的真实tile结果进入diag，再按9014/9066 mask/frac加权；共享writer与finalize一致carry diag.ae | 真实模块→diag→`_runtime_next_state_from_components`→下一split，两表示及窄JIT；正/负/0残差、frac0/1/中间、无tile/无植被。water2extract与grid总vevapnu原本已正确，不重复加subsink |

所有生产修改前，accepted before回执均为同一基线、相同三份生产文件SHA、空tracked
diff，且无error/skip：A12数值失败；BCD17失败/13通过；E2失败/4通过，另2条真实
cache选择失败；F7数值失败/2通过。F实际正残差例中旧diag tile1`.00275`、旧carry`.005`
应为`.02475`，下一split读旧蒸发`.005`应为`.0162`。不是仅复用split=.20/diag=.25
人工seam。F在baseline后另加强独立bare fraction和negative/no-old fallback断言，原七条
失败断言保留。D在原30项之后补4条numlayers边界；这些增项不追认为原before证据。
C的freeze=False/profile=.5仅为隔离赋值顺序的合成控制，源正常off路径profile为0，
不据此声称完成非零冻结profile的实际生命周期验证。

分包after为A12、原BCD30、E13、F9项通过；相关既有回归A8、BCD30、E21项通过。
BCD旧target首轮29通过/1失败保留：原routing综合fixture将tile4 mineral clipping与peat
forcing混用，修复后改成确有该阶段差异的tile6，保留全部positive-force/route断言；
新tile4 clipping及原tile4 positive-force检查仍保留。雪旧test只修源常数预期，保留
显式333600 override与任意soilcap输入；没有删除覆盖或放宽容差。详见逐项报告。

最终[组合focused回执](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-fixes/combined-focused.json)
exit0、72项通过、无失败/error/skip，进程76.38秒，TEMP已清理；覆盖四个新增文件
及四项既有module/grid直接检查，文件身份与补丁适用性核验见交付入口。
所有数值为轻量CPU组件或选定窄JIT，pytest计数不相加为独立oracle/分支覆盖率。
ORCJAX Python-B、禁pytest cacheprovider，系统TEMP运行/cache目录清理，既有cache_dir
warning保留。交付metadata核验与科学数值结论分开。

Verify图基线`E-ORCHIDEE-MAN-JAX` generation`2026-10-25T07:07:50Z`、12327nodes/
73750edges已确认ready；相关search无后页，Python metadata_changed、Fortran excluded、
template not_tracked均确切源回退。恢复后追加的两次trace调用返回Transport closed，
不宣称新增成功trace或完整性，无重索引。修复文件按原S22问题的hunk/测试分组交付，
hydrol.py由唯一writer整合，driver E/F共享文件的hunk归属明确，不反复apply/revert工作树。

旧显式雪10cases/17字段/39arms仍是harness333600条件下的历史结果，未重建为333700
oracle；Teacher/prepared/restart身份和序列化类/模块身份不重标。S22-FWET-CARRY、
S22-DUMMY-REFRESH、S22-TOPMODEL-TIDE继续开放；完整SAVE/active-layer、两项Stage20
peat候选、C8 tiny跨步/negative fatal/nonfinite和raw-restart旧容量未知不扩大。
C10源负库存局限保留，C4–C10准确历史SHA不变；研究同步仍待维护线程核对。

319年度失败`0.0008192662350707258 > 0.0003951849734783173`不变，六修复的实际年度
影响/因果未证，water-origin investigation paused。无新Fortran执行、整模型编译、模型
日/年、广泛suite/73oracle、SSH/服务器/调度、GPU/训练/付费、环境重建、其他研究工作树、
自动化或跨聊天消息。历史服务器4601/7200秒、余额2599秒、pending0保持。
### 主控接受与可同步提交（2026-10-25）

主控已审阅全部生产差异、四个新增回归文件、既有测试预期/fixture修改、有效before与
最终after记录、实际运行器、组件组合及patch分组。接受上述六项源/接口修复；
当前16份交付文件SHA均与manifest相符，其中三份核心生产文件也与组合运行启动快照一致。
复用owner的72项最终定向通过和相关既有回归，主控新增数值运行0次；56项交付metadata
核验独立于科学测试。图工具本轮返回Transport closed，使用前序图定位和本次确切源回退。

| 修复编号 | 接受的代码提交 | 内容 |
| --- | --- | --- |
| C11 | `dd6d6335ba5011fcd86285984136a6799f6916ac` | 雪融合潜热默认、相关harness参数及回归；独立土壤lhf保留 |
| C12–C14 | `d48b923e891946d3c9db77ba9b684666cb511b1d` | tile分阶段阈值、冻结dummy顺序、TOPMODEL前缀；共用一份回归文件 |
| C15 | `bbcdbf7a661cb11f5295883e54c3b1ea9d596d39` | Z0_BARE参数传递及两compiled缓存隔离 |
| C16 | `2f36dc7f7945d52387a2848aa124c043118ed45f` | surface到diag再到跨步carry的ae状态；不重复计入总蒸发 |

C12–C14合为一个可同步提交，其余三组独立。提交仅暂存相应已审阅hunk，没有apply/revert
工作树或重新运行模型。两个研究分支仍待其维护线程按实际接口核对同步，不能整文件覆盖。
本次接受范围不包含完整年度、梯度、全配置或新Fortran oracle认证；三个S22候选继续开放。
主控接受回执见[primary-acceptance.json](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-fixes/primary-acceptance.json)。

**下一步：按有限剩余矩阵收尾源码审查，优先核对CWRR表构造/冻结refresh、TOPMODEL候选及
尚未核对的参数/生命周期边界；保留当前生产准入，不自动开启模型年或新实验。**

## 第二十四阶段：有限剩余源码与条件域收尾（主控已接受有界证据）

2026-10-25从干净main `cc3afe325f0bb4378b664b3a5b7e3db7486ce88b`继续，
实际root/branch及ORCJAX导入均为固定根checkout。已先读取Stage23主控接受回执及
版本化报告，C11–C16四组接受SHA与72项定向通过直接复用；旧delivery的pending/null
时点记录不再作为阻塞，也不重标旧producer。本批仅新增
[Stage24目录](../../outputs/reviews/main-physics-20261025/main-source-closure/README.md)
中的诊断/参考/台账，并更新本页；执行方交付时没有生产修复、新C编号或共享登记修改。
随后主控完成下述验收与累计/分支证据登记修正；仍没有新增模型代码修复。

### 六包处置与数值证据域

| 包 | 有界结论 | 证据和限制 |
| --- | --- | --- |
| P1 / M07 表/参数/插值 | 默认10个参数声明、65个标量静态一致；保留最后D区间沿用frac及AKS_POWER校验的源quirk。非冻结源`INT(scaled)+imin`与当前`INT(imin+scaled)`在浮点边界不同 | [P1台账](../../outputs/reviews/main-physics-20261025/main-source-closure/hydrol-tables-tail/P1-ledger.md)12条。默认mcr.045/mcs.43、合法mc.1682时scaled15.999999999999998，实际CPU float64 eager选bin17，独立源表达式选16；freeze=True/零冰控制按不同源表达式选17一致。value-tag表只证明索引，无物理D幅度、peat数值、JIT或编译Fortran结论 |
| P2 / S22-DUMMY-REFRESH | 源6884–6886用normal末态mc刷新profile，随后6887–6905重建mcl/coef/setup；当前all_tiles10730–10749仍传入步初profile，返回10757也保留旧值 | [P2参考与实际观察](../../outputs/reviews/main-physics-20261025/main-source-closure/dummy-refresh/P2-ledger.md)。1点4节点6tile、明确温度/几何/default USDA和真实表，独立final-mc→profile→lookup→coef/setup→RHS/solve/两次重构：profile最大差.011622817271966679，dummy柱水量差.0296400206193006mm。实际结果匹配独立旧profile参考，传入独立新profile则匹配源刷新参考；normal solver本身未独立认证，不是paper真实步或年度触发 |
| P3 / M08 TOPMODEL/tide/fwet | 源7286–7293按cell的tile6面积选择水库并用于所有tile，当前7597–7603只对tile6加tide；组件条件差异确认。初始化文件IO明确不支持、生产module未传flag/四参数，true缺diag时runtime返回None | [P3台账](../../outputs/reviews/main-physics-20261025/main-source-closure/topmodel/P3-ledger.md)。1点11层6tile显式四参数，tile6面积.3、wt_ab10/wt_ab_tide50mm时meanwt误差−.028m，fwet误差−.013998822588127957；无tile6、tidesoff、等水库、peatoff四控制加权量一致。未开放生产TOPMODEL |
| P4 / M09 选定尾部/检查/入口 | 18条记录，明确check默认false、split/soil/qflux/infilt失败合同及tmc默认allowed_err差异；canopy PFT1和inactive peat-above源未赋值坐标不制造oracle | [P4台账](../../outputs/reviews/main-physics-20261025/main-source-closure/hydrol-tables-tail/P4-ledger.md)。五处是静态控制/默认值差异，未提供有效轨迹数值反例，不计为五个已触发物理bug。tail无生产caller仅入口支持；active-layer/root只审声明的初始化/动态入口，不扩到全部SAVE生命周期 |
| P5 / M10 CONDVEG参数/init/路由 | 非默认SNOWCRI_ALB未从wrapper/main/init转发；kernel已有参数。MIN_WIND/HEIGHT_DISPLACEMENT/Z0_ICE/SNOW_DENSITY等另按source-only接口条件及分支列明 | [参数/路由台账](../../outputs/reviews/main-physics-20261025/main-source-closure/condveg/parameter-ledger.md)。源OK_SECHIBA且IMPOSE_PARAM读入非默认值、nobio雪10时，设20的真实eager wrapper仍给.5，独立源应1/3；默认10一致。probe为effective-parameter mapping，未含gate键，不是完整run.def loader/driver准入；IMPOSE_PARAM=false不触发此非默认读取分支。C15数值/AST/cache证据不重跑、不混域 |
| P6 / M11 雪oracle/跨层证明 | 历史10cases/17fields/39arms保留333600 harness条件，原oracle生成Git身份缺项不臆造；C11独立333700与土壤333600证据复用。撤回以density≤750和早期几何推出全程不可达的证明 | [P6域台账](../../outputs/reviews/main-physics-20261025/main-source-closure/snow-domain/snow-domain.md)。旧compaction gate反例不重跑。实际maxmass入口需`R<m3`；充分条件`U=m1+m2<3000`，restart/transfer/phase/sublimation允许域尚未提供该不变量。未确认新的maxmass算术翻译缺陷 |

P2源7165–7174恢复normal `mc/mcl/tmc`，**不恢复已刷新的profile**。修复时须保留
该owner差别，不能只重算dummy后丢弃新profile，也不能把dummy积分写回normal状态。
本批脚本对后续诊断的非mutation检查不独自证明全部restore；恢复/返回路径的结论还依赖
精确赋值检查及已接受C13的有界证据。线性冻结也有8445–8455的mc饱和度修正依赖，
本批未另跑该模式。P2本例旧/新profile的retry均为false；不把条件代码的存在当作
本批已触发retry，相关旧C13有界证据仍按原域保留。

S22-FWET-CARRY不再泛称未知：当前shared writer5966–5969和ordinary3471–3473会用
priorfwet覆盖新diag，finalize3119则优先新diag。复用Stage22两种状态表示的`.8→.2`
观察及无prior/无新diag控制；本批未重跑seam，也不将旧ae观察当作C16之后的新失败。
新fwet只能由未准入的TOPMODELtrue路径创建，因此仍是**条件状态路由观察、生产触发
未准入**，不是本paper已达生产缺陷。参数文件owner是源weighted reader9871–10049、
getter9910及10003–10032的面积归一化；noweight调用873是注释，不能互换为生产依据。

P4静态边界中，soil budget/minimum-mc、split守恒及qflux fatal受check_cwrr开关约束；
infilt_tot负值/超入渗量的源fatal不受该开关约束，但未证合法输入实际触发。
tmc直接helper在check开启且省略allowed_err时默认0，源为2e−8、比较界限2e−7；
未找到其生产caller，不声称当前轨迹发生误拒。各行已给最小残差/状态owner和闭合条件。
水量残差的另一源分支仅warning，不能误报缺fatal。未定义canopy/peat坐标须明确source
赋值或consumer mask，不能以初始化零值宣称等价。

P5区分direct init、实际cold构造、ordinary/compact和generic supplied-mapping入口；
拒绝的non-MODIS/static/imposed/repeated init、不全的背景参数、fixed-snow fraction_veg
及imposed未定义临时量分别登记。generic helper要求调用方提供当步输入，不等于run.def
读取器或cold初始化器。参数AST实参、缓存stub和真实组件数值仍各限原证据域。

P6将source349true/355false/360true从普遍`source_unreachable`撤回为
conditional/domain-unproven；360false在loc∈{1,2,3}且loc≠3下的局部逻辑保留。
`R=min(M−3000,k*soilcap/333700)`，M>3600时k=3，否则1；R=m3的inclusive定位
已进入上一层但可能移除其零质量，分支可达与严格正跨层移除分开。若能证明transfer输出
U<3000、有效phase/下渗且sublimation非负，质量论证可闭合；当前restart/forcing没有
建立这些前提。执行方交付时共享disposition登记只读；主控验收后已将
`pft14_arm_dispositions_sechiba.yaml`的349true/355false/360true改为现有合法分类
`pft14_conditional`，在reason中明确domain-unproven，并修正相邻bottom-only说明。
360false局部逻辑保持，旧生成覆盖报告保留历史身份，不宣称新39/39或全域覆盖。
新Fortran全arm参考或物理准入跨层触发需要另批授权。

### 有限矩阵、验看与下一步

[更新矩阵](../../outputs/reviews/main-physics-20261025/main-source-closure/remaining-main-matrix.md)
及机器记录给出M01–M12处置和每一真正剩余量；不再保留无边界的“全部SAVE/所有参数
尚未核对”任务。custom runtime表参数绑定、tail实际caller/error owner、未定义输出mask、
TOPMODEL生产支持策略和雪质量不变量保留为具体合同/证据条件。C8 tiny、C10源负库存、
两项Stage20 peat候选和raw-restart旧容量未知均按原范围停放。

新执行仅P1一个eager索引探针、P2一个真实all_tiles组合及其独立/供参dummy比较、P3五个
显式诊断场景、P5两次wrapper；没有相加为独立oracle数或分支覆盖率。P1/P4的65个参数
标量、源/AST、JSON/链接/Git/TEMP核验都是静态/metadata，非科学测试。父级验看owner
真实脚本与保存值，未重复其执行；[命令与交付记录](../../outputs/reviews/main-physics-20261025/main-source-closure/commands-and-results.json)
保留各域及收尾核验。

图依据复用Verify project `E-ORCHIDEE-MAN-JAX` generation`2026-10-25T07:07:50Z`、
12327nodes/73750edges。本批一次index_status返回Transport closed，不重复恢复或重建；
Python metadata_changed、Fortran excluded及template not_tracked以当前确切源/rg回退，
不宣称新成功图查询或覆盖完整。Python-B、CPU小组件、系统TEMP cache并清理；
Stage23及73oracle/广泛suite均未重跑，无新Fortran、整模型编译、模型日/年、多点实验、
SSH/服务器/调度、GPU/训练/付费、环境重建、研究工作树或自动化；执行方没有跨聊天消息或提交推送。

V01当前完整XLA/跨点/restart数值、V02的319年度NPP失败/水分历史因果、V03的669点/
长期科学与C4–C16维护同步仍独立。319误差仍为
`0.0008192662350707258 > 0.0003951849734783173`，failed；water-origin暂停，
本批差异均未证明年度影响，旧Teacher/prepared/restart身份不重标。历史服务器累计
4601/7200秒、余额2599秒、pending0不变。

### 主控验收与下一批决定

2026-10-25验看实际四个探针、保存值、独立P2/P3工件审查及成对源码，复用owner
数值验证，未重跑JAX/Fortran/模型。额外仅一次Python标量算术核对：按源先算
`ratio=max(mc-mcr,0)/(mcs-mcr)`再乘50，本例仍为15.999999999999998、bin16；
确认P1探针中乘除重排未改变本例结果，但下一回归须逐步写出源ratio/INT/加法顺序，
不能依赖代数等价代表所有浮点边界。P1仍只证明索引；P2正常solver/restore、
P3生产准入、P5门控/loader及P6物理允许域均保留上文限制。

接受本批**审查证据和有限分类**，不等于新增修复已验收。共享登记已补齐16项/11个
代码提交的总表、生产路径和准确SHA；本批新发现继续使用发现ID，不计入C1–C16。
主控收据见[primary-acceptance.json](../../outputs/reviews/main-physics-20261025/main-source-closure/primary-acceptance.json)。

**唯一优先下一步：原Sol完成末态冻结profile刷新、非冻结查表INT顺序、已有TOPMODEL
诊断潮汐选择的最小修复，以及SNOWCRI有效配置门控补证后的参数传递修复。**
每项须有源条件、修前失败、独立参考、修后组件/调用及必要窄JIT证据；SNOWCRI若
实际准入合同不成立则明确停留候选，不以扩大支持或关闭门控凑修复。其他source-only
参数、check/fatal、fwet生产准入、雪不变量不自动扩为修改。保持原科学阈值、旧数据
身份和生产准入；不开模型日年、Fortran、服务器、训练或其他研究分支同步。

## 第二十五阶段：四项最小边界修复（主控已验收）

本批从main `7905b1165f6e812bf87514017765278f9df5da6d`的干净生产/index开始，
固定cwd与实际ORCJAX导入均为`E:/ORCHIDEE-MAN-JAX`。上节四项下一步已按本批
授权实施；Stage24的只读/pending是历史时点。2026-10-26主控验看实际补项、
修前失败、93项最终JUnit/log及八文件交付身份后接受本批。累计已接受口径为
**C1–C20、20项/13个代码提交**，旧准确SHA表完整保留。
以下四项依次登记为C17、C18、C19、C20，原发现ID保留供追溯。
C17–C19代码及相关测试提交为`ad0d322cc4951d4af158f4e8cbb6d306bdf374f3`；
C20为`f8d36e1f8c03046e12e146aed2abb8bd286403de`。
A-R1完成原C17状态传递，D-R1是C20新helper的输入合同补齐，不另增物理缺陷编号。
[主控验收回执](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/acceptance/primary-acceptance.json)
记录接受范围、复用验证和历史回执的引用更正。

### 必要性、实现和有界证据

| 原发现ID | 必要修改、源条件及位置 | 修前/修后证据与输入域 |
| --- | --- | --- |
| S22-DUMMY-REFRESH | `hydrol.f90::hydrol_soil`6884–6905按normal末态mc刷新冻结profile，7165–7174只恢复normal mc/mcl/tmc。`hydrol.py`的all_tiles在dummy前刷新并返回新profile；module/closure及`orchestration.py`四入口传实际dh与五项冻结控制，缺温度/dh/纹理明确拒绝。A-R1补齐实际writer，从soil结果持久化新profile，不再复制输入旧profile。低层显式profile helper仍保留固定profile合同 | 共同基线thermo/linear/module/actual precall closure四个有效失败，profile最大差分别.01954419464452073/.014218757224033346/.010042066601177013/.0007949450011098258，dummy RHS亦不同；off通过。修后真实normal返回mc/mcl/tmc、独立NumPy profile/真实表/矩阵/RHS/求解/重构与下游diag/moisture均通过。A-R1真实零雪7层closure→writer/packet→无THERMOSOIL fallback在初版实现上复现旧/新profile及writer/fallback误差均.000794945001109798，补齐后通过，normal mc/mcl/tmc保留。另有1点4层6tile thermo/linear/off及4层linear窄JIT；冻结律源8332–8455，土壤lhf333600保留。normal solver本身不是独立认证 |
| S24-CWRR-NONFREEZE-BIN | `hydrol_soil_coef`8291–8301先ratio、scale、INT再整数加imin。只改矿物/泥炭非冻结consumer，8261/8275冻结加法后INT保留；局部`jax.lax.optimization_barrier`保留ratio赋值的舍入边界 | .1682/.045/.43仍按独立逐步表达式得scaled15.999999999999998、bin16。共同基线矿物eager/泥炭eager及JIT失败，矿物JIT本来通过。仅改整数顺序后3pass/1泥炭JITfail，加入局部屏障后4pass；两family×eager/JIT、各16个边界两侧/端点输入和freeze控制，真实a/b/D/K均通过原限。原consumer AST隔离重放的D最大偏差为矿物4910.4347847302415、泥炭268.56503923288255mm²/day；K近连续。该重放不冒充共同修前或编译Fortran；无逐位等价、任意backend或屏障性能声明 |
| S22-TOPMODEL-TIDE | `hydrol_soil`7286–7293按cell的tile6面积严格`>min_sechiba`为所有tile选潮汐/普通水库，替换当前tile索引条件 | 共同基线eager/JIT最大meanwt差.04m；修后3cell的无tile6/恰等阈值/刚超过阈值，以及tidesoff/等水库/peatoff/已有5tile组件合同均通过独立meanwt/fwet参考和窄JIT。C14连续暖层前缀10项旧回归通过。仅已有显式四参数诊断；生产TOPMODEL=true、参数IO和fwet carry仍未准入/未修 |
| S24-CONDVEG-SNOWCRI | `control.f90`455、495–503、550–555和`intersurf.f90`2267、2302–2310、2330–2335建立SECHIBA状态及参数gate；`constantes_var.f90`129/697默认IMPOSE_PARAM=true/SNOWCRI=10，`constantes.f90`507–513读active值。新增显式initialized-state resolver；main/init/wrapper、cold直接fraction/main及两缓存键转发effective标量；D-R1补齐既有严格boolean输入合同 | gateon有效20+nobio雪10时，源893给1/3，旧wrapper仍.5，共同基线4fail/8controls。门控关闭忽略非法raw，缺IMPOSE_PARAM用源默认true；raw OK_SECHIBA不替代initialized状态。D-R1 active未知IMPOSE_PARAM从静默false改为ValueError，recognized true/false及initialized-off不读raw的5项控制保持；这是helper合同补项，不增加物理翻译缺陷。D50个唯一ID在新93项组合通过；初版D44=42pass+2夹具修正pass历史保留。数值wrapper eager/JIT、main/init/restart、五actual call表达式×三gate情况、两真实cache constructor stub分列。explicit scalar API表示已生效值；其他四个参数接缝及未支持init模式未扩展 |

[HYDROL报告](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/hydrol/README.md)和
[CONDVEG报告](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/condveg/report.md)
保留源subroutine/行段、当前函数位置、必要性、调用依赖和有限文件身份。
A四个driver调用到recording边界是实参装配证据；A实际closure数值另证。
A-R1直接执行真实合法零雪closure、`_add_hydrol_to_previous_fields`和packet，再进入
已支持的无THERMOSOIL fallback；普通有THERMOSOIL的下步会重算profile，不能由此
宣称每步数值都受影响。仅定向查看同名字段的科学输出/诊断writer，二者原已消费soil
新profile；这不是全状态writer完整性审计。
D的AST调用执行真实组件，只证明装配/组件；cache stub只证明构造选择，均不宣称
完整driver控制流或整模型compiled执行。

### 共同基线、最终组合与接受登记

所有生产写入前保存[共同修前回执](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/common-before.json)：
27项=17有效fail/10pass/0error/0skip；三生产及三tests身份稳定，生产diff/index前后空，
父级实际验看JSON/log后才放行。B/C在首断言失败后的loop不算修前已验证控制。
原27个ID全部保留；修后增加A2+D32共34项接口/gate/cache检查，加25项直接相关旧回归，
初版选择86个唯一ID，未重跑Stage23全72或73oracle/广泛suite。

初版实际运行是**84pass/2fixturefail**；两项新directmain夹具误请求不完整albedo输入。
仅该函数改为合法的无albedo主接口，保留全部分数/init/restart/拒绝断言和容差；
完整D文件逆还此hunk的SHA精确等于原组合测试SHA，三生产和其余target均未变。
随后仅该2项重跑为2pass。故86个唯一ID均有通过证据，**不是单次86pass**。
原失败回执不覆盖，不把fixture失败登记为物理数值反例；
[初版联合验证](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/verification-review.json)
继续绑定初版生产/测试身份，其入口工件另存`history/initial-delivery/`。

主控实际审查要求的A-R1/D-R1仍属本Stage25。先在与初版相同三生产SHA的dirty实现上
保存[补项修前7项](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/revision-before.json)：
2有效fail/5pass/0error/0skip，无fixture/import/TypeError；不是HEAD-clean共同基线。
父级验看后放行两处最小生产修改，三tests保持修前身份，再按原12个selectors一次
运行[新最终组合](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/revised-final.json)：
**93pass/0fail/0error/0skip**，保留原86个ID并增加A1+D6，25个直接相关旧ID不变。
[补项联合验证](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/revised-verification-review.json)
给出原/新ID映射、真实命令/rootimport、JUnit/stdout、稳定的生产/测试身份及系统TEMP清理。
这次93通过绑定补齐后的最终生产，未重跑共同修前27、原调查或AD。
[复用的B变换观察](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/acceptance/bin-transform-observations.json)
仅为CPU JAX0.10.1两consumer非边界grad/vmap共8观察，与旧consumer AST body在当前
module namespace及共享真表下重放匹配，不是历史整module重放或一般AD准入。
[交付核验](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/delivery-verification.json)
记录源码/测试/文档身份及补丁重构检查。

补丁按A刷新/接口+driver、B非冻结bin/局部屏障、C已有诊断选择、D gate/传参/cache
分四组；三份新测试和两份报告草案各为依赖组。A依赖已有C13、C依赖C14、D依赖C15，
共同基线已含这些提交；同一hydrol测试文件的A/B/C分节不拆成重复测试。
四项均已加入[已接受总表](shared_core_branch_review.md)。主控复用最终93项验证，
没有另跑数值实验；补项修前失败与最终测试身份对应，八文件与交付manifest一致。
实际提交分为C17–C19水文组、C20参数组；两研究分支均待维护线程核对同步。
执行方补丁/manifest保留交付时文档草案身份，主控登记更新单独提交，不回写历史回执。

仅本地ORCJAX CPU Python-B/必要窄JIT，pytest禁cacheprovider，运行/cache均位于
系统TEMP并已清理。无epsilon/生产nextafter/全局编译开关/容差调整，未改雪算法或
disposition、其他参数/checks、Teacher/prepared/restart身份、研究工作树；没有模型日年、
完整模型编译、新Fortran/oracle、SSH/服务器/调度/GPU/训练/付费/环境重建/自动化或
跨聊天回传。执行方图复用Verify `E-ORCHIDEE-MAN-JAX` generation2026-10-25T07:07:50Z、
12327nodes/73750edges；Transport closed不重试，Python metadata_changed、Fortran
excluded、templates not_tracked均沿用精确源回退，不宣称新图/全域覆盖。
主控验收时图服务已恢复，generation为2026-10-25T14:39:20Z；定向查writer/parser及
调用关系，coverage仍标记相关Python metadata_changed、Fortran excluded，结合当前
源码/diff核对，不把服务恢复当作全域覆盖证明。

319误差`.0008192662350707258 > .0003951849734783173`仍failed，water-origin暂停；
V01–V03、C8 tiny/负值fatal/nonfinite、C10源负库存、Stage20两peat候选、raw-restart
旧容量和维护同步边界全部保留。本批组件修复没有建立年度NPP因果或改善证据。

**有限收尾计划：**下一批仅处理Stage24已列出的参数/检查合同、选定peat与restart
边界及雪maxmass允许域，逐项给出已证一致、需修复、不支持或证据不足的处置。
只有有效支持域内、源码与独立反例同时确认的小缺陷进入修复；未证域明确登记，
不无限追查。完成这一批和集中验收/同步说明后结束本轮源码审查。
V01–V03整模型/跨点/restart、319年度归因及669点/长期验证另作科学验证计划。

## 第二十六阶段：最后一批有限源码边界（主控已验收）

固定cwd为`E:/ORCHIDEE-MAN-JAX`，main基线
`61c6b3170419bb2c5d22eab80f98bfe314e9fa74`。启动根目录/分支/实际ORCJAX解释器和
`jax_orchidee`导入均已核对。本批只处理A四参数、B五check合同、C两peat及一次raw
restart归类、D三雪臂允许域，未重开Stage25或扩大到全配置。

2026-10-26主控复用已审A/B/C/D证据，验看R1实际hunk、独立修前4fail/7pass及
最终132项JUnit/log。20个被测源码/测试身份和11个交付文件均匹配，未重跑数值。
本批接受C21–C28共8项；四参数分别编号，fraction carry及其R1时序补项合计为C25，
诊断默认和显式组件修复与生产轨迹影响分列，不把8项等同于8个已触发paper物理bug。
C21–C25提交`c214e43da57c21b68002186fbf71c67d45835bae`；C26提交
`84b5d425418a6d1b18fa6dbe080348c2faa72e13`；C27–C28提交
`9fc87c011a4e991f1637223b0f8fb2a224000595`。
[主控验收回执](../../outputs/reviews/main-physics-20261026/final-source-boundaries/acceptance/primary-acceptance.json)
保留接受范围及与原changes-requested回执的关系。

| 接受编号 / 原发现 | 最终有界结论 |
|---|---|
| C21–C24 / A-MIN_WIND、A-HEIGHT_DISPLACEMENT、A-Z0_ICE、A-SNOW_DENSITY（按此顺序） | 四参数各自源owner/default/gate及已支持入口数值差异确证，已接受。前三项constantes、密度constantes_soil自身gate；默认.1/.66/.001/330，密度保留源正值检查。main/init/wrapper、cold直接fraction/main及两真实缓存键转发有效标量；静态/其他原reject模式不开放 |
| C25 / S26-A-FRAC-CARRY（含S26-R1） | 源sechiba SAVE及1085–1091生成的fraction在下一DIFFUCO997–1005沿用。补齐实际writer、mapper、cold映射和DIFFUCO可选字段优先；集中审查S26-R1再使restart scaffold从成功HYDROL的post雪计算fraction，避免旧fraction与新雪混合。必需restart字段不变。真实组件/调用表达式及writer/packet→mapper→assembler有独立before/after，非完整rollout |
| C26 / S26-B-TMC-DEFAULT（P4-17） | 已有显式tmc check API的allowed_err恢复源2e-8，停止边界10倍即2e-7；真实tile迁移roundoff及阈值控制通过。未改check=false默认或显式override，不称paper数值缺陷。P4-02/03/04缺check/fatal owner，未支持；P4-05保留无条件bounds-fatal静态缺口，只建立条件有效域代数不变量，无合法触发反例/生产修复 |
| C27 / S26-C-PEAT-ACCUM | 已有显式caller-supplied is_peat组件的连续peat buffer累积/非peat reset/累积回加恢复源1126、2102–2141；独立buffers1/3与DOC11/23最终通过。生产仍nvm14单PFT14，不开放多peat生产 |
| C28 / S26-C-PEAT-OLT | helper/core默认min_stomate恢复源1e-8及严格>诊断，显式override保留；只证peat_OLT诊断差异，库存/转移不变，无下游物理轨迹影响声明，C10不变 |
| C raw-restart | 证据不足并结束。实际已有restart和reader缺pcapa_en，源初始化仍有定义；唯一最小缺项是该文件/初始化控制下、首次energy前源初始化pcapa_en及producer身份，保存数组或独立验证重构均可。不造restart或同值结论 |
| D雪允许域 | 349true/355false/360true保持pft14_conditional/domain-unproven；360false局部逻辑保留。R<m3（充分U=m1+m2<3000）须在实际maxmass入口由准入restart/forcing及transfer/phase/sublimation保持证明；这是唯一缺项。未确认新翻译错，不改雪算法/registry或构造任意高密度轨迹 |

[最终M01–M12及独立V01–V03矩阵](../../outputs/reviews/main-physics-20261026/final-source-boundaries/final-matrix.md)
逐行记录一致、确证修复、未支持/源未定义、或证据不足及最小缺项。
四包报告、当前补丁/hunk依赖及后续同步建议由[交付入口](../../outputs/reviews/main-physics-20261026/final-source-boundaries/README.md)索引。

初版[组合回执](../../outputs/reviews/main-physics-20261026/final-source-boundaries/verification/final-integrated.json)
为**121pass/0fail/0error/0skip**：A新60、公共carry新7、B新9、C新8、直接旧37，历史身份保留。
actual-call表达式与真实组件数值、两cache构造stub分别标明，不认证完整driver/compiled。
121是初版最终单次结果；A/B/C修前26/9/8是三个独立运行，不拼成共同baseline。
父级最初fraction参考漏改同时生效的SNOWCRI_ALB=20，原错误参考及117pass/3fixturefail
保留，不计有效反例；正确源式下精确原公共源码重放3fail/3pass，字节恢复回执保留。
中间120pass之后发现实际writer仍丢字段，新增真实writer-before1fail，再补两字段；
最终121绑定该整合身份。详见[公共carry记录](../../outputs/reviews/main-physics-20261026/final-source-boundaries/fraction-carry.md)。
没有为测试改生产公式或容差。Stage25接受的93项只复用，不整套重跑。

所有数值均为本地ORCJAX CPU、Python-B及必要窄JIT；pytest禁cacheprovider，系统TEMP
已清理，最终源码/测试身份稳定，原log/JUnit/失败均保留。
图Verify实际恢复后调用；Python metadata_changed及Fortran excluded使用当前精确源回退，
不宣称全覆盖。[交付核验](../../outputs/reviews/main-physics-20261026/final-source-boundaries/verification/delivery-verification.json)
绑定当前源/测试/两报告和工件。

C1–C28累计28项/16代码提交；原C17–C19的`ad0d322cc4951d4af158f4e8cbb6d306bdf374f3`、
C20的`f8d36e1f8c03046e12e146aed2abb8bd286403de`身份保留。C4–C28研究同步仍待维护线程核对，
本批未修改其他工作树。319 `.0008192662350707258 > .0003951849734783173`仍failed，
water-origin暂停，V01–V03、C8 tiny/负值fatal/nonfinite、C10源负库存、TOPMODEL/fwet
生产未准入和历史Teacher/prepared/restart/reference producer身份保留。
无完整模型编译/日年、新Fortran/oracle、广泛suite/旧73、服务器/GPU/训练/付费、
环境重建、同步、自动化或跨聊回传。**本轮有限源码审查已收尾；整模型科学验证
按下面独立方案执行，不把组件验收当作整模型已通过。**

### 集中审查补项S26-R1与V01实际验证

主控已复核初版121及其20个被测文件身份、三个独立before和原有限结论；仅要求修正
restart timestep scaffold 的 CONDVEG 调用。它原本在HYDROL之后仍读取restart旧雪；
这是既有producer问题，初版carry进一步让下一DIFFUCO保留该旧fraction，不能将全部
影响都归于本次新增carry。没有无雪paper触发或319年度因果声明。

修前新增实际调用表达式/真实CONDVEG组件11项得到
[4有效数值fail/7pass](../../outputs/reviews/main-physics-20261026/final-source-boundaries/verification/r1-before.json)：
mass/explicit各自的vegetation/nobio在不同pre/post雪下失败；默认未变化、无snow_state
回退和缺显式几何控制通过。只在该调用覆盖成功HYDROL且有snow_state的六雪字段，
包括与albedo同刻的两age；无state保持原输入，几何缺失仍报告缺输入，未扩配置。
原7项carry的精确字节前缀保留，测试/源公式/容差未为失败调整。

最新[一次132项组合](../../outputs/reviews/main-physics-20261026/final-source-boundaries/verification/r1-final.json)
全通过，0fail/error/skip；原121全部ID+11新项，pytest35.28秒、runner39.312秒，身份稳定、
CPU/root import正确、TEMP清理。相对旧121只有orchestration与公共carry测试两个
被测文件变化，18个其他文件身份不变；原before/Stage25整93/调查不重跑。
[补项报告及精确hunk](../../outputs/reviews/main-physics-20261026/final-source-boundaries/revision-r1/README.md)
将实际实参表达式、组件及writer/packet→mapper→assembler与完整控制流分列，
HYDROL边界数组只作合法声明输入，未运行HYDROL或完整scaffold。
新AST核对仅restart_payload实参改变；旧121和初版交付快照均保留，R1属原carry闭合，
不新增独立修复ID/接受编号。本次index_status实际Transport closed，沿用主控图上下文
及当前精确源回退，不将旧generation称本次查询成功。

源码收尾时保存的[V01准备方案](../../outputs/reviews/main-physics-20261026/final-source-boundaries/next-validation-plan.md)
保留其历史未执行身份；主控接受并发布后，另行授权本地CPU实际验证。固定生产者
`d5d487f101ac49c2b933c91f1cb08d42bb473bfd`、main/root、ORCJAX与原001输入，实际入口
`outputs/validation/v01-after-stage26/run_v01.py`直接由`conda run -n ORCJAX python -B`执行。
[实际交付](../../outputs/validation/v01-after-stage26/README.md)与[机器汇总](../../outputs/validation/v01-after-stage26/001-api-9day/summary.json)
记录一次模型调用/8个模型日，计划的19个模型日没有全部执行。

首段cold8/block7完整返回：第1日普通冷入口、第2–8日实际编译块；8个日界均48步，
无missing/stopped，末`tstep=383`。每日输出264个数值叶/3384个元素全部有限。
末packet扫描353个数值叶/23994个元素，`diffuco_previous_step_state`及
`sechiba_finalize_state`两处`roughheight_pft`均为`float64[1,14]`，各有一个`[0,0]=NaN`，
没有Inf。这是相同坐标/值的两个组件条目，不据此计为两个独立缺陷；原始验证未作inactive/mask
豁免或生产bug起源判断。**原始尝试未通过独立有限值门槛；当时第9日续跑、恢复第9日和普通9日
均未启动**，相应连续性/普通编译误差尚无结果，原`atol=1e-8,rtol=1e-10`未放宽。

实际动态`pcapa_en`为有限`float64[1,32,14]`，范围1343518.9620269747–2186328.39145471。
完整返回、8日输出、末packet（含provenance）与容量已先落盘；离线0/0存储一致性最大
绝对误差为0，但比较器允许匹配NaN，不能覆盖独立有限值失败。[原失败及traceback](../../outputs/validation/v01-after-stage26/001-api-9day/failure-37196.json)
完整保留，无模型重跑。首段205.9292175秒含编译/同步；失败尝试208.6074725秒含导入/
预检至失败，不是完整V01总量或纯模型吞吐。源码/config/run.def/runner/comparator及
输入身份在运行前后相同，生产diff为空；本次系统TEMP子目录已验证包含关系并清理。

随后仅做[有界非有限值调查](../../outputs/validation/v01-after-stage26/nonfinite-investigation/README.md)，
未运行任何新模型积分。实际坐标为001点、Fortran PFT1/bare ground；保存末状态的height、
veget、veget_max、lai该位置均为0，冻结配置IMPOSE_AZE=false、ROUGH_DYN=true、
HEIGHT_DISPLACEMENT=.66，OK_LAIDEV1–14全false。保存末状态不是此前CONDVEG调用输入，
因此不能据此确定首次出现NaN的日/步或冒充历史producer回放。

Fortran condveg.f90的静态1467–1472、动态1705–1717只为jv=2..nvm赋值，
对应INTENT(out)的PFT1未定义；不能推断实际Fortran内存必为NaN。JAX condveg.py:424/571
显式以NaN初始化，循环从索引1开始，PFT1保留sentinel；并非该坐标除法/log失效。
该机制在仓库根基线`7333b46c0b38650fb6c9250582876137831657b8`及Stage26直接父版本均存在，
不是Stage26新增；消费端safe mask来自`5910ba9af857a91e565d13a25e3de2e3019080d1`。
imposed分支会定义全部PFT，不能推广为所有PFT1字段或所有配置的豁免。

已沿CONDVEG→writer→两处packet→ordinary/fast-layout/恢复mapper核对保存和运输链；
DIFFUCO在log、乘除及sqrt前用`OK_LAIDEV & (index>=1)`选择安全roughheight_pft，
再进入输出选择；裸地/grid标量roughheight仍被物理计算使用。共11次局部函数调用，
其中1个窄CPU JIT封装调用2次：三条运输/消费路线的原NaN0与有限替代123对照均11叶/
63元素有限、maxabs0；动态/静态producer有限输入控制仅未定义列保留NaN。
反事实启用PFT14的局部守卫控制也不受NaN0影响，而向启用列注入NaN使3个输出数组
出现NaN。这是组件与运输证据，不是实际冻结配置的替代科学验证、完整compiled/restart
准入、所有配置或AD证明；该次调查未运行整模型第9日或普通9日。

主控已在[正式裁定](../../outputs/validation/v01-after-stage26/primary-domain-decision.json)接受
“源未定义且已审计消费者在算术前排除的坐标，未证明新生产缺陷”的有界解释及严格合同。
合同初始仅限本次001点/1961/d5d487f、legacy14/NVM14、IMPOSE_AZE=false/ROUGH_DYN=true
的实际动态重算producer，静态控制只作机制证据。对两处命名roughheight_pft的[0,0]单列未定义域并保留预期NaN
sentinel，不豁免Inf；全部源已定义坐标及其他字段仍须有限，不作通用mask
豁免或nan_to_num修复。原V01 failed/失败traceback不重标，生产物理及原脚本不变；
本批已另建[domain-v2验证器与回执](../../outputs/validation/v01-after-stage26/domain-v2/README.md)，
32个窄控制均得到预期接受/拒绝，旧stage1离线重评通过。按已接受的纯验证合同调整，
未变物理计算的首段保存结果可保留原producer身份复用，
由另版门槛生成派生回执；不能把旧failed改为passed。若修改物理代码，受影响状态/输出
须由新producer重新生成，旧packet仅留诊断身份。主控已授权在新回归及离线重评通过后
执行余下三段、最多增加11个模型日；不得重跑首段8日。原进程已退出，两条第9日路线分别为
“完整结果重新加载后的续跑”和“独立packet文件恢复续跑”，不称未中断同进程8+1。

domain-v2当时只完成第一条day9续跑，tstep384–431、48步，完整结果/daily/packet/动态容量均已同步落盘，
耗时104.8217936秒含编译/同步。随后独立域验证因packet schema/provenance差异停止；
[失败回执](../../outputs/validation/v01-after-stage26/domain-v2/execution-001/failure-48208.json)保留。
[只读schema核对](../../outputs/validation/v01-after-stage26/domain-v2/schema-failure-inspection.json)
确认全部数值字段shape/dtype与基线一致，每日33叶/423元素严格有限，packet353叶/23994元素
仅原两处roughheight_pft的[0,0]为NaN；这些数值扫描不覆盖provenance失败。

唯一结构差异为`provenance_by_component.slowproc_stomate_previous_step_state`从7条变6条，
少了`stomate.f90 lines 3288-3489 half-hour OK_LEAK state writeback`；其余6条顺序和值相同。
[原值对照](../../outputs/validation/v01-after-stage26/domain-v2/provenance-differences.json)及精确源码
将记录生成定位到orchestration.py:2775的2条base、8213–8239的1条half-hour追加、9275–9333
的4条day-end追加。不能仅凭此记录差异推断物理过程漏执行或数值错误；当时domain-v2合同要求
与seed的provenance精确一致，故保留显式失败，未改已执行验证器或数值容差。
在该历史停止点，独立packet恢复day9、ordinary cold9、两路线连续性及普通/编译比较均未运行。
domain-v2新增1模型进程/1调用/1模型日，当时累计2模型进程/2调用/9模型日；32控制和离线读取
不计模型积分。原source8不重跑，已完成day9保留供后续有身份复用，不自动重试或扩展。
新调查有独立manifest，原17工件及冻结索引SHA保持；图服务本次Transport closed，
仅用历史Verify上下文和当前精确源码回退，不称本次图覆盖成功。原raw pcapa_en、319、
TOPMODEL/fwet、支持域及研究同步限制均不因该调查解除。

### V01最终验收：001点九日及动态状态恢复

主控按[来源记录裁定](../../outputs/validation/v01-after-stage26/primary-provenance-decision.json)
复核[13字段写回证据](../../outputs/validation/v01-after-stage26/domain-v3/source-closure/README.md)：
最终半小时OK_LEAK结果经显式daily-state选择和day-end覆盖写回，后续覆盖字段集合与其不交。
17组真实组装控制包含221项字段断言及8项路线检查；这是有形状的标记值/组装证据，
不计作221次模型或独立科学验证。编译块保留首日7条来源，later/ordinary末状态生成6条；
这是本次路径的记录生成差异，没有证实数值字段漏写。

domain-v3按源码生成完整路线来源映射，数值结构/类型/离散值和producer身份仍严格检查。
12项新增路线/结构/非有限拒绝控制通过，原32项未变合同控制及11次NaN局部探针复用。
只有两处命名roughheight_pft的[0,0]须保留预期NaN，全部已定义坐标、其他状态及每日输出
仍有限；没有通用NaN豁免或容差调整。原strict-all-finite与domain-v2的失败回执保留。

[实际最终汇总](../../outputs/validation/v01-after-stage26/domain-v3/execution-001/summary.json)
记录复用原source8和重载day9，新运行独立packet恢复day9及ordinary cold9，共新增10模型日。
累计19模型日/4次调用/3进程，未重复已完成阶段。两种day9均为原进程退出后加载，
不称同进程未中断8+1；ordinary cold9为一次连续冷启动。恢复day9的完整packet（含来源）
和每日值最大差异0；普通/编译1–9日全部每日输出及最终数值状态通过原atol1e-8/rtol1e-10。
每日输出最大绝对差异2.3125945602942011e-13；全状态最大绝对差异
2.5588087737560272e-7位于大尺度pcapa_en，在原相对容差内；最大相对差异
3.9964615879614042e-12位于wat_flux。有限域、48步/日、最终tstep431、无missing/stopped均通过。

新两次模型调用分别141.2581363秒和85.6985076秒，含编译/同步；新进程完整模型窗口
249.3094987秒，不称纯吞吐或包含前三批工作的总耗时。主控验看实际验证器、runner关键调用、
数值回执及源码，核对46项新工件、13项运行源码和7项写回取证源码身份；未重跑模型/组件。
原17/59/43项工件保留与新保存对象0/0回读复用执行方核验。**主控接受本次有界V01通过。**

本批仅增加outputs验证工具/证据及两份状态报告，没有新增生产修复，C1–C28/16代码提交不变。
V01交付时下一项为原319点/1961/365日、冻结输入和原年度门槛的V02；V01批次未运行V02/V03。
随后另获授权的V02实际结果见下节，不回写V01历史回执。
raw NetCDF容量证据、319历史失败、water-origin暂停、TOPMODEL/fwet/其他支持域及研究同步
限制保留；九日路径/动态状态验证不构成年度、跨点、长期或全配置科学接受。

执行方交付时未stage/commit/push。主控验收后按上述三组提交代码/测试，接受登记
另行提交；原manifest与旧回执保留交付时文档身份，不回写历史。C1–C28累计28项/
16代码提交，所有原有限未知、C8/C10源限制、TOPMODEL/fwet未准入、319失败、
V01–V03和历史producer身份保留。研究分支待其维护线程按准确提交核对同步。

### V02实际结果：319点1961全年完成，NPP原科学门槛仍失败

V01主控验收/报告发布后，本批另获授权执行原319.0-057.0 / I5 /
S26_69.834_0.0019_0.3980_97.918、1961冷启动完整365日。固定生产基线
`876ac3bb8c49469ae9d9a9e0aff5982a1e4ebd64` / main；相对V01的`d5d487f`仅报告变化，
生产源码相同。[原方案](../../outputs/reviews/main-physics-20261026/final-source-boundaries/next-validation-plan.md)
仍为历史未执行方案，不改作运行回执。实际结果见
[V02交付](../../outputs/validation/v02-after-stage26/319-1961/README.md)、
[执行回执](../../outputs/validation/v02-after-stage26/319-1961/execution-receipt.json)及
[完整离线汇总](../../outputs/validation/v02-after-stage26/319-1961/summary.json)。

实际入口为`conda run -n ORCJAX python -B -m jax_orchidee.cli.main run`，
复用原config及319-current/inputs/used_run.def，SHA分别为
`3e3ec5c4518ebf4d470c60eda3e9bd6ddfd22e5c6dee473694f2409f17e619af`和
`5ecb4d2615dec4d54a88f2db5444ac2820d8327ec6c10c6af16945b4ff56374a`。
原15诊断字段、全部开关及compiled block7保留，只有输出/checkpoint目录变化；未重新物化run.def。
真实导入为本工作区包，ORCJAX/CPU进程PID53212，Python -B、持久编译缓存关闭。
一次预检后，实际1模型进程/1调用/365模型日，无年度重试；末tstep17519，48步/日，
365个日界及15字段各365日的float64(1,14)输出通过，未缺项或提前停止。
原runner计时264.5704782秒含编译/同步；CLI进程281.3368429秒另含导入、完整对象保存和汇总，
不作纯吞吐对照。两条XLA algebraic-simplifier循环警告保留，进程正常exit0。

完整年度实际run在原汇总前由只读观察器保存，并按0/0精确回读；原_summarize_year在保存对象上
离线重算，完整year payload与CLI保存值一致。年度checkpoint完整packet（含来源）与run精确一致，
pickle tuple/JSON list仅在元数据的原序列化边界规范化，数值/类型/来源的packet比较不放宽。
真实319配置为legacy14/NVM14、动态且non-imposed CONDVEG；只有两处命名roughheight_pft
float64[1,14]的[0,0]为源码支持的未定义NaN，其余定义坐标、每日输出及pcapa_en有限。
365=1+52×7的实际完整块路径保留首日来源，由源码helper生成完整来源合同并逐项核对，
未按点号直接放行或强制合法跨路来源与001 seed逐字相同。
源码必需字段合同及260个数值字段的shape/dtype核对通过；另92个额外字段没有完整的独立
源码shape合同，已逐项列出并保持严格有限域/完整存储检查，不称全部packet结构已获源码认证。

按原acceptance汇总语义，年度限为`1e-5 + 1e-3*abs(reference)`，normalized bias绝对值及
normalized RMSE均限1e-3。四项结果如下；原始精度及完整诊断保存在机器汇总中。

| 指标 | 新JAX | 同一参考 | 绝对误差 | 原年度限 | normalized bias / RMSE | 科学结果 |
|---|---:|---:|---:|---:|---:|---|
| AGB_model | 4.591235866253038 | 4.590915203094482 | 0.000320663158555945 | 0.004600915203094482 | 6.9847327683e-5 / 6.9847327683e-5 | 通过 |
| BGB_model | 4.120666562048175 | 4.120461463928223 | 0.000205098119952396 | 0.004130461463928222 | 4.9775521928e-5 / 4.9775521928e-5 | 通过 |
| GPP_model | 0.7776203738914573 | 0.7776592373847961 | 0.0000388634933388143 | 0.0007876592373847962 | −4.9974965217e-5 / 4.9974965217e-5 | 通过 |
| NPP_model | **0.3860031827055333** | **0.38518497347831726** | **0.000818209227216038** | **0.0003951849734783173** | **0.0021241981997049434 / 0.0021241981997049434** | **失败** |

所以本次运行完整，年度科学接受仍为false，normalized统计也因NPP失败。单年度序列仅有一对值，
不能当逐日时间序列表现；correlation无定义。另列的诊断限仍为atol2e-6/rtol1e-7，诊断总门槛false，
不替代或放宽科学门槛；所有年度字段及诊断差异均已完成统计。

原319-current producer `09968cf86551e8dfc039c1740b94fb3d42cf3f12`、原payload/receipt/analysis身份
及同参考/输入已核对。历史失败`0.0008192662350707258 > 0.0003951849734783173`保持，
旧payload直读保存误差`0.0008192662350708368`也单独保留，不用显示值相减改写历史。
新旧JAX的AGB/BGB/GPP/NPP分别变化约−5.7891e-8、−3.0846e-7、−1.22782e-6和−1.05701e-6；
NPP误差略减但仍超限。旧CLI只存摘要，没有补造其逐日向量。另复用已独立绑定相同输入的
Stage6 producer `c7110e030bae21512538b1ae45add8d0317ce0d7`四通量逐日捕获作辅助新旧比较：

| 逐日字段 | 新−历史JAX的日均变化 | 最大绝对日变化 | 所在日 |
|---|---:|---:|---:|
| GPP | −1.22781986075e-6 | 0.000263996068 | 218 |
| NPP | −1.05700785475e-6 | 0.000189970558 | 218 |
| MAINT_RESP | −1.95024679850e-8 | 6.23768007468e-7 | 364 |
| GROWTH_RESP | −1.51309538012e-7 | 7.38774392221e-5 | 218 |

指定早/晚年及历史脉冲窗口的有界离线统计保存在summary；这些变化和相关性不构成某项修复的
因果归因，历史Fortran记录未重标为本次producer。科学失败后仍完成全部离线分析，未再积分或展开归因矩阵。
一次pickle/JSON元数据表示差导致的离线harness失败记录保留，修正后继续读取同一年度对象。

源码/config/run.def、27个有限输入路径元数据及旧工件身份在执行前后不变；系统TEMP已安全清理。
本批[最终静态核验](../../outputs/validation/v02-after-stage26/319-1961/final-verification.json)
保存工件清单、实际argv/来源身份、报告链接和修复登记核对；该核验没有模型或统计重算。
图服务Transport closed，采用精确源码回退，不称图覆盖成功。仅两份报告为tracked改动，
无生产物理/参数/容差修改、无新修复，C1–C28累计28项/16代码提交及研究同步状态不变。
2026-10-27主控接受本批交付和运行/存储验证证据，明确保留年度科学接受false及严格诊断false，
不将“交付验收”写成“模型科学通过”。主控核对33份新工件及15份当前源码身份，复用执行方13项
离线检查，并审查模型外保存包装与原判据；未重新积分或重算统计。见
[主控验收](../../outputs/validation/v02-after-stage26/primary-v02-acceptance.json)及
[身份核对](../../outputs/validation/v02-after-stage26/primary-v02-identity-verification.json)。
260字段独立shape/dtype合同的通过范围、92额外字段合同缺口与两处源未定义槽位均保持原限定。
本次仅发布两份报告；原交付manifest及回执仍保留交付时点身份。
V03、其他点/年份/长期、raw NetCDF容量、TOPMODEL/fwet/支持域、water-origin暂停与研究分支
待同步均未由本批验证解除；后续有限执行检查须另行派发，不扩大669点或长期科学验证。

### V03a实际结果：继承数值合同通过，额外载体检查失败保留

V02主控验收及发布后，本批另获授权执行本地ORCJAX/CPU有限跨点一致性检查。固定main
`a317c77750f7f2f42db21edaf05e6fa7f7a0c3a2`；相对接受的V01 `d5d487f`、V02 `876ac3bb`
只有报告变化，实际生产源码相同。复用原config及两个实际used-file，不重新物化输入：

| 点 | 冻结run.def SHA256 | 实际W/E/S/N | 实际PFT14 VCMAX / ARJV / slopeC / alloc / residence |
| --- | --- | --- | --- |
| A：001.0-071.0 / I10 / S2_63.206_0.0876_0.2019_50.658 | `f75d64dcbe321a0934368a415f1820687e00d5e6e0829f2c808bd2313e5343c2` | −180 / −178 / −20 / −18 | 63.2061836 / 2.45712368323 / .0876862 / .2019211 / 50.6583452 |
| B：319.0-057.0 / I5 / S26_69.834_0.0019_0.3980_97.918 | `5ecb4d2615dec4d54a88f2db5444ac2820d8327ec6c10c6af16945b4ff56374a` | 138 / 140 / −34 / −32 | 69.8349644 / 2.30694746155 / .0019037 / .3980902 / 97.9182736 |

config仍为`configs/orchidee_man_250919.yaml`，SHA
`3e3ec5c4518ebf4d470c60eda3e9bd6ddfd22e5c6dee473694f2409f17e619af`。
真实域中心A为lat/lon −19/−179、contfrac .0625，B为−33/139、contfrac1；不是YAML示例区域。
两点legacy14/NVM14/NSTM6、non-imposed/dynamic CONDVEG、全部14个OK_LAIDEV=false及共同参数
均按实际used-file核对。源码、47个既有输入元数据路径及V01/V02原manifest/seed身份在前后保持一致。

一次充分预检确认实际Python为`C:/Users/admin/.conda/envs/ORCJAX/python.exe`，导入当前根工作区包，
设备`cpu:0`，Python -B及持久编译缓存关闭。通过公共API
`paper_1961_driver_cold_start_multiday_modelout_lite_run`运行，两个实际模型进程顺序执行：

| 实际调用 | Python PID | 路线 | 调用计时秒（含编译、同步及边界观测保存） |
| --- | ---: | --- | ---: |
| A-first | 58044 | 独立cold day1 + 一个真实compiled block7 | 176.7734517 |
| B-compiled | 58044 | 独立cold day1 + 一个真实compiled block7 | 24.1930783 |
| A-return | 58044 | 独立cold day1 + 一个真实compiled block7 | 12.1478658 |
| B-ordinary | 59312 | 独立cold8，compiled SECHIBA off/block0 | 107.1586996 |

共享进程无cache clear或module reload。普通B共同开关与接受的V01普通路线一致：
static daily carbon/module JIT/DIFFUCO local JIT/prebuild day payloads均true，single-pass daily fold及
NumPy accumulator均false。四个真实模型调用均完成8个48步日界，末tstep383，无missing/stopped；
完整run、daily、final packet、实际pcapa_en及精确回读先于下一次模型调用保存。
外部shared/ordinary进程分别286.9264433/156.1406413秒，另含导入、检查及工件写入；不作稳态吞吐比较。
ordinary进程exit1来自模型保存后的类型比较断言，模型本身已完整返回；没有自动重试或额外积分。

[合同与交付收尾](../../outputs/validation/v03a-after-stage26/contract-closure-v1/README.md)为本次补项入口；
[原汇总](../../outputs/validation/v03a-after-stage26/summary.json)、
[收尾前派生汇总](../../outputs/validation/v03a-after-stage26/final-summary.json)及
[原普通路线检查](../../outputs/validation/v03a-after-stage26/ordinary/gates.json)保留如下结果：

| 比较 | 最大绝对差 | 最大相对差 | 结果 |
| --- | ---: | ---: | --- |
| A-first、A-return各自对接受的V01 source8 | 0 | 0 | 全daily与final packet通过，含原类型/来源 |
| A-return对A-first | 0 | 0 | 完整对象通过，含原类型/来源 |
| B-compiled对V02年度对象的前8个完整daily | 0 | 0 | 通过 |
| B-ordinary对同一V02前8个完整daily | 3.5271785492341223e-13 | 8.150827422816209e-13 | 通过 |
| B-compiled对B-ordinary，本次额外载体检查 | 6.644986569881439e-7（已比较部分） | 2.404596709685098e-10（已比较部分） | **44个叶载体类型失败；该历史检查的数值扫描不完整** |
| B两路线，沿用V01数值状态合同的完整比较 | 6.644986569881439e-7 | 2.404596709685098e-10 | **通过原限，数值失败0**；复用另版完整比较证据 |

合同来源为接受的[V01 domain_verifier.py](../../outputs/validation/v01-after-stage26/domain-v3/domain_verifier.py)
中schema的23–36行：数值叶经未指定dtype的`np.asarray`
记录kind=numeric、原shape及dtype，不要求JAX/NumPy/Python载体类相同。V01实际
`run_remaining_v01.py::execute`的118–133行先经DomainVerifier准入两路线，再使用
`scripts/dev/audit_compiled_fast_path_acceptance.py::ComparisonAccumulator.compare`的113–187行比较
daily与final fields/tstep；其139–149行数值入口亦为np.asarray。shape/dtype保障来自schema与parity的组合，不能由历史比较器的
float64差值算术单独推出dtype精确。
本次冻结`validation.py::compare`的188行在数值检查前新增无条件`type(a) is type(b)`，因此44项
属于额外载体条件失败，并非V01既有数值状态合同失败。它们为43个JAX ArrayImpl对NumPy ndarray、
1个ArrayImpl对Python float（`slowproc_stomate_previous_step_state.dt_days_read`）；原summary/gates/failure、
冻结工具和manifest均不改写，额外strict carrier false永久保留。
当前公共packet/daily/lite协议在orchestration.py的1726–1731、2370–2378及2446–2463行，
数值payload以object表示而未固定载体类；原pickle保存实际对象，不规范化载体。
compiled重建直接保留返回值、ordinary日末合并保留载体及dt_days_read=float(dt_days)的源解释，
仅覆盖这些已核对边界，不追认44叶全部上游历史或认定新的物理bug。

已完成的[完整比较证据](../../outputs/validation/v03a-after-stage26/numeric-diagnostic-v1.json)在数值叶
以np.asarray读取原值，不指定转换dtype、不改保存数组；完整键/容器、shape/dtype与离散检查仍精确，
float仍用`atol1e-8 + rtol1e-10*abs(reference)`。全部620数值叶/27381元素、44个原跳过叶均通过，
6个负控制仍能拒绝容器、dtype、shape、键、离散值和超限float变化。
本次收尾复用这些已完成证据，按已接受V01数值状态合同的跨路比较结果为通过；仅对上述实际44个
numeric载体关系作有据等价，不通用放行其他类型、不忽略其他metadata、不放宽浮点限。
完整数值结果与额外载体检查的历史false分列，不伪造模型重跑或修改旧回执。
最大绝对差在末packet `thermosoil_previous_step_state.pcapa_en[0,2,0]`：compiled
1127009.884473888、ordinary 1127009.8844732235，差6.644986569881439e-7，原限
0.00011271098844732235。最大相对差在`hydrol_previous_step_state.wat_flux[0,7,3]`：
.00060882007618979对.0006088200760433933，原绝对组合限1.0000060882007604e-8。
各最差字段、坐标、原值、参考和实际allowance均见机器回执；未改物理、参数或容差来取得数值结果。

所有实际对象均先独立通过有限定义域及实际路线来源检查；仅两处命名roughheight_pft的
float64[1,14]、精确[0,0]保留源未定义NaN，其余定义坐标、全部daily及动态容量有限。
同路/存储完整来源精确；compiled 1+7末packet的slowproc来源7条、ordinary为6条，分别匹配
精确源码生成的路线合同。跨路数值诊断不把来源数量强制相等；两条B daily又各自与V02完整daily
作全来源比较。V02没有独立day8末packet，只比较其真实前8日对象，未重建或补造旧day8状态。
沿用260数值字段的独立源码shape/dtype合同与92额外字段合同未完整覆盖的限制；派生成对比较检查
全部实际字段的shape/dtype，不能升级为92字段已获独立源码认证。

模型外边界观察器保留原函数实参与返回对象，用`is`对实际cache前后对象核验。
prepared LRU依次为A miss、B miss、回A hit；A首/回context id2182045034000，B为2185814025808。
三个compiled段实际key相同（记录SHA `17e9f7c34a3c3f77daa82fdcfb7571591c4dc66a0b1bf607767f37a0cbfb0076`），
外层可执行对象id2183757175440；A首次新增1项，B及回A均复用原对象、cache增量0。
九类动态输入从真实caller边界保存为完整pickle并0/0回读，覆盖state、forcing、day numbers、
STOMATE参数、HYDROL表、landpoint、restart template、season及DIFFUCO参数；实际点位/参数绑定随段记录。
未代理可执行体、改写key或替换数值图。只在A-first观察到一次inner factory边界，B及回A外层命中
可跳过该边界，不能把其0次观测称新的inner命中证明或真实XLA编译/重跟踪次数。

此前[primary-v03a-acceptance](../../outputs/validation/v03a-after-stage26/primary-v03a-acceptance.json)
实际为本执行聊天的内部审查，生成时项目主控聊天尚未最终验收；该历史文件与身份保留，不能作为
项目主控最终接受凭据。2026-10-27项目主控已接受本次两点短窗的真实缓存复用、点位绑定与
沿用V01数值状态合同的完整比较，见
[项目主控最终验收](../../outputs/validation/v03a-after-stage26/controller-final-acceptance.json)、
[本次收尾入口](../../outputs/validation/v03a-after-stage26/contract-closure-v1/README.md)及
[主控补项审查](../../outputs/validation/v03a-after-stage26/controller-review-20261027.json)。主控复用此前
73份工件/24份当前源码身份核验和执行方检查，核对新增收尾工件及合同来源；没有重跑模型或数值统计。
额外载体检查的44项原失败仍为false，不回写旧工具/门槛，也不将本次有限执行验证称为年度科学通过。
原[静态核验](../../outputs/validation/v03a-after-stage26/final-verification.json)保留当时报告身份；
新收尾证据另存独立目录。系统TEMP已清理，图Transport closed下沿用精确源码回退；本次仅核对合同及
文档、链接、身份和修复登记，不新增模型/组件调用或数值重算。
本批只发布两份报告，无新生产修复/Stage27/C编号或研究分支同步，C1–C28仍28项/16代码提交。
V01原严格失败、V02年度NPP及严格诊断失败、raw NetCDF容量、TOPMODEL/fwet/支持域、
暂停water-origin与C4–C28研究同步限制均保留；两点单年短窗没有完成669点、年度/多年度科学验收，
也未形成某项修复的因果归因或新的物理过程缺陷证明。

## NPP局部分支敏感性：两次历史输入干预已完成并主控验收（2026-10-27）

本节保留上批主控验收时的证据、缺项及344日原建议；其中“未保存/未实现/未执行”是该批历史状态。
后续已获准并完成的365日捕获与当前Day338判别见下一节，不把新tuple改称旧producer原捕获。

**已证明：在两份真实历史Day338完整输入背景内，仅把PFT14 GPP在0与
1.2241381168386465e-6之间交换，就能触发约0.315的最终NPP/维护呼吸变化。
尚未证明：这项上游GPP差异来自浮点舍入，或它已解释当前319年度NPP失败。**
这两个层次不能合并为“已查明数值误差”，也不能因一侧未完整复现就否认局部分支放大。
完整结果、证据表与下一观测方案见[本轮README](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/README.md)，
实际两次返回见[机器汇总](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/response-summary.json)。

本轮在固定main `2c15d4b0d76af6333084c0a8e93a33369da4b8a6`、本地ORCJAX/CPU完成。
干预背景来自Stage8 producer `e05369c9e70b4c982f34545e37af7c1782d8142d`的实际JAX输入，
及原Fortran binary `64bd0e4b8f598e443ac8a1c839a66f1d7d15474c7b74b3302dd031ab0739774a`的实际输入；
两者时钟为1961 Day338，F step16224 / J零基日末16223。并非当前V02输入或新增Fortran观测。
整个`carbon_kernels.py`在Stage8、V02与本轮的SHA均为
`56516c8690ab9f656f940c106871a50beb5531f5ea881cc8d1280d0b67a5c1e1`；
原Day338/339/343三完整Fortran输入同源回放18/18比较沿用，不重放baseline。
源码相同支持这项局部机制的适用性，不证明上游状态、运行时或全部分支相同。

每次完整冻结biomass、f_alloc、resp_maint_part、pft_present、frac_growthresp、dt_days、tax_max、
min_stomate及GPP其余坐标，仅换`gpp[0,13]`；shape/dtype与原记录一致，dt=1天、tax=.8、min=1e-8。
六个真实返回数组全部保存为JSON和原NPZ。下表为PFT14原组件通量单位`g C m⁻² day⁻¹`；
Δ为干预减本背景原值，残差为干预减对方原值，原限为`1e-8+1e-10*abs(对方原值)`。

| 背景 / 输出 | 本背景原值 → 干预值 | 对方原值 | 有符号Δ | 有符号残差 / 原限 | 原限内 |
|---|---|---|---|---|---|
| F / NPP，GPP 0→1.2241381168386465e-6 | 9.999999994736442e-9 → −0.31504082549082607 | −0.3155346427512101 | −0.31504083549082607 | +0.0004938172603840552 / 1.0031553464275122e-8 | 否 |
| F / 维护呼吸 | −9.999999994736442e-9 → 0.3150419810772084 | 0.31553579833759243 | +0.31504199107720837 | −0.0004938172603840552 / 1.003155357983376e-8 | 否 |
| J / NPP，GPP 1.2241381168386465e-6→0 | −0.3155346427512101 → 2.7977620220553945e-13 | 9.999999994736442e-9 | +0.3155346427514899 | −9.999720218534236e-9 / 1.0000000001e-8 | 是 |
| J / 维护呼吸 | 0.31553579833759243 → −2.7977620220553945e-13 | −9.999999994736442e-9 | −0.3155357983378722 | +9.999720218534236e-9 / 1.0000000001e-8 | 是 |

因此仅J→零GPP在原限内复现对方的**两个主要输出**，且残差接近原绝对限，不能称exact。
两侧六返回全坐标比较均未全部通过，见[完整离线分析](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/mechanism-analysis.json)。
F→微小正GPP已有大幅跳变，但仍剩0.0004938172603840552，远超原限；非GPP维护/库存/分配背景及交互
仍影响完整原生差额。“更近”只表示方向，部分解释、背景依赖和严格复现分别报告；不新增通过阈值或年度贡献比例。

源条件解释来自`stomate_npp.f90::npp_calc`314–387/449–531及
`carbon_kernels.py::npp_closed_update`3954–4057。先按PFTpresent汇总原维护部件R，
令B=GPP×dt、T=tax_max×B；enough为`B>0且R×dt<T`，over_tax为`B>0且非enough`，no_alloc为`B<=0`。
由两份实际完整输入与这些源码谓词确定：两份零GPP情况走no_alloc，两份正GPP情况走over_tax，enough均false。
这是输入及谓词确定的结果，未伪称新增内部观测。零GPP路径把全部维护扣到碳储备；微小正GPP路径保留
税后少量分配，并按维护部件比例从指定活组织/储备泵取余量。随后源码对负库存补至min_stomate，
把创建量从维护呼吸中扣除，最终NPP=GPP−growth−maintenance，故不同扣库路径可产生很不同的最终净呼吸。
本轮直接观测的是六返回；反事实negative/pre-floor/bm_create局部量未直接输出，不从返回维护呼吸倒推成“实测floor”。

| 证据类别 | 本轮可确认 | 必须保留的边界 |
|---|---|---|
| 绝对量级 / 小分母 | V02年度参考约.385，绝对误差越过原限；近零日值同时报告绝对量 | 年度失败不能归咎于近零分母；小绝对GPP差也不等于已证舍入误差 |
| 输出采样 / 聚合 / 精度 | 复用Stage8 core→send36/36、完整NPP同输入18/18；Stage10同一NetCDF高精度重导恢复34/34 | 三项float32存储严格double诊断失败保留；输出精度/所选聚合排除不排除内部数值敏感性 |
| 内部阈值 / 状态扰动 | 本轮两次固定背景干预证明局部零/正分支敏感；旧Stage12液水跨阈值与48桥精确，Stage10完整65参DIFFUCO同输入通过 | 旧液水入口已不同，Stage13未保存完整配对soil tuple；不证明舍入起源、当前首分歧或未查边界无bug |
| 已确认源问题 / 修复相关性 | 原28修复/16代码提交及接受范围保持；本轮无新源级同输入反例 | 修复登记、年度轻微改善或同输入局部通过都不等于当前全年定因 |

当前年度基线仍是V02 producer `876ac3bb8c49469ae9d9a9e0aff5982a1e4ebd64`：
NPP0.3860031827055333、reference0.38518497347831726、误差0.000818209227216038，
大于原限0.0003951849734783173；科学及严格诊断失败均保留，不重算旧年度/逐日表。
本批此前有界读取的实际PFT14日值如下，单位仍为原组件日通量；不是历史反事实的输入替代。

| V02日 | GPP | NPP | MAINT_RESP | GROWTH_RESP |
|---|---|---|---|---|
| 337 | 0 | 9.999999994736442e-9 | −9.999999994736442e-9 | 0 |
| 338 | 4.564204140244641e-7 | −0.3155349337862027 | 0.3155353646470735 | 2.5559543185369996e-8 |
| 339 | 0 | 1.801669924361704e-12 | −1.801669924361704e-12 | 0 |
| 343 | 1.9396509450671202e-6 | −0.3119493121936352 | 0.3119511432241273 | 1.0862045292375871e-7 |

这些当前日值支持脉冲形态仍在，尚不足把当前年度失败定类为“已证数值分支敏感性所致”。
当前保存对象仅有day365/tstep17519终态，没有day336状态或逐日完整pre-NPP九参、floor/native水文输入；
不能从四通量补造它们。原Stage6与V02在Day218的新旧JAX最大日差继续保留其原比较身份，
不改称当前Fortran首次分歧，也不自动转移观测靶点。

唯一建议的后续判别先限于**与V02相同生产源码/输入、明确记录新producer的Day337–339完整NPP边界观测**：
九参、六直接返回及精确clock/phase；新观测须与保存V02同窗输出在原限内自准入，不把新生成输入改称旧producer原捕获。
若讨论floor则需直接保存对应pre-floor/negative/bm_create。须先找到有来源且完整的day336重放状态；
若不存在，只能另获准后从原cold-start演化。V02路线是cold1+52×7，337为块尾、338–344在同一7日块；
`orchestration._paper_run_compiled_later_day_blocks`11442–11445/11478只执行完整块，公开ndays=339会改走tail，
不能冒充原路线。因此最短保持7块路线的方案为cold-prefix344、只记录337–339；原块内安全停止尚未实现。
不能拿day365终态或旧输入替代，也不为再次见年度失败跑全年；该新capture runner仍待实现/静态准入和主控决定。
[最小观测合同](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/next-observation-plan.json)
给出真实API/实参、integration.py1695–1714及内部局部锚、动态日号和原路线自准入；
未来预算提案仅1次本地CPU prefix344/过程硬限900秒、0额外baseline/warmup，不是现成捕获命令或本轮执行授权。
比较须绑定同一forcing/config/有效参数和独立Fortran参考身份；不同输入分支不判bug，同输入输出超原限
才构成需源码核对的候选。若要进一步证明浮点起源，还缺上游完整相同原生输入、直接源输出及误差来源事实，
不能由旧液水阈值观测直接下结论。年度定因还需按原单位/聚合核对该机制解释的当前差额及其余全年残差；
局部响应不自动关闭年度失败，本轮没有为此重算旧全年表。此方案只交主控裁定，未实现/执行新轨迹捕获，water-origin新捕获仍暂停。

原[npp-next-diagnostic方案](../../outputs/reviews/main-physics-20261027/npp-next-diagnostic/diagnostic-plan.json)
及其dry-run保留当时“未执行、组件0”历史身份；本次明确授权后另目录记录**恰好2个组件调用**。
一个科学子进程加监督进程，120秒硬上限内exit0；0baseline/warmup/整模型/新参考调用，
0显式JIT wrapper/整模型图，仅两调用固有JAX算子；Python-B/CPU/x64、持久缓存关闭，系统TEMP已清理。
没有生产物理/参数/容差修改、服务器/SSH/GPU/训练/付费/环境重建；本次主控仅发布两份报告。
图本轮可用，Verify代次仍`2026-10-26T20:35:28Z`，目标JAX元数据变化、Fortran排除；上述机制使用精确源回退，
不作全分支或图完备性声明。交付完成条件为真实两调用/六返回、谓词与机制边界、证据表/当前缺项、
有限下一方案及两报告绑定齐全；raw NetCDF重启容量与跨年连续性仍待后续独立验证，不扩669点/长期矩阵。
28项修复/16代码提交、42行/28ID有界修复表及C4–C28历史同步状态不变；两研究分支对齐和交接包继续暂停。
最终工件、两报告及工作区绑定见[本轮静态核验](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/static-verification.json)。

项目主控已核对21份新工件、真实两次调用与完整单坐标干预、分析实现和原限，复用执行方存储及身份检查，
没有重跑组件或统计。[最终验收](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/controller-acceptance.json)
接受局部分支放大及背景剩余的证据；不接受“已证浮点起因”“已解释当前年度差额”或新的物理缺陷结论。
执行方原344日/三日观测提案保持其历史身份。后续主控选择一次原365日路线，增加最后21日是为保存全年
NPP机制记录并核对已有day365完整末状态；每日记录集中存储，不按日散建文件。该任务另行派发，
最多一次本地CPU年度观测和两次当前输入小组件干预，保留原限；不扩展上游水文/Fortran/远端捕获。

## 当前轨迹NPP：365日原生机制与两次组件判别（2026-10-27）

本批已完成**1次整模型调用/365日及2次原NPP组件调用**，没有额外baseline或warmup。
新producer固定main `84e1d8d71dd2110f35d36b525881e70a74a68f61`，cwd及实际包均在
`E:/ORCHIDEE-MAN-JAX`；ORCJAX/JAX0.10.1、CPU/x64、Python-B、持久缓存关闭。
原V02的model-call、runner argv、15字段和全部开关原样复用，仍为319点1961、cold1+52×7；
没有重新物化run.def。配置SHA `3e3ec5c4518ebf4d470c60eda3e9bd6ddfd22e5c6dee473694f2409f17e619af`，
run.def SHA `5ecb4d2615dec4d54a88f2db5444ac2820d8327ec6c10c6af16945b4ff56374a`。
输入与实际新调用身份见[preflight](../../outputs/reviews/main-physics-20261027/npp-current-capture/preflight.json)及
[actual-model-call](../../outputs/reviews/main-physics-20261027/npp-current-capture/actual-model-call.json)；
新worker PID57056，不能沿用原V02 PID53212。完整原始返回先保存，再作独立准入。
模型调用及保存245.062秒，年度监督总261.016796秒，低于900秒硬限；
另一个组件worker PID12200在独立120秒硬限内完成两调用，监督总3.257107秒。
这些是本批完整执行耗时，不是稳态吞吐。所有本批系统TEMP已清理。

### 原轨迹准入与观测边界

沿用`atol1e-8/rtol1e-10`，容器/键/shape/dtype/离散与来源合同精确；数值载体Python类不要求相同。
[独立准入](../../outputs/reviews/main-physics-20261027/npp-current-capture/annual-admission.json)
核验新旧完整对象及各自定义域，数值与独立验证失败均为0。

| 与保存V02比较的范围 | 数值叶 / 元素 | 最大绝对差 |
|---|---:|---:|
| 完整年度返回 | 12401 / 178392 | 2.7755575615628914e-17 |
| 全365日完整daily对象 | 12045 / 154395 | 2.7755575615628914e-17 |
| day365完整末packet | 353 / 23994 | 0 |

最大绝对差为Day312 NPP[0,13]，最大相对差7.332580063418541e-16；这是原限内轨迹一致，
不能称观测对全部内部浮点计算无扰动。两处命名roughheight_pft[0,0]源未定义NaN合同保留，
其余定义坐标仍须finite。260字段独立源码分类及92额外字段独立覆盖缺口不变。

本批只在内存装载由当前精确源码生成的观察overlay；逆变换后算术AST精确相等，生产文件未修改。
NPP源锚为`stomate_npp.f90::npp_calc`314–387/449–531，JAX核心3954–4057、
`stomate/integration.py`1695–1714；保存的是allocation之后、后续post-NPP链之前的真实九参、
分支/pump/reserve、pre-floor/negative/bm_create、原floor-rate操作数及六返回。
floor-rate直接lift原`sum(bm_create)/dt_days`，不是从维护呼吸倒推。
365条记录集中在一个[native-records.pkl](../../outputs/reviews/main-physics-20261027/npp-current-capture/native-records.pkl)，
准确唯一顺序1..365，完整shape/dtype/finite及分支互斥通过[native准入](../../outputs/reviews/main-physics-20261027/npp-current-capture/native-record-admission.json)。

动态日号从原outer scan的science_day_numbers传入model_day_number，再作为debug.callback显式操作数；
year1961，日末零基tstep=day×48−1。原later-block路线自身选择outer_compiled_static_dispatch，
直接调用非独立inner-jit的_daily_carbon_static_dispatch；请求的use_static_jit_daily_carbon=True未改。
cold day1是独立执行，不把其静态日号1复用到后续日。原executable九家族实参及对象不经代理；
逐块调用前纯观察hook取得实际end-Day330/tstep15839和end-Day337/tstep16175的完整packet、
九输入家族、context/domain/calibration及动态pcapa_en，不能用只触发首块的factory hook冒充。
[边界包索引](../../outputs/reviews/main-physics-20261027/npp-current-capture/boundaries.json)保存两处精确存储回读；
day365完整packet在annual-run.pkl中。它们是新producer实际边界，不是旧V02缺失状态的补造。

### 当前Day338局部分支放大

当前实际GPP4.564204140244641e-7>0，原生分支为over_tax，bm_tax_max3.6513633121957125e-7、
bm_pump0.31553499951074226、reserve_no_alloc_delta0。原生负库存为空、floor-rate=0；
pre-floor最小值9.998478064767094e-9仍为正，源码只修正负库存，不能因其低于min_stomate判bug。
完整不变输入控制先通过全部六返回/546元素比较，maxabs1.9852334701272664e-23；
随后第二次调用只将gpp[0,13]置0。[冻结核验](../../outputs/reviews/main-physics-20261027/npp-current-capture/local-verification.json)
确认另八参及其余GPP坐标逐值/shape/dtype不变。

| PFT14日速率，g C m⁻² day⁻¹ | 原生当前Day338 | GPP单坐标置0后的原组件返回 |
|---|---:|---:|
| GPP | 4.564204140244641e-7 | 0 |
| 维护呼吸 | 0.3155353646470735 | −2.7977620220553945e-13 |
| 生长呼吸 | 2.5559543185369993e-8 | 0 |
| NPP | −0.3155349337862027 | 2.7977620220553945e-13 |

NPP有符号变化+0.31553493378648245，维护变化−0.31553536464735327。
这证明当前完整背景下的局部放大，超出此前仅历史Stage8输入的证据域；不等于全年干预效应。
反事实仅直接保存六返回，内部floor未捕获，不能把原生floor0移作反事实floor实测。
完整输入/返回见day338-control.pkl和day338-gpp-zero.pkl，控制原限及全部坐标见
[组件准入](../../outputs/reviews/main-physics-20261027/npp-current-capture/component-control-admission.json)。

### 全年原生恒等会计与输出差额

全年PFT14分支为enough302日、over_tax51日、no_alloc12日；floor共11日且均为no_alloc，
另一no_alloc为cold day1、维护0。原生恒等式为
`NPP = GPP − growth − masked_pre_maintenance + actual_original_floor_operand`，
全年最大残差2.220446049250313e-16；原生NPP对实际daily输出逐日最大差0。
原年度指标是365个**日速率的算术平均**，单位g C m⁻² day⁻¹，不是年通量sum或kg/year。

| 原生项 | 365个日速率之和 | 年度日均 |
|---|---:|---:|
| GPP | 283.831436470382 | 0.7776203738914572 |
| masked pre-maintenance | 87.35472959313623 | 0.23932802628256497 |
| 原floor修正 | 3.4274549371626826 | 0.009390287499075842 |
| 最终维护呼吸 | 83.9272746559735 | 0.22993773878348914 |
| 生长呼吸 | 59.01300012688871 | 0.16167945240243484 |
| NPP | 140.8911616875197 | 0.38600318270553324 |

详细原操作数与逐日记录见[机制会计](../../outputs/reviews/main-physics-20261027/npp-current-capture/mechanism-accounting.json)。
固定这些已观测操作数扣除floor项，日均为0.37661289520645747；这只是代数扣项，
不是关闭floor重新积分的模型结果。没有全年配对Fortran原生floor/分支记录，
所以0.009390287499075842不是floor跨模型误差、年度解释份额或可修复量。

另用新年度summary和V02既有、身份绑定的参考汇总完成**输出层有符号差额**，不重算旧逐日表。
取Δ=当前JAX−参考，原日均单位不变：

| 输出项 | 有符号Δ |
|---|---:|
| GPP | −0.00003886349333881434 |
| 维护呼吸 | −0.0008469903396264022 |
| 生长呼吸 | −0.000010082380928505552 |
| 两呼吸合计（直接） | −0.0008570727205549078 |
| NPP | +0.000818209227216038 |

`ΔNPP = ΔGPP − Δmaintenance − Δgrowth`残差−5.551115123125783e-17；
参考输出恒等残差0，当前−5.551115123125783e-17。故年度NPP偏高在报告通量层对应较低的维护/生长呼吸，
部分由较低GPP抵消；这已完成输出会计，仍未把差额归到native floor、上游浮点起因或某项修复。
参考适用范围、各项原值/差额及残差见[annual-output-accounting.json](../../outputs/reviews/main-physics-20261027/npp-current-capture/annual-output-accounting.json)。

新四指标与V02汇总值相同：AGB4.591235866253038、BGB4.120666562048175、GPP0.7776203738914573
通过原年度限；NPP0.3860031827055333对参考0.38518497347831726，
绝对误差0.000818209227216038>原限0.0003951849734783173，**年度科学失败及严格诊断false保留**。
本批没有证明上游GPP/状态差异来自舍入，也没有新增同输入源级NPP反例；不预设有bug或无bug。
若后续继续判别，已保存end-Day337是可复用实际边界，尚需独立Fortran producer的同phase/单位/足够精度
上游输入与输出；历史tuple、水文阈值及旧48步记录不自动成为当前起因证据。
见[剩余最小事实](../../outputs/reviews/main-physics-20261027/npp-current-capture/next-minimal-facts.json)；本批科学预算已用尽，无后续调用。

工具首次静态dry-run的annotation表示差异失败及修复后通过均保留，两次均0科学调用。
年度运行期间只加强未调用的offline admission，annual函数与物理observer未变；pre-component全脚本快照
明确按该阶段命名，不伪称完整launch快照。组件前监督器补TEMP绝对父目录/prefix检查，
旧回执不回写；细节见[工具版本说明](../../outputs/reviews/main-physics-20261027/npp-current-capture/tool-version-note.json)。
Verify图代次2026-10-26T20:35:28Z/full/complete/best_effort，三JAX路径metadata_changed，
Fortran及outputs排除；沿用精确源码回退，不作图完备/全分支声明。
未修改生产物理、参数、容差；无新Fortran、SSH/远端/GPU/训练/付费或环境重建，执行方未commit/push。
28修复/16代码提交及42行/28ID修复表不变，无新Stage27或修复ID。
原raw NetCDF重启容量、跨年连续性、TOPMODEL/fwet/支持域限制不变；water-origin新捕获、
两研究分支对齐及交接包继续暂停。本批工件/两报告/固定HEAD/工作区与修复登记绑定见
[最终静态核验](../../outputs/reviews/main-physics-20261027/npp-current-capture/final-verification.json)，保留交付时的待主控状态。

2026-10-27项目主控已集中审阅本批实际观测、单坐标控制、年度核算、工具版本说明与报告差异，
核对64份清单工件及交付报告SHA，复用执行方的准入/存储/输入/源码验证，未重跑模型、组件或统计。
[本批主控验收](../../outputs/reviews/main-physics-20261027/npp-current-capture/controller-acceptance.json)
接受当前背景的局部分支放大、观测轨迹原限内准入、全年原生机制核算与输出通量差额闭合。
该接受不证明上游浮点起因、跨模型floor差额或年度因果归属；原年度科学失败保留，没有新增生产修复。
本轮版本化发布仅含两份报告，原64份清单及执行回执保持冻结。

## 上一批Day338 GPP上游定位：生产正值已定位，完整捕获缺一项（2026-10-27）

本批固定`E:/ORCHIDEE-MAN-JAX`、main `7acead83618e64483095e69621f11e70137a9658`，
实际包为该checkout的`jax_orchidee/__init__.py`；ORCJAX/JAX0.10.1 CPU/x64、Python-B、持久缓存关闭。
复用上批真实end-Day337/tstep16175的完整packet、动态pcapa_en、context及原executable九家族，
仅调用一次原`_paper_compiled_later_day_block_executable`生成的compiled7块，完成Day338–344。
原factory内部静态placeholder保留；真实science_day_numbers338..344显式进入观察clock，
跨原47步scan及DIFFUCO独立JIT边界，未取消原JIT或改用独立科学计算。
两次prepare均0科学调用；唯一科学worker PID65264，原调用及保存114.391秒，
监督总118.074427秒/900秒硬限、exit0，系统TEMP已清理。0年度/cold/额外组件/baseline/warmup/重试。
执行身份见[preflight](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/preflight.json)、
[block-call](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/block-call.json)及
[运行回执](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/supervisor-worker.json)。

### 原轨迹比较通过，捕获完整性失败

完整原始块返回先保存，再准入；`raw-block-return.pkl` SHA256为
`07a8bce50f7901e064307baa1f814e40910139450f919eb799f3916d9bd9b47f`。
实际保存七日完整daily、七日NPP完整九参/locals/六返回、Day338的Yin/canopy/C3/entry各48条共192条，
以及新producer的day344末packet。没有旧day344末packet，不声称其完成新旧数值比较。
两处roughheight源未定义NaN合同保留，其余定义域finite。

| 准入项目 | 实际结果 |
|---|---|
| 七日完整daily：231数值叶/2961元素 | 原atol1e-8/rtol1e-10通过，maxabs2.17010222251378e-15，0数值失败 |
| 七日NPP九参/locals/六返回：196叶/13251元素 | 同原限通过，maxabs同上，0数值失败 |
| 容器/键/shape/dtype/离散/来源 | 精确通过；数值载体Python类不要求相同 |
| 已保存NPP的enough/over_tax/no_alloc/negative | 七日均精确相同 |
| 完整mandatory capture | **失败**，缺唯一键`('accumulator',338,0)`，无其他unexpected keys |

实际`PEAT_OCCUR=FALSE`、`use_compiled_runtime=True`、`use_numpy_runtime=False`。
累加观察器挂在仅peat guard成立才调用的`_stomate_daily_accumulation_fold_runtime_peat_static_jit`，
因此没有运行；实际使用`daily.py:1432–1499`的generic ordered loop，位于原outer compiled7图内。
该观察点选择错误导致原生逐步gpp_d及累加前后值均未保存；初次离线准入的KeyError、原工具快照及版本说明保留，
后续只修离线分析，不重试科学块。[admission](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/admission.json)
明确`passed=false`、`mandatory_capture_complete=false`，同时单列可用轨迹数值/保存NPP分支通过。
新亚日内部支路此前未保存，不声称它们与上批逐步支路相同。

Day338/339/340/341/344的tiny GPP与上批逐值相同、0ULP，零/正分支保持。
Day342新1.5116582107170082e-6对旧1.5116582128871104e-6，绝对差2.17010222251378e-15、
相对差1.4355773044550261e-9、10,248,018ULP；Day343新1.9396509446861947e-6对旧1.9396509450671202e-6，
绝对差3.809255394568358e-16、相对差1.9638870613581155e-10、899,435ULP。
故七日并非bitwise相同；观察/重建可能影响fusion，原限内数值通过及关键保存NPP支路精确不变仅支持上述有界准入。

### Day338唯一实际正GPP的生产链

Day338区间为零基tstep16176..16223。48步中仅offset5（tstep16181）实际assimilate为true、GPP为正；
其余47步humrel为0、实际光合活动为false、生产GPP为0。对应Fortran一基kjit标签16182来自时钟惯例映射，
并非本批Fortran实测。完整65项C3直接输入和原生内部记录见
[production-chain](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/production-chain.json)及subdaily.pkl。

| offset5实际操作数/输出 | 数值及边界 |
|---|---|
| humrel及活动阈值 | 3.6402170609775793e-6 > 1e-8；距离3.6302170609775794e-6 |
| swdown | 888.8902888737543 > 1e-8 |
| LAI / veget / veget_max | 1.2954020484751498 > 0.01 / 0.7262122337533891 / 1.0 |
| temp_growth | 19.24361600534081，在(-4,55)内；全部活动门为true |
| 原生canopy assimtot | 7.028563100383167e-5 |
| salinity / inundation控制 | 0.5 / 0.6012772289745216 |
| 原生controlled_assimtot | 2.1130574723354817e-5 |
| 原生mesh单步GPP / entry GPP / 日NPP入口GPP | 均4.564204140244641e-7；日NPP入口与上批0ULP |

Yin向量化求解器20层可产生raw正候选；实际8个active叶层参与canopy积分，active及raw fallback均0。
inactive步或叶层的raw候选不能计作更早生产正GPP。该步humrel并非仅差1ULP贴活动阈值，
但仍不能排除更早扰动经历史/分支放大；本批没有观察humrel上游创建过程。
源锚为`diffuco.py`活动2296–2372、Yin2648–2783、canopy2786–2949、输出2952–3100、C3包装3103–3349；
Fortran为`src_sechiba/diffuco.f90::diffuco_trans_co2`活动2338–2384及GPP换算2884–2899。
原式`where(assimilate, controlled_assimtot * 12e-6 * veget_max * dt_sechiba, 0)`已含1800秒步长，
单步单位gC m⁻²(mesh) timestep⁻¹；日NPP入口使用归一化PFT日速率，分析没有再乘一次步长。

### 仅派生的日累计会计与独立参考缺项

保存的真实end337初始gpp_daily=0、48条entry及veget_cov_max=1，按源码`daily_inputs.py:171–220`和
`daily.py:1432–1499`离线代数推导：offset5的gpp_d=2.1908179873174275e-5，
乘dt_sechiba后中间量0.0394347237717137，最终除dt_stomate86400得到4.564204140244641e-7，
与实测NPP入口相同。数据独立保存为`derived-accumulator-accounting.pkl`，逐行标明DERIVED。
这支持所存操作数与源代数的一致；**没有原生累加器逐步观测，不能称原生累加链完整闭合**。
Fortran归一化来源为`stomate.f90::stomate_main`2921–2931/3020–3032/3207–3208，
累加来源为`stomate_accu_r1d/r2d/r3d`9341–9411。实际路线及缺口见
[累加源码说明](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/accumulator-source-note.md)。

[命名历史证据表](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/historical-evidence.md)
核对Stage8–13：Stage8 Day338为下游NPP，旧JAX GPP1.2241381168386465e-6并非当前值；
Stage10完整65参DIFFUCO和Stage12水文carry主要是Day337，不能移作当前Day338配对参考。
不同轨迹的状态差只约束边界，不形成同输入翻译反例。当前绑定的
`reference/OUT/orc_calibrate_250919_sen/arg2_1.0/319.0-057.0/I5/S26_69.834_0.0019_0.3980_97.918/stomate_history_1961.nc`
只有`time_counter=1`的float32年度聚合；[实际锚核验](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/reference-anchor.json)
没有Day338日值，不能据其宣称当前参考当天GPP为0。

最小独立Fortran方案限定Day338 kjit16177..16224、完整65输入及activity/叶层/controls/GPP/累加/交付，
使用float64或17位roundtrip及精确离散记录，并绑定preceding end337/prefix、source/binary、有效配置与输入。
历史入口和非执行命令模板已列入历史证据表；旧qhcess ifort/IntelMPI及用时不能直接当作Explore1000事实。
Explore1000新工作目录须在`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/`，编译器/arch/MPI/XIOS/NetCDF、
binary/overlay、输入部署、prefix状态、节点/资源/时限/费用均尚未绑定，也没有已准入的Fortran end337短重启。
详见[下一最小事实](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/next-minimal-facts.json)。
此为具体缺项及未执行方案，不构成后续运行授权；本批预算已用完。

本批可以确认当前正值在光合生产层及entry已经存在；不能认定翻译错误、浮点起因、有bug/无bug或年度因果。
原年度NPP0.3860031827055333对参考0.38518497347831726，误差0.000818209227216038>
原限0.0003951849734783173，科学失败不变；上批局部分支放大/floor0及年度会计保持其已接受身份。
图Verify代次2026-10-26T20:35:28Z/full/complete/best_effort已确认；目标JAX路径metadata_changed，
Fortran和outputs排除，使用相关精确源/工件回退，不作全分支或图完备声明。
无生产物理/参数/容差修改，无新Fortran、SSH/远端、GPU/训练/付费、环境重建、commit/push、自动化或主控聊天消息。
28项修复/16代码提交、42行/28ID修复表不变，无新Stage或修复ID；raw NetCDF重启、跨年连续性及原支持域限制保留。
本批未启动water-origin新捕获，研究分支对齐和交接包继续暂停。交付时实际工件与两报告绑定见
[最终静态核验](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/final-verification.json)；
上批64份manifest和历史待验收/主控验收回执均未回写。

2026-10-27项目主控已审阅实际生产链、运行/准入、累加观察缺口、历史参考边界及两报告差异，
核对58份清单工件与交付报告SHA，复用执行方验证，零模型/组件/统计重跑。
[独立主控验收](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/controller-acceptance.json)
接受当前Day338正GPP已在DIFFUCO生产层出现、唯一活动步及其humrel直接输入、有界七日轨迹证据；
完整捕获失败和无原生逐步累加器记录保持原状，不以派生会计补称完成。
后续优先追查humrel创建及独立同phase参考，不为单独补累加日志重跑全年；必要新捕获中修正实际路线观察点。
上游浮点/实现起因、年度因果及原科学失败不变。本次仅发布两份报告，无新增生产修复。

## 当前humrel起源与独立Fortran配对：水分胁迫域通过，完整参考有缺项（2026-10-28）

本批固定`E:/ORCHIDEE-MAN-JAX`、main `33ab88d8c636262033cd921bd58c670e8e281f7f`，
实际导入为该checkout的`jax_orchidee/__init__.py`；ORCJAX/JAX0.10.1 CPU/x64、Python-B、持久缓存关闭。
复用原真实end337/tstep16175 packet、动态pcapa_en、context及原executable九家族，
唯一原compiled7调用完成Day338–344；无本地annual/cold/baseline/warmup或科学重试。
worker PID28440，调用及保存168.719秒，监督172.67159175872803秒/900秒硬限，exit0。
四次prepare均0科学，系统TEMP已清。来源与调用见
[preflight](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/preflight.json)及
[执行回执](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/supervisor-worker.json)。

### 完整本地观测与保留的全raw失败

原始块返回先保存，再准入。15份观察overlay逆变换后的算术/控制AST精确相同；
真实day/step跨原DIFFUCO、HYDROL独立JIT传入，原静态参数与placeholder保留。
callback逐条追加到单个`native-records-journal.pkl`，effect barrier后一次合并，并精确核对journal/内存/保存值。
960条为20类×48步：Yin/canopy/C3/entry、完整水文diagnostic creator/aggregate/carry/generic accumulator，
及六tile的layer/stress完整入参与返回；另保存七日NPP九参/locals/六返回和新day344 packet。
完整creator为`hydrol_module_diagnostics`51参，六tile按源顺序携带更新的根权重；layer/stress/aggregate为18/17/4参。
动态根记录覆盖全部14 PFT候选、allowed/depth、denom/positive/safe、前后根权重，原生亚日记录无nonfinite。
见[捕获schema](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/capture-schema.json)、
[overlay审计](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/overlay-audit.json)和
[紧凑结果](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/local-summary.json)。

| 独立准入范围 | 实际结果 |
|---|---|
| mandatory本地捕获 | **完整**，20×48及七日NPP均在；补齐实际generic ordered accumulator |
| 完整七日daily/NPP vs原年度保存值 | 原atol1e-8/rtol1e-10通过；231叶/2961元素及196叶/13251元素，maxabs7.587966781058726e-16 |
| 完整daily/NPP vs上一批compiled7 | 原限通过，maxabs1.4113055444079075e-15；四类NPP保存分支七日精确不变 |
| 全192条既有raw亚日候选比较 | **FAIL，440项**；Yin160/canopy280，200离散候选及240浮点超原限 |
| 实际生产域 | 48步bool活动门精确；完整defined C3返回、entry和活动offset5完整输入/locals/返回maxabs0 |

`admission.json`总`passed=false`不改；全部440失败位于offset28/30/37/40/43，均`assimilate=false`且无active叶层，
最大raw差376.1873200765633。首比较器还对连续activity字段要求了额外bitwise相同，
该额外条件与实际bool门区分；全raw的独立失败仍成立。原限、shape/dtype/键/离散/来源精确合同和
两处命名roughheight源未定义NaN合同保留，Python数值载体类不额外要求相同。
见[原准入](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/admission.json)与
[失败分类](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/admission-classification.json)。

Day338 GPP仍为4.564204140244641e-7，与原年度及上批0ULP。Day340/342/343对原年度tiny GPP
分别差3.0603639197476773e-16、7.587966781058726e-16、4.2905988075045594e-16，
ULP距离90,326/3,583,316/1,013,089；其余四日精确。观察/重建可能影响fusion，不称七日bitwise相同。
新day344 packet真实保存，旧数值包不存在，不伪造末状态新旧比较。

### 原生writer → carry → reader与层水量来源

Day338零基tstep16176..16223。唯一正writer为offset4/tstep16180，经真实next_state到
offset5/tstep16181 DIFFUCO reader；Fortran实际对应kjit16181→16182。offset5自身HYDROL在其DIFFUCO之后，
供下一步读取；offset0读取真实end337。HYDROL transpiration `humrel`与传入STOMATE、名为humrel的`vegstress`分开。
本地原生carry链精确，见
[phase链](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/water-phase-chain.json)与
[源码契约](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/source-contract.md)。
源锚为`orchestration.py`2954起/5816–5819/6188–6435/10758–10797，
Fortran `sechiba.f90::sechiba_main`997/1049–1055/1184–1188；水分胁迫为
`hydrol.py`7043–7204和`hydrol.f90::hydrol_soil`6473–6717，聚合为7207–7224和9089–9107。

| offset4，tile4/PFT14/第6层 | 当前JAX原生值 | 独立Fortran原生值 |
|---|---:|---:|
| 层水量sm | 3.0827056142277756 | 3.079323123859772 |
| smw_tmp / smf_tmp | 3.0826979486351997 / 5.7149560143648 | 相同 |
| pcent / NEW_WATSTRESS / DYN_NROOT_LARIX | 0.8 / false / true | 相同 |
| PFT14根权重，进入/退出 | 仅第6层为1 | 相同 |
| stress factor / 动态根候选和 / humrelv | 3.6402170609775793e-6 | 0 |
| 下一步DIFFUCO读取humrel | 3.6402170609775793e-6 | 0 |

独立Fortran在Day338全部48步的writer/reader humrel和生产GPP均为0；
本地仅writer4/reader5为正，时钟、相位及实际读取均已配对。
presence为1、under-MCR为false；本地浅层虽有较大factor，根support为0而不贡献。
动态根本地正denom归一化后仍为原one-hot；Fortran零denom沿源分支保留原根。
humrel为非负的tile humrelv之和，不再乘tilefrac。本地tile4 vegstressv为2.9121736487820634e-6。
第6层本地sm在offset3/4/5为3.082652074939157/3.0827056142277756/3.0772549010334744；
对共同wilt阈值的差为−4.587369604269753e-5/+7.665592575811786e-6/−0.005443047601725315。
本步正margin远大于单个ULP，但更早扰动或历史分支放大尚未排除。

[层水量会计](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/layer6-source-accounting.json)
严格分列native与保存操作数派生值：零基节点4/5/6按原分组
`dz[5]*(3*mcl[5]+mcl[4])/8 + dz[6]*(3*mcl[5]+mcl[6])/8`生成第6层sm，当前原生与派生残差0。
实际post-solver MC/MCL等于下一步carry；before记录是前entry状态，不冒充内部solver的精确中间入参。
该会计未调用或重建土壤solver；根support在所观测三步之前已收缩到该层。

[跨侧保存值会计](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/cross-state-analysis.json)
进一步分解offset4的原生sm差0.0033824903680037544。两侧DZ完全相同，贡献如下；
权重与贡献均为保存操作数的源代数派生，不是新组件调用或原生中间量。

| 零基节点（Fortran一基） | JAX MCL | Fortran MCL | 原积分权重 | 派生sm差贡献（JAX−F） |
|---|---:|---:|---:|---:|
| 4（5） | 0.07310990413372333 | 0.07236800719033043 | 3.910068428 | 0.0029008678151902713 |
| 5（6） | 0.06503245070461967 | 0.06501876459708353 | 35.19061585199999 | 0.0004816225528136591 |
| 6（7） | 0.065 | 0.065 | 7.820136855999999 | 0 |

贡献和0.0033824903680039304，对原生sm差残差1.7607443281164592e-16。
Fortran本步wilt margin为−0.0033748247754279426，与本地正margin分别决定源factor/denom的零/正。
真实end337时tile4 MC/MCL已不同，最大绝对差0.0016603309705315594，PFT14根仍精确相同；
因此水状态差先于本地Day338块，尚不能把它的首次形成归给offset4当前soil update。

本地offset5原生光合生产/entry/NPP入口仍为4.564204140244641e-7。
此次观察器挂到实际`PEAT_OCCUR=FALSE`的generic ordered loop，48步原生gpp_d、累加前后、
do_slow/日末除法和送入NPP逐值闭合；最终native accumulator与NPP入口精确相同。
见[原生累加记录](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/native-accumulator.json)。
上一批缺失的累加记录及仅derived会计保持原身份，没有回写成完整。

### 独立Fortran执行、位域准入与组件判别

唯一新producer为`/public/share/qhcess/qhcess2/User/jcwu/orchidee_jax_319_humrel_current_20261027`，
从冻结Stage12 producer隔离必要源和全部可写构建/运行文件；只读复用既有XIOS/环境，不动zzhao。
无完整end337 restart，故该唯一必要轨迹沿原cold365 prefix与调度。
job125893716，a14r1n19，1节点/8CPU/16GiB、最大7200秒；实际allocation500秒，
4000 allocated CPU-seconds=1.1111111111111112 CPU-hours，RSS约3.173GiB。
编译376秒、原轨迹93秒，模型exit0并到达`END of dim2_driver`；事后检查不存在的旧名
`matched_send_capture.txt`使wrapper exit4/调度FAILED，原失败和archive错误保留。
仅取回现有文件并离线核验，无第二allocation/轨迹。
提交前idle/load2.23、实际启动load17.38及CPU affinity0–7均记录；不以快照解释历史性能。
allocated CPU-hours与未取得的实际CPU消费、货币费用区分。见
[来源回执](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/fortran/source-receipt-125893716.json)及
[资源](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/fortran/resources.json)。

32项producer检查通过，绑定五份实际预处理源、实际worker/binary前后、输入/新源/冻结旧文件前后；
run.def和used_run.def与冻结版本逐字节相同。旧producer的DIFFUCO使用`/den_man`和`weight_ven_a/b`，
仓库对应为`/1000`和`.5/.5`；本次实际used_run及numout确认1000/.5/.5，48个firstcall_diffuco均false。
这些证据只关闭本case公式/操作数和first-call条件，不建立全配置或源码字节等价。
原source-only审计当时pending的身份保留，当前运行条件见
[Fortran摘要](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/fortran/admission-summary.json)。

Fortran记录覆盖end337及Day338共49边界、六tile、14 PFT根分支、层/胁迫/聚合及下一步读取。
226257项检查中，**784记录/980标量的decimal/native负零往返失败保留**；原文把native −0打印为+0。
独立native-bit域以直接`TRANSFER`位记录解码，经一一键/shape/64bit配对和相关定义域检查通过，
不重算物理、不改文本或容差。原总数据状态仍failed_preserved；该域也不是完整51参creator或soil solver。
全年四通量各365×14个float32存储值对已接受Stage12精确相同，maxabs0；
Day338/PFT14存储为GPP0、NPP9.99999993922529e-9、MAINT_RESP−9.99999993922529e-9、GROWTH_RESP0。
这些是独立Fortran自身轨迹保持和实际日存储，不代替Day338原生NPP入口：后者九字段guard仍Day337。
C3虽有65实际输入，完整10最终输出缺证据，整对control/cross跳过。见
[Fortran入口与缺项](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/fortran/README.md)。

水域相关capture/来源/hidden/static依赖独立准入后，唯一组件worker PID3924先作不变JAX tuple控制，
通过后再接实际Fortran tuple；实际2调用/1.281秒，监督4.872981071472168秒/总300秒硬限，TEMP已清。
无warmup/重试，C3调用0。所有入参/返回/记录集中保存，
[域准入](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/component-domain-admission.json)与
[实际组件结果](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/component-admission.json)分开保留全raw=false。

| 原单tile hydrol_water_stress_diagnostics | 原限结果 |
|---|---|
| 不变JAX tuple：完整原生记录214叶/4109元素及完整6返回/348元素 | PASS，maxabs0；bool分支、零/正精确 |
| 实际Fortran17参：完整6字段返回/348元素 | PASS，maxabs0，0失败 |
| 跨侧相关分支/locals：101叶/814元素 | PASS，maxabs0，0失败；包含全部14 PFT动态根分支 |

Fortran的under-MCR增量是保留native前后计数之差的派生接口映射；部分allowed/depth/positive谓词
由保存native操作数按原IF组合，均有逐字段provenance，不冒称Fortran命名native变量。
同一JAX原组件在当前tuple给正值，在实际Fortran tuple给0并匹配其完整映射返回；
所测单tile胁迫域未复现翻译差。此前已存在的sm差足以改变本步factor/denom与光合活动门，
不据此证明六tile完整creator、内部soil solver、C3全返回、所有分支或全年起因。

原年度NPP仍0.3860031827055333对参考0.38518497347831726，误差0.000818209227216038>
原限0.0003951849734783173；上批局部分支放大/floor0/年度会计的有限接受保持独立。
更早水状态及根support形成未归因；下一最小事实限定为形成上述end337 tile4水状态的更早实际soil-update
完整入参与原生输出对应。现有post-solver层值及前entry状态不能补称该边界已捕获，不在本批追加轨迹。
当前图服务list_projects/index_status/search/coverage尝试为Transport closed；历史Verify代次
2026-10-26T20:35:28Z仅作历史身份，使用精确源/工件回退，不作当前图完备或全分支声明。
本批无生产物理/参数/阈值/容差修改，无研究工作树访问/修改、GPU、训练、环境重建、commit/push、
自动化或主控聊天消息。28修复/16代码提交、42行/28ID登记及原支持域不变，研究分支对齐/交接仍暂停。
交付时待主控状态保留；旧58/64份manifest和接受/发布回执冻结，交付工件/两报告绑定见
[静态核验](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/final-verification.json)。

### 主控验收与本轮收尾（2026-10-28）

主控已核对236份清单工件和交付报告SHA，审阅实际水分创建/传递/读取、分层操作数、原组件执行与域准入、
Fortran身份/位记录/缺项及两份报告，复用执行方检查，零模型/组件/统计重跑。见
[独立主控验收](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/controller-acceptance.json)。
接受范围是当前Day338的局部水状态跨阈值机制，以及所测单tile胁迫组件在两组完整输入上的精确复现。
本地440项raw失败、Fortran文本/wrapper失败、NPP338和完整C3缺项均保留，不把有限域通过记成整批全通过。

本轮已回答“微小输出差是否能由不同状态触发分支放大”：在该日有直接证据；尚未回答“最初状态差是否纯属浮点，
以及全年差额是否全由它造成”。后两者登记为已知未解决科学差异，原年度NPP失败保持，不构成新增生产修改依据。
**本轮专项诊断收尾，不自动追加更早状态追踪、模型运行或Fortran捕获。**
若以后需要定位首次土壤状态分歧，应另定可判别问题、完整对应soil-update输入/输出证据及资源预算，
不能因前史尚未查清就无限延长本轮。本轮收尾不表示主分支重启、跨年或全部科学验证已通过。
本次版本化发布仅含两份报告，28项修复/16个代码提交不变，研究分支对齐继续暂停。
