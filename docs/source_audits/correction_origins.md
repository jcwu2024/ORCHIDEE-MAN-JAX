# C1–C61修正与支持补齐来源登记（2026-11-18）


## 当前接受：C60／C61与参数状态收尾

2026-11-18，基线`a3551dac4dbb2f53bee2ef47c80b4995eb57af82`上的WATCHOUT原包、R1–R5修正与R2公开历史续包合并接受，代码／测试／CI提交`aa14d04210ab371aae6e959da193561f37f55d0e`。正式累计61项修正／支持补齐、34个相关生产提交；59项有复查前错误或支持缺项证据（含C60），C42／C57为复查期间已接受代码的遗漏。不能把61解释成61条当时均激活的物理公式bug。

| 登记 | 必要性、年代与证据 | 修复和接受边界 |
| --- | --- | --- |
| C60：WATCHOUT-TRANSPORT-01 | 完整复查前`7cae3e02adad23674767a590fdf4d27f239ce065`与a355快照中高层forcing／cache缺字段、driver拒绝该模式，低层参考不能证明高层已接通；这是旧支持缺口，不虚称旧可执行路径曾产生数值失败。准确最初引入未知 | 文件schema／geometry／高度／cadence与有效控制→插值／bundle→DIM2／intersurf→DIFFUCO／ENERBIL→compiled动态tuple和source条件driver9／13读写。actual first-substep初始化、once-only SWnet、static drag及standard／cold／native公开context交接一体接受；RELAXATION原fatal保留 |
| C61：FORCING-FIRST-READ-01 | 7cae `forcing_interpolation.py:264`与a355都有Python `round(86400/dt_force)`；原`readdim2.f90:1435–1452`要求NINT及完整首日。19200秒得到4.5时旧选record4、原源选5；四条20000秒虽index合法却不足源完整首日。新cadence接线暴露旧表达式边界，首次引入未知 | 正数NINT半值远离零，源完整首日guard恢复；实际public bundle及file输入验证。原插值公式、参考与容差保持；默认长度／TIME_LENGTH的新接线归C60，不按调用点增加C号 |

来源：[原运输缺口](../../outputs/reviews/main-physics-20261118/watchout-parameter-transport/origin-binding.json)、[R1–R5年代](../../outputs/reviews/main-physics-20261118/watchout-parameter-transport-repair/origin-binding.json)、[R2交接](../../outputs/reviews/main-physics-20261118/watchout-context-handoff/origin-binding.json)。本地Fortran未被旧Git提交跟踪，不伪称恢复其历史producer。

C60开发期间发现的SWnet assembler、compiled首tuple、NETRAD／SPRED入口、writer变量／模式、raw预读误当初始化、局部context丢失、old count与new split混用、非WATCHOUT override实参、continuous缺历史fallback和reference漏返回，均是**本轮未接受支持实现中的集成遗漏**，合并在同一链并保留修前证据，不倒写为7cae已有，不追加独立旧bug。夹具、缓存身份和lint修正另外记录。

最终公开交接同输入16失败／19通过→35通过，另既有7项通过；之前99＋1与原113各保留当时版本。完整stage反例与终审保留，主控复核未重跑数值；提交时3测试仅移除空EOF，AST不变。见[最终验收](../../outputs/reviews/main-physics-20261118/parameter-state-controller-final/controller-acceptance.json)。参数／状态有限工作面已完成，完整科学接受仍false；未来两研究分支按共同commit及基线依赖集中核对，尚未同步。

## 已接受快照：C55／C43扩展与C56–C59参数状态交接

[交付包](../../outputs/reviews/main-physics-20261118/parameter-state-closeout/README.md)基线`db8dd72eea30e3c2dc30dd3b7368a664d5a56058`，共同代码／测试／CI提交`5fe049ffa249f2c53daf669d576fe590ecbbb999`。按概念修复登记为C55与C43扩展及四个新编号，不能把六候选组机械算成六个新bug。该包接受时累计59项修正／33个相关生产提交，其中57项有复查前证据，C42与C57为复查期间引入。原精确来源、失败和限制见`origin-binding.json`及[主控验收](../../outputs/reviews/main-physics-20261118/parameter-state-closeout/controller-acceptance.json)；没有执行旧整个模块或旧模型。

| 登记 | 年代及旧反例 | 修复与接受边界 |
| --- | --- | --- |
| C55扩展：HD-DEPTH-PARAM-01 | 旧初始化接受深度覆盖，runtime重建却回默认，bottom kfact的0.07243975703425146回到0.1；同一参数传递链的扩展，精确首次引入未知 | 七活动标量＋一inactive metadata独立合同，贯通表／缓存／系数／driver；十向量合同不变；保留源AKS_POWER检查nk_rel及注释分支 |
| C56：HD-REFSOC-LIFECYCLE-01 | 复查前7cae exporter强制false、IO未识别条件字段；db8的[42500.125] tag丢失或被拒；首次引入未知 | 条件读取／ALLmarker初始化、双镜像carry、finalize和真实小文件运输；新resolver不是历史已有API |
| C57：COLD-CLOCK-CONTEXT-01 | **复查期间**`782535b4130f2ddc8cd30a251e0493709e1ab26a`（2026-11-14）引入ready出口未定义context；原局部AST两分支NameError | 已有prepared identity保留，否则从cold bundle建clock；上游完整物理ready未验证，不称全部冷启动已通过 |
| C43扩展：SCAN-CONTEXT-01 | 7cae的scan cache key漏捕获context identity，真实窄JIT的A/B串值；原C43已修later-day block，本次扩到scan；精确首次引入未知 | identity入key并由closure持有；同形状动态参数仍正确复用，无整模型数值声明 |
| C58：BLOCK-DISPATCH-01 | 7cae已有state派生phenology mask而block key未含它；同context／layout、不同mask复用旧分支；首次引入未知 | immutable dispatch同时入key和closure快照；真实窄compiled block使用物理witness |
| C59：STATE-TEMP-GROWTH-01 | 7cae已漏half-hour carry；782535b4后来新增必需calendar consumer，两历史事实分开；实际原clock入口可复现丢值 | 中央carry补字段，源slowproc588的pre-STOMATE读取和stomate5049日更新顺序不改；不声称旧标准例当时已失败 |

独立终审另发现**本包新集成**的disabled初始化仍carry旧refSOC，已保留两条真实红反例并在文件首writer／new_native双镜像清除。此子问题归C56同一链，不单列新C号，也不倒写成7cae已有。受影响36节点最终按35通过＋一项夹具修正后定向通过；生产仅改两个初始化owner，连续路径和strict exporter不改。

四项IO run.def夹具依赖、LRU对象identity断言、NamedTuple误用dataclass属性及空HYDROL组件访问均是验证设施问题，原失败保留，不计作模型缺陷。主控仅另删测试末尾空行，执行AST不变，无数值复跑。当前94节点最终通过是组合证据，不是单次全绿或hosted CI。

WATCHOUT尚需本地source→driver运输补齐；原封存包将外部producer合同视为必须先解决的阻塞，由补充源审查修正。真实外部producer身份未知继续保留，不登记为已证实的模型bug。研究同步暂停，准确文件／依赖见[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)。

## C53–C55 已接受来源

2026-11-18主控接受C55=`HD-SOIL-PARAM-01`，[参数链包](../../outputs/reviews/main-physics-20261117/hydrol-soil-parameter-closure/README.md)代码／测试／CI提交`474f52088be77a543ebf181b232acfc8a02ddc02`。Fortran hydrol_init2069–2252设置同一组矿物参数，var_init4071–4088派生有效点阈值，soil6395–6741／diag_soil9089–9284持续消费它们。旧初始化可用自定义值，模板／wrapper／诊断却回到USDA默认；原case1／5的drysoil_frac和vegstress四失败归一项参数传递遗漏。五例eager／真实JIT各191／191通过，原输入／参考／容差保持。复查前7cae3e02已有相关函数与硬编码默认，首次引入未知、旧整模型未重跑；见[来源核查](../../outputs/reviews/main-physics-20261117/hydrol-soil-parameter-closure/controller-origin-check.json)与[主控验收](../../outputs/reviews/main-physics-20261117/hydrol-soil-parameter-closure/controller-acceptance.json)。本轮也补齐同一十向量的refSOC、缓存及实际调用支持，不能声称全部配置或论文自然轨迹原来失败。

2026-11-17主控接受[水文／冠层包](../../outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/README.md)的两项有限接口／执行修复，登记C53=`HD-FWET-01`、C54=`HD-JIT-01`，共同代码／测试／CI提交 `139e1ba7ba2db46d6f1a3b00f0592335bf40cafa`。C53按hydrol_soil:7307–7342保留当前TOPMODEL湿地诊断，仅在没有新值时沿用旧值；原ordinary／runtime／fast布局曾将0.412613023955767覆盖为0.123。C54按diffuco_trans_co2:2214–2234在显式参数缺光照表时提前调用既有静态布局helper，修复真实JIT层数转换异常。两项在`7cae3e02adad23674767a590fdf4d27f239ce065`已有精确源码证据，首次引入未知；新参考设施不伪称旧版已有。前者当前wrapper TOPMODEL=True仍未贯通，后者论文默认getter本来预供表，故不声称论文自然路径曾触发两项失败。主控复用独立原源、19项交接和10项隔离回归／Ruff证据，无数值重跑；见[验收](../../outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/controller-acceptance.json)及[发布](../../outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/controller-publication.json)。当时水文四项自定义参数差异另列；现由C55同一参数链闭合。

C55接受时累计55项修正、32个相关生产代码提交，其中54项有复查前证据、1项C42属复查新增helper的默认遗漏；当前累计以上方C1–C61登记为准。C53为状态选择、C54为有限编译入口修复，计数不等同活动物理公式错误数。原C1–C52来源和接受范围保留；C51／C52共同提交 `9a856c6cf859437d2436f4f3748d4c9e770838a1`。C49初次提交 `ea75bde31012cf519d4b0fa5c689d865c9d9c2b7` 与US_INIT扩展 `d6bad452932d1a7179ff5ab5fd80c771da89e336` 仍算同一初始化链。

C45–C47共同提交3f7b482765b7b41339ee56a818b1f164490a5d1e，具有配对源、保存旧反例和新identified native参考；不恢复历史001 producer。详见[累计登记](shared_core_branch_review.md#累计修改位置与提交索引)和[helper接受记录](../../outputs/reviews/main-physics-20261114/native-clock-helper/controller-acceptance.json)。旧C41–C43提交8a772de6f4977018a875093d535b6088317716c8、C44提交9c15c042a5ade9c91d91ac91a8c0ea39ed87c8f2保留。生产接线C48见下节，不并入这三个已接受helper修复。

## C51–C52：完整热组件及driver蒸发单位边界已接受

基线main `82d622722feea3cfe5034d8b15f98cda78e04872` 的[整包](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/README.md)分别保存两个复查前来源。主控于2026-11-17接受为C51、C52，共同代码／Oracle／fixture／CI提交 `9a856c6cf859437d2436f4f3748d4c9e770838a1`。见[主控验收](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/controller-acceptance.json)及[发布记录](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/controller-publication.json)。交付时pending回执保留其原始身份。

| 编号 | 复查前证据与最小修复 | 验证／边界 |
| --- | --- | --- |
| C51 thermal final coef裸地权重 | `7cae3e02`两个full-step final coef实参均为raw veget_max；源thermosoil_main1002–1009传veget_max_bg。仅两个实参改现有helper，C50 energy调用保持 | [来源](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/thermal/origin-binding.json)；修前52失败字段，修后原8参考eager／JIT552字段过；不同于C50的energy消费者，两个coef调用点合计一项 |
| C52 driver vevapp amount→rate | `7cae3e02`共享writer直接复制ENERBIL每步amount；源intersurf743–749除xrdt后才返DIM2。新增纯除法helper、实际dt元数据、共同writer一次转换 | [来源](../../outputs/reviews/main-physics-20261116/thermal-snow-coupling-batch/integration/origin-binding.json)；非零原源与实际writer／单点IO通过；SECHIBA evapora仍amount，peq外部producer gap不因此关闭 |

只执行保存旧writer的单一函数AST负控制，未import／执行整个旧模块。两项的精确首次引入提交均未知；当前新建参考基础设施和dt metadata接线不伪称旧版已有。积雪没有新确认翻译错，O0／完整JIT grain严格失败独立保留，不能用strict xfail增加通过数或新C号。主控将该xfail限制到已知case的有限grain差异，其他字段先正常断言；定向1项仍为xfail且Ruff通过，属于测试保护，不增加模型修正数。没有整模型运行、研究树同步或NPP／DOC改善结论。

## C50：thermal energy消费者的裸地权重已接受

2026-11-16，[独立thermal包](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/README.md)从原`thermosoil.f90:602–603,913,2442–2455`确认energy必须使用补足裸地的`veget_max_bg`。复查前`7cae3e02adad23674767a590fdf4d27f239ce065`及本包基线`551cc8cdbc4f16fa37ed2c89f260c36b57c596a8`的`thermosoil_explicit_step`、`thermosoil_no_explicit_snow_step`均传raw `veget_max`；kernel直接按入参加权，未做补足。已保存两个完整文件及实参AST，[来源绑定](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/origin-binding.json)证明复查前已存在，精确引入提交未调查；旧整个模块／模型未执行。

独立本地原Fortran稀疏案例`veget_max[0,[0,1,13]]=[.05,.125,.25]`，补足后bare=.625：原源`surfheat=216899.3529053959, coldcont=0`，旧JAX eager／窄JIT反置，4个energy字段失败已封存。生产仅改两个调用实参为`thermosoil_veget_max_bg(veget_max)`，kernel接口和其余执行AST不变；不是C34初始化helper修复，也不按调用点分别计两项。修复后288字段对照、真实隔离thermal组件／carry及95通过／2旧控制skip见[验证](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/verification-results.json)。没有据此声称PFT14标准案例、NPP或DOC改善。

主控于2026-11-16接受为**C50**，代码／Oracle／fixture提交 `6a5d4a7f2092f32ef9de064e61ee3224c5b48b01`，见[验收](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/controller-acceptance.json)与[发布](../../outputs/reviews/main-physics-20261116/thermal-initialization-oracle/controller-publication.json)。累计50项／29个相关生产代码提交，其中49项有复查前证据。新脚本与fixture是本轮建立的独立参考基础设施，不能倒写成旧版已有；验收另补脚本的已有目录日志保护，2项CLI检查通过，不另算模型修正。审查者早期误读daily同名文件后在main独立重核，未写入／同步研究树；范围纠正保存在本包审查记录。组件证据不改变`numerical_acceptance=false`或93／126／181分母，旧交付回执仍保留当时pending状态并由主控接受记录更新结论。

## 此前已接受：C49上游US_INIT支持域与实际入口补齐

基线main `512d2a1134b96586d41528aea0376e81fe42e72e`，见[当前包](../../outputs/reviews/main-physics-20261115/us-init-production-closure/README.md)及[来源绑定](../../outputs/reviews/main-physics-20261115/us-init-production-closure/origin-binding.json)。只读git show保存复查前`7cae3e02adad23674767a590fdf4d27f239ce065`的完整HYDROL owner、cold helper、USDA配置helper及raw_run_def通道：旧完整owner直接`jnp.sum(us,axis=3)`，这三个有界函数没有US_INIT setvar，证明该上游缺项在复查前已存在；精确首次引入仍未知，不外推为全项目负面结论，也未执行旧模块。

C49接受的是已物化us子域，并明确拒绝合法但尚未实现的整层marker分支。该guard是当时已接受的显式边界，不能改称今天新发现的旧版隐藏保证或新的C49失败。本包是源支持域扩展，加上新的scalar adapter、真实参数通道、完整state写回及年度consumer镜像接线；调用点不分别计旧bug，也不把今天新增adapter／接线伪称旧版已有。

主控于2026-11-16接受声明域内的上游补齐，代码／测试提交 `d6bad452932d1a7179ff5ab5fd80c771da89e336`，见[验收记录](../../outputs/reviews/main-physics-20261115/us-init-production-closure/controller-acceptance.json)及[发布回执](../../outputs/reviews/main-physics-20261115/us-init-production-closure/controller-publication.json)。复用唯一组合[84／84通过](../../outputs/reviews/main-physics-20261115/us-init-production-closure/verification-results.json)、独立终审和执行AST绑定，核对当前13个交付文件，无重跑。正式49项／28个相关生产代码提交，仍48项有复查前证据／C42新helper遗漏；本次作为C49已声明边界的支持域扩展，不另编号。账本90＋1＋1／92和93／126／181、numerical_acceptance=false保持。getincom本体、源NONE、历史producer／构建、DOC／thermal／maxmass／peqAcoef与整模型门不变，研究同步暂停。原C49、此前源／通过／失败工件95个原字节保留，以下C49接受快照不回写。

## C49：vegstress源初始化与实际入口接线已接受

2026-11-15，本包基线main `194d91350bf375f2c9e49360401b807e07d8a02e`；[上一包](../../outputs/reviews/main-physics-20261115/vegstress-read-path-closure/README.md)取得同树lookup／gather源码，闭合源whole-local ALL999999原值加S，并保存旧真实owner失败和21项检查。[来源绑定](../../outputs/reviews/main-physics-20261115/vegstress-read-path-closure/origin-binding.json)证明pre-review `7cae3e02adad23674767a590fdf4d27f239ce065`与BASE完整owner均遗漏显式all999999分支；真实首次引入提交仍未知。

[主控反馈](../../outputs/reviews/main-physics-20261115/vegstress-read-path-closure/controller-review.json)指出该包仅完成局部owner／reader，实际论文入口仍raw、dispatcher默认cold helper且结果未展开。本次是**同一修复组的集成补齐**：[交付](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/README.md)把共享规则接入真实paper file-init与new-native年入口，默认dispatcher改完整owner并以as_payload展开科学state。既有raw codec／strict93／writer保持；continuous／同年内存继续／checkpoint解码不重放。新adapter、接线与as_payload是本次实现，不将其伪称成旧版已有代码；局部规则与入口集成合并登记为C49，不按调用处重复计数。

[本轮证据](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/verification-results.json)累计32不同检查有通过证据（29＋2＋1，总监督19.358秒），首轮两项新guard夹具的归约1 ULP失败及原输入源保留。只将该guard控制改为精确整数加零，原精确门及生产算法不变，生产源码各轮一致。四真实NC论文入口、默认owner选择／后续callback写回、l_first重复拒绝、新进程与连续相位／pickle及US_INIT前置域均有有界入口证据；旧21按原绑定保留复用。

主控已在声明的源码及materialized-us适配器域接受C49，代码／测试提交 `ea75bde31012cf519d4b0fa5c689d865c9d9c2b7`，见[验收记录](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/controller-acceptance.json)。正式为**49项／27生产修复提交**（48项有复查前证据、1项C42属新helper默认遗漏）。账本90＋1＋1／92、93／126／181及`numerical_acceptance=false`不变。us未完成整层US_INIT是Fortran合法但当前materialized-us子域未实现的上游边界，显式拒绝；源NONE、native数值／历史身份、DOC／thermal／maxmass／整模型门保留。没有新模型步／完整物理／Fortran／GPU／训练／调度／付费／环境重建，研究同步仍暂停；原失败和成功工件保留。提交及远端核对另记[发布回执](../../outputs/reviews/main-physics-20261115/vegstress-production-entry-closure/controller-publication.json)。

## C48（原 FIELD-CANDIDATE-01）生产接线已接受

2026-11-14 的[接线包](../../outputs/reviews/main-physics-20261114/calendar-driver-wiring/README.md)完成共享 calendar、真实 metadata／DIM2、实际 caller／compiled／年与 IO 接线。来源沿用复查前 `7cae3e02` 已存在的 driver owner 调用遗漏。本包保存 b5 `b5eff879e00154c936a1709d5ab91aae65f158d6` 中完整 `_runtime_next_state_from_components` 函数，在相同物理替身下调用其旧 signature，复现 Jan1／Jan2 丢失／stale 日历值，再执行当前完整 builder 验证双镜像修复；fixture、原函数 SHA 与跨度见[来源绑定](../../outputs/reviews/main-physics-20261114/calendar-driver-wiring/baseline-replay-identity.json)。没有执行旧模块／旧模型，也不追认未受 Git 跟踪 Fortran 的历史身份。

当前 `slowproc.f90:556–610` 的 pre-STOMATE 顺序、Jan1／Jan2／午夜、全部PC分支及严格边界为源依据，C45–C47 primitive/native证据复用。120个唯一检查最终通过：首次112通过／7夹具失败、修夹具后定向7通过，再补1个实际host factory；原回执保留、生产SHA未改。[C48主控接受](../../outputs/reviews/main-physics-20261114/calendar-driver-wiring/controller-acceptance.json)限于接线语义，不宣称标准案例数值回归或其排除，不恢复原001／旧archive构建／命名实际重启producer。正式累计48／26，研究同步暂停。下方旧包表格保留交付当时状态，当前C48状态以本节为准。

## 原C1–C40来源审计快照

以明确的复查前边界 `7cae3e02adad23674767a590fdf4d27f239ce065` 为准，**40项的错误表达式或源合同遗漏均能直接在该基线中定位**。这支持“这40项所修的问题在本轮共同核心复查前已存在”，但不支持“40条独立、当时均激活的物理公式bug”。没有在这40项中证实复查期间新引入的回归；未审计所有未接纳候选或每个中间实现。

审计冻结已接纳登记HEAD为 `765c75fc51af981b1f687f8e36b05f8dd382dc77`。先读[累计登记](shared_core_branch_review.md#累计修改位置与提交索引)，再逐项核对基线源码及22个准确修复提交；没有切换版本、运行旧测试/旧模型或追溯到最初移植提交。下表“首次已知”统一为基线 `7cae3e02`，真实引入提交未知且至迟在该边界已存在；不能把修复日期当成引入日期。

来源标签计数：`baseline-defect-present`=40；`review-era-new-or-extended-interface-omission`=0；`confirmed-regression-introduced-during-review`=0；`insufficient-origin-evidence`=0。基线存在性的置信度为高，准确引入提交仍未知，零unknown不代表已找出40个最初引入提交。

性质按一个主维度互斥计数：源过程计算/常数/单位/顺序13项；默认/参数/状态/restart接口20项；显式诊断API4项；执行缓存/AD稳定性3项。源过程计算组也包含条件分支，不等同13项都在纸面案例自然激活。C2/C3尤其不应称为两条新增物理公式错误。40个概念修正对应22个代码提交，多字段、多caller或一次修复尝试不重复计数。

机器证据与可复现脚本：[origin-audit.json](../../outputs/reviews/main-physics-20261113/final-gas-runtime-and-fix-origins/origin-audit.json)、[build-origin-audit.py](../../outputs/reviews/main-physics-20261113/final-gas-runtime-and-fix-origins/build-origin-audit.py)。JSON逐项保存完整revision、函数跨度/身份及精确基线摘录；表内行号都是**基线版本**，可用 `git show 7cae3e02adad23674767a590fdf4d27f239ce065:<path>` spot-check，不能直接当作当前工作树行号。修复列保留完整SHA；`git show <repair> -- <path>`可复核改动。

| ID | 修正性质 | 基线直接证据（路径省略`jax_orchidee/`） | 首次已知/引入边界 | 准确修复提交 | 来源标签/置信度 |
| --- | --- | --- | --- | --- | --- |
| C1 | 执行缓存 | `sechiba/diffuco.py:3472–3473` `_diffuco_pft_column_mask`：host lru_cache直接保存JAX数组；可保存tracer | `7cae3e02`；最初引入未知、至迟基线 | `db4dd8f1cc86a33dc8e343725fda2803f63fb358` | `baseline-defect-present` / 高 |
| C2 | AD稳定性 | `stomate/carbon_kernels.py:4000–4013` `npp_closed_update`：维持呼吸份额仍用普通除法，尚未用稳定比值原语 | `7cae3e02`；最初引入未知、至迟基线 | `e317fa81b97b8b8c959f2e7dfdfba525bb54dec8` | `baseline-defect-present` / 高 |
| C3 | AD/份额条件化 | `stomate/carbon_kernels.py:4563–4576` `turnover_leaf_age_fall`：叶龄份额从带共同叶质量的分子/分母求商，尚无relative质量路径 | `7cae3e02`；最初引入未知、至迟基线 | `2a3e303845154576c6f4b5c0b8155106e5bcc70b` | `baseline-defect-present` / 高 |
| C4 | restart单位/默认 | `stomate/restart_io.py:790–803` `read_stomate_readstart_states_from_template`：tau缺值上限把系数3当作天数 | `7cae3e02`；最初引入未知、至迟基线 | `4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99` | `baseline-defect-present` / 高 |
| C5 | 调用参数 | `sechiba/thermosoil.py:3153–3171` `thermosoil_explicit_step`：getdiff调用缺原水文nslm；同函数较早profile调用已传nslm，不能混淆两处 | `7cae3e02`；最初引入未知、至迟基线 | `a117cd5afc92ea55c9aeb528eec6f724de780111` | `baseline-defect-present` / 高 |
| C6 | 过程时钟单位 | `driver/orchestration.py:7933–7946` `_paper_ok_leak_boundary_from_sources`：TF_DOC实际producer传整日dt而非半小时dt | `7cae3e02`；最初引入未知、至迟基线 | `27ff14b38aafc7df91ce52b8393642ca08952395` | `baseline-defect-present` / 高 |
| C7 | 过程率/质量单位 | `stomate/integration.py:3995–4011` `stomate_ok_leak_explicit`：DIC聚合直接接收四项litter/soil率，缺步长转换 | `7cae3e02`；最初引入未知、至迟基线 | `27ff14b38aafc7df91ce52b8393642ca08952395` | `baseline-defect-present` / 高 |
| C8 | 过程共享输入顺序 | `stomate/integration.py:3958–3971` `stomate_ok_leak_explicit`：soilcarbon接收原bulk_dens，缺litter及soil两次共享赋值 | `7cae3e02`；最初引入未知、至迟基线 | `27ff14b38aafc7df91ce52b8393642ca08952395` | `baseline-defect-present` / 高 |
| C9 | 默认/装配参数 | `driver/orchestration.py:8325–8338` `_paper_compiled_ok_leak_fold`：独立源默认cutoff错误地绑定水文层数 | `7cae3e02`；最初引入未知、至迟基线 | `27ff14b38aafc7df91ce52b8393642ca08952395` | `baseline-defect-present` / 高 |
| C10 | 过程赋值顺序 | `stomate/soilcarbon_kernels.py:3547–3560` `soilcarbon_leak_doc_diffusion`：第二峰值通量仍用旧f0构成的共同分母 | `7cae3e02`；最初引入未知、至迟基线 | `225cb95a6a44334390d1656d85f66fd54d342a19` | `baseline-defect-present` / 高 |
| C11 | 过程常数 | `sechiba/hydrol.py:2445–2458` `snow3lheat_explicit`：雪融化潜热默认333600而非333700；与土壤冻结lhf分开 | `7cae3e02`；最初引入未知、至迟基线 | `dd6d6335ba5011fcd86285984136a6799f6916ac` | `baseline-defect-present` / 高 |
| C12 | 过程分支阈值 | `sechiba/hydrol.py:10576–10615` `hydrol_soil_all_tiles_explicit_step`：统一peat阈值传入各阶段，丢失tile4/5和4/5/6差别 | `7cae3e02`；最初引入未知、至迟基线 | `d48b923e891946d3c9db77ba9b684666cb511b1d` | `baseline-defect-present` / 高 |
| C13 | 过程重构顺序 | `sechiba/hydrol.py:11774–11787` `hydrol_alt_residual_solve_step`：dummy只从原mc重构一次，缺冻结条件首赋值后再读更新mc | `7cae3e02`；最初引入未知、至迟基线 | `d48b923e891946d3c9db77ba9b684666cb511b1d` | `baseline-defect-present` / 高 |
| C14 | 显式诊断API | `sechiba/hydrol.py:7583–7596` `hydrol_peat_water_table_diagnostics`：暖层总数代替地表连续暖层前缀 | `7cae3e02`；最初引入未知、至迟基线 | `d48b923e891946d3c9db77ba9b684666cb511b1d` | `baseline-defect-present` / 高 |
| C15 | 参数传递/缓存 | `sechiba/condveg.py:1094–1107` `condveg_main_minimal`：main签名/roughness调用不转发已有Z0_BARE | `7cae3e02`；最初引入未知、至迟基线 | `bbcdbf7a661cb11f5295883e54c3b1ea9d596d39` | `baseline-defect-present` / 高 |
| C16 | 状态/调用传递 | `sechiba/hydrol.py:8019–8038` `hydrol_module_diagnostics`：诊断读取split旧ae而非surface更新值；driver也沿用旧split | `7cae3e02`；最初引入未知、至迟基线 | `2f36dc7f7945d52387a2848aa124c043118ed45f` | `baseline-defect-present` / 高 |
| C17 | 过程末态刷新 | `sechiba/hydrol.py:10709–10733` `hydrol_soil_all_tiles_explicit_step`：normal末态后直接进入dummy，无final-mc冻结profile刷新 | `7cae3e02`；最初引入未知、至迟基线 | `ad0d322cc4951d4af158f4e8cbb6d306bdf374f3` | `baseline-defect-present` / 高 |
| C18 | 过程整数/浮点边界 | `sechiba/hydrol.py:6441–6454` `hydrol_soil_coef_mineral_profile_from_tables`：非冻结表达式先加imin再INT；peat同类表达式同在基线 | `7cae3e02`；最初引入未知、至迟基线 | `ad0d322cc4951d4af158f4e8cbb6d306bdf374f3` | `baseline-defect-present` / 高 |
| C19 | 显式诊断API | `sechiba/hydrol.py:7592–7605` `hydrol_peat_water_table_diagnostics`：潮汐水库只对当前tile6选择，非按cell的tile6面积选择全部tile | `7cae3e02`；最初引入未知、至迟基线 | `ad0d322cc4951d4af158f4e8cbb6d306bdf374f3` | `baseline-defect-present` / 高 |
| C20 | 门控参数/缓存 | `sechiba/condveg.py:1131–1144` `condveg_main_minimal`：main没有snowcri_alb入口/实参，固定消费helper默认 | `7cae3e02`；最初引入未知、至迟基线 | `f8d36e1f8c03046e12e146aed2abb8bd286403de` | `baseline-defect-present` / 高 |
| C21 | 参数传递/缓存 | `sechiba/condveg.py:1153–1170` `condveg_main_minimal`：动态roughness调用缺MIN_WIND | `7cae3e02`；最初引入未知、至迟基线 | `c214e43da57c21b68002186fbf71c67d45835bae` | `baseline-defect-present` / 高 |
| C22 | 参数传递/缓存 | `sechiba/condveg.py:1153–1170` `condveg_main_minimal`：roughness调用缺HEIGHT_DISPLACEMENT | `7cae3e02`；最初引入未知、至迟基线 | `c214e43da57c21b68002186fbf71c67d45835bae` | `baseline-defect-present` / 高 |
| C23 | 参数传递/缓存 | `sechiba/condveg.py:1153–1170` `condveg_main_minimal`：roughness调用缺Z0_ICE | `7cae3e02`；最初引入未知、至迟基线 | `c214e43da57c21b68002186fbf71c67d45835bae` | `baseline-defect-present` / 高 |
| C24 | 参数传递/缓存 | `sechiba/condveg.py:1131–1144` `condveg_main_minimal`：质量雪fraction调用缺SNOW_DENSITY | `7cae3e02`；最初引入未知、至迟基线 | `c214e43da57c21b68002186fbf71c67d45835bae` | `baseline-defect-present` / 高 |
| C25 | 状态/雪相位传递 | `driver/orchestration.py:2932–2944` `_add_condveg_to_previous_fields`：packet仅保存四项roughness，未保存snow fractions；restart首CONDVEG仍接旧雪 | `7cae3e02`；最初引入未知、至迟基线 | `c214e43da57c21b68002186fbf71c67d45835bae` | `baseline-defect-present` / 高 |
| C26 | 显式诊断默认 | `sechiba/hydrol.py:8429–8442` `hydrol_tmc_update`：可选水量检查默认零容差而非源2e-8 | `7cae3e02`；最初引入未知、至迟基线 | `84b5d425418a6d1b18fa6dbe080348c2faa72e13` | `baseline-defect-present` / 高 |
| C27 | 过程跨PFT顺序 | `stomate/soilcarbon_kernels.py:3738–3751` `soilcarbon_leak_doc_export`：每个PFT内部把peat累计buffer置零，并只回加本PFT loss | `7cae3e02`；最初引入未知、至迟基线 | `9fc87c011a4e991f1637223b0f8fb2a224000595` | `baseline-defect-present` / 高 |
| C28 | 显式诊断默认 | `stomate/soilcarbon_kernels.py:2706–2719` `soilcarbon_perma_peat_redistribute`：peat_OLT诊断默认0而非源1e-8 | `7cae3e02`；最初引入未知、至迟基线 | `9fc87c011a4e991f1637223b0f8fb2a224000595` | `baseline-defect-present` / 高 |
| C29 | 过程根归一化 | `sechiba/hydrol.py:2048–2060` `hydrol_nroot_from_humcste`：ALT0时对全部原根求和触发归一化，缺源znh<ALT累计条件 | `7cae3e02`；最初引入未知、至迟基线 | `ea86d3fbe97b3a1ecf5c5d8dda931aea5ea2ae69` | `baseline-defect-present` / 高 |
| C30 | 过程网格几何 | `sechiba/hydrol.py:1922–1934` `hydrol_first_step_profil_froz_from_arrays`：冻结profile从相邻节点差构造dh而非真实dlh*1000 | `7cae3e02`；最初引入未知、至迟基线 | `ea86d3fbe97b3a1ecf5c5d8dda931aea5ea2ae69` | `baseline-defect-present` / 高 |
| C31 | 过程末态温度 | `sechiba/thermosoil.py:3579–3592` `run_thermosoil_first_step_module`：结束直接返回downstream，缺显雪grid温度clamp写回 | `7cae3e02`；最初引入未知、至迟基线 | `cb4f76817acd852cde8cd8d4195e1e52bfd8c770` | `baseline-defect-present` / 高 |
| C32 | restart packet所有权 | `driver/restart_bundle.py:94–105` `paper_restart_bundle_state_from_day_end_packet`：gas/remainder直接沿用base，未用当前packet；standalone同类 | `7cae3e02`；最初引入未知、至迟基线 | `77c3ac7d30ddf466a3cc2b4d6824824713a632b1` | `baseline-defect-present` / 高 |
| C33 | restart派生量 | `driver/restart_export.py:88–101` `stomate_restart_states_from_day_end_packet`：合并pool后缺失total仍留旧base，不从实际pool重建 | `7cae3e02`；最初引入未知、至迟基线 | `77c3ac7d30ddf466a3cc2b4d6824824713a632b1` | `baseline-defect-present` / 高 |
| C34 | restart初始化/持久状态 | `driver/orchestration.py:11513–11533` `paper_1961_driver_restart_year_start_day_result`：直接rebase进入main，缺非持久热初始化；file路径同缺且首packet漏e_soil_lat | `7cae3e02`；最初引入未知、至迟基线 | `fd0fe51495626b39ce42d50dec29b3a6cef744b3` | `baseline-defect-present` / 高 |
| C35 | restart目录传递 | `driver/orchestration.py:7109–7122` `paper_1961_driver_timestep_scaffold`：STOMATE boundary调用未传first_step_restart_state.run_dir | `7cae3e02`；最初引入未知、至迟基线 | `fd0fe51495626b39ce42d50dec29b3a6cef744b3` | `baseline-defect-present` / 高 |
| C36 | restart active初始化/保持 | `driver/orchestration.py:5390–5404` `_paper_first_step_restart_entry_sources`：active STOMATE也用inactive derivvar覆盖；缺恢复assim/height及月温初始化growth | `7cae3e02`；最初引入未知、至迟基线 | `fd0fe51495626b39ce42d50dec29b3a6cef744b3` | `baseline-defect-present` / 高 |
| C37 | restart整数接口 | `sechiba/restart_io.py:162–166` `sechiba_finalize_source_state_from_restart`：NC REAL映射直接返回，未恢复njsc/veget_year的INTEGER owner类型 | `7cae3e02`；最初引入未知、至迟基线 | `fd0fe51495626b39ce42d50dec29b3a6cef744b3` | `baseline-defect-present` / 高 |
| C38 | compiled forcing绑定 | `driver/orchestration.py:9934–9947` `_paper_1961_later_day_transition_inputs`：静态模板按year/tstep48取天气；start0盐/潮位用day_start_payload而非实际series0 | `7cae3e02`；最初引入未知、至迟基线 | `45654fea50aec6edf7774ac0f0b90170eea617db` | `baseline-defect-present` / 高 |
| C39 | gas缺值默认 | `stomate/restart_io.py:795–808` `read_stomate_readstart_states_from_template`：O2/CH4缺值默认都为0，而非源298.3/0.001267 | `7cae3e02`；最初引入未知、至迟基线 | `76e406e0e0e75402d221bb281908fa4260bc5d66` | `baseline-defect-present` / 高 |
| C40 | gas创建/状态carry | `driver/orchestration.py:3451–3464` `driver_previous_step_state_from_next_step_hydrol_scaffold`：首日后继scaffold白名单末尾无四gas；cold producer/extractor也无创建/携带 | `7cae3e02`；最初引入未知、至迟基线 | `76e406e0e0e75402d221bb281908fa4260bc5d66` | `baseline-defect-present` / 高 |

C40的三个边界需分清：基线已没有cold gas producer/extractor，且首日后继`driver_previous_step_state_from_next_step_hydrol_scaffold`的carry清单缺四gas；基线的`STOMATE_FIXED_PATH_CARRY_FIELDS`及初始restart提取却已有gas。因此“某处已有gas”不能证明完整首日生命周期。初版修复没有闭合该既有遗漏，是不完整修复，不自动成为复查引入回归或新C编号。本审计不改C40已接纳的`76e406e0e0e75402d221bb281908fa4260bc5d66`身份，也不把本轮后续运行候选预先接纳。

C34非持久热初始化、C35显式目录、C36 active初始化及growth保持、C37 INTEGER解码、C38静态模板/实际forcing绑定，均在基线已有对应入口且缺少所修合同；不是仅凭2026-11修复时间判作新增接口遗漏。C32/C33的bundle/standalone packet合并和派生total接口也直接存在于基线。C15/C20–C24新增参数接线是在修补基线已有component/helper与caller的断接；不能因修复增加签名或resolver就反推缺陷由复查新增接口造成。

此表只审计来源和性质；数值、运行覆盖、活动gate、历史参考producer及研究分支同步状态仍以原登记和[主报告](main_physics_review.md)为准。没有由源码存在性推出年度影响、所有分支通过或gas动力学激活。

## C41–C43及未修日历候选的有界来源核查

本节新增来源核查与上方原40项快照分开。主控已将P-CTX-OKPC、P-REFSOC-DEFAULT、P-COMPILED-CLOSURE分别接纳为C41、C42、C43；FIELD-CANDIDATE-01仍未修复。完整JAX提交边界、精确源码跨度、当前未跟踪Fortran文件SHA256及限制见交付时的[机器记录](../../outputs/reviews/main-physics-20261113/parameter-repairs-calendar-design/candidate-origins.json)和[说明](../../outputs/reviews/main-physics-20261113/parameter-repairs-calendar-design/candidate-origins.md)，原交付候选ID保留以便追溯。

| 候选ID | 有界来源分类 | 结论与限制 |
| --- | --- | --- |
| `P-CTX-OKPC` | 复查前已存在的JAX参数接口遗漏 | `7cae3e02`仍把raw `OK_PC`复制进context；初始仓库提交`7333b46c`已出现该表达式。当前本地Fortran `control.f90:255–278`要求`OK_LEAK`有效时将`ok_pc`置false，但该源文件不受Git跟踪，不能声称它在复查前边界的Git历史中存在。 |
| `P-COMPILED-CLOSURE` | 复查前已存在的JAX缓存／闭包绑定遗漏 | `7cae3e02`已有跨context复用compiled constructor缓存的路径，缓存闭包可捕获首个context；初始仓库提交`7333b46c`已包含该路径。已接受反例替换了编译与物理delegate，不是实际compiled物理数值失败。 |
| `FIELD-CANDIDATE-01` | 复查前已存在的选定driver日历owner接线遗漏 | `7cae3e02`已有JAX `slowproc_main_pft14_step`日历owner，但选定orchestration未调用；day-end adapter也未接入Jan-1 reset、Jan-2 `peatC_ok`写入及普通日末`growth_day`增量。当前本地Fortran `slowproc.f90:556–594`给出对应顺序，文件未跟踪，不能据此恢复Fortran的Git历史。普通日末部分patch仍不完整，未应用。 |
| `P-REFSOC-DEFAULT` | 复查期间新helper缺少源默认 | helper及缺键`FALSE`回退由`fd0fe51495626b39ce42d50dec29b3a6cef744b3`于2026-11-12引入。当前本地Fortran `thermosoil.f90:337–374`在可选override前将`use_refSOC`初始化为true；该源文件未跟踪，历史限制同上。此项是新增restart准入边界默认遗漏，不证明热初始化数值错误或标准算例回归。 |

这四组不应统一描述为“旧翻译遗留”或标准案例回归。前三组只确认JAX行为已存在于复查前Git边界；准确初始作者提交在可用历史中未知。第四组属于新加入helper的默认遗漏。原C1–C40的40项／22提交接受范围保持不变；新增C41–C43共享上述一个代码提交。所有生产修复的准确提交、验证范围和研究同步状态以累计登记为准。

## C44及2026-11-13包交付时的clock候选来源

本节记录2026-11-13包交付时的状态：COLD-THERMAL-INPUT已接纳为C44，两个clock设计候选当时仍未修。原C1–C43的来源审计不重做。详细边界和哈希见交付时的[机器记录](../../outputs/reviews/main-physics-20261113/clock-authority-cold-inputs/cold-candidate-origins.json)及[说明](../../outputs/reviews/main-physics-20261113/clock-authority-cold-inputs/cold-candidate-origins.md)，候选ID保留以供追溯。

| 草案ID | 来源分类 | 有界结论 |
| --- | --- | --- |
| `COLD-THERMAL-INPUT` | 复查前已存在的cold source-default/static-input adapter遗漏 | 两个cold `use_refSOC`缺键`FALSE`分别用于静态读取门及cold initializer实参，均由`git blame`追到初始JAX基线`7333b46c`，且存在于`7cae3e02`。新restart helper的C42是另一个review-era问题，已在C41–C43接受提交中修复，不在本候选重复计数。当前Fortran source未跟踪；无标准case数值回归结论。 |
| `CLOCK-YEAR-LENGTH-DESIGN` | 复查前JAX表达式用day单位`one_year`除以second单位`dt_sechiba`，与恢复IOIPSL duration语义不同，活动影响未定 | JAX用`rint(one_year / dt_sechiba)`；恢复的IOIPSL `tlen2itau('1Y',dt,date0,...)`按实际日期转换。noleap、Jan1、dt=1800的源算术为17520步，JAX表达式为0；这是公式推导，不是运行观察。历史001的date0/calendar仍缺。 |
| `CLOCK-GREGORIAN-EPOCH-DESIGN` | 复查前JAX Gregorian helper与恢复IOIPSL日序epoch不同，活动影响未定 | JAX用astronomical Julian split；恢复的IOIPSL Gregorian日期路径使用Lilian日数及`2299160`偏置。历史001的date0及source-to-archive桥仍缺。 |

上述cold项现为C44，按一组cold控制／静态输入接口修复登记，相关satsoil传参和多caller不重复计数。按2026-11-13包交付时的边界，clock两项仍未修；其时没有生产日历修复或原输出失败结论，既有C1–C43接受范围保持不变。后续本地helper修复状态见下一节。

## C45–C47时钟helper来源（2026-11-14已接受）

`CLOCK-YEAR-LENGTH-DESIGN`与`CLOCK-GREGORIAN-EPOCH-DESIGN`仍分别保留其既有来源分类：二者的旧JAX表达式均在初始仓库提交`7333b46c`及复查前边界`7cae3e02`中，且不是本轮新引入的生产错误。新增识别的IOIPSL参考在41条clock owner/primitive记录和4个合成restart元数据case上与配对源码一致；这支持有界helper合同，不代表原始`001`、source-to-archive构建身份、生产driver日历接线或年度物理验收。完整证据、来源跨度与hash见[native helper来源记录](../../outputs/reviews/main-physics-20261114/native-clock-helper/helper-origins.md)和[机器记录](../../outputs/reviews/main-physics-20261114/native-clock-helper/helper-origins.json)。

C47 fractional-second修复有独立baseline证据，正式与C45年长、C46 Gregorian起点分开登记：`7333b46c`、`7cae3e02`及`a5748f6a`中的noleap callback/state将秒标为整数；唯一已确认运行时截断发生在`intersurf_time`构造返回状态时的`second=int(second)`。year-offset helper直接使用传入值，其`int`注解本身不执行截断。组合验证用native `nlfrac` owner值执行保存的旧源码，旧helper返回状态为900秒而IOIPSL owner返回900.2499952912331秒。其值来自请求900.25秒序列化后的`date0=715765.0104195601`，故也记录标量精度与owner量区分；不推出历史001曾使用fractional date0。候选审查中发现并修正的`floor(x+0.5)`近半值NINT边界属于新候选实现审查，不归因旧生产源码，也不另计正式修正。既有synthetic 2001 stub仍是独立旧fixture，native参考不验证其身份或历史值。组合local receipt为80项PASS、无error/skip，baseline控制3项。
