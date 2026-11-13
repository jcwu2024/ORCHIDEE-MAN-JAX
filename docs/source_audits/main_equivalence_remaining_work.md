# PFT14 剩余源码与状态覆盖

## S-NEXT-01：当前状态／参数合同交付

2026-11-13，基线main `8f3c7dbab3ce22af5cd2a828d1ea5416b0489035`。主控已接受本包92字段逐行处置、三个选定入口的有效参数合同及有界接口反例；**尚不满足PFT14等价发布合同**。92个身份全部复用，新增身份0；87项在选定源域内合同归一，5项保留精确缺证。纯适配器检查复现4组源／接口语义矛盾，尚未修复生产；C1–C40／22个代码提交保持原接受范围。逐项工件、原始回执与候选补丁见[本包目录](../../outputs/reviews/main-physics-20261113/state-parameter-contracts/README.md)。

当前字段域为PFT14／USDA12／CWRR／显式雪／IMPOSE_VEG，分开记录cold、restart、首日、普通、compiled、日年界与IO相位。上一包92项身份关联及26项精确合同候选是有效的局部证据，并未认证这92项的全部七维源语义；本包不把它们重计为“此前全部未知”，也不据此推算完成百分比。

| 字段合同维度 | 选定源域已解析 | 精确缺项 |
| --- | ---: | ---: |
| 单位 | 91 | 1 |
| dtype | 92 | 0 |
| 轴 | 92 | 0 |
| mask／活动条件 | 92 | 0 |
| 默认／sentinel优先级 | 91 | 1 |
| 更新相位 | 90 | 2 |
| 实际生产／carry路径 | 90 | 2 |

两份YAML的消费者为开发审计／测试，选定运行入口不加载它们；本次只改文档合同。已有静态validator首次发现3个实际producer漏列，补入`frac_snow_nobio`、`frac_snow_veg`、`e_soil_lat`的既有producer合同后，canonical SECHIBA条目由123增至126，STOMATE lifecycle仍181。`cgrnd/dgrnd`四条canonical记录改为实际`ngrnd−1`接口轴，源递推单位分别为K／无量纲；q_sol_pot、tsol_rad及带dt水量等按公式归一，原磁盘注释作为冲突证据保留，没有运行时换算或生产修改。

有效参数范围为`prepare_paper_1961_driver_context`→cached实际体、`_paper_compiled_later_day_block_executable`、`restart_thermosoil_initial_fields`及其可达resolver／override／consumer／cache闭包：**179个canonical knob，103个indexed family按族各计一次**。179项均保留精确resolver及adapter投影和单位处置；177项有可执行initializer／MTC链，另2项命名边界保留，独立物理／真实compiled数值认证为0／179。必需materialized键缺失必须报错，其源默认归上游materializer负责；不把旧3,266个used_run键扩张为本包分母。READ_LAI与源LAI_MAP的域外别名、固定NLEAFAGES及SECHIBA_VEGMAX／SLOPE源setvar责任分别记录。

| 已复现的语义矛盾组 | 同输入源期望与当前接口观察 | 当前交付／限制 |
| --- | --- | --- |
| SLOWPROC日历记忆 | Jan1的growth_day `[7,8,9]`应清零而实际carry不变；Jan2的peatC_ok应置1而仍0；日末pre-STOMATE温度`[6,5,−2]`应使growth变为`[8,8,9]`而仍`[7,8,9]`。源写入在DYN_PEAT判断之前，不能以该开关false排除 | 一组、两个字段、三个反例；实际calendar owner未接入选定driver。仅提供普通日末增量的**部分补丁**；Jan1／Jan2仍需真实clock接线，不能猜tstep取模 |
| P-CTX-OKPC | OK_LEAK／OK_PC均true时，`control.f90:261–278`要求有效OK_PC=false，cached context仍保留raw true | 候选override补丁；未应用／执行 |
| P-REFSOC-DEFAULT | 有效雪／soil-C／leak门下，源`thermosoil.f90:365–374`先置use_refSOC=true；当前缺键回退false而拒绝，显式true对照可进入 | 候选默认补丁；不关闭热初始化独立真值缺项 |
| P-COMPILED-CLOSURE | 同config／layout、两个不同且不变的run.def，FR_DT分别2／3；缓存B仍向spy传A的2，fresh B传3 | 使用真实Python构造体，JIT／lower／compile／scan及物理delegate全部替换为spy；非真实compiled数值失败。候选按context实例分区并显式强引用保活，等价独立context可能少共享编译 |

参数补丁覆盖三项接口问题，并列出同context复用、不同context分区、默认正负控制及weakref／淘汰回归方案。其余daily-carbon dispatch与projector callable闭包责任另列。prepared-context的path cache遵守不可变输入约定，**仅path作为key本身未被判为缺陷**。两份候选补丁均只保存在outputs中；字段补丁不构成完整日历修复。

| 仍开放的5个字段 | 精确缺证／下一最小动作 |
| --- | --- |
| growth_day | 绑定源month／day／sec及pre-STOMATE temp_growth，使Jan1 reset和日末增量在cold／普通／compiled／年重启的实际路径中各执行一次；已有`lifecycle_completion.intersurf_time`为源clock权威，当前payload尚未绑定 |
| peatC_ok | 同一真实clock接入Jan2首步且`!dynpeat_PC`的源写回，并把结果传入finalize镜像 |
| vegstress | 原Fortran缺restart分支对sentinel直接累加的疑点，尚非确认翻译缺陷；需原restget缺字段观测及后续hydrol_var_init／首消费者证据 |
| veget_year | 当前IMPOSE_VEG分支无科学年值定义；保留sentinel／int32 codec，需map-vegetation协议再认证科学含义 |
| peqAcoef | 选定非relaxed零系数明确；非零大气relaxation的rate／单位约定需原耦合合同，不能从当前零值推定 |

检查最终逻辑集合为15／15：11项正常assertion及4项预期反例。只运行过一次组合及两次因新失败而进行的定向复查：初次14／15；第一次定向schema通过、validator失败且runner误派参数产生1个TypeError（参数检查体未执行）；修正source span和派发后，最后仅validator复查通过。13项未受影响的初次通过结果复用；最终记录按每项原始receipt／模块SHA绑定，**不是声称最后版本整套重跑**。三份原始回执及全部失败／错误保留，系统TEMP均清理。

本包模型步／模型日／积分、完整物理kernel、真实compiled物理块、Fortran、远端、GPU、训练、付费及环境重建均0。生产／参数／容差／CI未改，未新增C号、未commit/push、未动研究工作树；独立审核仅接受有界合同和接口反例，不是生产数值或等价认证。热边界独立真值、maxmass域限、001历史身份、外部IOIPSL、旧oracle到当前源码／compiled的桥，以及既有gas／DOC严格状态与守恒限制继续开放。下一最小代码动作是主控审核候选参数修复及纯回归方案；字段完整修复须先绑定真实clock。任何后续实际物理／compiled验证另按合同准入。

主控已核对成对源条件、候选补丁、消费者审计、实际逐项回执与独立审核，复用已执行检查，未重复模型或测试；[接受记录](../../outputs/reviews/main-physics-20261113/state-parameter-contracts/controller-acceptance.json)只接受合同和接口反例，不接受未运行的补丁。下一包落实三项参数／缓存修复及回归，同时完成真实源clock的日历接线方案；普通日末部分补丁不得当作完整修复合入。

来源补充：`P-REFSOC-DEFAULT`所在准入helper于2026-11-12提交`fd0fe51495626b39ce42d50dec29b3a6cef744b3`新增，该提交已含缺键回退FALSE。它属于本轮重启修复新增逻辑的缺省边界遗漏，不应算作复查前缺陷；尚无由此导致已通过标准算例数值回归的证据。其余三组的最早来源待限定追溯，不能统一称为旧翻译遗留；C1–C40既有来源结论仍只适用于原40项修正。

## 上包已接受的覆盖快照

2026-11-13：主控已验收本次既有源码、状态台账和接受记录的有限对账；**尚不满足 PFT14 等价发布合同**。本包新增确认的待修语义缺陷为0，缺少验证的条目仍需逐项处理。C1–C40／22个代码提交保持原接受范围。

基线为main `034733109e015ec7330adbf1f75c39c616d9b804`。依据[当前执行计划](pft14_completion_execution_plan.md)和[等价交付合同](pft14_equivalence_delivery_contract.md)，本包只做本地源码／保存证据分析，模型积分、模型日、Fortran执行、远端、GPU、训练和付费均为0；不改生产、参数、容差、CI或修复编号，不commit/push。详细工件、可复现脚本和独立审核见[交付目录](../../outputs/reviews/main-physics-20261113/equivalence-gap-reconciliation/README.md)。

主控已核对当前报告、机器摘要、选定原始对账行、23项源控制／10项状态控制回执及独立审核，未重跑分析器、模型或测试。接受范围见[主控记录](../../outputs/reviews/main-physics-20261113/equivalence-gap-reconciliation/controller-acceptance.json)：本表组织后续工作，不把扫描分母或缺证数量称为bug数量，也不把当前证据关联未完成视为撤销旧接受。下一包执行S-NEXT-01，补齐92字段源合同和有效参数到实际消费者的关联。

## 分母与证据层级

下列视图彼此重叠，分别计数。过程条目、过程片段、分支位点、分支臂和状态字段不能相加；没有由修复数量或测试数量推算完成百分比。

| 源视图 | 分母 | 含义 |
| --- | ---: | --- |
| 既有reachable过程条目 | 52 | 37 active、11 conditional、4 inactive；不是52个原子分支 |
| 精确源owner片段 | 308 | 原351条仅去除43条完全相同源行集合的别名；重叠但不同的片段保留 |
| 过程扩展记录 | 12 | 扩展声明单列，不作为过程闭合证据 |
| 源分支臂 | 26,428 | 七目录、126个源文件：119个`.f90`、7个`.F90` |
| 源过程 | 1,084 | 扫描器可识别的过程声明 |
| 分支位点 | 13,606 | 与分支臂是不同视图 |
| microcase设计行 | 40 | 两份矩阵的设计／历史描述，不是40次已执行验证 |

逐行ID、模块、源范围、reachability、当前kernel/caller身份、缺证及下一动作见[source-reconciliation.json](../../outputs/reviews/main-physics-20261113/equivalence-gap-reconciliation/source-reconciliation.json)；所有模块与各层计数见[denominator-inventory.json](../../outputs/reviews/main-physics-20261113/equivalence-gap-reconciliation/denominator-inventory.json)。

证据采用两轴：`saved_evidence_level`保留旧producer已有的有界证据；`current_contract_level`／`reconciliation_level`按当前合同所需的完整关联计数。保存的48个过程oracle、840个分支臂oracle不能自动成为当前JAX或当前compiled路径的通过。当前源码不同且尚无可核验桥接的记录，保留原数值结果和缺项，不能把它改称当前失败。`source_mapped`还要求reachability、源范围、kernel和状态合同；符号存在本身不足以达到该层。

<!-- source-level-counts:start -->
| 源视图 | 严格当前合同计数 | 保存证据计数（原producer范围） |
| --- | --- | --- |
| 52过程条目 | inventoried 52 | fortran_oracle 48；jax_regression 4 |
| 308精确owner片段 | inventoried 308 | jax_regression 293；inventoried 15 |
| 12扩展记录 | inventoried 12 | source_mapped 9；inventoried 3 |
| 26,428源分支臂 | inventoried 26,428 | fortran_oracle 840；source_mapped 7,009；inventoried 18,579 |
| 1,084过程／13,606位点／40设计行 | 各自全部inventoried | 各自全部inventoried |

本源overlay的当前source_mapped、jax_regression、fortran_oracle、production_verified计数各为0，表示尚未按逐行完整合同建立关联，不撤销既有有界接受。状态视图的独立计数及四gas实际路径证据见下节。
<!-- source-level-counts:end -->

无剪枝源图可达20,000个臂；固定paper开关剪枝后可达11,085个臂，其81条禁用边另列。旧active/conditional分类与当前paper图相交为4,217个数值臂：这部分仍缺逐臂当前源码桥和实际compiled同oracle／写回认证，其中824个已有保存oracle、3,393个尚无已关联的通过oracle。另363个`paper_static_active`是源dispatch边，不替代数值验证。旧`process_span_unmapped=4732`是映射诊断span数，不能解释成4,732个未实现过程。

原剪枝helper拒绝当前`HYDROL_CWRR=true`，因为它只接受字面`y`。本包分析器单独核对等价逻辑写法、LIMIT十进制数值及源码支持的缺省`VEGET_UPDATE=0Y`，恢复81条匹配边并保留原拒绝；非等价／未知配置不剪枝。原helper、run.def和生产代码未改。

固定paper图不等于合同的全部PFT14条件域。改变forcing、点位或连续参数可激活的路径仍需源条件分类；当前false开关、旧peat_hydro inactive记录不能证明所有潜在域不可达。OK_PC／deep-carbon的局部条件证据与完整driver中OK_LEAK控制冲突分别保留，不通过改Fortran或缩小目标制造闭合。

本次扫描尚不能给出完整目标Fortran分母：七目录外源码、外部IOIPSL、build/include/XML、完整预处理与跨行语句、全部带类型FUNCTION、隐含SAVE、完整参数与状态mask尚未穷尽。Windows的`rglob('*.f90')`也匹配本次7个`.F90`文件，平台枚举行为与实际suffix清单单独记录。正则call/generic关联也不是可执行控制流证明。未扫描范围作为后续源清单工作保留；无记录缺口的图索引不证明覆盖完整。

## 状态与参数接缝

| 状态／输入视图 | 分母 |
| --- | ---: |
| Fortran SAVE记录 | 1,094 |
| Fortran restart名称 | 167 |
| SECHIBA合同字段 | 123 |
| STOMATE生命周期字段 | 181 |
| 已保存319 producer的实际used_run键 | 3,266 |
| 专项年度记忆字段 | 56 |
| 历史额外字段 | 92 |
| 指定API签名输入 | 210 |
| 真实IO schema字段 | 285 |
| prepared-context声明 | 81 |

共5,555条机器记录，仅表示这些重叠视图的行数；其中5,103条`inventoried`、452条`source_mapped`，没有新增逐字段Fortran oracle或全域production认证。逐行源声明、单位、轴、mask、创建／更新／写回相位及缺项见[state-seams.json](../../outputs/reviews/main-physics-20261113/equivalence-gap-reconciliation/state-seams.json)。未归一的单位保持`null`，签名默认、context声明和旧实际used_run值不冒充完整有效参数闭包。

92／92历史字段身份已关联：26项精确合同候选、75项显式当前源关联，重叠9项，未关联0项。其中59项finalize已接显式别名、实际Fortran writer、当前更新／条件carry及磁盘维度。原`all_extra_source_schema_complete=false`保持；身份关联没有认证全部单位、mask、有效默认、相位和数值。旧`veget_year` int64与当前int32 codec分别记录。

| 接缝 | 已有可复用证据 | 剩余范围 |
| --- | --- | --- |
| 有效配置／参数 | 旧actual used_run、签名和context三视图；C38选定绑定修复 | 所有有效默认、override、consumer、cache-key及闭包的完整关联 |
| cold／restart创建 | 已有reader／schema／source合同；四gas有已接受真实启动 | 其他字段的源默认、单位、同相位独立真值与非默认输入 |
| 首日专用carry | 四gas真实首日遗漏已修并验收 | 其他字段不能仅凭普通carry常量证明首日extractor正确 |
| 普通／compiled更新 | 四gas实际days2–8的Compiled调用／返回及逐坐标保持 | 其余字段逐oracle当前production路径与完整writeback |
| 日界／年界 | 既有packet、年度记忆及集成保存物 | 完整字段相位、严格连续性、单位一致balance及当前身份桥 |
| 真实三文件IO | 四gas故意错误base覆盖及实际writer→reader已接受 | 其他字段数值、更多合法模板／点位域、外部IOIPSL身份 |

最终gas producer `765c75fc51af981b1f687f8e36b05f8dd382dc77`至本包基线仅五份文档变化，生产与tests源码桥成立。可复用的是四字段、两条8日路线、首次／compiled／日末和真实三文件写读的原有限证据，不外推active gas动力学、全状态或669点。此项旧compiled文档欠账已通过证据对齐关闭，无需重跑。

## 按模块保留的工作

全部源行的模块计数、required标记及缺证分组在机器分母中完整保留；`required_remaining=0`只表示该视图没有标记数值ARM要求，不代表对应模块已通过合同。以下是执行顺序和应补边界。

| 模块／边界 | 保留工作与最小动作 |
| --- | --- |
| driver／forcing／parameters／global | 完整有效默认与case override、输入身份、时间／索引规则、当前caller／compiled缓存关联；先做S-NEXT-01 |
| SECHIBA／SLOWPROC／CONDVEG | 123字段合同及finalize别名的单位／mask／相位闭包；首日与普通更新分别检查 |
| HYDROL／HYDRO_SUBGRID | 现有源oracle到当前调用与状态写回的逐case桥；冻结、mask、条件域保留；不重开已接受C29–C31因果 |
| DIFFUCO／QSAT | 源分支同输入tuple、PFT／植被mask、返回状态及真实production消费者桥 |
| ENERBIL | 现有源微oracle和返回状态的当前production认证；热初始化字段不误标为其直接输入 |
| THERMOSOIL／EXPLICITSNOW | 独立初始化／首消费者真值、非默认热参数域、显雪maxmass合法条件域及同相位写回 |
| STOMATE／carbon／DOC | 生命周期与SAVE／年度记忆逐字段闭包、独立oracle与compiled写回桥；保留DOC严格失败和真实碳通量缺证 |
| restart／modelout／IO | 真实IO schema的单位／轴／phase／非默认覆盖、外部实现与输入身份；之后才选择代表性集成门和669-case发布门 |
| 其余源模块／域分类 | 逐ID保留固定paper外、未扫描、非PFT14、其他配置、diagnostic/error及源未定义契约；只按源依据排除，不能把未知默认为不可达 |

| 稳定未决组 | 性质与边界 |
| --- | --- |
| S-GAP-EXTRA-SCHEMA92 | 身份关联已完成；92字段单位／mask／有效默认及完整相位／数值schema仍未完成 |
| S-GAP-COMPILED-STATIC | 完整有效参数与consumer／cache／闭包分母缺项；保留C38和选定gas证据 |
| S-GAP-REFERENCE-BRIDGES | 剩余旧oracle／当前kernel与caller／状态／真实compiled的逐case身份桥及认证 |
| S-GAP-THERMAL-TRUTH | 原始Fortran初始化终点及首实际消费者的独立数值真值、source／binary／input身份缺项 |
| S-GAP-THERMAL-DOMAIN | custom FR_DT、热表／常数、未审flags、缺coeff的准入尚未认证 |
| S-GAP-SNOW-MAXMASS | 合法显雪剖面、阈值和forcing下的PFT14可达域及源数值case未证明 |
| S-GAP-001-HISTORICAL | 缺历史001／1961实际used_run及输入producer；2010归档边界不是该历史起点 |
| S-GAP-IOIPSL-EXTERNAL | 外部restini/restget等实现、build和NONE实际输出缺项；caller不能提供默认数值真值 |
| S-GAP-DOC-STRICT | 已接受有界浮点敏感性；78坐标maxabs2.2346858610422515e-4和累计诊断差0.02923796884715557仍超原严格门 |
| S-GAP-CARBON-FLUX | 实际边界DOC、大气／DIC、floor／约束碳、异养呼吸及同相位完整库观测未齐；物理守恒残差仍为null |
| S-GAP-IO-DOMAINS | 1×1／index_g=[1]与选定模板不外推任意多点routing、未知模板或669-case集成 |

热真值必须区分创建和消费者：`stempdiag`随后供HYDROL冻结／`temp_hydro`；`pcapa_en`和`temp_sol_beg`供`thermosoil_energy`。三者虽在首main ENERBIL前创建，却不是它的直接输入。独立参考还需level1 capacity、soilcap_initial、post-ENERBIL temp_sol_new和veget_max_bg。完整coeff的最终var_init薄雪overlay与cold／缺coeff随后coef覆盖分别保留；C34接受范围不变。

已接受[DOC核算](../../outputs/reviews/main-physics-20261113/doc-carbon-accounting/controller-acceptance.json)的局部机制与库存定义保持闭合，不启动其准备中的捕获。`carb_mass_total`差不是等量碳损失；真实年末审计库存差0.00286422847186 gC/m²也不替代守恒与严格状态门。未分摊年度差仍未知。历史年度失败保留原producer，不覆盖后来接受的C31／当前科学结果；当前年度通过也不回写旧失败。

C1–C40的准确原commit、接受来源与适用行见[accepted-corrections-map.json](../../outputs/reviews/main-physics-20261113/equivalence-gap-reconciliation/accepted-corrections-map.json)，复用既有来源审计，不重做首次引入调查。上述缺证、数值限制、配置域和历史身份各自计账，不合称待修bug。

## 后续CI与下一包

当前CI仅有CLI／配置／paper-landpoint smoke和相应Ruff。已选择五个现有轻量静态节点：reachable-ledger、evidence-scope、constants原始片段组装、thermosoil源span和未定义OK_ZIMOV策略门。它们依赖已跟踪的源码／YAML；constants节点会导入JAX参数模块。详细节点、未来命令和未保护类别见[CI候选](../../outputs/reviews/main-physics-20261113/equivalence-gap-reconciliation/ci-candidates.md)。本包未执行这些测试或修改workflow。

constants闭合／writeback和thermosoil缓存结果测试另依赖被忽略的`outputs/reference_mode`资产，不能直接加入仅有Git checkout的无条件CI。读取旧pass也不认证当前production。轻量CI不能覆盖完整源分母、状态相位、独立数值oracle、真实compiled写回或669-case发布门。

下一最小包为[S-NEXT-01](../../outputs/reviews/main-physics-20261113/equivalence-gap-reconciliation/next-package.json)：复用已完成的92身份关联，补齐单位、mask、有效默认与cold／首日／普通／compiled／日年界的源相位合同，并把选定used_run键接到实际consumer与cache／闭包。入口为`prepare_paper_1961_driver_context`、`_paper_compiled_later_day_block_executable`、`restart_thermosoil_initial_fields`。完成条件是每字段／选定knob均有源owner、单位、相位、默认／override及消费边，或明确稳定缺项；相位差不能被无条件复制抹平。

此处只提出离线包。发现实际同输入传递遗漏后，才准备一项局部共享修复和回归，并单列未来首次、普通、实际compiled、日年界／真实IO及独立Fortran验证的有界执行合同。逐模块oracle桥接随后推进；热真值、maxmass与历史外部身份可独立处理，不等待另一条DOC轨迹。

源／状态分析器的正负控制、独立复核、报告补丁、文件预算和最终版本身份见[交付回执](../../outputs/reviews/main-physics-20261113/equivalence-gap-reconciliation/delivery-receipt.json)。本包的有限清单不替代合同的完整分母、零悬空状态边、全部活动／条件臂production认证和669-case集成门。
