# 共享物理核心与研究分支核查

同步安排（2026-10-27用户决定）：**暂缓与两个研究分支对齐，优先稳固main**。
本阶段不安排分支同步或交接包工作；28项修复/16个代码提交的登记与适用范围持续维护，
待基础分支本轮加固后集中评估交接。C4–C28仍是待维护方核对的历史登记状态，
并不表示本次核查过研究分支。**当前humrel起源与独立Fortran配对已执行**：本地960条原生捕获完整，
正值由offset4 tile4/PFT14/第6层写入，offset5 DIFFUCO读取；独立Fortran同phase为0。
原JAX单tile胁迫组件对不变JAX tuple和实际Fortran tuple均复现完整返回，maxabs0；差已在层水量输入存在。
七日daily/NPP通过原限；全raw候选440项失败、Fortran负零文本及Day338原生NPP/完整C3输出缺项保留。
更早水状态及年度起因未定，年度NPP仍失败。见
[本批交付](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/README.md)及下文当前节；
主控已[接受有限机制与同输入证据](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/controller-acceptance.json)。
**NPP起因定位尚未完成，继续追踪首次土壤水状态分歧**；局部机制已知不能替代起因与年度归因，原失败/门槛保持。
上一批已[接受有限生产层证据](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/controller-acceptance.json)，
其原生累加器缺失、完整捕获失败及58份清单保持历史状态。
上批365日捕获、当前背景局部分支放大及年度输出核算已由项目主控
[验收](../../outputs/reviews/main-physics-20261027/npp-current-capture/controller-acceptance.json)，其64份清单及回执冻结。
该接受不作为本批验收；上游起因与年度因果未知。此前两次历史输入交换的
[主控验收](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/controller-acceptance.json)保留原身份。
不新增修复或同步；原始NetCDF重启与跨年验证仍待独立裁定。

Stage26当前：四包边界及集中审查S26-R1已主控验收，最终132项定向通过，**本轮有限源码审查收尾**。
见[Stage26交付入口](../../outputs/reviews/main-physics-20261026/final-source-boundaries/README.md)及下方接受分组。
新增C21–C28；Stage25的C17–C20及93项原证据保留。
最新修复登记：2026-10-26。累计28项已接受修复、16个代码提交，仍未建立319年度归因。
**V01已主控验收通过**：001点1961年前9日的编译/普通路径与动态状态恢复通过原数值容差。
[最终交付](../../outputs/validation/v01-after-stage26/domain-v3/README.md)及
[验收回执](../../outputs/validation/v01-after-stage26/primary-v01-acceptance.json)保留完整证据。
两种day9恢复结果最大差异0；累计19模型日/4调用/3进程。原NaN及来源记录两次失败保留，
经源码核验的定义域/路线合同另版通过。本次没有新增物理修复，28项/16代码提交不变。
**V02交付已由主控验收，年度科学验证仍未通过**：固定319点1961冷启动全年运行的
新NPP`0.3860031827055333`，参考`0.38518497347831726`，绝对误差
`0.000818209227216038 > 0.0003951849734783173`；AGB/BGB/GPP年度门槛通过。
见[本次交付](../../outputs/validation/v02-after-stage26/319-1961/README.md)、
[主控验收](../../outputs/validation/v02-after-stage26/primary-v02-acceptance.json)及下方V02结果。
实际1模型进程/1调用/365模型日；原年度失败、raw重启及跨点/长期限制保留。
**V03a数值执行检查已由项目主控验收**：同进程独立cold A8→B8→A8，
再独立ordinary B8，4调用/32模型日。真实outer复用及点位绑定通过；B两路线沿用V01数值状态
合同的完整比较在原限内通过。本次额外载体检查触发44项失败，其历史false保留，不称V01原科学门槛失败。
见[合同与交付收尾](../../outputs/validation/v03a-after-stage26/contract-closure-v1/README.md)及下方V03a结果。
此前primary-v03a-acceptance保留为执行聊天内部审查；真正的
[项目主控最终验收](../../outputs/validation/v03a-after-stage26/controller-final-acceptance.json)另存独立回执。
无新增生产修复，28项/16代码提交和研究分支待核对同步状态不变；不是669点或长期科学验收。

此前Stage23的C11–C16六项HYDROL/雪/CONDVEG修复已主控验收，最终定向组合72项通过；
四组准确提交及适用域见下文Stage23。此前C6–C10五项OK_LEAK修复均已完成主控验收，最终focused组合49项通过。
C6–C9提交为`27ff14b38aafc7df91ce52b8393642ca08952395`；C10独立提交为
`225cb95a6a44334390d1656d85f66fd54d342a19`。实施基线`493e830`不是修复提交。
C10源顺序修复单独分组，明确保留可能产生负剩余库存的源算法局限。C5（THERMOSOIL原水文层数传递）已完成主控验收，独立修复提交为
`a117cd5afc92ea55c9aeb528eec6f724de780111`。C4（restart tau天数回退与配置传递）已在main独立提交为
`4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99`，供日尺度和全球效率分支后续核对同步。
本次未检查或修改两个研究工作树；其C4–C28同步状态均为**待维护线程核对**，不代表已同步。
C1–C3的分支同步记录截至下述2026-10-11核查，不能当作两个研究分支的最新状态。

Stage23六项修复登记为**已接受C11–C16**，原S22 IDs保留用于证据追溯。
四组代码已独立提交，不代表两个研究分支已同步。

累计口径：**28项已接受修复，16个代码提交**；同一提交可以包含多项修复，
测试数量也不等于独立物理过程或完整模型覆盖率。以下总表、各项源码/回归依据及
精确SHA共同构成修复登记。Stage24四项候选经Stage25实现与补项验收，现计入C17–C20；
其历史处置和本批接受范围分别见下文。

## 当前humrel起源与独立Fortran配对（2026-10-28）

本批固定main `33ab88d8c636262033cd921bd58c670e8e281f7f`及`E:/ORCHIDEE-MAN-JAX`实际包，
ORCJAX CPU/x64。复用原end337 packet/动态pcapa_en/context/九家族，唯一compiled7完成Day338–344，
worker168.719秒、监督172.67159175872803秒/900秒硬限、exit0。四次prepare均0科学，TEMP已清；
无本地annual/cold/baseline/warmup/科学重试。

| 当前证据域 | 结果与限制 |
|---|---|
| 本地原生捕获 | mandatory完整，20类×48=960条，另七日完整NPP；含六tile、51参creator、动态根及真实generic累加器 |
| 七日完整daily/NPP | 对原年度maxabs7.587966781058726e-16、对上批compiled7最大1.4113055444079075e-15，原限通过；四NPP分支精确 |
| 不受限raw候选比较 | **FAIL 440项**：200离散/240浮点，全部在assimilate=false且无active叶的Yin/canopy域，原总false不改 |
| 实际生产域 | 全48 bool门/defined C3返回/entry及唯一活动offset5全记录精确；Day338 GPP仍4.564204140244641e-7 |
| 独立Fortran自身 | 49边界/六tile/14 PFT的native-bit域通过；全文本往返784记录/980标量signed-zero失败保留 |
| 两次原水组件 | JAX控制与Fortran cross的完整6返回、全部相关根分支/locals均PASS/maxabs0；仅单tile stress域 |

本地raw先存再准入，15 overlay逆AST通过，显式clock跨原独立JIT，append-only journal与合并结果精确。
actual PEAT_OCCUR=false的generic ordered accumulator此次完整保存48步，最终原生值与NPP入口精确；
旧批缺失及derived会计不回写。Day340/342/343 tiny GPP与原年度分别相隔90,326/3,583,316/1,013,089 ULP，
绝对差均小于8e-16；不能称七日bitwise一致。新day344 packet无旧包可比较。
原atol1e-8/rtol1e-10及结构/shape/dtype/离散/来源合同保持，连续activity额外bitwise检查与实际bool门区分。
见[原准入/失败分类](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/admission-classification.json)。

offset4 HYDROL writer/tstep16180经真实state carry到offset5 DIFFUCO/tstep16181；
实际Fortran对应kjit16181→16182。tile4/PFT14第6层JAX sm3.0827056142277756，
Fortran sm3.079323123859772；共同smw_tmp3.0826979486351997、smf_tmp5.7149560143648、pcent0.8。
两侧根入/出均仅该层为1、presence1、under-MCR=false，NEW_WATSTRESS=false、DYN_NROOT_LARIX=true。
本地factor/denom/humrelv3.6402170609775793e-6，F为0，下一步各自读取同值。
本地正margin7.665592575811786e-6，F在阈值下方；根正denom归一化/零denom保留分支不同而最终根相同。
浅层无根support，aggregate humrel不额外乘tilefrac；STOMATE接收的名为humrel参数实际为HYDROL vegstress。
保存MC/MCL/DZ按原第6层积分式的代数会计与native分列，未执行内部土壤solver；更早根support形成未知。
Fortran全48步writer/reader humrel及生产GPP均0。本步sm差0.0033824903680037544，
由零基节点4/5/6的保存MCL差，经共同原权重贡献0.0029008678151902713/0.0004816225528136591/0，
派生和与原生sm差残差1.7607443281164592e-16；F wilt margin为−0.0033748247754279426。
实际end337 tile4 MC/MCL已不同（maxabs0.0016603309705315594），PFT14根精确相同；
当前writer诊断解释不建立该更早状态差的首次起因。见
[跨侧原生/派生会计](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/cross-state-analysis.json)。
详见[源码/phase契约](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/source-contract.md)
和[主报告](main_physics_review.md)本批节。

独立producer为Cancon jcwu目录下`orchidee_jax_319_humrel_current_20261027`，从冻结源隔离构建，
唯一job125893716/a14r1n19，1节点8CPU16GiB/最大2h；实际allocation500秒=1.111111 allocated CPU-hours，
RSS约3.173GiB。编译376秒、原cold365轨迹93秒，模型exit0；事后旧send文件名检查令wrapper4/调度FAILED，
错误回执及archive失败保留，仅取回现有文件离线验收。32项来源核验通过，实际PP源/binary/worker/输入/冻结文件绑定；
run.def/used_run.def与冻结版相同。实际numout及有效配置1000/.5/.5、48个firstcall=false，
仅支持旧源`den_man/weight_ven_a/b`与仓库对应literal的本case等价，原静态pending审计身份保留。
实际四通量各365×14 float32存储值与已接受Stage12精确相同，maxabs0；Day338/PFT14 GPP0、
NPP9.99999993922529e-9、MAINT_RESP−9.99999993922529e-9、GROWTH_RESP0。
完整Day338原生NPP九字段guard仍337，C3虽65输入在而完整10输出缺证据；两缺项保留。
直接TRANSFER位解码独立域与失败decimal域分列，不改文本/容差。来源、资源和捕获边界见
[Fortran入口](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/fortran/README.md)。

组件域独立准入后，PID3924唯一300秒worker实际2调用，合计1.281秒、监督4.872981071472168秒，TEMP清。
先不变JAX完整记录214叶/4109元素及整个返回通过；再原`hydrol_water_stress_diagnostics`接实际F17参，
完整6字段/348元素和101叶/814元素相关分支/locals均maxabs0。under-MCR增量是native前后counter之差的
派生接口映射，部分谓词由native操作数按源IF组合并有provenance；C3整对跳过，调用0。
见[组件域准入](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/component-domain-admission.json)及
[实际比较](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/component-admission.json)。
所测单tile胁迫域未复现翻译差，输入层水量差已存在；该证据不覆盖完整51参creator、内部soil solver或全年因果。
下一最小事实是形成该end337 tile4水状态的更早实际soil-update完整入参与原生输出对应，
现有post-solver及前entry记录不足以代替；本批不追加轨迹。

原年度NPP误差0.000818209227216038仍超原限0.0003951849734783173；28修复/16代码提交、
42行/28ID登记及原支持域不变，无新Stage/修复、生产物理/参数/阈值/容差修改。
两研究工作树未访问/修改，对齐/交接继续暂停；无GPU/训练/环境重建、commit/push、自动化或主控聊天消息。
图当前Transport closed，历史代次仅作历史记录，使用精确源回退。交付时待主控状态保留，
旧58/64清单和主控回执冻结；最终工件/两报告绑定见
[静态核验](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/final-verification.json)。

2026-10-28主控核对236份实际清单工件、报告SHA及上述机制/组件/参考证据，复用执行方检查，零科学/统计重跑。
[独立主控验收](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/controller-acceptance.json)
只接受当前局部水状态跨阈值机制和所测单tile组件的同输入一致；全部raw/文本/wrapper失败与参考缺项保留。
用户进一步明确未解决问题仍需继续；主控撤回收尾安排，改为从初始状态与连续共同边界定位首次有意义分歧，
再核对对应soil-update完整输入/输出及传播。已完成批次不等于起因定位完成，上游浮点/实现原因和全年归因仍未知。
历史主控验收/发布回执不回写，其收尾安排由
[后续目标更正](../../outputs/reviews/main-physics-20261027/npp-humrel-origin-current/controller-followup-decision.json)替代；有限证据仍有效。
年度NPP原失败及门槛保持，没有新增生产修复或容差调整；28项/16代码提交及暂停研究对齐的决定保持。

## 上一批Day338 GPP：生产层定位与完整捕获缺口（2026-10-27）

固定`E:/ORCHIDEE-MAN-JAX`、main `7acead83618e64483095e69621f11e70137a9658`及该checkout实际包，
ORCJAX/JAX0.10.1 CPU/x64、Python-B/无持久缓存。复用上批真实end-Day337/tstep16175的packet、
动态pcapa_en、context及原九家族，只运行一次原factory的compiled7块Day338–344；
真实science_day_numbers跨原scan/JIT传入观察clock，原静态placeholder和独立JIT保留。
worker PID65264，调用及保存114.391秒、监督118.074427秒/900秒硬限、exit0，TEMP已清理。
两次prepare均0科学；年度/cold/额外组件/baseline/warmup/重试为0，唯一科学预算已用尽。

| 实际证据 | 准入与限制 |
|---|---|
| 七日完整daily，231数值叶/2961元素 | 原atol1e-8/rtol1e-10通过，maxabs2.17010222251378e-15，0失败 |
| 七日NPP完整九参/locals/六返回，196叶/13251元素 | 同原限通过，maxabs同上，0失败；enough/over_tax/no_alloc/negative七日精确不变 |
| 结构及定义域 | 容器/键/shape/dtype/离散/来源精确，数值载体类不要求相同；两处源未定义NaN合同保留 |
| Day338原生亚日记录 | Yin/canopy/C3/entry各48条、合计192；此前无同轨迹亚日内部支路可逐步比较 |
| 完整mandatory capture | **FAIL**，缺`('accumulator',338,0)`；可用轨迹与保存NPP支路PASS单列 |
| 新day344末packet | 已真实保存；旧day344末packet不存在，不伪造末状态新旧数值比较 |

实际`PEAT_OCCUR=FALSE`、compiled runtime开启、numpy runtime关闭，走`daily.py:1432–1499`
generic ordered loop（仍在outer compiled图内）。观察器仅挂到peat-static JIT，guard不成立而未调用，
没有原生逐步gpp_d及累加前后记录。初次离线准入KeyError、原工具版本和错误均保留，只修离线分析，无科学重试。
见[准入](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/admission.json)和
[实际路线/缺口](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/accumulator-record-gap.json)。

Day338/339/340/341/344 tiny GPP与上批0ULP，零/正保持；Day342差2.17010222251378e-15、
相对差1.4355773044550261e-9、10,248,018ULP，Day343差3.809255394568358e-16、
相对差1.9638870613581155e-10、899,435ULP。因此不称七日bitwise一致；观察/重建可能影响fusion，
关键保存NPP支路不变及数值原限通过仅形成有界轨迹准入。

Day338唯一活动/正GPP为offset5、第6个半小时、JAX零基tstep16181；Fortran kjit16182仅为时钟惯例映射。
实际humrel3.6402170609775793e-6，距1e-8阈值3.6302170609775794e-6；
swdown888.8902888737543、LAI1.2954020484751498>0.01、veget0.7262122337533891、veget_max1，
temp_growth19.24361600534081在(-4,55)内，全部活动门true。其余47步humrel0、实际活动false/GPP0。
Yin20层raw候选可正，实际8个active叶层参与积分，raw/active fallback均0；inactive raw候选不算生产正GPP。
canopy原生assimtot7.028563100383167e-5，经salinity0.5及inundation0.6012772289745216，
controlled_assimtot2.1130574723354817e-5，再按原式乘12e-6、veget_max和dt1800，
得到mesh单步GPP4.564204140244641e-7。entry逐值相同，日NPP入口也为该值且与上批0ULP。
单步单位gC m⁻²(mesh) timestep⁻¹，已经包含dt；日入口为归一化PFT日速率。
来源为`diffuco.f90::diffuco_trans_co2`2338–2384/2884–2899及`diffuco.py`2296–2372/2648–3349。
该步humrel并非仅差1ULP贴阈值，但更早扰动经分支放大仍未排除，其上游创建不在本批观测内。

真实保存end337初始累加0、48条entry和veget_cov_max1，按源代数离线推导offset5的gpp_d
为2.1908179873174275e-5，加权中间量0.0394347237717137，最终除86400得到4.564204140244641e-7，
与实测NPP入口一致。`derived-accumulator-accounting.pkl`明确仅DERIVED，不能补作原生逐步累加或宣称该链完整观测闭合。
来源`stomate.f90::stomate_main`2921–2931/3020–3032/3207–3208及`stomate_accu_*`9341–9411，
对应JAX `daily_inputs.py:171–220`、`daily.py:1432–1499`；细节见
[源码说明](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/accumulator-source-note.md)。

命名历史证据中Stage8 Day338只有下游NPP，旧JAX GPP1.2241381168386465e-6并非当前值；
Stage10完整65参DIFFUCO、Stage12水文carry主要为Day337，不能冒充当前Day338配对参考。
当前绑定stomate_history_1961.nc只有time_counter1/float32年度聚合，没有该日零/正GPP锚。
所以本批确认正值在光合生产层/entry已存在，仍不能判定翻译错误、浮点起因或年度因果。
后续最小独立参考需Fortran Day338 kjit16177..16224、全部65输入、activity/叶层/controls/GPP/累加/交付，
float64或17位roundtrip及精确离散，绑定end337/prefix及实际source/binary/输入。
历史入口和非执行命令模板见[历史证据](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/historical-evidence.md)，
Explore1000的arch/编译器/MPI/XIOS/NetCDF、输入与prefix、节点/资源/时限/费用仍未绑定，无已准入短重启；
旧qhcess资源/性能不迁作新平台事实。方案见
[下一最小事实](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/next-minimal-facts.json)，本批未执行任何远端动作。

年度NPP仍0.3860031827055333对参考0.38518497347831726，误差0.000818209227216038>
原限0.0003951849734783173；上批局部分支放大/floor0/年度会计保持其独立接受身份。
Verify图代次2026-10-26T20:35:28Z/full/complete/best_effort，JAX元数据变化、Fortran/outputs排除，精确源回退不等于全分支覆盖。
无生产/参数/容差修改、Fortran/SSH/远端/GPU/训练/付费/环境重建、commit/push、自动化或主控聊天消息。
28修复/16代码提交、42行/28ID表及支持域限制不变，无新Stage/修复ID；研究分支对齐、交接继续暂停，本批无water-origin新捕获。
交付时实际工件/两报告绑定见
[最终静态核验](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/final-verification.json)，上批manifest/回执保持冻结。
2026-10-27主控核对58份清单工件、交付报告SHA及实际结果，复用执行方验证，零科学/统计重跑。
[独立主控验收](../../outputs/reviews/main-physics-20261027/npp-gpp-origin-current/controller-acceptance.json)
接受有限生产层/唯一活动步/humrel输入与有界七日轨迹证据；完整捕获失败、未观测累加器及上游/年度未知保持。
下一重点是humrel创建和独立同phase参考，不为单独补日志重跑全年；本次仅发布报告，无新生产修复。

## 当前轨迹NPP：原生机制与年度输出会计（2026-10-27）

本批固定main `84e1d8d71dd2110f35d36b525881e70a74a68f61`、`E:/ORCHIDEE-MAN-JAX`实际导入，
ORCJAX/JAX0.10.1 CPU/x64、Python-B/无持久缓存。原319点1961输入/15字段/全部开关复用，
run.def未重新物化，仍cold1+52×7；新年度worker PID57056，区别于V02 PID53212。
实际**1整模型调用/365日、2原NPP组件调用**；年度调用及保存245.062秒，监督261.016796秒/900秒硬限，
独立组件worker两调用合计监督3.257107秒/120秒硬限；系统TEMP已清理，无额外baseline/warmup。
完整原始年度返回先保存，再按V01接受的定义域与原`atol1e-8/rtol1e-10`作独立准入。

| 原限内自准入范围 | 最大绝对差 | 失败数 |
|---|---:|---:|
| 完整年度返回 / 全365日完整daily对象 | 2.7755575615628914e-17 | 0 |
| day365完整末packet | 0 | 0 |
| 当前Day338不变输入原组件六返回 / 546元素 | 1.9852334701272664e-23 | 0 |

容器/键/shape/dtype/离散及来源仍精确，数值载体类不要求相同；两处命名roughheight未定义NaN合同、
260独立源码字段分类/92额外字段覆盖缺口保留。观察可能改变fusion，本结果只证明原限内轨迹一致。
可逆源码overlay保存365日完整九参、原生分支/pump/reserve/pre-floor/negative/bm_create、
原floor操作数及六返回，算术AST逆变换精确相等。动态scan日号进入callback，准确1..365，
year1961、日末tstep=day×48−1；原static_daily_carbon请求未变，later-block自身原outer dispatch路线未改。
原executable不经代理，实际end-Day330/337完整packet、九家族、context及pcapa_en已保存，
day365完整packet在年度对象中。新tuple保留新producer身份，不回填为旧V02原捕获。

当前Day338走over_tax：GPP4.564204140244641e-7，原生pre/post-maintenance均0.3155353646470735，
growth2.5559543185369993e-8、NPP−0.3155349337862027；**原生floor修正0**。
pre-floor最小9.998478064767094e-9仍为正，源只修正负值。完整控制通过后仅gpp[0,13]置0，
另八参和其余GPP坐标逐值/shape/dtype冻结；NPP变2.7977620220553945e-13、
maintenance变−2.7977620220553945e-13，NPP有符号响应+0.31553493378648245。
当前完整背景的局部放大已证；反事实内部floor未直接捕获，不能搬用原生floor0。

全年enough302日、over_tax51日、no_alloc12日；11日floor非零且均在no_alloc。
原生`NPP=GPP−growth−pre_maintenance+原floor-rate`残差最大2.220446049250313e-16，
原生NPP对每日输出最大差0。**年度指标为365个日速率的算术平均，单位g C m⁻² day⁻¹**。
floor之和3.4274549371626826、日均0.009390287499075842；固定原操作数扣除floor项的
NPP日均0.37661289520645747仅为代数会计，不是关闭floor的年度反事实，也不是相对参考误差。

输出层Δ取当前JAX−既有绑定参考：GPP−0.00003886349333881434，维护−0.0008469903396264022，
生长−0.000010082380928505552；两呼吸合计−0.0008570727205549078，NPP+0.000818209227216038。
`ΔNPP=ΔGPP−Δmaintenance−Δgrowth`闭合残差−5.551115123125783e-17。
因此年度NPP偏高在报告通量层对应较低维护/生长呼吸，部分被较低GPP抵消；
没有全年配对Fortran原生floor/分支记录，尚不能向floor差额、上游浮点起因或某fix分配因果。
新四指标与V02相同，AGB/BGB/GPP过原限；NPP0.3860031827055333对参考0.38518497347831726，
误差0.000818209227216038>原限0.0003951849734783173，年度科学失败及严格诊断false保留。

详见[当前README](../../outputs/reviews/main-physics-20261027/npp-current-capture/README.md)、
[年度输出差额](../../outputs/reviews/main-physics-20261027/npp-current-capture/annual-output-accounting.json)、
[剩余最小事实](../../outputs/reviews/main-physics-20261027/npp-current-capture/next-minimal-facts.json)
及[main报告当前末节](main_physics_review.md)。若后续继续，end-Day337是实际可复用边界，
但仍需独立参考producer的同phase/单位/足够精度的上游输入输出；历史tuple不自动证明当前起因。
本批科学预算已用尽，未启动后续。工具分阶段版本和首次静态失败保留，不改原始receipt。
Verify图2026-10-26T20:35:28Z为best_effort，JAX元数据变化、Fortran/outputs排除，精确源回退不等于全分支覆盖。
无新生产/参数/容差修改、Fortran/远端/GPU/训练/付费/环境重建，执行方未commit/push。
28修复/16代码提交、42行/28ID表及原支持域不变，无新Stage27或修复ID；
water-origin新捕获、两研究分支对齐和交接继续暂停。实际工件/两报告/固定HEAD/工作区绑定见
[最终静态核验](../../outputs/reviews/main-physics-20261027/npp-current-capture/final-verification.json)，保留交付时的待主控状态。
2026-10-27项目主控已核对64份清单工件、交付报告SHA及实际结果，复用执行方验证，零模型/组件/统计重跑。
[独立主控验收](../../outputs/reviews/main-physics-20261027/npp-current-capture/controller-acceptance.json)
接受本批有限机制证据和输出差额核算，不将其提升为上游浮点或年度因果结论；年度失败与28项/16提交不变。
本轮版本化发布仅含两份报告，原清单和回执保持冻结。

## NPP历史局部分支机制：已由主控验收，年度起因保留（2026-10-27）

本节保留上批验收时的缺项和344日原建议；其中“未保存/未实现/未执行”是该批历史状态。
后续365日捕获及当前Day338结果见上一节，不能与历史Stage8输入或原回执合并。

本轮从固定main `2c15d4b0d76af6333084c0a8e93a33369da4b8a6`复用原计划和完整历史载荷，
按主控明确授权完成恰好两次`npp_closed_update`。这是Stage8 Day338的两个实际完整背景，
仅交换PFT14 `gpp[0,13]`的0与1.2241381168386465e-6；另八参及其余GPP坐标逐值/shape/dtype冻结。
两个历史producer、时钟F16224/J零基16223与当前V02分别保留，不能把干预称当前轨迹或新Fortran参考。
六真实返回及完整输入见[本轮交付](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/README.md)，
连续数值/残差/原限见[机器结果](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/response-summary.json)；
上批细表和当时四日缺项见[main报告的NPP历史节](main_physics_review.md)。项目主控已接受该批局部分支敏感性证据，
保留上游浮点起因和当前年度归因未知，见[最终验收](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/controller-acceptance.json)。

**小绝对GPP变化触发约0.315的局部响应已证；上游差异来自浮点舍入尚未证明。**
下表为PFT14、单位`g C m⁻² day⁻¹`；残差取干预减另一实际背景原值。

| 冻结背景及干预 | NPP原值→干预值 / 有符号Δ | 维护呼吸干预值 | 对方原值残差 / 原限 | 两主要输出严格复现 |
|---|---|---|---|---|
| F：GPP0→1.2241381168386465e-6 | 9.999999994736442e-9→−0.31504082549082607 / −0.31504083549082607 | 0.3150419810772084 | NPP +0.0004938172603840552 / 1.0031553464275122e-8；维护 −0.0004938172603840552 / 1.003155357983376e-8 | 否，已大幅响应但仍有背景剩余 |
| J：GPP1.2241381168386465e-6→0 | −0.3155346427512101→2.7977620220553945e-13 / +0.3155346427514899 | −2.7977620220553945e-13 | NPP −9.999720218534236e-9、维护 +9.999720218534236e-9 / 各1.0000000001e-8 | 是，接近原限；非exact或六返回全相同 |

规则沿用`atol1e-8/rtol1e-10`，不以“更近”代替复现，不设置新阈值。F原维护−9.999999994736442e-9，
J原维护0.31553579833759243；维护有符号Δ分别+0.31504199107720837、−0.3155357983378722。
两次大响应与只一侧严格复现并存；两侧六返回全坐标比较均未全部通过，见[完整分析](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/mechanism-analysis.json)。
不能否认分支放大，也不能把全部原生差额归给GPP；无年度解释比例。

源锚为`stomate_npp.f90::npp_calc`314–387/449–531与`carbon_kernels.py::npp_closed_update`3954–4057。
由实际完整输入和源码谓词确定，零GPP走`no_alloc`储备扣除，正GPP走`over_tax`的tax/多组织pump，
`enough`均false。源码随后对负库存补至min并从维护呼吸扣创建量，因此零/正扣库路径可放大最终净通量。
直接新增观测仅六返回，反事实pre-floor/negative/bm_create未直接输出；不从post-maint反推为实测floor。
整个carbon_kernels.py的SHA在Stage8、V02与当前均为
`56516c8690ab9f656f940c106871a50beb5531f5ea881cc8d1280d0b67a5c1e1`，原18/18同输入基线复用，未重放。

| 证据类别 | 有界结论与不能跨越的限制 |
|---|---|
| 绝对量级 / 小分母 | 当前年度reference约.385且绝对误差超原限，不是近零分母假失败；小GPP绝对量不证明其为舍入噪声 |
| 输出采样 / 聚合 / 精度 | Stage8所选core→send36/36及同输入18/18，Stage10原NetCDF高精度恢复34/34；三项float32严格诊断失败保留。不能据此排除内部数值敏感性 |
| 内部阈值 / 输入状态 | 新两次干预证明局部分支敏感；旧液水跨阈值/创建读取48桥及DIFFUCO同输入通过只约束旧窗口，未确定当前上游误差起点 |
| 已确认源问题 / 因果 | 28修复/16代码提交及接受域保留；没有新NPP同输入源级反例，也未把任何修复或局部通过认定为年度原因 |

当前V02 Day338实际GPP4.564204140244641e-7、NPP−0.3155349337862027，区别于上述历史J输入。
此前仅读取337/338/339/343四个保存日的通量，零/正脉冲形态仍在；只有day365/tstep17519终态，
缺day336完整状态、逐日pre-NPP九参/floor/native水文tuple。当前年度NPP0.3860031827055333对参考
0.38518497347831726，误差0.000818209227216038>原限0.0003951849734783173，科学与严格诊断失败不变。
当前年度最多将分支敏感性登记为有局部因果证据的候选机制，尚不能定类为已证浮点数值起因，也不预设有bug或无bug。

下一方案仅交主控：使用与V02相同生产源码/输入并记录新producer，在Day337–339直接保存完整九参/六返回/clock及所需源锚定floor局部；
须对保存V02同窗输出按原限自准入，不能把新生成tuple改称V02原捕获。
无合格day336状态则须另获准从cold-start演化；V02的337是块尾、338–344同一7日块，公开ndays339会切到tail。
因此最短保持原7块路线的是prefix344、仍仅观测337–339；块内安全停止及新capture runner尚未实现/准入。
不使用day365或历史旧tuple代替。[最小合同](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/next-observation-plan.json)
列实际API/字段/源码锚，未来预算提案1次本地CPU prefix344/900秒、0额外baseline/warmup；没有现成捕获命令或本轮新授权。
配对参考须核定输入/控制/producer；
上游舍入起因另需完整相同原生输入及独立源输出/误差事实，旧不同液水入口不能替代。
年度定因还需按原单位/聚合闭合该机制解释的当前差额及其余全年残差，本轮不重算旧表。未启动新捕获，water-origin仍暂停。
本次一个science子进程加监督进程，120秒硬限内exit0、CPU/x64/Python-B/持久缓存关闭、系统TEMP清理；
0baseline/warmup/整模型/新参考调用，无显式JIT wrapper或整模型图，仅两调用固有JAX算子。
原计划目录保留“未执行、组件0”的历史回执，新授权执行另目录保存；无生产/参数/容差修改、远端/GPU/训练/付费/环境重建。
Verify图代次2026-10-26T20:35:28Z，JAX元数据变化、Fortran排除，机制依赖精确源回退，不声称全分支覆盖。
本轮有限交付到两调用/真实结果/机制边界/缺项与后续条件/两报告核验齐全为止；raw NetCDF重启容量和跨年连续性
仍待独立验证，不展开669点或长期矩阵。28修复/16代码提交、42行/28ID表保持，无新修复编号；本次仅发布报告。
C4–C28同步仍是历史待核对登记，两研究分支对齐及交接包继续暂停；
最终工件/两报告/工作区与修复登记绑定见[静态核验](../../outputs/reviews/main-physics-20261027/npp-gpp-swap-approved/static-verification.json)。
主控核对21份新工件、实际调用及分析，复用既有验证，新增组件/统计重跑为0。上述344日提案保留为执行方
原建议；后续派发改为一次原365日观测，以同时保存全年NPP机制并与V02年末完整状态核对。
最多一次本地CPU积分及两次当前输入小组件干预，不改生产或容差、不扩上游水文/Fortran/远端捕获。

## Stage26 接受修复与接入建议（C21–C28）

基线main `61c6b3170419bb2c5d22eab80f98bfe314e9fa74`；2026-10-26主控已核对
S26-R1实际修改、独立修前及最终132项JUnit/log、20个被测文件与11个交付文件身份。
复用既有A/B/C/D审查及验证，无主控数值重跑。以下8项加入C21–C28；R1属于C25，
不重复编号。代码/测试按三组提交，报告接受登记单独提交，未修改两研究工作树。
C21–C25为`c214e43da57c21b68002186fbf71c67d45835bae`，C26为
`84b5d425418a6d1b18fa6dbe080348c2faa72e13`，C27–C28为
`9fc87c011a4e991f1637223b0f8fb2a224000595`。

| 分组 / 临时发现 | 接入依赖及证据边界 |
|---|---|
| C21–C24：A-MIN_WIND / A-HEIGHT_DISPLACEMENT / A-Z0_ICE / A-SNOW_DENSITY（按此顺序） | 各源owner/gate/default已独立核对；condveg有效API/main/init/wrapper、orchestration cold调用及两cache。基线已有C15/C20。A新60通过，其中实际组件/调用表达式与构造stub区分，不扩原reject/init模式或全bucket生命周期 |
| C25：S26-A-FRAC-CARRY（含S26-R1） | 与参数组一起接入writer→mapper→assembler及cold fraction；S26-R1补齐restart scaffold从成功HYDROL的post雪生成fraction，避免旧fraction与新雪混合。必需restart变量不变。18新项=原7+补项11，含实际call/真实组件及writer/packet；完整compiled/restart跨点仍V01 |
| C26：S26-B-TMC-DEFAULT / P4-17 | hydrol_tmc_update只恢复源allowed_err2e-8及说明，显式check API默认修复，未改算法/override/check=false。B9新+14旧通过；P4-02/03/04未支持check/fatal owner，P4-05条件不变量与静态guard缺口有界保留，无新支持入口 |
| C27：S26-C-PEAT-ACCUM | 既有显式组件的连续buffer/非peat reset/累积回加；不开放多peat生产。当前生产仍单PFT14 |
| C28：S26-C-PEAT-OLT | helper/core源默认1e-8，仅诊断；显式override与库存/转移保留，其他min参数及C10不变。C8新+4直接旧通过，另选C10源顺序8控制通过；两C项共用新测试文件并一组提交 |

[提交分组/hunk依赖](../../outputs/reviews/main-physics-20261026/final-source-boundaries/submission-suggestions.md)
和[当前工件核验](../../outputs/reviews/main-physics-20261026/final-source-boundaries/verification/delivery-verification.json)
绑定五个生产文件、四个新测试及两报告。补项后最终单次132通过，0fail/error/skip；
早期错误fraction参考的117pass/3fixturefail、正确参考下原源码重放、120pass中间版本
及writer-before有效失败完整保留，不混作共同baseline或最终writer证据。
原121及其manifest/被测源与测试快照保留；R1独立before为4有效fail/7pass，
最终132保留原121全部ID，18个旧被测文件不变，仅公共orchestration/carry测试更新。
该补项只改实际restart CONDVEG实参的六雪字段覆盖，缺state/显式几何边界保留；
是原carry接入闭合，不另计独立修复或接受C编号。
Stage25整93不重跑；测试数不代表全模型/全分支覆盖。

raw-restart容量以唯一源初始化pcapa_en/producer身份缺项停止，不称源未定义；
雪349true/355false/360true以入口质量不变量缺项停止，360false局部逻辑保留，无新雪补丁。
[最终M01–M12/V01–V03矩阵](../../outputs/reviews/main-physics-20261026/final-source-boundaries/final-matrix.md)
与本次主控验收一起结束有限源码审查；不把有限未知改成全配置已验证。

原[V01准备方案](../../outputs/reviews/main-physics-20261026/final-source-boundaries/next-validation-plan.md)
保留历史未执行身份。主控发布后已另行授权实际V01，生产者固定
`d5d487f101ac49c2b933c91f1cb08d42bb473bfd`，仅本地ORCJAX CPU/001点1961。
[实际汇总](../../outputs/validation/v01-after-stage26/001-api-9day/summary.json)为一次模型调用、
8个模型日：cold8/block7完整返回，8个48步日界、无missing/stopped、末tstep383通过，
每日输出264个数值叶/3384个元素全部有限。末packet扫描353个数值叶/23994个元素，
diffuco_previous_step_state及sechiba_finalize_state的`float64[1,14] roughheight_pft`
各有一个`[0,0]=NaN`，无Inf；相同坐标/值的组件条目不计为两项独立缺陷，不作mask豁免。
**原始尝试的V01独立有限值门槛未通过**；当时第9日续跑/恢复、普通cold9均未运行，连续性及普通/编译
误差尚未得到，原atol1e-8/rtol1e-10未改。后续模型停止，交主控判断，不自动启动V02/V03。

首段205.9292175秒含编译/同步，失败尝试208.6074725秒含导入/预检至失败，不能称完成
19个模型日或稳态吞吐。完整结果、daily、packet/provenance、实际有限容量和原traceback
已保存；离线0/0存储最大绝对误差0不能覆盖有限值失败。运行前后生产/输入/runner身份
一致，系统TEMP已清理，无生产修改或模型重跑。raw NetCDF容量缺项、原319年度失败、
TOPMODEL/fwet及原支持域限制、研究同步待核对均保留；没有年度改善/归因或raw重启准入声明。

后续调查未新增整模型积分或生产修改。实际NaN为001点/PFT1 bare ground，保存末状态
height/veget/veget_max/lai均0；冻结IMPOSE_AZE=false、ROUGH_DYN=true、
HEIGHT_DISPLACEMENT=.66、全部OK_LAIDEV=false。Fortran condveg.f90静态1467–1472/
动态1705–1717只赋值PFT2..nvm，INTENT(out)的PFT1未定义；JAX动态/静态初始化NaN、
循环从索引1开始，故此处保留显式sentinel，不能称源内存必为NaN。该写法来自仓库根
基线`7333b46c0b38650fb6c9250582876137831657b8`，Stage26前已存在；消费端safe mask来自
`5910ba9af857a91e565d13a25e3de2e3019080d1`。末边界保存值不能确定首次NaN日/步。

[调查入口及复用建议](../../outputs/validation/v01-after-stage26/nonfinite-investigation/README.md)
核对producer/writer/两packet及ordinary、fast-layout、恢复mapper；DIFFUCO的
`OK_LAIDEV & (index>=1)`在log/乘除/sqrt之前选择安全输入，grid标量roughheight仍有效。
11次局部函数调用（含窄CPU JIT封装2次调用）中，三条运输/消费路线NaN0与finite123
对照11叶/63元素均有限、maxabs0；反事实启用PFT14的守卫控制仍通过，启用列NaN
阳性控制使3个输出数组非有限。后者是构造组件控制，不是冻结配置的科学准入；没有
完整ordinary/compiled/restore、所有配置或AD结论，也未确定原历史producer实参。

主控正式接受的合同初始仅限本次001/1961/d5d487f、legacy14/NVM14、
IMPOSE_AZE=false/ROUGH_DYN=true的实际动态重算producer；静态控制仅机制证据。
只检查两处命名roughheight_pft的[0,0]未定义域；保留预期raw NaN、不豁免Inf，
全部已定义坐标及其他字段仍严格finite。imposed分支
定义全部PFT，不可通用豁免PFT1或inactive字段。未证明新生产缺陷；本批已另建验证器，
32窄控制及首段离线重评通过，随后按授权开始余下阶段，未重跑source8。纯验证合同改变
用另版门槛对未变物理的旧首段结果生成派生回执，保留
旧producer和原failed；若改物理，受影响状态/输出须新producer重生成，旧packet仅诊断。
原17工件/索引SHA保持，新调查独立manifest、TEMP已清理；图服务本次Transport closed，
结论依赖精确源码回退。两条第9日路线都在原进程退出后加载，须分别称完整结果重新加载续跑
与独立packet文件恢复续跑，不宣称同进程8+1边界。C1–C28仍28项/16提交，不新增C编号。

domain-v2当时只完成完整结果重新加载后的day9：tstep384–431/48步，104.8217936秒含编译/同步，
完整结果/daily/packet/动态容量先同步保存，随后packet schema/provenance门槛失败。
[只读差异](../../outputs/validation/v01-after-stage26/domain-v2/schema-failure-inspection.json)
显示数值字段shape/dtype无差异，daily33叶/423元素严格finite，packet353叶/23994元素
仅保留原两NaN；唯一结构差异为slowproc_stomate_previous_step_state的provenance从7条变6条，
缺`stomate.f90 lines 3288-3489 half-hour OK_LEAK state writeback`，其余6条完全相同。
生产记录生成位置为orchestration.py:2775的2条base、8213–8239的1条half-hour、9275–9333
的4条day-end；尚不能由记录差异推断物理漏过程或科学数值错误。当时domain-v2按seed精确比较而显式失败，
故保留[失败回执](../../outputs/validation/v01-after-stage26/domain-v2/execution-001/failure-48208.json)，
未改验证器、生产或容差放行。在该历史停止点，独立packet恢复day9、ordinary cold9及两组路径比较未运行。
domain-v2新增1模型进程/1调用/1日；当时连同旧source8累计2模型进程/2调用/9日，回归/离线不计模型。

主控现已验收[domain-v3最终结果](../../outputs/validation/v01-after-stage26/domain-v3/execution-001/summary.json)。
13个半小时字段经源顺序选择并覆盖写回，17组实际组装控制/221字段断言和8路线检查通过；
这些是标记值组装证据，不是221个独立科学试验。编译块7条与later/ordinary6条来源分别由
真实helper生成并精确核验，未全局忽略来源；12项新增拒绝控制通过，原32控制按身份复用。
原两处roughheight未定义坐标单列，其余状态/每日输出有限，原数值容差不变。

实际复用source8及旧重载day9，只新增独立packet恢复day9和ordinary cold9，共10模型日；
累计19模型日/4调用/3进程。恢复day9的完整packet含来源和每日输出最大差异0；普通/编译
1–9日及最终状态通过atol1e-8/rtol1e-10。每日输出maxabs2.3125945602942011e-13，
全状态maxabs2.5588087737560272e-7位于大尺度pcapa_en且在原相对容差内；maxrel
3.9964615879614042e-12。两种恢复都经过跨进程加载，不认证未执行的同进程8+1或raw NetCDF。
主控核对实际源码/验证器/回执及46新工件、13运行源/7取证源身份，无数值重跑。
两次历史失败、旧producer及存储原值保留，无生产代码/参数修改、无新修复ID。
V01只接受本次短窗路径与动态恢复；V01批次未执行V02固定319全年或V03跨点/长期。
随后另获授权的V02实际结果见下节，不回写V01历史回执。

参数及fraction carry影响非默认值和运行state字段；今后相应维护线程需按拟支持用途
评估compiled/restart/Teacher消费者兼容性，不能用文本合并代替数值准入或整文件覆盖。
C4–C28同步仍待维护线程核对，既有Teacher/prepared/restart/reference producer与执行版本
分别保留。本批没有同步授权/操作。319仍failed，water-origin暂停；V01–V03、C8 tiny/
负值fatal/nonfinite、C10源负库存、TOPMODEL/fwet未准入边界保留，无年度归因。

### V02实际结果：319点全年完成，NPP科学门槛仍失败

V01验收发布后另获本批授权，固定main `876ac3bb8c49469ae9d9a9e0aff5982a1e4ebd64`
（相对V01仅报告变化，生产源码不变），原319.0-057.0/I5/S26_69.834_0.0019_0.3980_97.918、
1961 cold-start365日、compiled block7和原15字段/开关完整执行一次。
[V02交付](../../outputs/validation/v02-after-stage26/319-1961/README.md)、
[执行回执](../../outputs/validation/v02-after-stage26/319-1961/execution-receipt.json)及
[四年度指标/完整离线统计](../../outputs/validation/v02-after-stage26/319-1961/summary.json)记录真实结果；
旧next-validation-plan仍为历史方案，未回写成执行回执。

实际ORCJAX/本工作区包/CPU进程PID53212，Python -B、持久编译缓存关闭；原CLI入口、
config SHA3e3ec5c4518ebf4d470c60eda3e9bd6ddfd22e5c6dee473694f2409f17e619af及
319-current/inputs/used_run.def SHA5ecb4d2615dec4d54a88f2db5444ac2820d8327ec6c10c6af16945b4ff56374a保留。
仅改输出/checkpoint目录，未重新物化run.def。1模型进程/1调用/365模型日，无年度重试；
365个48步日界、末tstep17519、无missing/stopped、15字段各365个有限float64(1,14)输出通过。
runner264.5704782秒含编译/同步，CLI进程281.3368429秒另含导入/完整对象保存/汇总；不作纯吞吐对照。
两条XLA simplifier警告保留，实际exit0。年度完整run精确回读0/0、原年汇总离线exact replay、
checkpoint完整packet含来源0/0一致。pickle tuple/JSON list仅规范化元数据，packet门槛不放宽。

实际319 legacy14/NVM14及动态/non-imposed配置支持两处roughheight_pft[0,0]的原未定义NaN；
其余定义坐标/每日输出/capacity有限。365=1+52×7的真实完整块路线由源码helper生成来源合同并精确比较，
未机械照搬001点元数据。源码必需合同及260数值字段shape/dtype通过；另92个额外字段的
独立源码shape合同未完整覆盖，逐项列出并保持有限域/完整存储检查，不声称全部packet结构认证。

AGB/BGB/GPP的新JAX依次为4.591235866253038、4.120666562048175、0.7776203738914573，
年度误差依次0.000320663158555945、0.000205098119952396、0.0000388634933388143，
均通过原`1e-5+1e-3*abs(reference)`及normalized bias/RMSE限1e-3。
**NPP=0.3860031827055333，参考0.38518497347831726，绝对误差0.000818209227216038
仍大于原限0.0003951849734783173**；normalized bias/RMSE均0.0021241981997049434，失败。
因此全年完整但科学接受false；单年度统计只含一对值，不代表逐日表现。诊断atol2e-6/rtol1e-7另列且false，
不代替科学门槛，全部字段/诊断已完成统计。

原319-current producer09968cf8及同一参考/输入身份保留；原NPP失败
`0.0008192662350707258 > 0.0003951849734783173`和旧payload直读误差末位差另列，未回写。
新NPP比旧JAX低约1.05701e-6，仍未闭合年度偏差。旧CLI只有摘要；辅助逐日比较使用有独立身份的
Stage6 JAX producer c7110e03，四通量最大日变化分别为GPP2.63996e-4/day218、NPP1.89971e-4/day218、
MAINT_RESP6.23768e-7/day364和GROWTH_RESP7.38774e-5/day218，原字段单位不变。
其窗口统计/相关性不构成修复因果，历史Fortran未补称本次producer。科学失败后继续完成离线交付，未第二次积分。

本次无生产修复/参数/容差修改，28项/16代码提交及C4–C28待同步状态不变；
两份报告更新。执行前后有限源/config/run.def/输入及历史身份一致，
系统TEMP清理；图Transport closed下用精确源回退。V03/跨点/长期、raw NetCDF容量、TOPMODEL/fwet/
支持域和water-origin暂停限制保留，未检查或修改两个研究工作树。
本批[最终静态核验](../../outputs/validation/v02-after-stage26/319-1961/final-verification.json)
记录保存工件/身份/报告链接与42行、28个修复ID原登记；无模型或统计重算。
2026-10-27主控验收本批交付，复核33份新工件与15份当前源码身份，复用13项离线检查，
未重复运行或重算统计；[主控验收](../../outputs/validation/v02-after-stage26/primary-v02-acceptance.json)
保留年度科学接受false、严格诊断false及260/92字段合同边界。该验收没有新增已接受修复，
也没有解除319科学失败、raw重启、跨点/长期或研究分支待同步限制；原交付回执保持历史身份。

### V03a实际结果：跨点复用及继承数值合同通过，额外载体检查失败保留

V02验收发布后另获授权执行本地ORCJAX/CPU批次，固定main
`a317c77750f7f2f42db21edaf05e6fa7f7a0c3a2`；生产树与V01 `d5d487f`、V02 `876ac3bb`相同。
一次预检核定真实root包/ORCJAX Python/cpu:0及47个已有输入元数据路径，原manifest和seed保持身份。
原config SHA `3e3ec5c4518ebf4d470c60eda3e9bd6ddfd22e5c6dee473694f2409f17e619af`不变；
A为001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658，used-file SHA
`f75d64dcbe321a0934368a415f1820687e00d5e6e0829f2c808bd2313e5343c2`，实际W/E/S/N −180/−178/−20/−18；
B为原319/I5/S26_69.834_0.0019_0.3980_97.918，used-file SHA
`5ecb4d2615dec4d54a88f2db5444ac2820d8327ec6c10c6af16945b4ff56374a`，实际138/140/−34/−32。
实际域、PFT14五校准值及所有legacy14/NVM14/NSTM6/CONDVEG开关见
[完整交付](../../outputs/validation/v03a-after-stage26/README.md)及[边界观测](../../outputs/validation/v03a-after-stage26/shared/cache-observations.json)，
未用YAML示例域，未重新物化run.def。

公共cold-start API的真实进程PID58044顺序运行独立cold A8→B8→A8，每次cold day1+真实block7；
原进程中不清cache、不reload。随后独立PID59312执行ordinary B8，compiled off/block0，
其他common flags含static daily carbon true与接受的V01普通路线一致。两进程顺序而非并发。
四次调用全部完成8个48步日界、末tstep383、无missing/stopped；累计4调用/32模型日，
完整run/packet/实际pcapa_en先保存并精确回读再进入下一模型调用，没有重试。
四调用计时176.7734517/24.1930783/12.1478658/107.1586996秒含编译、同步及边界保存；
外部进程286.9264433/156.1406413秒另含导入、验证及写盘，不作稳态吞吐声明。

两个A各自对接受的V01 source8及互相比对均完整exact0；B-compiled与V02前8个完整daily差0，
B-ordinary与同一daily最大绝对差3.5271785492341223e-13、相对差8.150827422816209e-13，均通过。
V02没有独立day8末packet，本批没有补造。两条新B末packet则均已真实保存。
本次B compiled/ordinary额外载体检查拒绝44个叶类型差异：43个JAX ArrayImpl对NumPy ndarray、
1个ArrayImpl对Python float；普通模型已返回且保存，exit1来自保存后的比较断言。
原[summary](../../outputs/validation/v03a-after-stage26/summary.json)、
[gate/failure](../../outputs/validation/v03a-after-stage26/ordinary/failure.json)和冻结工具均不改写，
**本次额外strict carrier false永久保留**，不伪造模型重跑通过。它不是接受的V01原科学门槛。

V01实际合同由`domain-v3/domain_verifier.py::schema`的23–36行与
`run_remaining_v01.py::execute`的118–133行实际调用的`ComparisonAccumulator.compare`（113–187行）组成：schema将数值叶用
np.asarray记录原shape/dtype，再作数值parity，不要求数值载体类相同。历史比较器自身的float64
误差算术不能独立证明dtype精确；本次仍保留完整键/容器、shape/dtype、离散精确及原浮点限。
V03a冻结`validation.py::compare`的188行在数值前增加无条件type相同条件，才产生上述44个载体失败。
公共packet/daily/lite协议的payload为object，未固定数值载体类；原pickle保留实际对象。
compiled直接重建与ordinary合并/float(dt_days)的解释仅限已核对边界，不证明44叶完整上游历史或新的物理bug。

已有[完整数值比较](../../outputs/validation/v03a-after-stage26/numeric-diagnostic-v1.json)未指定dtype地
用np.asarray读取数值叶，不改变保存数组或其dtype；全部620数值叶/27381元素及原44个跳过叶
均在原atol1e-8/rtol1e-10组合限内，数值超限0，6项负控制通过。因此按已接受V01数值状态合同，
本批跨路比较结果为通过；额外载体条件仍false。仅等价这43个ArrayImpl/ndarray和1个ArrayImpl/float
实际数值关系，不扩展为其他类型问题通用豁免，不通用忽略metadata或放宽浮点限。
最大绝对差为pcapa_en[0,2,0]的6.644986569881439e-7：compiled1127009.884473888对ordinary
1127009.8844732235，原限0.00011271098844732235；最大相对差为wat_flux[0,7,3]的
2.404596709685098e-10，原绝对组合限1.0000060882007604e-8。原值、参考、字段/坐标和allowance
逐项保存在回执。本次合同收尾复用已有完整结果、诊断和控制，无新模型/组件调用或数值重算，
额外类型检查的原失败及全部manifest身份保持不变。

每个实际对象先独立核定定义域；仅两个命名float64[1,14] roughheight_pft的精确[0,0]保留
源未定义NaN，其余定义值/daily/动态capacity有限。同路/存储来源精确；compiled末slowproc7条与
ordinary6条分别按精确源路线核验，不把合法来源差异当物理漏过程或全面忽略metadata。
源码shape/dtype合同仍为260字段已分类/92额外字段未完整独立覆盖；派生成对全字段shape/dtype
检查不消除此来源合同缺口，更不认证raw NetCDF capacity。

真实prepared LRU为A miss、B miss、回A hit；A context首/回相同、B独立。实际outer key相同，
同一executable id2183757175440由A首次创建，B和回A均按`is`命中原cache对象、cache增量0；
三个边界保存真实caller的九类动态输入及实际域/校准值并0/0回读。原函数实参、返回对象与数值图
保持原样。仅A-first出现一次inner factory边界，另两次外层命中可跳过它；不据此声称新的inner命中
或XLA编译/重跟踪次数，也不把历史Stage16替身869断言升级为真实XLA证据。

此前[primary-v03a-acceptance](../../outputs/validation/v03a-after-stage26/primary-v03a-acceptance.json)
由本执行聊天生成，是执行方内部审查，生成时项目主控尚未最终验收；该文件和身份保留。
2026-10-27项目主控已接受本次有限缓存复用、绑定和沿用V01数值状态合同的完整比较。见
[项目主控最终验收](../../outputs/validation/v03a-after-stage26/controller-final-acceptance.json)、
[合同与交付收尾](../../outputs/validation/v03a-after-stage26/contract-closure-v1/README.md)及
[主控补项审查](../../outputs/validation/v03a-after-stage26/controller-review-20261027.json)。主控复用已完成
73份工件/24份源码身份核验及执行方检查，核对新增收尾和合同来源，不重跑模型或数值统计。
44项额外载体检查原失败永久保留；原
[静态核验](../../outputs/validation/v03a-after-stage26/final-verification.json)保留当时报告身份，新核验单独保存。
系统TEMP已清理，图Transport closed使用精确源回退。本批只发布两份报告，无新增已确认生产问题/修复、新C编号或两个研究工作树同步，
42个有界登记行/28个ID、28项接受修复/16个代码提交不变。
V01原失败、V02年度NPP/诊断失败、raw重启、TOPMODEL/fwet/支持域、water-origin暂停及
C4–C28研究待同步均保留；本次两点1961八日不能宣称669点、年度/长期科学通过或修复因果已建立。

初始核查日期：2026-10-11。当时结论：共同物理模型只需维护一套；研究中的共享代码
差异可以拆成三项通用修复、一个可选 Teacher 接口和两项批处理运行能力。
初始核查完成分类、证据复核与轻量测试。C1、C2、C3 已分别以独立提交
`db4dd8f`、`e317fa8`、`2a3e303` 接入主分支并通过针对性回归。日尺度已有
这三项同等实现；全球效率线已有 C1，仍需由其维护线程同步 C2、C3。
可选接口和批处理能力继续按下述范围分别评估。

## 初始核查基线

| 工作线 | 核查提交 | 固定目录 |
| --- | --- | --- |
| main | `7cae3e02adad23674767a590fdf4d27f239ce065` | `E:/ORCHIDEE-MAN-JAX` |
| 日尺度 | `7463642eb3ff20f2352bc5732e9904b5181f11ff` | 根目录下 `runtime/worktrees/daily-coarse-graining` |
| 全球效率 | `52326bd5e67629269a4a4a1c39885276eefd5ec8` | 根目录下 `runtime/worktrees/global-throughput` |

以三个提交的文件内容直接比较，`jax_orchidee/` 的差异只有以下四个路径。
初始 main 基线最近的物理源码提交为 `5910ba9`，其后至该基线的提交是管理文档更新。
研究工作树中其他任务的未提交修改不属于本核查基线。

| 路径 | 日尺度相对 main | 全球效率相对 main |
| --- | --- | --- |
| `jax_orchidee/driver/orchestration.py` | +4 行：可选通量捕捉与回调 | +63 / -23 行：提取可转换的日块函数 |
| `jax_orchidee/stomate/carbon_kernels.py` | +36 / -1 行：两项数值 / 梯度修复 | 相同 |
| `jax_orchidee/sechiba/diffuco.py` | 相同 | +7 / -1 行：缓存修复 |
| `jax_orchidee/driver/global_batch.py` | 不存在 | 新增 325 行：批处理研究适配器 |

其他差异包括研究脚本、测试和研究数据合同。日尺度还增加 SciPy 依赖、研究
配置 / manifests，并更新 manifests 导航；这些不等于共同物理公式发生变化。
既有生产配置和数据合同文件、`uv.lock` 没有出现在此次差异中。
提交、逐文件统计和配置差异记录在
[inventory.json](../../outputs/reviews/shared-core-20261011/inventory.json)。

## 逐项结论与接入顺序

| 编号 | 改动 | 应归属 | 本轮结论 |
| --- | --- | --- | --- |
| C1 | DIFFUCO PFT 掩码改为缓存不可变 NumPy 数组 | 通用执行正确性修复 | main `db4dd8f`；日尺度 `120c637` 已同步且 14 项通过；全球已有同等实现 |
| C2 | NPP 维持呼吸份额使用已有稳定比值原语 | 通用物理核 AD 修复 | main `e317fa8`，12 项通过；保留极小分母导数策略；日尺度已有，全球待同步 |
| C3 | 叶龄周转份额消去共同叶质量因子 | 通用物理核数值修复 | main `2a3e303`，20 项通过；保留原测试、适用范围和源身份；日尺度已有，全球待同步 |
| C4 | restart tau回退复用天数上限，并传递当前季节配置 | 通用状态读取正确性修复 | main `4b92d3d`，31项相关测试及1项默认入口补查通过；日尺度/全球均待核对同步（2026-10-23） |
| C5 | THERMOSOIL显式步向getdiff转发原水文nslm | 通用物理装配正确性修复 | main `a117cd5`，新wrapper回归9项、既有窄检查2项通过，主控已验收；日尺度/全球均待维护线程核对（2026-10-24） |
| C6 | TF_DOC实际producer使用半小时干沉降dt | 通用调用单位修复 | main `27ff14b`，主控已验收；实际装配/小组件carry，完整compiled模型未证；两研究分支待核对 |
| C7 | DIC聚合仅将四个litter/soil率转步质量 | 通用过程接口修复 | main `27ff14b`，主控已验收；独立及真实混合三出口/no-flow，维护/root不重复乘dt；两研究分支待核对 |
| C8 | 共享bulk_dens按litter→soil源时刻两次赋值 | 通用输入/装配修复 | main `27ff14b`，主控已验收；raw/normalized/边界及marker回归，tiny跨步/负值fatal未闭合；两研究分支待核对 |
| C9 | sro_bottom默认5独立于nslm11/ndeep32 | 通用参数装配修复 | main `27ff14b`，主控已验收；普通/compiled/water/core/litter统一，已有API覆盖保留；两研究分支待核对 |
| C10 | DOC峰值扩散第二赋值读取已更新左通量 | 通用源行为恢复，独立分组 | main `225cb95`，主控已验收；四源算例×eager/JIT，源负库存局限保留；两研究分支待核对 |
| C11 | 雪融化潜热及两个harness恢复源值333700 | 通用源常数修复 | main `dd6d633`；独立土壤冻结lhf333600保留；旧参考不重标，两研究分支待核对 |
| C12 | HYDROL不同阶段保留tile4/5与4/5/6阈值差别 | 通用条件分支修复 | main `d48b923`；不全局取消tile6泥炭行为；两研究分支待核对 |
| C13 | 冻结dummy按源顺序执行两次含水量重构 | 通用源赋值顺序修复 | main `d48b923`；只证明固定profile片段，末态profile刷新另列Stage24；两研究分支待核对 |
| C14 | TOPMODEL解冻深度使用地表连续暖层前缀 | 已有诊断API修复 | main `d48b923`；未开放生产TOPMODEL；两研究分支待核对 |
| C15 | Z0_BARE贯通调用层并隔离compiled缓存 | 通用参数传递修复 | main `bbcdbf7`；真实组件、AST装配和缓存stub证据分列；两研究分支待核对 |
| C16 | 当步surface的ae进入诊断、carry和下步split | 通用状态传递修复 | main `2f36dc7`；真实组件组合通过，不重复计算总蒸发残差；两研究分支待核对 |
| C17 | normal末态mc刷新冻结profile，并传至dummy、输出和后续状态 | 通用源顺序/状态传递修复 | main `ad0d322`；保留normal mc/mcl/tmc，真实closure→writer→fallback及有限冻结控制通过；两研究分支待核对 |
| C18 | 矿物/泥炭非冻结查表先INT再整数加imin | 通用浮点边界表达式修复 | main `ad0d322`；局部ratio屏障、真实表及CPU窄JIT通过，冻结表达式保留；两研究分支待核对 |
| C19 | TOPMODEL诊断按cell tile6面积为所有tile选择潮汐水库 | 已有诊断API修复 | main `ad0d322`；严格阈值/多cell/控制项通过，生产TOPMODEL和fwet仍未准入；两研究分支待核对 |
| C20 | 有效SNOWCRI_ALB贯通main/init/wrapper/cold及缓存 | 通用参数门控/传递修复 | main `f8d36e1`；源默认/gate、严格布尔输入及真实组件通过，整模型compiled未证；两研究分支待核对 |
| C21 | 有效MIN_WIND贯通粗糙度入口及缓存 | 通用参数传递修复 | main `c214e43`；低风速组件源公式及门控控制通过；两研究分支待核对 |
| C22 | 有效HEIGHT_DISPLACEMENT贯通入口及缓存 | 通用参数传递修复 | main `c214e43`；保留源默认.66及非默认高度公式；两研究分支待核对 |
| C23 | 有效Z0_ICE贯通入口及缓存 | 通用参数传递修复 | main `c214e43`；冰区动量/热粗糙度组件通过；两研究分支待核对 |
| C24 | 有效SNOW_DENSITY贯通CONDVEG质量雪入口及缓存 | 已支持组件参数传递修复 | main `c214e43`；独立soil配置gate/正值检查，非全部密度consumer或bucket生命周期准入；两研究分支待核对 |
| C25 | 保存雪覆盖率并修正restart入口的post-HYDROL雪时序 | 通用状态/时序传递修复 | main `c214e43`；实际call→组件→writer/packet→DIFFUCO的18项通过，原raw缺字段回退保留；两研究分支待核对 |
| C26 | tmc可选诊断恢复源allowed_err默认2e-8 | 显式诊断API合同修复 | main `84b5d42`；接受真实迁移roundoff并保留超限停止、check关闭和显式override，不改科学门槛；两研究分支待核对 |
| C27 | 连续peat PFT间累积DOC_RUN_2_PEAT并按源回加 | 已有显式组件源行为修复 | main `9fc87c0`；连续/隔断/单PFT控制通过，未开放多peat生产；两研究分支待核对 |
| C28 | peat_OLT诊断采用源min_stomate默认1e-8 | 诊断默认修复 | main `9fc87c0`；阈值两侧/相等和实际core通过，库存/转移不变；两研究分支待核对 |
| I1 | 日块可选捕捉 `daily_flux_labels` | 共同 Teacher / 数据导出接口 | 先处理回调兼容性，再与日块函数提取协调接入 |
| R1 | 提取 `_paper_later_day_block_transition_factory` | 共同 CPU / GPU 运行接口候选 | 保持原标量路径，按既定前向合同核查提取后的行为 |
| R2 | `global_batch.py` 的 lane 组装与 `vmap` | 全球运行能力候选 | 继续留在全球研究线；满足支持范围、生命周期和运行接口要求后可进入 main |

C1–C3 已接入 main；I1、R1、R2 仍是接入候选。三个通用修复均按具体源码和
回归单独整理，没有整体合并研究检查点；C3 原提交 `5fb886c` 包含大量其他研究
改动，C1 原提交 `ad6c0e1` 也包含容量实验准备材料，均不应整体用于同步修复。

### 累计修改位置与提交索引

下列路径均相对`jax_orchidee/`，测试及必要harness随对应提交一起同步。
各项详细段落和[main审查报告](main_physics_review.md)保留修改必要性、源位置、
修复前后证据与适用范围；这里用实际Git提交复核修改位置，不重复历史数值实验。

| 编号 | 主要生产文件 | 准确代码提交 |
| --- | --- | --- |
| C1 | `sechiba/diffuco.py` | `db4dd8f1cc86a33dc8e343725fda2803f63fb358` |
| C2 | `stomate/carbon_kernels.py` | `e317fa81b97b8b8c959f2e7dfdfba525bb54dec8` |
| C3 | `stomate/carbon_kernels.py` | `2a3e303845154576c6f4b5c0b8155106e5bcc70b` |
| C4 | `coupled.py`、`driver/restart_state.py`、`stomate/restart_io.py` | `4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99` |
| C5 | `sechiba/thermosoil.py` | `a117cd5afc92ea55c9aeb528eec6f724de780111` |
| C6–C9 | `coupled.py`、`driver/orchestration.py`、`stomate/integration.py`、`stomate/soil_properties.py` | `27ff14b38aafc7df91ce52b8393642ca08952395` |
| C10 | `stomate/soilcarbon_kernels.py` | `225cb95a6a44334390d1656d85f66fd54d342a19` |
| C11 | `sechiba/enerbil.py`、`sechiba/hydrol.py`；另有两个雪harness模板 | `dd6d6335ba5011fcd86285984136a6799f6916ac` |
| C12–C14 | `sechiba/hydrol.py` | `d48b923e891946d3c9db77ba9b684666cb511b1d` |
| C15 | `sechiba/condveg.py`、`driver/orchestration.py` | `bbcdbf7a661cb11f5295883e54c3b1ea9d596d39` |
| C16 | `sechiba/hydrol.py`、`driver/orchestration.py` | `2f36dc7f7945d52387a2848aa124c043118ed45f` |
| C17–C19 | `sechiba/hydrol.py`、`driver/orchestration.py` | `ad0d322cc4951d4af158f4e8cbb6d306bdf374f3` |
| C20 | `sechiba/condveg.py`、`driver/orchestration.py` | `f8d36e1f8c03046e12e146aed2abb8bd286403de` |
| C21–C25 | `sechiba/condveg.py`、`sechiba/diffuco.py`、`driver/orchestration.py` | `c214e43da57c21b68002186fbf71c67d45835bae` |
| C26 | `sechiba/hydrol.py` | `84b5d425418a6d1b18fa6dbe080348c2faa72e13` |
| C27–C28 | `stomate/soilcarbon_kernels.py` | `9fc87c011a4e991f1637223b0f8fb2a224000595` |

### Stage24验收时的处置（历史；四项修复现见Stage25）

2026-10-25主控验看实际探针、保存值和成对源码，接受[有限审查证据](../../outputs/reviews/main-physics-20261025/main-source-closure/README.md)
的声明范围。下一批修复按下表处理；在实现和回归验收前，不分配新的已接受C编号。

| 发现ID | 已证实的范围 / 修改依据 | 下一批处置 |
| --- | --- | --- |
| S22-DUMMY-REFRESH | 源hydrol_soil6884–6886用normal末态mc再次刷新冻结profile；当前dummy和返回状态沿用旧profile，真实表组件对照已复现差异 | 修复刷新及状态传递；保留normal mc/mcl/tmc，补热力/线性冻结及直接调用组合回归 |
| S24-CWRR-NONFREEZE-BIN | 源8291–8301先INT再加imin，当前先加再INT；合法浮点边界选bin16/17 | 只对齐非冻结矿物/泥炭表达式；真实表、邻近边界与窄JIT补证，冻结表达式保留；现有value-tag不证明物理幅度 |
| S22-TOPMODEL-TIDE | 源7286–7293按cell的tile6面积为所有tile选水库；当前仅tile6选tide，显式参数组件差异已复现 | 修已有诊断API选择条件；保留生产TOPMODEL未准入，不扩展参数IO/生产支持 |
| S24-CONDVEG-SNOWCRI | 源已生效的非默认SNOWCRI_ALB在wrapper/main/init丢失；组件20给.5而源为1/3，默认10一致 | 先补实参入口与OK_SECHIBA/IMPOSE_PARAM有效配置门控证据，再修传递/缓存及相关调用；不能直接采纳门控关闭的raw覆盖值 |

fwet覆盖、其他四个参数接缝和五处检查/默认合同保持各自条件/静态证据域，
不自动计为生产bug或修复。雪maxmass三条普遍不可达证明撤回，登记为条件域未证明；
这次是证据登记修正，不是新的模型算法修复。319年度NPP原科学失败保留，
C1–C16及本批新差异均未建立该年度差异的原因或修复后改善。

### Stage25接受登记：C17–C20

基线main `7905b1165f6e812bf87514017765278f9df5da6d`，固定根及ORCJAX实际导入
均为`E:/ORCHIDEE-MAN-JAX`。Stage24上表保留验收当时的处置；四项已按后续授权
实施。2026-10-26主控已验看两补项实际代码/测试、修前失败、最终JUnit/log和八文件
交付身份，接受下表四项，按顺序对应C17–C20。**累计20项、13个代码提交**，
原C1–C16准确SHA完整保留。A-R1属C17状态闭合，D-R1属C20新helper合同，不另计缺陷。
C17–C19提交`ad0d322cc4951d4af158f4e8cbb6d306bdf374f3`，C20提交
`f8d36e1f8c03046e12e146aed2abb8bd286403de`；两研究分支同步仍待维护线程核对。

| 原发现ID | 必要性与修改位置 / 补丁组 | 输入条件、修前/修后及未证域 |
| --- | --- | --- |
| S22-DUMMY-REFRESH | 源hydrol6884–6905按normal末态mc更新profile、7165–7174只还原normal水状态。`hydrol.py`刷新/dummy/返回及module接口，`driver/orchestration.py`四调用传实际dh和五冻结控制；A-R1实际state writer改从soil结果持久化新profile；A组，依赖已有C13 | 冻结开启、normal mc改变；thermo/linear/module/实际closure四个修前数值失败，off通过；修后独立profile/真表/RHS/两次重构与normal mc/mcl/tmc还原、diag/下游组合、linear窄JIT和缺真实输入拒绝通过。A-R1真实零雪closure→writer/packet及无THERMOSOIL fallback在初版复现.000794945001109798误差，补齐后通过并保留normal状态；普通有THERMOSOIL下步重算，不能泛化逐步影响。土壤lhf333600和固定profile低层API保留；不认证normal solver、完整thermo JIT/refSOC/模型年 |
| S24-CWRR-NONFREEZE-BIN | 源8291–8301非冻结ratio→scale→INT→整数加imin；只修`hydrol.py`两个真实表consumer，B组。非冻结ratio局部屏障防编译重排，冻结8261/8275表达式保留 | 合法浮点边界.1682给源bin16、旧eager17；矿物/泥炭各16个边界/端点×eager/JIT及freeze控制均通过。整数顺序初版3pass1peatJITfail，局部屏障后4pass；原consumer AST隔离重放D差为4910.4347847302415/268.56503923288255mm²/day，不冒充共同修前/编译Fortran。无epsilon/生产nextafter/全局flags/容差调整；任意backend、逐位/性能未证 |
| S22-TOPMODEL-TIDE | 源7286–7293按cell tile6面积严格阈值为所有tile选择水库，`hydrol.py`已有诊断API，C组，依赖已有C14 | 3cell含无/等于/略高阈值，tidesoff/等水库/peatoff/既有5tile组件合同，eager/JIT及连续暖层旧回归通过；修前meanwt最大差.04m。生产TOPMODEL=true、四参数IO和fwet carry仍未准入/未修 |
| S24-CONDVEG-SNOWCRI | 源control455/495–503/550–555、intersurf2267/2302–2310/2330–2335和constantes_var129/697、constantes507–513核准effective gate/default。`condveg.py`resolver/main/init/wrapper，driver cold直接调用及两effective缓存键；D-R1补齐严格boolean输入合同；D组，依赖已有C15 | 已初始化SECHIBA且IMPOSE_PARAM=true才读raw，缺impose用源默认true，关闭gate不消费raw；explicit标量API表示已生效。有效20+雪10旧.5→源1/3；修前4fail/8controls。D-R1未知active gate改为ValueError、既有true/false及initialized-off不读raw保留，不计新物理缺陷。D50个唯一ID在新93组合通过，初版D44历史身份保留。实际组件/五call表达式×三gate情况/两个cache构造stub分列，不宣称完整driver或其他四参数/unsupported init准入 |

修前共同原状27项为17有效fail/10pass，保存三生产SHA、空diff/index和实际rootimport，
父级验看后才放行生产。初版选择61个新测试ID（保留原27，新增A2+D32）与25个直接相关
旧回归。实际组合84pass/2fixturefail，随后仅修directmain测试函数的合法无albedo输入，
所有断言/容差及生产保持，逆还全文件SHA证明无其他D测试修改，再仅重跑该2项为2pass。
**86个唯一ID均有pass证据，不是单次86pass**；原失败回执保留，不登记为物理反例。
[初版联合验证记录](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/verification-review.json)
及原失败回执保持历史身份，原入口工件另存`history/initial-delivery/`。

主控两项补充仍属Stage25。在与初版三生产SHA一致的dirty实现上先保存
[补项修前7项](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/revision-before.json)，
2有效fail/5pass/0error/0skip，无fixture/import/TypeError；父级验看后才改A writer和D gate。
三tests身份不变，按原12个selectors在最终生产一次完成
[93项全通过](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/revised-final.json)：
旧86 ID全部保留，新增A1+D6，仍为25个直接相关旧ID；
[补项联合验证](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/verification/revised-verification-review.json)
分列命令、ID、JUnit/stdout、稳定身份、rootimport和TEMP清理。未重跑共同修前27或原调查。
B局部屏障另复用主控CPU JAX0.10.1两consumer非边界grad/vmap的8个有限观察：
与旧consumer AST body在当前namespace/真表中重放匹配，不是历史整module或一般AD准入。

三生产文件、三新测试和本两份报告组成可审修改；A–D生产hunk、联合测试和报告依赖组
分别打包，并在系统TEMP内从基线副本依次apply/check和逐文件重构核对。
[main报告](main_physics_review.md)的第二十五阶段及
[交付核验](../../outputs/reviews/main-physics-20261025/main-boundary-fixes/delivery-verification.json)
保留具体源/当前函数与有限文件身份。执行方交付时未stage/commit/push，主控在验收后
按上述两组提交代码及测试，接受登记单独提交；不修改或同步其他研究工作树。
原manifest/patch保留交付时草案身份，主控接受更新不回写历史回执。

后续同步按C17–C19水文组和C20参数组核对。前者在含C13/C14的基线上建立，后者承接
C15接口/缓存；若研究线已有同等实现，先比较hunk与调用合同，不整文件替换。
三份Stage25测试和直接相关旧回归随组核对；93项CPU组件结果不代替研究线的GPU、
梯度、compiled运行验证。既有Teacher/prepared/restart保留原producer身份。

319年度NPP仍failed、water-origin暂停，V01–V03、C8/C10/Stage20/raw-restart及旧
Teacher/prepared/restart身份保留；本批未建立年度影响。无模型日年/完整模型编译、
Fortran/oracle/广泛suite/SSH/服务器/GPU/训练/付费/环境重建或自动化。
执行方复用既有Verify图generation2026-10-25T07:07:50Z及Transport closed后的精确源回退。
主控验收时服务恢复至generation2026-10-25T14:39:20Z，定向writer/parser/调用查询配合
coverage及当前源码核对；Python metadata_changed、Fortran excluded限制仍保留，
没有重建图或全域覆盖声明。剩余一批有限源码边界审查及集中收尾见main报告；
整模型科学验证另列V01–V03，不因本次修复登记而视为完成。

### C1：与 GPU 批处理无关的缓存缺陷

核查基线 main 的 `_diffuco_pft_column_mask` 在 `lru_cache` 内创建 JAX 数组。若首次调用
发生在追踪中，缓存会保留该次 `DynamicJaxprTracer`。全球分支仅改为创建相同
布尔值的只读 NumPy 数组，由消费处在各自追踪中转换；没有改变物理公式或掩码值。

本轮用从分支回归改编的小数组测试检查 14 个 PFT 和两种独立追踪形状。
main 在 `jax.checking_leaks` 下 14 项全部复现泄漏；在同一 main 模块内注入
精确候选实现后 14 项全部通过，包括列替换结果、dtype、掩码内容与只读属性。
这不是“所有正常主模型运行都会失败”的断言。

源码及正式回归已作为独立提交
`db4dd8f1cc86a33dc8e343725fda2803f63fb358` 接入 main。该提交只包含
`jax_orchidee/sechiba/diffuco.py` 和 `tests/unit/test_diffuco_mask_cache.py`，
便于日尺度分支单独同步。修复后的 DIFFUCO 文件与全球分支 `52326bd` 完全一致。

在实际修复后的 main 上执行
`conda run -n ORCJAX python -B -m pytest -q -p no:cacheprovider tests/unit/test_diffuco_mask_cache.py`，
14 项全部通过；测试文件 Ruff 检查和 `git diff --check` 通过。
测试使用 `PYTHONDONTWRITEBYTECODE=1`；关闭 cacheprovider 产生的 `cache_dir`
配置警告不影响结果。记录见
[主分支回归日志](../../outputs/reviews/shared-core-20261011/main-diffuco-fix-pytest.log)。
预审[补丁](../../outputs/reviews/shared-core-20261011/global/diffuco-mask-host-cache.patch)
和 [global-evidence.md](../../outputs/reviews/shared-core-20261011/global/global-evidence.md)
保留初始复现过程。

日尺度已由其维护线程同步为 `120c637`，保留来源记录并通过同一组 14 项回归；
核对 main / 日尺度的 DIFFUCO 文件无差异。全球分支已有同等修复。
三条开发线的这项修复已一致，历史实验执行版本仍保留原身份。

### C2、C3：共同修复，不依赖神经网络架构

C2 来自 `9023de2a3fbb3ddb49f4476d9bdac4b003502d1b`。它保持 NPP 前向源商、
泵出碳库及掩码，改用 main 已有的稳定比值原语。该原语在
`abs(denominator) < sqrt(finfo(dtype).tiny)` 时把商的切线设为零；FP64 阈值约
`1.49e-154`。这是一项明确的导数策略，不是任意小尺度下精确求导的证明。
独立方向探针中 `n=d/4, dn=d, dd=0` 的数学方向导数为 1，而实现于阈值下为 0、
阈值及以上为 1，前向值均为 0.25。本次接入保留原有阈值和上述导数策略。

C3 保留 Fortran 顺序的叶、根、果等 stock / turnover 更新，只在有效正叶质量
区域以相对质量计算叶龄份额，避免数学上会抵消的巨大梯度项。实数代数一致，
浮点份额的计算顺序有变化；零、负、阈值以下及退化情况保持原路径。

本轮复核了 Fortran `stomate_npp.f90::npp_calc` 299–386、447–531 和
`stomate_turnover.f90::turn` 569–665，以及命名报告和已保存的数值证据。
C3 当前函数源身份与已接受执行版本一致。四个实际边界的 stock / turnover
未变，份额前向最大差异不超过 `3.33e-16`；后续四项完整损失的 CPU JVP 与
CUDA 目标坐标 VJP 对照通过原容差。全参数及前向回归通过，但其他状态差异仍保留。
它们不证明所有 PFT、所有状态或完整神经训练通过。

初始本地复核通过 6 项碳过程回归及全部 8 项周转条件测试。当前神经合批的
梯度失败不应覆盖 C3 已独立通过的证据，也不能由 C3 的通过反推整个神经训练合格。
源版本、原始证据路径和测试命令见
[daily-evidence.md](../../outputs/reviews/shared-core-20261011/daily/daily-evidence.md)。

两项修复随后在实际 main 导入路径下分别接入并验证：

- C2：`e317fa81b97b8b8c959f2e7dfdfba525bb54dec8`，仅修改
  `jax_orchidee/stomate/carbon_kernels.py` 和
  `tests/unit/test_stomate_carbon_kernels.py`。12 项 NPP 源表达式、负碳库阈值、
  掩码、微小维持呼吸梯度及生产适配器回归通过。
- C3：`2a3e303845154576c6f4b5c0b8155106e5bcc70b`，仅修改同一碳过程源码并新增
  `tests/unit/test_turnover_fraction_conditioning.py`。20 项周转、临界叶龄、
  非活动掩码、微小叶质量及 turnover / gap / kill 适配器回归通过，其中包含
  日尺度已有的全部 8 项条件测试。

接入后的碳过程源码和原碳过程测试文件与日尺度 `120c637` 完全一致；新增测试仅做
格式及等价可调用语法整理，归一化 AST 与原测试一致。NPP 函数 AST 与来源提交一致；
周转函数 SHA256 为已接受的
`f6240cf5ce64a47b9bbe0647b5e5715305e9a9230ea77f2bd9f42981ae35a93c`，
去除新增份额计算后，原物理 stock / rate / mask / 更新顺序语句的 AST 未变。
原测试模块的 Ruff 检查未新增诊断，新增周转测试文件 Ruff 检查通过。

命令、运行环境、源身份及原始日志见
[接入证据](../../outputs/reviews/shared-core-20261011/carbon-integration/integration-evidence.md)
和[机器记录](../../outputs/reviews/shared-core-20261011/carbon-integration/integration-receipt.json)。
本次复用上述已接受的有界 CPU / CUDA 证据，没有重跑历史 GPU 或完整模型任务。

### C4：restart tau回退的单位与配置传递（2026-10-23）

独立修复提交：`4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99`。只包含三份生产文件
`jax_orchidee/stomate/restart_io.py`、`jax_orchidee/driver/restart_state.py`、
`jax_orchidee/coupled.py`（相关docstring）及两份直接回归
`tests/unit/test_stomate_restart_io.py`、`tests/parity/test_driver_1961_restart.py`。
审查报告另行提交，便于研究分支单独同步该修复。

触发条件是restart中长期温度有效，但`tau_longterm`为未初始化sentinel；partial helper
也允许tau字段缺失，而完整生产入口仍会先拒绝缺字段。原helper把系数3当成3天；
Fortran `stomate_data.f90`697–700和`stomate_io.f90`620–637要求系数乘年天数。
现在helper复用`SeasonTimeScales`的天数默认值（本配置1095天），生产reader传入
已有配置解析器给出的上限。显式天数覆盖保持原单位，自定义系数2.5对应912.5天；
有效tau原样保留，长期温度缺失/all-sentinel时仍按源优先初始化tau=2。

ORCJAX一次相关检查31/31通过，包括真实三文件生产读取/首次season选择；主控针对
新增配置解析依赖补跑原有无`run_def_path`入口测试，1项通过。验证仅涉及状态读取，
没有重跑模型日/年或服务器任务。命令及证据见
[Stage15入口](../../outputs/reviews/main-physics-20261023/restart-tau-fix/README.md)、
[31项检查](../../outputs/reviews/main-physics-20261023/restart-tau-fix/checks/attempt-01.json)及
[默认入口补查](../../outputs/reviews/main-physics-20261023/restart-tau-fix/checks/primary-default-entry.json)。

所核定的319/1961第三阶段输入本来含有效tau=1095，显式旧上限3/新上限1095读取
五组状态及report精确相同；因此不触发此缺陷，原年度NPP失败仍未关闭。
其他触发回退的输入可能改变后续模型结果；不得把旧Teacher、prepared数据或实验
producer改标为修复后版本，也不由这次读取检查宣称全部历史结果不变。

日尺度与全球效率维护线程后续可核对并使用`git cherry-pick -x 4b92d3dcb5dc534f5bfba1a64b1ea9fae296da99`。
同步应同时保留helper默认值、生产配置传参和相关回归，按各分支实际接口解决冲突，
不整体覆盖共享文件或替换正在运行的checkout。核查各自有效配置和输入是否触发回退，
通过相关状态读取回归后，将本节同步状态更新为各分支实际提交与验证结果。
本次只登记修复，未向其他聊天发送消息或自动合并研究分支。

### C5：THERMOSOIL原水文层数与深层潜热容量截断（2026-10-24）

状态：**最小实现、本地回归及主控验收完成；独立修复提交为
`a117cd5afc92ea55c9aeb528eec6f724de780111`**。本轮基线为
`04441e7d70f8e396757d79aa6dd363b4fc4eb214`，不是修复提交。Stage18已验收
S18-THERM-NSLM的源码原因及独立Fortran保存值，详见
[Stage18入口](../../outputs/reviews/main-physics-20261024/core-process-sweep/README.md)。

Fortran `thermosoil.f90::thermosoil_getdiff` 2742–2743要求 `jg>nslm` 的
`pcappa_supp` 清零。JAX `thermosoil_explicit_step` 在humlev已扩到ngrnd后未传
原水文nslm，direct helper默认推断遂使用扩展层数。最小修复只在该调用转发
`nslm=_as_float64(shumdiag_perma).shape[1]`；direct公式、默认值、精度、阈值和
已经正确传参的cold初始化调用均保持原语义。回归必须经过实际wrapper，不能将
Stage18的显式nslm direct comparator通过当作修复后wrapper通过。

生产改动仅 `jax_orchidee/sechiba/thermosoil.py:3171` 一行。新增独立测试
`tests/unit/test_thermosoil_nslm.py` 及小fixture
`tests/unit/fixtures/thermosoil_nslm_fortran.json` 已随独立修复提交；运行只读取该fixture，
不依赖ignored outputs或其诊断脚本。fixture保存4层pcapa/supp的8个独立Fortran值，
由原104值确认自然槽1–13逐层相同后压缩；自然PFT14明确按源自然分支角色映射槽1，
原oracle的crop槽14不作自然PFT14参考。源文件、过程跨度、编译单元及CSV身份均留存。

同一新回归在旧wrapper实测2项失败：自然槽1–13为52/104不符，自然PFT14为4/8，
浅层先行断言通过。只补传参数后新模块9项通过，涵盖相同独立比较、两个含端点相变
边界、暖/冷区间外、长期湿度0、等水热层数及一次真实wrapper容量投影JIT。既有
direct getdiff与wrapper composition窄检查2项通过；JIT只验证所选容量投影和traced
原湿度形状，不升级为生产模块全部输出或完整块编译等价。原容差1e-14/1e-12保留。

三次检查均在ORCJAX、Python -B、禁用pytest cacheprovider及系统TEMP下运行；TEMP
均已清理。cacheprovider禁用后的既有cache_dir配置warning原样保留，没有据此修改
项目配置或重跑。没有新的Fortran编译、模型日/年、服务器或研究执行。实际命令、
旧失败/新通过日志及身份见
[Stage19入口](../../outputs/reviews/main-physics-20261024/thermosoil-nslm-fix/README.md)、
[实现与验证回执](../../outputs/reviews/main-physics-20261024/thermosoil-nslm-fix/implementation-and-validation.json)。

日尺度和全球效率分支的潜在同步范围是这一共享thermal wrapper及其独立组件回归/
小fixture。各维护线程可据上述独立修复提交，按实际差异核对或同步准确
提交，保留各自执行优化和训练接口，检查是否真实经过该wrapper或独立thermal副本；
不得整体覆盖共享文件或原地替换正在运行的checkout。本轮不操作两工作树或向其聊天
发消息，不将旧Teacher/prepared数据改标为修复后生成。

paper的nslm11/ngrnd32及冻结开启只证明静态层数条件；深层自然活动mask、正长期
湿度及含端点相变区间是否在实际轨迹同时出现、年度影响和319 NPP因果均未证。
原319失败及原阈值保留。raw-restart首步旧热容量回退仍是未证接口限制，不并入C5。

主控验收已核对实际新测试/fixture、执行器、旧失败和新通过日志、三个当前受测文件
身份及一行生产差异，复用执行方9+2检查，未重跑数值。原交付回执的待验收状态保留
其交付时点身份；本节和主审查报告记录最终验收及修复SHA。

### C6：TF_DOC实际producer的半小时干沉降时钟（2026-10-25）

状态：**主控已验收，修复提交`27ff14b38aafc7df91ce52b8393642ca08952395`**。`stomate.f90::stomate_main:3380–3388`
向soilcarbon传半小时dt，`stomate_soilcarbon.f90::soilcarbon_leak:1206–1233`只在干项
乘dt。初始化、普通fold及compiled wrapper三处改传`runtime.dt_sechiba/86400`，降水
定义保留。leaf100/cover.4独立预期干项`.0007666666666666667`，旧为`.0368`；湿项、
TF关闭、真实ground/storage两步carry有界通过。compiled参数/窄scan使用明确core seam，
不宣称完整compiled模型数值等价。C6旧3失败/3控制通过；与C9同15项修复后全通过。
实现位于`driver/orchestration.py`，回归`tests/parity/test_ok_leak_producer_contracts.py`。
见[Stage21说明](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/implementation-and-validation.md)。

### C7：DIC聚合的litter/soil率与即时maintenance/root质量（2026-10-25）

状态：**主控已验收，修复提交`27ff14b38aafc7df91ce52b8393642ca08952395`**。litter源2667–2670、2740–2750、
2778–2781及soilcarbon1671–1704返回日率，stomate3314–3315、3409–3410先乘半小时dt
再3442–3456聚合。resp297–298的即时maintenance已含dt，stomate3258的floodroot也是
步质量；日累计是另一消费者。`stomate_ok_leak_explicit`只将四个litter/soil聚合参数乘
`dt_days`，保留核心rate对象，maintenance/root不再乘。新独立fixture混合三出口为
`[.03529411764705882,.09895833333333333,.7058823529411764]`，另有真实混合组件和no-flow。
两项旧混合失败已修复，回归/小fixture在`tests/parity/test_ok_leak_dic_units.py`及同目录
fixtures；既有manual-chain仅同步该边界期望。未证年度效应。

### C8：bulk_dens两次源赋值、共享owner与prepared marker（2026-10-25）

状态：**主控已验收，修复提交`27ff14b38aafc7df91ce52b8393642ca08952395`**。litter1971–1973及soilcarbon1473–1478
各执行一次`<500 ×1000`。raw static/map/default保留raw，controls前执行litter赋值；
generic/same-step合并采用prepared值，adapter跳过已完成的litter后执行soil赋值，compiled
以tracer-safe`jnp.where`尊重marker，防止tiny值第三次转换。新增`soil_properties.py`
含精确Fortran provenance；`peat_bulk_density`分离，不在低层adsorption偷偷转换。
同原10项旧8失败/2通过→同10通过：4真实adapter数值、2controls实参，另2仅为新增marker
缺失的KeyError。当前20项含真实producer/controls/adapter、generic/peat及窄scan参数检查，
最后组合全部通过。raw1.65、1650、500、.5及tiny`.0001→.1→100`有界覆盖；tiny仅证
一次调用，未引入通用跨步bulk carry，负值源fatal/nonfinite及跨步回写仍未闭合。

### C9：独立于网格层数的runoff cutoff（2026-10-25）

状态：**主控已验收，修复提交`27ff14b38aafc7df91ce52b8393642ca08952395`**。源`constantes_var.f90:294`默认
`sro_bottom=5`，独立于水文11/深层32。初始化、普通和compiled装配统一水量及core/litter
cutoff；compiled独立static参数，排除重复packet键。已有内存API覆盖3仍可用，不新增
run.def能力，不改11/32形状。producer哨兵水量5层55、旧11层506、覆盖3为14仅证明
参数；真实decomposition消费捕获cutoff，独立标量layer5/6 DOC`.075/.1125`，旧compiled
为`.075/.075`。C9旧9失败，修复后与C6同15项全通过；不是新Fortran oracle或全年验证。

### C10：DOC峰值扩散顺序恢复及负剩余库存局限（2026-10-25）

状态：**主控已验收，独立修复提交`225cb95a6a44334390d1656d85f66fd54d342a19`**。主控已授权按源`stomate_soilcarbon.f90::
soilcarbon_leak:2002–2006`恢复先2005后2006；第二分母读取更新后的左通量。只改该
分母和来源注释，WHERE/循环/非峰值/冻结保留。默认半小时D、正薄层与`[1,3,1]`
给flux`[1.5,12/7]`，末库存`[2.5,-3/14,19/7]`。**负中央库存是源算法局限，未clamp或
改造limiter。** 新独立fixture四算例×eager/窄JIT8项及既有直接检查通过，旧峰值两项
失败真实保留；原simultaneous1.5/1.5期望按源改正。

独立[四文件patch](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/C10-source-order.patch)
和[分组身份](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/C10-group.json)
仅含soilcarbon扩散hunk、旧测试期望hunk、新source-order测试与fixture，不混C7/C8。
原交付只作正/逆向适用性检查；主控已将该四文件单独提交为上述C10修复。

C6–C10最终[组合49项](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/combined-focused.json)
exit0，绑定最后实际文件身份；[交付核验](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/verification.delivery.json)
只作元数据/链接/Git/补丁检查，不重跑数值。fixture前置错误与有效缺陷失败分开，重叠
检查不累加为独立覆盖率。图工具不可用，复用Stage20 Verify generation
`2026-10-24T15:26:49Z`（Python metadata_changed、Fortran excluded），确切源补足，
不宣称新图覆盖。两研究分支待维护线程按最终真实提交及各自接口核对，不整体覆文件、
不替换运行checkout、不重标历史Teacher/prepared。本批无研究工作树或跨聊天操作。
原319失败、C5实际轨迹/年度影响未证、raw-restart旧容量未知及暂停水分起源调查保留；
两个Stage20 candidate与完整SAVE生命周期不并入，科学门槛/预算不改。详见
[Stage21入口](../../outputs/reviews/main-physics-20261025/ok-leak-core-fixes/README.md)。

主控验收已复核实际代码/测试/fixture、有效旧失败和最终49项日志；14份当前受测
文件身份一致，未重复数值运行。C6–C9共享一个九文件修复提交，C10使用另一个四文件
提交，审查文档另行提交。执行回执中“待主控验收/SHA待定”保留原交付时点身份，
本登记给出最终接受状态与准确提交。同步时按这两组真实差异处理，不能以整文件替换
覆盖研究分支的独立实现，也不能把历史Teacher/prepared改标为修复后产出。

### Stage23：六项共享修复登记（已接受C11–C16）

实施基线为main `c94da5b623da4f6da377520ce08436933b6ed1b7`，不是修复SHA；
Stage22历史数值producer保留`1229de7b41711a5177d572feef5cc95f4cbc2a32`。
下表不更改C4–C10的接受状态、准确提交或研究同步记录。详见
[Stage23实现/证据/补丁入口](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-fixes/README.md)和
[main审查Stage23](main_physics_review.md)。

| 修复编号 / 原发现ID | 已接受改动 | 依赖与接受边界 |
| --- | --- | --- |
| C11 / S22-SNOW-LF | 共享derived333700、8显式雪+1bucket默认及两个harness参数关系 | hydrol+enerbil共享常数，原雪test预期；独立土壤lhf333600保留。12新增/8既有通过；不重建或重标旧333600 oracle |
| C12 / S22-TILE6-THRESHOLDS | 保留mineral阈值，top/clip/部分residual仅4/5，deep/coef/liquid保留4/5/6 | hydrol各阶段hunks及旧routing fixture；BCD共用一个自含回归文件，不全局取消tile6 peat行为 |
| C13 / S22-FROZEN-DUMMY-ORDER | freeze分支第一次赋值后无条件第二次重构，真实caller传switch | hydrol dummy helper/caller；固定profile/retry/restore有界证据，不关闭DUMMY-REFRESH候选 |
| C14 / S22-TOPMODEL-PREFIX | 地表连续暖层前缀及numlayers边界 | hydrol diagnostic API；源>6/undef保留，不开放生产TOPMODEL或关闭潮汐/carry候选 |
| C15 / S22-CONDVEG-Z0 | wrapper/main/init和cold绑定Z0_BARE，两compiled cache隔离 | condveg+driver E hunks；13新增/21既有通过。物理wrapper与AST实参/cache stub证据分列，不称完整compiled模型验收 |
| C16 / S22-AE-STATE | 真实post-surface ae进入diag，shared writer carry与finalize相同diag阶段 | hydrol+driver F hunks；9项真实组件→writer→下次split通过；grid总vevapnu残差不重复加 |

A12失败、BCD17失败/13通过、E2失败/4通过及2 cache失败、F7失败/2通过的before均在
共同生产barrier前保存，无error/skip；之后原BCD30项通过并加4条numlayers边界。
最终组合以[focused回执](../../outputs/reviews/main-physics-20261025/hydrol-snow-condveg-fixes/combined-focused.json)
记录的72项通过为准（exit0、无error/skip，TEMP清理）；重叠测试数不当作独立oracle或
完整覆盖。旧雪预期按源常数改；旧tile4综合routing
夹具改为source真正有阶段阈值差异的tile6，保留原数值断言及独立tile4检查，不放宽容差。

主控已验看全部生产/test改动和有效before/after，复用数值检查、没有重复积分或组件运行。
16份交付文件身份均匹配，最终三份生产文件与组合启动时SHA一致。代码分为四组提交：

| 编号 | 精确修复SHA |
| --- | --- |
| C11 | `dd6d6335ba5011fcd86285984136a6799f6916ac` |
| C12–C14 | `d48b923e891946d3c9db77ba9b684666cb511b1d` |
| C15 | `bbcdbf7a661cb11f5295883e54c3b1ea9d596d39` |
| C16 | `2f36dc7f7945d52387a2848aa124c043118ed45f` |

交付patch保留A/B/C/D/E/F问题hunk分组；B/C/D共用回归随C12–C14同一提交。
hydrol、driver的同文件hunk不能以整文件覆盖研究分支，不自动同步daily/global。
运行checkout、Teacher/prepared、旧Fortran输出及
原333600 harness结果继续保留各自producer身份。

三个S22候选继续开放；319年度失败、暂停water-origin、C10源负库存、C8 tiny跨步/
negative fatal/nonfinite、两项Stage20 peat候选及raw-restart旧容量未知均保留。
无年度改善/因果、全配置/完整compiled模型或新生产准入结论；无新Fortran、模型日年、
服务器/GPU/训练/付费、研究工作树、自动化或跨聊天操作。执行方交付后，由主控完成上述
接受与提交；交付时的pending/SHA null回执保留历史状态，以本登记和主控回执为后续接受依据。

### I1：四行改动包含回调接口变化

I1 来自 `2eb63842aefc5a685451e502815039bdc24b0ae1`，把日通量标签接入已有的
compiled block，并把捕捉开关加入缓存 key。底层标签类型与单日捕捉能力 main
已经存在。它是可选的 Teacher 导出能力，不需要搬入神经网络实现。

但日尺度版本无条件把 `training_output_projector` 从两个位置参数改为三个，
即使通量捕捉关闭也传入第三个 `None`。日尺度保留的
`scripts/dev/probe_ok_leak_outer_block_reentry.py::target_projector` 仍只有两个
参数；签名检查可确认三参数调用不兼容。这是日尺度旧诊断调用点的具体缺口，
该脚本不在 main，不能表述为 main 现有生产调用已经失败。

集成时需显式保持旧回调合同或采用可选 / 版本化的新接口，并检查相关调用点；
不要用捕获任意 `TypeError` 的方式掩盖回调内部错误。I1 与 R1 修改同一段日块
组织，应一起设计接口，但可以分别验证和提交。

### R1、R2：已有真实前向证据，尚不是完整生产入口

全球分支保持物理过程方程，把原日 scan 提成函数并在外层批量调用。
当前适配器只接受每点 `npts=1`、48 半小时 / 日、连续 7 日和兼容的静态配置，
拒绝 peat / TOPMODEL，且模块明确未提供生产 CLI 和持久可执行缓存。

单元测试使用模拟标量函数检查组装、拒绝条件与 lane 独立性；研究线另有真实
GPU 前向证据，不能把两者混为一谈：

- 固定 `bd9e4c04...` 的 job15662711：B512 / B1024 两阶段各 1027 门槛通过，
  保留 760 个原生输出叶和 1024 次 FULL64 对照。
- 固定 `8aa2e05a...` 的 job15671732：B1024 对照阶段通过 2051 门槛及 2048 次
  FULL64 对照；B2048 在第二连续块执行时 OOM，没有完整验收结果。

以上为 64 个真实点重复扩展的容量负载，不能替代真实 669 异质点、独立集合成员、
长期状态 / restart 和生产 IO 的对应验收。原证据分别见全球工作树的
[B1024 结果](../../runtime/worktrees/global-throughput/outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/CURRENT_RESULT.md)
和 [B2048 结果](../../runtime/worktrees/global-throughput/outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/CURRENT_RESULT.md)。

未来接入 main 应按拟支持的运行合同复用已有前向状态、输出、掩码、原生 dtype
证据并补实际缺口。全球当前研究正向吞吐，完整梯度验收不是本轮新加的接入
前置条件；若声明批量可微性，再按其明确用途补相应证据。

## 数据、执行版本与同步

固定 v6 Teacher 的生产报告记录 producer
`19087b171564f44b49eb96fa1217169deca5d57a`，其 `carbon_kernels.py` 与初始 main
核查基线 `7cae3e0` 相同，早于本次 C2、C3 接入。报告里的生产 / 哈希验收和 CPU smoke，不等于 main README
中仍待完成的 669 点长期 Fortran 等价验收；两种验收的目标不同。

Teacher 生成版本、prepared 对象构建版本、当前执行版本可以不同，必须分别保留
身份和消费者相容性。C2 保持源前向商，C3 允许上述有界浮点差异；这些事实本身
既不要求重新生成 Teacher，也不证明所有 prepared 包可无条件混用。本轮没有
核定当前活动 prepared 包的完整生产清单。既有 v6 / prepared 复用工作单、类和
模块序列化身份继续保留；不要改写历史 producer 标签。

同步状态：C1 在三条开发线上已有同等实现；C2、C3 已进入 main，日尺度原本已有
对应实现，不需盲目重复应用。全球效率线 `52326bd` 尚缺 C2、C3，由其维护线程在
完整阶段边界依次同步 `e317fa8`、`2a3e303`，并核查实际依赖的前向行为。正在执行、
排队或用于历史对照的代码和可执行文件继续保留原版本，不在原地替换。

主分支的这轮已知通用修复收尾完成，后续可开始按 Fortran 来源和分支覆盖账本
检查已有物理过程代码。I1 / R1 的兼容接口及 R2 的全球运行支持范围仍由相应
工作线推进，尚未自动纳入 main；本轮不能称为三个分支全部文件已经一致。

## 本轮验证与局限

本轮使用 ORCJAX / Python 3.11.15 / JAX 0.10.1 CPU。DIFFUCO 原 main 的 14 项
失败是有意复现；候选及接入 main 后均为 14 项通过，日尺度同步回归也为 14 项通过。
碳过程初始复核为 6 + 8 项通过；实际 main 接入后，C2 的 12 项、C3 的 20 项通过。
独立解析探针还核查了导数阈值、旧回调签名和已接受周转源身份。
保存证据链接位于上述分项报告和接入记录；
这些 `outputs/` 文件按项目规范不入 Git，本文保留可审查的结论、版本与复现入口。

知识图仅覆盖 main，generation `2026-10-10T08:46:50Z`；目标文件元数据已变更，
研究新增文件不在 main 图中，因此关键结论使用直接 Git 差异和源码回退核查。
本轮不是全模型 / 全分支科学验收，没有重新执行历史 GPU 任务、训练或生成数据。
本页随共同修复接入更新，不追加研究实验时间线。
