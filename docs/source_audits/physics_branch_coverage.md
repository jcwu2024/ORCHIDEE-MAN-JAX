# 当前物理计算与条件分支覆盖

2026-11-19，当前正式累计**64项修正／支持补齐、36个相关生产提交**，含C62–C64。冠层／能量[首包与CI修订已获主控有限接受](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-acceptance.json)；没有宣布整个阶段或PB-01／PB-02完成。测试／参考／CI提交为`ebd1141d5b214ab0b874d124eab2466eef8e0a77`，当时不改变生产模型或61／34计数。范围与门槛遵循[当前计划](physics_branch_closeout_plan.md)和[PFT14交付合同](pft14_equivalence_delivery_contract.md)：独立单点、forcing值／来源、连续参数和支持状态下的源条件，不能缩成默认点或旧319轨迹。

执行资产以[CI交付修订](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/controller-ci-repair/README.md)绑定：缺原源的隔离checkout仍执行全部数值／声明输入检查，仅两项source/span门跳过；实际本地uv三文件结果110 passed／2 source skipped／2 strict xfailed。有源静态门另2通过，hosted Ubuntu／uv sync未执行。主控核对实际修订与9份执行资产并复用检查；JSON夹具经Git换行规范化后值一致，当前管理文档另随验收更新。原科学证据和覆盖等级不升级，后续PB-03进度见下文。

| 工作组 | 当前可复用与新增证据 | 尚需的最小动作 |
| --- | --- | --- |
| PB-01 冠层／光合 | 复用 C53／C54 和旧 10 例；新增 DIFFUCO 20、盐潮 8、CONDVEG 8 例，当前 eager／组件 JIT；真实 water100 到能量消费者 | 低光 cim 严格失败保留，已有同系数浮点放大证据；极低正 LAI 原参考 trap 的受支持前驱／参考合同未证；根通气及部分 albedo 分支缺参考 |
| PB-02 能量／QSAT | 原完整能量过程与 QSAT 的 5×8 点、24 节点；420 字段／运行检查全部通过，实际 downstream／carry／量率边界单列 | 独立 beta tuple 不证明自然联合可达；tiny-Milly false 与部分雪消融 conjunct 需合法 tuple 或不变量证明；外部 PBL 身份单列 |
| PB-03 水文／土壤热／雪 | [第三包已接受](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase03-hydrol-thermal-closeout/README.md)复用既有参考，补dry／joint同输入有界浮点归因、冻结阈值→液态水及最终热钳制→真实writer证据；C64冷启动ZWT_FORCE逐土壤块传递已修补并过隔离输入／消费者检查 | 全部strict坐标保留；独立雪groundflux、水文首差土柱／5个JIT蒸发限值、私有peat历史有效build、自然雪前驱和其余mask仍未证；本次修复限npts1/nstm6；新证据均按有限范围接受 |
| PB-04 植被／碳 | 原 daily／season／phenology／allocation／NPP／gap／turnover／litter／soilcarbon Oracle families，已有日历和状态工作保留 | 具体 owner 按日相位／增长周转／凋落物土壤碳分派，补当前完整同输入和顺序写回关联；DOC 严格状态及真实 flux 缺证仍在；OK_PC 完整 driver guard 与局部微参考分开 |
| PB-05 跨组／诊断 | 本包 6 项真实组件消费者与单位写回检查；旧 forcing／静态聚合／module order／modelout／normalized restart 证据保留 | 精确关联剩余 forcing 数值、日表面、科学 restart 消费者、modelout 单位和组间边；多点 routing、一般 native IO、历史 producer 身份独立登记 |

[当前机器总账](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/coverage/current-coverage.json)为 52 个历史过程条目保留具体 owner、源条件、JAX caller、相位、旧证据与下一动作；第二包仅增补 13 个 PB-03 条目的精确处置。7 种旧集合稳定 ID 原样保留；308 去重 span、12 扩展、26428 arms 与旧 4217 数值 arms／824 Oracle 关联不是同一分母，不能相加或称缺关联就是 bug。旧 HYDROL 109-arm 参考没有与这 13 条登记 span 的直接行号交集，不能广播为其逐 arm 通过。宽基础设施 arm 仍有目标域判定缺项，当前表不宣称分母完整。

证据等级 `inventoried`、`source_mapped`、`jax_regression`、`fortran_oracle`、`production_verified` 按实际边界分别记录。旧有界接受不因当前 overlay 默认 inventoried 而作废，也不广播到全文件；站点候选 arm 关联不等于逐 arm pass。物理组件、外围输入组装和严格数值门的结果分别保留，详见[分批验证](../../outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/verification-results.json)。

[具名剩余动作](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/coverage/remaining.json)保留 PB-03 未解门和其余四组缺项。
主控要求先完成[生产集成与配置修订](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/README.md)：
潮汐两份原始文本按 native-run/source-istp 进入真实 driver／编译强迫／precall／owner，
`CHECK_CWRR` 默认 false 与完整 true 拒绝跨缓存绑定；土柱求和顺序候选已撤回至接受基线。
原覆盖表和科学回执不改写历史；修订重测受求和恢复影响的原 19 例并保留每个严格失败坐标，
不重跑已接受土壤参数五例、不重开 NPP／DOC／snow 既有因果。两个生产原因的年代及
C13／C17／C18／C55旧参考影响已有有界核查，未整体撤销原接受。修订现获[主控接受](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/controller-repair/controller-acceptance.json)，两项登记C62／C63，共同提交`7b35196462fbfd53f531fd1dc751699092f80104`；土柱求和候选不计修复。
当前隔离135通过／195精确strict xfail／4身份skip，19例水文100 eager／93 JIT失败坐标保持原限。旧机器表中的151坐标属于撤回求和候选的历史版本，以修订numeric回执为当前数值身份；不回写冻结表。
[当前紧凑覆盖层](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase03-hydrol-thermal-closeout/coverage.json)保留13个PB03稳定过程ID，按源条件、真实caller／writeback、具名案例和剩余动作关联新证据；52条父表和七种历史ID集合保持原身份，不复制巨型arm列表或更改分母。dry／joint同输入的底层递推1ULP差异与表面相消支持有界浮点归因，不能把严格失败改作通过。水文保存342个tile阶段证明下游check／校正／聚合传播，尚未确定tmci／tmcf的首差源头；另5个JIT蒸发限值仍在。

私有peat原声明与传递USE的全局同名符号在最小编译检查中冲突；本地ppsrc也保留该形式。缺的是与历史可执行文件绑定的build／module边界，不能把已有trace false叫原源默认或据此排除true目标条件。冷启动ZWT_FORCE首值广播已接受为C64，完整向量及默认等价声明接入真实构造器／饱和消费者，提交`e275a33fd5864957e9b241138359c1f6001a5756`。源等价限定npts=1、nstm=6；原setvar先判第一轴，等长轴／多点未获证。字面IOIPSL单值扩展、混合优先序与部分非默认索引仍明确未支持。19项隔离uv检查无skip通过，64个导入在隔离副本；[主控](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase03-hydrol-thermal-closeout/controller-acceptance.json)复用验证并接受dry／joint热浮点归因，严格门不改。其余组及[具名剩余动作](../../outputs/reviews/main-physics-20261119/physics-branch-closeout/phase03-hydrol-thermal-closeout/remaining.json)保留，不能称PB-03完成。
正式64项／36相关生产提交；`numerical_acceptance=false`，整模型综合验证和669尚未开始。
