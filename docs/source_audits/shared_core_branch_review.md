# 共享物理核心与研究分支核查

Stage26当前：四包边界及集中审查S26-R1已主控验收，最终132项定向通过，**本轮有限源码审查收尾**。
见[Stage26交付入口](../../outputs/reviews/main-physics-20261026/final-source-boundaries/README.md)及下方接受分组。
新增C21–C28；Stage25的C17–C20及93项原证据保留。
最新修复登记：2026-10-26。累计28项已接受修复、16个代码提交，仍未建立319年度归因。

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

[V01准确短运行/状态连续性与重启方案](../../outputs/reviews/main-physics-20261026/final-source-boundaries/next-validation-plan.md)
仅为未执行准备，复用现存入口/冻结输入/参考和原门槛，说明真正later-day块与
raw restart容量缺项限制；不当作整模型验证证据。源码接受后由主控另行安排实际
资源，随后依次进行固定319原1961全年/原阈值回归及跨点/669长期验证，
不重开water-origin或原年度调查。此次仍无模型日年/XLA/服务器运行或同步。

参数及fraction carry影响非默认值和运行state字段；今后相应维护线程需按拟支持用途
评估compiled/restart/Teacher消费者兼容性，不能用文本合并代替数值准入或整文件覆盖。
C4–C28同步仍待维护线程核对，既有Teacher/prepared/restart/reference producer与执行版本
分别保留。本批没有同步授权/操作。319仍failed，water-origin暂停；V01–V03、C8 tiny/
负值fatal/nonfinite、C10源负库存、TOPMODEL/fwet未准入边界保留，无年度归因。

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
