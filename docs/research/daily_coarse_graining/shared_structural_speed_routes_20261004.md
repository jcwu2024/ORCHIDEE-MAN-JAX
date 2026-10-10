# 解码计算结构与架构决策

日期：2026-10-04。**用户已选择 latent32/reader64 作为下一研究原型，开始实施与实测。**
原始设计阶段完成源码、静态布局、参数与公式核算及独立审查，没有运行新模型。
当前授权包含可选实现、小布局权限/AD、真实T7优先及有依据的daily/T30/完整更新扩展，
并合并原版双卡调度连续更新。统一免费预算5 GPU小时（各卡占用墙钟求和），
单次占用最多2小时；编译、加载、普通修复和运行计入同一预算，不重置。
每个诊断模型链最多16次实际更新；不是生产默认、正式训练或长期学习授权。
T30 expert5 chunk-vmap 扩展暂缓；统一 decoder384→128 未选。保留已有 V4/V7 及学习/科学证据。

## 1. 已测成本与本轮判断

同工作量原版完整更新约 **12.94秒**；V4/V7 的16步三臂诊断提高吞吐 **16.56–16.72%**，
但严格参数/状态梯度失败978/287，留出开发面板在 step16/T30 的 GPP/NPP/AGB/BGB 误差
高于原版重复漂移；保留研究机制，停止直接替换。详见
[学习结果](shared_v4_learning_sensitivity_results_20261004.md)及
[科学比较](shared_v4_heldout_science_results_20261004.md)，不追加同类试验。

[前向/AD测量](shared_forward_reverse_results_20260930.md)给出匹配组件中位数之和：
前向2.104821秒（18.57%），额外AD/重算9.228212秒（81.43%）；AdamW约0.02–0.03秒。
最大的机会在模型计算及其AD。81.43%含反向和checkpoint重算，既不是纯反向分账，
也不是decoder占比；这组中位数没有精确分割一次12.94秒更新。

源码说明宽投影和FFN在每个坐标上重复。expert5有7,542个静态可用坐标，
占8,550个静态可用输出坐标的88.21%。K/V复用已经存在；继续缓存相同bank不是本轮新收益。
新方案将宽非线性移到7×32个latent，把所有坐标的私有条件计算限制在64维。
这保留每个输出与信息路由，却改变函数类，值得先证伪、再决定是否投入全模型编译。
不以“辅助输出”标签证明某个输出可删或可降为不同头。

## 2. 当前张量轴及实际执行频率

记B为同时处理的样本数，E=7，源S=393，专家轴A=140，decoder宽D=384，
完整输出39,006；当前布局的静态可用R=8,550，expert128执行行Rpad=8,716。
这些是源码/静态fixture核查，吻合已有布局资产；没有新读取真实prepared输入。
实际active/defined掩码随输入变化，静态可用不等于每个样本非零。
当前active PFT槽[0,13]只描述这份布局，不能成为未来模型的硬编码限制。

| 张量/操作 | 每个样本日的轴与数量 | 频率及共享边界 |
| --- | --- | --- |
| forcing tokenizer | 45=5记录×9字段，宽256 | 1前驱+4原生六小时记录；两层在展平45token上做masked self-attention，family相同才可互读；源形状仍为45×45分数，不能误写为九条独立5×5 |
| 其余来源token | 257连续+8离散+30静态+1年度+49过程参数+2PFT+1日历 | 与45 forcing合计393；按类型编码与来源掩码，没有393token全局self-attention |
| axial tokenizer | [B,7,18,32,384] | 18物理轴，140真实位置/专家；pad至18×32=576，七流共4,032行、3,052 padding；一层每专家/每物理轴内部self-attention，轴之间不合并 |
| backbone过程query | [B,7,1,768] | 八层cross-attention；七专家共用每层权重，激活及权限隔离；Q/O、FFN只在七个query上算 |
| backbone K/V bank | source [B,393,768]，axis [B,7,140,768] | 每层source K/V算一次，expert-axis分别算；八层有不同权重，不能跨层复用投影 |
| family/context bridge | family [B,7,20,384]，axis [B,7,140,384]，process [B,7,1,384]，PFT [B,2,384] | 先按专家合法来源汇总family，再768→384；合计1,129行，一次/日 |
| coordinate query features | [B,8550,384] | 每行有process、field、label、轴角色/位置/slot、PFT、对齐local(value,defined)；同形状有不同数值 |
| decoder共同K/V | 7×20 family+7 process，GPP另2386 raw carbon | 两层各自投影/cache，一次/专家/日；raw carbon仅GPP读取 |
| decoder私有K/V | 1,380 field-axis+3,249 field-local bank行 | 两层各自按field身份缓存，再按每query索引gather；不把不同field或local值混成公共bank |
| coordinate block | [B,Rpad,1,384] | 两层Q/O、cross-attention、FFN1536、residual/norm在每行独立执行；没有坐标间self-attention |
| readout | 384→256→1 | 每执行行一次，再恢复全部字段形状及掩码 |

两个coordinate block顺序依赖：第二层query由第一层attention和FFN生成，
所以不能把两次条件读取预先算成同一个结果。共同参数不会使这些query或FFN输出相同。
非GPP有160字段/8,548行，按expert合并执行但保留各field的dynamic K/V索引；
GPP的两个当前可用坐标保留独立canonical路径。

| 非GPP expert | 字段 | 有效静态行 | 执行行 | map次数 |
| --- | ---: | ---: | ---: | ---: |
| canopy_gpp | 8 | 14 | 14 | 1 |
| surface_energy | 22 | 28 | 28 | 1 |
| hydrology_snow | 73 | 722 | 768 | 6 |
| soil_thermal | 6 | 146 | 256 | 2 |
| maintenance_respiration | 1 | 24 | 24 | 1 |
| litter_poc_doc / expert5 | 42 | 7542 | 7552 | 59 |
| daily_interface_finalize | 8 | 72 | 72 | 1 |
| 合计，另加独立GPP | 160+1 | 8548+2 | 8714+2 | 71+1 |

expert5的59×128中仅10行padding；去掉这10行解决不了主要成本。
当前所有专家合计166行coordinate padding。旧9,046 padded行与68.487G账本属于旧field/
广播source配置，不能作为本轮reference。

## 3. 唯一推荐的可审查架构

保留backbone768×8、既有384维bridge/坐标/轴/local/carbon编码、全部输出和canonical绑定。
声明以下固定首个假设，不展开宽度/latent数量搜索：

1. **每专家32个384维latent，两层。** 初始query为共享slot embedding加本专家process。
   每层依次cross-attention读取本专家20个family槽+1process、norm，再专家内部32-slot
   self-attention、norm，最后FFN1536、norm。两种attention各有独立Q/K/V/O权重。
   七组激活始终隔离；slot参数在专家之间共享，不表示PFT编号。
2. **坐标reader宽64，两层，八头，FFN256。** 一套共享384→64 affine分别处理每个不同query
   和bank值；最终latent/process与field-axis/local、GPP carbon在当前调用内转换/cache。
   reader读取所属专家32latent+1process及原有query-private axis/local，GPP另读raw carbon；
   每层对所有合法segments使用一次联合softmax，随后residual/norm、FFN/norm。
   读出64→256→1，所有字段使用同一路径，恢复full shape及所有适用性掩码。
3. 保留FP32神经、FP64物理与既有归一化策略。latent每头48维；reader每头8维，
   cosine-Q/K分数的温度随head-width成为sqrt(8)，旧decoder为sqrt(48)。
   这是显式架构改变，不能隐称数值等价。epsilon和物理因子不因提速而调整。

32是首个有限容量假设：给21槽共同bank多于一组的学习查询，避免在每个坐标上重复宽交互；
64使私有矩阵项按宽度平方明显缩小。二者没有科学最优性证据。
本候选总参数略增，收益来自计算位置改变。它与统一缩宽128的区别是仍有384维的共享非线性交互。
本包不保留第二个候选；没有足够证据支持一个实质不同、同样可核算的备用结构。

**信息权限必须成立。** family先按专家source-mask筛选，再pool、bridge及bias后remask。
20是槽数，各专家可读的family子集不同。latent不新增direct axis/local/raw-carbon bank；
现有process已经经backbone读取合法轴/PFT信息，family也有合法全局摘要，因此不能宣称latent
完全没有轴/PFT依赖。GPP direct raw-mask仍为defined且owner=-1或query同PFT，再与query-active
相交；其他输出不含raw segment。原有全局摘要可以合法依赖其他PFT，禁止路径测试必须隔离
direct raw route或追踪完整来源闭包，不能一概要求GPP对所有其他PFT的端到端导数为零。

latent、投影与K/V必须每专家/日构建一次，保持JAX参数/状态依赖；不得每field/chunk重复，
不得host固化、stop-gradient或跨日/跨更新复用。reader的local-defined与coordinate-active
是不同掩码；空bank、padding、inactive/undefined、PFT重排和输出变换保持有限/零语义。
跨日记忆继续由物理状态承担，完整T7/T30递归、公共loss/AD和retained exact tail保留。

**真实容量风险。** 宽latent看不到一个坐标的私有local值，坐标特异的宽非线性变成64维。
输入/输出身份保留与参数数目增加，都不能证明局部碳响应、库存差异或GPP条件交互仍足够。
32latent还改变共同bank的softmax竞争。本候选需要自己的可学习性与科学证据，
不以逐坐标复现旧随机网络函数作为新架构验收条件。

## 4. 参数与MAC账本

参数从当前初始化字典的symbolic shape核验：300个数值leaf、两个PFT-position静态None，
**64,884,481**。tokenizer3,667,840；八backbone block56,702,976；
两个旧coordinate block3,548,928；其余964,737。

一个attention+FFN4w+两norm block为 `12w²+13w`；
含cross+self attention+FFN4w+三norm的latent block为 `16w²+19w`。
候选参数逐项为：
`64,884,481 - 3,548,928 - 98,817 + 12,288 + 4,733,184 + 24,640 + 99,968 + 16,897
= 66,123,713`。
依次替换旧两block/readout，加入共享slot、两latent block、共享384→64、
两reader block和新readout。增加1,239,232（1.91%）；每份FP32参数/梯度树约多4.96MB，
两AdamW moment约多9.91MB。原设计数值按声明拓扑推导；后续公共initializer的实际
eval-shape核验已确认66,123,713、FP32 leaves及选定轴，证据在
[原型汇总](shared_latent_reader_results_20261004.md)。MAC仍是源形状估计。

下表是**每样本日源形状的dense/dot前向MAC**，一乘加计一个MAC；masked/padded dots照源形状计入。
不是GPU kernel计数，不含pool/reduction、bias/add/embedding、GELU/norm/softmax、
mask/gather/scatter、physics、loss、AD/重算、优化器或编译。
XLA可能消除或改变这些运算，不能用本表乘一个固定反向系数得到更新时间。

| 项目（G MAC） | 当前expert128/source-reuse | 候选latent32/reader64 |
| --- | ---: | ---: |
| tokenizer dense+self-attention | 7.317451 | 7.317451 |
| 输入投影+backbone | 13.738906 | 13.738906 |
| 768→384 context bridge | 0.332956 | 0.332956 |
| 保留coordinate/field-local/raw-carbon输入编码 | 0.010894 | 0.010894 |
| 宽coordinate两层Q/O | 5.140906 | 0 |
| 宽coordinate两层FFN | 20.563624 | 0 |
| 旧family/process/raw/field-axis/local K/V | 4.224319 | 0 |
| 旧coordinate attention score/value | 0.363368 | 0 |
| 旧readout | 0.859049 | 0 |
| 新latent cross+self+FFN（含全部投影） | 0 | 1.029784 |
| 共享384→64，15,796个query/保留field-bank身份行 | 0 | 0.388202 |
| 新reader K/V、Q/O、FFN、attention及readout | 0 | 1.065104 |
| **合计** | **52.551473** | **23.883296** |

新FFN合计0.528482+0.571212=1.099694G，旧coordinate FFN20.563624G，
减少94.65%；全部计入神经dots减少54.55%，算术比2.200344。
本候选静态账本沿用expert128的Rpad=8,716。当前canonical field执行有不同
padding和调度；23.883296G不能冒充该实际图的完整运算数。参数数目不受此影响，
真实GPU时间以原型实测为准，不为重算静态MAC推迟真实测量。
无coordinate padding下界分别52.039392/23.865398G。
保留tokenizer/输入/backbone/bridge共21.389312G（当前计数40.70%）；
即使其余全部消失，这个口径的理想比也只有2.456903。**这不是完整AD的时间上界。**
axial tokenizer的padding及backbone axis K/V仍是明显的源形状成本，未被本方案解决；
没有追加另一个优化路线或声称这些padding都实际执行。

| 同工作量前向账本（T MAC） | 当前 | 候选 |
| --- | ---: | ---: |
| daily32，八个不同micro4 | 1.681647 | 0.764265 |
| 完整T7 | 0.367860 | 0.167183 |
| 完整T30 | 1.576544 | 0.716499 |
| **一次更新的69 sample-day前向等价** | **3.626052** | **1.647947** |

69=32独立日+7+30递归日；37个递归转换不能当成可任意独立并行的37条日样本。
没有重算/反向计数，也没有把这张表称为真实完整更新MAC。

## 5. 激活增长与完整更新收益的限制

| 一层逻辑激活，micro4 FP32 | 当前 | 候选 |
| --- | ---: | ---: |
| 全coordinate query | [4,8716,1,384]，53.551MB | [4,8716,1,64]，8.925MB |
| 全coordinate FFN hidden | [4,8716,1536]，214.204MB | [4,8716,256]，35.701MB |
| expert5单128行FFN hidden | 3.146MB | 0.524MB |
| 新wide latent / FFN hidden | — | [4,7,32,384] 1.376MB / 5.505MB |
| 保留384 coordinate features / raw carbon | 52.531MB / 14.660MB | 同左；另64 raw carbon为2.443MB |
| latent cross / self scores | — | 0.602MB / 0.918MB |

这些是“假设所有行拼在一起”的单层体积；lax.map顺序执行及AD residual改变同时驻留量，
不能将其相加为峰值显存。主要项随B与R线性增长；latent dense为O(BEMD²)，
self scores为O(BEhM²)，私有reader dense为O(BRu²)，reader scores为O(BhR(M+nprivate+1))。
GPP额外为O(BhRgpp×2386)分数；保留原始carbon编码与物理状态。
日窗口H在线性前向计数上乘H；不checkpoint的AD保存量可能随BH增长，
当前公共transition-checkpoint仍会发生重算。新方案没有实测AD峰值，也不能据此宣布更大batch可装入。

完整更新用 `S = 1 / ((1-f) + f/r + δ)` 做敏感性：
f是实际更新中被重构部分的时间份额，r是该部分**实测AD**速度比，
δ是新增/调度净成本占原更新的比例。当前没有测出f、r或δ；
2.200344源MAC比和81.43%额外AD均不能代入f或r。

| 假设，非预测；δ统一取0.03 | 完整速度比S | 以12.940825秒作情景基准 |
| --- | ---: | ---: |
| f=0.50，r=2 | 1.282x | 10.094秒 |
| f=0.75，r=4 | 2.139x | 6.050秒 |
| f=0.90，r=8 | 4.124x | 3.138秒 |

同样δ下即使r无限大，三种f的上限分别1.887/3.571/7.692x。
10x要求 `(1-f)+f/r+δ≤0.1`；没有证据表明本结构满足。
因此建议为累积改进候选，不承诺10倍或把本轮静态结果当实际更新改善。
未知项包括decoder AD份额、tiny-query/kernel利用率、tokenizer/backend消除、physics/retained-tail、
完整损失、checkpoint residual及调度。用候选局部AD与完整更新配对测量判断即可，
不要求先精确拆清每个内部耗时。

## 6. 最小验证顺序：先否定错误结构，再投入真实编译

以下是原始分阶段设计。当前已选原型的实现、必要真实GPU图和技术性连续更新
按NEXT_STEPS执行，资源采用新的统一5 GPU小时/单次2小时上限，取代下表的分项提议。
私有响应256步拟合及正式/长期学习并未随原型选择自动获批。
保留数据、损失、精度、容差、采样和完整递归；新架构与旧架构各用自己的fresh权重。
同架构实现等价比较沿用现有严格门；跨架构不要求旧随机函数的loss/gradient逐坐标相等。
新模型仍要有独立AD、物理和科学验收，不用较小训练loss代替这些门。

| 阶段 | 最少检查与明确决策 | 提议的资源/预算 |
| --- | --- | --- |
| 小布局技术否定 | 两专家、两个active PFT+inactive、不同同形状field/local、undefined local、global/两PFT carbon、空bank；检查full scatter、mask、合法/禁止source闭包、隔离direct raw route、联合softmax、PFT/coordinate一致置换及参数/状态JVP-VJP/有限差分。禁止路径非零、NaN、断AD或错误恢复则先修实现；结构做不到则停止。 | 本地CPU≤300秒，tiny layout/必要的eval-shape；保留真实width/head定义 |
| 私有响应容量否定 | 预声明fresh小问题：固定共同context，改变多个field/轴的local值，包含场依赖非线性、local×context交互及defined切换；同输入/预算384 reader对照。固定两seed、最多256步和完整学习曲线；目标/尺度/停止标准在运行前写定。对照能拟合而候选持续失去局部响应则停止A，讨论私有通路；两者都未学会则该证据不具判别力。 | 首选本地CPU≤600秒；本阶段权重更新需独立学习授权，超时记未测，不判容量失败 |
| 首个真实T7全AD | 先部署runtime preflight；只编译一个候选T7，保存binary/ABI/结果。完整loss、所有参数/状态导数、物理/finite/mask、≤24GiB、三组稳定warm速度；验证实际参数/状态依赖贯通。候选不达技术门或T7无可重复收益，暂停昂贵扩展并据实际成本判断。 | 免费单V100、OMP4/OpenBLAS1，提议累计4200秒；旧T7 lower146s/backend2528s及加载成本支持此规划，未承诺新图同样耗时 |
| 同工作量完整更新验收 | 复用T7结果，补daily micro4和T30；daily32/eight不同micro4+完整T7/T30，公共loss、AdamW、完整梯度与物理检查；至少三组交替配对、相同卡，报告两模型各自loss/梯度/显存/稳定时间，固定权重的proposal先discard。若有效更新速度低于1.5x且无显存/科学理由，停止该提速路线；1.5x是本提案工程优先级门，不是旧科学容差。 | 提议另一累计7200秒免费单V100窗口，仅必要的新图一次/图；旧daily/T30 backend约720/2546秒作预算依据，缓存可选，编译成功成果保存 |
| fresh学习与长期科学 | 技术+同工作量速度值得继续后，单独声明fresh种子/初始化、full-network规则、相同daily/T7/T30 mix、学习/停止与选择标准；证明学习和物理状态反馈。固定development的flux/stock/累计/终点与daily tendency，再365日、季节/年际、留出点/年及D2、最终sealed验收。短16步或全零RH面板不足，应覆盖有定义且非零呼吸条件。 | 另作科学/资源选择；本包不批准正式训练或长期运行 |

固定权重proposal用于技术筛查。最终吞吐验收仍要在获批的fresh学习阶段实测changing-weight
完整更新，包含所有实际AdamW工作，比较至少15个first-update之后的更新；模型各自学习结果
另按预声明科学指标判断，不能把discard的proposal当成已完成学习或最终训练速度验收。

预算包含启动、lower/backend、普通修复、保存、退出，以持久累计deadline控制，不每次重置；
超时是未取得结果，不能据此宣布架构无效。真实GPU请求要在实现、CPU/目标shell检查及
实际命令准备后具体化；付费后备仍逐项批准资源/时间/目录/命令/最坏CPU+GPU费用。
同架构代码问题在获批窗口内自行修；改变M/u、attention/精度/损失或采样则返回科学选择。
缓存复用只提高效率，重新编译一次并保存结果可作为合理选择。

首个真实T7的当前执行选择是canonical `latent_reader_execution="field"`。
expert128 grouping的小布局严格参数AD有一个坐标失败；有界context-VJP发现
差异已在decoder汇合处存在，但未确定具体浮点指令原因。保留失败并停止深入追查，
逐字段路径保持同一M32/u64架构、合法信息路由和全部动态参数/状态依赖。
它不采用未通过的grouping，也不放宽精度或容差；真实速度必须据实测判断。

## 7. 版本纠正与可复算证据

| 旧证据的正确scope | original→candidate warm | candidate lower / backend |
| --- | --- | --- |
| [daily micro4 V4，仅参数AD](shared_chunk_vmap_daily_results_20261001.md) | 0.375510→0.221913秒，1.692144x | 101.564 / 719.661秒 |
| 同报告后续daily完整参数/状态AD | 0.382541→0.228810秒，1.671871x | 不与上行混为同一scope |
| [真实T7完整参数/状态AD](shared_chunk_vmap_recurrent_results_20261001.md) | 1.578531723→0.906023597秒，1.742263x | 146.183800 / 2527.957784秒 |

0.221913秒及719.661秒原来被此报告误标为T7，现已纠正。
组件收益、16步更新收益和本轮源MAC各有独立scope，不相互代替。

主证据为 [audit.json](../../../outputs/diagnostics/shared-decoder-structure-20261004/audit.json)、
[300-leaf参数形状](../../../outputs/diagnostics/shared-decoder-structure-20261004/parameter-shapes.json)、
[独立设计审查](../../../outputs/diagnostics/shared-decoder-structure-20261004/design-review.md)；
生成脚本 [audit_shared_decoder_structure.py](../../../scripts/hpc/audit_shared_decoder_structure.py)。
JSON保留逐字段/expert、每项公式、源码位置、静态资产核对及实际CPU命令。
原始静态核算阶段的模型/GPU/update计数均为零；其CPU耗时只说明核算成本，不是模型性能。
当前实现与实测须另记录真实结果，不能将本节原始MAC账本冒充速度。

源码定位（行号供核查，函数与当前source为准）：
shared_transformer.py初始化148–279、family/bridge664–702、projection/cache379–448、
coordinate block537–607、dynamic identities810–967、query970–1043、carbon1090–1138、
GPP/raw mask及decode1141–1329；shared_decoder_expert_execution.py60–241、
shared_decoder_macro_packed.py25–76及288–371；
shared_backbone_projection_reuse.py19–83；
strong_baseline_architecture.py forcing993–1037、axial1040–1097；
shared_recurrent_training.py152–240及rollout_loss.py79–93。
Graph Verify按exact symbols定位和coverage检查，metadata_changed后用当前精确source回退；
这是有界核查，不声称全代码/所有Fortran分支完整性。

**已选研究原型：每expert共享32×384 latent，同时保留64×2私有reader。**
此次包括有界技术性连续更新与免费实测，仍不批准正式/长期学习或付费作业。
不得自行变更M32/u64或科学接口；保留原模型与证据，依据真实T7及≥1.5x同工作量
完整更新工程门决定是否值得进入另行固定的fresh学习与365日科学计划。
