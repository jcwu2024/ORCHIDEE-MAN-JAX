# 主线进度

2026-11-24，**已验收 001 暖点 1961 年冷启动的两日参考。JAX 在这个窗口仍有超差。** 留存的 Fortran 程序、按原配方重建的程序、加观测的重建程序和本地 JAX 都完成了 96 个半小时步。三套 Fortran 有源定义的重启末态逐值一致。加观测的程序实读水文私有泥炭开关为关，首日抽样步只有 4 号土块参与求解。首日 PFT14 的 GPP 和 NPP 都为零，第二天结束由三份重启确认。年度通量和 2025 年原作业的实际开关仍缺对应证据。[两日实测](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-two-day-reference/README.md) · [主控验收](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-two-day-reference/review/controller-acceptance.json)

主线 JAX 的常规入口遇到原气象文件的两条时间轴时在第一步前退出。此次在包内按留存 IOIPSL 的顺序选轴后运行了生产计算。1,369 条两版配对记录中，1,287 条通过严格门限，4 条入射短波在其已登记容差内，78 条超过相应门限。第一步净短波差 2.66×10⁻⁸ W/m² 可由两边各自的入射量和同一个吸收系数复现。两日末有 30 个重启字段超差，另有 47 个 STOMATE 字段由 JAX 写出默认值补齐。接下来核对正常气象读取入口、相同辐射输入下的热通量，以及土壤容重和融土湿度从写出到重新读取时的实际值。[逐项比较](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-two-day-reference/compare/RESULTS.md)

2026-11-24，**001 点原作业只读取证已验收。** 作业 `98386440` 的启动路径仍保留一份 Fortran 程序。其私有 `peat_hydro` 在已核对的水文路径上为关闭，与当前 JAX 论文入口一致。远端和本机的 20 份原件字节一致，包括 Slurm 日志和 1961 年史。已查目录缺当次程序副本、1961 有效配置及逐步日志，年史只有一个全年样本。当年作业实际读值仍缺直接证据。日志里批量清理年度目录的指令是注释，`out1961` 缺失的原因待证。[主控验收](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-cancon-originals-resumed/review/controller-acceptance.json) · [取证报告](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-001-cancon-originals-resumed/README.md) · [两日方案](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-trace-reader-and-reference-preflight/observation/README.md)

第十八包的追踪工具修正 `17db2adc` 已验收。独立 1961 追踪的土块求解掩码按原写出读为 `resolv`，144 条实档和 24 份旧报告回放保留原数值和采样位置。累计模型修正或功能补齐 92 项、48 个生产提交。[本地补查](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-trace-reader-and-reference-preflight/review/controller-acceptance.json) · [后续分支同步](shared_core_branch_review.md)

2026-11-24，**第十八包已验收，物理阶段部分交付。** 17 项本地待办完成 11 项，余 6 项需要同一原运行的水文观测或真实诊断端点。已接受土温诊断和末态地表温度写回两项限定验证。三处代表点已匹配原输入和参考文件。正式整模型验收仍需确认所选论文构建的水文私有泥炭开关，并配对同一步的水热接收和日末碳写回。此包更新两份测试夹具，累计登记 92 项修正或支持补齐、48 个生产提交，其中 90 项有复查前证据。[主控验收](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-physics-terminal-review/controller-acceptance.json) · [代表点与最短观测](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase18-physics-terminal-review/representative-gate.md)

2026-11-24，**物理检查第十七包已验收。** 上一包留下 50 项本地动作，此次完成 33 项，另 17 项仍缺具名证据。冠层叶内 CO₂ 分母的回退和无 priming 土壤碳分解在相应组件输入上通过，剩余 72 个源码组和 33 个调用组，其中 12 个调用涉及外部后端。本轮只提交审计记录。累计修正或支持补齐 **92 项、48 个生产提交，其中 90 项有复查前证据**。[主控验收](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase17-remaining-local-actions/controller-acceptance.json) · [发布记录](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase17-remaining-local-actions/controller-publication.json)

实测中，原 Fortran 与 JAX 的 584 个单点潮位记录逐项对应，冠层在分母过小时写回同一叶内 CO₂ 值。备用蒸发的 56 项差异对临时土柱储水量敏感，热通量有 49 个坐标仍超原门限。第十八包结合论文的有效配置和实际调用次序，对 17 项本地余项及全表其余依赖做物理阶段终审，判断开始整模型验证还需要哪些证据。[逐组余项](../../outputs/reviews/main-physics-20261124/physics-branch-closeout/phase17-remaining-local-actions/open-groups.json)

2026-11-24，**物理检查第十六包已验收。** C91 修正灌溉的格点和需求检查，C92 修正无冻结泥炭选导水系数时的浮点运算顺序。灌溉 6 组原生对照一致。泥炭同输入第 4 层由第 32 档恢复到原第 33 档，关联的 14 项差异消失。42 项本地回归通过。累计 **92 项修正或支持补齐、48 个生产提交，其中 90 项有复查前证据**。[主控验收](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase16-soil-thermal-and-hydrology/controller-acceptance.json) · [累计修正](correction_origins.md) · [发布记录](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase16-soil-thermal-and-hydrology/controller-publication.json)

本包 180 项中办结 147 项，15 个源码组在相应组件和输入范围内完成。各通道合计还有 50 项本地动作，分布在 21 个组。全部余项为 74 个源码组和 33 个调用组，其中 12 个调用涉及外部后端。下包集中处理其余 17 项冠层、土壤碳和输入读取检查，同时推进 33 项水热余项中可复用同输入或现有记录的部分。自然年际轨迹和年度通量影响留待综合验证。[逐组余项](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase16-soil-thermal-and-hydrology/open-groups.json)

2026-11-23，**物理检查第十五包已验收。** C90 修正季节统计的年末结转。旧 JAX 在主时钟已到年末时仍可能用普通日逻辑处理降水等统计量。五种两格点同输入情形各有 41 个字段与 Fortran 相符，年度 NPP、GPP 和 DOC 的实际影响待综合验证。累计 **90 项修正或支持补齐、47 个生产提交，其中 88 项有复查前证据**。[主控验收](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase15-season-thermal-doc-and-ci/controller-acceptance.json) · [累计修正](correction_origins.md)

土温诊断的裸地权重在单点矿质无雪范围内完成核对。DOC 的导出和扣库分别在给定底排跨过约 30.03 和 45.04 mm/步时受库存限制，两侧相符。还余 89 个源码组和 33 个调用组。下包集中查土壤热传导和水文分支，可复用输入的两组工作共涉及 250 个原条件。本地发布流程 16 批共 1,417 次通过、227 次预期失败、26 次跳过。[逐组余项](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase15-season-thermal-doc-and-ci/open-groups.json) · [发布记录](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase15-season-thermal-doc-and-ci/controller-publication.json)

2026-11-23，**物理检查第十四包已验收。** C87 修正日末和重启中的叶龄组比例取样，C88 修正固定季节过程接收的日平均值，C89 补齐通用凋落物接口的前次日量。两套论文组件样例从 108/118 项通过变为 118/118。累计 **89 项修正或支持补齐、46 个生产提交**，其中 87 项有复查前证据。[主控验收](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase14-water-heat-carbon-receivers/controller-acceptance.json) · [累计修正](correction_origins.md)

本地 94 项回归通过。原 Fortran 的四处径流 DOC 库存限制取得组件对照，热状态的写回也传到了下一次冻结和液态水计算。仍余 90 个源码组、33 个调用组，后者含 12 个外部缺口。第十五包优先检查季节过程的实际返回和 LPJ 接收、裸地面积为零时的土温及 DOC 排水限制。[具名余项](../../outputs/reviews/main-physics-20261123/physics-branch-closeout/phase14-water-heat-carbon-receivers/open-groups.json)

2026-11-23，**物理检查第十三包已验收。** C86 修正植被碳模块写出最终状态的两个入口。主控单点复验中，植被覆盖调整后的 GPP 和两个输出均为 3.75。累计 **86 项修正或支持补齐、45 个生产提交**，其中 84 项有复查前证据。[主控验收](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase13-vegetation-carbon-receivers/controller-acceptance.json) · [累计修正](correction_origins.md)

第十二包的 6 项叶片差异来自进入下游时已不同的状态。分别固定每套输入后，原 Fortran 与 JAX 的周转、最终叶面积和最大羧化速率均在原容差内相符。该格点在两套输入下分别得到 50 和约 26.67，原始极端叶片比例与另设的合法对照分别保留。自然轨迹和全年影响仍待综合验证。余 90 个源码组、33 个调用组，其中 12 个调用组缺外部证据。第十四包优先核对水、热、碳过程间的实际传值。[叶片对照](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase13-vegetation-carbon-receivers/lai-vmax/README.md) · [具名余项](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase13-vegetation-carbon-receivers/open-groups.json)

2026-11-22，**物理检查第十二包已验收。** C84 修正根系导水率公式，C85 补齐入渗参数轴，通用耦合诊断的配置传递归入 C83。累计 **85 项修正或支持补齐、44 个生产提交**，其中 83 项有复查前证据。两个研究分支的同步依赖已登记。[主控验收](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase12-root-water-and-vegetation-carbon/controller-acceptance.json) · [累计修正](correction_origins.md)

2026-11-20，第六包经主控接受，五项修补登记C73–C77，生产／测试／CI提交`f724dd1a6cf8f4edb3b5d1a1eeb780783d8d3e7b`。累计**77项修正／支持补齐、39个相关生产提交**；75项有复查前错误或支持缺口证据，C42／C57为复查期间遗漏，不是77项物理公式错误。本轮五项均有复查前`7cae3e02adad23674767a590fdf4d27f239ce065`只读源码证据，首次引入日期未断言，未恢复运行历史代码。

| 工作组 | 当前结果 | 尚需工作 |
| --- | --- | --- |
| 参数／状态 | 既有有限合同已接受；新增6份科学restart文件到实际reader／下一组件的有限往返 | 真实IOIPSL／native producer与自然年际历史等范围另留 |
| PB-01冠层／光合 | 新增restart／held／维护LAI前驱与有限消费者 | 自然低LAI历史、外部PBL及fixedsnow原未定义合同 |
| PB-02能量／QSAT | 实际热容量和通量写回后进入下一次 ENERBIL，低风气动差异已有有界归因 | 31 项默认 JIT 气动严格失败，及其余源域与外部 PBL |
| PB-03水文／热／雪 | C91、C92 已登记。三种导热率接线、裸土蒸发对实际土柱含水的读取已对照，15 个源码组限定闭合 | 水热余 33 项本地动作。热通量和土柱水量预算的严格残差、自然潮汐和泥炭前驱、原外送接口仍有缺项 |
| PB-04植被／碳 | C86–C90 涵盖 LPJ 最终输出、叶龄和日量取样、季节输入及年界。径流和给定底排的 DOC 库存限制已有原生对照 | 完整 LPJ 原生接收、自然跨年和排水前驱、年度 DOC 收支待验证 |
| PB-05跨组 | C75–C77输入处理已修，forcing／静态控制／modelout／routing与restart有限接口验证 | maintained完整调度、跨模块源调用归属及阶段终审 |

[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-acceptance.json)核对46份产品、交付与独审封存身份，并复核配对原源、实际写回及历史源码。保存的隔离CI按精确节点合并为454通过、9项科学strict xfail、10项外部身份skip、0失败／错误；这不是一次不间断全通过运行。首次漏复制已跟踪restart schema的9个setup错误保留，修正复制后仅重测受影响的10节点，未重测此前通过项。配置内Ruff通过，生产lint 62前／62后无新增；hosted CI未运行。主控解析JUnit并复用owner／独审验证，未再积分；12份JSON仅Git换行规范化、语义相同。[发布记录](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/controller-publication.json)保留最终版本。

保存的嵌套JIT整返回差异现有有界递推取整与表面通量抵消证据；13/14 soilflx_pft及其别名仍严格失败，不声明唯一HLO／机器指令原因。两个新增合成相等浓度DOC元组捕获到入口约1e-17取整变化，实测入口重放能复现严格相等分支变化及4／5个失败坐标；没有定位最早上游算术，也不外推旧年度轨迹。第六包验收时，8项新JIT完整系数通量门／60坐标留待第七包归因。旧HYDROL／热／雪／低光／DOC／NPP／grain严格门各守原范围，不同配置DGVM诊断不计新修复。

52父过程的整体路线等级仍为source_mapped／open；4,059词法候选、565源审查组和187跨模块／外部调用组不等于数值覆盖或缺陷数。新有限证据包括14生长元组2611比较、扩散实际arm、非零冻融carry、冠层／capacity前驱、forcing／静态输入／modelout／routing和6份科学restart文件往返（1402检查，含147独立字段）。真实IOIPSL／native producer、maintained完整调度、自然历史、私有peat／PBL、原未定义合同与完整源分母仍有缺证。C64限npts1／nstm6；年度DOC真实边界／残差／异养呼吸均为null，原mass warning与负中间量保留。

第七包已通过[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase07-source-dispositions-and-energy/controller-acceptance.json)。两组相同输入下，土壤底层递推系数在二进制浮点的最后一位出现差异，传到表层后，两项接近的量相减使热通量出现残差。保存的操作数、两个输出别名及雪土加权结果支持这一解释。8项比较的60个坐标仍按原容差记录为失败。该结论限于这两组输入。新增3项隔离回归通过，外部原源身份检查在隔离环境跳过、在本地通过，配置的Ruff检查通过。本轮只增加测试、原参考和CI配置，累计仍为77项修正或支持补齐、39个生产提交。

第八包登记 C78 冻结水文分箱、C79 泥炭湿度分箱，代码提交 `9591652668df1afb4955d7282a6091b96d844e4b`。上次批量复核有 90 个失败坐标记录（含重复单点），修订后的同一脚本 18 行结果全部通过。原生边界核对、实际含水量写回和 STOMATE 分解率均使用原参考。隔离测试 18 项通过，1 项外部原源身份检查跳过，该项在本地通过。[主控验收](../../outputs/reviews/main-physics-20261120/physics-branch-closeout/phase08-hydrol-snow-and-consumers/controller-acceptance.json) · [累计修复及同步依赖](correction_origins.md)

第九包登记 C80。暖干 PFT14 的论文分解率使用原构建的双精度下限，实际凋落物和土壤碳写回在匹配输入上通过。组件对照 348 项、本地测试 48 项通过。隔离环境 8 项通过、1 项外部源码身份检查跳过。剩余 142 个源码组和 34 个调用组，其中 12 个调用组缺外部证据。原局部收支警告仍出现，年度 DOC 和 NPP 影响未知。整模型 `numerical_acceptance=false`。[主控验收](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase09-carbon-activity-and-consumers/controller-acceptance.json) · [剩余清单](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase09-carbon-activity-and-consumers/open-groups.json)

第十包接受 C81 叶层 CO₂ 重启写回。36 组同输入下，修后完整字段和 11 项公开物理量通过原容差。50 个冠层与能量源码组中 37 组获得有限支持，另有 4 个相邻组获支持。冠层结果传到下一次气动计算的调用已核对。剩余 101 个源码组、33 个调用组，其中 12 个调用组缺外部证据。低风近中性时 31 项严格气动对照仍失败。年度 NPP/DOC 影响未知。[主控验收](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase10-canopy-energy-and-consumers/controller-acceptance.json) · [剩余清单](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase10-canopy-energy-and-consumers/open-groups.json)

第十一包核对 707 个原源码条件，剩余 92 个源码组及 33 个调用组，含 12 个外部缺口。实际水文含水状态和雪字段进入热土壤，后者的容量、通量和网格温度写回并被下一次能量计算读取。新增土柱对比保留 16 项严格失败。低风气动的 31 项失败追到默认 CPU JIT 的融合舍入，原科学容差继续适用。五处重启夹具补齐后，完整文件和两项生产回归纳入本地 CI 入口。[覆盖差额](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase11-soil-water-heat-and-aerodynamics/coverage-delta.json) · [独立复核](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase11-soil-water-heat-and-aerodynamics/review/independent-review.md)

第十一包登记 C82 基岩热容量的冻土开关、C83 水文土类表的逐点索引。主控用相同数组核实，初交超限的 26 个字段在返修后全部通过，单点参考保持一致。混合及重排的 12 点经 212 项对照，在原容差内匹配。年度 NPP、DOC 影响待验证。下一包优先核实配置导水率的根系公式及其真实调用，并行推进土柱严格差异和植被碳的局部收尾。[主控验收](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase11-soil-water-heat-and-aerodynamics/controller-acceptance.json) · [修订余项](../../outputs/reviews/main-physics-20261121/physics-branch-closeout/phase11-soil-water-heat-and-aerodynamics/controller-repair/open-groups.json)

第十二包对账 27 个仍获部分支持的源码组及 1,026 个原成员。土柱 270 项对照中保留 16 项严格差异。植被碳的两组上游结果送入同一个 JAX 接收方后，第二次调用的周转和 Vmax 在同一格点出现 6 项差异。第十三包先把相同完整实参分别交给原 Fortran 和 JAX，核实接收方的计算，再沿实际调用核对碳过程写回。余下 92 个源码组、33 个调用组，其中 12 处调用缺外部证据。全年 NPP、GPP、DOC 通量与收支影响待验证。[具名余项](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase12-root-water-and-vegetation-carbon/open-groups.json) · [下游对照](../../outputs/reviews/main-physics-20261122/physics-branch-closeout/phase12-root-water-and-vegetation-carbon/npp-lpj/README.md)

修复位置与未来同步依赖见[来源登记](correction_origins.md)、[累计索引](shared_core_branch_review.md#累计修改位置与提交索引)和[当前覆盖表](physics_branch_coverage.md)。Sol每包封存后回传一次，主控集中验收发布后继续派发，不定时轮询。565／187是源码审阅分组，不是752个已知bug，也不是要求752次模型运行。
