# main 物理正确性检查入口

核查日期：2026-10-16。**本轮没有发现已确认的 main 生产物理实现 bug；
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
DIFFUCO读取，保留48步邻域；已准备字段、插入点、命令和资源方案，尚未实现或执行。
局部同输入闭合不能确定全年NPP因果。
命名日值 trace 的具体历史文件身份已取得；两条长期链 producer 和年度因果
仍未全部核定，尚未确认源级错误。**
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
| 完整七日块 / executable reuse | [14-day cold / restart、365-day 记录](../../outputs/reviews/main-physics-20261011/block-gates.json) 均 passed；[跨点 reuse](../../outputs/performance/compiled_fast_path_acceptance/reuse_3points_current.json) 全部 strict parity 通过且后续点/重复运行不新增 executable | cold 对 strict source order；restart/annual 对 compiled daily no-block，浮点 `1e-8 + 1e-10*abs(ref)`、离散/模式精确。公共 [CLI](../../jax_orchidee/cli/main.py) 注入 on / block=7；底层 multiyear 的 block=0 默认不能据此判为公共 CLI 未启用。 |
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
保留 raw 配置与 HYDROL 局部 SAVE 的区别；319 有效 Fortran 局部值仍未核定，
未据 case001 False trace 改开关。**本阶段没有确认可达源级物理错误或提出生产修复。**
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
trace为False均不能证明319本次Fortran局部值。该值仍缺失，尚未确认配置不匹配。
原生液水/阈值、动态根、clip、tile缺席及under-mcr均是未决区分量。最终比较器两侧
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

## 已确认限制与优先检查清单

完整 669 点验收未完成。保存 certificate 的 `equivalent=false` 同时引用旧
stage6 2/3 代表点结果，不能把该旧 false 直接当作当前实现失败；反过来也不能把
七点通过扩写为 669 点通过。`process_span_unmapped=4732` 是诊断性 span 计数，
证书已有另一套 arm 分类/owner 路由，不是 4,732 个未实现过程。
HYDROL peat hydraulic coefficient 条目虽然命名含 conditional，但实际 reachability
为 inactive，并给出了 audited runtime `peat_hydro=F` 的来源理由；其 regression-only
不能脱离有效配置解释成可达缺口。该 False 的点位范围须保留：第三阶段仅确认
所关联 case001 trace；对 319 的有效 Fortran 局部值尚未建立身份桥，见上节。
更一般的潜在输入域仍受原交付合同约束。

下列四项按下一步价值排序；它们是检查对象，非四个待修 bug。

1. **证据到当前物理版本的关联（证据待补）。**
   七点、source/owner 和 state 档案已有很强的通过证据，但若干原始 JSON 未内嵌
   JAX producer SHA。入口为本轮 inventory 的 production_change_history、
   `scripts/dev/audit_pft14_equivalence_certificate.py::build_certificate`、
   `deba027` / PFT identity 与 restart 提交 / `5910ba9` 以及已接受 C1–C3。
   本轮以 HYDROL 示例核对原 Fortran身份、当前函数 AST 和具体 guards。
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
   成对源链支持HYDROL创建→下步TRANS；关键原生输入/写读桥缺失。下一唯一优先方向为
   已准备的offset6–8原生HYDROL/carry观测及48步轻量上下文，不另做孤立光合作用重放。
   humrel创建来源及年度因果未定；原NPP核心回放与本次局部证据均不扩展为全分支正确性证明。
   不从科学失败或最后半小时解释比例直接认定物理函数 bug。
   当前 HYDROL/年边界已核实的来源和反证继续有效，不从旧失败直接修改 root 或门槛。

3. **当前三文件 restart 与年接续的完整状态合同（有通过证据，待版本关联）。**
   入口为 `_paper_day_end_state_packet`、`paper_1961_driver_restart_year_start_day_result`、
   [restart lifecycle 合同](restart_state_lifecycle.yaml) 和 stage5 的 181 字段记录；
   Fortran writer 来源为 ledger 所记 `stomate_io.f90::writerestart` 2200–2948，
   HYDROL writer / 年初始化范围如上。已有 001/069/319 全年直接/split restart
   与离散/模式检查通过，nroot 删除疑点已排除。
   最小办法对照 PFT identity/restart 接入后的 reader/writer/字段元数据，复用已有
   年边界与序列化检查；只针对实际变更核对 axes、mask、时间归属和重算字段。
   尚未独立重审全部 181 字段在 ee42eb1 的读写链，不能据此说状态缺失。

4. **七日 compiled 块跨点绑定与年输出（有通过证据，待当前入口/全点接入核对）。**
   入口为 `_paper_compiled_later_day_block_executable` 11185–11355、
   `_paper_run_compiled_later_day_blocks` 11358–11471、
   `read_forcing_model_step_cached`、`audit_paper_landpoint_binding`、
   `compute_modelout_from_fields`。账本给出 Fortran `sechiba_main` 997–1224、
   `forcing_read_interpol` 660–1687/1899–1968；本轮已核对 paper postprocessor
   48–76/115–118 的 PFT14 选择、碳库相加及 AGB/BGB `0.02` 转换。
   已有 14/365 日 A/B、三点 executable reuse、七点 block=7 的完整 50 年科学通过。
   当前 block 的 forcing/science day、参数、landpoint payload 与 restart template
   是显式动态参数，`year=1961` 的静态 metadata 本身不是锁定所有年 forcing 的证据。
   最小办法核对缓存绑定和公共 CLI→runner 的实际参数流，复用已有 forcing transport /
   block parity / landpoint binding 检查；669 点数据绑定审计与 669 点长期科学验收分别记录。
   尚未完成 ee42eb1 的完整跨点消费者审计及 669 点积分验收。

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
也不证明根因。下一方案仍为准备态，须实现观测器并验证自身轨迹后才有新科学证据。
本次验收未新增积分、重放、构建或分配；仅提交本页报告，保持手动跟进和原科学失败。
