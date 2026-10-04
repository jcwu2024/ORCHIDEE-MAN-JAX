# 全球物理模型吞吐研究

更新：2026-10-05。分支：`research/global-throughput`；起点：`main@5910ba9`。
本文件是本分支的研究入口和当前工作单；[EXECUTION_AUDIT.md](EXECUTION_AUDIT.md)
记录具体实现边界。日尺度神经算子研究继续由原分支管理。

## 目标与范围

在保持原有物理过程、48 个半小时时步、日 STOMATE 更新、精度和验证标准的前提下，
降低论文 669 个 PFT14 网格、1961–2010 年模拟的总耗时与计算成本。
这里的“全球”首先指现有 669 个独立论文网格；不代表全陆地 PFT 或开启横向耦合的全球模型。
若任务增加 routing 或网格间交换，必须重新审计独立性和通信需求。

优先研究正向模拟。可微性的后续用途是参数校准、敏感性分析与数据同化；
这些工作需另行定义可微参数、观测损失和长期梯度验证，不能由“使用 JAX”自动推定完成。

## 初步判断

有值得检验的结构性提速机会，核心是**同时计算一批网格**，而非只替换运行设备。
JAX 编译、数组批处理和设备分片支持这一实现；它们与自动微分是不同能力。
Fortran 本身也支持向量化、MPI/OpenMP 乃至 GPU，比较对象应明确为当前论文版本和当前运行方式。

建议计算结构：

```text
网格独立状态 + 各自 forcing/参数
  -> 按兼容的静态配置分组，装成一批
  -> 同一批内并行计算网格；每个网格按原顺序推进时间
  -> 块边界汇总输出、检查状态；必要时保存恢复点
  -> 单设备收益成立后，再把不同批分配到不同 CPU/GPU
```

50 年的状态递推有因果依赖，不能直接把每年作为互不相关任务并行。
`lax.scan` 能减少调度与数据传输，却不消除这种依赖。网格、独立情景和参数集合才是
最自然的并行轴。CPU SIMD/批处理仍是候选；GPU 不保证获胜。
物理模型应按实际 FP64 算力、访存和小 kernel 调度成本选择设备，
不能借用神经网络低精度 Tensor Core 的峰值来预测速度。

## 现有证据及限制

| 证据 | 已知结果 | 不能推导的结论 |
| --- | --- | --- |
| 主分支 README 与当前入口 | 已有七天完整块、编译复用；七个新网格完成 50 年验证，最大 2010 AGB/BGB/GPP/NPP 相对误差 8.5e-5；完整 669 点等价验收待完成 | 不能称全分支、全 PFT 或全部 669 点已经通过 |
| 主分支旧 SECHIBA 编译 A/B [报告](../../source_audits/compiled_fast_path_production_readiness.md) | 三点一年 compiled 79.33–94.16 s，strict JAX 237.84–268.63 s | 2.65–3.00 倍是相对 strict JAX；不是相对 Fortran，也不是当前七天块的新实测 |
| 更晚的完整多日块验收原始记录，2026-07-19 | 本地 Windows CPU；365 日热运行 31.1364 s，冷编译加首轮 158.13 s；冷启动/全年/restart 的有界状态门槛通过 | 缺精确 commit、CPU 型号和 JAX 版本；不是本次 main 实测，也不是同机 Fortran 对比 |
| 历史物理 Teacher GPU capture，2026-07-21 | V100 / JAX 0.4.38 / FP64；15 日热运行 4.840 s，29 日 7.631 s；冷运行约 903/923 s | 带额外训练数据采集；不是当前 main 纯正向基线，短窗热速率不是 50 年速度 |
| 历史 Teacher v6 CPU capture，2026-08-06 | 五个单 CPU 常驻 worker；热年约 82 s/点年，首次 capture 约 2058 s；换网格未增加可执行缓存项 | 历史研究提交、额外输出与不同环境；不能与上述 GPU 行直接作公平 A/B |
| 立项轻量检查 | 新 worktree 导入成功；ORCJAX Python 3.11.15 / JAX 0.10.1，仅显示 CPU；公开 `run --help` 成功 | 仅为立项环境检查；后续本地完整短窗实测见下 |

历史报告可在同仓库提交 `b0080d3215bce7cae349d1bcf76b37450ac94ceb` 中恢复：

```bash
git show b0080d3:docs/research/daily_coarse_graining/teacher_capture_gpu_benchmark_20260721.md
git show b0080d3:docs/research/daily_coarse_graining/teacher_v6_reuse_benchmark_20260806.md
```

GPU 两个原始 JSON 本次已核读：原工作区
`outputs/performance/daily_coarse_graining/compiled_training_capture_in_process_gpu_15d_optimized.json`
与 `compiled_training_capture_in_process_gpu_29d_block28.json`；记录冷/热时间、350 个末态叶子、
重复运行零差异和缓存计数。CPU 数据本次依据历史报告，未重新执行该任务。
其 31.2 GiB/worker 是包含编译的进程峰值，不能直接当作稳态状态内存。

更晚的主模型证据位于原工作区
`outputs/performance/compiled_multiday_block_acceptance_20260719.json` 与
`complete_day_landpoint_reuse_20260719.json`。后者是三点两天块的复用与短窗一致性，
不是三个点都完成七天块全年验收。不要用旧性能文档覆盖这些更晚结果；具体范围见执行审计。

用户提供的 Fortran 粗基线为 2 分钟/点年、近 2 小时/点/50 年。
按 669 点估算约 1115–1338 单核小时（每点占一核的假设）；这是原始经验值的算术换算，
不是同硬件复测或 JAX 节省量。2 分钟乘 50 等于 100 分钟，两个口述数字的差距须在
基线测量中由启动、IO、年份差异等实际记录解释。

## 当前工作单

首个本地实现已加入 `driver/global_batch.py` 与
`scripts/dev/benchmark_global_batch.py`：在 AOT 前提取原完整 transition，保持内层
`npts=1`，外层 `vmap` 的八组数值输入逐点堆叠，科学日序只在逐点相等后共享。
四个真实点 001/069/319/331 的各自 cold Day1、静态科学 closure、schema/dtype/None
审计已经通过；13 项轻量单元测试和新增文件 Ruff 检查通过。B1 原 AOT 与提取后的
两块完整状态/输出通过（760 leaves，有限值零差异，严格匹配原 bare-soil roughheight
NaN sentinel）。四点各自 context 的 scalar reference 在逐点清空内层 cache 后生成；
B4 逐点、置换和 lane 替换隔离通过，最大逐元素组合容差占比 0.08167。
物理 core 冻结在 `a33b82d`；独立参考来自 `b8bb285`，最终 harness `80831ea`
保全并复用这些参考，先验证共享 scalar 对每个独立参考零差异，再做完整 batch gate 和计时。

本地 Windows CPU、i7-13700（16物理/24可见逻辑核）、JAX/jaxlib 0.10.1、FP64，
每次从同一冻结 Day1 末态重置，连续推进两个7日块；五次热运行同步全部末态/输出和 effects：

| 同工作量 | batch 热 core 中位数 [min,max] | JAX 常驻串行中位数 [min,max] | batch 吞吐 |
| --- | --- | --- | --- |
| B1，14 point-days | 1.873 s [1.864,1.970] | 2.007 s [1.830,2.483] | 7.475 point-days/s |
| B4，56 point-days | 3.442 s [3.216,3.551] | 7.818 s [6.924,8.827] | 16.268 point-days/s |

B4 本次相对同工作量 JAX 串行的 core 比为 **2.271倍**。含已准备输入的 stack/transfer/
host materialization wrapper 中位数4.015 s；B4 lower23.575 s、compile95.345 s、
first call4.490 s。完整四份独立 scalar 编译进程峰值 RSS7.72 GiB，复用参考后最终计时
进程峰值5.30 GiB；GPU数与card-hours均0。初态准备和独立参考编译另计，wrapper不含
新cold-start、年度聚合或生产checkpoint。Fortran同机对比、GPU吞吐、年度/669/50年和AD
仍未由此短窗验收。聚合证据为 `outputs/research/global_throughput/local-milestone-summary.json`。

Explore1000免费诊断完成四点独立Day1/static/schema审计（test02，wall267.49 s）；
随后GPU `b8bb285`复用其准备输入，在gln01实际节点`ibc13b02n04`使用1张
V100-SXM2 32GB、affinity0–3、1200 s预算。原scalar reference在650.393 s保存，
extracted scalar compile在659.739 s开始；预算边界结束时尚未完成GPU batch parity或计时。
wrapper exit1，原report停在`running`，stderr仅有slow-compile alarms，不能解释为数值失败。
进程已退出；此免费GPU诊断预算1/3 card-hour，四核affinity容量4/3 core-hours，
快照实耗CPU下界0.375 core-hours、进程峰值RSS下界9.254623 GiB。1200个逐秒GPU样本
的节点card0峰值987 MiB，属于编译中设备观测，不是完整batch显存需求。
原证据在 `outputs/research/global_throughput/remote-gpu-b8bb285/`；后续launcher cache路径
隔离修复发生在运行结束后，不属于该运行条件。

本地与远端合并记录见
[MILESTONE.md](../../../outputs/research/global_throughput/MILESTONE.md)和
[milestone-summary.json](../../../outputs/research/global_throughput/milestone-summary.json)。

可复现命令（数据根只读，生成物在本 worktree；默认禁持久 compilation cache）：

```powershell
$env:ORCHIDEE_DATA_ROOT = 'E:/ORCHIDEE-MAN-JAX/data'
$env:ORCHIDEE_REFERENCE_ROOT = 'E:/ORCHIDEE-MAN-JAX/reference'
conda run --no-capture-output -n ORCJAX python scripts/dev/benchmark_global_batch.py --asset-root outputs/research/global_throughput/local-assets --truth-root E:/ORCHIDEE-MAN-JAX --output-dir outputs/research/global_throughput/local-audit --mode audit
conda run --no-capture-output -n ORCJAX python scripts/dev/benchmark_global_batch.py --prepared-input outputs/research/global_throughput/local-audit/prepared_inputs.pkl --truth-root E:/ORCHIDEE-MAN-JAX --output-dir outputs/research/global_throughput/local-b1-b4-final --sizes 1 4 --warm-repeats 5
```

`--asset-root` 使用 `landpoints/{id}/runtime_used_run.def` 与逐点 `reference/`，也可用
`--input-manifest` 指定每点 `landpoint_id/run_def/reference_dir`。未提供两者时使用现有
raw-server reference resolver。准备输入与 cold Day1 checkpoint 用集合 pickle 保存，
普通审计错误修复可复用；这不是生产恢复格式。`report.json` 与 `events.jsonl`
在失败时保留实际阶段、异常和峰值 RSS，不以不完整执行报告加速。
如完整独立 references 已保存而仅诊断代码失败，可在同输出目录使用
`--resume-references`；它要求原来每点的 cache-clear 记录、完整输入身份和科学源码一致。
本次最终成功是此模式修复 JAX 置换索引后得到的实测，原 report 保存为
`reference_origin_report.json`，没有重做已通过的四份独立 scalar 编译。

1. 固定基线（本地短窗已完成）：同一组真实网格、forcing、初态、参数、输出和精度，测起点完整块。
   分开记录进程启动/数据准备、冷编译、缓存加载、热执行、数据传输、输出与恢复点 IO。
   首轮用小窗口和原有状态资产，避免先跑 669 点或 50 年。
2. 最小实现（本地已通过）：在编译前暴露纯 transition，先做 `B=1` 等价，然后 `B=4` 不同网格的批处理。
   优先考察对单点函数的 `vmap`；不要把已经 `.compile()` 的可执行文件直接套入变换。
   审计所有闭包中的静态配置，连续地点参数必须逐点传入；仅形状相同不足以证明可合批。
3. 有界吞吐筛选（本地CPU B1/B4完成，其余待验收）：真实异质点 `B=1,4,16,64`，按内存和编译结果逐级扩展；同时测 CPU/GPU。
   远端GPU尚在scalar参考编译时用完预算；下一步先解决可复用编译/参考和time-to-valid-measurement，
   再在有界资源合同内完成GPU正确性与同工作量计时，不直接投入年度或更大batch图。
   主测七天块，先过正确性再判断是否值得测 28 天块。保留原精度、全部有效过程和输出。
   重复相同网格只可作设备负载诊断，不能代替异质全球吞吐证据。
4. 只有批处理确有收益，才扩展代表性一年、跨年恢复和多设备规模曲线；
   最后处理 669 点完整科学验收和 50 年生产规模。先明确“快在哪里”，再投入长作业。

首轮比较至少包含：现有单点 JAX 常驻串行、JAX CPU 批处理、JAX 单 GPU 批处理。
当前论文 Fortran 的同工作负载结果是最终速度基线；拿不到匹配环境的实测时，
单列用户历史值，不发布已证实的 Fortran 加速比。

## 不变的验收条件

- 批处理逐点对齐独立执行的完整末态、年度/日输出、schema、整数与逻辑 mask；
  复用原 fast-path `atol=1e-8, rtol=1e-10`，离散字段精确一致。
  Fortran 科学验收另循[现有政策](../../source_audits/paper_scientific_acceptance_policy.md)。
- 覆盖异质参数、干旱、显式积雪、边界日、冷启动及 restart；保留静态 Fortran 分支 ledger。
  批中 padding 不得污染有效网格；不得让某点的参数或初态被广播给其他点。
- 每次计时都等待设备计算完成。至少报告重复热运行的中位数与波动，
  编译缓存身份、CPU 核数/亲和性、GPU 型号/数量、主机负载、峰值内存/显存。
- 核心指标：有效点年/秒、完整任务 wall time、CPU core-hours、GPU card-hours、费用；
  将编译、IO 和 checkpoint 纳入端到端结果。多卡提速不等于单卡效率提升。
- 缓存键区分影响编译的配置/布局/批大小；连续状态与参数不应制造不必要重编译。
  不为追求占用率偷偷降低精度、省略过程、缩短递推或增加采样数据量。
- 逐块加载 forcing、常驻当前状态，按任务需要保留输出；普通正向模拟无需保存所有半小时全状态。
  输出/恢复点采用可恢复的集合文件与索引，并核算字节及文件数；不按变量或时步制造大量碎文件。

## 可微性的独立价值

若损失是单个标量，反向 AD 通常可用一次前向和一次反向传播求很多参数的梯度，
避免有限差分随参数数量成倍重复完整模拟；实际代价受重计算、存储和分支影响。
这有望使全球联合参数校准、观测约束和归因更可行，但不承诺优化迭代次数或总时间。
50 年反向传播的内存/数值稳定性、离散阈值附近的导数和参数路径完整性需单独验证。
正向运行不应为了“可微”而额外求梯度或保存反向轨迹。

## 工作目录与操作约束

本分支目录：`E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`。
父目录仍是活跃的日尺度研究工作区，不能在那里切分支或修改研究状态页。
外部数据、Fortran source 和参考资产仍在父工作区，按已声明绝对路径只读复用；
新生成物放本分支 `outputs/`，不要复制一份大型数据树。

本次完成立项、来源审计、最小 batch 实现、本地静态输入及 B1/B4 有界数值/计时门槛。
远端免费输入审计通过，GPU诊断受编译预算限制。没有执行训练或付费任务。
后续 Explore1000 操作继续服从项目部署说明和专用技能；付费作业仍需明确资源、
时限、命令、目录和最坏费用授权。该分支不继承另一个研究线程的特定 GPU 实验额度。
