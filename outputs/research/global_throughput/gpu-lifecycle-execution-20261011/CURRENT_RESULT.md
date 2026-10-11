# B1024 双臂生命周期诊断：首臂数值失败终态

更新：2026-10-11。job **15769234** 终态 **FAILED 1:0 / 1140s**。
首臂初始FULL64严格浮点容差检查在lane45（真实点283.0-091.0）停止；初始执行已返回，
编译和原内存guard通过。第二臂未派发，没有warm样本、封存stage或双臂吞吐比。
完整失败数组保留；主管已接受有界CPU/NumPy逐叶只读定位：唯一超差为Day8末
`diffuco_previous_step_state.rveget`、float64、inner `[0,13]`（PFT14），32重复一致。
本轮科学执行关闭，原因未归属；此数值字段差异不是旧ABI故障证据。

## 授权、执行版本与目录

用户在当前主管聊天明确回复“批准。”，授权来源与完整范围见
[批准记录](APPROVED_EXECUTION.md)和[付费执行账本](PAID_EXECUTION_LEDGER.json)。
独立执行聊天仍为 `01a128df-4038-72b0-83a7-144dd939cd76`，
主管为 `01a128d2-3ec2-77a1-8d51-1665c220d2ca`。
本地固定 global-throughput 工作树、`research/global-throughput`；无 main/daily 写入。

唯一执行 SHA：`3b4ae35e695f85480ecb72461e4ba93f4534bbd8`。
后续文档/记账 HEAD 不替代这个执行 pin；原候选的 unapproved 属批准前历史状态。
57项本地有效检查及旧199/37/354证据复用，未重跑，也没有新的独立GPU诊断。

服务端固定 cwd：
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-lifecycle-v1`。
新输出：
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-15769234`。
共享 Slurm stdout/stderr：
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_lifecycle_15769234.txt`。
提交前已核查日志父目录存在且可写。

## 部署与实际目标准入

现有 VPN 路由及 cln01 SSH 认证实际成功。部署前读取当前服务端 README/registry/bare store，
按精确完整历史 v2 bundle 校验 SHA256、ref、bytes后无force导入并新建该唯一执行目录。
checkout干净、无overlay、无alternate依赖；旧不可变树和main/daily refs均保留。
实际回执见[部署记录](deployment-after/gpu-lifecycle-20261011-deployment.json)与
[实际部署 stdout](04-exact-deployment.stdout.txt)。bundle约4.86MB保留本地/服务端，不进Git。

目标 GNU Bash4.2.46直接/模拟spooled展开相同，进入真实coordinator纯dry-run；
源准入在test02完成，104个科学源文件为精确Git blob，新canonical SHA256
`8c8b1dfd0b207e96bbb81c2d3eb85ae6edccb9ed3c6b77fb470dabc58c5fa77d`。
gln01现有容器/venv JAX与jaxlib0.4.38、x64及host CUDA/NVML库绑定检查通过，
仅import/library load；没有GPU枚举/初始化、模型或JIT执行。
两张gln测试卡均被其他用户占用，未在那里进行GPU探测计算。
见[目标Bash/runtime](05-target-bash-runtime.stdout.txt)、[目标源准入](06-target-source-admission.stdout.txt)。
本地conda Windows包装曾改写含`%`/引号的远程元数据命令；改用SSH stdin脚本，未触发作业。

## 单次提交与终态

从gln01经现有SSH入口提交，仅一个gnall allocation：1node/1task、1 V10032GB、
4 Slurm CPUs、120min、no-requeue、非exclusive，主/保守worst-case CNY4.80/4.96。
最新选择节点 `ibc13b03n02`：提交前IDLE、CPUAlloc0/112、load0.01；不等于节点预约。
资源、cwd、节点、完整实际命令先展示，再按已有批准直接提交。见
[提交前快照](07-pre-submit-snapshot.stdout.txt)、[实际提交](08-submission.stdout.txt)。

job15769234在该节点启动：load0.08/0.03/0.05、CPU affinity0,1,56,57，
selected UUID `GPU-18a4abb8-43d5-5142-c32f-a1898866d9bd`、V10032GB、driver525.85.12；
启动selected-card GPU进程表为空。实际分配见[startup allocation](startup-evidence/allocation-identity.txt)。
首臂已完成输入传输、lowering与compile；原guard估算22,957,020,592B，
原preallocation free33,721,155,584B。post-transfer free仅遥测，不重设budget或代表peak。
完整物理图只在首臂执行；[终态Slurm](terminal-evidence/terminal-sacct.txt)父allocation为FAILED1:0、
ElapsedRaw1140s，cluster-local Start16:47:51、End17:06:51，Requeue0/Restarts0。
[协调器](terminal-evidence/run/orchestration.json)为first_arm_failed，只有control child，
comparison_allowed/ratio_published均false。完整[首臂report](terminal-evidence/run/phased_resident_control/capacity/B1024/report.json)、
[manifest](terminal-evidence/run/phased_resident_control/capacity/B1024/stage_manifest.json)和[日志](terminal-evidence/slurm-log.txt)已回收。
manifest记录ValueError：`independent saved-array FULL64 tolerance failed at lane 45`；
原报告只保存首个失败lane；具体叶子、误差幅度与全部失败范围已由
[保存数组解释](SAVED_ARRAY_EXPLANATION.md)补齐并获主管接受。

[时间/内存/费用机器记录](terminal-measurements.json)保留实际可用证据：

| 边界 | 秒 | 含义 |
| --- | ---: | --- |
| prepared input reload | 4.532402543 | 首臂读取 |
| reference/memory reload | 2.094281498 | 冻结数组与历史memory元数据 |
| input transfer事件边界 | 2.168199584 | 含边界记录，不当作warm |
| 实际lower call | 90.272038259 | 单次lower |
| 实际compile call | 866.515529865 | 单次compile；phase差866.602808371s另存 |
| initial collection start | 983.600361001 | child elapsed，不是执行时长 |
| coordinator wall | 1136.942398310 | 含首臂及协调开销 |
| 父allocation Slurm elapsed | 1140 | 唯一费用基准 |

失败路径没有独立的initial execution、validation、serialization及IO完成计时，记为未知，
不把差额当作warm吞吐。GNU time子进程wall18:56.08，user/system CPU1235.00/105.32s；
host peak RSS24,584,675,328B（ru_maxrss与GNU time相符）。
实际compiler临时buffer估算4,567,176,144B，原preallocation free33,721,155,584B，
whole-stage估算22,957,020,592B及原2GiB slack门槛通过；快照和compiler估算不证明peak VRAM。

本地command receipts的Windows UTC与服务器UTC约有8m21s时钟差，二者分别标注。
费用/时长只采用同一Slurm/allocation的Elapsed，父allocation只计一次；
不跨主机UTC相减，不重复叠加batch/extern/srun steps。
全job成本按1140/3600×(1GPU×2.20+4CPU×0.05/0.07)计，主CNY0.76、保守CNY0.785333333，
对应1140GPU-card-seconds及4560allocated-CPU-count-seconds，账单未核实。
本次额度在唯一作业终态后关闭，低于cap不恢复余额或重试资格。

## 保存与追加数值解释

[终态收集脚本](collect-terminal.sh)取回87,684B紧凑archive并记录全部新输出SHA256；
archive SHA256为`538f4254a9b1d03afc4c010d4942278b9153f5d1cf98d4043c02f650251e07b5`。
[回收核验](COLLECTION_VERIFICATION.json)确认8个紧凑run文件与服务器checksum一致、
执行checkout终态干净且仍为固定3b4ae35。

完整失败数组880,948,565B保留于原输出下
`phased_resident_control/capacity/B1024/failed_initial_arrays.pkl`，SHA256
`698d3dece201424778d9cf4b0f10c1ac3b1802a1ce35b7d231f73b751641ddc5`。
报告hash与独立收集清单一致；原终态元数据回收未加载科学数组。
主管授权在既有免费CPU节点用NumPy与冻结27,587,892B FULL64参考只读比较，
保持原容差，定位全部失败叶/字段/时间/坐标及重复lane，解释新科学pin相对旧参考的差异。
该一次比较已在test03完成；未在login节点计算，无模型/JAX/jaxlib导入、JIT或GPU执行，
未下载原始科学数组。[紧凑回收](EXPLANATION_COLLECTION.json)核对10个回收文件的SHA256与大小，
archive为12,481B、SHA256 `64cdcf2537c01897fea0e56da255d63df0d313f7f3daaf7005cc041c85fb9148`。

760个原生叶×2048lane检查完整，唯一失败叶为 `[0][0][0][7]`，实际shape `[2048,1,14]`、
参考shape `[64,1,14]`，两者float64。它是第一窗口Days2–8的Day8末状态，
不是逐日观察输出；第二窗口Days9–15及其他759个叶未超固定容差、无schema/dtype/离散失败。
真实点仅 `283.0-091.0`（source45），其余63个真实点未失败；
失败lane为 `45+64k, k=0..31`，范围45–2029，对应repeat0–31。
32次均只有inner `[0,13]`（PFT14）一个失败坐标，坐标及actual/reference原生数值逐字节一致。

| 定位/数值 | 结果 |
| --- | --- |
| 字段/时间 | `diffuco_previous_step_state.rveget` / Day8结束 |
| lane45完整零基坐标 | `[45,0,13]`，PFT14 |
| actual / reference | `515.7648516411803` / `515.7648515535959` |
| 绝对差 | `8.758433978073299e-08` |
| 原容差界 `1e-8+1e-10*abs(ref)` | `6.157648515535959e-08` |
| 误差/原界 | `1.4223666641536081` |

完整[机器summary](saved-array-explanation/comparison/summary.json)及
[压缩逐坐标记录](saved-array-explanation/comparison/all-failing-coordinates.jsonl.gz)保留所有32个lane来源，
不是抽样错误表。比较匹配8,192个原声明bare-soil roughheight NaN sentinel，其余非有限值无失败。
test03单核、外层600s/内部540s、8GiB虚拟内存上限；Python3.11.15、NumPy2.2.6。
脚本内部22.631719476s，GNU外层29.54s，峰值RSS1,008,807,936B；这些是解释运行记录，不是模型吞吐。
[实际命令/环境](21-compare-saved-arrays.stdout.txt)及本地/服务器原时间戳保留。
第一次启动器在数组加载前因旧Git不支持`-C`停止，原stdin及20号回执保留；
改成已验证的先cd后git后，21号命令完成唯一一次数组比较，无模型或作业重试。

主管已验读summary/gzip并接受该差异定位，不新增检查或扩大溯因。
本轮科学执行关闭；原FULL64门槛仍失败，无科学/性能验收。
此数值字段差异不是旧ABI故障证据；没有因果证据把失败归给碳修复或生命周期优化，
不自动修科学代码、调容差或重跑，原执行pin与旧版本证据保留。

## 不变的停止与验收合同

先 `phased_resident_control`，五core再五wrapper，保留输入根；exit0、seal且独立数组验收全过，
才派 `phased_release`，相同顺序，仅wrapper前释放根。第二臂紧邻派发前至少剩3600s。
失败停止并保存，不retry/requeue/延期/第二作业或扩大shape。

仅B1024×2、64真实FULL64输入重复32次：2048lanes、28672lane-days，
Days2–8连续9–15、48步/天、全部物理/native outputs/masks/dtypes，
atol1e-8/rtol1e-10、离散精确、仅声明bare-soil roughheight NaN sentinel。
common identity只去掉strategy/root持有策略，保留sampling order及其余所有身份。
旧goldens/certificates不覆写，旧timings不混样；x64不等于全算术FP64。

仅两臂独立通过后发表同allocation的core/wrapper吞吐比。
setup/compile/initial execution/validation/serialization/IO与warm样本分别报告，
总allocation费用单列。snapshot不是peak，逻辑root释放不证明物理allocator回收。
本屏仍不构成669异质点/独立成员、长窗口、训练或普遍科学/性能验收。
