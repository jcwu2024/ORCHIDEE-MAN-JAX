# 独立 B1 CPU pool：接口与本地验证

2026-10-14。此记录是 harness/协议验证，**没有新物理 lower、compile、run，
没有远端部署、预检或付费提交**。下一次运行及代码 pin 以单独审批对象为准。

实际 checkout 为 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`，
branch `research/global-throughput`；ORCJAX Python 导入的
`scripts.dev.benchmark_global_server_cpu.__file__` 也位于该 checkout。
父任务传入 Verify 图证据；新增 helper/test 路径的 coverage 再查结果仍为
`excluded/not_tracked`，祖范围 `runtime/worktrees`，generation
`2026-10-14T11:04:08Z`。相关结论使用精确源码读取，不把图的无结果解释为不存在。

## 实现边界

- 扩展现有 `PoolRunner/pool_worker`。每 worker 是独立 spawn 进程，固定一个已准入物理核，
  thread 环境按 1 归一化；独立调用现有 `independent_scalar_executable` 创建 B1 程序。
  每个点调用原 `run_two`：第二 window 继承第一 window 的状态；下次任务从冻结输入重置。
  不使用多点 vmap。控制器分享 worker0 的核，全部进程仍在同一 allocation 预算内。
- `--pool-workers` 可在 allocation 内选择 1/2/4 等数量；48 CPU 计划必须显式给出该值，
  不默认启动 48 个 resident workers。任务仍是原 FULL64 唯一真实点，轮转唯一分配。
  现有 scalar/batch 路由保留，CPU48 B64 coordinator 属历史路线。
- `--pool-start-concurrency` 默认 1：一个 worker ready 后才启动下一个。
  可显式提高有界并发；每 worker 启动超时默认 1800 秒，任务/完整输出传输超时默认 1800 秒。
  pipe 接收在后台线程中完成，控制器等待时继续检查超时和内存；失败回收本 runner 的所有 workers。
- 物理 pool 必须显式给出 `--pool-memory-limit-gib`，检测控制器及已启动 workers 的总 RSS。
  shared pages 会重复计数；等待期间约 0.1 秒采样，可能错过峰值。启动前和等待期间还检查
  host `MemAvailable`。两者均非内存 reservation、OS 硬 RSS 上限或 OOM 保证。
  可选 `--pool-worker-address-space-gib` 使用 Linux `RLIMIT_AS`；虚拟地址空间不等于 RSS，
  未验证真实 JAX 兼容性，pilot 默认不设。
- 每 worker 记录 pre-import、保留选定输入后的 pre-compile、post-compile、first-call、warm
  RSS/HWM/VmSize，启动 wall/CPU、实际 thread affinity、lower/compile 计时、StableHLO SHA、
  cache events 和实际 observed persistent-cache hits。控制器另记录启动批次前后总 RSS 及观测峰值。
  现有 pickle 必须临时加载完整输入；screen 后仅保留 assigned lanes，不能称“从未加载全部64点”。
- `configure_cache(persistent, root)` 的 public reset 只重置进程 cache 初始化，不删除 disk cache。
  每 worker 仍调用 lower/compile API；同 shape 不证明相同 cache key，实际命中以记录的日志事件为证。
  本地协议测试没有提供物理跨 worker cache-hit、编译节省或 resident 容量证据。
- warm CPU 利用率使用控制器加所有 workers 的 CPU delta，除以实际 group wall。
  worker CPU receipt 在完整 payload serialization/IPC 后产生；最后小型 CPU receipt 的发送除外。
  core 边界包括 dispatch/barrier，不传完整输出；wrapper 包含完整输出 materialization/serialization/IPC。
  总 CPU 数值是观测量，不能替代并行效率或 48 个有效线程的证明。
- 完整输出、native dtype、mask、原固定容差 `atol=1e-8, rtol=1e-10`、五次 warm repeats、
  frozen reset 与 strict CPU scalar gate 保留。GPU 仅既有 exploratory finite-only 规则可继续；
  CPU strict gate 失败会保存 raw output 后阻止热计时。本地 toy 测试不等于真实 760 叶验证。

## 参考与启动接口

旧 scalar 通过 `--cpu-reference-role cpu-internal --validation-mode exploratory` 消费。
pool4/48 FULL64 必须提供新的 `--cpu-reference-admission-record` 与精确
`--expected-cpu-reference-admission-sha256`；原 `load_reference` 校验不变：source/candidate
science/input/reference/commit 身份及来源账目继续核验，不能复用绑定旧 execution SHA 的 admission
冒充新 pin。原 scalar 不重生成。admission 的构造继续由既有
`global_server_reference.bind_cpu_reference_admission` 和本次本地版本准备负责；新的精确 pin/admission
交主管审查，参见[方案](CPU_SCALAR_POOL_PLAN.md)。

现有 `scripts/hpc/slurm_global_server_cpu_4core.sh` 默认只运行 pool：
4 CPU、90 分钟、无默认 node pin、非 exclusive/no-requeue。必须显式设置 `CODE_DIR`；
默认 `BENCHMARK_SIZES=64`、`POOL_WORKERS=4`、`POOL_START_CONCURRENCY=1`、
`POOL_MEMORY_LIMIT_GIB=64`、`POOL_MEMORY_GUARD_GIB=64`、
`CACHE_MODE=persistent`，独立 cache namespace 默认为 `$RUN_ROOT/scalar-worker-cache`。
另需 `CPU_REFERENCE_DIR`、`CPU_REFERENCE_ADMISSION_RECORD`、
`EXPECTED_CPU_REFERENCE_ADMISSION_SHA256`、`EXPECTED_GPU_REFERENCE_SHA256`。
`CPU4_METHODS=batch` 仅显式保留历史入口，不自动跟跑。

已填入审查对象环境变量后的只读命令示例：

```bash
bash "$CODE_DIR/scripts/hpc/slurm_global_server_cpu_4core.sh" --dry-run
```

生产 Python 的 pool 参数示例（其余冻结输入/参考/身份参数仍须完整给出）：

```text
--mode pool --cpu-budget 4 --sizes 64 --pool-workers 4
--pool-start-concurrency 1 --pool-start-timeout-seconds 1800
--pool-command-timeout-seconds 1800 --pool-memory-limit-gib 64
--memory-guard-gib 64 --cache-mode persistent --cache-root <new-shared-worker-cache>
--validation-mode exploratory --cpu-reference-role cpu-internal
--cpu-reference-admission-record <new-exact-pin-admission.json>
--expected-cpu-reference-admission-sha256 <sha256> --expected-gpu-reference-sha256 <sha256>
```

metadata-only 预检入口为 `scripts/dev/diagnose_global_cpu_references.py`，可给同一 pool4/48
计划参数及新 admission；cache 固定 `off`、warm repeats 固定 5。
实际资源门仍要求 `cntest/ibtest`、1 task、1 CPU、affinity 1，thread 环境 1。
它只调用共享 `prepare_references`，报告实际 diagnostic allocation 与计划 pool 预算/worker 数；
在 StageStore/PoolRunner/物理 lower-compile-run 前退出。新远端 metadata-only 预检本轮未运行。

## 本地已运行检查

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/admin/.conda/envs/ORCJAX/python.exe' -B -X utf8 -m pytest `
  tests/unit/test_global_server_benchmark.py `
  tests/unit/test_global_server_cpu_launchers.py `
  tests/unit/test_global_cpu_scalar_pool.py `
  tests/unit/test_global_cpu_reference_preparation.py `
  tests/unit/test_global_cpu_child_environment.py -q -p no:cacheprovider
```

81 项通过。唯一 warning 是禁用 cacheprovider 后 pytest 不认识现有 `cache_dir` 配置。
新增纯 stdlib spawn 测试验证 1/2/4 workers、64 点唯一分配、两 window 顺序、完整 toy tree、
五次 reset/wrapper、worker CPU 汇总、串行/并发2的 ready barrier、启动和任务超时回收、
RSS 阈值回收、RSS/HWM/VmSize 解析、重复任务/缺失输出拒绝、48 核假拓扑的 allocation/billing，
以及 48 CPU dry-run 必须显式选择 worker 数。其 JAX/model import 为 false。
既有 toy/reference 测试保留 CPU strict gate、探索 GPU gate、输入/参考身份及错误 execution SHA 拒绝；
pool4 与 batch48 均覆盖共享参考加载边界。
本机 Git Bash 实际展开 launcher 数组，完整命令经仅记录参数的本地 srun/Python stubs；
归一化环境 wrapper 和真实 Python parser 的 `--dry-run` 也执行通过。
这些不是目标服务器 Bash、Linux affinity/RLIMIT 或真实物理程序验收。临时文件在系统 TEMP，
禁用项目 bytecode/pytest cache。

## 尚缺的物理证据

4 worker pilot 的 64 GiB RSS abort/64 GiB host guard 是待审批保护参数，尚无新的物理容量保证。
旧 scalar HWM 19.94 GiB 含编译/完整64点输入与输出，不能当作最小单 worker resident 内存；
旧 postcompile/warm RSS 也不能推导 48 worker 一定可驻留。需实际小 pool 的逐 worker 编译峰值、
增量 RSS、cache events 和完整760叶 strict CPU 比较后，再判断 resident 上限。
原 FULL64 只有896 point-days；不是与 GPU B1024 等总工作量比较。
本轮没有 task-repeat/2048 新功能；后续扩展必须区分独立 task_id 和 point_index，
重复点不能称 member 或异质真实点，且需相同输入分布/总量和完整输出比较。
