# 同一真实 summary_input：1200s GPU 确认执行准备合同

2026-10-11。**本轮仅本地准备，尚非 GPU ready，未批准执行。**
本合同接续 [GPU confirmation plan](T7_batch_numerics_GPU_confirmation_plan_20261011.md)，
不采用生产修复，不重复既有 CPU affine probe、不打开整网/训练。
旧 scope/occupation 已关闭。主管已验收本地包，按13文件范围版本化；实际提交推送SHA见
[postcommit交付receipt](../../../outputs/T7-batch-gpu-preparation-20261011/postcommit-delivery.v1.json)；
不得用起始 HEAD `048c963d9812d77a7c59291ff6374b6e37201738` 代替新增源码身份。
原真实 NPZ/receipt 的旧 bytes 未恢复、旧路径仍缺失；独立授权的一次新 thin capture
已完成，产生下列 recovery 新身份，未改写旧证据。历史 CPU12测量/91数组保持历史身份，
不以新 capture 冒充原 bytes 恢复或已重跑完整 affine 验证；仍不得宣称 GPU ready。

## 冻结输入与最小部署清单

本地 cwd 为 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining`，
branch 为 `research/daily-coarse-graining`，实际 `jax_orchidee.__file__` 在该 checkout。
以下新目标均为**将来计划路径，尚未创建/传输/部署**：

| 内容 | 新目标绝对路径 |
|---|---|
| 版本化代码 checkout | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-batch-numerics-20261011-v1` |
| 输入目录 | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-summary-recovery-20261011-v1` |
| 独立结果目录 | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-gpu-20261011` |

输入目录只需 `summary-recovery.v1.operands.npz` 和 `summary-recovery.v1.capture.json`。
本地来源是上述 daily checkout 的 `outputs/T7-batch-gpu-preparation-20261011/`。
NPZ SHA256 `bf36a098b4e6a89d555f8d18521555f2da66135425d9435cfe01878adeec6129`；
receipt SHA256 `f863a2d964f626cec327981d86a8832e73aa45af62d708a342805c4aa4220e00`。
capture-source 是 checkout 内 `scripts/hpc/capture_shared_latent_T7_summary_operands.py`，
SHA256 `70619b0a6445cde87eccaeb31feac46bbb34cdc0d7a6f68c68938c4d09901072`。
新 capture execution HEAD `048c963d9812d77a7c59291ff6374b6e37201738`；
captured probe source SHA256 `05e2ee66b1905fd238722d3a4cf645e74639f7134de4af5a700cf0b657305ee0`。
授权的一次新 thin capture18.670917s，六个 numeric arrays SHA 与独立提供的 incident 前
SHA 相符；metadata/receipt 的新身份不可等同原 bytes。未重跑完整 affine probe。
旧 capture execution `9de13774df05ddf015557daee3b8540a07ddb497` 与原 `c599236`
及其 overlays 保留为历史身份；新 capture HEAD 也不是最终 GPU 代码部署身份。

新代码须固定最终验收 commit，使用 Git bundle/已验证精确 commit checkout；
保留 tracked Python 包及其传递导入，不手工抽取 `_linear` 或重写公共 runtime。
必要执行根文件为 `scripts/hpc/run_shared_latent_T7_batch_numerics.py`、同名 `.sh`、
`probe_shared_latent_T7_batch_numerics.py`、`orcjax_gpu_runtime.py`、
`check_daily_training_runtime.py`、`shared_gpu_budget.py`、capture-source；
`research/daily_coarse_graining/daily_operator/` 与 `jax_orchidee/` 的 tracked
源提供公共 `_linear`/preflight 传递依赖。最终监督器如增加 helper，须补入代码 manifest。
旧 `shared_latent_T7_memory.sh` 只作审查依据，不将其旧 scope/index/log 限制作为新监督器。
八个历史 source assets、context/parameters/Teacher shards、旧 binaries 不传输、不加载。
versioned checkout 不覆盖旧 `runtime/checkouts/shared-split-width-v1`。

## 冻结 argv 与资源

未来从新 checkout 调用以下 argv；本 owner 的 Bash mock 证据使用旧输入 paths，
新 recovery argv 已完成实际本地parser验证，receipt为
[bash-real-parser.v1.receipt.json](../../../outputs/T7-batch-gpu-preparation-20261011/bash-real-parser.v1.receipt.json)；
这些历史mocks不替代该实际parser证据：

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-batch-numerics-20261011-v1/scripts/hpc/run_shared_latent_T7_batch_numerics.sh \
  --checkout-root /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-batch-numerics-20261011-v1 \
  --operands /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-summary-recovery-20261011-v1/summary-recovery.v1.operands.npz \
  --capture-receipt /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-summary-recovery-20261011-v1/summary-recovery.v1.capture.json \
  --capture-source /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-batch-numerics-20261011-v1/scripts/hpc/capture_shared_latent_T7_summary_operands.py \
  --output /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-gpu-20261011/result.v1.json \
  --scope-json /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-gpu-20261011/scope.v1.json \
  --ledger /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-gpu-20261011/T7-summary-affine-gpu-20261011-v1.ledger.json \
  --dry-run
```

`--dry-run` 保持不批准、不触卡；将来的 live argv 仅在独立 scope 获明确批准、
目标环境核验后去掉它。scope JSON 必须由最终 runner schema 产生/核验，不能复用旧 approval。
`source_files` 必须包含执行源码的当前 SHA256；`code_git_head` 须为最终验收的
精确 commit，postcommit的scope/deployment v2填写实际新SHA，execution保持false。部署 checkout 必须 tracked
与 untracked 均 clean，通过准确身份检查后才能考虑下一步；历史 overlays 不默默替代。
监督器内固定 probe 参数为 `--backend gpu --seed 20261011 --repeats 3`，
真实 `(4,257,4) @ (4,384)`，sample3 raw434，人工 diagnostic cotangent，非 T7 loss。
比较 scalar/vmap/lax.map 三 operator 各 compile 一次及 serial merge 一次；禁止 retry、
algorithm 搜索、新层/attention/prefix/T7/T30/整网 reverse、optimizer update 或 paid fallback。

未来资源：`gln01` 单张空闲 V100，非独占，≤1200 occupied seconds，
≤0.333334 card-hours，free CNY0，GPU memory≤24GiB，OMP4/OpenBLAS1。
preflight≤240s；小图 compile/calls≤780s；保存/释放保护≥180s。
shell 第一个时间戳 `--shell-started-epoch` 包含 CPU interpreter 启动开销，
且用户自传该 flag 被拒绝。timestamp 不是启动 watchdog：live 外层 Bash
先以 `setsid` 创建自己的 CPU process group，并启动独立 deadman，覆盖 CPU
interpreter/顶层 import 阻塞。自 shell 起始第1010s TERM 本 CPU group，
第1190s KILL，保留180s cleanup 与末10s记录保护；剩余等待扣除已花启动时间。
只在 `ps` 核实 PGID 等于本次 CPU child PID 时发送负 PG signal；不杀其他 group。
Python wrapper 独立负责 GPU phase groups 的 bounded cleanup，不将它们当 CPU group。
此外 runtime-child 在任何 driver/container 操作前读取 supervisor 给出的
`ORCJAX_NUMERICS_SHELL_STARTED_EPOCH`，验证 phase leader PGID 等于其自身 `$$`。
同 phase PG 内 timer 对 TERM/INT 免疫，独立在 start+1010s TERM、start+1180s KILL
整个自己所在的 phase PG，早于 outer CPU1190s；timer 存活防止 PGID 被复用。
它逐轮核实自身 BASHPID 仍在该 PG，正常 phase leader 退出/reap 后也 KILL 余下
descendants；timer stdout/stderr 指向 null，避免正常 leader exit 后阻塞 CPU pipe。
因此 supervisor 已 blocked/被杀时，GPU phase deadline 不依赖 Python handler。
正常 CPU exit 不宣称 GPU release 成功。生产入口没有测试 cap 或 live interpreter override；
实际 Linux `setsid` 不可用则不启动 live CPU entry。
每个 driver readlink/container startup 已在监督器计时的 child process group 中。
新 scope 固定物理 GPU index/UUID，container device0；不将 host index 等同 container index。
先记录 affinity/load/GPU processes，确认空闲，独立累计 ledger，不把旧余额变成新授权。
host 只允许 `gln01`/`ibc13b02n04`；initial/release 都严格要求所选 UUID 0MiB、0%、
无 compute apps。active groups 结束后移入 retired/history 保留，不能删掉审计历史。

冻结 strict `rtol=1e-5, atol=1e-6`、FP32 neural/FP64 independent oracle，
保留完整 primal/JVP/小矩阵参数及 input 梯度、masked objective、per-round 原数组/HLO。
outer 到期另以 noclobber 写 `result.v1.outer-timeout.json`，schema 为
`T7_batch_numerics_outer_timeout_v1`，`status=deadline`、`scope_closed=true`、
`automatic_retry_allowed=false`、`reservation_state=charged_or_startup_unreserved_unknown`、
`GPU_release_verified=false`、`resource_state=unverified`。不覆盖 existing primary JSON/ledger。
即使 wrapper 在 reserve 前阻塞，此 marker 也关闭 one-shot，不允许自动重试；
已存在 marker 时 shell live 与 Python preparation 都须拒绝。marker 路径严格在 runtime outputs。
数值非有限/strict/AD失败、NVML异常、24GiB超限、phase/total deadline 各自分类停止；
保存已完成证据、TERM→bounded KILL 所属组、检查所选 UUID 的释放。
释放无法确认则标记未知，不宣称已释放、不退还/重置占用，不自动再试。
此项组件结果不证明原 T7 因果、完整更新提速或长期科学验收。

## Shell 实现与本地准备证据

普通入口从自身目录解析同 checkout runner，dry-run exec、live 外层 deadman 运行 canonical
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python`。
本地 `ORCJAX_NUMERICS_LAUNCHER_PYTHON` 只允许带 `--dry-run` 的 mock/local Python；
live override 在 driver/container 启动前拒绝。
内部 `--runtime-child` 由监督器发起并要求 canonical `.venvs/orcjax_gpu/bin/python`、
同 checkout `check_daily_training_runtime.py`/`probe_shared_latent_T7_batch_numerics.py`
两项 exact allowlist、单 UUID；其他 `.py` 与任何 `--dry-run`/`--dry-run=...` 均在 driver 操作前拒绝。
随后动态 `readlink -f /usr/lib64/libcuda.so.1` 与
`libnvidia-ml.so.1`，bind 到 accepted 容器库位置。
accepted image `/apps/soft/sif/foundationpose`，toolchain
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/toolchains/cuda-nvlink-12.6.85`。
container 清除旧 host LD_LIBRARY_PATH、固定 cuda,cpu/x64/no-preallocate，
PYTHONPATH/repo root 指向新 checkout；PJRT 在 `jax.numpy`/模型导入前由公共 helper 注册。

本地 receipt：
`E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining/outputs/T7-batch-gpu-preparation-20261011/shell-evidence.v3/receipt.json`。
同目录保留每项 stdout/stderr/exit/timing、NUL 原始 argv 和可读 argv文本。
实际 Git Bash syntax exit0/0.067732s，旧目标 paths 的 argv mock dry-run exit0/0.213197s；
十四项独立 shell tests exit0，pytest11.18s/十四项通过（cacheprovider关闭导致已有
`cache_dir` config warning，不生成项目 cache）。tests 覆盖 quoting/精确 argv、
CPU-only普通入口禁止 driver/container 调用、live override 拒绝、内部非 canonical
参数拒绝、mock 动态 driver bindings/UUID/container argv 转发、runtime-child exact allowlist/
dry flag拒绝、epoch spoof拒绝、实际blocking CPU command的TERM→KILL deadline/exit124/
同owned target/marker不覆盖。Git Bash 无 `setsid`，deadline test 在函数层 mock
setsid/ps/kill；实际 child 忽略TERM，最后KILL真实终止。它证明本地deadline分支，
不证明目标 Linux 的实际 session/PG isolation。新增 phase tests 覆盖 blocked
CPU supervisor 不受 phase signal、正常 phase leader exit 后清理 descendants、
caller TERM 后 timer 保持 deadline，以及 non-session-leader 拒绝。每个真实 mock
process 均由当前 harness 创建；没有向其他任务或真实 GPU processes 发信号。
旧 v1/v2 receipt 保留；v3记录当前 shell/tests SHA 和变更前 mock argv 原记录。
最新 recovery 输入及 canonical output/ledger schema 已由实际本地 Git Bash→真实 ORCJAX
parser 验证：`outputs/T7-batch-gpu-preparation-20261011/bash-real-parser.v1.receipt.json`，
exit0/0.436603s，计划12测量/91数组/4次 compile；local schema/real-input parser passed。
此 dry-run 未启动 supervisor/GPU。旧 mock argv 不作最新 parser 证据；原资产缺失
保留为 incident，新 capture 已通过六数组身份对比。目标 Linux/session-PG、
Singularity/PJRT/CUDA/NVML 仍未验证，未重跑完整 affine probe，仍非 GPU ready。
mutation audit 与当前14项 stdout/exit/timing 收入 v3 receipt；tests 的 mkdir/write/chmod
全来自 tmp_path，源码只读，marker/signals 只落 test temporary directory。
验证命令 `conda run -n ORCJAX python -B -m pytest -p no:cacheprovider
tests/unit/test_shared_latent_T7_batch_numerics_shell.py -q`。

这里的 container/driver 是私有临时 mock；本地 Bash 对 target 绝对参数的展开
不等于 Explore1000 target-shell、真实 Singularity、JAX0.4.38/PJRT/CUDA/NVML 验证。
本阶段无 Explore服务器SSH/transfer/deploy/GPU/整网forward或reverse/训练/paid。
主管已验收，按明确13文件范围提交推送；实际身份另存postcommit交付receipt及
scope/deployment v2，不在同一commit内伪填自身SHA。
薄capture仅按单独授权反序列化既有源并执行相同canonical预处理/JVP。
监督器30项CPU/mock tests通过（1.31s），证据为
`outputs/T7-batch-gpu-preparation-20261011/launcher-tests.v2.receipt.json`；
v1的2failed/28passed原日志保留，不放宽runtime gate来修复测试。
daily graph excluded/not_tracked（main generation `2026-10-10T08:46:50Z`），
使用父任务提供 coverage 与 known exact-path 源码 fallback，无重索引/图零命中否定。
