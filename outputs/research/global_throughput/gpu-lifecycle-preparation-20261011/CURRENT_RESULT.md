# B1024 双臂生命周期诊断：本地准备

更新：2026-10-11。固定工作树为
`E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`，分支
`research/global-throughput`。本轮只准备可审阅的本地代码、轻量检查和未来资源方案。
未连接 Explore1000、传输或部署代码、修改服务器镜像、运行 GPU/完整物理图、提交作业。

## 当前交付与版本

专用本地代码、保存证据接口及两臂串行流程已完成；检查结果与命令见同目录验证记录。
完整交付提交 SHA、远端一致性、最终源码准入 dry-run 和未来静态 argv 将保存在
`EXECUTION_CANDIDATE.json`；该提交后收据不进 Git，避免提交 SHA 自引用。
资源方案仍未获执行批准，详见[GPU 提案](../memory-lifecycle-implementation-20261011/GPU_VALIDATION_PROPOSAL.md)。

前置科学同步已完成，复用[有界 CPU 验证](../carbon-core-sync-20261011/CURRENT_RESULT.md)：
37 项检查与 354 个生产接口返回记录通过，范围不含完整全球 driver 或 GPU。
科学内容固定在 `1fdacde59d86f696f384b56d69292c8f6fa8d30b`，
新 canonical scientific SHA256 为
`8c8b1dfd0b207e96bbb81c2d3eb85ae6edccb9ed3c6b77fb470dabc58c5fa77d`。
本轮没有修改 `jax_orchidee` 科学过程或旧 `global_source_compatibility.py`。
旧 DIFFUCO-only execution-equivalence 证书继续拒绝新碳核心。

新合同为 `GPU-B1024-phased-memory-lifecycle-new-science-v1`；
进程策略为 `sequential_fresh_process_per_strategy`；
唯一 stage 为 `capacity_B1024_lifecycle_v1`。独立科学身份明确声明碳过程变化，
只给出对旧冻结 FULL64 的严格比较准入，不宣称科学来源等价或 GPU 数值已通过。
旧参考 `B4_group_references` SHA256 固定为
`a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6`。

## 执行与验收合同

同一未来 allocation、不可变代码、runtime、GPU UUID/total、allocator 与输入下，
串行运行两个全新进程：

| 顺序 | strategy | wrapper 前的 core 输入根引用 |
| --- | --- | --- |
| 1 | `phased_resident_control` | 保留 |
| 2 | `phased_release` | 清空 |

各臂均先五次 core，再五次 wrapper。`common_identity` 仅排除
`measurement_strategy` 与 `wrapper_resident_input_policy`；
`sampling_order=five_core_then_five_wrapper` 必须保留并严格匹配。
仅使用 B1024×2：64 个真实 FULL64 输入重复 32 次，共 2048 lanes、28672 lane-days，
Days2–8 后连续 Days9–15、每天 48 步。保留完整物理、全部 native outputs/masks/dtypes、
`atol=1e-8, rtol=1e-10`，离散字段精确；仅允许来源明确的 bare-soil roughheight NaN mask。
x64 不表示所有算术均为 FP64；重复输入负载不等于 669 异质点或独立集合成员。

新 runner 在初始数组 collection 完成后、任何 warm sample 前，独立检查实际
760-leaf schema 与 2048 FULL64 比较。有限数值超原容差也立即失败，保存失败初始数组，
不沿用旧 capacity 路径的 exploratory continuation 语义。旧工作流语义保留。
实际初始、reset、permutation、replacement 输出及两个初始 chunk 的连续状态交接另行存证。
warm 运行使用原 executable 与未改动的 `run_two`。

每臂须 exit 0、完整封存并通过 CPU/NumPy 独立保存数组 validator。
validator 不导入模型/JAX，重新检查 2051 gates、760 个实际叶子、2048 个参考比较、
reset/permutation/replacement、实际连续状态、源码与运行身份、timing、采样顺序及内存证据。
首臂独立通过后第二臂才可派发；紧邻第二臂派发前必须尚余至少 3600 秒。
任意失败停止，无重试、requeue、延期、放宽容差、改变 allocator 或扩大 batch。
仅双臂独立通过后发布同 job 的 core/wrapper 中位吞吐比；否则无比值。

## 代码与证据地图

| 文件 | 职责 |
| --- | --- |
| [global_lifecycle_science.py](../../../../scripts/dev/global_lifecycle_science.py) | 固定新科学 blob、显式旧参考比较准入、受限 NumPy 反序列化 |
| [benchmark_global_gpu_lifecycle.py](../../../../scripts/dev/benchmark_global_gpu_lifecycle.py) | 专用 B1024-only runner、严格初始早停与实际数组存证 |
| [validate_global_gpu_lifecycle.py](../../../../scripts/dev/validate_global_gpu_lifecycle.py) | 独立保存数组重查 |
| [benchmark_global_gpu_capacity.py](../../../../scripts/dev/benchmark_global_gpu_capacity.py) | additive 计时外 observer 与非 warm 验证 callbacks |
| [global_gpu_lifecycle_coordinator.py](../../../../scripts/hpc/global_gpu_lifecycle_coordinator.py) | 串行 fresh-process 派发、独立验收、时间/身份 guard、终态与比值 |
| [slurm_global_gpu_lifecycle.sh](../../../../scripts/hpc/slurm_global_gpu_lifecycle.sh) | 专用 canonical CUDA/NVML launcher |
| [global_batch_gpu_entry.py](../../../../scripts/hpc/global_batch_gpu_entry.py) | canonical entry allowlist 添加新 runner |

launcher 传入绝对 `CODE_DIR/scripts/hpc/global_batch_gpu_entry.py`，
由 `ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_lifecycle.py` 选择 runner，
runner 文件名不是额外 argv。旧服务器 Git 使用 subprocess cwd 或 `cd CODE_DIR && git`，
不依赖 `git -C`/worktree。真实本地 Bash 展开必须进入真实 coordinator dry-run；
canonical driver bind 在本地记录为待目标节点解析，不能称已验证服务器驱动。
Slurm stdout/stderr 均固定到共享绝对路径
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_lifecycle_%j.txt`；
部署后、提交前必须核查该共享日志父目录存在且可写。Slurm 在脚本执行前打开日志，
不能依靠脚本内的 `mkdir` 建立该父目录。当前没有执行目标端检查。

## 已完成的本地验证

[真实旧参考格式检查](reference-format-compatibility.json)已通过，使用现存本地冻结
`gpu-round4/model/stage_B4_group_references.pkl`（27,587,892 bytes、上述固定 SHA256）。
受限解码器用精确四字段 inert `ModeloutResult` DTO读取旧 NamedTuple，不导入科学模块/JAX。
实际 64×760 native leaves 含 744 float64、4 int32、8 bool、4 int64；
严格自身比较为零差异，四个声明 roughheight NaN 路径各含64个允许坐标。
该项是格式、schema/dtype、mask与冻结来源的兼容检查，不是新核心数值比较。
可复现命令为 `conda run -n ORCJAX python -B` 执行同目录
[reference-format-check.py](reference-format-check.py)。

[observer 检查](observer-validation.json)的三项 fake 测试已通过：
计时外 observer人为增加50s仍得到原 core7s/wrapper15s warm样本；
验证 callback仅用于非warm collection，根持有策略和UUID失败早停符合合同。
source/runner/validator/observer 当前有效43项：容器桥接修复后的 runner7+validator23
共30项重跑通过（28.35s），复用未变化的 science10+observer3；此前42项批次
（28.20s）保留为修复前记录，不能称43项同时新跑。详见
[source/runner 收据](source-runner-validation.json)和[最新30项日志](runtime-container-pytest.txt)。
检查覆盖严格初始有限数值早停、完整schema/dtype/mask、跨lane混合NaN sentinel记录、
成功生产容器 runner→StageStore→独立validator、实际状态交接与失败封存规则。

[真实参考容器贯通](runtime-reference-container-compatibility.json)另行通过：
GPU runner 在 runtime admission 后才恢复精确生产 `ModeloutResult` 类；
760 leaves、两个容器与每个数组对象保持不变，实际 `base.compare` 与独立比较器均通过。
CPU validator 仍使用 inert DTO，不导入 JAX/模型。该贯通检查仅导入生产容器/JAX，
没有模型、JIT或GPU执行；复现入口为[runtime-reference-container-check.py](runtime-reference-container-check.py)。

[coordinator/launcher 收据](coordinator-launcher-validation.json)记录 fresh-process 顺序、
独立验收与checksum、严格 common identity、失败无比值及 journal IO后紧邻派发的3600s guard。
日志路径及POSIX规划小修后，两文件最新14项通过（1.84s）；本轮有效合计57项，
不是57项单批重跑。最新[launcher/coordinator 日志](coordinator-launcher-tests.log)保留完整结果。
实际 Git Bash 展开同时覆盖 checkout 与 Slurm-spooled 启动位置，进入真实 coordinator dry-run。
Windows fixture 映射项目路径，并把mock容器Python token映射回canonical Explore1000 Python；
这是本地参数集成证据，不是目标Linux Bash、Slurm或CUDA/driver bind验证。
未来 `/WORK` 输出路径另作纯dry-run检查，避免Windows本地盘符混入canonical计划。
Ruff、格式及diff检查通过，具体本地数量和命令保存在两份验证收据；
[最终集成审阅](integration-review.json)记录提交范围和边界。
不重跑旧199项或碳核心37/354项，不编译或运行完整物理模型。

严格source准入曾拒绝六个科学 `__pycache__` 目录内60个既有 `.pyc`。
经每个绝对源/目标路径和reparse point检查后，完整移动保存在本报告目录的
`local-cache-quarantine/`，映射见[保留回执](local-cache-quarantine.json)；未删除或提交缓存。
该拒绝属于本地源码打包准入，不是数值或GPU失败。两策略真实CLI纯dry-run随后通过。

图服务 `index_status` 与 `check_index_coverage` 本轮仍返回 `Transport closed`。
已知图项目 `E-ORCHIDEE-MAN-JAX`、generation `2026-10-10T08:46:50Z` 属main，
runtime worktrees被排除；本轮采用Verify层级的相关精确源码/测试文件回退，
不宣称图覆盖完整，也未修复服务或重建索引。

## 内存、计时与局限

guard 保持 `2R + 3O + scaled_temp + 2 GiB` 对原始 preallocation physical-free budget，
postcompile snapshot 仅作 telemetry。historical B1024 checkpoint 只流式核对 hash/bytes，
未来 runner 不加载其数组或计时。初始/结束 collection、每次 sample 和失败阶段及时落盘。
core root 释放前后及每次 wrapper start/end 在计时外保存 logical ownership/bytes、host RSS、
同卡 physical snapshots。observer 与验证 materialization 不进入 warm timer。
wrapper timer仍包括 host stack、H2D、两块执行、D2H 与同步。

离散 snapshot 不是峰值；真实 peak VRAM 不可用。没有新增 per-chunk/per-block/per-transfer
的完整显存轨迹，也没有可靠 allocator live/reserved 采样。不把 logical reference 消失
写成物理显存已释放。独立保存数组校验增加诊断 IO/host 数组量，其耗时另计。
本地 fake/fixture、源码与真实旧参考格式检查均不替代部署 JAX0.4.38/CUDA 环境、
新科学核心 FULL64 数值、GPU allocator 行为或吞吐收益验收。

## 下一步资源块：未批准

未来 cwd：
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-lifecycle-v1`。
未来单次输出根：
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-<new-job>`；
每臂位于 `<root>/<strategy>/capacity/B1024`。

拟 `gnall`、1 node/1 task、1 V10032GB、4 Slurm CPUs、120 分钟、no-requeue、非 exclusive。
主 worst-case CNY4.80，保守 CNY4.96；费率/账单假设及完整 canonical 命令见 GPU 提案。
未来节点须基于新的 scheduler occupancy/CPU-load 证据选择，当前未查询或部署该目录。
批准仍须明确 code SHA、cwd、完整 command、输出、资源、有限 walltime 与最大费用。
旧 job/预算保持关闭，本轮普通本地 commit/push 不更新服务器执行树或镜像。
