# 下一次 JAX CPU / GPU B64同量对照计划

当前单 GPU 工程选择为 **B1024**。下一主问题是同一 JAX 物理模型的 CPU/GPU 速度及单位成本；B1280/B1536、显存优化以及新旧科学版本的小 tolerance 失败均不作为吞吐工作的前置条件。Fortran 性能对照仍另缺。本轮预算关闭，仅规划；保留全部旧 pin、结果和失败证据，不实施 runner、模型运行、SSH 或作业。

## 已有证据与公平比较边界

| 已完成记录 | 可引用的测量 | 限制 |
| --- | --- | --- |
| [CPU job15591356](../cpu-performance-results-15591356/) | 1 CPU、B16真实点×14天；wrapper `7.052818167587637 point-days/s`；compile `688.378443662 s`，core `31.458503 s`，wrapper `31.760354 s` | code `4bc82a0747c2fd156430956e41ba426271a88b20`，science `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`；2个 cross-GPU rveget tolerance失败，科学接受count为0。CPU-internal reader修复不证明科学接受，也不证明CPU4速度。 |
| [已接受GPU B64 compact authority](../cpu-performance-results-15591356/server-cost-results.json) 的 `accepted_stages[stage=B64_candidate_first_v1]` | job15579869，unique64×14天=`896 point-days`；wrapper median `6.222332641482353 s`，`143.99744462818447 point-days/s` | execution `2b14506992e9c9c7c851cef7b905606e55ac4e04`，science `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`，baseline `133e35aec7b83b9d79888c46092f64d518e97299`。 |
| [GPU job15671732 terminal summary](../gpu-capacity-2048-preparation-20261009/terminal-summary.json) | FULL64真实64点repeat32，2048 lanes×14天=`28672 lane-days`，B1024×2；wrapper `299.1398250995909`、core `350.629529 lane-days/s` | code `8aa2e05aa24b65dc4d30a27d4ee857bba6312221`，同一原science `11b6…`；2051 gates / 2048 saved FULL64 checks通过，只是capacity阶段接受。后续B2048 OOM使parent失败，不能把整个parent费用归给B1024。 |

已核对：上述B64 stage的64条 `native_input_lanes`，其 `landpoint_id` / `inputs_sha256`逐序64/64精确等于 [collected-fixed2048.json](../gpu-capacity-2048-preparation-20261009/collected-fixed2048.json) 的 `valid_exploratory_stages[0].identity.lanes`；双方science均为同一 `11b6…`，命令绑定同canonical `prepared_inputs_remote_64.pkl`。这证明保存metadata的点序和完整input digests绑定同一FULL64候选；未来CPU真实读取、同版本reference及资源admission仍pending，不证明CPU4已经运行。

下一次 CPU 绑定已接受GPU B64的 **science版本、canonical FULL64完整输入与点序、Days 2–8 / 9–15两块、48 half-hours/day、760完整leaves、native masks和dtypes**，不能直接使用当前已改变science的checkout冒充同模型比较。本轮仅核对既有metadata，不测试模型；复用已有GPU证据，不强制重跑GPU。

CPU4/B64尚未运行；未来metadata绑定和完整scalar reference准入后，可直接与已有GPU B64比较相同896 point-days，并提供并行可行性证据。与当前B1024×2的299.139825吞吐仅作不同collection量的归一化参考，不能冒称严格同总量端到端对照。旧B16单核/GPU2048比值也不认证同负载加速或单位成本优劣。

## 最小下一步：B64 CPU4持久池准入

1. **准备同版本的完整B64 CPU scalar snapshot。** 先检查是否已有符合身份绑定的snapshot可复用；当前已列CPU B16记录不能代替完整B64 reference。若缺失，未来用单核scalar、`--sizes 64 --warm-repeats 5`建立。跨GPU固定门仍逐点完整报告；若只存在有限值小tolerance差异，使用现有 `exploratory` 路径，不改门或归因为旧ABI。该snapshot仅作CPU内部对照，不能标为accepted。
2. **只准入现有pool4+B64。** `--mode pool --cpu-budget 4 --sizes 64 --warm-repeats 5 --validation-mode exploratory --cpu-reference-role cpu-internal`，显式提供上一步 `--cpu-reference-dir`，与同版本GPU reference比较并保留全部门结果。CPU内部reference只能配exploratory；若未来有严格接受reference，可另按strict规则使用。测量core和wrapper分别至少5次同步重复，记录四个worker及controller的资源、内存与编译证据。初步不增加batch/vmap大编译。

仅在以后需要与B1024×2严格同总量端到端对照时，再选择2048 repetition adapter/collector；现有sizes只支持16/64，该缺口不是本次B64对照前置，也不立即开发。

## 现有并行、线程和内存依据

入口为 [benchmark_global_server_cpu.py](../../../../scripts/dev/benchmark_global_server_cpu.py)、[global_server_resources.py](../../../../scripts/dev/global_server_resources.py)，以及 [1core launcher](../../../../scripts/hpc/slurm_global_server_cpu_1core.sh) / [4core launcher](../../../../scripts/hpc/slurm_global_server_cpu_4core.sh)。SingleRunner scalar只编译一份executable、各点顺序resident执行。PoolRunner用multiprocessing `spawn`、round-robin分配点，每核一个持久worker，各自独立初始化runtime并编译scalar一次；controller共享worker0的已分配核，不申请第5核。

资源准入要求实际paid cnall/cnmix分配、one task、恰好1或4个不同物理核，禁止login/diagnostic节点性能运行；SMT不是额外物理核。每个worker在JAX导入前设精确affinity、CPU backend、x64及OMP/MKL/OPENBLAS单线程；XLA线程策略仍须记录实际flags，不能仅凭环境变量声称线程数。batch4拒绝继承强制单线程XLA flags；vmap lane数不等于核数。线程census证明已观察mask没有越过分配范围，须保留compile后和warm后证据。

既有单核peak RSS `21,393,784,832 bytes ≈19.925 GiB`，compile-end RSS `5,127,684,096 bytes ≈4.776 GiB`。四个worker按历史peak简单相加约79.7GiB，尚不含controller/IPC，也不是保证；沿用pool **96GiB MemAvailable准入guard**并测量完整worker/controller RSS，避免同时编译超内存。48个独立进程按compile-end RSS朴素相乘约229GiB，超过192GB节点；这不是实测steady，不能提出这种方案。CPU节点56物理核、约187GiB OS可见；Slurm RealMemory=1失真，现有guard不是内存reservation，不编造`--mem`或whole-node exclusive。历史固定node和负载不代表当前occupancy。

## 未来资源、时限和成本准备（尚非提交请求）

优先候选为 **cnall、1节点、1 task、4 allocated CPUs、无GPU的单一有界allocation**：若缺同版本FULL64 scalar snapshot，先以单核step生成，然后四worker完成B64 pool实测，即可提供与GPU B64的公平同量比较及并行证据。单核step须确认准确的step分配环境、物理核affinity及现有admit的1CPU约束，集成launcher/调度方式尚未准备，不能当现有可直接执行命令。已有1core/4core入口是可复用组件，不是硬性两job阶段。

旧1core默认30min不能直接套B64：按旧B16吞吐跨点集线性外推，B64一次约127s，5core+5wrapper加旧compile已有约32.65min，尚未计first run、完整gates、启动/输入准备与输出IO。45min只可作为B64待实际成本确认的费用情景，不是reference加pool流程的完成保证；父allocation若为4CPU，连单核step也按4CPU计费：主费率情景 `4×0.75×0.05=0.15元`，保守情景 `0.21元`，不能按单核价低估。无本轮授权或提交，不请求新审批。未来须核对实际计费、当前node occupancy/CPU负载，优先合适空闲node，并完整呈现资源、wall、实际命令、绝对cwd与最坏费用后再获付费批准。

未来pool准入可复用 `BENCHMARK_SIZES=64`、`CPU4_METHODS=pool`、`VALIDATION_MODE=exploratory`、`CPU_REFERENCE_ROLE=cpu-internal`。`CODE_DIR`必须解析为`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/`下同science、干净immutable checkout；CPU解释器用canonical `.venvs/orcjax_cpu/bin/python`；prepared输入、GPU B64 reference、CPU scalar reference和新输出目录全部显式绑定。**实际checkout/cwd、reference身份/路径、单核step与四核step资源admission、wall及集成展开命令尚pending**，不得拿默认旧checkout/node直接提交；准备时需在目标shell检查真实`--dry-run`展开，本轮未执行。

最终collector需分别报告实测lane-days/s、allocated CPU-hours / GPU-hours、core与wrapper边界、startup/compile和端到端费用；单位成本为`实际可归属费用 / 实际完成lane-days`，稳态估算为`费率 / (3600×实测lane-days/s)`并计入GPU辅助allocated CPU收费。GPU费用不得将后续OOM阶段混入B1024；CPU/GPU费率加算规则和分阶段wall归属未证实前，成本结论保持未完成。当前算术证据见同目录 [CURRENT_RESULT.md](CURRENT_RESULT.md)，由结果owner维护。

本计划使用主管给定的Verify层证据及上述限定源码fallback；graph transport关闭、global worktree不在已知main图coverage内，不声称图覆盖完整。本地确认cwd/branch及stdlib `find_spec`路径指向global worktree，未导入模型。仅新增本文及轻算术，没有运行模型、科学测试或作业。
