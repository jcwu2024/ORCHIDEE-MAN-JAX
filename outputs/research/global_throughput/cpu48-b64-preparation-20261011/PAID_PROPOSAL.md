# CPU48 / FULL64 B64：具体运行提案

本页是供主管审阅的准备结果，**尚未部署或提交**。旧预算已关闭；只有独立明确授权后才执行下列远端写入及付费命令。本轮不启动 GPU、不改物理过程或容差。

## 固定版本和负载

- execution commit：`4f6b2fc906c453b9030acff489637a8f83605dca`，ref `codex/global-cpu48-b64-v1`；审核 source 为 `2f8544d40cef306c4582f5ed58bd7ab7e6a8905f`，完整封装证据见 [execution-code-pin.json](execution-code-pin.json)。
- base：`8aa2e05aa24b65dc4d30a27d4ee857bba6312221`；science `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`；baseline `133e35aec7b83b9d79888c46092f64d518e97299`。
- 精确 code/cwd：`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v1`。
- Python：`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python`；必须是实际 CPU JAX/jaxlib0.4.38，开启 x64。
- 输入：`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl`。
- GPU 数值 reference：`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-round4/model`，SHA256 `a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6`。
- config：`<上述code>/configs/orchidee_man_250919.yaml`。64 unique FULL64 点，Days2–8 / 9–15，共896 point-days；原48半小时步、760完整输出叶、masks/native dtype和固定门保留。

只将审核的 harness/tests 加到历史执行 base；整个物理源码子树与 base 相同，当前研究树的碳修复保留。先前 source identity 不能移授新碳核心；本次 CPU/GPU 对照明确使用同一历史 science。

## 资源和预算

| 项目 | 请求 |
| --- | --- |
| partition / node / task | cnall / 1 / 1 |
| allocated CPUs / GPUs | 48 / 0 |
| wall / requeue / exclusive | 01:30:00（5400s）/ no-requeue / 非exclusive |
| scalar / batch MemAvailable guard | 24GiB / 64GiB；不是内存reservation或OOM保证 |
| 主情景最高费用 | 48×1.5h×CNY0.05 = **CNY3.60** |
| 保守最高费用 | 48×1.5h×CNY0.07 = **CNY5.04** |

父allocation在scalar1 step期间仍按全部48核计费。费率未核实实际账单。Slurm RealMemory=1失真，不加 `--mem` 或 whole-node exclusive。协调器 start/end 的区间费用不是完整 Slurm Elapsed；终态 accounting 只计父一次，失败费用也保留。

## 启动前和具体付费命令

[只读收据](target-readonly-preparation.json)证实新code目录当时不存在、绝对logs目录已存在、server bare Git已有bundle prerequisite。节点 `ibc12b08n35` 在收据时间为健康IDLE、CPUAlloc=0/56、load=0.01、ThreadsPerCore=1；这不是reservation。提交前重新读取该node state/occupancy/load，健康且足够48核时才使用本页命令。旧n04已满、n01为DRAIN，不恢复旧默认node。

授权部署后，运输版本pin所列的已验证bundle，fetch到现有 `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/deployment/repository.git`，在上述全新code路径以完整execution SHA创建detached immutable checkout；不更新正在运行的checkout，不覆盖当前研究树或数据。GitHub不可达时用bundle，不用第三方镜像。确认完整HEAD和clean状态，以及canonical输入/reference存在，再执行一次目标shell `--dry-run`；准入和runner在物理compile前再次检查版本/资源/reference。

审批对象是以下一个作业命令，cwd已经解析为固定immutable code-dir。输出为全新 `.../server-benchmark/cpu48-b64-v1-${SLURM_JOB_ID}`，含scalar、batch、orchestration和两个step日志；Slurm日志使用绝对路径。

该 SHA 的启动器 Git blob 已经通过目标 GNU Bash4.2.46 stdin `--dry-run` 实际展开；本地实际 coordinator→scalar/batch CLI dry-run 也通过。见 [固定版本接口收据](pinned-interface-review.json)。本地仅替换 Python/runner 路径；没有目标 Python/JAX 测试或部署，真实 code-dir/HEAD、Slurm 资源及物理运行准入仍待未来实测。

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v1
EXPECTED_CODE_COMMIT=4f6b2fc906c453b9030acff489637a8f83605dca \
CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v1 \
/rmprog/slurm/v22.05.7/bin/sbatch \
  --partition=cnall --nodes=1 --ntasks=1 --cpus-per-task=48 \
  --time=01:30:00 --nodelist=ibc12b08n35 --no-requeue \
  --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v1 \
  --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_cpu48_b64_%j.txt \
  --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_cpu48_b64_%j.txt \
  scripts/hpc/slurm_global_server_cpu_48core.sh
```

## 阶段和硬停止规则

同一allocation先srun scalar1生成FULL64 CPU内部参考，沿用5 core + 5 wrapper，最多2700秒；完整stage密封后再运行一个48核native B64进程，5 core + 5 wrapper。候选最低剩余协调器窗口1800秒。48核step使用 `OMP_NUM_THREADS=48`、Eigen multithreading，MKL/OPENBLAS/NUMEXPR均1；一个native CPU JAX device，实际并行效果未知。startup/cold compile/first call与热core/wrapper分列，保存CPU时间、RSS/HWM/MemAvailable和线程affinity。

版本不符或dirty、未实际分配cnall/1node/1task/48CPU、AllocTRES或48不同物理核不符、缺冻结输入/reference、GPU数值reference SHA不符、runtime版本不符、MemAvailable guard不足、参考不完整、scalar失败或超时、候选窗口不足、候选失败/超时/信号中断时停止。超时/信号清理自己的本地srun进程组，不自动提交第二次job或requeue。真实Slurm step清理仍待未来目标环境核验。

只有有限值的跨GPU小容差失败允许探索计时并原样记录；CPU内部完整整组比较、permutation、replacement/isolation、frozen-reset及结构/非有限值门仍硬停止。完整成功也标成exploratory measurement，不自动变成CPU科学接受。CPU4或CPU2048不作为前置，不重跑GPU。GPU B64为同896 point-days比较；B1024的299.139825只作不同总量归一化参考，不称严格端到端加速比或48核线性收益。

## 终态采集和费用

采集器的具体协议/CLI与normalized accounting schema见 [PROTOCOL.md](PROTOCOL.md)。终态取得父job的sacct `JobIDRaw,State,Elapsed,AllocTRES` 原始parsable2行，保留完整行及SHA，绑定orchestration中的parent job ID、cpu=48/noGPU和两个step目录。整作业费用由终态Elapsed推导，两个情景都乘48；没有accounting时仅发布协调器覆盖区间估算、whole-job pending。失败也生成费用记录，扣留性能结论。

未来在获准的诊断CPU节点test01–test04上，用immutable code/canonical CPU Python、受限CPU affinity执行独立collector；login节点仅做scheduler/transfer，不做数组比较。一次完整成功的命令形态如下，JOB_ID以实际返回值替换：

```bash
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUN_ROOT=$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/cpu48-b64-v1-${JOB_ID}
cd $PROJECT/runtime/worktrees/global-throughput-cpu48-b64-v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONDONTWRITEBYTECODE=1 \
taskset -c 0-3 $PROJECT/.venvs/orcjax_cpu/bin/python -B \
  scripts/dev/collect_global_cpu48_b64.py --run-root "$RUN_ROOT" \
  --parent-accounting "$RUN_ROOT/parent-accounting.json" \
  --output "$RUN_ROOT/collected-cpu48-b64.json"
```

collector重新核验manifest/checkpoint、科学与输入身份、64 unique×14天分母、760输出schema、实际运行观测和保存数组的GPU/CPU比较；复用Linux loadavg tuple/list精确白名单，CPU内部按runner实际保存的整组gate重算。完整测试仅证明合成协议可读写；真实Linux CPU48资源、利用率、全物理数值、编译耗时、内存和吞吐仍pending。
