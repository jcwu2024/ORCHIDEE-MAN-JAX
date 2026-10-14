# CPU pool遥测修复与待审新候选

更新：2026-10-15。**修复、独立部署及免费非模型验证已完成；新增paid提交/授权均为0。**
最终待审pin为`bae03d9ef78726f00987737cdc58eac705a922ae`，ref
`codex/global-cpu-scalar-pool-v3`。相同4CPU/4worker/90分钟单次pilot已备妥，
等待主管审查；不恢复旧job15866488的额度，不联系或自动唤醒主管。

## 实际代码边界

旧[唯一paid pilot](CPU_SCALAR_POOL_PILOT_RESULT.md)在可选`nvidia-smi`处FAILED1:0/1秒，
尚未派发。现在`slurm_global_server_cpu_4core.sh`以显式`if ! command`捕获GPU
遥测非零；缺命令或失败会warning后继续。相邻`uptime`、`taskset -pc`、`lscpu`
也只用于日志，失败同样warning。真正的物理核准入来自现有Python的sysfs/affinity，
不依赖这些日志命令。没有把整个日志管道或执行链加`|| true`。

保留`set -euo pipefail`，date/hostname、scontrol/TRES、Git/clean checkout、
必需路径、CPU预算、tee写入及实际srun错误仍fatal。
新增[启动测试](../../../tests/unit/test_global_server_cpu_startup.py)执行实际launcher
启动路径；只把PROJECT和SLURM_BIN两处安装前缀映射到临时目录，使用受控遥测/
scheduler stub。srun只记录参数，绝不调用下游Python、模型或真实调度器。
模拟4CPU元数据不证明真实4核资源/物理准入。

可审查的[完整执行差分](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/execution-harness.patch)
相对旧pin`50864f065363f6cd0de685c18d9c257723bf2838`只有launcher和新测试两文件。
`jax_orchidee`、`configs`、`scripts/dev`树OID逐一相等，全部Python benchmark/harness未变。
执行物理tree仍为`75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`；研究分支两项碳修复
及其tree`62e90547ee8eadab387713397bb111c399c81b6c`未回退。源码阶段提交
`f7b4688`/`60e4d8fac3dfd589842860882a07164a7c2c0456`分别保存修复与可移植测试断言。

本轮Verify图检查沿用项目generation`2026-10-14T11:04:08Z`；工作树被图排除，
coverage明确为excluded/not_tracked，相关launcher/test/identity均用实际源文件补查，未重建图。

## 已执行验证及保留的中间失败

| 检查 | 结果与边界 |
| --- | --- |
| 最终本地startup矩阵 | 14/14通过，Git Bash5.2.37；[本地收据](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/LOCAL_VALIDATION_V3.md)、[逐case JSON](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/local-startup-matrix-v3.json) |
| 既有launcher回归 | 本轮修复后18/18通过；未改变这些源文件/参数门。[首轮本地收据](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/LOCAL_VALIDATION.md) |
| 最终目标检查job15866807 | cntest/ibtest01、1CPU、请求300秒，完整父10秒COMPLETED0:0；GNU Bash4.2.46、真实affinity `[0]`。14/14启动矩阵通过，实际launcher/parser dry-run及新candidate identity匹配通过 |
| 目标启动语义 | NVIDIA命令存在但失败、缺失、成功均派发一次；其它三项可选遥测失败也派发。NVIDIA失败且srun退出47时脚本仍47；missing input、dirty、错误CPU预算、缺TRES、scontrol、Git/date/hostname、tee错误均零派发且非零退出 |
| 复用既有证据 | [81项接口检查](CPU_SCALAR_POOL_VALIDATION.md)、免费参考加载job15866448（GPU/scalar各760叶、1CPU/63秒）保留复用；本轮不再解序列化/审计参考数组，无scalar生成、物理runner、JAX导入或物理lower/compile/run |

[目标核验JSON](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/target-verification.json)
保存实际pin、模块路径、affinity、prepared/GPU/science身份、parser结果和完整参数展开；
[目标startup矩阵](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/target-startup-matrix.json)
保存每例stdout/stderr、trace和派发参数。
[sacct全部行](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/free-check-accounting.psv)
单列本轮两个免费父作业，不与旧paid父成本合并。

中间v2 pin`08c9fd8079f1438f36b816de32d089ce38edc811`和免费job15866780
（cntest/1CPU/9秒FAILED1:0）原样保留：[11通过/3失败矩阵](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/target-startup-matrix.json)。
三例硬错误都已阻断派发，但Bash4.2把`group | tee`错误返回1，测试原要求本地Bash5
保留的45/46/48。最终v3仅把这三例改成“非零、零派发、失败命令trace存在”，
没有改变生产launcher；srun47/scontrol44/资源门2的精确断言保留。
另有一次v3辅助脚本路径替换误把Slurm v22写成v32，未进入调度；
[原错误收据](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/free-verification-dispatch-path-error-transport.log)
已保存，修正安装路径后才执行最终免费作业。两种失败均不属于科学/数值或内存结果。

## 新pin、admission与部署

[execution-code-pin.json](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/execution-code-pin.json)
固定源提交、树/文件身份、bundle校验、工作量、admission和proposal。
新[scalar-admission.json](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/scalar-admission.json)
SHA256为`259936bb09f5d0006895b1f5ae17fa298c0baada07edcb23b2355ed892ef796a`。
它相对旧admission只改顶层candidate_identity内execution_commit/execution_code_dir；
其他内容逐项相等。旧scalar来源job15820467、stage SHA、审计收据、strict false、
reset限制、费用、scientific false及数组身份全保留，未重新生成来源。

新目录为
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu-scalar-pool-v3`。
增量bundle经本地/目标校验，向共享bare非force导入后`clone --no-hardlinks`并detach精确pin；
checkout clean、无alternates。[部署收据](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/deployment-receipt.json)
与[登记终态](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/registry-finalization.json)
记录两个版本及免费结果，验证全部旧registry记录不变；旧checkout/数据/结果均保留。
复用canonical CPU环境，没有环境重建或GPU任务。

[制品核验收据](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/integrity-verification.json)
验证两版本差分、只重绑两字段、目标身份和旧失败制品SHA未变；
该输出目录的Git属性为`-text`，提交前核对index blob与本地收据字节相等。

## 仅供审批的一次pilot

| 项目 | 精确候选 |
| --- | --- |
| 资源 | cnall；1node/1task/4allocated CPUs/4独立B1 worker/0GPU；nonexclusive、no-requeue、自动放置；无新增mem/nodelist |
| 时间/停止边界 | wall5400秒；启动并发1，每worker启动1200秒、命令300秒；采样总RSS abort64GiB、host MemAvailable guard64GiB；RLIMIT_AS不设 |
| 工作量/科学门 | FULL64，Days2–8接9–15，896 point-days；每日48半小时步、760原生输出/masks/dtypes、atol1e-8/rtol1e-10和全部原严格门保持 |
| 最坏成本 | 4×1.5=6 allocated-core-hours；费率0.05/0.07均未核实，对应CNY0.30/0.42；含启动/编译/失败的完整父只计一次 |
| cwd/环境 | 上述v3目录；`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python` |
| 授权 | 本工作单paid allowance=0、submission_count=0；精确对象只备妥，须主管审查 |

[PROPOSED_SUBMISSION.sh](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/PROPOSED_SUBMISSION.sh)
SHA256为`db9a5a7bb4a382f373e7cf7cb4d340d185d32293a305a086c585dd6db57070f9`，
固定pin/admission/参考SHA、全部参数并清除继承SBATCH约束；目标只执行过`--dry-run`。
批准后唯一拟命令：

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/PROPOSED_SUBMISSION.sh
```

完整sbatch及srun参数见[目标提交展开](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/submission-target-dry-run.txt)
和[launcher展开](../../../outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3/launcher-target-dry-run.txt)。
未执行上述提交命令。仍无4worker驻留/cache/编译/热吞吐或48核扩展实证；
这些启动检查不授予科学接受、GPU同量对比或192GB可行性结论。
