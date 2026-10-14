# 独立B1 pool首个pilot终态

2026-10-14。**唯一付费job15866488已FAILED 1:0/1秒，止于CPU launcher的可选GPU遥测，尚未启动任何物理worker。**
这次授权已消耗，remaining=0；没有修复、重提、取消、扩48或新模型运行。
不是物理编译、数值或内存失败；不能据此判定48worker在192GB节点不可行。

## 实际执行与失败边界

执行pin `50864f065363f6cd0de685c18d9c257723bf2838`，物理tree
`75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`。
部署在 `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu-scalar-pool-v1`，
detached clean；回传的launcher原始bytes与该pin完全相同。
新旧scalar admission、历史准备制品及研究分支碳修复均未修改。

免费预检job15866448在cntest/ibtest01、1 CPU完成0:0，sacct父ElapsedRaw **63秒**。
`reference_preparation_passed`，GPU/scalar各760叶、execution身份相符；无物理runner/lower/compile/run。
它证明保存参考加载边界，不是4核物理或内存验收。

付费实际为cnall、ibc11b04n04、1 node/1 task/4 CPU、90分钟上限、no-requeue、无node/exclusive/新增mem约束。
父分配核验 `cpu=4,node=1,billing=4`，与审批一致。集群+0800时间23:08:58开始、23:08:59终止。
ReqMem的1024K是调度器默认记账字段，不是对192GB物理内存的测量或新增reservation。

精确失败点为 `scripts/hpc/slurm_global_server_cpu_4core.sh:177`：

```bash
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
```

CPU节点存在该命令，但返回“couldn't communicate with the NVIDIA driver”。
该命令位于launch-context管道中，`set -euo pipefail`使整个launcher退出；194行的
`run_case pool "${pool[@]}"`尚未达到，日志没有`starting mode=`。
不是GPU任务或物理模型调用；没有为归因重演launcher或运行nvidia-smi。

原RUN_ROOT `.../runtime/outputs/research/global_throughput/server-benchmark/cpu4-15866488`
**只有launch-context.txt**；`scalar-worker-cache`、`pool4`、worker目录、report、manifest和pickle均不存在。
因此逐worker lower/compile/ready/cache hit/StableHLO、增量RSS/HWM、first-call、warm CPU/wall、
strict CPU/GPU/reset gates、完整760叶物理输出全部 **not reached**，无吞吐或warm单位成本。
Slurm TotalCPU只含launcher活动，不能当物理CPU利用率；早晚快照粒度不同的原值均保留。

## 原始证据与费用

- [授权](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/authorization.json)、[部署](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/deployment-receipt.json)、[预检完整报告](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/reference-preflight-report.json)。
- [终态compact结果](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/terminal-evidence/terminal-result.json)、[原始证据manifest](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/terminal-evidence/terminal-evidence-manifest.json)、[独立closure收据](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/terminal-evidence/closure-receipt.json)。
- [完整原日志](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/terminal-evidence/global_scalar_pool_15866488.txt)，SHA256 `a879dd2f396cd8ef9f272e756603c9548c783e01dd38382288bc046d59280e86`。
- [全部sacct行](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/terminal-evidence/sacct-all.psv)、[唯一完整父行](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/terminal-evidence/sacct-parent.psv)、[目录清单](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/terminal-evidence/run-root-inventory.txt)、[缺失制品收据](../../../outputs/research/global_throughput/cpu-scalar-pool-pilot-20261014/terminal-evidence/physical-artifact-presence.txt)。

原始bytes打包回传SHA256 `ca7721e91ad57bfcbfe16d005c217887832b4c04a4406ac98d764873e19c5043`，
18份远端原始文件逐一SHA验证，原RUN_ROOT及日志在远端保留。scontrol记录随后过期，首次完整原始快照仍保存；
sacct终态与父独立核验一致。只做本地stdlib metadata/hash/tar审计，没有科学pickle可反序列化。

完整付费父只计一次：`4×1/3600 = 0.001111111... allocated-core-hours`。
费率0.05/0.07元每allocated-core-hour的未核实情景为 **¥0.00005556/¥0.00007778**，不是账单。
免费预检单列；旧scalar job15820467的完整父2197秒/48 CPU与¥1.464666667/¥2.050533333仍独立保留，未重复收费。

## 下一决策

**本次没有取得192GB单节点支持48个resident B1 worker的实测证据，也没有证明内存不足。**
完整CPU/GPU同量比较仍缺；计划FULL64只有896 point-days，不等于既有GPU B1024×2的28672 lane-days。

最小可审查修改建议仅针对177行：令CPU任务的可选GPU遥测失败留下明确warning后继续，
建议替换为下列显式条件；此处仅为审查文本，未修改生产代码：

```bash
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    printf 'warning: optional GPU telemetry failed for CPU job\n' >&2
  fi
fi
```

保留所有资源/科学硬门。
先在目标CPU Bash用非模型替身验证“命令存在但返回非零”控制流，再审查新immutable pin、参考身份与
**一次相同4 CPU/4worker/90分钟pilot**：串行启动、RSS abort/host guard各64GiB，RLIMIT_AS不设；
最坏6核时、¥0.30/¥0.42（费率未核实）。这是新的待审批对象，本次未改代码、未执行或授予新额度。
待实际小pool的启动、cache、内存与完整输出证据后才能决定48核资源；不作线性容量或吞吐承诺。
