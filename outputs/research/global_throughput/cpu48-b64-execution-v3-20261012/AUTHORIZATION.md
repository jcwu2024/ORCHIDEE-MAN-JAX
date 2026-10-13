# CPU48 v3 新单次作业授权

授权日期：2026-10-13，Asia/Shanghai。执行接续时间记录为2026-10-13 10:07:56 +08:00，
不把此记录时间当作用户授权的精确时刻。主管聊天为`01a128d2-3ec2-77a1-8d51-1665c220d2ca`。

用户在主管聊天明确授权：“他好像给了一个新作业，你看没问题就可以批准，授权你。”
主管完成实际产物审查后，通过本执行聊天收到的`codex_delegation`批准已审提案中的
新单次CPU48 v3作业，以及必需的精确部署、正常预检、运行、采集和交付。
此次权限直接来自用户，不再索取重复批准；旧未批准措辞由本次授权取代，旧v1/v2额度不恢复。

执行SHA：`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`；ref `codex/global-cpu48-b64-v3`；
base `b1b5cf0835da298bddb388e61d754feeb6c14cf2`。准备交付SHA为
`93f7f5cd45c12344177685f314bc75978274107e`，不是模型执行pin。
已审[提案](../cpu48-b64-preparation-v3-20261012/PAID_PROPOSAL.md)、
[协议](../cpu48-b64-preparation-v3-20261012/EXECUTION_PROTOCOL.md)和
[pin清单](../cpu48-b64-preparation-v3-20261012/execution-code-pin.json)绑定完整版本及数据。
bundle SHA为`940a6f7e2aa6bf545661ef1c815cbd0156fa92b62b18452cbad7398d7c49d47c`，3380B，依赖上述base。

批准cnall、1节点、1task、48 allocated CPUs、0GPU、5400秒，no-requeue、非exclusive、
无新增`--mem`、无overcommit、无node pin/exclusion，由Slurm选择同规格合适健康节点。
提交前保留新的scheduler/占用/负载快照；普通IDLE+PLANNED/PENDING允许正常提交和等待。
全父allocation含scalar全程按48核计费，worst-case情景为48×1.5h×0.05=CNY3.60，
保守0.07情景为CNY5.04，费率及账单未核实。

批准部署到`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3`，
复用共享bare store和已验证bundle、按旧Git1.8.3.1规范登记；不更改v1/v2或其他运行/历史checkout。
若路径已有同SHA干净且未提交版本，确认来源后复用；不覆盖未知内容。不改环境或物理输入。
canonical Python为`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python`。
必要的实际部署identity/import/环境核验在受限CPU test节点完成，复用已验JAX0.4.38接口证据，
不在正式运行前完整预编译。登录节点仅调度、传输和元数据读取；数组比较/collector限test01–test04。

唯一批准提交命令从cln01或cln02登录节点执行，cwd为PROJECT根目录：

```bash
(
  set -euo pipefail
  cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
  for name in $(compgen -A variable SBATCH_); do unset "$name"; done
  for name in CODE_DIR RUN_ROOT EVIDENCE_DIR PYTHON ENTRY COORDINATOR SLURM_BIN PREPARED_INPUT GPU_REFERENCE_DIR CONFIG; do unset "$name"; done
  export CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3
  export EXPECTED_CODE_COMMIT=03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8
  exec /rmprog/slurm/v22.05.7/bin/sbatch /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3/scripts/hpc/slurm_global_server_cpu_48core.sh
)
```

仅允许这一次真实sbatch；返回不明先查同一标记/调度记录，不能盲重试。失败/超时即关闭尝试，
无第二次提交、自动重试、requeue或余额恢复；版本/科学合同/资源预算有必要变化时先停止报告。
run root固定为`runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/<job-id>`；
日期标签保留，授权及实际运行时间如实记录2026-10-13和实际时间。

64 unique FULL64点、连续Days2–8/9–15、48半小时步/天、896 point-days、760完整outputs/masks/native dtypes、
`atol=1e-8/rtol=1e-10`、science `11b6…`/actual source `b334…`及冻结输入/reference hash原样保留。
scalar cap2700s，B64起跑至少余1800s，各5 core+5 wrapper，单native CPU JAX device、48不同物理核。
CPU内部/结构/非有限值/隔离/reset硬门不变，有限跨GPU差异仅可探索计时；x64不声称全算术FP64。

保存原始scontrol、实际node/affinity/load/资源、job ID回执、完整父sacct及一次记账、输出/失败证据。
同量比较对象为GPU B64 job15579869的896 point-days/6.222332641482353s；
B1024 job15671732的299.1398250995909 lane-days/s仅单列不同工作量归一化参考。
本授权不是完整物理/科学/吞吐接受。执行聊天自主跟进至终态交付，不新建聊天、heartbeat或自动任务，
不联系/轮询主管或旧聊天。
