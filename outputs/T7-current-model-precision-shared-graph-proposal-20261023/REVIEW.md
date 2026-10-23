# T7 shared-graph：部署就绪，待新执行授权（2026-10-23）

**approved=false；未运行GPU、未sbatch、未创建真实执行批准。** 主管2026-10-23已审阅并接受96978ca本地工程修复及dd352d7终态收尾；此接受不构成GPU、梯度或科学验收，也不沿用旧“继续”或任何已消耗批准。

| 固定项 | 身份/边界 |
|---|---|
| 执行源码 | `96978caf04c9a56273ea7e9b9031f57aebf999ac`，未引入后续Teacher reader改动 |
| probe SHA256 | `3db6eaebbe7e0000e93f661d3264900e4d1d8ac92efe851a667bdc7392d1c863` |
| 提案SHA256 | `069e8dcc114580314dffc173fdaab1158dc0ad2931520ec3a41e26061bee7a24` |
| 新远端cwd | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-shared-graph-20261023` |
| 独立未来output | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-shared-graph-20261023`；当前不存在 |
| 费用/资源 | gnall，1node/1task/4实际Slurm CPU/1V100，非独占，90min，24GiB GPU上限；`(4×0.07+2.20)×1.5=¥3.72`，实际账单未知 |
| 预算 | load300+启动/discovery/preflight240+共享graph3300+calls1380+save/release180=5400s；lower/compile同一绝对截止，compile不续时；启动/准备开销计入外层剩余时间 |
| 调度/次数 | 自动择节点，无固定/排除/独占；普通Priority/Resources排队可接受，忙卡不阻止将来获准后入队；no-requeue，单次sbatch，无重试；最多1次模型compile、52调用(16native+36FD) |

**准确入口与命令：** [proposal.json](proposal.json)保存完整submission/wrapper/outer-runner/probe argv及全部数据路径/哈希；[submission.command.txt](submission.command.txt)是精确未来`sbatch --wrap`命令，**当前不得执行**。[approval.proposed.json](approval.proposed.json)只是拒绝执行的草案，绑定相同argv/阶段预算/probe哈希；真正的`approval.v2.json`不存在。

**已验证：** test01新checkout HEAD/clean/import路径/probe哈希、Git bundle与旧依赖链；原参数/上下文/prepared manifest/dataset manifest/artifact299/receipt/forcing/CO2逐字节身份未变（[asset-identities.json](asset-identities.json)）。固定loader直接用已准备artifact299数组，`_training_provenance`只核对manifest元数据，不读已退役的逐年Teacher文件；未恢复Teacher或重新消费全套输入。元数据、overlay、FP32参考额外固定当前哈希；不变case/state和七日输入绑定证据复用。

[linux-tests.json](linux-tests.json)：**test01实际Linux 42 passed in4.77s**，包括700s受控lower进入compile/只用余量、阶段超时JSON保留/真实短子进程exit124、全局/调用/释放保护、旧预算approval拒绝。准确命令在该receipt，入口为`CPU_PYTHON -B -m pytest -q -p no:cacheprovider --assert=plain tests/unit/test_current_T7_precision_deadlines.py tests/unit/test_run_shared_current_T7_precision_admission.py tests/unit/test_shared_current_T7_precision_shell.py --tb=short --basetemp <checkout>/outputs/T7-current-model-shared-graph-20261023/cpu-validation/pytest-tmp`。临时1.74MB纯Python工具ZIP通过PYTHONPATH使用；无环境安装。初次缺pytest及ZIP缺py shim的准备失败已记录，最终42通过才是验证结果；cacheprovider禁用产生既有cache_dir配置警告。

[argument-chain.json](argument-chain.json)：实际目标Bash展开`--wrap`、真实wrapper `--dry-run`打印的outer argv完全一致；真实runner用CPU subprocess记录替身核对向probe的转发，probe解析通过。**未运行会真实消费输入的probe `--stage dry-run`**。实际未改shell在独立进程组中把原allocation epoch和`CURRENT_T7_PROBE_DEADLINE_EPOCH`导出至`SINGULARITYENV_*`；CPU singularity替身记录并模拟前缀映射后，probe剩余计时约1999.66/2000s。替身用空driver fixture，不加载CUDA/NVML，不执行GPU Python或真实容器；自有进程组已清理。[final-check.json](final-check.json)确认草案被`approved`门拒绝、新output/approval不存在、CPU/GPU Python和nvlink仅文件可执行性检查。

**复用与保留：** retained-data-root仍为`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-current-model-forcing-asset-recovery-20261016/inputs`。旧f2242fc5 checkout、结果、缓存和所有Git对象依赖不变；新checkout通过shared alternates依赖旧checkout，必须一并保留。因源码强制cache在本checkout/outputs内，仅将旧cache唯一文件原字节复制到新cache：`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-shared-graph-20261023/outputs/T7-current-model-shared-graph-20261023/source-cache`，SHA256 `5d77141bad1359d3a71174ba03eba57527d7b0e94cf3e42bce73a2dc650c740d`，不重新生成。

**仍待未来GPU分配验证：** 唯一UUID discovery/runtime preflight、实际FP64 lower/compile与显存、52次数值/physics/完整3854state比较及释放。66,123,713参数/B1 artifact299/window0/T7公共(flux+stock)/9、方向434/2382/2621/control200与六步FD、rtol1e-5/atol1e-6均不变；state20 rawNaN严格失败独立保留，FP64仅诊断。3300s可能仍超时，不能保证成功；届时保存真实阶段/attempt/completed/raw结果后收尾，不能自动追加探针或更长作业。主管须先审阅此产物，再决定是否请求**新的**执行授权。当前停止于此交付点，无watcher/heartbeat。
