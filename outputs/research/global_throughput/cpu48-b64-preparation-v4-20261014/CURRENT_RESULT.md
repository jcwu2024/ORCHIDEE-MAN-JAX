# CPU48 v4 准备结果

更新：2026-10-14。当前正在完成不付费修复及证据集成，尚未生成最终执行 pin 或提交新作业。
范围见[授权](AUTHORIZATION.md)，合同见[执行协议](EXECUTION_PROTOCOL.md)，资源与停止边界见[提案](PAID_PROPOSAL.md)。

已确认的启动缺口是：v3 生产 Popen 未显式传 mode 环境，srun 使用含重复键的 `--export=ALL`，而 dry-run 显式传入了环境。
[实际安装 Slurm22.05.7 的免费单CPU复现](environment/slurm-export-precedence.json)输出继承值，证明该命令的重复键优先级。
历史 job15820467 只报告 OMP 线程预算门失败，未记录实际 child OMP 值；两项证据不能混为一次历史观测。

正式启动与预检现已共用 post-transport child 边界，记录白名单环境与 affinity，保留实际48核/独立物理核硬门。
启动/资源/替身回归的108项本地检查通过；免费单CPU目标诊断与shell展开通过，见[环境修复验证](environment/ENVIRONMENT_VALIDATION.md)。
目标当前记录仍是明确标注的占位pin快照，最终固定pin检查待绑定后完成。
根封装的14项最终轻量测试已通过：[实际回执](package-local-check-receipt.json)及[原始日志](package-final-binding-tests.stdout.txt)。
首次较长临时目录触发 Windows Git 路径限制，[失败日志](package-admission-tests.stdout.txt)保留，未修改断言或科学合同。

保存 scalar 的目标端独立数组审计已完成：免费 cntest/ibtest02 的 job15852254，1CPU、实际affinity[0]、23s。
[实际审计回执](scalar-audit/scalar-audit-receipt-v3.json)共146,678字节，SHA256为`cb1331edb7094e74ec1814c6813a0eee218c58ea4e89f1c0c1f630e981479431`。
64点输入digest与精确点序独立重算通过，全部760叶shape/native dtype核对，64个GPU门重算为61通过、3有限超差（4处失败），严格科学状态仍false。
reset只能核对原封存门证明；原重复数组未独立保留，不称独立重算reset。实际执行信息见[执行说明](scalar-audit/audit-execution-notes.md)。
采集器runner侧车门名称/数组重算、早停顺序、batch-only旧新parent成本核算及审计执行信息已完成，见[scalar admission验证索引](SCALAR_ADMISSION_VALIDATION.md)及其中原始测试输出；尚未绑定admission。
新候选只跑 B64，显式引用旧 scalar 目录与独立 admission/SHA，不重生成 scalar，也不称同 job 测量。

拟议单次资源为 cnall/1node/1task/48CPU/0GPU/3600s、非exclusive/no-requeue/无新增mem或节点限制。
未核实费率的最坏费用 CNY2.40/3.36；60分钟是探索停止边界，真实B64编译、硬门、热吞吐/RSS及CPU48/GPU比仍未测量。
最终固定代码、审计绑定、目标shell检查和精确命令齐备后交付主管审核，本聊天交付后停止等待派工。
