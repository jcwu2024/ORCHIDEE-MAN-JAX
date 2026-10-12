# CPU48 v3 执行线程有界产物验收

验收日期：2026-10-13。准备产物已达到供主管审阅新的单次作业提案的条件。
本次接受的是独立执行pin、准备回归、入口证据和具体提案；生产部署和付费提交仍待后续授权。

执行SHA为`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`，ref `codex/global-cpu48-b64-v3`，
parent为v2 `b1b5cf0835da298bddb388e61d754feeb6c14cf2`，研究源检查点为
`8a860c144f6263fed5b9bf246a88f6d83d36797f`。实际包、ref/parent、patch/bundle及独立源码审查均无阻断。
104个物理文件和整个`jax_orchidee`树与v2完全一致，tree OID为`75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`，
science证书`11b6…`和actual source hash `b334…`保留；研究树两项碳修复未回退。
AOT文件SHA `bc8ce1d62f6947228c8c8438344987345bf18095f307c90a2f45dc4655c6891e`
与主管已接受的`d33fa65…`候选完全相同。

| 实际证据 | 核验范围及结果 |
| --- | --- |
| [pin清单](execution-code-pin.json)、[执行patch](execution-harness.patch) | 六个必要harness/tests；bundle为3380B，SHA `940a6f7e…`，Git验证通过，依赖v2 base |
| [本地回归记录](regression-recheck-20261013/REGRESSION_RECEIPT.md) | clean detached v3，Python3.11.15/JAX+jaxlib0.10.1/x64；准备15通过，**原10个失败node IDs全部通过**，另AOT/证书10通过；Ruff/Bash/source admission通过 |
| [原十项结构记录](regression-recheck-20261013/original-capacity-protocol-tests.json)及邻接原始stdout | 原十项与旧AOT报告失败清单一致，10通过/0失败/5.68s/exit0；与补充筛选分开，不称整模块60项重跑 |
| [目标入口记录](TARGET_SMOKE.json)、[命令](TARGET_SMOKE_COMMANDS.md)及target原始stdout | ibtest01的四个精确pin文件临时staging；launcher展开与独立coordinator→scalar/batch CLI dry-run通过，无JAX/模型/srun/sbatch；临时目录删除并确认不存在 |
| [单次作业提案](PAID_PROPOSAL.md)、[完整协议](EXECUTION_PROTOCOL.md) | pin/bundle/runtime/data/cwd/命令已具体绑定，subshell清除继承SBATCH及harness override；cnall/1node/1task/48CPU/0GPU/5400s，非exclusive/no-requeue/无新增mem，不锁n35 |

完整父allocation含scalar均按48核计费，未核实费率的worst-case为CNY3.60/5.04。
同64 unique FULL64点、连续Days2–8及9–15、896 point-days，scalar cap2700s，B64须至少剩1800s，各5 core+5 wrapper；
760完整native outputs/masks、x64/原dtype及`atol=1e-8/rtol=1e-10`和原硬门保持。
GPU B64是同工作量对象，B1024的299.139825只作不同工作量归一化参考。

复用[已接受的Linux JAX0.4.38候选元数据/替身接口证据](../cpu-aot-signature-repair-20261012/linux-jax-0.4.38-candidate-full-metadata.json)，
不把本地回归或目标CLI检查升级为完整物理lower/compile、科学一致性、CPU48热吞吐或CPU/GPU比。
v3生产checkout仍未部署，未提交任何新作业；旧v1/v2授权已消耗，失败证据及旧pin不改。
图工具当前为Transport closed，本次按已知路径直接源码/实际Git包核验，不作全仓结构或覆盖完整性结论。
路径专用Git attributes保留原始JSON、patch及日志字节；31份原始产物在index中与磁盘字节完全一致。
bundle的SHA/3380B大小已核对，保留在本地本目录，不把bundle、pickle或模型数组加入Git。
