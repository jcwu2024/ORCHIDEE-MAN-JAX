# CPU48 v4 准备结果

更新：2026-10-14。环境传递修复、保存 scalar 独立审计、固定执行包及最终目标端非模型检查已完成。
范围见[授权](AUTHORIZATION.md)，合同见[协议](EXECUTION_PROTOCOL.md)，精确资源、命令与停止条件见[提案](PAID_PROPOSAL.md)。
本轮没有 sbatch、GPU、完整物理编译、新生产部署或 scalar 重生成；本聊天交付后停止等待派工。

执行 pin 为 `60dccbddde10be4295aab7e7fe55fa7864d3c3f3`，ref `codex/global-cpu48-b64-v4`。
审核源码首先提交于 `01b09549cea3ff133f53d72a2a9444a0a31d7ed4`；执行包来源检查点为
`493252e418f3fd827af4a6f7748a63d3157e680d`。
[pin 清单](execution-code-pin.json)绑定全部身份、15 个 harness/test 原始 blob、patch 和 prerequisite bundle。
[本地实际封装核验](local-package-verification.json)确认执行版的 104 个物理 blob 与 v3 基底逐字节相同，
tree `75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`、science `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5` 不变；
研究树 `62e90547ee8eadab387713397bb111c399c81b6c` 和两项 carbon 修复保留。
生成命令见[封装回执](package-creation-receipt.json)，核验脚本见[verify-local-package.py](verify-local-package.py)。

v3 的生产 Popen 未显式传 mode 环境，srun 使用含重复键的 `--export=ALL`，dry-run 则显式传环境。
[Slurm 22.05.7 免费单 CPU 复现](environment/slurm-export-precedence.json)确认被测试命令的继承值优先级；
历史 job 15820467 没有记录实际 child OMP 值，不能据此声称当时观察到 OMP=1。
修复使正式启动与预检共用 post-transport child 边界，记录白名单环境/affinity，并保留真实 48 个独立物理核硬门。
启动/资源/替身回归 108 项通过；占位 pin 的早期目标记录保留在[环境验证](environment/ENVIRONMENT_VALIDATION.md)。
固定 pin 的最终检查在免费 cntest/ibtest03、job 15852487 上完成：1 task/1 CPU、实际 affinity [0]，
终态 COMPLETED 0:0、4s。提案与 launcher 的 Bash 语法、launcher→coordinator→runner dry-run、
真实旧 scalar 封存校验均通过，见[单独的固定 pin 回执](environment/exact-pin/CPU48_B64_V4_EXACT_PIN_CHECK_RECEIPT.json)。
该回执 SHA256 为 `ef32710afcf98560283970adf2d6115a575cc718147a9887d30c13b07db61445`。
实际 driver、原始/映射 argv、18 份日志 SHA 和原始调度记录由该回执关联；根审核确认六个 staging 源文件
与固定 Git 原始 blob 一致，18 份保留日志哈希匹配。原 `allocation.txt` 的 `host=1` 是格式缺陷，
节点依据原 scontrol 的 NodeList/BatchHost=ibtest03 和已执行 hostname 断言；原日志保留。
原 source/input 远端 hash 比较输出未保留，transfer map 的远端一致性属于先前观察记录；封包期间未独立远端重查。
该检查使用与提案 SHA 相同的 staging admission 副本，仅映射 coordinator executable 和 `--entry` 到 staging，
未来 code/config/results 参数保留。validator 依据封存 pin 的声明 candidate identity/job_id=null 核对旧 seal，
未从尚未部署的未来 checkout 重算源身份。单 CPU 上的 OMP48 是环境归一化证据；真实 48 独立物理核仍须运行时准入。
未来 production checkout、results 和 persistent admission 未创建；不把占位检查或 staging 成功称为生产/科学/吞吐验收。
此前 15852458 的未使用 allocation、15852476 语法检查后驱动失败、15852482 多余 override 成功分别保留。
根封装 14 项轻量测试通过，见[测试回执](package-local-check-receipt.json)及[原始输出](package-final-binding-tests.stdout.txt)。
较长临时路径触发的 Windows Git 限制保留[失败记录](package-admission-tests.stdout.txt)，未放宽断言或科学合同。

保存 scalar 的独立审计在免费 cntest/ibtest02、job 15852254 上完成：1 CPU、实际 affinity [0]、23s。
[审计原始回执](scalar-audit/scalar-audit-receipt-v3.json)为 146,678 B，SHA256：
`cb1331edb7094e74ec1814c6813a0eee218c58ea4e89f1c0c1f630e981479431`。
64 个 prepared lane 输入 digest 和精确点序独立重算、64 unique 点及全部 760 叶 shape/native dtype 核对通过。
64 个 GPU 门独立重算为 61 通过、3 个点有限容差失败（4 处比较），严格科学状态仍为 false。
原 reset 重复数组未保留，reset 仅核对原封存门证明，不称独立数组重算。
实际绝对路径 srun 命令、原回执错误 shorthand 和初次 PYTHONPATH 设置失败分别见[执行说明](scalar-audit/audit-execution-notes.md)。

[scalar admission](scalar-audit/scalar-admission.json)已绑定上述审计和新执行 SHA，145,538 B，SHA256：
`f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d`；
状态 `admitted_for_internal_comparison_only`，candidate job_id 为 null，原 job 15820467/code/checkpoint 身份独立保留。
[验证索引](SCALAR_ADMISSION_VALIDATION.md)保留最后 runner/collector 修改后的 43 项通过，以及较早且明确标注版本的 62 项通过。
合成集成使用生产 gate emitter 与 sealer，并验证改数组/索引后重封和 digest 篡改拒绝；
这不是实际 B64 科学门验收。新候选 B64-only，无 scalar 回退或重试。

单次提案是 cnall/1 node/1 task/48 CPUs/0 GPUs/3600s，非 exclusive、no-requeue、无新增 mem 或节点限制。
未核实费率下最坏 CNY 2.40/3.36；60 分钟保留约 3209s 原拟议候选窗口及入口余量，是停止边界。
旧父 allocation 2197s × 48 = 29.293333333 核时、费用情景 CNY 1.464666667/2.050533333 已发生，只计一次；新父另计。
真实 B64 编译、硬门、热吞吐/RSS、同量 CPU48/GPU 速度与费用比仍未测量；主管按已有用户授权审查具体提案后决定单次部署/提交。

此前多目录递归 fixture 清理被自动审批拒绝，理由仅为 “blocked by policy”；未删除且不重试。
忽略的 fixture 目录（含 `outputs/tmp/cpu48v4p-final2`）保留，不纳入 Git 或执行 bundle。
