# 新日尺度项目开场指令

项目目录：`E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining`。
下面整段可作新主管线程第一条消息，不必复制旧聊天历史。

```text
请接任日尺度神经算子研发的主管，负责研究判断、任务安排、证据验收和向我解释。
目标是高效替代ORCHIDEE-MAN亚日尺度过程，使日尺度模型长期模拟的
GPP/NPP/呼吸和AGB/BGB轨迹可靠，同时能用足够数据规模训练。

固定工作目录：E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining
固定分支：research/daily-coarse-graining
根目录现在是main，不要在那里写研究代码或切其他工作树的分支。
先确认git顶层目录、分支、工作树状态，以及ORCJAX实际Python导入路径。

遵循AGENTS.md和docs/WORKSPACE_MANAGEMENT.md，依次阅读：
1. docs/START_HERE.md
2. docs/NEXT_STEPS.md
3. docs/CODE_MAP.md
4. docs/DOCUMENT_STATUS.md和docs/current-status.md
5. docs/research/daily_coarse_graining/SUPERVISOR_HANDOFF.md
只补读与本次判断相关的结果和合同，不遍历历史实验。

已有同四条7日轨迹合批约2.55倍组件提速，但状态梯度验收未通过；
最新结果将有效差异定位到原始神经输出阶段，具体原因未定。
这不是完整训练提速，也不是长期精度验收。GPU诊断已结束，旧额度已关闭。
项目目录迁移和代码同步已完成，旧交接待提交清单是历史快照，以当前Git状态为准。

这次先接手、做必要的只读证据核查，简明告诉我：有效成果是什么，
真正挡路的问题是什么，推荐下一块工作的理由和交付物。
不要自动开GPU实验、长训练、付费任务或再次清理项目。
不要把每次微小失败变成一次新申请，也不要无止境地切图诊断。
后续授权后连续完成实现、相应验证、修复和汇总，直到约定交付点。

执行优先按subagent-steward技能委派给明确选择模型/强度的Sol/Luna子智能体，
你保留主管判断和实际产物验收。先核对可用模型，不自动复用仍以根目录为cwd的
旧执行聊天。我缺少神经网络工程经验，请给出明确建议，不让我替你判断方案。
保持当前精度、梯度、物理和数据边界，重大改变先讨论。
```
