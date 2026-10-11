# 新项目开场指令

在以 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput` 为目录的新Codex项目中复制下段：
添加这个现成目录后，新聊天选择本地/Local使用已有工作树；无需clone或再从main派生分支。

```text
你接替主管【研究 JAX 全球运行效率】，原聊天ID：01a1075c-aa28-7c11-a242-9ca5128673bc。
本线研究完整物理模型正向吞吐与单卡点位×独立集合成员的单位成本甜点，不接手日尺度NN。
沿用两个独立聊天协作：本主管线程使用GPT-6 Astra，独立执行线程的主模型使用GPT-6.1 Sol；各自effort沿用我的设置。
你负责研究方向、任务分配、沟通和验收，独立Sol线程负责具体研发、测试、服务器执行和结果交付。
这是用户指定的跨线程协作方式；subagent-steward已经安装，各线程可按该技能使用内部子代理，但不能用临时Sol子代理替代独立执行线程。
先核对cwd、git top-level、branch和ORCJAX实际jax_orchidee.__file__。
必须在 E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput，branch research/global-throughput；
若cwd/branch不符，先明确定位到这个已有工作树，不往main写研究或自动switch他人的树；添加项目不证明旧聊天cwd改变。
请先仅读：
1. AGENTS.md与docs/research/global_throughput/SUPERVISOR_HANDOFF.md。
2. outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/CURRENT_RESULT.md。
3. outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/CURRENT_RESULT.md。
4. docs/WORKSPACE_MANAGEMENT.md及handoff链接的共享部署入口；其他历史按需。
当前B1024是最佳已测batch，未证明最优；B2048已编译却在wrapper第二块执行OOM。
repeat64容量负载不是真669异质点/独立member验收，最新与旧job的1024样本不能混合。
保持原物理/时间递推、native dtypes、full outputs/masks和固定严格容差；x64不等于全算术FP64。
先用中文确认你理解目标、已知证据/限制、固定版本和资源停止边界。
然后提出具体下一步：保存证据能查什么、还缺什么，以及中间batch或执行内存优化如何选择。
这些只是讨论候选；本开场不授权新GPU诊断、付费作业、长运行、改变科学合同或重跑B2048。
不要恢复旧额度、启动自动任务、SSH更新服务器镜像/执行树，或自动唤醒原Sol执行聊天。
原独立执行线程是【用Sol接手研究JAX全球运行效率】，ID：01a1077d-30a0-7792-b7d1-a67660a7b889。
后续开始研究时，按我的授权指挥明确的独立Sol执行线程、读取进展并核查交付；保持这套主管/执行关系。
旧聊天的默认cwd仍在根main，开始执行前必须解决正确工作目录；需要新独立执行聊天时由我创建或明确授权创建，不能把旧聊天视为已自动迁移。
保留已完成数值证据；普通文档/日志维护不反复全检、模型编译或绕回已修DIFFUCO旧ABI问题。
当前请完成只读接班理解与下一步建议后等待我的研究选择，不执行候选方案。
```
