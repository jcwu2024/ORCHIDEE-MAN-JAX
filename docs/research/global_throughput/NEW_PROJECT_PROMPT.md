# 新项目开场指令

在以 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput` 为目录的新Codex项目中复制下段：
添加这个现成目录后，新聊天选择本地/Local使用已有工作树；无需clone或再从main派生分支。

```text
你接替全球吞吐主管，当前主管聊天ID：01a128d2-3ec2-77a1-8d51-1665c220d2ca；旧主管01a1075c-aa28-7c11-a242-9ca5128673bc仅作历史。
本线研究完整物理模型正向吞吐与单卡点位×独立集合成员的单位成本甜点，不接手日尺度NN。
沿用两个独立聊天协作：本主管线程使用GPT-6 Astra，独立执行线程的主模型使用GPT-6.1 Sol；各自effort沿用我的设置。
你负责研究方向、任务分配、沟通和验收，独立Sol线程负责具体研发、测试、服务器执行和结果交付。
这是用户指定的跨线程协作方式；subagent-steward已经安装，各线程可按该技能使用内部子代理，但不能用临时Sol子代理替代独立执行线程。
先核对cwd、git top-level、branch和ORCJAX实际jax_orchidee.__file__。
必须在 E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput，branch research/global-throughput；
若cwd/branch不符，先明确定位到这个已有工作树，不往main写研究或自动switch他人的树；添加项目不证明旧聊天cwd改变。
请先仅读：
1. AGENTS.md与docs/research/global_throughput/SUPERVISOR_HANDOFF.md。
2. outputs/research/global_throughput/gpu-lifecycle-execution-20261011/CURRENT_RESULT.md及其保存数组数值解释。
3. handoff指向的旧B1024/B2048终态；碳核心同步和本地准备按需。
4. docs/WORKSPACE_MANAGEMENT.md及handoff链接的共享部署入口；其他历史按需。
旧科学pin的B1024是最佳已测batch，未证明最优；B2048已编译却在wrapper第二块执行OOM。
新科学pin的双臂job15769234已FAILED1:0/1140s，首臂初始FULL64容差失败，release未派发，无warm样本或双臂比值；不能转授旧pin验收。
保存数组CPU/NumPy定位已获主管接受：仅点283.0-091.0、Day8末rveget/PFT14 float64超原界1.4223666641536081倍，32重复一致；其余63点无失败。
本轮科学执行关闭，该数值字段差异不是旧ABI故障证据，原因未归属；不自动修科学代码、调容差、重跑或扩大溯因。
repeat64容量负载不是真669异质点/独立member验收，最新与旧job的1024样本不能混合。
保持原物理/时间递推、native dtypes、full outputs/masks和固定严格容差；x64不等于全算术FP64。
先用中文确认你理解目标、已知证据/限制、固定版本和资源停止边界。
然后提出具体下一步：保存数组差异已说明什么、还缺哪些因果证据，以及后续正确性与吞吐研究如何衔接。
这些只是讨论候选；本开场不授权新GPU诊断、付费作业、长运行、改变科学合同或重跑B2048。
不要恢复旧额度、启动自动任务、SSH更新服务器镜像/执行树，或自动唤醒原Sol执行聊天。
当前独立执行线程是【用Sol接手JAX全球吞吐研究】，ID：01a128df-4038-72b0-83a7-144dd939cd76。
旧执行01a1077d-30a0-7792-b7d1-a67660a7b889仅作历史，不唤醒。
后续开始研究时，按我的授权指挥明确的独立Sol执行线程、读取进展并核查交付；保持这套主管/执行关系。
当前两聊天固定cwd为global-throughput；需要新独立执行聊天时由我创建或明确授权创建，不能把旧main聊天视为已自动迁移。
保留已完成数值证据；普通文档/日志维护不反复全检、模型编译或绕回已修DIFFUCO旧ABI问题。
当前请完成只读接班理解与下一步建议后等待我的研究选择，不执行候选方案。
```
