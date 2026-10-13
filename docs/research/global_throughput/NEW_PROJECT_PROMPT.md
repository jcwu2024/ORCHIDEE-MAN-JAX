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
2. outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/CURRENT_RESULT.md、AUTHORIZATION.md、execution-code-pin.json、PAID_PROPOSAL.md及其指向的环境修复/独立scalar验收证据。
3. handoff指向的已接受AOT修复与CPU/GPU比较；旧B1024/B2048终态、生命周期保存数组解释和碳核心同步按需。
4. docs/WORKSPACE_MANAGEMENT.md及handoff链接的共享部署入口；其他历史按需。
旧科学pin的B1024是最佳已测batch，未证明最优；B2048已编译却在wrapper第二块执行OOM。
新科学pin的双臂job15769234已FAILED1:0/1140s，首臂初始FULL64容差失败，release未派发，无warm样本或双臂比值；不能转授旧pin验收。
保存数组CPU/NumPy定位已获主管接受：仅点283.0-091.0、Day8末rveget/PFT14 float64超原界1.4223666641536081倍，32重复一致；其余63点无失败。
本轮科学执行关闭，该数值字段差异不是旧ABI故障证据，原因未归属；不自动修科学代码、调容差、重跑或扩大溯因。
CPU48 v2的job15805735已FAILED1:0/109s：真实allocation通过，scalar lower后签名helper对Python float读取.shape失败；compile及B64未运行，单次付费授权已消耗。
主管已接受研究提交d33fa65bb263b1564ca98f72b81a69e93aabc33f的有界AOT修复：首个失败叶子.values_by_component[4][67]为Python float，helper改用JAX shaped_abstractify。
Linux JAX0.4.38上确切候选通过两组64点、各536叶元数据及实际scalar/B2连续两块core/wrapper替身接口，完整carry保留；构造weak_type审计和native call行为不变。
该接受只覆盖元数据/接口，复用已完成证据，不称完整物理编译、科学接受或吞吐已验证。
用户已明确授权主管审核后批准合适新作业；2026-10-14说“怎么样了。继续。”后主管派工原独立Sol线程完成CPU48 v4不付费修复、保存scalar独立验收和可执行准备。
先查看v4交付的具体状态，不恢复旧额度或自动提交；主管复核新固定pin/命令/证据后可按用户已有授权决定单次提交，无需把同一方案重复问回用户。
CPU48 v4不付费准备已生成固定执行SHA60dccbddde10be4295aab7e7fe55fa7864d3c3f3，ref codex/global-cpu48-b64-v4；来源研究检查点493252e418f3fd827af4a6f7748a63d3157e680d，具体命令在v4准备目录PROPOSED_SUBMISSION.sh，从未执行sbatch。
执行pin直接以03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8为基底，仅纳入15项harness/tests；104物理blob及tree75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba、science11b6…、actual sourceb334…逐字节保持，研究树两项碳修复保留。
独立scalar审计job15852254在免费cntest/ibtest02单CPU/affinity[0]/23s完成，64输入身份和点序、760叶及GPU门独立重算为61通过、3点有限超差（4处比较），reset仅原封存证明。审计SHAcb1331edb7094e74ec1814c6813a0eee218c58ea4e89f1c0c1f630e981479431；已绑定新pin的admission SHA f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d，科学false、candidate job_id null、旧job/code身份保留。
环境/资源替身108项、最后runner/collector改动43项和封装14项检查通过，较早reader-inclusive 62项明确标记版本；这些不是实际B64科学/性能验收。最终固定pin目标staging检查与早期占位记录分开，不能称未来production checkout已经部署或通过真实48核准入。
未来代码路径为/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v4；未来结果namespace为runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014。
旧job15820467已FAILED1:0/2197s：scalar FULL64完整seal，B64在线程预算资源门失败，未JAX初始化/物理编译/热计时；旧child实际OMP值未记录。
原源码缺口是生产Popen不传mode环境、srun --export=ALL含重复键，dry-run却显式传env。v4已在post-transport child边界确定mode环境，生产Popen显式传env，预检复用同一launch；保留实际48核/独立物理核硬门和成功/失败白名单回执。Slurm22.05.7免费单CPU重复键复现只证明被测试命令的继承值优先级，不证明历史child的OMP值。
下一候选B64-only，显式引用旧job15820467的原scalar目录和独立验收artifact/SHA，不能冒充同job测量、覆盖旧身份或静默重跑scalar。
提案cnall/1node/1task/48CPU/0GPU/3600s，非exclusive/no-requeue/无新增mem或节点限制，全父计费最坏CNY2.40/3.36，费率未核实。B64编译时间未知，60min只是探索性停止边界。
64 unique FULL64点、Days2–8和9–15、48步/天、896 point-days、760完整native leaves/masks，各5core+5wrapper，严格容差及全部CPU内部整组/permutation/replacement-isolation/reset/结构/非有限值硬门不变。
GPU B64 job15579869为同量比较对象；B1024 job15671732的299.139825仅不同总量归一化参考。CPU48科学/热吞吐和同量CPU48/GPU比仍待真实B64运行。
旧scalar各5core/5wrapper，中位127.07498395815492s/121.38244984298944s；61/64GPU参考门通过、3有限容差失败、冻结reset保存证明通过。v3 collector本身仅失败记账；独立数组验收和可复用结论以v4新artifact为准，科学状态仍false。
旧父allocation2197s×48=29.293333核时，未核实费用CNY1.464667/2.050533已经发生只计一次；新job费用另列。scalar HWM19.94GiB，与checkpoint当前RSS5.586GB分开。
repeat64容量负载不是真669异质点/独立member验收，最新与旧job的1024样本不能混合。
保持原物理/时间递推、native dtypes、full outputs/masks和固定严格容差；x64不等于全算术FP64。
先用中文确认你理解目标、已知证据/限制、固定版本和资源停止边界。
然后说明v4环境传递修复/真实child回归、保存scalar独立验收结论、具体新pin/命令/资源/费用/停止条件，以及仍待真实B64验证的部分；不持续等待/轮询执行聊天。
这些只是讨论候选；保存数组差异已接受且科学执行关闭，不自动扩大溯因。本开场不授权新部署、GPU诊断、付费作业、长运行、改变科学合同或重跑B2048。
不要恢复旧额度、启动自动任务、SSH更新服务器镜像/执行树，或自动唤醒原Sol执行聊天。
当前独立执行线程是【用Sol接手JAX全球吞吐研究】，ID：01a128df-4038-72b0-83a7-144dd939cd76。
旧执行01a1077d-30a0-7792-b7d1-a67660a7b889仅作历史，不唤醒。
后续开始研究时，按我的授权指挥明确的独立Sol执行线程、读取进展并核查交付；保持这套主管/执行关系。
当前两聊天固定cwd为global-throughput；需要新独立执行聊天时由我创建或明确授权创建，不能把旧main聊天视为已自动迁移。
保留已完成数值证据；普通文档/日志维护不反复全检、模型编译或绕回已修DIFFUCO旧ABI问题。
当前请完成只读接班和实际v4交付审阅，按已有用户授权处理主管审批边界；不因这段开场自动部署/提交、创建自动任务或执行其他候选方案。
```
