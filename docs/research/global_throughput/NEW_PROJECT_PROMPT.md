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
2. outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/CURRENT_RESULT.md、execution-code-pin.json和PAID_PROPOSAL.md。
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
最新授权仅为CPU48 v3未付费准备。执行SHA为03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8，ref codex/global-cpu48-b64-v3，直接基于v2 b1b5cf0835da298bddb388e61d754feeb6c14cf2。
只移入六个必要harness/tests，104个物理文件及整个jax_orchidee树与v2一致，science11b6…不变；研究树两项碳修复保留。
ORCJAX/JAX0.10.1/x64准备测试15通过；实际detached v3 source_identity通过，原10项容量protocol/launcher失败按具体ID全部通过；另10项AOT/source-certificate筛选通过（50未选）。原始回归记录区分两组，不是整模块60项重跑或目标runtime接受，不回写原研究50通过/10身份门失败记录、不放宽guard或回退碳修复。
新代码目录为/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3；结果namespace为runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012。
目标小型launcher/coordinator/runner CLI dry-run仅用临时staging，无JAX/模型/srun/sbatch，staging已移除。v3未生产部署，bundle/pin及新提案供主管核验。
拟议cnall/1node/1task/48CPU/0GPU/90min、非exclusive/no-requeue/无新增mem；全父48核计费CNY3.60/5.04情景，费率未核实，健康匹配节点正常排队、不锁n35。
64 unique FULL64点、Days2–8和9–15、48步/天、896 point-days、760完整native leaves/masks；scalar先行且cap2700s，B64须剩至少1800s，各5core+5wrapper，原硬门不变。
GPU B64为同量比较对象；B1024的299.139825仅不同工作量归一化参考。完整物理lower/compile、CPU48科学/热吞吐和同量CPU/GPU比尚未知。
先核验实际准备与提案，后续由主管取得用户新的单次作业批准；本开场不自动部署、lower/compile、GPU或sbatch，不复用已消耗v2额度。
repeat64容量负载不是真669异质点/独立member验收，最新与旧job的1024样本不能混合。
保持原物理/时间递推、native dtypes、full outputs/masks和固定严格容差；x64不等于全算术FP64。
先用中文确认你理解目标、已知证据/限制、固定版本和资源停止边界。
然后说明CPU48 v3准备是否具备审阅条件、仍需何种部署核验与新的单次审批，以及测量与科学接受的区别。
这些只是讨论候选；保存数组差异已接受且科学执行关闭，不自动扩大溯因。本开场不授权新部署、GPU诊断、付费作业、长运行、改变科学合同或重跑B2048。
不要恢复旧额度、启动自动任务、SSH更新服务器镜像/执行树，或自动唤醒原Sol执行聊天。
当前独立执行线程是【用Sol接手JAX全球吞吐研究】，ID：01a128df-4038-72b0-83a7-144dd939cd76。
旧执行01a1077d-30a0-7792-b7d1-a67660a7b889仅作历史，不唤醒。
后续开始研究时，按我的授权指挥明确的独立Sol执行线程、读取进展并核查交付；保持这套主管/执行关系。
当前两聊天固定cwd为global-throughput；需要新独立执行聊天时由我创建或明确授权创建，不能把旧main聊天视为已自动迁移。
保留已完成数值证据；普通文档/日志维护不反复全检、模型编译或绕回已修DIFFUCO旧ABI问题。
当前请完成只读接班理解、v3准备和提案审阅，随后等待我的新单次作业决定，不执行候选方案。
```
