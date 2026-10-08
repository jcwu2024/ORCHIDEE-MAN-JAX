# 单卡批量容量：B256 已验证，B512 未完成

2026-10-08 获批作业 **15643868 已结束**：15:31:39–15:53:15 Asia/Shanghai，`ibc13b03n02`，`FAILED`，exit `1:0`。B256 完整阶段通过并密封；B512 在 lowering 后的 AOT 签名审计发现 `expected536, internal673`，在 compile/run 前终止，没有完整调用、数值输出、计时样本或检查点。没有自动重试、新作业或新 automation；既有 `gpu-15600405` heartbeat 保持 `PAUSED`，旧 Explore1000 heartbeat 不恢复。

当前目标是找到未来“点位 × 参数/情景集合成员”独立轨迹的单卡批量费用/吞吐区间。**B256 是目前两个同工作集合配置中费用较低的候选，最优批量尚未找到。** B512 缺少测量，无法判断平台或下降。B128 使用历史 lazy C++ dispatch，B256 使用 bound Python dispatch；以下比较包含执行路径差异，不能把增益全部归因于 batch size。

## 相同工作集合的热计时与单位费用

两项都是有序 fixed512 collection：64 个真实 FULL64 点各重复 8 次，每 lane 连续推进 Days2–8、9–15，每天 48 步，合计 **7168 lane-days**。唯一真实点对应 896 unique-point-days；重复计算不增加科学样本，也不代表已实现不同成员输入。

| 配置 | core 中位数；五次全范围 (s) | wrapper 中位数；五次全范围 (s) | wrapper lane-days/s | 主口径 CNY/百万 lane-days |
| --- | --- | --- | --- | --- |
| B128，已封存基线 | 29.617320834；29.605399007–29.623860434 | 33.838030240；33.780365391–33.889177628 | 211.832661 | 3.147138229 |
| B256，本次完整阶段 | 27.024156772；27.022209791–27.044812379 | 30.468557430；30.319014769–30.538280688 | 235.258923 | 2.833757201 |
| B512 | 无有效样本 | 无有效样本 | 不发布 | 不发布 |

主指标 wrapper 包含已准备输入的整理、host/device 传输、完整双块执行及完整输出返回；core 吞吐用于解释执行成本，B256 为 265.244169 lane-days/s。两种计时均有一次完整同步 warmup 后的五次同步完整 collection，原始 5+5 样本见[精确汇总](collector-continuation-actual-summary.json)。wrapper 不含重新生成 Day1/初态、年度汇总或生产检查点。

费率情景为 1 V100 × CNY2.20/card-hour，加 4 allocated Slurm CPUs × CNY0.05（保守 0.07）/CPU-hour，即 CNY2.40/2.48 每小时；不是核实账单。B256 主/保守 wrapper 为 **CNY2.833757201/2.928215775 每百万 lane-days**。按同费率，观测 wrapper 吞吐增加 11.0589%，单位费用降低 9.9577%，成本与吞吐互为倒数。完整作业费用不除以单次 7168 lane-days。

## 设置、IO 和完整作业费用

| B256 单独记录的边界 | 秒 |
| --- | --- |
| host stack / device transfer | 2.773223046 / 0.883987846 |
| lower / compile | 93.373039044 / 686.842093194 |
| 首次完整 fixed512 调用 | 27.168201223 |
| checkpoint serialization / atomic write | 0.294933960 / 0.271023320 |

首调用不作为热吞吐；checkpoint IO 不含 manifest/report/event 写入。B512 lower 为 63.639648404 秒，之后签名审计拒绝，未 compile。时间/资源准入曾通过（剩余 4183.28 秒）；这次失败不是 OOM、timeout 或数值门槛失败。B256 compiler temporary buffer estimate 为 1,151,609,352 bytes，累计 process peak RSS 为 16,139,980,800 bytes；它们均不是实测 VRAM 峰值。

[实际 sacct 父行](job-15643868-terminal-parent-final.txt)为 1296 秒、4 CPUs、1 GPU；只计父行一次，不叠加 child steps。完整 job 为 0.36 GPU card-hours 和 1.44 allocated CPU-hours，估算 **CNY0.8640 主口径 / CNY0.8928 保守口径**，包含启动、编译、门槛、重复测量、IO 及失败阶段。长期复用的热单位费用与短任务总费用分开；缺少完整独立观察到的设置/验证费用模型，不给 K 复用估计。

## 验证和接受范围

test02 的 canonical CPU 环境只读取已保存结果，未编译或执行模型。完整 checkpoint SHA、science/input identity、native dtype/schema、760 leaves ×512 lanes、两个完整七日块、515 项原严格 gates、512 项独立 saved-output/B4-reference comparisons 及 5+5 样本均通过。[完整验证](job-15643868-terminal-validation.json)和[collector 验证记录](collector-continuation-verification.json)保留可复核证据；collector 的 56 项测试与 Ruff 通过。

B256 checkpoint 为 220,429,675 bytes，SHA256 `ee7210f2c83ea11346e9aa752320e8760560b2a6c1e377aa2058aa53da16b08c`。完整[checkpoint](job-15643868-B256-evidence/capacity/stage_capacity_B256_fixed512_v1.pkl)、[manifest](job-15643868-B256-evidence/capacity/stage_manifest.json)、[report](job-15643868-B256-evidence/capacity/report.json)、启动/GPU 诊断及日志已保全；[终态 capture](job-15643868-B256-evidence/capture.json)的 12 项文件大小和 SHA 均由[本地传输验证](local-transfer-verification.json)核对通过。大型 snapshot 保留在本地 outputs 和 remote capture，未加入 Git。

代码保持不可变：`global-throughput-gpu-sweetspot-v3`，commit `12b495b9425de3ad1dc16efa3a5a0e69d7e89222`；物理源码相对 `a33b82d` 无差异，原生精度、完整输出、mask 和容差未改。B128 原检查点/执行 provenance 只读重验，没有重编译或重计时。历史 job15600405 的 B256 自然触发原因仍未证明；本次 B512 是新观察到的 lower-metadata mismatch，见[失败诊断](job-15643868-failure-diagnostics.json)。接口修复与 v2 启动失败的证据见[修复报告](REPAIR_REPORT.md)及[付费执行记录](PAID_RUN_REVIEW.md)。

容量筛查保持 `scientific_acceptance=false`。CPU 没有已接受的同工作量性能阶段，本轮不发布 CPU/GPU 加速比、成本胜负或 44 核 CPU 结果。Fortran、年度、真实 669 点、50 年、AD/训练及 10× 均未验收。

## 真实 669 点任务与后续方案

669 是单情景空间任务数，不是未来集合并行的 batch 上限。真实 669 点任务需另测异质输入、初始化、设置、尾批/填充、聚合和生产 IO。仅作 **14 日热算术投影**：B256 的两满批加 157 真实点尾批，若尾批填充 99 lanes，则处理 768 slots；假定每 chunk 时间与本次相同，wrapper 约 45.702836 秒、CNY0.030468557/0.031484176。该假定未在真实 669 点验证，未含 Day1、设置或 IO；未填充 B157 也未测量，不把投影称作总任务结果或外推 50 年。

后续先提出有界免费 B512 metadata 诊断范围，再为任何付费补测单独审查资源、时间和费用；本次不执行诊断或重试。先补齐 fixed512 的 B512。若最大有效批量仍明显上升，可提出新的共同 **fixed1024** 集合，比较 B256/B512/B1024（每 collection 分别 4/2/1 满批），沿用双七日、48 步、完整输出、原容差及五次同步样本。64 点重复 16 次只能作为容量筛查；不能用现有 fixed512 假称满载 B1024。若需细化 256–512 区间，可在共同 fixed1536 上比较 B256/B384/B512（6/4/3 满批），另做资源/时间准入。平台附近优先选接近最高吞吐且显存/编译负担较低的范围。最终候选需不同真实点和实际不同成员输入验证；不改变分辨率、不实现集合科学方法，也不假设 50 年可拆成独立年份。

[执行账本](preparation-ledger.json)保留批准范围、已消耗的一次提交、终态费用和下一步设计；本次跟进已完成。
