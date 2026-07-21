# Daily Coarse Persistence Baseline Review

状态：**persistence_baseline_failed / neural_learnability_inconclusive**。
Teacher commit `7333b46`，实验日期 2026-07-21，结论复核于同日。

## 结论

本实验只否定 persistence baseline，不否定日尺度 operator 或神经网络的可学习性。
不得据此启动大规模训练或训练集生成；是否值得批准一个极小 supervised
learnability + synthetic operator cost pilot，需要单独决策。

无神经 persistence baseline 的计算速度远高于 `3x` 门槛，但 8 日自由 rollout 已
出现预期的科学失败。按照该 baseline 的预注册退出条件，未继续运行 30 日。这里的
退出对象是 persistence baseline，不是整个日尺度研究方向。

## 实验边界

baseline 每天只接收：

- 自己上一日产生的完整动态 state；
- 当天 48 步 forcing。

它不接收 Teacher 日末状态、daily interface 或 OK_LEAK target。Teacher capture
只用于运行后的误差评分和零推理 replay 对照。

日末 SECHIBA 状态采用 persistence；17 个 daily/maintenance 字段由当天 forcing
统计和日初 carry 构造；13 个 common OK_LEAK 字段及条件 `deepC_peat` 采用
persistence。保留 Teacher tail 从 season/STOMATE 日碳开始执行。跨年 Day 1 使用
独立、无 target 的单日 executable，Day 2--8 使用一个可复用的 7 日 executable。

执行命令：

```powershell
conda run -n ORCJAX python -m research.daily_coarse_graining.persistence_baseline `
  --state-cache outputs/acceptance/compiled_1961_12point_wetdiaglong_fix/001.0-071.0/compiled_checkpoints/paper_driver_1961_year_end_state.pkl `
  --year 1962 --days 8 --repeats 3 `
  --output outputs/performance/daily_coarse_graining/persistence_baseline_go_no_go.json
```

## 性能

CPU：Intel Family 6 Model 183，JAX `0.10.1`，单 `cpu:0`。

| 路径 | 8 日热运行中位数 | 秒/日 | 相对 Teacher |
| --- | ---: | ---: | ---: |
| complete-day Teacher | 0.9045 s | 0.1131 | 1.00x |
| dynamic zero-inference replay | 0.1668 s | 0.02085 | 5.42x |
| real persistence baseline | 0.03105 s | 0.003882 | 29.13x |

baseline 短窗线性外推为 `70.84 s/50 years`。该数字说明轻量日算子有足够计算预算，
但不代表能达到相同速度的科学日尺度 operator 已经存在。

分项时间：Teacher target capture `113.84 s`；forcing 准备 `0.0643 s`；host-to-device
`0.00082 s`；baseline 编译 `3.26 s`；replay 编译 `4.40 s`；计时内磁盘 IO 为零。

这三条计时路径不是严格的相同边界 A/B：dynamic Teacher replay 运输并重建每日
Teacher boundary，persistence baseline 在 executable 内从当前 state/forcing 构造
边界，而且原样持久的 state 可能被 XLA dead-code elimination。因而：

- `5.42x` 是当前 dynamic replay 实现的端到端测量，不是该架构的绝对速度上限；
- `29.13x` 是当前 persistence 实现的低成本参考，不是可学习 operator 的预计速度；
- 二者的差异说明需要一个同 PyTree、同 retained-tail、动态 synthetic operator 的
  cost pilot，不能解释为科学模型本身的速度优势。

## 自由 Rollout

最终 complete state schema 不相等，baseline 缺少 Teacher 日末
`hydrol_previous_step_state.nroot`。这是 fixed-spec/adapter schema 缺陷，必须修复，
但不是方法不可行性或神经网络不可学习的证据。

8 日物理状态最大 field-relative-L2：

| 状态族 | 最大相对误差 |
| --- | ---: |
| DIFFUCO | 121.76 |
| ENERBIL | 2.81 |
| THERMOSOIL | 2.81 |
| HYDROL | schema 缺失 `nroot` |

关键 daily interface 在第 8 日的相对误差包括：`soilhum_daily=97.3%`、
`resp_maint_part=96.1%`、`resp_maint_radia=88.8%`、`wspeed_daily=270%`。

关键 OK_LEAK 状态在第 8 日的相对误差包括：`DOC=67.8%`、
`interception_storage=60.0%`、`deepC_peat=59.4%`、`carbon_32l=58.0%`、
`dead_leaves=50.3%`。

8 日累计 forcing precipitation 为 `1.8415`，而 baseline water-storage proxy 从
`2.813375` 精确保持不变；Teacher 降至 `2.401870`。因此 baseline 没有真实水文
forcing response。选定库存没有 NaN、Inf 或负值，但“数值合法”不能替代过程响应。

这里的水文无响应、DOC/`deepC_peat`/`carbon_32l` 漂移是 persistence 定义的直接
后果：`_baseline_boundary` 原样携带完整 SECHIBA state，OK_LEAK pools 也原样携带。
这些数值如实证明 persistence 不可用，但不能外推出非持久性日尺度 operator 不可学。

PFT14 modelout 到第 8 日仍看似温和：AGB/BGB 误差约 `0.05%`，GPP `15.6%`，NPP
`8.5%`。这证明只看 AGB/BGB/GPP/NPP 会掩盖内部水、能量和 OK_LEAK 状态崩坏。

## 复核判定

已证实：

1. persistence baseline 很快，但在 8 日内科学失败；
2. dynamic Teacher replay 数值精确，但包含 boundary 运输/重建成本；
3. 两条路径不是严格的同边界性能 A/B；
4. 当前 adapter 丢失 `hydrol.nroot`。

尚未证实：日尺度 operator 是否可学、一个小模型的真实 compiled cost、自由 rollout
能否稳定，以及 residual 神经网络是否值得开发。

因此结论修正为：`persistence_baseline_failed / neural_learnability_inconclusive`。
当前没有可靠显式物理 baseline，若继续，模型需要学习 fast-day operator 的大部分
动态，而不是在可靠基线上做小 residual。这显著提高风险，但不构成不可行性证明。

当前不批准大规模训练。一个可供另行决策、但本轮不执行的最小下一实验是：使用极小
Teacher 样本做 one-step supervised learnability 检查，同时用相同输出 PyTree 的
动态 synthetic operator 替换学习器，测量同 retained-tail、同 executable 复用下的
真实 cost。必须先修复 `nroot` adapter；pilot 只做 one-step holdout 和短 7 日自由
rollout，预先限制样本量、模型规模和 CPU 时间。只有 accuracy 与 `>=3x` 同时通过，
才讨论 residual/learned operator；否则停止。

机器可读证据：
`outputs/performance/daily_coarse_graining/persistence_baseline_go_no_go.json`。
