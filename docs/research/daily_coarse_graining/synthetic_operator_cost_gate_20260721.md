# Synthetic Daily Operator Cost Gate

状态：**synthetic_cost_gate_passed / neural_learnability_inconclusive**。

本实验只回答一个问题：具备现实 encoder/MLP 形状、动态参数和完整 coarse boundary
输出的日尺度算子，连同 retained Teacher tail，是否仍有稳定 `>=3x` 的端到端加速
空间。它不训练模型，不评估精度，也不证明日尺度 operator 可学。

## 实现边界

- Teacher 固定为 commit `7333b46`；没有修改 `jax_orchidee/` 科学代码或默认路径。
- 输入仅为真实动态 day-start state、当天 48-step forcing 和动态参数数组；没有
  Teacher target、每日纠偏或固定答案输入。
- 算子含 14x64 forcing encoder、16x64 state encoder、
  `192x128 -> 128x128 -> 128x64` MLP 和 64x251 输出头，共 67,835 参数。
- 输出覆盖 6 个 fast-day state component、17 个 daily/maintenance 字段、13 个
  common OK_LEAK 字段、`deepC_peat` 和末步 `t2mdiag/temp_sol`。
- 输出进入 retained season/STOMATE daily carbon/modelout/day-end writeback；未被 tail
  最终保留的叶子进入一个显式有限化 validation reduction。最终 state、modelout 和
  8 个逐日 checksum 都执行 `block_until_ready`，避免无消费者输出被消除。

## `hydrol.nroot` 生命周期修复

年初 restart rebase 仍按 Teacher 语义删除 `nroot`。研究 adapter 不填默认值：Day 1
synthetic operator 从动态 `hydrol.us`、forcing encoder 和动态参数产生形状
`[1,14,11]` 的有限 `nroot`，严格验证后把 producer 输出提升为 Days 2--8 的 fixed
runtime spec。该字段进入 retained day-end packet 和下一日 state。

Teacher owner 是 `_add_hydrol_to_previous_fields`；下一 HYDROL transition 通过
`run_hydrol_first_step_module_from_precall(nroot_state=hydrol_state.nroot)` 消费。定向
测试同时保留“rebase 后缺失”和“producer 后存在并精确 roundtrip”两个阶段。

## 命令与环境

```powershell
conda run -n ORCJAX python -m research.daily_coarse_graining.synthetic_operator_cost `
  --state-cache outputs/acceptance/compiled_1961_12point_wetdiaglong_fix/001.0-071.0/compiled_checkpoints/paper_driver_1961_year_end_state.pkl `
  --year 1962 --days 8 --repeats 5 `
  --output outputs/performance/daily_coarse_graining/synthetic_operator_cost_gate.json
```

Windows 10，Intel Family 6 Model 183，JAX `0.10.1`，单 `cpu:0`。实验基于分支
`research/daily-coarse-graining` 的 `e0cc618` 加本轮未提交研究改动执行。

## 结果

| 项目 | 结果 |
| --- | ---: |
| synthetic compile，Day 1 | 13.7067 s |
| synthetic compile，7-day block | 3.6369 s |
| synthetic compile，总计 | 17.3436 s |
| Teacher first call，compile + run，未拆分 | 155.5355 s |
| host preparation，热计时外 | 1.6163 s |
| host-to-device，热计时外 | 0.00575 s |
| timed disk IO | 0 s |
| Teacher hot median，8 日 | 0.88788 s |
| synthetic hot median，8 日 | 0.01804 s |
| synthetic 秒/日 | 0.002255 s |
| 中位 Teacher A/B 加速 | 49.21x |
| 5 次最小配对加速 | 47.33x |
| synthetic 50 年线性外推 | 41.16 s |
| Teacher 50 年线性外推 | 2025.47 s |

synthetic 热运行 5 次为 `0.01678, 0.01804, 0.01600, 0.01839, 0.01907 s`；
Teacher 为 `0.88585, 0.89154, 0.84176, 0.88788, 0.90228 s`。8 个 blocked checksum
均有限并逐日变化，范围 `73.4627--73.8356`；runtime `nroot` 有限。

首次编译期间 XLA 两次报告 algebraic simplifier circular-loop warning，但编译正常
结束，5 次热运行均正常完成。该警告属于编译成本/稳定性限制，不能忽略，但不改变
本次热运行成本门禁。

## 判定

最保守稳定加速 `47.33x >= 3x`，所以**成本门禁通过**。正确结论仅是：可以另行
提议一个极小 supervised learnability pilot。`neural_learnability` 仍为
`inconclusive_not_tested`；不得据此开始大规模训练、生成正式数据集或使用 GPU/服务器。

机器证据：
`outputs/performance/daily_coarse_graining/synthetic_operator_cost_gate.json`。
