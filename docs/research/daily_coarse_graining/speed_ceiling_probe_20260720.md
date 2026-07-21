# Pre-Daily-STOMATE Replay Speed-Ceiling Probe

状态：Gate 1 canonical-minimal 边界探针已完成，Teacher commit `7333b46`，
2026-07-21 更新。

Replay 不是日尺度模型。它使用已经由 Teacher 计算出的正确边界值，不预测未知日期；
作用是测量“假设一个日尺度模型能以很低成本提供这些值”时的端到端速度上限。

## 实现

研究侧实现位于：

```text
research/daily_coarse_graining/replay_ceiling.py
```

它不修改 `jax_orchidee/`。Teacher capture 通过临时运行时 hook 记录以下现有 owner
返回值：

- 48 步 SECHIBA transition；
- daily accumulator/maintenance fold；
- OK_LEAK pre-step boundary；
- maintenance respiration stack；
- 48 步 OK_LEAK fold 结果和状态更新。

replay 在相同位置注入记录值，并让原始 season、STOMATE 日碳、modelout、日末写回
和跨日 carry 继续执行。hook 只在单次函数调用的上下文中存在，不改变 Teacher 默认
runner。

同等编译层级模式进一步保留 Day 1 在 block 外，并将 Day 2 起固定 7 天的 replay
tail 放入一个 JIT executable。固定边界模式会把 Teacher 值编译成常量，只能作为
绝对乐观上限。动态模式把每日边界作为 `[7, ...]` 数组输入，并让所有 7 日 block
复用同一个 executable。

canonical-minimal 动态模式不再传输完整 Teacher fold/OK_LEAK 对象图、预先构造的
OK_LEAK boundary 或重复的 13-field update dict。它只传输：

- 日末 SECHIBA fixed-spec state packet；
- 最后一步 `t2mdiag`、`temp_sol` 两个不可由当前 carry 无歧义恢复的诊断量；
- 17 个 daily/maintenance 字段；
- 13 个通用 OK_LEAK carry 字段，以及 active `PERMA_PEAT` 路径的 `deepC_peat`。

Teacher tail 从当前 state/context 重建 pre-step boundary、空成功 fold scaffold 和
13-field update dict。`resp_hetero_soil` 仅用于生成 reset 零数组，其 shape 由已有
`slowproc.resp_hetero` carry 提供，不作为 coarse 输出。

## 命令

单日 smoke：

```powershell
conda run -n ORCJAX python -m research.daily_coarse_graining.replay_ceiling `
  --state-cache outputs/acceptance/compiled_1961_12point_wetdiaglong_fix/001.0-071.0/compiled_checkpoints/paper_driver_1961_year_end_state.pkl `
  --year 1962 --days 1 --repeats 1 --compiled-day-block-size 0 `
  --output outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_1day_smoke.json
```

30 日生产 block 对照：

```powershell
conda run -n ORCJAX python -m research.daily_coarse_graining.replay_ceiling `
  --state-cache outputs/acceptance/compiled_1961_12point_wetdiaglong_fix/001.0-071.0/compiled_checkpoints/paper_driver_1961_year_end_state.pkl `
  --year 1962 --days 30 --repeats 3 --compiled-day-block-size 7 `
  --output outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_30day_block7.json
```

30 日同等编译层级上限：

```powershell
conda run -n ORCJAX python -m research.daily_coarse_graining.replay_ceiling `
  --state-cache outputs/acceptance/compiled_1961_12point_wetdiaglong_fix/001.0-071.0/compiled_checkpoints/paper_driver_1961_year_end_state.pkl `
  --year 1962 --days 30 --repeats 3 `
  --compiled-day-block-size 7 --compiled-replay-block-size 7 `
  --output outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_30day_compiled_block7.json
```

30 日 canonical-minimal 动态边界：

```powershell
conda run -n ORCJAX python -m research.daily_coarse_graining.replay_ceiling `
  --state-cache outputs/acceptance/compiled_1961_12point_wetdiaglong_fix/001.0-071.0/compiled_checkpoints/paper_driver_1961_year_end_state.pkl `
  --year 1962 --days 30 --repeats 3 `
  --compiled-day-block-size 7 --compiled-replay-block-size 7 `
  --dynamic-replay-boundary-inputs `
  --output outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_30day_canonical_dynamic_block7.json
```

## 数值结果

逐日 Python replay 满足：

- complete day-end state schema 相同；
- 350 个 day-end state 叶子最大绝对误差为 `0`；
- 26 个 modelout fields 最大绝对误差为 `0`；
- 4 个最终 modelout 最大绝对误差为 `0`；
- 离散字段精确相等。

compiled replay 也通过相同的 schema、离散字段和浮点容差门禁；由于跨日 JIT 改变
浮点运算编排，不再要求位相等，其最大误差见下文。因此当前 hook 注入位置对已测
1962 Day 1--30 是数值一致的。

## 性能结果

| 实验 | Teacher 中位数 | Replay 中位数 | Speedup |
| --- | ---: | ---: | ---: |
| 1 日，逐日 compiled Teacher | 0.449 s | 0.111 s | 4.04x |
| 3 日，逐日 compiled Teacher | 1.717 s | 0.392 s | 4.38x |
| 30 日，生产 7 日 block Teacher | 3.584 s | 2.482 s | 1.44x |
| 8 日，Teacher/replay 均为 7 日 block | 0.834 s | 0.0838 s | 9.95x |
| 30 日，Teacher/replay 均为 7 日 block | 2.755 s | 0.194 s | 14.18x |
| 8 日，动态边界、单 7 日 executable | 0.812 s | 0.177 s | 4.59x |
| 30 日，动态边界、单 7 日 executable 复用 4 次 | 2.669 s | 0.556 s | 4.80x |
| 30 日，canonical-minimal 动态边界、单 executable | 3.173 s | 0.559 s | 5.67x |

30 日逐次计时：

```text
Teacher: 4.192, 3.584, 3.333 s
Replay:  2.482, 2.539, 2.351 s
```

Teacher target capture 用时 `199.3 s`，生产 7 日 block 的首次 warm/compile 用时
`91.9 s`；两者均排除在热运行 speedup 之外。运行期间 XLA 两次报告 algebraic
simplifier 达到 50 次循环上限，但运行完成且数值门禁通过。

同层级 30 日实验的 Teacher capture 为 `127.0 s`，四个固定 replay block 的总编译
时间为 `31.5 s`，Teacher warm/compile 为 `72.7 s`。热运行逐次结果为：

```text
Teacher:        2.755, 2.731, 2.764 s
Compiled replay: 0.194, 0.193, 0.194 s
```

最终 day-end state 最大绝对差为 `7.28e-12`、最大相对差为 `1.38e-14`；每日
modelout fields 最大绝对差为 `7.11e-15`，最终 modelout 最大绝对差为 `2.22e-16`。

动态边界实验将所有日变化数值叶子作为 `[7, ...]` JAX 输入，并让四个 7 日 block
复用同一个 executable。30 日 replay 编译时间为 `5.38 s`，热运行结果为：

```text
Teacher:       2.666, 2.669, 2.688 s
Dynamic replay: 0.556, 0.556, 0.583 s
```

最终 day-end state 最大绝对差仍为 `7.28e-12`；每日 modelout fields 最大绝对差为
`2.13e-14`，最终 modelout 最大绝对差为 `1.78e-15`。

canonical-minimal 30 日实验的边界为每日期 `343` 个动态叶子、`266426` bytes：

| 边界族 | 动态叶子 | bytes/day |
| --- | ---: | ---: |
| 日末 SECHIBA fixed-spec packet | 310 | 187306 |
| 最后一步诊断量 | 2 | 16 |
| daily/maintenance interface | 17 | 2048 |
| OK_LEAK interface（含 `deepC_peat`） | 14 | 77056 |

编译时间为 `5.02 s`，Teacher target capture 为 `121.1 s`，两者不计入热运行。
热运行逐次结果为：

```text
Teacher:          3.173, 3.829, 3.057 s
Canonical replay: 0.571, 0.515, 0.559 s
```

最终 350-leaf day-end state 最大绝对/相对误差分别为 `7.28e-12` 和
`1.38e-14`；每日 modelout fields 最大绝对差为 `2.13e-14`，最终 modelout 最大
绝对差为 `1.78e-15`。所有 schema、离散字段和浮点门禁通过。

## 解释

逐日 Teacher 对照显示 fast-day 本身确实昂贵，但这不是当前生产基线。生产默认的
7 日 complete-day block 已经把 SECHIBA、daily fold、OK_LEAK、STOMATE 和写回放进
跨日 compiled scan，大幅降低 Python 日循环成本。当前 replay 仍逐日执行保留尾部，
所以相对生产 Teacher 只有 `1.44x`。

这些结果给出三个结论：

1. pre-daily-STOMATE 边界可以在不修改 Teacher 的情况下准确注入；
2. 仅实现日尺度近似而继续使用当前逐日 Python tail，不会产生数量级端到端加速。
3. 将保留的 STOMATE tail 放入同等 7 日 compiled block 是必要条件；固定边界常量
   可达到 `14.18x`，但不能作为性能生死门结论。

固定边界的 `0.00647 s/day` 线性外推约为 2 分钟/50 年，但它把不同日期的 Teacher
边界作为常量，并为 30 天编译了四个 executable；真实 coarse model 必须提供动态
输出并跨 block 复用 executable，因此不能据此宣布通过继续标准。

动态边界和单 executable 复用实验把该 replay 实现的测量从常量折叠的 `14.18x`
修正到约 `4.8--5.7x`。canonical-minimal 的 replay 时间为 `0.01865 s/day`，与完整动态
边界实验的 `0.01854 s/day` 实质相同；短窗线性外推约为 5.7 分钟/50 年，而且仍未
计入未来 coarse operator 的推理成本。Teacher 计时存在 run-to-run 波动，因此不能
把本轮 `5.67x` 与上轮 `4.80x` 的差异解释为接口精简带来的确定加速。因此：

- `14.18x` 只能说明固定 Teacher 边界的常量折叠效果；
- `4.8--5.7x` 是当前 dynamic Teacher replay 的端到端测量，不是任意日尺度 operator
  的架构绝对上限；
- 日尺度路线有明显加速空间，但尚未满足数量级改善或 1--2 分钟/50 年目标。

后续 persistence baseline 达到 `29.13x`，说明它与 replay 没有执行相同的 boundary
运输/重建路径，且持久状态可能被 XLA 消除。两者不能作为直接 A/B；未来 cost pilot
必须使用相同输出 PyTree、动态 operator 输出和同一 retained tail。

接口对象图精简没有改变 retained tail 的主要成本，说明继续做 Python/JAX 容器级
微优化价值有限。下一步仍不应直接开发神经网络，而应先实现无神经日尺度 baseline，
并同步补齐逐字段 canonical-owner 规则和水/能量/碳预算 ledger。当前 310-leaf state
transport 是 fixed-spec packet envelope，不等于已经冻结的 learnable schema；在
duplicate/finalize mirror 重建规则完成前，不得据此生成正式训练集。

## 证据

- `outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_1day_smoke.json`
- `outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_3day.json`
- `outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_30day_block7.json`
- `outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_8day_compiled_block7.json`
- `outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_30day_compiled_block7.json`
- `outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_8day_dynamic_block7.json`
- `outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_30day_dynamic_block7.json`
- `outputs/performance/daily_coarse_graining/pre_daily_stomate_replay_30day_canonical_dynamic_block7.json`
- `tests/unit/test_daily_coarse_replay_ceiling.py`
