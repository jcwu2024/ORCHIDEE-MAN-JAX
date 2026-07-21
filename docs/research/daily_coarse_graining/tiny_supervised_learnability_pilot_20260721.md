# Tiny Supervised Daily-Boundary Learnability Pilot

结论：**tiny_overfit_passed_holdout_failed**。

本实验从 `d0554d0` 开始，只测试完整 fast-day coarse boundary 的极小监督可学习性。
Teacher target 只作标签；输入只有真实 day-start state 和当天 48-step forcing。没有
Teacher 日末状态输入、rollout 纠偏、GPU、服务器或大规模数据集。

## 数据与模型

- 训练：1962 Day 1--12；holdout：Day 13--16。归一化统计只来自训练日。
- 未执行 7 日 rollout，因为 one-step holdout 未通过。
- continuous schema：250 leaves、19,838 elements，覆盖六个 fast-day state
  component、17 daily/maintenance、14 OK_LEAK（含 `deepC_peat`）和
  `t2mdiag/temp_sol`。
- 离散/static/provenance 不学习；carry owner 规则在训练和 holdout 分别 exact 检查
  36 和 12 个元素，零 mismatch。`nroot` 保持 Day 1 producer 后进入 runtime spec。
- forcing encoder 是保留 48 步顺序的 32-state recurrent encoder；state 使用逐元素
  train-only 标准化和有序 64-element chunk encoder；融合 latent 为 128。
- decoder 按过程族设置独立 dense head，对每个 continuous element 输出 residual，
  不是每字段广播系数。总参数 2,624,286，其中监督拟合 decoder 参数 2,559,102。
- 为使极小 overfit 门禁确定且可诊断，decoder 使用训练日 SVD pseudoinverse 拟合；
  encoder 固定初始化。该设计测试表示容量和最小 one-step 泛化，不代表正式训练方案。

## 顺序门禁

训练日 overfit normalized RMSE 为 `2.132e-15`，低于 `1e-6` 阈值；逐族均接近机器
精度，说明 schema 和逐元素 decoder 能表示目标。

Day 13--16 holdout 结果：

| 过程族 | normalized RMSE | 最大绝对误差 |
| --- | ---: | ---: |
| overall | 1.5164 | 83,467.6 |
| DRIVER | 1.9199 | 21,226.8 |
| DIFFUCO/ENERBIL | 1.4604 | 6,813.7 |
| HYDROL | 0.8458 | 25.586 |
| THERMOSOIL | 1.7486 | 83,467.6 |
| SECHIBA finalize | 1.3268 | 6,813.7 |
| 17-field daily interface | 9.4512 | 26.288 |
| OK_LEAK | 0.5003 | 0.1125 |
| final diagnostics | 1.9155 | 26.884 |

预注册阈值是 overall `<=1.0` 且每族 `<=2.0`。总体和 daily interface 失败，因此
不进入 rollout。

关键字段包括：`hydrol.mc/mcl/soil_mc` normalized RMSE `0.871`，`nroot` `2.416`，
`thermosoil.ptn/tdeep` `2.152`，`daily.gpp_daily` `0.217`，
`resp_maint_part` `0.801`，但 `soilhum_daily` 达 `45.41`。AGB/BGB 的慢态 biomass
在该边界按 owner exact carry；GPP/NPP 的前置 daily/maintenance 驱动如上报告，不能
用最终 modelout 掩盖内部失败。

四个 holdout 日的 HYDROL 和 carried slow-state 选定库存均无 NaN、Inf 或负库存。
水库存 proxy 最大相对误差 `8.28%`，OK_LEAK 碳库存总量 proxy 最大相对误差
`0.768%`；后者没有替代逐 pool 合法性门禁。这些只是诊断 proxy，不是冻结的单位
闭合预算。

## Trained Runtime Cost

结构改变后重新编译了真实 2.62M 参数 decoder + retained Teacher tail 的一日
executable。compile `1.493 s`，3 次 hot run 为 `0.004189, 0.004078, 0.004689 s`，
中位 `0.004189 s/day`；blocked checksum `3642.009` 且有限。该复测只用于确认新
decoder 成本，没有做 7 日 rollout，也没有复用或引用 `d0554d0` 的 `49x` 作为新
结构速度。由于没有同层级多日 Teacher A/B，本结果不声明新 speedup。

Teacher capture 用时：训练 12 日 `111.41 s`，holdout 4 日 `1.03 s`；SVD fit
`0.313 s`。首次 Teacher 编译仍出现已知 XLA algebraic-simplifier warning，但正常完成。

机器证据：

- `outputs/performance/daily_coarse_graining/tiny_supervised_learnability_pilot.json`
- `outputs/performance/daily_coarse_graining/tiny_supervised_sample_manifest.json`

## 判定

当前表示具备训练集记忆容量，但没有最小跨日期 one-step 技能。最终结论只能是
`tiny_overfit_passed_holdout_failed`。本轮在此停止，不增加训练日、不调大网络、
不执行 rollout。
