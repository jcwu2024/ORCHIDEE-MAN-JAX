# Parameter-Conditioned Gradient Training Readiness Gate

结论：`training_pipeline_blocked_local_overfit`。

本轮只整理服务器前架构并验证真实梯度链路。Teacher 固定为 `7333b46`，未修改
`jax_orchidee/` 科学代码；实验只使用 Windows CPU、16 个连续 Teacher 日和既有
`ORCJAX` 环境，没有服务器、GPU、rollout 或大规模数据生成。

## 四参数 owner/consumer

| 网络输入 | Teacher 静态 owner | fast-day 直接影响 | retained tail consumer | 下一日动态路径 | base / 参考点 / 论文样本范围 |
| --- | --- | --- | --- | --- | --- |
| `VCMAX25` | `pft_parameters.f90:3234-3240` | 否 | `stomate_vmax::vmax:327-354` | 写出动态 `vcmax`，经 `assim_param` 进入下一日 DIFFUCO | `50 / 63.2061836 / [20.0180641, 70]` |
| `MAINT_RESP_SLOPE_C` (`S1`) | `pft_parameters.f90:4220-4226` | 是，半小时 maintenance | tail 只消费已累计的 `resp_maint_part` | maintenance 改变 retained carbon state | `0.05 / 0.0876862 / [1.1e-6, 0.3]` |
| `ALLOC_MIN` (`fa_min`) | `pft_parameters.f90:4532-4538` | 否 | `stomate_alloc::alloc:623-627` | allocation 改变 biomass/leaf state | `0.3 / 0.2019211 / [0.1000685, 0.4]` |
| `RESIDENCE_TIME` (`tau`) | `pft_parameters.f90:4163-4169` | 否 | `lpj_gap::gap:226-234` constant-mortality arm | active 时改变 biomass/litter carry | `80 / 50.6583452 / [5.0023366, 100]` |

JAX 对应 consumer 分别为 `carbon_kernels.vmax_step`、
`_maintenance_respiration_core`、`allocation_step` 和 `gap_mortality_step`。完整机器 ledger
见 `manifests/coarse_graining/parameter_conditioned_training_v1.json`。

`VCMAX25` 是 run.def 控制的静态 PFT 参数；`vcmax/assim_param` 是 retained STOMATE
根据叶龄等动态状态生成的过程状态。新输入契约同时保留两者，不能用静态参数替代
日初动态状态。

## 输入与输出契约

`ConditionedDayInput` 是动态 JAX PyTree，包含：

- 完整日初 state vector；
- 保留顺序的 48-step forcing，实际 shape 为 `[48, 597]`；
- 四参数动态数组，共 56 个元素；
- 103 个 soil/hydrology/PFT landpoint 物理元素；
- 4 个 calendar/season 元素。

契约不含 landpoint ID，也不读取论文 AGB/BGB/GPP/NPP 或 Teacher 日末 target。所有
normalization 只由 16 日 train split 计算。定向测试证明只改变参数数组会改变 JIT
模型输出，因此参数没有被当作静态常量或忽略。

完整 continuous boundary 原为 250 leaves / 19,838 elements。以下 6 个逐点 proven
exact 字段移出 learned target：`t2m_daily`、`precip_daily`、`snowfall_daily`、
`t2m_min_daily`、`t2m_max_daily`、最终 `t2mdiag`。它们来自
`enerbil_t2mdiag`、runtime entry payload 和 `stomate.daily` 的 source-order fold；16 日
对 Teacher 标签均为 bitwise exact，最大绝对误差为 0。learned target 因而是
244 leaves / 19,832 elements。其余无法证明 exact 的量全部保留为 learned target。

## 梯度训练结果

生产训练路径为纯 JAX：shared chunk state encoder、48-step recurrent forcing encoder、
parameter/static/calendar condition encoder、fusion MLP，以及按过程族分头的逐元素
residual decoder。旧 SVD pseudoinverse 仅保留为历史诊断，不参与本轮训练。

固定配置：seed `20260721`，float32，16 日 full batch，Adam 600 steps，learning rate
`0.003`，参数量 519,312。checkpoint 包含 model/optimizer PyTree、train-only
normalization、Teacher commit 和 boundary width；位置为
`outputs/checkpoints/daily_coarse_graining/gradient_training_ready/checkpoint.pkl`，不提交
Git。

- initial gradient norm：`0.0725313`；
- encoder / decoder 参数变化范数：`13.2571 / 83.8851`；
- loss：`0.502948 -> 0.00993169`；
- overall normalized RMSE：`0.0873787`；
- 预注册 overfit threshold：`<= 0.05`，未通过；
- Teacher capture / gradient training：`132.98 s / 30.56 s`；
- 参数 bytes / 近似 peak bytes：`2,077,248 / 18,462,976`；
- discrete/static carry：48 个元素 exact，0 mismatch。

过程族 RMSE 为 DRIVER `0.1104`、DIFFUCO/ENERBIL `0.1366`、HYDROL `0.1013`、
THERMOSOIL `0.1268`、SECHIBA finalize `0.1224`、daily interface `0.07264`、
OK_LEAK `0.01193`、final diagnostics `0.04337`。这证明真实梯度、更新、checkpoint
链路可运行，但不能把未达阈值写成 `training_pipeline_ready`。

## Linux/GPU 收束

CLI 与 config 不依赖 Windows 绝对路径；未来 Linux 环境使用 `pyproject.toml` 已有
`cuda12` extra。GPU smoke 命令模板：

```bash
python -m research.daily_coarse_graining.gradient_training_ready \
  --state-cache <paper_driver_1961_year_end_state.pkl> \
  --experiment-config research/daily_coarse_graining/configs/local_gradient_gate.json
```

当前客观 server gate 未打开，因为 16 日 overfit RMSE 未达到 `0.05`。本轮不增加
训练步数、不调参、不扩大数据、不上服务器。机器结果位于
`outputs/performance/daily_coarse_graining/gradient_training_ready_gate.json`，样本清单
位于同目录 `gradient_training_ready_samples.json`；二进制样本和 checkpoint 均不提交。
