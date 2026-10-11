# T7 合批数值：本地准备结果

2026-10-11。**主管已接受本地验证方法和后续准备；真实操作数/部署仍待完成。**
本轮只完成 bounded source audit、独立小算子判别方法、输入 admission 与后续计划。
原模型首个分歧、GPU contraction 因果和原集成 T7 导数参考均未建立；
whole-prefix scalar mapping 不作为当前推荐修复。接受不表示原模型数值修复、
GPU readiness、完整训练提速或长期科学验收，后续操作仍需主管另行决定。

## 源码与输入结论

固定 cwd 为 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining`，
branch 为 `research/daily-coarse-graining`，本轮起始 HEAD 为
`7463642eb3ff20f2352bc5732e9904b5181f11ff`；ORCJAX Python 和
`jax_orchidee` import 指向此 checkout。此 SHA 保留为本轮起始身份，不随版本化收尾
提交更新。当地 CPU JAX 是 0.10.1，历史 deployed GPU
JAX 是 0.4.38，二者的验证效力分开。图只覆盖 main，既有 generation 为
`2026-10-10T08:46:50Z`；复核仍显示 daily subtree excluded/not_tracked，采用精确源码 fallback。

[input audit](../../../outputs/T7-batch-numerics-20261011/input-audit.json)
限定到以下函数，不宣称全网络或所有 branch 排除：

| 源码边界 | 本轮结论 |
| --- | --- |
| `successor_training.py::rollout_input_from_template:161–185` | 验证单样本 state `(3854,)`，刷新 compact PFT views；重建发生在 scalar vmap body 内 |
| `input_assembly.py::assemble:149–208` | FP64 物理视图重建、固定 PFT selection、单样本 flatten；所查函数没有显式 sample-axis reduction |
| `input_conditioning.py::standardize_continuous:12–57` | elementwise；保留 FP32 legacy 与选定 FP64 tail 后 cast 的原公式 |
| `strong_baseline_architecture.py::_masked_summary:957–973` | pool coordinate reduction，variance 使用中心化平方偏差，非 `E[x²]−E[x]²` |
| `_layer_norm:890–894`、`_stable_head_l2_normalize:901–910` | 分别归约最后 feature/head axis；stable L2 实数公式含 `epsilon²*max(abs(x))²` 与 zero-output threshold |

11 项输入检查中，10 项 serial/vmap primal、JVP、实际 JVP/VJP pairing 与 sample
isolation 全通过；第 11 项 defined-boundary 通过。6 个独立 FP64 smooth oracle 的
FD 在两个最小步长均通过。精确 extrema tie/zero variance 的 selected AD tangent
不等于普通单侧导数；不能用 kink FD 指认 AD 错误。

四条真实 artifact299 scalar raw/mean/scale 元数据的 primal 与 cast-scale analytic
JVP 检查通过；434/2382/2621/200 的 raw-coordinate JVP 分别约为
5,874,869 / 2,260,285.5 / 161,124.703125 / 2,849.056396。所选四坐标自身不是
初始 extrema tie。完整原 state、normalization/pool 数组、实际 head 激活均未重建，
这些结果不能归因原 GPU 故障。

保留一个既有 synthetic near-constant LayerNorm sensitivity witness：serial/vmap
仍一致，但 FP64 oracle **64 primal / 15 JVP 坐标失败**；FP32/FP64 mean 分别为
1.0000298023223877 / 1.0000300407409668。测试通过表示失败被诚实保留，
不表示这个 witness 或真实模型已满足 accuracy gate；没有扩展合成例子搜索。

## 小 matrix 判别方法与保存门槛

[probe](../../../scripts/hpc/probe_shared_latent_T7_batch_numerics.py) 仅调用未改动的
生产 `_linear`；默认 synthetic `(4,128,64) @ (64,64)`。captured bundle 接受 bounded
`(4,N,K) @ (K,M)`，可用于早期 tokenizer 4→384。exact seven-key NPZ、finite FP32
operands、Boolean masks、sample3-only 非零 direction、≤2M elements 和 captured
identities 的合同见 [GPU plan](T7_batch_numerics_GPU_confirmation_plan_20261011.md)。
metadata 的 source identities 是调用者声明，本 probe 不独立重算完整来源文件 SHA。

四次 B1 scalar 调用加 device merge、B4 vmap、B4 lax.map 共三个主图加一个 merge
图，**总计四个小 compile**。独立 NumPy FP64 analytic primal/input JVP、全部小
kernel+bias/input gradients、central FD 与实际 `<Jv,c>`/`<v,Jᵀc>` 配对共同验收；
两端分别对 oracle。保留 raw outputs 和 masks；没有参数 JVP、整网 AD、物理、
rollout 或 optimizer。B1 是比较 route，不是真值。

最新 [CPU v3 receipt](../../../outputs/T7-batch-numerics-20261011/affine-cpu.v3.json)
通过：三 route untimed baseline 和三 warm cycles 共12测量，12 oracle/12 isolation、
8 cross-route、9 within-route repeats 均通过；全部结果加独立 oracle 共
[91 arrays](../../../outputs/T7-batch-numerics-20261011/affine-cpu.v3.results.npz)。
每条 route 的 optimized HLO 分别保存为 `.v3.b1/vmap/map.optimized-hlo.txt.gz`。
首版 final-only receipt 和 v2 均保留，以 v3 为本轮最终 method evidence。

最终补充拒绝检查污染中间轮的 raw primal、JVP、input/kernel gradients 和 loss，
包括 inactive raw 坐标：NPZ 保存原 NaN/Inf，逐轮 oracle/cross/repeat/AD pairing
继续失败，后来通过不覆盖失败。JSON 明确 actual/reference/derived nonfinite
计数，无法表示的 max/dot 标量为 `null`，严格 `allow_nan=False` 可保存；
状态 `numerical_failed`，CLI 退出码3。finite subtraction overflow 同样拒绝，未豁免 gate。

## 真实资产缺口与最小获取边界

[literal availability record](../../../outputs/T7-batch-numerics-20261011/asset-availability.json)
只对六条已声明 local paths 做 Test-Path；全部缺失，未做数据扫描、bulk hash、复制
或 remote 检查。以下 relative paths 基于 `E:/ORCHIDEE-MAN-JAX/runtime/`：

| 资产 | relative path / expected identity |
| --- | --- |
| 参数 | `outputs/diagnostics/shared-latent-reader-20261004-T7-assembled-input.parameters.pkl`；SHA `4f95d18d97d2feb4dcab8e1dda6caeb242a4869f044a00b6a7ae6bc689806c14` |
| context | `outputs/diagnostics/shared-expert-real-daily-context-c599236.pkl`；SHA `ff3fdabe755d599bc2ceee72087c03983d9a629add93166519dcda1f56be9026` |
| TRAIN299 artifact | `outputs/training/target-oriented-prepared-1000-a243029/chunks/251-438/update-0299.pkl.gz`；SHA `ee7fa6e0f537fcd75dc8d02bbf4aa4f0438119f1ca73368e8f7f9d64af1f18aa` |
| prepared manifest | `outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json`；dataset/schedule/source/contract bindings 全值见 JSON，非 whole-file SHA |
| dataset manifest | `outputs/training/pft14-daily-teacher-v6/dataset_manifest.json`；prepared binding `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99` |
| 原 T7 manifest | `transfer/shared-latent-reader-20261004/latent-reader-T7-20261005.manifest.json`；SHA `252d610a8df4b78963111244edf7ef3414471883bdfe126f436275beaef57bc4` |

远端对应路径仅记录，availability 为 `not_checked`。742/275/482 artifact paths
须由原 prepared manifest 解析，不猜 shard。sample299 原 day0 FP64 `(3854,)`
state array SHA 是 `54d01d0f376b0ceed8a843740328dba9d61d2a8989507436bda1a0d16c624f5d`；
本地 saved state derivative 和另一个 biomass initial state 均不能替代。
已有 raw24 pickles/HLO/metadata 不含真实内部 matrix operands。

最小下一项准备是 **尚未实现的 CPU thin extractor**：从原 manifest/context、参数
和四窗口 day0 state 取得 exact normalization/layout/PFT/pool masks，核验
`parameters["tokenizer"]["summary_input"]` leaf；仅复用 `standardize_continuous →
_mask_inactive_pft_elements → _masked_summary`，保存 x、原 kernel/bias、一个真实
direction 的 CPU input JVP、cotangent/mask 与 capture receipt/source identities。
优先434；不构造 shared backbone、reader64 激活或整 prefix。资产不可得即保留缺口。

## 验证与交付状态

实际轻量命令（cwd 为固定 daily worktree）：

```text
conda run -n ORCJAX python -B scripts/hpc/audit_shared_latent_T7_input_numerics.py
conda run -n ORCJAX python -B -m pytest tests/unit/test_shared_latent_T7_input_numerics.py -q -p no:cacheprovider
conda run -n ORCJAX python -B scripts/hpc/probe_shared_latent_T7_batch_numerics.py --backend cpu --output outputs/T7-batch-numerics-20261011/affine-cpu.v3.json --operands outputs/T7-batch-numerics-20261011/affine-cpu.operands.npz --repeats 3
conda run -n ORCJAX python -B -m pytest tests/unit/test_shared_latent_T7_batch_numerics.py -q -p no:cacheprovider
```

Input tests **6 passed /16.63s**，matrix tests 最终 **6 passed /3.80s**；两者只有
禁用 cacheprovider 后已有 `cache_dir` config warning。本机 optional historical test
确实执行通过；普通 synthetic fixture 使用不存在的 tmp asset directory，干净
checkout 上只有明确的 external-history test 会按缺失资产 skip，缺失不被计作通过。

[GPU plan](T7_batch_numerics_GPU_confirmation_plan_20261011.md) 给出一次未来 bounded
free-V100 component 判别、outcome→repair/stop 决定，以及另行决定的原66,123,713
参数候选 T7 public objective/native full AD 回归。真实 capture、deadline/NVML wrapper、
deployed PJRT bootstrap 与目标 shell dry-run 尚未完成；当前命令不称 GPU ready。
严格 precision/tolerance、24GiB、完整 masks/参数/state AD、roughness state20 NaN
admission 和 daily32+T7+T30 AdamW 最终工作量保持。本轮没有 remote/GPU、训练、
Teacher 生成、生产 patch、清理或 merge。主管已审阅并接受本地准备，现有12项
针对性测试与 CPU v3 证据足够；本次状态文案/版本化收尾复用这些证据，不重跑模型。
按 `docs/WORKSPACE_MANAGEMENT.md`，仅获准将8个已审阅任务文件在现有 research
分支提交并正常 push；不包含 outputs/资产，不 merge main，不开启新研究或服务器工作。
