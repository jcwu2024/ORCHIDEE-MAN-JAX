# T7 early summary_input：本地提取与验证

2026-10-11。主管已接受本包的本地提取方法、fixture 接入与失败保存，授权提交/推送
七个任务文件。**真实捕获尚未执行；这不是 CPU 未复现原问题的结果。**
参数、context 和 prepared 数据缺失，不以 synthetic 或已保存导数替代原 initial state。
没有选择生产修复，不排除 later64→64/attention 原因，不构成原 T7 或完整训练提速验收。

## 身份和有界资产定位

固定 cwd 为 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining`，branch 为
`research/daily-coarse-graining`，本轮起始 HEAD 为
`120c6378966b060f4da6c6763250e43f2d26e2e4`。ORCJAX 解释器实际为
`C:/Users/admin/.conda/envs/ORCJAX/python.exe`，`jax_orchidee` import 来自本 daily
checkout。历史执行仍是 `c599236bcf57b745e9bafed678298010f4c3b58f` 加 declared overlays，
不改写历史报告、manifest、source overlays、HLO、checkpoint 或数据身份。

[资产记录](../../../outputs/T7-summary-capture-20261011/asset-inventory-20261011T064831Z.json)
记录 root outputs、root runtime outputs 和 daily 复用路径中的15条精确路径，以及
10个相关目录内 depth0 exact-basename 查找；未递归遍历项目历史或 Teacher 数据。
只对原 manifest 的两条实际候选路径核对完整 SHA，未 bulk hash。
原 `latent-reader-T7-20261005.manifest.json` 在 root 与 daily 的
`outputs/diagnostics/shared-latent-reader-20261004/` 找到，两者 SHA 均为
`252d610a8df4b78963111244edf7ef3414471883bdfe126f436275beaef57bc4`。
之前六条 runtime 字面路径的缺失记录仍成立；它不是穷尽不存在证明。

最小未来获取清单：

| 资产 | 需要的身份 |
| --- | --- |
| 原 candidate parameters | `4f95d18d97d2feb4dcab8e1dda6caeb242a4869f044a00b6a7ae6bc689806c14` |
| 原 prepared context | `ff3fdabe755d599bc2ceee72087c03983d9a629add93166519dcda1f56be9026` |
| prepared manifest 与 canonical v6 dataset manifest metadata | dataset/schedule/source/contract bindings 全值见资产记录；manifest 内解析 artifacts，不猜 shard |
| TRAIN742/275/482/299 四个原 prepared artifacts | 路径与 SHA 从匹配 prepared manifest 解析；299 SHA 为 `ee7fa6e0f537fcd75dc8d02bbf4aa4f0438119f1ca73368e8f7f9d64af1f18aa` |

sample299 window0/day0 原 FP64 `(3854,)` state bytes SHA 必须是
`54d01d0f376b0ceed8a843740328dba9d61d2a8989507436bda1a0d16c624f5d`。其余 initial
states 来自各自原 prepared window0/day0。只需要既有 TRAIN 数据，禁止 held-out、
Teacher 再生成或原权重/context 重造。原 input_bundle/source_profile 的声明 basename
也未找到；薄层从身份匹配 parameters/context 内的 method 交叉核验，不构造完整 loss。

## 提取与证据边界

目标边界是 canonical rebuild/assembler 的 `inputs.state.other`、defined 和
active PFT mask，经 `standardize_continuous → _mask_inactive_pft_elements →
_masked_summary` 得到 `(4,Nfield,4)` 输入和 state-field validity；只对这段 preprocessing
做 raw coordinate434、sample3-only CPU JVP。投影参数须实际验证为保存的
`parameters["tokenizer"]["summary_input"]` kernel `(4,384)` 与 bias `(384,)`，保留
normalization、tail 配置、pool/PFT/定义掩码和 dtype。诊断 cotangent 明确标记，
不冒充原 T7 loss；Boolean output mask 来自真实 field validity。

主管已核对历史 c599236→当前起始 HEAD：input_conditioning、input_assembly、
successor_training、strong_baseline_architecture 四个 base helper 文件无差异；
shared_transformer 的差异是既有 latent-reader opt-in。原 raw prefix 的
`make_raw_prefix:51–59` 依次 rebuild→assemble→head.assemble_context。
这些对应关系不自动覆盖历史 overlays；新提取器/current helper source 与实际执行
SHA 单独记录，不能用当前 HEAD 替换旧实验 identity。

图工具本轮 Transport closed；此前 generation `2026-10-10T08:46:50Z` 对 daily
subtree 为 excluded/not_tracked，使用上列精确 source fallback，不重索引 main。

## 实现与验证记录

实现入口为 `scripts/hpc/capture_shared_latent_T7_summary_operands.py`。真实模式仅需
显式传入 parameters、context、T7 manifest、prepared manifest、dataset manifest
五条既有路径。先检查文件身份再反序列化；从原 method 和 context 交叉验证历史
head、overlays、窗口与 prepared identity，用既有 TRAIN provenance 验证器拒绝
held-out、越界和重叠窗口。直接读取 prepared `seven_day.inputs[0,0]`，不构造
targets/loss、完整 model 或 Teacher。捕获器不调用 neural head/reader/physical step。

`JAX_PLATFORMS=cpu` 在 JAX 导入前强制设置，不能被继承的 cuda 值覆盖。
subprocess 测试在继承 `JAX_PLATFORMS=cuda` 时只得到 CPU。
source receipt 除四个 base helper 外还记录 contract、资产验证器、既有 probe
及 `jax_orchidee/ad_primitives.py`，明确使用
`source_sqrt_with_finite_zero_tangent` 和注册的 `_finite_zero_sqrt_jvp`。
这些是本次执行来源，未宣称 current helper 自动等同全部历史 overlay。
当前 extractor 只生成 sample3/raw434 方向，没有其他方向选择接口；
2382/2621/control200 是以后另选的扩展，不属于本包实现或验证。

七键 NPZ 是 `x/kernel/bias/direction/cotangent/mask/metadata`。metadata 区分
`synthetic_method_fixture` 与真实已捕获来源；receipt 绑定六个数组的 shape、dtype、
bytes SHA 和实际 source。非法数值/方向隔离拒绝时保留全部原数组为
`.failed-arrays.npz`，并保存失败 receipt；输出 stem 不覆盖既有证据。

针对性检查：

```powershell
conda run -n ORCJAX python -B -m pytest tests/unit/test_shared_latent_T7_summary_operands.py -q -p no:cacheprovider
```

[最终测试日志](../../../outputs/T7-summary-capture-20261011/summary-tests.v3.stdout.txt)
为 **10 passed / 13.46s / exit0**；
[launcher receipt](../../../outputs/T7-summary-capture-20261011/summary-tests.v3.launch.json)
完整 wall time 为16.8121085s，stderr 为空。唯一 warning 是关闭 cacheprovider 后
pytest 不识别仓库的 `cache_dir` 配置；不影响测试结果。
检查包括真实 canonical 对象重建/PFT refresh、独立 host normalization/tail/pool/
defined/PFT mask 与 summary oracle、raw434 的解析链式导数和其他 sample 精确零、
七键 round trip、投影/布局/normalization 拒绝、SHA 检查在 pickle 前执行、TRAIN/
held-out/range 约束，以及 x NaN、direction NaN、其他 sample 方向非零三种
admission failure 的完整原数组留存。未重复执行既有整网、GPU 或长测试。

最终 fixture 与接入命令：

```powershell
conda run -n ORCJAX python -B scripts/hpc/capture_shared_latent_T7_summary_operands.py --output outputs/T7-summary-capture-20261011/summary-fixture.v3 --synthetic-fixture
conda run -n ORCJAX python -B scripts/hpc/probe_shared_latent_T7_batch_numerics.py --backend cpu --operands outputs/T7-summary-capture-20261011/summary-fixture.v3.operands.npz --output outputs/T7-summary-capture-20261011/summary-fixture-probe.v1.json --repeats 1
```

| 实际检查 | 结果与证据 |
| --- | --- |
| fixture capture | exit0 / wall10.0525053s；[capture receipt](../../../outputs/T7-summary-capture-20261011/summary-fixture.v3.capture.json)、[NPZ](../../../outputs/T7-summary-capture-20261011/summary-fixture.v3.operands.npz)、[CLI/日志记录](../../../outputs/T7-summary-capture-20261011/summary-fixture.v3.launch.json) |
| 既有 affine probe 接入 | PASS / exit0 / wall5.8815583s；[结果](../../../outputs/T7-summary-capture-20261011/summary-fixture-probe.v1.json)、[49数组](../../../outputs/T7-summary-capture-20261011/summary-fixture-probe.v1.results.npz)、[CLI/日志记录](../../../outputs/T7-summary-capture-20261011/summary-fixture-probe.v1.launch.json) |
| 真实源缺失拒绝 | exit2 / wall5.3901677s；[failure receipt](../../../outputs/T7-summary-capture-20261011/summary-real-missing.v1.failure.json)、[完整五路径CLI/日志记录](../../../outputs/T7-summary-capture-20261011/summary-real-missing.v1.launch.json)；`assets_unavailable / not_executed / load_sources`，无真实 NPZ |

实际缺失资产拒绝命令（未进入 preprocessing）：

```powershell
conda run -n ORCJAX python -B scripts/hpc/capture_shared_latent_T7_summary_operands.py --output outputs/T7-summary-capture-20261011/summary-real-missing.v1 --parameters E:/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-latent-reader-20261004-T7-assembled-input.parameters.pkl --context E:/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-expert-real-daily-context-c599236.pkl --t7-manifest outputs/diagnostics/shared-latent-reader-20261004/latent-reader-T7-20261005.manifest.json --prepared-manifest E:/ORCHIDEE-MAN-JAX/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json --dataset-manifest E:/ORCHIDEE-MAN-JAX/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
```

fixture summary/direction 是 FP32 `(4,21,4)`，kernel/bias 是 FP32 `(4,384)`/
`(384,)`；Boolean mask 和诊断 cotangent 是 `(4,21,384)`，每个 sample 的21个
field 中20个有效。21是这组 synthetic contract 的 field 数，未宣称是缺失真实
context 的布局。原 raw states/direction 保持 FP64 `(4,3854)`。

本次接入只用 `--repeats 1`：三 routes 各一 baseline 和一 warm cycle，共6测量，
不是前包三 warm cycles 的12测量。4次小编译包括3个 affine graphs 和 serial
merge；6个 oracle comparison、4个 cross-route、3个非平凡 repeat、6个独立
AD pairing/direction 以及6个其他 sample 精确零检查全部通过，保留49数组。
`rtol=1e-5, atol=1e-6` 不变。probe 文件未改；父级核对其绑定的 operand SHA
与实际最终 NPZ 一致。此小型接入证明提取产物能进入已接受的判别器，未提供
原数据复现、GPU 因果或训练速度结果。

最终捕获脚本/current fixture source SHA 是
`f4136665209844236c7168c40be8d486e3ff164ddb63cef30677f8f85bc2d920`，父级已核对
与 v3 capture receipt 一致。AD helper whole-file SHA 是
`b6e07de35e555befd77e2c536e68b786496d80e7bb1a0a7bce54377e1c2e98f4`；其余
source/array identities 见 receipt。缺失路径 CLI receipt 的实际脚本 SHA 为
`1b007dd892e6181679f080f80f1fb60594a5e7e6587deeade2988ca6977a0288`；此后仅
修正 failure origin 日志，使已尝试 preprocessing 的拒绝不被标成未执行。
未改数值路径，因此复用已完成测试，不改写旧 receipt/source identity。
早期 fixture.v1 的 Windows source-path key `KeyError/exit1` 实现错误日志保留，
后续用 `as_posix()` 修正；这不是科学数值失败。

主管已直接核对实现、10项测试与实际日志、v3 capture/source、fixture probe 与
真实源缺失 receipt，接受范围仅本地方法与失败保存。仅提交/推送指定七个任务
文件到 daily research branch；本轮 outputs 证据不加入 Git。起始/执行 HEAD 和
历史身份保持上述原值，交付提交不替代旧实验执行版本。

## 后续决定

取得最小缺失资产后，仅运行同一薄 CPU capture 和既有三 routes/三 warm cycles
affine probe；每轮原数组与失败保存，`rtol=1e-5, atol=1e-6` 不变。资产未取得时
只交付方法/拒绝路径验证，真实状态为 `not_executed`，不追加后层或大图。
本机 CPU JAX0.10.1 不代替 deployed GPU0.4.38；GPU wrapper/bootstrap 是另一包。
原 strict state 失败、roughness state20 NaN、2.55× 仅组件证据与完整梯度/物理/
精度/数据边界全部保持。先提高日尺度训练执行速度，用固定 Teacher/prepared 版本
比较；正式训练等待 main Teacher 定下、新数据与相称兼容性/准入验证。
