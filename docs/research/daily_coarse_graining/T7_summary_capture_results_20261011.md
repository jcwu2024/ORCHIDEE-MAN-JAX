# T7 early summary_input：本地提取与验证

2026-10-11。前包的本地方法、fixture 接入与失败保存已获主管接受并提交。
**本次后续授权的八文件获取、真实薄 CPU capture 和三轮 affine probe 已完成，通过。**
真实 summary 为 `(4,257,4)`；12测量/91数组的独立 oracle、AD、cross/repeat/isolation
全部通过。此早期 projection 在本机 CPU 上未复现超出 strict gate 的差异。
没有选择生产修复，不排除 later64→64/attention 原因，不构成原 T7 或完整训练提速验收。

## 身份和有界资产定位

固定 cwd 为 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining`，branch 为
`research/daily-coarse-graining`。前 fixture 包起始 HEAD 为
`120c6378966b060f4da6c6763250e43f2d26e2e4`；本次真实获取/执行 HEAD 为
`9de13774df05ddf015557daee3b8540a07ddb497`，执行时包含下述局部路径兼容改动。
ORCJAX 解释器实际为
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
前包六条 runtime 字面路径的缺失记录保留为当时历史；本次八文件已获取，
该历史查询不是当前缺失结论或穷尽不存在证明。

前包确定的八文件获取清单已全部完成，身份不变：

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

## 前包实现与 fixture 验证记录

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
field 中20个有效。21是这组 synthetic contract 的 field 数；本次真实field数为257。
两者 raw states/direction 均保持 FP64 `(4,3854)`。

前 fixture 包接入只用 `--repeats 1`：三 routes 各一 baseline 和一 warm cycle，共6测量，
不是前包三 warm cycles 的12测量。4次小编译包括3个 affine graphs 和 serial
merge；6个 oracle comparison、4个 cross-route、3个非平凡 repeat、6个独立
AD pairing/direction 以及6个其他 sample 精确零检查全部通过，保留49数组。
`rtol=1e-5, atol=1e-6` 不变。probe 文件未改；父级核对其绑定的 operand SHA
与实际最终 NPZ 一致。此小型接入证明提取产物能进入已接受的判别器，未提供
原数据复现、GPU 因果或训练速度结果。

前包最终捕获脚本/fixture source SHA 是
`f4136665209844236c7168c40be8d486e3ff164ddb63cef30677f8f85bc2d920`，父级已核对
与 v3 capture receipt 一致。AD helper whole-file SHA 是
`b6e07de35e555befd77e2c536e68b786496d80e7bb1a0a7bce54377e1c2e98f4`；其余
source/array identities 见 receipt。缺失路径 CLI receipt 的实际脚本 SHA 为
`1b007dd892e6181679f080f80f1fb60594a5e7e6587deeade2988ca6977a0288`；此后仅
修正 failure origin 日志，使已尝试 preprocessing 的拒绝不被标成未执行。
未改数值路径，因此复用已完成测试，不改写旧 receipt/source identity。
早期 fixture.v1 的 Windows source-path key `KeyError/exit1` 实现错误日志保留，
后续用 `as_posix()` 修正；这不是科学数值失败。

前包主管已直接核对实现、10项测试与实际日志、v3 capture/source、fixture probe 与
真实源缺失 receipt，接受范围仅本地方法与失败保存。前包指定七个任务文件已提交/
推送到 daily research branch；前包 outputs 证据未加入 Git。该前包起始/执行 HEAD
和历史身份保持上述原值，前包交付提交不替代旧实验执行版本。本次八文件/真实
capture后续包也已获主管验收，授权按七文件范围提交推送。

## 本次真实获取与捕获

本次只读获取授权仅含八个既有文件。当前 TAP 接口 `192.168.111.108` Up、
`192.168.11.0/24` route，以及正常 host-key verification/既有 `train` 身份的
cln01 authenticated metadata read 共同验证连接；cln01成功，未追加cln02尝试。
远端仅执行 `stat` 和 SCP读取，未执行 remote hash/Python/反序列化/计算/调度。
未下载原 input_bundle/profile、Teacher shards 或 GPU assets；原T7 manifest复用
已匹配的本地文件，不是第九个传输文件。

[八文件 ledger](../../../outputs/T7-summary-real-20261011/retrieval.20261011T074542Z.json)
保存逐条 source/target、remote bytes/mtime、local SHA、完整 argv、stdout/stderr、
exit/timing。全部经 unique `.part` 下载、本地验证后落位，没有冲突覆盖或清理。
local mirror 根为 `E:/ORCHIDEE-MAN-JAX/runtime/`，表内均为其 relative path；
远端根是 `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/`。

| 文件 | bytes | 本地 SHA256 | SCP wall秒 / exit |
| --- | ---: | --- | --- |
| `outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json` | 350163 | `8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31` | 0.390351 / 0 |
| `outputs/training/pft14-daily-teacher-v6/dataset_manifest.json` | 60610898 | `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99` | 8.031424 / 0 |
| `outputs/diagnostics/shared-latent-reader-20261004-T7-assembled-input.parameters.pkl` | 264518171 | `4f95d18d97d2feb4dcab8e1dda6caeb242a4869f044a00b6a7ae6bc689806c14` | 27.260246 / 0 |
| `outputs/diagnostics/shared-expert-real-daily-context-c599236.pkl` | 14465399 | `ff3fdabe755d599bc2ceee72087c03983d9a629add93166519dcda1f56be9026` | 1.876545 / 0 |
| `outputs/training/target-oriented-prepared-1000-a243029/chunks/627-813/update-0742.pkl.gz` | 3332079 | `61747a584c11186a7016ef010199997949de733db2c0030a197e097e1fe0fda5` | 1.024944 / 0 |
| `outputs/training/target-oriented-prepared-1000-a243029/chunks/251-438/update-0275.pkl.gz` | 3366175 | `03faafa481d0d63b913efa3eaaa649fa8bfa689ddf01d97c95ec403006128610` | 0.697049 / 0 |
| `outputs/training/target-oriented-prepared-1000-a243029/chunks/439-626/update-0482.pkl.gz` | 3371650 | `db06c10eb82864021c3313a17727ccea1d16f52e3da5f661196f14dd4b9cbf39` | 0.899106 / 0 |
| `outputs/training/target-oriented-prepared-1000-a243029/chunks/251-438/update-0299.pkl.gz` | 3391246 | `ee7fa6e0f537fcd75dc8d02bbf4aa4f0438119f1ca73368e8f7f9d64af1f18aa` | 0.762374 / 0 |

共353405781 bytes，八次传输全部exit0；prepared identity的dataset/schedule/source/
contract四值与前包声明完全一致。四artifact路径和SHA实际从该manifest解析。
TRAIN/train四首窗口互不重叠：742=`171.0-095.0/1991/start225`，
275=`293.0-093.0/1970/start178`，482=`297.0-081.0/1971/start105`，
299=`345.0-069.0/1967/start202`；各窗口7天、window0/day0，sample299原state SHA
仍为 `54d01d0f376b0ceed8a843740328dba9d61d2a8989507436bda1a0d16c624f5d`。

首次 `summary-real.v1` 在context反序列化时因Windows不能实例化Linux `PosixPath`
退出1/wall5.490513s，尚未进入preprocessing；[完整receipt/log](../../../outputs/T7-summary-real-20261011/summary-real.v1.launch.json)
保留，实际source仍是前包 `f413666...2d920`。局部 `HistoricalAssetUnpickler` 仅在
Windows把该class恢复为 `PurePosixPath`，保留historical lexical path且不把remote
path重绑定为本机文件。仍先验证文件SHA，再读取pickle；数字、掩码和gate不改。
针对性Linux-path pickle roundtrip/数值bytes测试加入后，
[完整11项测试](../../../outputs/T7-summary-real-20261011/summary-tests.v4.stdout.txt)
PASS/16.16s；[launch](../../../outputs/T7-summary-real-20261011/summary-tests.v4.launch.json)
exit0/wall19.840629s。相同cache_dir warning不影响检查。

真实capture仍用前述五条路径，只将output改为
`outputs/T7-summary-real-20261011/summary-real.v2`；
[capture receipt](../../../outputs/T7-summary-real-20261011/summary-real.v2.capture.json)、
[NPZ](../../../outputs/T7-summary-real-20261011/summary-real.v2.operands.npz)、
[完整CLI/日志](../../../outputs/T7-summary-real-20261011/summary-real.v2.launch.json)
为exit0/wall21.786731s，origin=`captured_component_operands`。
实际capture source SHA为
`70619b0a6445cde87eccaeb31feac46bbb34cdc0d7a6f68c68938c4d09901072`；
NPZ SHA为 `7af399244e7ae864cb666fdf942705bb2d1b5854cec17e8e41572a770360d1df`。
真实x/direction为FP32 `(4,257,4)`，saved kernel/bias为FP32 `(4,384)`/`(384,)`，
mask/cotangent为 `(4,257,384)`。四sample有效field数为250/251/251/251；mask保留。
normalization文件保存的mean/scale本身为FP64，canonical neural计算转换为FP32；
未重建统计或改变precision。raw434 sample3-only JVP与其他sample精确零成立。

```powershell
conda run -n ORCJAX python -B scripts/hpc/probe_shared_latent_T7_batch_numerics.py --backend cpu --operands outputs/T7-summary-real-20261011/summary-real.v2.operands.npz --output outputs/T7-summary-real-20261011/summary-real-probe.v1.json --repeats 3
```

[真实probe结果](../../../outputs/T7-summary-real-20261011/summary-real-probe.v1.json)、
[91原数组](../../../outputs/T7-summary-real-20261011/summary-real-probe.v1.results.npz)、
[CLI/日志](../../../outputs/T7-summary-real-20261011/summary-real-probe.v1.launch.json)
为PASS/exit0/wall7.638124s，CPU0/JAX0.10.1。probe未改，source SHA
`c175841fcf9131122adebebc683cf33ca6f980e145cdecfdff0cd77e1bf2e2e8`。
三routes各baseline加三warm cycles，共12测量；12oracle/12isolation/12实际AD
pairing与oracle方向检查、24 FP64 FD、8cross/9repeat全部通过。四次小compile
为三个operators加serial merge，三个optimized HLO gzip均保存。
cross primal/tangent/input-gradient精确一致；loss和parameter gradients有微小
route差别，均在strict内，cross最大allowance ratio为0.004743；所有repeat精确。
独立FP64 oracle最大ratio为0.102922。人工diagnostic cotangent仍不是T7 loss。

[saved-only NumPy复核](../../../outputs/T7-summary-real-20261011/summary-real-review.v1.json)
重新从91数组验算既有判别器，15项身份/数组/完整cycle/receipt对应检查全部通过；
未重跑JAX、compile或模型。warm median为B1 0.003302s、vmap 0.002439s、map
0.003863s，仅小算子本机诊断记录，不是训练吞吐或原GPU结论。
旧包的 `not_executed` 缺失源receipt保持历史身份；当前真实状态是captured/CPU passed。
主管已最终验收本次七个源码/测试/文档改动，授权按此范围提交推送；outputs/
八源资产留在Git外。验收仅覆盖真实薄capture和这个early CPU边界，无原模型修复/
完整提速验收或新GPU诊断授权。

## 后续决定

八文件与真实薄capture/CPU判别已完成；不重复下载或重跑已通过的相同检查。
本包授权止于此边界，没有GPU、后层、prefix、T7/T30、训练或paid操作。
`rtol=1e-5, atol=1e-6`、原数据/权重/历史overlays保持，未选生产修复。
本机 CPU JAX0.10.1 不代替 deployed GPU0.4.38；GPU wrapper/bootstrap 是另一包。
原 strict state 失败、roughness state20 NaN、2.55× 仅组件证据与完整梯度/物理/
精度/数据边界全部保持。先提高日尺度训练执行速度，用固定 Teacher/prepared 版本
比较；正式训练等待 main Teacher 定下、新数据与相称兼容性/准入验证。
