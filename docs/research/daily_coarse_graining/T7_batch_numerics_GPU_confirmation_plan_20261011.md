# T7 合批数值：最小目标 GPU 确认与原集成回归

2026-10-11。**没有选定生产修复；当前不采用 whole-prefix scalar mapping。**
主管已接受本轮作为本地验证方法和后续准备；真实操作数/部署仍待完成。
这不是原模型数值修复、GPU readiness、完整训练提速或长期科学验收。
本轮只准备本地方法与计划，没有新GPU授权。先取得early tokenizer真实operands，
再做一次bounded V100确认；局部修复有因果/AD证据才另行决定原T7验证，不自动追加整prefix图。

## 证据与假设

[raw JVP结果](T7_raw_neural_JVP_results_20261010.md)在canonical rebuild→assembler→context→raw
确认active差异：434/2382/2621/200每轮失败1769/10/36/21个tangent坐标及四个primal坐标。
161 masks exact、16次非平凡重复exact、12次隔离通过；首个分歧与整T7因果仍未证明。
HLO中B1 `[128,64]×[64,64]` 算法18、B4 `[512,64]×[64,64]` 算法19，以及
attention contraction差别，只支持“batch lowering改变FP32舍入/归约”假设。
434 normalization scale `1.7021657286023486e-7` 可放大敏感性；route equality
也不能证明accuracy。attention目前无causal结果，不属于下一次GPU scope。

[四窗口集成结果](shared_latent_reader_results_20261004.md)保留固定权重组件
`5.025831s → 1.967053s`、`2.555005×`，但state AD gate失败，未采用。
B1不是真值；NumPy FP64小算子oracle是独立诊断，不改变生产reference/precision。

## 真实输入：必须完成的下一项本地准备

64→64 reader的内部x/direction未保存；不为获取它构造shared-backbone大图。
推荐先提取最早的tokenizer state-field `summary_input` projection：
原day0 state3854 → `standardize_continuous` → `_mask_inactive_pft_elements` →
`_masked_summary`，得到 `x[4,Nfield,4]`、原kernel `[4,384]`、bias `[384]`。
边界是 `strong_baseline_architecture.py:1111–1128`，helpers分别位于
`input_conditioning.py:12`、`strong_baseline_architecture.py:984/957`，
参数初始化来源 `shared_transformer.py:177`，实际saved parameter leaf path须核验。
这是early projection干预；通过或未复现均不能排除later64→64/attention原因。

**thin CPU extractor尚未实现，是下一项准备，不是已经到位的输入。**只反序列化
原context的head/contract/layout、参数与TRAIN742/275/482/299四个首T7窗口day0，
复用canonical rebuild/assembler的state定义和PFT mask。保留原
`head.input_normalization` mean/scale/tail threshold/tail mask、state-field pool、
element-PFT index；只执行上述三个helpers，不调用完整head/context forward、
backbone、raw prefix或物理步。对该小preprocessing函数做CPU JVP，取方向
434/2382/2621/control200的dsummary（仅sample3激活），不做整网reverse/T7/T30。
若真实资产无法取得或小边界不能执行，明确缺口并停止；人工方向不得标成captured。

输入从历史 `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/` 下解析：

| 输入 | relative path |
| --- | --- |
| 参数 | `outputs/diagnostics/shared-latent-reader-20261004-T7-assembled-input.parameters.pkl` |
| context | `outputs/diagnostics/shared-expert-real-daily-context-c599236.pkl` |
| prepared manifest | `outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json` |
| dataset manifest | `outputs/training/pft14-daily-teacher-v6/dataset_manifest.json` |
| 原manifest | `transfer/shared-latent-reader-20261004/latent-reader-T7-20261005.manifest.json` |

input bundle取原manifest的`input_bundle`；四个artifact路径/rows从prepared manifest
解析，不猜shard路径。身份导航 `probe_shared_latent_reader_T7_pair.py:203 assets`
还会绑定overlays并检查profile/reference/admission/head/source，不能直接当thin loader；
`shared_latent_T7_batch_assets.py:114 load_training_T7_cases`构造完整case并作小
identity shape检查，无模型forward，亦不提供内部operand。extractor实现后需固定
CPU-only parser/argv、保存capture receipt/source hash；本轮没有执行提取或传输。
本地declared shared-root参数/context/TRAIN299 artifact/prepared manifest/原manifest
均缺，且无runtime junction；已有raw24结果/HLO2 gzip不能反推内部operands。

## 与现有 harness 完全一致的 NPZ 合同

入口 `scripts/hpc/probe_shared_latent_T7_batch_numerics.py` 仅接受七个exact keys：

| key | 必要类型 / shape |
| --- | --- |
| `x` | finite FP32 `(4,N,K)` |
| `kernel`, `bias` | finite FP32 `(K,M)`、`(M,)` |
| `direction` | finite FP32，同x；前三sample exact zero，sample3 nontrivial |
| `cotangent`, `mask` | finite FP32 / Boolean，均`(4,N,M)`，active objective非平凡 |
| `metadata` | Unicode scalar JSON；`schema_version=T7_affine_operand_bundle_v1` |

全部数组元素总数≤2,000,000。origin只能是`synthetic_method_fixture`或
`captured_component_operands`。captured metadata必须绑定：
`parameter_file_sha256=4f95d18d97d2feb4dcab8e1dda6caeb242a4869f044a00b6a7ae6bc689806c14`、
`prepared_context_sha256=ff3fdabe755d599bc2ceee72087c03983d9a629add93166519dcda1f56be9026`、
`artifact_ids=[742,275,482,299]`，且非空 `parameter_path/capture_boundary/
capture_source_sha256/capture_receipt_sha256/direction_coordinate/mask_meaning`。
direction_coordinate只能434/2382/2621/200；一个bundle一个方向。保持同一geometry，
后续方向可复用同一批小executable，但目前CLI一次仅消费一个bundle；
下一次最小GPU确认优先434，不把四方向变成四轮新compile。

默认synthetic固定为 `(4,128,64) @ (64,64)`，仅验证判别方法。
捕获真实包后先CPU验证，再冻结未来GPU命令：
`<canonical-GPU-python> -B <declared-checkout>/scripts/hpc/probe_shared_latent_T7_batch_numerics.py
--backend gpu --operands <identity-bound.npz> --output <unique.json>
--seed 20261011 --repeats 3`。这是参数主体，尚非已验证GPU launcher；不覆盖旧证据。

## 一次 bounded V100 确认

同一operands/进程/卡比较：四次B1 scalar调用+device merge、B4 `vmap`、B4
`lax.map`。原 `_linear` 不变；每route joint结果包含raw primal/input JVP、
masked scalar mean objective及完整小matrix kernel+bias/input梯度。
这是scalar-objective VJP，不宣称已有参数JVP或任意cotangent整网AD。
NumPy FP64独立analytic primal/JVP/全部小矩阵梯度，加输入方向central FD和
JVP/VJP pairing共同校验；不得以B4接近B1替代oracle检查。目标均值是全output
坐标数作分母、inactive项置零；serial参数梯度均值、input梯度stack÷4，raw JVP不÷4。

建议未来独立批准：`gln01` 一张空闲V100、非独占，累计1200 GPU seconds
（≤0.333334 card-hours，free估计CNY0）：preflight≤240s、小图/calls≤780s、
保存释放≤180s。**三个operator compile各一次，加一个small serial merge compile**；
没有retry、algorithm搜索、attention/prefix/T7/T30或paid fallback。
现有harness无外层deadline/memory sampler；批准执行前须准备并验证bounded wrapper，
新runtime preflight与24GiB监测/终止规则，不能声称CLI自身已实施这些资源门槛。

记录CPU affinity/load、OMP4/OpenBLAS1、GPU processes、JAX/CUDA/toolchain、实际
HLO shape/algorithm、lower/backend/首次有效测量时间及同步热调用中位数。
harness保存每route untimed cycle0和warm cycles1–3共12轮，加NumPy FP64 oracle，
同一`.results.npz`共91数组；逐轮strict/oracle/isolation/实际JVP-VJP pairing验收，
共8项cross-route、9项非平凡within-route repeat，后来通过不覆盖中间失败。
非有限原数组仍写入NPZ；JSON列actual/reference/derived nonfinite计数，不可表示的
max/dot标量用null，strict失败为`numerical_failed`/exit3，不改gate。
`rtol=1e-5, atol=1e-6`、FP32neural/FP64physics、161输出/masks、容量/参数、
loss/sampling、public native-checkpointed AD及完整跨日梯度边界全部保持。
非有限、内存或deadline分别分类并停止，不被其他gate通过豁免。使用canonical
`.venvs/orcjax_gpu`/`.venvs/orcjax_cpu`与已接受container/toolchain：deployed JAX0.4.38
须在`jax.numpy`/模型import前注册PJRT，复用accepted bootstrap及canonical launcher
动态`readlink -f` driver-library binds；此入口接入和deployed验证尚未完成，本地
CPU JAX0.10.1通过不替代它。任选可用cln01/cln02作jump，冻结argv须CPU参数校验/
目标shell `--dry-run`；真实提取、deadline/NVML wrapper及bootstrap验证到位前不称GPU ready。

## 结果决定与原集成 T7 回归入口

| 结果 | 决定 |
| --- | --- |
| captured差异复现，oracle/AD支持一个局部候选且strict通过 | 主管选定局部修复；完成显式接入/身份验证后另行决定一次集成T7 |
| synthetic通过、真实包缺失、小HLO不匹配或early projection未复现 | 只保留机制/该边界结论；不排除later contraction，停止交主管，不采用wholeprefix map |
| strict/AD失败、map收益丢失、非有限或资源超限 | 保留失败/代价证据并停止；不改tolerance/reference、不自动加大图 |

原集成指同一个66,123,713参数latent-reader候选架构，非pair脚本名为`original`
的64,884,481参数旧架构。入口 `probe_shared_latent_T7_batch.py:195
_candidate_objective` →原turnover/backbone overlays→
`shared_recurrent_training.make_shared_recurrent_objective`；
`shared_latent_T7_batch_ad.py:14/25/29`提供mean目标、完整参数/state AD与serial merge。
`probe_shared_latent_T7_state_jvp.py:54/67`共享公共目标及primal桥，不替代全参数AD。

历史argv从 `outputs/diagnostics/shared-latent-reader-20261004/
latent-reader-T7-batch-20261008.serial-saved.target-shell-dry-run.txt` 读取；closed
job15672280的proposal/readiness/submission保留paid身份。raw
`T7-raw-neural-jvp-20261010.execute.python-dry-run.json`仅是prefix argv。
原remote cwd是`runtime/checkouts/shared-split-width-v1`，source
`c599236bcf57b745e9bafed678298010f4c3b58f`加已执行overlays，不能用本地HEAD替代。
batch entry必需manifest/context/parameters/adapters/modules/turnover/preflight、
prepared/dataset manifests、candidate/batch binaries、output和对应readiness/ledger。
现有selector没有本轮repair接入；CPU readiness含整模型eval_shape，本轮不运行。

候选改变源码/数学身份必须新编，**旧B4 binary不能装新修复**。数学身份未变的旧
B1可作对照；若B1也须新编，重新明确scope。独立局部NP导数oracle仍需通过，
再检查原四窗口三轮全部loss/aux/masks、66,123,713参数梯度、15,416 state坐标、
cross-route与within-route repeats，保存reference/state/参数全数组。roughness state20
NaN独立admission失败不能豁免。历史B4 lower435s/backend7595s不适合20分钟scope；
未来最多一次新candidate T7 compile可按10800s、1V100/8 allocated CPU/64GiB、
单非独占gnall规划（3h×(2.20+8×0.07)=CNY8.28，新独立批准），保护≥600s
比较/保存/退出时间，到期停，不保证成功。仍不证明完整daily32（八differentmicro4）
+一个T7+一个T30 AdamW更新速度或科学准确性；组件收益不可相乘。

## 本轮验证

daily root/branch/HEAD/import、入口/schema已核对；CPU证据选`outputs/T7-batch-numerics-20261011/affine-cpu.v3.json`，
synthetic PASS、6项matrix tests PASS（3.80s，含中间轮mutation及nonfinite保存/exit3 guard），input owner
6项PASS；首版final-only及v2结果保留，以v3为最终method evidence。这些本地证据不归因原模型。
未运行remote/GPU/真实模型forward、整网tracing/AD或Teacher生成。daily图索引
excluded/not_tracked，源码精确fallback；thin extraction、wrapper、GPU因果/原集成gate未验证，旧额度关闭。
