# 同一 early summary_input GPU 本地执行准备

2026-10-11。本包准备一个独立1200s、单V100、free CNY0的小 affine 确认入口。
**仅本地准备，尚非 GPU ready，未批准执行。** 无远程连接、传输、部署、GPU、
训练、整prefix/T7/T30或 paid fallback。主管已验收，按明确13文件范围版本化；
实际提交推送身份见[postcommit交付receipt](../../../outputs/T7-batch-gpu-preparation-20261011/postcommit-delivery.v1.json)，
scope/deployment v2填写实际新commit，execution保持false；不在同一commit内伪填自身SHA。
入口是 [execution contract](T7_batch_numerics_GPU_execution_contract_20261011.md)。
前一 accepted stage 为 `048c963d9812d77a7c59291ff6374b6e37201738`，
该起始HEAD不是新增源码版本；新增版本身份以postcommit交付receipt为准。

## 测试事故与新输入身份

子代理 `/root/input_numerics_audit/gpu_supervisor_tests` 的初版负面测试将真实
NPZ、receipt和capture-source作为可写测试目标，误删
`outputs/T7-summary-real-20261011/summary-real.v2.operands.npz`、
`summary-real.v2.capture.json`，并把
`scripts/hpc/capture_shared_latent_T7_summary_operands.py` 改为19bytes
`invalid replacement`；子代理也报告其missing-source variant调用unlink。
父级仅从 accepted HEAD 恢复source，实际SHA重回
`70619b0a6445cde87eccaeb31feac46bbb34cdc0d7a6f68c68938c4d09901072`。
随后一次“已修订”测试仍引用REAL_SOURCE；完整静态审查在执行前截住它。
最终所有负例目标均为独立tmp_path文件，真实source只read/copy，
无需finally恢复真实资产；hash负例使用临时文件真实hash，不伪装真实资产身份。

原两文件没有已知精确副本。核查实际本包输出/pytest temp、system TEMP精确候选，
以及父级root outputs/runtime outputs指定路径后收束；本包未创建原资产的
deployment/staging/tar/zip副本，不VSS/扫盘。原两个路径保持缺失。
原误操作的两次命令均为同一launcher pytest（先conda captured output，
再PYTHONUTF8/`--no-capture-output`）；第一轮exit1且Conda打印GBK异常，
第二轮保留tool尾行`9 failed, 6 passed, 1 warning in 0.62s`。
当时未保存文件日志、PID或准确起止时间，不能补造完整原始日志。
事故记录及已保留原tool文本范围见输出目录`incident.v1.json`与
`incident-closeout.v1.json`；后续每次验证均保存实际stdout/stderr/exit/timing。

主管明确授权一次相同薄CPU capture到全新stem，使用仍完整、身份核验的同八本地
源与未改capture-source。新capture在`2026-10-11T11:21:42.536036Z`启动，
exit0、18.670917s，实际HEAD为`048c963d9812d77a7c59291ff6374b6e37201738`。
只执行原canonical预处理/JVP，没有完整affine probe重跑。
本地新资产位于`outputs/T7-batch-gpu-preparation-20261011/`：

| 文件 | 全文件SHA256 |
|---|---|
| `summary-recovery.v1.operands.npz` | `bf36a098b4e6a89d555f8d18521555f2da66135425d9435cfe01878adeec6129` |
| `summary-recovery.v1.capture.json` | `f863a2d964f626cec327981d86a8832e73aa45af62d708a342805c4aa4220e00` |

六数组bias/cotangent/direction/kernel/mask/x的dtype、shape、C-order byte SHA
全部与主管事故前`exec-403f5a31-2dd1-44ae-b888-f9b9ae77759c`读取证据一致。
主管另外saved-only独立核验6/6数组、receipt绑定及9项capture source身份，
正式接受新输入数值同一性；**原文件字节身份未恢复**。
完整六SHA见`summary-recovery.v1.numeric-verification.json`，
新receipt记录当前probe source
`05e2ee66b1905fd238722d3a4cf645e74639f7134de4af5a700cf0b657305ee0`。
原历史NPZ `7af399...d1df`、receipt `e69592...f4a3`、capture HEAD
`9de13774...b497`及历史probe `c175841f...e2e8`只作历史证据；
旧CPU12测量/91数组结果仍留原输出，不冒充新capture执行结果。

## 有界入口及资源判定

Python supervisor仅导入NumPy/公共probe admission，不导入JAX/模型。
它验证新NPZ/receipt/source全文件身份与六数字数组绑定，固定真实
`(4,257,4) @ (4,384)`、TRAIN742/275/482/299、sample3/raw434/window0/day0。
新增GPU入口复用`orcjax_gpu_runtime.initialize_gpu_backend()`，在
`jax.numpy`/模型导入前注册PJRT；CPU路径保持独立。原数学、mask、
precision、tolerances、oracle和AD方法不变。

未来范围固定为gln01（实际host也允许`ibc13b02n04`）、一张空闲V100、非独占；
preflight≤240s、小图compile/calls≤780s、释放reserve180s、总≤1200s、
24GiB，OMP4/OpenBLAS1。同一scope只有一个固定exclusive-create ledger，
旧budget从不打开。三route各compile一次、serial merge一次，repeats3，
计划12测量/91数组；人工diagnostic cotangent不是T7 loss。
host index/UUID与container device0分别记录，外国GPU进程出现时只停本任务组。
初始与释放都严格要求所选卡0MiB、0%、无compute process；未知释放
不能通过、不能退款，数值exit3与释放失败分别保留。

外层Bash从第一个shell timestamp计时，以setsid隔离CPU supervisor：
第1010s TERM、第1190s KILL；独立owned phase timer在driver/container前建立，
免疫TERM/INT，第1010s TERM、第1180s KILL phase PG，先于外层CPU KILL。
phase leader已退出时timer清理同PG descendants；timer留在该PG期间防止PGID复用。
Python确认组释放后从active集合删除，另保留history，不再信号历史组。
outer-timeout marker独立noclobber，不覆盖primary JSON/ledger，并关闭one-shot。
正常退出不宣称释放；最终UUID观察与全部owned组退出共同判定。

live入口要求clean tracked/untracked checkout、明确最终commit、冻结source manifest、
open/authorized scope和批准说明，拒绝login node/旧scope/另一个ledger/旧输出。
scope仍是`execution_authorized=false`、UUID未选择；v1预提交快照的commit未填写，
postcommit v2仅补入实际代码身份，不授予GPU执行。
完成轮次的完整raw arrays逐次写atomic partial NPZ，计时外写盘，超限仍保留已完成证据。
不retry、不搜索algorithm、不追加层/整网/训练，不放宽strict gate。

## 本地验证与交付入口

输出根是daily checkout的`outputs/T7-batch-gpu-preparation-20261011/`：

| Receipt / artifact | 意义 |
|---|---|
| `summary-recovery.v1.launch.json` / `.numeric-verification.json` | 一次授权薄CPU捕获及六数组原SHA对照 |
| `shell-evidence.v3/receipt.json` | 14项shell tests通过，11.18s；真实blocking mock TERM→KILL，逐项tmp/process mutation audit |
| `bash-real-parser.v1.receipt.json` / stdout/stderr | 真实Git Bash→实际ORCJAX Python parser，新输入dry-run exit0/0.436603s；未启动Supervisor/GPU |
| `launcher-tests.v2.receipt.json` / stdout/stderr | 30项launcher CPU/mock tests通过，1.31s；输入真实fixture hash负例、one-shot、timeout、owned cleanup、manifest/dirty checkout、0MiB释放、数值3与资源未知分别记录 |
| `launcher-tests.v1.receipt.json` / stdout/stderr | 2failed/28passed、1.25s的失败原日志；修正旧host文案期望及缺少假phase报告的test fixture，runtime gates未变 |
| `scope.proposed.v2.json` / `deployment-manifest.v2.json` | postcommit实际代码身份、新输入与源码SHA、固定checkout/transfer/output；execution=false/未选GPU；v1保留预提交快照 |
| `postcommit-delivery.v1.json` | 实际Git命令/exit/timing、13文件commit范围、local clean与remote精确HEAD核验 |
| `incident-closeout.v1.json` | 三文件事故、新capture授权与身份修复、日志局限，旧bytes未恢复 |

Python AST解析通过；本地针对性验证合计30launcher+14shell tests通过。
关闭pytest cacheprovider后，既有`cache_dir` config产生warning，不创建项目cache。
真实Bash dry-run计划12测量/91数组/4小compile，source manifest与新输入绑定。
Git Bash缺少实际Linux setsid；本地PG测试使用私有函数层mocks及真实blocking child，
不证明目标Linux session/PG isolation、真实Singularity/driver/PJRT/JAX0.4.38/NVML。
未来actual target preflight必须在独立1200s范围内执行一次；本地14shell tests
和CPU mocks不替代它。完成准备不构成GPU授权或原T7数值因果/完整更新速度/科学验收。

delegate使用Sol/medium（shell/tests/contract）与Luna/high（launcher CPU mocks）；
执行owner保留集成、真实资产隔离、唯一薄capture和实际parser验证。
daily graph excluded/not_tracked（main generation `2026-10-10T08:46:50Z`），
使用既有图coverage和exact-path source fallback，无重索引或图零命中否定。

验收版本化文件共13项：两个launcher、probe bootstrap/partial
修改、两个unit tests、execution contract、本报告、GPU confirmation plan，以及
五个当前入口文档（START_HERE/NEXT_STEPS/current-status/CODE_MAP/DOCUMENT_STATUS）。
精确路径、size、SHA与最终diff-check见
[verification-final.v2.json](../../../outputs/T7-batch-gpu-preparation-20261011/verification-final.v2.json)；
`verification-final.v1.json`保留为此前已保存快照。v2仅收束文档，复用44项通过证据，未重跑测试或数值。
outputs下的capture/receipts/HLO和历史八源资产不进入Git。
