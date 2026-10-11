# 已保存初始数组的失败定位（2026-10-11）

已完成并取回的比较将全部失败定位到真实点 `283.0-091.0`（零基 source index `45`）的 `diffuco_previous_step_state.rveget`：第一块 Days 2–8 的 **Day 8 结束状态**，零基内部坐标 `[0,13]`，即 inner npts `0`、PFT 轴索引 `13`（PFT14）。这不是逐日观测输出。

原生树路径为 `[0][0][0][7]`，leaf index `7`，component index `0`、field index `7`；actual/reference 均为 `float64`，形状分别为 `[2048,1,14]` 与 `[64,1,14]`。首个失败的完整零基坐标为 `[45,0,13]`。

| 量 | 已保存比较的值 |
| --- | --- |
| actual | `515.7648516411803` |
| frozen reference | `515.7648515535959` |
| actual − reference | `8.758433978073299e-08` |
| absolute delta | `8.758433978073299e-08` |
| 固定 bound | `6.157648515535959e-08` |
| absolute delta / bound | `1.4223666641536081` |

这些值全部有限。失败原因是既定浮点 tolerance 门，而不是离散 exact 门或非有限值；门值保持 `atol=1e-8`、`rtol=1e-10`，`bound = atol + rtol × abs(reference)`。

全部 32 个失败 lane 为 `45 + 64*k`，`k=0..31`，范围 `45..2029`：

`45, 109, 173, 237, 301, 365, 429, 493, 557, 621, 685, 749, 813, 877, 941, 1005, 1069, 1133, 1197, 1261, 1325, 1389, 1453, 1517, 1581, 1645, 1709, 1773, 1837, 1901, 1965, 2029`。

每个 lane 的来源都是 source index `45`、同一个真实点，repeat index 为 `0..31`。已完成的数组比较逐字节验证这些失败坐标及其 actual/reference 值在重复 lane 间完全一致；gzip 将其表示为一个 exact-repeat group、一个内部坐标记录，即 `1 coordinate × 32 repeats = 32 native failing coordinates`。本文整理返回的协议证据，没有再次加载原始数组独立验证重复性。

比较明确检查了 **760 leaves × 2048 lanes**，仅一个 failing leaf、一个 failing unique point、一个 failure group。其余 759 leaves 及其他 63 个真实点在这一次保存数组比较中没有门失败；声明的 bare-soil roughheight sentinel 共 8192 个坐标。这一范围不构成全分支科学正确性、旧科学输出等价、GPU 性能或科学验收。

直接证据为 [summary.json](saved-array-explanation/comparison/summary.json) 和 [all-failing-coordinates.jsonl.gz](saved-array-explanation/comparison/all-failing-coordinates.jsonl.gz)。gzip 有三个记录，依次为 `leaf_failure`、`exact_repeat_group`、`failed_coordinate`，压缩文件为 1327 bytes，SHA-256 为 `4c6b7916acbe2979e55026dea0665542dfd3c0e5f0ce01680739eb02cfe6e8e2`。

字段位置沿用 [state-layout.json](state-layout.json) 的 canonical prepared-state 插入顺序绑定；其 SHA-256 为 `8fcdaa71441ac5e9c193f54801bd2e583290ff544c90220aab620e5c2f944a89`。已有精确 pin 源码证据包括 `fast_state.py::fast_state_from_previous_packet`（143–169，顺序序列化）、`orchestration.py::_add_diffuco_local_to_previous_fields`（2870–2879，直接保存 rveget）及 `_paper_later_day_block_transition_factory`（11185–11307，块结束状态）。PFT14 坐标含义采用本轮已经接受的映射证据，未启动新的源码溯查。

| 保留的身份绑定 | pin / SHA-256 |
| --- | --- |
| 原执行 code pin | `3b4ae35e695f85480ecb72461e4ba93f4534bbd8` |
| declared science source pin | `1fdacde59d86f696f384b56d69292c8f6fa8d30b` |
| actual declared science source digest | `8c8b1dfd0b207e96bbb81c2d3eb85ae6edccb9ed3c6b77fb470dabc58c5fa77d` |
| frozen old-science source digest | `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5` |
| failed initial pickle | `698d3dece201424778d9cf4b0f10c1ac3b1802a1ce35b7d231f73b751641ddc5` |
| original job report | `672f5a990dfc5d541d65bbe1f054030f82c300a06c81d0e3f4c3ca9ed67f9ca0` |
| frozen reference pickle | `a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6` |
| frozen reference manifest | `cce5ed91f94c19cd576856b223e63830f91e3e0fa6def046d786e013317f8319` |
| canonical prepared pickle | `be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7` |
| canonical source metadata audit | `259ddf5358c69b3d01b05924c17b709764f95a8ceebdfa993f00c39df01d0180` |

已完成的解释运行在免费 `ibtest03`、单 CPU 上，耗时 `22.631719475612044 s`，peak RSS `1008807936 bytes`；使用 Python `3.11.15`、NumPy `2.2.6`，未导入 JAX、jaxlib 或模型，未执行模型/JIT/GPU。本轮后续仅整理这些已取回证据。

**本轮科学执行关闭。** 本文只定位新科学保存数组相对冻结旧科学 reference 的固定门失败；原因尚未归属，这份数值差异定位不构成旧 ABI 故障证据。原执行 pin `3b4ae35`、冻结 reference 和旧版本保留，不作科学修复、容差调整或重跑。新增 `verify-saved-explanation.py` 与 `EXPLANATION_VERIFICATION.json` 未创建、未执行；没有新增独立验证收据。
