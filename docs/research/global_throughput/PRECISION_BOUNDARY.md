# 第四轮输入精度边界

2026-10-05。当前结论是**精度边界已确认，两个后续路线均未采用，GPU仍暂停**。
本记录不改变harness、物理core、任何保存的数值输入或历史数值记录。
已有“FP64”标签记录了`jax_enable_x64=True`及主要state/forcing/parameter的双精度设置，
不能由该标签推断所有中间物理算术都为FP64。

## 冻结源码与实际输入

诊断时HEAD为`105f6b9913f7f5277f8a826e38d2732b4dc0e3b8`。
`git diff --name-only a33b82da9a1233f38e00d09435aecde4ba088111 -- jax_orchidee`
为空，`git ls-files --others --exclude-standard jax_orchidee`亦为空；
这项边界存在于已接受的冻结物理源码，早于第四轮新输入gate。
知识图服务Transport closed，worktree未索引；本核查采用精确源码fallback，
不声称完成全模型或全部Fortran分支的算术精度审计。

实际只读加载old local4、old remote4、当前local16与已完成local64，分别检查每条lane的
两个完整BlockArguments。全部176个lane/block具有同一非float64浮点schema：

| 零基group / BlockArguments字段 | 精确path | 原生dtype | 每点shape |
| --- | --- | --- | --- |
| 5 / landpoint_payload | `['contfrac']` | float32 (`<f4`) | `(1,)` |
| 5 / landpoint_payload | `['lalo']` | float32 (`<f4`) | `(1,2)` |
| 5 / landpoint_payload | `['lat']` | float32 (`<f4`) | `(1,1)` |
| 5 / landpoint_payload | `['lon']` | float32 (`<f4`) | `(1,1)` |

在上述完整输入trees中未发现其他非float64浮点leaf。
这些字段来自forcing/domain原生坐标与continent fraction，按原dtype保存；
不能通过cast改变旧数组，再把它们称为相同完整输入。
证据逐lane/block列表及每份prepared pickle SHA保存在
`outputs/research/global_throughput/scale-implementation-checks/precision-boundary.json`。

## 活跃solar计算边界

冻结源码链如下，行号指本次核读源码：

1. `driver/domain.py:786–795`从原forcing文件zoom lon/lat/contfrac并组装lalo；
   `driver/sechiba_boundary.py:178–181`将四个原生字段直接放入payload。
2. `driver/orchestration.py:10208–10209`提取compiled landpoint payload，
   `:6546–6552`在compact runtime step保留`payload.lalo`为metadata，
   `:10681–10684`将first-step metadata返回给compiled day。
3. `driver/orchestration.py:11110–11115`的现有`gdd_init_date`分支把
   `first_step_metadata.lalo[:,0]`送入`season_update_gdd_init_date`。
   old001实际保存的该字段存在且为float64；因此不能把latitude路径归为仅host metadata。
4. `stomate/season.py:227–234`用`jnp.asarray(latitude)`保留原float32，并将
   astronomy constants与julian day显式设为`latitude.dtype`；`:235–268`计算solar flux。
5. `stomate/season.py:278–287`将solar结果与已有float64 recurrent state比较、组合，
   最终更新后的`gdd_init_date`恢复为float64。这种最后的promotion不使上游solar计算成为FP64。

本地ORCJAX / JAX0.10.1、x64开启、CPU，直接用**原old001的lalo与gdd state**，
science day为int32的2，执行`jax.eval_shape`和`jax.make_jaxpr`：

| 边界 | 实际abstract dtype / shape |
| --- | --- |
| 原latitude输入 | float32 / `(1,)` |
| 原gdd_init_date state | float64 / `(1,2)` |
| `season_midwinter_solad`结果 | float32 / `(1,)` |
| `season_update_gdd_init_date`最终state | float64 / `(1,2)` |

solar JAXPR primitive outputs计数为float32 105、float64 2、int32 2、bool 1。
它是含FP32算术的既有计算，不能称整个solar子图统一FP64，也不称全部primitive统一FP32。
这里只做abstract tracing，没有执行完整物理forward或编译物理executable；
本地dtype证据不替代部署JAX0.4.38的运行时/数值验收。

`contfrac`还有不同的消费边界：`stomate/carbon_kernels.py:13406,13488`保留其原生dtype，
随后乘以float64 biomass/turnover/veget_max导出的litter flux；
`sechiba/main.py:641,659–663`亦与float64 vegetation fraction组合。
这种混合类型promotion与上述保留float32的solar子计算需要分别记录，不能合称四字段均先转FP64。

本计算的Fortran provenance取自冻结`season.py:219–224`注释：
`src_global/solar.f90::downward_solar_flux` 290–597，
`nband=1,rtime=12,cloud=0`；调用来自
`src_stomate/stomate_season.f90::season` 537–543。
本checkout及已检查的本地truth路径没有这两份Fortran文件；本次未独立重读Fortran原文件，
也未从JAX注释推断Fortran自身实际浮点kind。

## 两个可审阅路线，均未开始

**路线A：精确原生schema入场。** 保持冻结core和所有原始数值对象、dtype、shape及值。
仅在group5允许上表四条精确float32路径/shape，其他state/forcing/parameter浮点leaf仍要求
float64，离散dtype/None、所有lane的static/schema与原anchor完整输入身份仍严格核对。
公开精度描述须披露继承的native输入及solar FP32子表达式，不能称uniform FP64 arithmetic。
有界测试应包含：原四字段通过；其他group或未知path的float32、原字段float16/错误shape/
dtype变化、非有限值均拒绝；验证reader/gate前后原输入字节未改；实际local16/shared screen
通过后再按原parity/permutation/isolation政策验证same-workload比较。
这一路线只修正harness对既有输入schema的描述与检查，不修正物理计算。

**路线B：严格全FP64修复。** 先明确uniform FP64的计算边界，审计该solar路径及可能受影响
的cold Day1/recurrent state，提出有Fortran来源的精度修正并形成新的science/input身份。
不能只在reader中cast旧输入，或把现有独立references当新精度计算的等价证据。
需要独立own-context参考、受影响prepared states及完整两块输出重新验证，保留现有numerical/
sentinel/schema门槛，并核对solar结果与最终state的dtype；通过之后才测B16/B64。
这是科学数值执行条件的改变，其工作量和资源与路线A分开评估。

当前未选任何路线，harness的严格浮点gate维持原样，GPU未开始。
本记录只准备用户精度决策所需的具体证据，不把scope或精度选择自动定为已批准。

## 复现

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
conda run -n ORCJAX python outputs/research/global_throughput/scale-implementation-checks/check_precision_boundary.py
```

该脚本只读prepared inputs与冻结源码身份，并写同目录的`precision-boundary.json`；
没有物理core/harness修改、物理forward、GPU、reference重编译或容差变更。
