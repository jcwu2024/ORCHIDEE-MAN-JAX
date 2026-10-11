# 碳核心同步：本地有界验收

2026-10-11，Asia/Shanghai。**生命周期补丁已提交；main 的两项碳核心修复已依次 cherry-pick，本地新增37项回归及354/354个完整返回叶子记录通过。** 348个记录完全相同，六个 turnover leaf_frac 记录有最大绝对差3.33e-16的舍入差异，全部57个碳库/turnover记录精确相同。本结论限下述小型生产适配器边界，不授予完整全球运行或GPU科学准入。

实际cwd、Git top-level为 `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput`，分支 `research/global-throughput`。实际本地导入为该树的 `jax_orchidee/__init__.py`。本轮只执行本地有界检查和普通Git同步；未SSH、部署、提交作业、编译完整物理图或读取FULL64大pickle，旧额度保持关闭。

## 版本与实际改动

| 阶段 | 完整commit SHA | 范围 |
| --- | --- | --- |
| 原开发HEAD | `52326bd5e67629269a4a4a1c39885276eefd5ec8` | 本轮生命周期实现的起点 |
| 生命周期检查点 / 同步前科学基线 | `277aa35793e5b6dd3a1d436a32b4cbc97a305106` | 五个已验收文件；复用此前199项轻量测试 |
| NPP修复 | `c6a88afd41120dfd6434bdf5d442b72b58b34ccd` | `cherry-pick -x e317fa81b97b8b8c959f2e7dfdfba525bb54dec8` |
| leaf fraction修复 / 本轮科学测试pin | `ef271cef8551f47f82cccf234949d021e958917e` | `cherry-pick -x 2a3e303845154576c6f4b5c0b8155106e5bcc70b` |

两次pick无冲突，独立提交保留来源。NPP maintenance fraction采用已有稳定比值AD规则；leaf fraction在可表示的正叶质量分支以顺序存活因子消去共同质量尺度，原物理stock/turnover更新顺序保留，原阈值以下策略保留。没有额外物理代码修改。原Fortran provenance及导入回归仍随源提交保留。

主管复核了stable patch ID；本聊天复核最终 carbon_kernels.py 的Git blob为 `11cf1dc18bf27fc6780697aec3935a90ba548624`，与main叶分数修复相同；导入NPP回归文件blob为 `8bf6aa239606c7838e18eb5f1aa8faa8a99b1d6f`，与来源提交相同。最终记录提交可由本报告的Git历史定位；上述科学测试pin与未来GPU执行pin分开。

## 实际正向比较

[复现脚本](../../../../scripts/dev/verify_global_carbon_core_sync.py)通过Git-backed importer，在两个独立子进程中加载各自精确pin的全部已导入 jax_orchidee 模块；不复制baseline工作树，也不借用HEAD科学模块。相同fixture SHA256为 `8860b5f54e2b62e3d3255062043143a48962179adb054285a47a3b68f9f39ca3`。

已执行命令（exit 0）：

```powershell
conda run -n ORCJAX python -B scripts/dev/verify_global_carbon_core_sync.py --baseline 277aa35793e5b6dd3a1d436a32b4cbc97a305106 --candidate ef271cef8551f47f82cccf234949d021e958917e --output-dir outputs/research/global_throughput/carbon-core-sync-20261011
```

复现须改用新的output目录，保留本轮原始证据。运行环境为ORCJAX Python3.11.15、NumPy2.4.6、JAX/jaxlib0.10.1、CPU、x64开启。每个entry实际覆盖scalar eager、scalar JIT、vmapped JIT三种模式。

| 实际生产entry | 跨版本返回叶子记录 | 结果 |
| --- | ---: | --- |
| integration.stomate_daily_carbon_explicit | 78 | 全部精确相同 |
| integration.stomate_daily_carbon_kill_gap_turnover_explicit | 243 | 全部通过；仅三个turnover.leaf_frac记录有舍入差 |
| carbon_kernels.turnover_step | 33 | 全部通过；仅三个leaf_frac记录有舍入差 |

14个fixture为ordinary_growth、ordinary_pumping、tiny_active_mass、tiny_active_maintenance、zero_mass、negative_mass、below_cutoff、cutoff_previous、cutoff_exact、cutoff_next、absent_pft、laidev_mask、bare_soil、natural_grass。显式边界为npts1、14 PFT、12 parts、一个element、四个leaf ages，外层vmap14 lanes；lpj_gap_const_mort=True、ok_dgvm=False、cover/vmax关闭、residence_time=正无穷（零gap mortality）、min_stomate=0。该输入在allocation之后，不代表全分支覆盖。

固定 `atol=1e-8, rtol=1e-10`，离散/text精确、shape/dtype及完整PyTree结构一致；None和空结构保留，requires_trace字符串作为static PyTree metadata保留并精确比较。stock字段即使在容差内有变化也会拒绝。全部数值输出有限；84个离散/text记录和57个stock记录全部精确相同。

六个差异记录均为float64 leaf_frac，shape `[14,1,14,4]`：最大绝对差 `3.3306690738754696e-16`；非零baseline坐标最大相对差 `4.803236536610133e-16`；最大原容差bound ratio `3.300198168921189e-08`。leaf_frac记录累计baseline零坐标13176，变化的零坐标为0。除leaf_frac外全部返回记录精确相同。完整指标见[comparison.json](comparison.json)。

初次脚本准备修正了静态text的JIT表示和fixture的IROOT=5、IFRUIT=6索引，再完成上述成功运行；这些是验证工具准备修正，不是物理代码更改或数值gate失败。之后的报告元数据补充复用已完成workers，没有重复编译未改变的科学边界。

## 全球生产来源与边界

精确Git AST、行号、条件runner绑定及route kwargs见[source-callchain.json](source-callchain.json)。实际静态调用路径为：

```text
global_batch.make_batch_transition
  -> make_scalar_transition
  -> orchestration._paper_later_day_block_transition_factory
  -> paper_1961_driver_later_day_runtime_result
  -> _paper_day_stomate_daily_carbon_from_bundles
  -> 条件选择outer_compiled / static_jit / explicit runner
  -> integration._daily_carbon_static_dispatch（outer_compiled分支）
  -> stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit
  -> stomate_daily_carbon_kill_gap_turnover_explicit
  -> stomate_daily_carbon_explicit / npp_closed_update
     与 turnover_step / turnover_leaf_age_fall
```

global_batch的batch/scalar调用点为325/312；orchestration block factory→runtime→bundle runner调用点为11232/11072/7627；7620–7626保存条件runner绑定；integration outer→static dispatch→explicit调用点为3548/3474；post-NPP适配器的NPP和turnover调用点为2425/2588，NPP适配器kernel调用点为1702。block factory直接调用runtime，没有额外complete-day factory。

上述链路是静态来源证据。数值脚本只执行表中三个现有较低生产entry，没有执行prescribe/allocation、外层driver、完整日或多日图。图工具list_projects/index_status/search_graph/check_index_coverage均曾返回Transport closed；收尾时再次list_projects仍失败。上次已知图generation为2026-10-10T08:46:50Z，属于main，global工作树曾excluded/not_tracked；不能作为当前索引完整性证明。本轮使用精确Git源码回退，不重建或重索引。

## 新增回归与验证记录

实际新增37项通过：34项focused回归耗时17.61s（11 comparator失败门、8 turnover conditioning、10 NPP/turnover kernel、5 integration），再加3项全球factory/stack/vmap薄层测试耗时2.45s。后者使用现有test doubles，没有编译真实完整日图。199项生命周期测试是同步前已完成证据，未重新执行，也不算作新核心数值验收。

完整实际命令、环境和选择见[validation.json](validation.json)，原始pytest结果见[focused-regressions.log](focused-regressions.log)与[global-entry-regressions.log](global-entry-regressions.log)。Ruff check、Ruff format check及diff检查通过。关闭pytest cacheprovider后只出现既有cache_dir配置warning，无数值warning。收尾只复核工件及以下已测试源码hash，不机械重跑。

| 已测试文件 | SHA256 |
| --- | --- |
| scripts/dev/verify_global_carbon_core_sync.py | `6392423d2d2783c6cf378fa5e882823d116f9429b09bc88654b712fa31602bfb` |
| tests/unit/test_global_carbon_core_sync.py | `43f49e00319926d5508e0386afc2b0c6faa45071448ff39f0136979fc676396a` |

## 来源准入与下一步

独立实际调用 `global_source_compatibility.source_identity(Path.cwd())` 在科学测试pin退出1，helper第53行抛出 `ValueError: science source changed outside certified execution-only cache repair`；命令与分类见[source-guard-check.json](source-guard-check.json)。这是预期的科学版本隔离拒绝，与上述通过的数值比较分开。旧DIFFUCO-only execution certificate及helper原样保留；未把碳过程修复归入execution-only equivalence，未创建新证书或绕过检查。

GPU历史fixed1024/fixed2048的pin `bd9e4c04af68d77e79b984bfc8aa064977f893d3` / `8aa2e05aa24b65dc4d30a27d4ee857bba6312221`、冻结FULL64 goldens及旧结果继续保留，其验收不能转授新核心。本地JAX0.10.1 CPU结果不能替代canonical JAX0.4.38 CUDA、新核心FULL64严格比较、两块连续窗口、异质全球输入或长期科学验收。

[GPU验证提案](../memory-lifecycle-implementation-20261011/GPU_VALIDATION_PROPOSAL.md)已更新同步lineage与本轮边界，仍未就绪、未执行。后续准备需要独立新科学核心identity和对旧冻结FULL64的显式严格comparison admission、同一新核心/pin的resident/release两臂、专用两组launcher、strategy-aware saved-array validator及目标Bash参数展开dry-run。新完整全球runner准入和GPU证据均未完成；拟1 V10032GB+4 CPUs、120min、CNY4.80/4.96仍只是未批准提案。

本轮按具体文件保存脚本、测试、交接、提案及必要小报告/JSON/log；NPZ数组、各worker原始文件及既有大结果继续保留在本地ignored输出中。最终commit、普通push及远端核对收据另存本目录VERSION_SYNC_RECEIPT.json，避免把记录提交误作科学测试或GPU执行pin。

自动审批复核拒绝了两次经过路径核验的原生PowerShell pytest TEMP清理，返回理由仅为“blocked by policy”。`C:/Users/admin/AppData/Local/Temp/orcjax-carbon-sync-20261011`因此保留；未换机制绕过，不影响本地验收或Git同步。
