# CPU48 / FULL64 B64 preparation

更新：2026-10-11。本轮已完成可审阅的 runner、资源准入、scalar→batch
协调器、启动器、独立结果采集器和本地合成协议验证。真实 CPU48 编译、
物理数值、并行效率、吞吐与完整作业成本仍待测量；没有部署或提交作业。

主候选为一个原生 CPU JAX device、单进程 FULL64 B64 batch/vmap、48 个不同
物理核。一次 cnall/1node/1task/48CPU/noGPU 父 allocation 上限 90 分钟，先
运行完整 scalar 内部参考（上限 45 分钟），剩余候选窗口至少 30 分钟时再
运行 B64；两阶段均保存 5 core + 5 wrapper 样本。所有阶段按父 48 核计费：
主情景最高 CNY3.60，保守最高 CNY5.04。具体路径、命令、停止规则与批准
对象见 [PAID_PROPOSAL.md](PAID_PROPOSAL.md)。

## 验证边界

- [harness-validation.json](harness-validation.json)：91 个有效定向检查，含
  旧 CPU1 launcher 环境的真实 admission→benchmark identity→reference reader
  合成链。CPU48 仍严格绑定父 allocation，固定数值门保持。
- [checks.json](checks.json)：71 个定向检查通过，包含已有资源、版本、reader
  检查；独立 collector CLI fixture 4 个检查通过。这些计数相互重叠，不相加。
  成功路径实际读写 pickle/JSON/manifest，重算 760 个输出叶、64 个 GPU 对照门
  和一个完整 CPU 内部整组门；拒绝数组、身份与终态 accounting 篡改。
- 历史执行封装工具 7 个定向检查与 scoped Ruff 通过：临时 Git index 保留
  原研究 checkout，不把当前碳核心修复移入旧科学对照。最后仅将合成 fixture
  的 scalar wrapper median 改为与样本一致的 1.2；该测试数据调整未重跑，
  生产代码未随之变化。
- [target-readonly-preparation.json](target-readonly-preparation.json)：GNU
  Bash4.2.46、server bare Git 的历史 prerequisite、新 code-dir 当时不存在与
  logs 目录已存在均已只读核验。节点 ibc12b08n35 当时健康 IDLE、0/56 核、
  load0.01、ThreadsPerCore1；提交前须重新检查，这不是 reservation。

## 执行版本与科学对照

执行 commit 为 `4f6b2fc906c453b9030acff489637a8f83605dca`，ref
`codex/global-cpu48-b64-v1`；审核 research source 为
`2f8544d40cef306c4582f5ed58bd7ab7e6a8905f`。实际封装结果见
[execution-code-pin.json](execution-code-pin.json)。历史 base 固定为
`8aa2e05aa24b65dc4d30a27d4ee857bba6312221`，认证 science 为
`11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`。
仅移入 13 个显式审核的 harness/tests；104 个物理源码文件及整个
`jax_orchidee` 子树与 base 完全相同，当前 research 分支的碳修复保留。
本地 incremental bundle `cpu48-b64-4f6b2fc906c4.bundle`（39,681 bytes）已
通过 Git verify，server bare Git 已有其 prerequisite。原始 harness patch 和
bundle 保留在本目录且不加入 Git；执行 ref、manifest 和接口收据保留可恢复身份。

[固定版本接口收据](pinned-interface-review.json)记录实际 Git blob 经目标
GNU Bash4.2.46 stdin `--dry-run` 的完整展开：精确执行 SHA、绝对 code/cwd、
canonical Python、冻结输入/reference 和 5400/2700/1800 秒窗口均绑定。
随后本地实际 coordinator→scalar/batch runner CLI dry-run 通过，未导入 JAX、
未生成 run-root；仅为该本地检查替换 Python/runner 路径。执行 checkout 尚未
部署；目标 Python/JAX、实际 HEAD/clean 和 Slurm step 清理没有被此验证替代。
准备验收见 [preparation-acceptance.json](preparation-acceptance.json)。

冻结输入为 canonical `prepared_inputs_remote_64.pkl`，64 unique 点、连续
Days2–8/9–15、48 半小时步，共 896 point-days；760 完整 native 输出、masks、
dtype、`atol=1e-8` 和 `rtol=1e-10` 保持。与 GPU B64 job15579869 的同工作量
wrapper6.222332641482353s / 143.99744462818447 point-days/s 比较；GPU B1024
299.1398250995909 lane-days/s 仅作不同总量的归一化参考。

有限跨 GPU 容差失败可保留探索计时；CPU 内部整组、permutation、replacement/
isolation、reset、结构与非有限值门仍硬停止。完整成功仍是探索性能结果，
不自动成为 CPU 科学接受。费用采集协议见 [PROTOCOL.md](PROTOCOL.md)：
终态父 job accounting 未取得时，仅发布协调器覆盖区间估算；失败费用保留。
真实 Slurm step 清理、CPU48 内存/利用率、数值与速度均待执行环境验证。

本轮未运行远端 Python/JAX toy 或物理模型，未启动 GPU、安装环境、转运 bundle、
写服务器、sbatch 或自动任务；主管取得新的部署与付费授权后才进行实际运行。
图工具 transport 不可用，使用既定路径的精确源码 fallback，无图完整覆盖声明。
