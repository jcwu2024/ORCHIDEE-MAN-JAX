# Explore1000 项目部署入口

本页是 `main` 的共享管理入口，不是新科学运行或付费作业授权。
项目服务端统一入口为
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/deployment/README.md`；
其 `registry.json` 登记开发分支镜像、实际执行 checkout、source overlays、数据、
结果、环境和共享 Git 对象依赖。服务器维护副本可在本地
[部署 README](../outputs/maintenance/20261010/server/README.md) 和
[registry](../outputs/maintenance/20261010/server/registry.json) 核查；
副本属于一次验收证据，服务器现存登记才是后续部署状态。

## 版本同步与科学执行分离

统一 bare 分支镜像为
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/deployment/repository.git`。
先声明完整 commit SHA，核验 Git bundle 的 checksum、refs 和完整历史，再无 force
导入、读回 refs 并更新登记。镜像同步不改变实际执行目录，不需要为每次管理更新
新建代码副本。服务器 Git 1.8.3.1 没有 `git -C` / `git worktree`；使用部署 README
中的 `git --git-dir=...` 命令，不把本地现代 Git 工作树命令直接搬到服务器。

保留旧 execution checkouts、已批准 source overlays、成功 executables、科学数据、
结果、环境和 Git alternate 依赖。记录实际执行 SHA 与 overlay 身份，不用今天的
分支 HEAD 隐式替代历史运行源码。运行中或已排队的代码保持不变；需要新执行版本时
先按研究合同声明目录和版本。任何代码归档/回收都先检查依赖、活跃任务和可恢复档案。

## 两条研究线的完整合同

- 日尺度：在 `research/daily-coarse-graining` 固定 cwd 阅读其
  [完整部署规范](../runtime/worktrees/daily-coarse-graining/docs/deployment-explore1000.md)、
  `docs/NEXT_STEPS.md` 和命名实验合同。
- 全球物理吞吐：在 `research/global-throughput` 固定 cwd 阅读其
  [研究入口](../runtime/worktrees/global-throughput/docs/research/global_throughput/README.md)、
  [服务器基准合同](../runtime/worktrees/global-throughput/docs/research/global_throughput/SERVER_BENCHMARK.md)
  和该入口指定的当前协议/执行账本。该分支没有同名 `deployment-explore1000.md`，
  不引用不存在的文件，也不继承日尺度的特定实验授权。

以上工作树链接用于本机布局；其他 clone 应进入对应 Git 分支查阅同一相对文件。
科学结论、具体启动参数和资源额度保留在研究分支，本页不复制或更新这些状态。

## 环境与权限

仅写 `/WORK/liwei_work/jcwu/`。全部项目 runtime 文件保持在
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime`；复用 canonical 环境
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu` 和已接受的
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu`，不擅自重建环境。
`cln01` / `cln02` 仅作连接、传输和调度入口，按连接质量选择并复用 SSH 身份；
不在登录节点计算。小 CPU/GPU 检查使用既定 test 节点/gln01。
paid `cnall` / `cnmix` / `gnall` 提交仍须明确批准资源、时限、cwd、命令和最坏费用。
进一步操作遵守本地 AGENTS 与 `explore1000-job-script` 技能。
