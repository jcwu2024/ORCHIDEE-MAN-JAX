# Runtime 与共享资产

根 `runtime/` 保存本地传输资产和两个固定研究工作树；它不是可随意清空的缓存。
`worktrees/daily-coarse-graining` / `worktrees/global-throughput` 使用各自研究分支；
`worktrees/main-teacher-sync` 是保留的 detached 历史快照。它们共享根 `.git` 对象。
根 main 不承担这两个研究线程的科学编辑。

大型 data、reference、Fortran、Teacher 和历史输出保持原位置。daily 的部分 ignored
目录为 junction，部分文件为 hardlink；经链接操作影响同一资产，不能当独立副本删除
或改写。缓存不共享，也没有把根 runtime 链接进其后代工作树的循环。
具体路径和恢复见[工作目录管理](../docs/WORKSPACE_MANAGEMENT.md)。

服务器 runtime 独立位于 `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime`，
统一入口为其 `deployment/README.md` / `registry.json` / `repository.git`。
服务端旧 execution checkout 和 overlay 不随开发镜像 refs 更新。
按[部署入口](../docs/deployment-explore1000.md)声明准确版本并验收，
没有本地↔服务器目录自动镜像或自动新建执行树。
