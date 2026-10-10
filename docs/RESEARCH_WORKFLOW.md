# 主分支与研究工作流入口

`main` 负责已接受 Teacher 和项目管理；研究科学代码及状态留在各自分支。
本页是主分支导航，不复制日尺度历史或科学结论。
固定目录、阶段提交/同步、资产与恢复规范见
[WORKSPACE_MANAGEMENT.md](WORKSPACE_MANAGEMENT.md)；服务器入口见
[deployment-explore1000.md](deployment-explore1000.md)。

| 用途 | 本机 cwd | 分支 |
| --- | --- | --- |
| Teacher / 管理 | `E:/ORCHIDEE-MAN-JAX` | `main` |
| 日尺度神经研究 | `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining` | `research/daily-coarse-graining` |
| 全球物理吞吐 | `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput` | `research/global-throughput` |

日尺度从其五个当前入口读取：`docs/START_HERE.md`、`docs/NEXT_STEPS.md`、
`docs/CODE_MAP.md`、`docs/current-status.md`、`docs/DOCUMENT_STATUS.md`。
详见其[研究工作流](../runtime/worktrees/daily-coarse-graining/docs/RESEARCH_WORKFLOW.md)；
只按具体问题读取命名历史报告，不把 chronology 当作当前运行授权。

全球研究从其 `docs/research/global_throughput/README.md` 读取当前工作单，
按需要阅读 `EXECUTION_AUDIT.md`、`SERVER_BENCHMARK.md`、
`PRECISION_BOUNDARY.md`、`FROZEN_ARGUMENTS_ADMISSION.md`；
这些文件的责任与日尺度五入口不同，科学状态由该分支自己维护。
其[实际入口](../runtime/worktrees/global-throughput/docs/research/global_throughput/README.md)
和命名合同给出证据、范围及授权边界。

每个聊天固定 cwd，编辑前核对 Git top-level、branch 和 Python import 路径。
已有聊天的 cwd 未被本轮维护修改；不要在根目录继续写研究分支的代码。
每个完整阶段审查并按明确路径提交，再普通 push；交接记录完整 SHA、dirty paths
和科学/资源停止边界。研究结果不自动合并 main，分支推送不授权新实验。
本地工作树链接依赖上述布局；其他 clone 应进入对应分支读取原相对路径。
