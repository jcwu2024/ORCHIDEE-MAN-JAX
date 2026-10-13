# CPU48 v3 Terminal Evidence Manifest

Local staging root: `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012`.
The remote run root was
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467`.
Run outputs were copied read-only after job `15820467` reached terminal state.

| Staged path | Bytes | SHA-256 | Contents |
|---|---:|---|---|
| `15820467/orchestration.json` | 12003 | `b50a0f5645509dd54c281bf4ca94ae0c36499cf4941b5d2116a283c437ffd478` | Terminal coordinator state and step accounting |
| `15820467/slurm-15820467.txt` | 12003 | `b50a0f5645509dd54c281bf4ca94ae0c36499cf4941b5d2116a283c437ffd478` | Slurm job stdout; byte-identical to the terminal orchestration capture |
| `15820467/batch.log` | 4440 | `fa4ba1c7ffec87986338cd8199f6784989a75d59ebbc06f7a4fdb867a5397ab6` | Pre-JAX batch-step failure and launcher timing output |
| `15820467/scalar/report.json` | 1029739 | `81bd4ceb413c079eb8f1fa7c00c61f4a8d74ddb75278d2a49bd1e5fd82db4da0` | Completed scalar gates, timings, signatures, and runtime measurements |
| `15820467/scalar/stage_manifest.json` | 265095 | `269c4e04b2404177dbb0c2b350673be0f41fa9a63dbfa5055af35f2d9550a641` | Scalar stage identity and checkpoint hashes |
| `15820467/scalar/events.jsonl` | 922 | `78c2d813bef1a489ba3cd4e68e0273dbccd9f326942b63b1a18a5353d5508833` | Scalar runtime events |
| `15820467/scalar.log` | 4135 | `64dea6f2662cc1caa3ff07b9c286108b5a8710435688b0d99c2578857d586422` | Scalar command output and `/usr/bin/time -v` |
| `15820467/sacct-all-steps-15820467.terminal.txt` | 538 | `62ab527cc455c3d597526322364fafe2002847cf3d181c1a3c4e42c6efa23ee9` | Terminal parent and step rows with exit, elapsed, CPU, node, TRES, CPU time, and MaxRSS |
| `15820467/sacct-parent-15820467.terminal.txt` | 85 | `e051d4e11cf52baa5cf82b79041a5b4d6a03b50ccfd683e31ea2698c66f2e7bb` | Terminal `sacct -X` parent-only row |
| `15820467/parent-accounting.json` | 713 | `a04c687530f390ad1d5f1d3668293a552b3a9b37f0cd05ef1f264660c2f37113` | Collector accounting record bound to the terminal parent row |
| `15820467/collector-result.json` | 6801 | `fb7c6e5baa0ba87765fe1660dad65161331c3c87aaccaaf6db327ac44e21c624` | Bounded test01 failure-preserving collection result |
| `15820467/scontrol-terminal-15820467.txt` | 49 | `5a9d81a48db8412ccbdbbbf6d72e0fe0bfdd93adc78b9b58b7f004afc3febfc0` | Post-terminal `scontrol` response; job record had expired |

The parent-only raw row is
`15820467|FAILED|00:36:37|billing=48,cpu=48,node=1\n`; its UTF-8 SHA-256 is
`e059a19bb41ff88ee685de0e76a13727f9861039c6373832e320dad541321bd6`. The
parent `AllocTRES` from the terminal query does not include `mem=1M`; the full
terminal rows are retained above. Earlier live scheduler and clock receipts
remain in this directory, including
`clock-and-job-snapshot-20261013T030949Z.txt` and
`clock-and-job-snapshot-20261013T031850Z.txt`.

This compact index contains no serialized payloads, execution bundle,
environment dump, or credentials. The requested scalar checkpoint files were
preserved separately under `15820467/scalar/`; their content hashes are
recorded in `15820467/scalar/stage_manifest.json`.
