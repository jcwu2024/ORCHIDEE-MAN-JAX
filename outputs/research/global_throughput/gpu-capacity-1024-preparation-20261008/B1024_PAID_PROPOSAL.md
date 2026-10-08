# B1024 single-card continuation proposal

Status: approved single submission, job15662711 RUNNING as observed at2026-10-08T17:02:31Z (2026-10-09 Asia/Shanghai). The user approved this exact node/resource/command/code/cost envelope; submitted once from gln01 via192.168.11.2. No monitor is created and earlier budgets remain closed. [Submission receipt](submission-receipt.json) retains preflight, initial scheduler accounting and startup evidence. No complete stage or terminal cost is yet validated.

B512 is the best measured single-card capacity point so far, not a confirmed optimum. The next comparison measures the exact same ordered fixed1024 workload in one new process and allocation: B512 as two serial chunks, followed by B1024 as one chunk. Each collection advances all 1024 lanes continuously through Days2–8 and then9–15, 48 half-hour steps/day, for 14336 lane-days. These are the original 64 real inputs repeated16 times, not 1024 unique points or independent ensemble members. Native dtypes, all outputs/masks, physical processes and the fixed numerical tolerances remain unchanged.

The accepted job15657822 B512 stage remains immutable historical evidence. Its exact checkpoint, FULL64 numerical outputs, compiler buffers and cumulative process RSS can support admission. Its old 7168-lane-day timing is not doubled or relabeled as a new fixed1024 sample. The old executable was process-local and persistent compilation cache was off, so a fresh B512 compile is needed for a real fixed1024 control; only B512 and B1024 compile. No B64/B128/B256 baseline compile, input regeneration, training or AD is part of this proposal.

## Requested allocation

| Item | Proposed value |
| --- | --- |
| Partition | `gnall` |
| Nodes / tasks | 1 / 1 |
| Auxiliary allocated Slurm CPUs | 4; not a claim of 4 physical cores |
| GPUs | 1 Tesla V100 32GB |
| Hard wall limit | 01:30:00 (5400s) |
| Approved and actual node | `ibc13b03n05` |
| Code checkout | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-fixed1024-v1` |
| Output root | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1` |
| Scheduler working directory | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX` |
| Code SHA | `bd9e4c04af68d77e79b984bfc8aa064977f893d3` |

The checkout is a separately deployed clean immutable source snapshot; [deployment receipt](deployment-receipt.json) records exact execution pin, independent Git objects and unchanged clean accepted source. The [final code pin](final-code-pin.json) binds this execution commit, source-file digests and the verified local [incremental Git bundle](fixed1024-bd9e4c0.bundle), 273676 bytes, SHA256 `a84da4a91faab7ca2925e7aaabe6815654f1eca9feb4467dc6898bf94b36113d`. The bundle requires preserved accepted commit `ad6c0e1faa7e22b590627844e9cb0ad492e9deb3`; it is sufficient for a new checkout cloned from that immutable B512 repository. A later documentation-only commit is not the execution-code revision. Deployment and actual-target clean/source/interface verification passed on test02, with no JAX import, model work or GPU initialization. The new output root remained absent; recheck it before any authorized submission. A failed or partial experiment preserves complete stages and failure evidence without retry, requeue, time extension, cancellation or another submission.

The 2026-10-08T15:40:15Z [scheduler snapshot](node-snapshot.txt) shows no idle eligible gnall node. Candidate `ibc13b03n05` was MIXED, CPU96/112 allocated, CPULoad19.37, GPU0/8 allocated, reported FreeMem337213MB. Among eligible nodes with unallocated GPUs and at least4 free CPUs, it had lower observed load than `ibc13b02n02` (load23.78), `ibc13b03n01` (GPU7/8 allocated, CPU100/112 allocated, load32.19) and `ibc13b03n04` (load48.44). Lower-load `ibc13b03n03` and `ibc13b03n07` each had all8 GPUs allocated; `ibc13b03n02` had no unallocated CPUs. All eligible nodes were MIXED or ALLOCATED; drained/down nodes were excluded. This is a scheduler snapshot, not a reservation or an idle-node claim. The final17:01:13UTC [submission preflight](submission-preflight.txt) rechecked approved n05 as MIXED,CPU96/112,load19.69,GPU0/8 allocated,FreeMem336420MB; exact checkout was clean, run root absent and no user jobs were queued. The approved node and envelope were retained. At startup retain actual node, allocation and affinity/SMT mapping, host load, selected physical GPU UUID/model/driver and selected-card process inventory. This snapshot cannot attribute any historical slowdown to contention.

## Cost ceiling

GPU1 ×1.5h ×CNY2.20 = CNY3.30. CPU4 ×1.5h ×CNY0.05 = CNY0.30: primary maximum **CNY3.60**. With CPU CNY0.07, the CPU component is CNY0.42 and the conservative additive maximum is **CNY3.72**. The ceiling is 1.5 GPU card-hours and6 allocated CPU-hours. These are unverified billing scenarios and requested-time ceilings, not final money or runtime consumed; execution has started, final parent accounting remains pending and queue time is unbilled. Terminal accounting will count the sacct parent once, retaining but not summing .batch/.extern/step rows.

## Resource and stage guards

The old B512 job measured lower98.739s, compile892.739s and parent1416s. New B512 fixed1024 and B1024 compile/execution times are unknown. The 90-minute ceiling accommodates two compiles without asserting an expected finishing time.

Require at least2700s remaining before starting a shape; B1024 starts only after the new B512 fixed1024 stage is atomically sealed and numerical/resource admission passes. There is no speedup prerequisite for attempting B1024. If the time or memory guard denies B1024, preserve B512 and report B1024 unmeasured.

[Planning arithmetic](resource-estimate.json) uses the exact accepted B512 stage: 1967886392 resident input bytes for its fixed512 collection, 220209152 full two-block output bytes, 2288514256 compiler temporary bytes, and19496943616 cumulative host peak RSS bytes. Doubling the full collection gives an upper resident estimate3935772784 bytes and440418304 output bytes. The device planning formula is `2*resident + 3*full-output + max linearly scaled actual compiler temporary`, then2GiB physical slack. It requires about12.69GiB free for B512×2 and14.82GiB free for B1024×1. Runtime guards use actual stacked bytes for each shape and the newest admitted actual compiler analysis. Doubling the historical cumulative host peak plus8GiB gives47583821824 bytes (44.32GiB); actual MemAvailable must pass the host guard. Shared static groups make the doubled resident value a conservative planning bound, not an observed fixed1024 allocation.

For B1024, the host guard uses the larger of that historical projection and twice the new fixed1024 B512 cumulative process peak, then adds8GiB. The outer stage decision may use a conservative repeated-leaf bound, while the immediate compilation guard checks actual stacked bytes; neither is a measured fixed1024 peak.

Compiler buffer analysis is not measured VRAM peak, RSS is host memory, and snapshots are not peak evidence. These guards do not guarantee no OOM or completion. No large-model B1024 lowering/compilation diagnostic is run during preparation. Two full consolidated checkpoints are about0.9GB total, with an additional transient atomic-write container of about0.45GB; retain existing inputs/checkpoints and use bounded reports/manifests, not per-lane or per-variable files.

## Proposed command

The user approved the following complete allocation, command, directory and worst cost. It was executed once as job15662711; this is a record, not an instruction to resubmit:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
/rmprog/slurm/v22.05.7/bin/sbatch --nodelist=ibc13b03n05 --export=ALL,CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-fixed1024-v1,RUN_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-fixed1024-v1/scripts/hpc/slurm_global_gpu_capacity_1024.sh
```

The wrapper resolves helpers from declared CODE_DIR after Slurm spooling. The [verification](verification.json) records 159 passed lightweight tests and the [small canonical helper probe](canonical-helper-probe.json). The [actual direct/spooled argument expansion](canonical-launcher-interfaces.json) passed in a PROJECT-relocated source fixture; its recorded CRLF-to-LF shell conversion is bound to the committed Linux shell bytes in the final code pin. The [actual deployed interface receipt](deployment-interface-receipt.json) now supplies deployment admission: exact clean SHA and four committed interface hashes, real CLI help/dry-run, direct/spooled command equality, existing source certificate and deployed collector code-pin admission all pass. Its10 subprocess checks took3.206605205s on test02 affinity0–3/BLAS1, without model work or GPU initialization. The relocated fixture remains historical evidence. The separate human approval authorized the single recorded invocation; no retry, cancellation, requeue, extension or additional job is authorized.

## Collection and interpretation

A new explicit fixed1024 collector must validate the exact complete checkpoint SHA, pinned execution code/source certificate, original FULL64/input identity, runtime/allocation/physical GPU origin, native 760-leaf two-block schema, all1024 lane comparisons, three internal gates and five synchronized core and prepared-wrapper samples. Historical fixed512 collectors cannot silently accept this contract. Report both stage medians/full ranges and cost per lane-day on the common14336 denominator; keep lower/compile, first call, stack/transfer, checkpoint I/O and complete job cost separate.

B1024 may improve, plateau or regress; the outcome is unknown. The new comparison informs the lowest measured warm unit cost for this repeated-input capacity curve. It does not establish the optimum for independent ensemble members or production IO. No K-reuse estimate is justified without observed full stage setup. There is no new CPU/GPU ratio, Fortran comparison, annual/real669-point/50-year result,10× result, training/AD or general scientific acceptance.
