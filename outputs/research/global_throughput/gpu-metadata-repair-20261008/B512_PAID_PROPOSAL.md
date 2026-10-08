# Separate paid allocation: B512 only, completed and collected

Status: **explicitly approved, submitted exactly once, completed and collected**, job15657822, from gln01 through login endpoint192.168.11.2. The [submission receipt](submission-receipt.txt) and [initial status](job-15657822-initial-status.txt) record initial PENDING (Priority) at2026-10-08 12:51:09UTC, with ReqNodeList=ibc13b03n02. It completed0:0 with1416s parent elapsed, actual node `ibc13b03n02`, 4 allocated CPUs,1 V100, and no requeue. Execution was20:57:08–21:20:44 cluster local on2026-10-08 (12:57:08–13:20:44UTC); queue time was unbilled. The strict collector admitted1 stage, withheld0 and exited0. Accepted deployed code SHA is `ad6c0e1faa7e22b590627844e9cb0ad492e9deb3`; the checkout was clean before/after all9 actual declared-checkout checks. See [deployment receipt](deployment-receipt.json) and [CLI/Bash/spool receipt](deployment-interface-receipt.json). Those interface checks did not import JAX or perform model preparation/lowering/compilation/invocation or GPU initialization.

Deployed Linux actual full-source SHA is `b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8`, with active exact execution-only cache certificate and admitted baseline science SHA `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`. Native-output equivalence and canonical real lowering evidence are described in [current result](CURRENT_RESULT.md); the repaired GPU executable also completed the bounded B512 compile/run and strict saved-stage admission below.

Resources: Explore1000 `gnall`,1 node,1 task,4 auxiliary CPUs,1 V100,45-minute hard wall (`00:45:00`), no requeue. Maximum0.75GPU card-hours and3CPU-hours. Additive approved ceiling wasCNY1.80 atGPU2.20/card-hour+CPU0.05/core-hour, orCNY1.86 withCPU0.07/core-hour. At1416s parent elapsed, planning cost isCNY0.944 primary/CNY0.975466667 conservative; billing is not verified.

Exact scheduler working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.
New immutable `CODE_DIR`: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-metadata-v1`.
Actual run root: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-capacity-512-15657822`. The launcher rejects an existing run root at launch.
Actual capacity destination: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-capacity-512-15657822/capacity`.
Actual scheduler log: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_capacity_512_15657822.txt`.

Recorded approved submission command, **already executed once; no automatic retry**:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
/rmprog/slurm/v22.05.7/bin/sbatch --nodelist=ibc13b03n02 --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX --export=ALL,CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-metadata-v1 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-metadata-v1/scripts/hpc/slurm_global_gpu_capacity_512.sh
```

The actual 2026-10-08 12:48:06UTC [submission snapshot](submission-node-snapshot.txt) records selected `ibc13b03n02`: ALLOCATED, CPUAlloc112/112, CPULoad22.74, FreeMem321383MiB, 1/8 GPUs allocated. No idle eligible gnall node was observed. The earlier11:21:20UTC [proposal snapshot](proposal-node-snapshot-approval.txt) was provisional and does not describe submission-time occupancy. Neither snapshot reserves the node or requests whole-node exclusivity. The launcher records actual node, CPU allocation/affinity/load and visible GPU processes at startup; this later snapshot does not establish the cause of any past slowdown.

The wrapper resolves the shared helper from `CODE_DIR` even in Slurm spool and expands the actual command through `srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores`, Singularity CUDA runtime, and the canonical GPU environment. It invokes `global_batch_gpu_entry.py` with `ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_capacity.py`, `--continue-512`, `--budget-seconds 2700`, five repeats, and these sources:

- prepared input: `runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl`;
- FULL64 reference: `runtime/outputs/research/global_throughput/gpu-round4/model`;
- old B64 host memory origin: `runtime/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first`;
- sealed B128: `runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity`;
- sealed B256: `runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512/capacity`.

All relative source paths above are under the exact working directory. Configuration is the new checkout's `configs/orchidee_man_250919.yaml`; asset root is `runtime/outputs/research/global_throughput/round4-inputs/selected64-assets-v2`. Output is `${RUN_ROOT}/capacity`; logs are `runtime/logs/global_gpu_capacity_512_%j.txt`.

Workload: same ordered512 lanes=64 genuine points×8, continuous Days2–8 then9–15,48 steps/day, full native outputs and physical masks. Import sealed B128/B256 origins without compiling or measuring them; B256 compiler temporary bytes1,151,609,352 provide the device estimate, and old B64 RSS supplies the host estimate. Keep2GiB device/8GiB host slack and the1800-second remaining-window admission gate. One full synchronized B512 warmup precedes5 core+5 prepared-wrapper collections and complete strict reference comparisons. No fake B1024 comparison is made.

The completed run passed admission and execution with unchanged physics baseline, native dtypes, masks, outputs and tolerances. The exact-byte certificate distinguishes the host-cache repair's actual source digest. The saved-capacity report records GPU compiler temporary memory2288514256 bytes and host process peak19496943616 bytes; these are not VRAM peak evidence, and pre/post GPU snapshots do not establish a peak. No training, AD or science-capacity change was made.

The read-only [B512 saved-result collector](collect_b512_result.py) is prepared and deployed before approval, with 41 lightweight tests and Ruff passing. Independent review found no blocking issue. Its deployed raw SHA is `36e8971fd94f9883c579d1fedb4c155f27298db595742b5aae2bcb347b6a6926`. It remains external to immutable execution checkout `ad6c0e1`; no runner redeployment was needed. Exact HEAD and clean status are required before importing executed helpers; bytecode writes are disabled. It pins Linux source `b33450ca...` and the active exact certificate, binds report/manifest/checkpoint identity, environment, allocation, gates and common startup timings, and admits only a newly measured B512 stage. B128/B256 origins are revalidated without changing their original hashes, jobs, code, raw samples or memory. [Actual historical-origin verification](b512-collector-origin-verification.json) passes and contains no B512 timing.

The [canonical collector interface receipt](collector-interface-receipt.json) verifies the deployed tool, actual checkout/source certificate, exact legacy dependencies and real CLI help without JAX, model work or GPU initialization. The [first attempt](collector-interface-first-attempt.json) found an absent prior helper; its exact source bytes were copied only after confirming absence. The existing legacy helper was hash-verified and left unchanged. The final strict collection used the existing canonical admission path `target-input-admission.json` under `gpu-sweetspot-preparation-20261006`; it admitted1 stage and withheld0, exit0, after80.06s saved-array CPU wall. Preparation remains367.0582004s of600s and is separate. See [collected result](job-15657822-collected-result.json), [raw terminal accounting](job-15657822-terminal-sacct-raw9.txt), [terminal job record](job-15657822-terminal-scontrol.txt), and [terminal log](job-15657822-terminal-log.txt).

The approved command above was executed once. Terminal accounting is preserved as the complete raw9 parent/step record: parent job15657822 is `COMPLETED`, elapsed1416s, CPU4/GPU1/node1, actual node `ibc13b03n02`, exit `0:0`; child records are retained but not added to parent cost. See [raw terminal accounting](job-15657822-terminal-sacct-raw9.txt) and [terminal scheduler record](job-15657822-terminal-scontrol.txt).

```bash
/rmprog/slurm/v22.05.7/bin/sacct -n -P -j "$APPROVED_JOB_ID" --format=JobIDRaw,JobName%64,Partition%64,State%64,ElapsedRaw,AllocCPUS,AllocTRES%512,NodeList%128,ExitCode > "$SACCT_RAW9"
```

This was the raw accounting command used to preserve that record (the executed job ID is fixed):

The exact executed collection wrapper is preserved in [collect-job-15657822.sh](collect-job-15657822.sh). It uses the verified allocated node and GPU identity, the raw terminal record above, and the existing canonical input admission file `runtime/outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/target-input-admission.json`; the final command and stdout/stderr/time/exit evidence are retained with the collection artifacts.

Actual collection ran on allowed CPU `test02` through [collect-job-15657822.sh](collect-job-15657822.sh), using the existing canonical input-admission file `runtime/outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/target-input-admission.json`. The exact observed GPU identity was global ID4, UUID `GPU-949cb714-8c53-353c-1f6a-1832c629230e`, Tesla V100-SXM2-32GB, driver525.85.12. The [strict collector result](job-15657822-collected-result.json) admitted1 stage and withheld0, exit0; saved-array CPU wall was80.06s. The stage passed515 strict gates,760 native leaves×512 lanes and512 actual FULL64 reference comparisons; original FULL64, B128 and B256 identities were preserved. The fixed ordered workload is64 genuine points×8, Days2–8 then9–15,48 steps/day,7168 lane-days, not512 unique points. Collection did not compile or run the model or initialize a GPU. Core/wrapper warm cost, startup, checkpoint IO and whole-job cost remain separate. Full-phase setup/validation was not recorded, so K-amortization is unavailable; no CPU/GPU ratio, annual Fortran,10×, training, AD or general science acceptance is produced.
