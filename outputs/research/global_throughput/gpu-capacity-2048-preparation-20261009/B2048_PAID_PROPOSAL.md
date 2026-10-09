# Fixed2048 B1024/B2048 one-card paid proposal

Status: **human-approved, submitted once as job15671732; RUNNING on n03**.
Submitted2026-10-09T04:54:47Z, started04:55:15Z (28s queue); latest04:58:34Z
shows the first B1024 child compiling. No retarget or resubmission was required.
See [actual submission/startup receipt](submission-receipt.json). No automation.
Old accepted job15662711 and all previous budgets remain closed. Exact implementation
pin, independent immutable deployment and actual test02 interfaces passed; see
[pin](final-code-pin.json), [interface receipt](deployment-interface-receipt.json) and
[local verification](LOCAL_VERIFICATION.md).

Approved resources: gnall1node/1task,4 allocated Slurm CPUs,1Tesla V10032GB,
120min/7200s hard wall,no-requeue,no-exclusive. Scheduler cwd:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.
Checkout: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-fixed2048-v1`.
Single-use run root: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1`.
Execution SHA: `8aa2e05aa24b65dc4d30a27d4ee857bba6312221`. Actual selected node: `ibc13b03n03`.
Earlier preparation basis `2026-10-09T04:18:18Z`: MIXED,64/112 CPUs allocated,0/8 GPUs allocated,
CPULoad13.24,FreeMem188243MB. No idle eligible node was observed; this is no
reservation or exclusivity claim. Immediate actual submission preflight04:54:47Z
confirmed n03 CPUAlloc100/112,GPUAlloc3/8,12freeCPUs/5freeGPUs,hostFreeMem198685MB,
load9.91 and no reservations. The one approved job started after28s. Test-only
forecasts were anomalously2041 for every tested node and unrestricted gnall; no
node-specific avoidable queue or retarget was observed. Resource/cost envelope unchanged.

Same28672 lane-days for both fresh shapes: ordered64 genuine FULL64 points×32,
continuous Days2–8/9–15,48steps/day,native dtype/full outputs/masks/tolerances.
New B1024×2 then conditional B2048×1; each shape gets a fresh process with default
allocator, same parent/job/node/selected physical GPU/resources/runtime. First child
must exit and strict2051gate/760leaf/atomicstage admission must pass before candidate;
both shapes need3600s remaining,2GiB physical GPU slack and8GiB host slack. No
speedup prerequisite. Missing candidate retains control without a fabricated curve.
See [protocol](PROTOCOL.md) and [planning origin](resource-estimate.json).

Measured prior fixed1024 B1024 lower64.84s/compile990.70s/first41.10s/wrapper47.41s
are planning provenance only. New2048 run/sampling/validation doubles the ordered
work; there are two fresh compiles/reloads/imports, with uncertain B2048 compile
scaling.120min is a finite proposed envelope, not predicted completion. Conservative
B2048 device requirement plus2GiB=29675924096B (~27.64GiB); host2×historical peak+
8GiB=64321847296B (~59.90GiB). Actual per-shape stack/compiler/resource checks
replace projections. Free snapshots/compiler buffers are not measured VRAM peaks,
and a conservative denial does not establish physical impossibility.

Worst requested-time cost:1GPU×2h×CNY2.20=CNY4.40 plus4CPU×2h×CNY0.05=CNY0.40,
primary maximum **CNY4.80**. CPU CNY0.07 gives CNY0.56 and conservative additive
maximum **CNY4.96**.2GPU card-hours/8 allocated CPU-hours; billing unverified,
queue time unbilled. Actual terminal cost counts the parent once, never adds children.
No retry, extension, requeue, cancellation or second job is included.

Approved command executed once from gln01 (unchanged resources/time/cost):

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
/rmprog/slurm/v22.05.7/bin/sbatch --nodelist=ibc13b03n03 --export=ALL,CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-fixed2048-v1,RUN_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-fixed2048-v1/scripts/hpc/slurm_global_gpu_capacity_2048.sh
```

No historical timing doubling, proven optimum, peak VRAM, CPU/GPU ratio, Fortran,
annual/669-point/50-year,10×,trainingAD or general science acceptance is claimed.
New B2048 results and scientific acceptance remain absent.
