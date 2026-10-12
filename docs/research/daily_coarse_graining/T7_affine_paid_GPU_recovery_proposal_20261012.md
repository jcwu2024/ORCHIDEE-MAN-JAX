# Paid affine v3 recovery: preparation and one-allocation proposal

Updated: 2026-10-12. The compatibility candidate is implemented, focused CPU
tests pass, and a new exact-commit checkout/false scope passed target CPU shell
admission. **No job is submitted or authorized for execution by this preparation.**
The proposed single continuous allocation performs startup gates first, then
the unchanged affine diagnostic only if every startup gate succeeds. Old
job15782177/f4ae is terminal and its scheduler attempt remains consumed.

## Compatibility evidence and limits

On explicitly authorized gln01, actual host `ibc13b02n04`, `/usr/bin/nvidia-smi`
`--help` reports NVSMI525.85.12. Complete `--help-query-gpu` lists `index`,
`uuid`, `name` but no `minor_number`: that field is invalid for this captured
selective-query interface. The separately requested `--version` exits255 with
invalid-combination text; this is flag evidence, not a driver/GPU failure.
[Help capture](../../../outputs/T7-affine-paid-recovery-preparation-20261012/help-capture.v1.json)
links complete stdout/stderr, exact argv, host/path and command-time receipts.
No device/process query, scheduler inventory, JAX or CUDA launch occurred.
Historical fixed n01 tool/driver identity was not established by this help.

The [official NVIDIA manual](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
documents UUID selection with `-i`, full attributes with `-q`, XML with `-x`,
Linux Minor Number as `/dev/nvidia[minor number]`, and exit2 as an invalid
supplied argument/flag. Exact HTTP200 bytes and text are preserved in
[official receipt](../../../outputs/T7-affine-paid-recovery-preparation-20261012/official/nvidia-smi.official-manual.v1.receipt.json).
This supports the CLI candidate below. It does not recover missing stderr or
prove the exact historical cause of job15782177. The
[accepted terminal report](T7_affine_paid_GPU_failure_results_20261012.md)
retains8s/4CPUs/1GPU/FAILED2:0/estimated CNY0.0055111111, no affine result and
absent live GPU cleanup observation. Earlier0-run and queue evidence stay history.

## Minimal candidate and failure evidence

The launcher first uses supported CSV `index,uuid,name`, requiring one row and
the declared Slurm physical index. It then runs `nvidia-smi -i UUID -q -x`
against that exact captured UUID. It requires root `nvidia_smi_log`, exactly
one direct GPU and no extra nested GPUs, exactly one direct UUID/product_name/
minor_number each, exact CSV/XML UUID and name agreement, and a decimal minor.
Missing, duplicate, conflicting, malformed or unavailable fields fail closed;
there is no minor=index inference, alternate-card choice or fallback interface.
The actual character-device major/minor, exact job cgroup isolation and later
CUDA-driver/PJRT single-device UUID gates remain unchanged. No new dependency,
environment rebuild, neural graph, numerical method or device-isolation relaxation.

Both CSV/XML command/returncode/stdout/stderr are retained in order before
parsing. Failed-query, malformed-XML and later kernel/cgroup mapping rejection
JSON retain the successful earlier outputs too; success stores observations
in the Supervisor record. Existing timeout partial-byte diagnostics remain.
Admission failure returns2, retains the consumed attempt and ends before the
affine probe. Early owned-lock release and uncertain-child cleanup rules remain;
unknown cleanup keeps the cooperative lock and never permits a retry.

Focused mocked Python checks pass30 cases/91 deselected, including index3 vs
minor5, duplicate UUID/minor, wrong root/nested GPU/identity, raw failure logs,
missing kernel node, consumed reentry and existing device-isolation guards.
[Python receipt](../../../outputs/T7-affine-paid-recovery-preparation-20261012/tests/focused.v4.receipt.txt)
and raw logs preserve the successful command/UTC/exit0; earlier test receipts
are retained. The changed Slurm path/attempt fixtures pass10 cases/22 deselected;
[shell receipt](../../../outputs/T7-affine-paid-recovery-preparation-20261012/shell-tests/receipt.json)
preserves argv/UTC/source stability. Each run has one existing `cache_dir`
warning. These are CPU/mock admission evidence, not live GPU/scientific acceptance.

## Concrete new allocation proposal

| Item | Exact proposal |
|---|---|
| Partition / inherited fixed node |gnall / `ibc13b03n01`; retained from the prior contract, no new idle/occupancy claim |
| Resources |1node/1task/4 actual Slurm CPUs/1 V10032GiB/nonexclusive |
| Wall / attempts |20min/no-requeue/exactly one new scheduler attempt/no automatic retry |
| Total / phase / memory caps |1200s from actual scheduler allocation StartTime; preflight240s/probe780s/release≥180s;24GiB;OMP4/OpenBLAS1 |
| Frozen code |`58a2d93885e991fc8e5e751f8a52b27aaade08e0` |
| cwd |`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-recovery-20261012-v3` |
| Output / false scope |`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-recovery-gpu-20261012-v3` /`scope.v1.json` |
| Ledger |`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-recovery-gpu-20261012-v3/T7-summary-affine-paid-recovery-gpu-20261012-v3.ledger.json` |
| Recommended new approval ceiling |CNY1.00; prior consumed approval is not a new grant |

Exact full command, **proposed only**:

```bash
sbatch /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-recovery-20261012-v3/scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh
```

Known additive rates: allocated CPU CNY0.07/hour, GPU CNY2.20/hour.
CPU `4×20/60×0.07=0.0933333`; GPU `20/60×2.20=0.7333333`; total
`0.8266667`, rounded upward **CNY0.83**. Recommend a new CNY1.00 ceiling.
Billing minimum/rounding is unverified. No exclusivity, extra node/card/CPU,
longer wall, second allocation or automatic paid/free fallback is proposed.
Four Slurm CPUs are the actual billed CPU contract; do not add nomultithread
and accidentally allocate8 CPUs. No new scheduler policy/resource refresh occurred.

After specific approval, the sole dispatch owner binds approval to the exact
false scope/manifest, records intent and invokes sbatch once. Accepted submission
consumes the scheduler attempt even PENDING. The job's first filesystem action
exclusively writes its new Slurm attempt; Python writes a separate exclusive
attempt. Startup failure consumes the attempt; no old scope/attempt can be reused.
No attempt marker is precreated by preparation. Approval/proposed status is
checked before live entry, source/clean-HEAD identities remain mandatory, and
the cooperative free/paid management boundary remains read-only and fail-closed.

Within that same future allocation: validate exact actual Slurm/node/TRES/clock,
CSV→UUID-boundXML→kernel/cgroup mapping, then the unchanged CUDA/PJRT preflight.
Only after every startup gate passes, run original scalar/vmap/lax.map graphs
once each plus one serial-merge compile, repeats3,12 measurements/91 arrays.
Phase caps are additionally constrained by the remaining total/release budget;
startup never renews allocation time. Failure retains full logs and ends.
FP32 neural/independent FP64 oracle, `rtol=1e-5, atol=1e-6`, complete primal/
JVP/input/parameter derivatives, masks/isolation/repeats/raw arrays/HLO remain.
This fixed diagnostic cotangent is not T7 loss; no full-model update/training.

## Exact deployment, inputs and readiness

Source/test stage committed/pushed normally as `58a2d93885e991fc8e5e751f8a52b27aaade08e0`. An independently
verified full Git bundle created the new checkout on test01, actual `ibtest01`,
without old Git-object alternates. No old/main/shared checkout was updated.
[Deployment receipt](../../../outputs/T7-affine-paid-recovery-preparation-20261012/deployment.v1.json),
[source manifest](../../../outputs/T7-affine-paid-recovery-preparation-20261012/deployment-manifest.v1.json)
and [false scope](../../../outputs/T7-affine-paid-recovery-preparation-20261012/scope.proposed.v1.json)
bind exact code/source/input identities. Scope is proposed/false/unconsumed,
not approval. New paths were absent before creation; only new prep/log files exist.

The existing two recovery files are reused in
`runtime/transfer/T7-summary-recovery-20261011-v1/`:
operand SHA256 `bf36a098b4e6a89d555f8d18521555f2da66135425d9435cfe01878adeec6129`;
receipt `f863a2d964f626cec327981d86a8832e73aa45af62d708a342805c4aa4220e00`;
capture source `70619b0a6445cde87eccaeb31feac46bbb34cdc0d7a6f68c68938c4d09901072`.
Probe SHA256 is `05e2ee66b1905fd238722d3a4cf645e74639f7134de4af5a700cf0b657305ee0`.
Fixed TRAIN742/275/482/299 window0/day0, sample3/raw434 and `(4,257,4)@(4,384)`
remain unchanged. No capture search/recapture or old CPU/numerical-reference rerun.

Actual test01 Bash `--dry-run` expanded both submission and downstream argv,
then the CPU paid parser passed in1.860166s. Source/input/
false-scope hashes before/after agree; imports resolve into this exact checkout;
JAX was not imported. Attempt/output/ledger/preflight paths remain absent.
[Target CPU receipt](../../../outputs/T7-affine-paid-recovery-preparation-20261012/target-cpu-admission.v1.json)
links actual argv/imports/source identities and returned parser logs. This is
real Bash argument expansion, not only syntax checking; sbatch was only printed.
Clock receipts retain their separate local/server sources without reconciliation.

Exact525 XML output, actual n01 version/Slurm GPU/kernel/cgroup binding,
container CUDA/PJRT identity, real GPU numerical comparison and release remain
unverified future gates. No allocation/GPU preflight/probe/ledger reservation
or submission occurred. Preparation cost/card-hours are0; a future approved
job would be paid. Free b676 stays paused and approval unconsumed; old f4ae,
old attempts/scopes/results and all recovery inputs stay preserved. Graph Verify
generation2026-10-10T08:46:50Z marks these worktree paths excluded/not_tracked;
exact source fallback was used without reindex or completeness claims.
Stop at this concrete proposal; no monitor, automatic next experiment or training.
