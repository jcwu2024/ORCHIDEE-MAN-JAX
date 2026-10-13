# T7 v2 paid single-GPU affine result

Date: 2026-10-13. Sole job **15822347** completed `0:0` on `ibc13b03n04`
after **110 allocated seconds**. Discovery, actual runtime preflight and the
unchanged GPU affine diagnostic report passed: **12 measurements, 91 preserved
arrays and four small graph compilations**. This is a component result; the
original neural/T7 state-derivative failures remain unresolved.

## Authorization and immutable execution

The human said: **“那边停了，你看没问题就可以批准，授权你了。”**
The supervisor then approved the exact proposal. Authorization derives from
that human delegation plus the supervisor decision, not an automatic message.
`approval.v1.json` binds the words, decision, full submission argv, resources,
caps, CNY1 ceiling, immutable execution SHA
`7efe4b443942bfafed6625e4dbabd44ff3583a7e` and approval-documentation SHA
`065c64cbfe656cb953bf2cd280041ca8389fbba6`.

The [preparation](T7_paid_v2_GPU_deployment_preparation_20261013.md) and
[v2 method](T7_paid_GPU_identity_local_repair_20261012.md) retain exact command
and method provenance. One nonexclusive gnall allocation used one node/task,
four allocated Slurm CPUs, one V100, fixed n04, `--no-requeue`, and 1200s.
Startup discovery/preflight shared the remaining240s window; affine cap780s,
release reserve180s and memory cap24GiB were unchanged. Worst-case additive
quote CNY0.8266666667 (upward0.83), approved ceiling1.00.

CPU metadata admission on test01 checked the clean immutable checkout, exact
source/input/manifest, unused scope, absent mutex and paused free scope before
opening. Sole `sbatch` invoked from gln01; all inherited `SBATCH_*` settings
were recorded and cleared (none existed). No redeployment or source mutation.
Queued from11:18:27, actual Slurm start11:38:54/end11:40:44; the earlier queue
start estimates were not guarantees. No node polling, extra GPU allocation,
resubmission, node change or fallback occurred.

## Actual GPU binding and numerical evidence

Unchanged original scheduler/CUDA mask was `0`. Discovery reported runtime
count1/ordinal0, PCI `0000:1A:00.0`, driver count1/physical handle0 and UUID
`GPU-49cb0d50-5428-f9f8-987d-200c34c47f30`. UUID-selected CSV/XML/kernel bind
Tesla V100-SXM2-32GB, host index0, major195/minor0. Fresh preflight and the
affine child each reverified runtime/PJRT/PCI/UUID before science; deployed
JAX0.4.38 exposed `cuda:0`/one PJRT device. Discovery, preflight and probe exited0
in2.016963s,68.339625s and26.514771s respectively.

This actual job has matching numeric mask/ordinal/index/minor0 and driver count1.
It does **not** demonstrate numeric remapping or driver enumeration greater
than1. Allocation association uses the accepted operations trust in the original
scheduler mask in that runtime namespace; independent GRES index→UUID mapping
is not verified. Actual-job cgroup `devices.list` is retained as `a *:* rwm`.
Selected-device read/write passes; OS permission width is broad, and opening
denial for other GPUs is not verified. Broad access is not evidence of execution
on another GPU.

The real captured `(4,257,4)` summary and sample3/raw434 direction retain exact
operand SHA `bf36a098b4e6a89d555f8d18521555f2da66135425d9435cfe01878adeec6129`,
receipt `f863a2d964f626cec327981d86a8832e73aa45af62d708a342805c4aa4220e00`
and capture source `70619b0a6445cde87eccaeb31feac46bbb34cdc0d7a6f68c68938c4d09901072`.
FP32 operands/results/AD, FP64 host oracle, seed20261011, three repeats and
`rtol=1e-5, atol=1e-6` are unchanged. Three operator graphs plus one serial merge
compiled; optimized b1/vmap/map HLO and full/partial NPZs are preserved.
The report passes12 oracle/12 AD-direction/12 other-sample isolation checks,
eight cross-route checks and nine repeat checks. Cross primal/tangent/input
gradients/masks are exact; small loss/parameter-gradient differences pass strict
tolerances. All within-route repeats are exact. Saved-only NumPy verification
in `execution-summary.v3.json` independently rebuilds the FP64 oracle from the
hash-verified operands, recomputes all saved-array comparisons, JVP/VJP pairings
and FP64 finite differences. It passed in3.375878s, with no JAX/GPU graph.
The local analyzer first raised a boolean-mask arithmetic TypeError. Its repair
passed; the worker accidentally overwrote the first failed capture files when
saving that successful run. The first error text was read before overwriting
and remains in the tool transcript/`analysis-harness-incident.v1.json`; original
failed raw capture bytes are unavailable. V2 successful captures remain;
acceptance then corrected the analyzer's independent-pairing variable and
required finite coordinates/exact boolean masks. The same saved evidence passed
the final v3 run with exclusive captures/receipt. Remote scientific evidence,
strict gates and arrays were unchanged.

This separately compiled affine projection passes under the captured operands
and diagnostic cotangent. It neither reproduces the original full-neural
failure nor identifies its first differing operation or establishes B1 derivative
truth. It does not certify a repair, full-T7 gradient, complete optimizer update,
training throughput, architecture/precision change or production adoption.

## Release, accounting and preservation

Supervisor primary/numerical status is passed/executed. Owned process groups
were released; final UUID observation has0MiB,0% utilization and no processes.
Ledger closed/release verified at109.72439998202026 occupied GPU seconds,
within1200s. Sampled peak757MiB is below24GiB; transient discovery memory before
UUID was known was unobserved, so757MiB is not a complete discovery-inclusive peak.
CPU affinity `[0,1,56,57]` and start host load `[2.91,6.5,5.87]` are preserved;
the submission occupancy snapshot cannot establish past contention.

Slurm terminal `sacct` owns the110s allocation/accounting. Terminal `scontrol`
was already purged (“Invalid job id specified”); its missing result/error bytes
are retained, while original RUNNING allocation snapshots are separately checked
with source paths/hashes. No terminal control snapshot was synthesized.
Main allocated card-hours110/3600 =0.0305555556; estimated additive cost:
fourCPUs×110/3600×0.07 = CNY0.0085555556, GPU110/3600×2.20 = CNY0.0672222222,
total **CNY0.0757777778**. Actual bill is unknown. The extern step111s and
Supervisor109.7244s remain distinct; neither replaces main allocation accounting.
Original host timestamps are retained; cross-host clocks must not infer ordering.

CPU-only terminal closeout preserves the executed-open scope bytes and closes
`T7-summary-affine-paid-single-gpu-20261012-v2` as consumed/false/no-retry.
Mutex absence is recorded; no lock deletion or new live GPU query occurred.
Old c8/58a/f4ae anchors, old scopes and all input/source identities are unchanged.
The separate free b676 grant remains paused/unconsumed.

Evidence is under `outputs/T7-paid-v2-GPU-execution-20261013/`: exact approval,
opening/admission/intent/invocation/accepted submission, four bounded job
observations, complete49-file `raw-terminal/`, original arrays/HLO/phase/cgroup/
identity/resource reports and closeout. Full archive SHA:
`5d9c674411fad230afa1d46b51c1c3f22e29563988b60b32e04611fecd84286c`.
Inventory, archive bytes and safe extraction were verified. Proposed/open/closed
scope SHA values are respectively `04173f4d...71a`, `2bfe02d8...5a91`,
`06313c12...9dbc`; full values are retained in their receipts.

The one approved attempt is finished. No new diagnostic, repair/redeployment,
fallback, monitoring or training is authorized by this result. The five current
entry documents own this settled status; historical preparation reports retain
their original dated snapshots.
