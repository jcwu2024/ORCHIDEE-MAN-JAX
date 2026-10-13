# V2 paid GPU deployment: CPU preparation and one-allocation proposal

2026-10-13. Accepted execution code is immutable
`7efe4b443942bfafed6625e4dbabd44ff3583a7e`; later documentation HEAD is separate.
The [corrected identity method](T7_paid_GPU_identity_local_repair_20261012.md)
remains the method authority. This stage prepares its independent deployment,
false scope and real test01 CPU dry-run. It grants no paid/GPU execution.
No sbatch/srun, GPU query/API initialization, discovery, preflight, affine,
training or free fallback ran. Old c8/58a/f4ae code/scopes/results and recovery
inputs remain unchanged; their approvals cannot migrate to v2.

## Deployed identities and CPU evidence

Project runtime is `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime`.
The independent code path is
`checkouts/T7-affine-paid-single-gpu-20261012-v2`; diagnostics are
`outputs/diagnostics/T7-summary-affine-paid-single-gpu-20261012-v2` below that runtime.
Both paths and the new delta path were unused before writes. The existing
verified c8 full bundle was cloned on test01, then a61100-byte delta fast-forwarded
to7efe4b44. The new Git store has no alternates/shared-old-object dependency;
it contains2449 files at deployment. Existing environments/data were reused.
No large asset/environment copy, old checkout write or scope replacement occurred.

`scope.v1.json` is `proposed`, `execution_authorized=false`, bound to exact7efe
source hashes and the independent v2 token. Its existing schema field
`approved_cost_ceiling_cny=1.00` is a proposed ceiling, not received approval.
The local [deployment package](../../../outputs/T7-paid-single-gpu-deployment-20261013/)
contains its exact false scope, manifest, bundle/deployment receipts and raw
target CPU stdout/stderr. The complete proposal is
[proposal.v1.json](../../../outputs/T7-paid-single-gpu-deployment-20261013/proposal.v1.json).

ibtest01 recorded UTC2026-10-13T02:38:08.399916Z; the canonical CPU parser
returned0 in0.613901s. The [actual CPU admission](../../../outputs/T7-paid-single-gpu-deployment-20261013/target-cpu-admission.v2.json)
and [command receipt](../../../outputs/T7-paid-single-gpu-deployment-20261013/target-CPU-admission.v2.receipt.json)
preserve host timestamps and raw shell/parser output; different host clocks
are retained without inferring approval order. The check ran the real Slurm
shell with `--dry-run`, compared the full printed
submission and downstream argv, and actually invokes the downstream paid Bash
entry through the canonical CPU Python. Imports resolve to the new checkout;
source/input/manifest/scope identities agree before/after. Discovery/preflight/
probe commands are planned only; original12 measurements/91 arrays/4 small
compiles and all caps remain unchanged. No attempt, ledger, result, Supervisor,
discovery or preflight artifact exists. No JAX import or CUDA-library/API call
is allowed in the admission process. This proves target CPU argument/readiness
only, not real GPU/container/PJRT/numerical or release acceptance.

The first CPU harness stopped before shell/parser execution because it assumed
all historical `execution_authorized` fields were false. Its raw failure is
retained. Historical f4ae scope deliberately retains original `open/authorized=true`
approval bytes, while matching job15782177 scheduler/Python one-shot receipts
consume it and prohibit retry. The separately named v2 harness binds those exact
hashes/flags/consumption receipts; c8/58a retain their closed/false checks.
The successful retry never edits/reopens the old scope. No deployed source
change or69/270-case/static-symbol rerun was needed.

## Proposed single complete allocation

One authorized preparation snapshot, cln01 UTC2026-10-13T02:21:59Z, reports
n04 IDLE, CPUAlloc0/112, load0.01 and empty AllocTRES; gnall UP, installed8V100.
The raw snapshot is retained. This is not a reservation or future placement
guarantee. Fixed n04 is intentional; queueing is acceptable. There is no polling,
node change or extra GPU-only discovery allocation in this stage.

| Proposed resource | Bound |
|---|---|
| Placement | gnall, fixed `ibc13b03n04`, one node |
| Tasks / charged CPUs / GPUs | one task, four actual Slurm CPUs, one V100 |
| Sharing / restart | nonexclusive, no requeue, no retry/fallback |
| Allocation |1200s/20min; scheduler StartTime/EndTime anchors deadlines |
| Discovery + preflight | shared remaining240s startup window |
| Original affine / release | probe≤780s, protected release≥180s,24GiB |
| Worst-case additive estimate |4×(1/3)×CNY0.07 +1×(1/3)×CNY2.20 = CNY0.8266666667, rounded upward0.83 |
| Requested future ceiling | CNY1.00; no execution approval received |

Working directory and complete future command, **not executed**:

```bash
CODE=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-single-gpu-20261012-v2
DIAG=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-single-gpu-20261012-v2
cd "$CODE"
# Record inherited SBATCH_* first; clear every such variable for this one request.
while IFS= read -r name; do unset "$name"; done < <(compgen -A variable SBATCH_)
sbatch --partition gnall --nodes 1 --ntasks 1 --cpus-per-task 4 \
  --gres gpu:1 --time 00:20:00 --no-requeue --nodelist ibc13b03n04 \
  --chdir "$CODE" --output "$DIAG/slurm-%j.stdout.txt" \
  --error "$DIAG/slurm-%j.stderr.txt" \
  "$CODE/scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh"
```

The one job's fixed downstream paid entry uses the deployed checkout and
`scope.v1.json`, `result.v1.json` and the independent v2 ledger in DIAG.
Operands/receipt stay at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-summary-recovery-20261011-v1/summary-recovery.v1.{operands.npz,capture.json}`;
capture source stays in CODE at `scripts/hpc/capture_shared_latent_T7_summary_operands.py`.
Exact input/source SHA values, full shell-expanded argv and planned phase commands
are in the target receipt/proposal. The accepted foundationpose, canonical GPU
venv and existing driver/toolchain bindings are reused, not rebuilt.

After separate new user approval, bind its exact resources/cost/argv/identity
in a preserved receipt before opening this scope; retain the false snapshot.
Record and clear inherited SBATCH_* as shown, retain submission intent/invocation/
returned JobId and submit once. A fresh actual allocation must independently
match n04/gnall/one GPU/four allocated CPUs/task/time/sharing and scheduler clock;
the present snapshot cannot satisfy that live gate. The job creates its one-shot
Slurm attempt marker first, then uses existing cross-scope mutex/free-paused
management and source/input/scope admission; no automatic scope creation.

Keep original scheduler mask/order unchanged. Count1/runtime PCI→physical driver
UUID→UUID-selected NVML/XML/kernel binds execution under trusted scheduler-mask
provenance; it is not independent GRES configuration or OS-isolation proof.
Record actual-job cgroup rules and separate OS width; selected rw is mandatory.
Fresh preflight and actual affine repeat runtime/PJRT/PCI/UUID checks before
upload/compile. Driver total enumeration may exceed1 under explicit v2 semantics.
Site configuration contents, target GPU API/--nv bindings and transient discovery
memory remain unverified despite the prior parent static-symbol evidence.

Retain phase/raw identity,12/91 arrays/results, Slurm terminal status and actual
occupation/accounting. Preserve failures; no retry, changed tolerances, reference
or sampling. Existing owned-group cleanup and UUID release observations govern
closure. Unknown release retains mutex/full1200s conservative charge; discovery
memory before UUID is known is unobserved. Close/consume the scope after the sole
attempt, retaining false/open/closed snapshots and results; no cleanup deletion.
This stage ends with CPU readiness and an unconsumed false scope, pending
supervisor acceptance and a separate new user approval for that exact run.
