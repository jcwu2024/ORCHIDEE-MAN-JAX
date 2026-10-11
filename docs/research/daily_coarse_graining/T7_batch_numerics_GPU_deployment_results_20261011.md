# Bounded affine GPU deployment result — 2026-10-11

用户已批准，CPU readiness通过，GPU runtime/数值对照待空闲卡，本次one-shot未消费。
Status: `approved_waiting_resources`. The last resource refresh admitted no
card; this is resource non-admission, not execution or numerical failure.
No GPU launch, runtime preflight, open scope or ledger reservation occurred.
Occupied GPU seconds/card-hours/cost are0/0/CNY0 because none was launched or
reserved; they are not calculated by subtracting host clocks. No GPU release
operation was necessary. This task stops without further polling or automatic
monitoring; approval persists for the unchanged, unconsumed one-shot.

## Approval and frozen scope

On2026-10-11 the user in the supervisor chat replied “批准。” to this separate
gln01/actual host `ibc13b02n04` scope: one idle nonexclusive V100, CNY0, at most
1200 occupied GPU seconds, preflight≤240/probe≤780/release reserve≥180,
24GiB, OMP4/OpenBLAS1. Frozen sample3/raw434 recovery operands, three affine
routes plus one merge compile, repeats3, planned12 measurements/91arrays,
unchanged math/precision and `rtol=1e-5, atol=1e-6` remain binding.
No recapture, historical large-asset transfer, full-prefix/T7/T30/training,
paid fallback or retry is included. The later approval supersedes the earlier
preparation's unapproved status; the original false proposal/manifest snapshots
remain unchanged historical evidence. The
[local authorization/closure receipt](../../../outputs/T7-summary-affine-gpu-20261011-execution/authorization-closure.v1.json)
records approval and input/source bindings, but is neither an open scope nor a
runnable launcher input. [NEXT_STEPS](../../NEXT_STEPS.md) owns current action.

## Exact deployment

Accepted/deployed HEAD: `b676353bbaeb633ba9ccab075aaeadcf9408c7d7`.
Access used authenticated SSH through `192.168.11.1`, existing identity and
normal host-key verification; current VPN subnet route and successful SSH were
recorded. The independent checkout was fast-forwarded from `688e14cd...2697`
using a verified incremental Git bundle,3186046bytes,
SHA256 `1329a023669338ced743beb248b8e35c822cb06442683140f35f26af453a7cb2`.
The canonical root and older shared-split checkout were not changed.

| Purpose | Remote path |
|---|---|
| Frozen code | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-batch-numerics-20261011-v1` |
| Two small inputs | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-summary-recovery-20261011-v1` |
| Output directory | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-gpu-20261011` |

The target checkout is clean and all seven execution-source SHA values match
the accepted manifest. Actual import resolves to that checkout's
`jax_orchidee/__init__.py`. It uses shared Git objects through the canonical
root `.git/objects`; preserve that dependency. Canonical CPU/GPU interpreters,
container/toolchain and driver-library paths exist, but existence does not
verify container/PJRT/GPU runtime. See
[deployment receipt](../../../outputs/T7-summary-affine-gpu-20261011-execution/deployment.v1.json)
and [access receipt](../../../outputs/T7-summary-affine-gpu-20261011-execution/access.v1.json).

Fresh operand SHA256 is
`bf36a098b4e6a89d555f8d18521555f2da66135425d9435cfe01878adeec6129`;
fresh capture receipt SHA256 is
`f863a2d964f626cec327981d86a8832e73aa45af62d708a342805c4aa4220e00`.
Unchanged capture source is `70619b0a...901072`, current probe is
`05e2ee66...305ee0`. No new capture or full affine numerical probe occurred.
The [preparation report](T7_batch_numerics_GPU_preparation_results_20261011.md)
retains the original-file loss, six independently matching numeric hashes and
separate historical CPU12/91 evidence; original NPZ/receipt bytes remain lost.

## Accepted target CPU readiness

Actual target Bash→canonical CPU Python `--dry-run` passed exit0, checking
source/input identity and planned12 measurements/91arrays/4compiles. The
5.049263s receipt wall time includes SSH invocation overhead; it is not the
parser's isolated runtime.
The raw dry-run stdout retains its production generic text “local preparation
only; target shell/container/bootstrap/preflight not verified”. Preserve it:
the separate target checks now verify CPU shell and private phase supervision;
actual container/PJRT/GPU/preflight remain unverified.

The first CPU-only phase test failed in its harness parser (`IndexError` on a
blank `ps` row), after its actual private child had ended. The v1 failure is
retained. Ignoring short/blank rows in the isolated harness yielded v2 pass:
production `start_phase_deadman` terminated its real Linux session/group with
short caller deadlines, child exit-9,2.324224874s, leader=PGID12182,
descendant12188, no remaining rows. Production source was unchanged.
Saved-only collection verified both private CPU groups11818/12182 absent and
downloaded four CPU artifacts with size/SHA verification. The supervisor
accepted target CPU readiness. These checks used no GPU/preflight/ledger.

Evidence: [CPU readiness](../../../outputs/T7-summary-affine-gpu-20261011-execution/target-readiness.v2.json),
[real shell receipt](../../../outputs/T7-summary-affine-gpu-20261011-execution/target-shell-real-parser.v1.receipt.json),
[retained harness failure](../../../outputs/T7-summary-affine-gpu-20261011-execution/target-Linux-owned-phase.v1.receipt.json),
[v2 raw receipt](../../../outputs/T7-summary-affine-gpu-20261011-execution/target-Linux-owned-phase.v2.receipt.json),
[saved-only retrieval](../../../outputs/T7-summary-affine-gpu-20261011-execution/readiness-retrieval.v1.json).
Initial old-server `git -C` rejection and local GBK stdout exception remain in
access logs; `cd`-based Git and UTF-8 printing resolved them without source changes.

## Final resource admission and stop

The one final refresh after CPU readiness recorded local UTC
`2026-10-11T12:15:07.622703Z` and server UTC `2026-10-11T12:06:46Z`
separately. Actual host `ibc13b02n04` reported load43.48/43.64/43.70 and
CPU affinity0–55. Neither clock difference establishes occupation duration.

| Host index / V100 UUID | Memory / utilization | External compute PID |
|---|---|---|
| 0 / `GPU-daa2988f-72e7-d5e0-45e5-1bef9c1cc688` |31707MiB /100%|74375 (31704MiB)|
| 1 / `GPU-07ee30e1-0b7d-4067-5c0a-3384e598e751` |31707MiB /100%|73253 (31704MiB)|

No0MiB/0%/noapps card was eligible. Metadata confirmed absent target
`T7-summary-affine-gpu-20261011-v1.ledger.json`, `scope.v1.json`,
`result.v1.outer-timeout.json`, `result.v1.json` and
`result.v1.runtime-preflight.json`. There is no GPU result to validate or
retrieve. [Resource admission receipt](../../../outputs/T7-summary-affine-gpu-20261011-execution/resource-admission.v1.json)
preserves the raw observation and non-admission. The approved one-shot remains
unconsumed; no resource fallback, further refresh, reservation or automatic
monitor is active. GPU/numerical validation is pending. No original-model
repair, derivative reference, precision change, training or throughput/science
acceptance follows.

Local source is unchanged. Reuse the accepted44 local tests (30CPU/mock,
14shell) and real local parser dry-run from
[postcommit delivery](../../../outputs/T7-batch-gpu-preparation-20261011/postcommit-delivery.v1.json);
this six-document result stage requires no numerical rerun.
