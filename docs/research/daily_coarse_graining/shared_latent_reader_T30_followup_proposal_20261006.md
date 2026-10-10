# Corrected latent-reader T30 and complete-update follow-up proposal

Status: the user approved continuous execution of the three-stage free T30
scope on 2026-10-06:15600 cumulative GPU seconds, protected compile12000 /
matched3600 pools, sequential occupations at most7200 seconds and ordinary
repairs within those same pools. Optimizer updates, formal training and paid
submissions remain outside this grant. The completed
[T7 result](shared_latent_reader_results_20261004.md) supplies the measured
starting point. Scientific choices, resource approval and final acceptance
remain with the lead/user. Fresh receipt
`59a57e60-91e1-434a-9787-7189dc602e93`, signed local UTC
`2026-10-06T10:41:07.987Z`, cites the current user delegation from
source thread `019fdc7a-5f6e-7620-a3be-28c855800369`.
The original false template is preserved as
`latent-reader-T30-20261006-free-approval.unapproved-template.json` locally
and remotely; the canonical receipt now has `approved=true`.
No stage boundary needs another approval. The original compile started on
gln01 GPU1 with UUID `30610006-0000-4000-8000-000000000001`.
Both corrected T30 graphs are now saved (original UUID0001, candidate UUID0002)
and first-call values are finite with zero hard counts. Candidate release bookkeeping was reconciled with
its full7200-second conservative charge after proving its processes had gone
while external GPU1 occupancy persisted. The saved-only matched stage then
completed8calls/3pairs on GPU1 with0/0 model compilations: warm medians
original6.650839463s/candidate5.189474726s,1.281601668× component throughput.
Status `numerical_failed`: strict state repeats fail original2/3 and candidate
3/3, while forward/gate and parameter repeats pass. Final charge10537.140655
seconds, compile remaining1954.560441/matched3108.298904. The current
[result block](shared_latent_reader_results_20261004.md) preserves the exact
timings, raw interrupted receipts and accepted no-refund repair. Cold calls
are excluded from the ratio; complete-update speed/scientific admission is
not established by this component measurement.

The lead's completed-measurement decision is to retain the candidate for
low-cost full-update source-reuse preparation. Strict state failures remain
open. The transparent `max(parent,T30)+H` planning proxy gives1.2816× at zero
overhead and decreases for equal nonnegative overhead; complete-update speed
is unmeasured and the1.5× engineering screen is unmet. This preparation does
not authorize a new GPU graph, optimizer update, formal training or paid job.

## Starting evidence and limitations

The real artifact742 T7 comparison completed three alternating same-V100
full native parameter/initial-state AD pairs. Original warm median1.571274640s,
candidate1.225237451s, ratio1.282424593×. Sample standard deviations are
0.083326519s and0.006800359s (CV5.146% and0.554%). This is a bounded component
measurement, not a complete-update throughput result. Warm T7 saves
0.346037189s per call. Observed lower+backend costs are original2678.220265s
and candidate5868.251324s, from different occupations/environments, so they
do not form a controlled cold-start cause comparison. The candidate's greater
compile cost makes saved-binary reuse across later runs an important condition.
The28.24% warm component gain does not include first compilation or loading.

Both architectures retain finite forward/gradient values, all hard physical
and mask counts0, full parameter gradients, and memory below24GiB. Strict
`rtol=1e-5, atol=1e-6` remains: original state coordinate33 snowgrain fails in
all three repeats; candidate hydrol snowgrain508 fails in pair1. Candidate
pairs2/3 state repeats pass. DOC remains in the complete state comparison and
has no failed coordinate in this particular T7 result. These four failures
remain failures; the graphs are diagnostic assets, not scientific admission.

Saved actual public losses on identical T7 inputs are original0.08665352921509632
and candidate0.02362536586305439. Their diagnostic update0/fresh weight trees
differ, so loss differences do not establish learned accuracy. No equality
between different architectures' loss or gradients is required.

Useful repeatable timing justifies preparing this follow-up. It does not
waive state gates or trigger T30 automatically.

## Frozen comparison and old T30 compatibility

Use the real prepared artifact742 T30 day69–98 case, the declared original
update0 checkpoint, and exactly the candidate initialization already used by
T7. Both retain768×8 backbone, FP32 neural/FP64 physics, public checkpointed
recurrent objective, native `argnums=(0,1)`, all legal/private/GPP outputs,
physical/source/science masks and original updater/tail. Original runs
expert128/non-GPP128, canonical GPP64 and field dynamic-projection reuse with
source-reuse backbone. Candidate runs canonical field/chunk64 with32×384
latents and private width64/two-block readers. No precision, architecture,
loss, sampling, horizon truncation, gradient or tolerance change is proposed.

The old original T30 full-state reference
`shared-original-t30-dual-reference-20261001.reference.executable.pkl` is
historical route/provenance evidence. Its wrapper forces T30/reference and
original_field_only, uses the immutable c599236 source, and does not install
the guarded turnover overlay. Its identity lacks the accepted
`f6240cf5ce64a47b9bbe0647b5e5715305e9a9230ea77f2bd9f42981ae35a93c` function.
The old source base is
`bb8674f3d2ceadb28df907346d18f6300dd87f46d09ff6010c795f72dbee6485`.
Consequently **one corrected original T30 graph is needed**; the old binary
must not be relabeled as the corrected denominator. Candidate T30 has no
saved graph. The source and identity audit uses the exact local private
wrapper/snapshot and [old reference report](shared_original_t30_dual_reference_results_20261001.md),
plus the [accepted guarded correction](shared_original_full_state_schedule_followup_results_20261005.md).

## Resource and phase proposal

Proposed next authorization is **one coherent free15600 cumulativeGPU-second
scope**, covering these three stages and ordinary repairs/necessary retests
without resets. Protect3600 seconds cumulatively for matched comparison,
preservation and exit; compile stages and their repairs may spend only the
other12000 seconds cumulatively. Unused compile time may be reused across
original/candidate repairs within that pool. Matched retries share the same
3600 pool. Settle actual occupied time before reuse; an unresolved reservation
blocks another occupation. Each single-card occupation remains≤7200 seconds. Successful
graphs are immediately reused; a later repair cannot recompile a saved graph.
There is no paid fallback in this proposed execution package. If the free card
is busy, stop and prepare a separate exact paid resource/command/cost proposal
for the human user instead of submitting automatically.

Free work runs on `gln01` (`ibc13b02n04`) using one genuinely idle V100 with
explicit GPU selection from the stable grant's `allowed_gpu_indices=[0,1]`,
`OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`,
selected CPU affinity and recorded host load. It has no Slurm CPU allocation
or whole-node reservation. The64GiB main-memory figure is a requirement, not
an enforced RSS bound. Idle observation does not reserve the node or card.
Every occupation records its actual single GPU index; changing to the other
idle card does not re-sign the grant, reset the ledger or allow overlap.
The entire matched occupation executes both architectures on its one card.
This free scope now has its fresh explicit grant; paid gnall work requires a separate human
approval covering its node, resources, wall, command, workdir and worst cost.
Neither old free ledger nor the completed T7 paid approval funds these phases.
No occupation exceeds7200 seconds. Record actual affinity, load, node and
selected-card processes at launch. Only a future paid hypothesis uses one
node/task,8 allocated CPUs, one V100,64GiB Slurm memory and nonexclusive
scheduling.

| Separate occupation | Preflight cap | Main phase cap | Setup reserve | Cleanup reserve | Total wall | Paid worst at¥2.76/h |
|---|---:|---:|---:|---:|---:|---:|
| Corrected original T30 build/save/cold AD |120s|4500s|60s|120s|4800s|¥3.68|
| Candidate T30 build/save/cold AD |120s|6900s|60s|120s|7200s|¥5.52|
| Saved-only matched T30,3pairs/8calls |120s|3300s|60s|120s|3600s|¥2.76|

All three stage maxima total15600GPU-seconds/4.333card-hours. Once the coherent
free scope is approved, proceed through stages and routine repair without
another permission question, while retaining its cumulative ledger. Each next
stage requires its predecessor's saved graph and a current card-availability
check. The table's hypothetical paid costs total¥11.96 at8×¥0.07 CPU/hour
plus¥2.20 GPU/hour; they are planning arithmetic only. This package has no paid
submitter or pending paid command to approve. Actual billing rounding would
remain independent in a later explicitly requested paid proposal.

Original inner probe wall4440s, lower600s/backend3600s/call180s. Candidate
inner wall6840s, lower600s/backend6000s/call180s. Saved-only inner wall3240s,
call180s,3pairs. Total wall governs assets, imports, graph serialization,
ABI/memory/cost checks, full result validation and cleanup; lower/backend/call
maxima are individual stopping bounds and are not a promise that all maxima
can be consumed and every later stage still completes.

The measured corrected T7 lower/backend are original176.017/2502.203s and
candidate324.936/5543.316s. Candidate helper compile phase took5996.788s;
the difference includes actual asset loading, serialization, cost/ABI/memory
checks, complete cold result and validation. Candidate backend6000s is only
8.24% above the measured T7 backend. T30 may exceed it, and success is not
promised. A deadline without a graph is a compile/resource limitation, not
evidence against the architecture or scientific strategy.

Measured matched T7 loads were original89.937s and candidate169.774s;
matched first calls37.402/1.674s and total occupied matched phase401.481s.
The separate T30 matched occupation preserves substantial room for both loads,
eight full AD calls, comparison of all arrays, failure-range receipts, result
serialization and process release. Compile phases cannot spend this reserve.
An additional run requires a new decision; a failed attempt never resets a
ledger or silently receives another7200 seconds.

## Executable preparation and recovery contract

The existing `scripts/hpc/probe_shared_latent_reader_T7_pair.py` is generalized
with explicit `--horizon 30`; horizon7 remains the default and retains the old
T7 schema and mathematical identity. T30 requires a separate
`shared_corrected_latent_reader_recurrent_pair_manifest_v1`, horizon30,
artifact742, exact native AD/precision/turnover pins and explicit external
execution authorization metadata. That metadata declares the separation of
science and funding; it is not human approval.

Prepared manifest:
`outputs/diagnostics/shared-latent-reader-20261004/latent-reader-T30-20261006.manifest.json`.
Its runtime paths remain under `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/`;
code workdir is `runtime/checkouts/shared-split-width-v1` at immutable
`c599236bcf57b745e9bafed678298010f4c3b58f` and overlays are in
`runtime/transfer/shared-latent-reader-20261004`.

Separate stages remain `compile-original`, `compile-candidate`, `matched`.
Save each sole successful executable and exact identity sidecar immediately,
before ABI or numerical admission. Save every complete result before checking
it. Successful graphs are recovered only by explicit load under the compile
guard; no successful candidate/original recompile is permitted. Matched
requires both corrected T30 graphs and must report model_compiles0/0.
Cross-architecture parity is not a gate; repeat each architecture against its
own cold-first complete result with unchanged tolerances. Numerical failures
are preserved while finite/physical valid timing continues through3pairs.

Current T7 successful graphs stay reusable:
`latent-reader-T7-20261005-compile-original.executable.pkl` and
`latent-reader-T7-20261005-paid-replacement-compile-candidate.executable.pkl`.
They cannot supply T30 shapes or serve as a T30 compile shortcut.

The new recurrent phase launcher and fresh approval-bound one-occupation
helper are separate from the historical T7 launcher/paid ledgers. They will
print actual target-shell argument expansion with `--dry-run`, require a fresh
external authorization receipt before true execution, retain separate caps and
receipts, and forbid reuse/reset of a phase output/ledger. One canonical
cumulative free ledger caps the entire scope15600 seconds, protects its
compile12000/matched3600 pools and records each occupation, including failed
attempts. Template approvals remain false and
cannot execute. Paid execution is outside this package; no script calls sbatch.

The helper records measured elapsed occupation time separately from the
conservative charged time, which includes a60-second finalization reserve.
Unconfirmed process-group or final selected-GPU release retains the full
occupation reservation and blocks another occupation. Release needs a valid
final snapshot with no compute PID and low memory, plus a gone process group;
raw final utilization is recorded because its sampling window can lag release.
The stricter start and post-preflight idle checks still require utilization0.
Budget refunds therefore depend on release evidence, not merely child exit.

## Concrete three-phase entry points

The launcher is deployed at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-latent-reader-20261004/run_shared_latent_reader_recurrent_pair.sh`.
Its default authorization receipt and sole cumulative ledger are respectively
`runtime/outputs/diagnostics/latent-reader-T30-20261006-free-approval.json`
and `runtime/outputs/diagnostics/latent-reader-T30-20261006-free-budget.json`
under the project root. Its preserved historical template has `approved=false`;
the current approved receipt funds these commands under the one coherent grant.

Run sequentially on gln01 from the declared immutable checkout. These review
UUIDs fix the expected output paths; ordinary retries use fresh UUIDs without
resetting the cumulative ledger. If a retry produces either saved graph,
the matched command must use that graph's actual successful UUID and identity
sidecar. The launcher rejects compile retries after a saved graph exists.

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-latent-reader-20261004/run_shared_latent_reader_recurrent_pair.sh compile-original --gpu-index 1 --cpu-affinity 8-15 --occupation-seconds 4800 --run-id 30610006-0000-4000-8000-000000000001
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-latent-reader-20261004/run_shared_latent_reader_recurrent_pair.sh compile-candidate --gpu-index 1 --cpu-affinity 8-15 --occupation-seconds 7200 --run-id 30610006-0000-4000-8000-000000000002
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-latent-reader-20261004/run_shared_latent_reader_recurrent_pair.sh matched --gpu-index 1 --cpu-affinity 8-15 --occupation-seconds 3600 --run-id 30610006-0000-4000-8000-000000000003 --original-executable /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T30-20261006-compile-original-30610006-0000-4000-8000-000000000001.executable.pkl --candidate-executable /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T30-20261006-compile-candidate-30610006-0000-4000-8000-000000000002.executable.pkl
```

Append `--dry-run` to inspect target-shell expansion without creating an
occupation or running the preflight/model. GPU1 is an initial example;
`--gpu-index 0` is the permitted alternate when independently found idle.
Every real occupation checks the selected card before the preflight and again
after it, and records its actual selection without changing the grant.

## Same-workload changing-update integration

After the T30 component is measured, prepare a separate complete-update
diagnostic. It executes exactly daily32 as eight distinct micro4 samples,
one complete T7 and one complete T30 per update, unchanged public daily/recurrent
loss weights, full parameter/state AD and public AdamW. Inputs, artifact order,
counts and two-card placement must match on both architectures. Parent
daily32+T7 and worker T30 reuse the existing schedule/IPC boundary. State
gradients are checked and saved but never enter AdamW.

Each architecture starts at its own declared baseline weights/optimizer
state, advances its own version through every complete update, and compares
each changed-weight result against an independent same-architecture replay.
Do not compare different architectures' parameter values or optimizer tensors
as numerical equivalents. Bind every branch to the same input parameter
version; retain component losses, complete mixed and component parameter
gradients, all32 daily states, both recurrent state gradients, optimizer
state, physical/mask checks and failure ranges. Preserve strict failures;
finite timing is not whole-model or trained-accuracy acceptance.

Interface changes needed are explicit, not already certified:

- Add a separate corrected component-admission view for the saved T7/T30
  pair reports/sidecars, whose identities and full ABI differ from historical
  `validate_original_admission`/`validate_candidate_admission` layouts. Accept
  preserved numerical_failed diagnostic evidence only in a clearly scoped
  performance route; do not change scientific admission to passed.
- Prepare corrected original and candidate daily micro4 native dual-AD graphs
  under the same guarded function. Historical daily graphs must undergo the
  same source identity check; old uncorrected binaries cannot silently enter
  a corrected full update. Compile any missing daily graph in its own bounded
  approved occupation, then load/save reuse it.
- Extend the original-only `probe_shared_original_full_state_schedule.py`
  interface through a separate architecture-aware diagnostic entry point;
  it currently rejects candidate manifests and hardcodes original64884481
  parameters. Use `shared_full_state_ad_execution.FullStateADEvaluator` and
  existing public serialized daily/recurrent update components instead of
  constructing new losses, cotangents or AD. Preserve state attachment and
  complete snapshots while selecting candidate66123713/355 leaves explicitly.
- Saved JIT arguments keep parameters dynamic. Load exact source/ABI identities
  once, then supply changed arrays without compiling per update; initialization
  provenance remains baseline provenance, not permission to freeze weights.
  Confirm branch versions and synchronize all returned AD outputs before
  timing. Keep startup/load, comparison/persistence, stable update time and
  card-hours as separate measured fields.

T7's1.2824× does not imply a28.24% complete-update gain. T30 has historically
dominated the two-card critical path. Measure complete changing updates after
all required corrected components exist; no complete-update speed is claimed
by adding component medians. Dataset accuracy, long-horizon sensitivity,
scientific acceptance, full-method adoption and the10x target remain open.

## Preparation verification

Seven focused CPU contract tests pass: legacy T7 defaults/schema, explicit
T30 schema/horizon and rejection of changed artifact, native AD, guarded
function or external authorization contract. No GPU model was constructed by
these tests. Exact T30 assets validated on test01 CPU with status
`assets_validated`, artifact742/horizon30 and0/0 model compilations; report
`latent-reader-T30-20261006.cpu-assets.json` is mirrored in the existing
diagnostic folder. A second CPU asset check verifies both generalized default
T7 mathematical identities exactly match the already completed T7 identities
after excluding the expected CPU/GPU device-kind difference; no model executes
or compiles. Receipt:`latent-reader-T30-20261006.T7-identity-compatibility.json`.
Twelve focused CPU budget/launcher tests pass, including protected pools,
unfinished-reservation blocking, failed release evidence, utilization sampling
lag at release, conservative finalization charging and changed/duplicate
internal-entry rejection. Their stdout is
`latent-reader-T30-20261006.cpu-tests.stdout.txt`. The three concrete commands
above pass target Bash `--dry-run`; full expansion is retained in
`latent-reader-T30-20261006.{compile-original,compile-candidate,matched}.dry-run.stdout.txt`.
The deployed direct internal entry rejects the false authorization template
before GPU query or preflight; receipt
`latent-reader-T30-20261006.direct-guard-rejection.stdout.txt`.
The preserved original template is
`latent-reader-T30-20261006-free-approval.unapproved-template.json`, with `approved=false`, no
approval UUID, timestamp or human reference. It grants no runtime execution.
The final consolidated receipt is
`latent-reader-T30-20261006.preparation-checks.json`; the same deployed helper,
launcher and monitor source versions are pinned in
`latent-reader-T30-20261006.phase-budget-checks.json`. Independent review
[passes the preparation controls](../../../outputs/diagnostics/shared-latent-reader-20261004/T30-free-preparation-independent-review.md).
All19 focused CPU checks (7 probe contracts +12 phase-budget fixtures) pass.
Preparation completed before the fresh human grant for the coherent
15600-second free scope. No T30 model execution, optimizer update or paid
submission occurred during preparation. Execution now proceeds continuously
under the approved receipt; preparation review itself was not authorization.
These checks establish
preparation and input/source binding, not CUDA, graph memory or scientific
acceptance. Existing deployed-runtime checks are reused where unchanged; an
actual newly authorized graph phase still performs the required small runtime
compatibility check once on its execution environment.
