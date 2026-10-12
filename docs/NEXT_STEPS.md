# Current Work Order

Updated: 2026-10-12. First-day localization completed both graphs/24 JVPs.
Finite, defined next-state tangents differ for every direction, including control 200;
raw state coordinate 20 is nonfinite. Numerical exit 3, GPU released, scope
closed. Raw-neural v3-final saved-evidence audit is accepted with active numerical
failures; 24 results saved/GPU released/scope closed. Handoff complete; stop.

The later project-maintenance authorization establishes the fixed daily checkout
at `runtime/worktrees/daily-coarse-graining` under the root workspace. See
[workspace management](WORKSPACE_MANAGEMENT.md). Commit/push and checkout
maintenance do not reopen any scientific run or closed GPU scope below.

## Main task and final acceptance

Improve T1/daily, T7 and T30 effective training throughput before formal
training. The10x target is cumulative research. Final acceptance uses changing
weights, the same GPU count and daily32/eight distinct micro4 plus one T7 and
one T30: public losses, complete parameter/initial-state gradients, physics,
masks and AdamW. State derivatives do not enter AdamW but remain a gate.
Keep strict `rtol=1e-5, atol=1e-6`, finite values and24GiB. No automatic
candidate adoption, reference/tolerance change or formal training is authorized.

User sequencing decision (2026-10-11): focus daily research on training
throughput while the main-branch Teacher may still receive small changes.
Use existing, version-pinned Teacher/prepared assets for performance and
correctness comparisons; keep their identities fixed within each comparison.
Main Teacher stabilization and daily execution optimization can proceed in
parallel. Formal training waits for those Teacher changes to settle and the
new versioned Teacher dataset to be ready. Before training, check its schema,
units, masks and prepared-input/normalization compatibility, then perform the
bounded admission checks appropriate to the changes. Small Teacher changes
do not automatically invalidate execution work or certify its compatibility.
This sequencing decision does not reopen closed GPU scopes or authorize a
new GPU experiment, long training, paid work, or Teacher regeneration here.

## Effective evidence

| Evidence | Result and limit |
|---|---|
| [Forward/AD timing](research/daily_coarse_graining/shared_forward_reverse_results_20260930.md) | Matched medians: forward2.104821s/18.57%, additional gradients/recompute9.228212s/81.43%; AdamW0.02–0.03s. Pure reverse and exact whole-update partition remain unmeasured. |
| [Original complete-update scheduling](research/daily_coarse_graining/shared_original_full_state_schedule_followup_results_20261005.md) | Same two cards9.810149→7.647413s/+28.28%; state gates fail. Specific common-leaf-mass cancellation repair is independently closed. No complete admission. |
| [Selected latent components and T7 B4](research/daily_coarse_graining/shared_latent_reader_results_20261004.md) | Corrected T7/T30 component throughput improves about28%. Same742/275/482/299 T7 inputs: serial5.025831→B4 1.967053s/2.555005x. Loss/all parameters/aux/physics/masks/memory pass; states fail9/9/11. Corrected daily/full updates remain unmeasured. |
| [Saved-B1 synchronization](research/daily_coarse_graining/T7_saved_B1_dispatch_results_20261009.md) | 1.024075x over three pairs, state cross/self fails; separate scope closed. Keep helper for later whole-workload use, without polishing this small gain now. |
| [Saved-state/input analysis](research/daily_coarse_graining/T7_state_repeats_analysis_20261009.md) | Three sample299 B4/B1 ranges stay separated beyond observed B1 variation. Inputs FP64/normalization FP32; selected masks valid/no DOC tail. None of four selected coordinates lies on an initial field/axis extrema tie; later recurrent ties and derivative contributions remain unproved. |
| [Current JVP completion](research/daily_coarse_graining/T7_B1_state_JVP_completion_results_20261009.md) | 32 B1/B4 JVPs/four full VJPs complete; cycles repeat exactly. B1/4–B4 fails at 434/2382, passes 2621/control 200; own JVP/VJP and full-state gates fail. Full parameter self repeats, primal/aux/physics/memory pass. Reverse repeat variability alone does not describe the whole discrepancy; cause and derivative reference remain unresolved. |
| [First-day localization](research/daily_coarse_graining/T7_first_day_JVP_localization_results_20261010.md) | Next-state tangents fail 200/16/14/16 coordinates for input 434/2382/2621/control 200 in every cycle. Finite primals/observations and exact auxiliaries pass; raw state 20 nonfinite remains a separate failure. Differences occur before later recurrence/loss; no cause or derivative reference/adoption inferred. |
| [Accepted raw-neural result](research/daily_coarse_graining/T7_raw_neural_JVP_results_20261010.md) | Active tangent coordinates fail 1769/10/36/21 for 434/2382/2621/control200; four active primal coordinates fail. All 161 masks exact; 16 nontrivial repeats exact and 12 B4 isolation checks pass. Numerical gate failed; scope closed/released. |

No complete-update optimization has passed every gate. Fixed-weight four-T7
batching is a component screen, not a changed formal sampling mix. T30 trace
and source MAC shares do not isolate pure AD/module time or a launch bottleneck.
V4/V7 direct substitution stays stopped; T30 chunk-vmap and decoder128 remain
unselected. A precision/reference change is not selected by the JVP result.

## Current action and next gate

The [single-GPU terminal result](research/daily_coarse_graining/T7_paid_single_GPU_execution_results_20261012.md)
owns the latest paid status: sole approved n04/4CPU/1V100 job15812634 at
frozen c8a44c3 ended FAILED2:0 after6 allocated seconds. The first
`nvidia-smi -i 5` CSV query returned6/“No devices were found”; no XML,
actual-job cgroup, Supervisor, preflight, CUDA/PJRT or affine12/91 ran.
Numerical status is not_executed; OS opening denial is not assessed.
Scope closed/consumed, mutex absent; live GPU cleanup/peak memory unobserved.
Allocation0.0016666667card-hours, additive estimate CNY0.0041333333; bill unknown.
No retry, mapping query, repair/redeploy, free fallback, monitoring or training.
The [preparation/proposal](research/daily_coarse_graining/T7_paid_single_GPU_preparation_results_20261012.md)
retains historical false scope, exact argv and88 CPU cases; these do not
establish GPU or scientific acceptance.

The [paid v3 terminal result](research/daily_coarse_graining/T7_affine_paid_GPU_recovery_results_20261012.md)
owns historical paid v3 execution status. Direct human “批准。” authorized the exact58a2d938
single fixed gnall n01/4actualSlurmCPUs/1V100/20min/nonexclusive/no-requeue job.
Sole job15806646 ended FAILED2:0 after7s. CSV→UUID XML succeeds on n01
(driver525.85.12/minor0); actual-job cgroup wildcard GPU access fails the unchanged
single-device-isolation gate. Exact devices.list/configuration cause was not
retained; no Supervisor/preflight/CUDA-PJRT/affine12/91 ran. Numerical status is
not_executed, not a numerical failure. Allocation0.0019444444card-hours,
estimated additive CNY0.0048222222; actual bill unknown. Scope closed and consumed,
mutex absent; no live post-exit GPU cleanup observation. That old scope grants no repair-and-rerun,
replacement, monitoring, card change, tolerance relaxation, free fallback or training.
Old v2/f4ae evidence remains; free b676 stays paused/approved/unconsumed.

The [v3 preparation/proposal](research/daily_coarse_graining/T7_affine_paid_GPU_recovery_proposal_20261012.md)
retains frozen58a2d938, Python30/shell10 and actual test01 CPU admission, source/
recovery-input manifests and original false scope. Those preparation snapshots
remain historical; latest approval, opened/closed identities and consumed attempt
are in the terminal result. No numerical repair or speed/adoption acceptance follows.

The [accepted preparation](research/daily_coarse_graining/T7_affine_paid_GPU_execution_proposal_20261011.md)
retains scheduler allocation-clock1200s/preflight240s/probe780s/release≥180s/
24GiB, exact actual Slurm/cgroup/kernel/UUID and CUDA/PJRT gates, physical
constraints and strict numerical comparisons. Prior136 compatibility cases
and historical CPU12/91 are not this dispatch's GPU acceptance. Actual device-isolation/
runtime/numeric gates remain unverified. Free b676 approval
is preserved/unconsumed and dispatch stays paused. Cooperative mutex/read-only
management admission does not mechanically constrain legacy arbitrary copies.

The previous accepted [early summary_input capture package](research/daily_coarse_graining/T7_summary_capture_results_20261011.md)
completed the authorized eight-file read-only retrieval, real thin capture
and unchanged CPU affine probe. Exact TRAIN742/275/482/299 window0/day0 inputs
produce FP32 summary `(4,257,4)`; raw434 sample3-only preprocessing JVP is
admitted. CPU JAX0.10.1 passes12 measurements/91 saved arrays, all independent
oracle/AD/isolation, eight cross and nine repeat checks. Eleven focused tests
pass after a Windows historical-PosixPath deserialization compatibility fix;
the first platform exception and prior missing-source `not_executed` receipts
remain historical evidence. That capture source was `70619b0a...901072`,
execution HEAD `9de13774...b497`; its identities stay separate from new recovery.
The supervisor accepted that seven-file stage, committed/pushed as `048c963...1738`;
outputs and eight historical assets remained outside Git.
Preserve retrieved assets and strict gates; no later-layer graph, GPU wrapper/
bootstrap, remote computation, training or automatic repair is part of it.

The separately authorized [local numerical preparation](research/daily_coarse_graining/T7_batch_numerics_local_preparation_20261011.md)
is accepted by the supervisor as completed local methods and preparation:
bounded input source audit, independent affine method/identity admission and
retained nonfinite rejection evidence.
Input6/matrix6 local method tests pass; the historical real capture/CPU result above remains evidence. No production
repair is selected. The [GPU discrimination plan](research/daily_coarse_graining/T7_batch_numerics_GPU_confirmation_plan_20261011.md)
now has a deployed bounded wrapper; target CPU shell/private-phase checks pass,
while container/PJRT/GPU verification remains pending. The earlier local
method stage was accepted/committed; this completed follow-up grants no new
GPU run, training or reopening of closed scopes, and establishes no original-
model repair, complete-update speedup or long-term scientific acceptance.

The [current GPU preparation report](research/daily_coarse_graining/T7_batch_numerics_GPU_preparation_results_20261011.md)
and [execution contract](research/daily_coarse_graining/T7_batch_numerics_GPU_execution_contract_20261011.md)
bind only fresh `outputs/T7-batch-gpu-preparation-20261011/summary-recovery.v1.*`:
NPZ `bf36a098...6129`, receipt `f863a2d9...0e00`, unchanged capture source
`70619b0a...901072`. The test incident removed the original two capture files
and changed source; source restoration and one authorized thin recapture are
documented separately. All six numeric hashes match independent preincident
evidence; original NPZ/receipt bytes were not restored. The existing CPU12/91
result was not rerun. The original proposed scope remains an unchanged historical
`execution_authorized=false` snapshot. The local package was supervisor accepted
and versioned in its explicit13-file scope. Actual
commit/push identity and final scope/deployment v2 are recorded in the
[postcommit delivery receipt](../outputs/T7-batch-gpu-preparation-20261011/postcommit-delivery.v1.json).

The [current deployment result](research/daily_coarse_graining/T7_batch_numerics_GPU_deployment_results_20261011.md)
owns the later execution status: the user replied “批准。” on2026-10-11 for
one nonexclusive idle gln01 V100, CNY0, a separate1200s one-shot
(preflight240/probe780/release reserve180),24GiB and unchanged strict math.
Accepted b676353 code and the fresh two-file recovery input were deployed to
an independent immutable checkout. Actual target Bash→CPU Python dry-run and
owned Linux CPU phase-group termination pass; the v1 harness parser failure
and v2 correction are preserved. Actual container/PJRT/GPU readiness remains
unverified. The single final post-readiness resource refresh found both cards
at31707MiB/100% with external compute processes. No card was selected and no GPU
launch/preflight/open scope/ledger reservation occurred; occupation/card-hours/
cost are0 from no launch/reservation. This is resource non-admission, not an
execution or numerical failure. Status is `approved_waiting_resources`; approval
persists and this one-shot is unconsumed. Stop here without further resource
polling, monitoring, reservation, retry, paid fallback or renewed approval request.

The [accepted result](research/daily_coarse_graining/T7_raw_neural_JVP_results_20261010.md)
reproduces active differences in the separately compiled canonical neural prefix,
before physical transforms/update. First differing operation, integrated-graph
cause and B1 derivative truth remain unproved. Batch-dependent FP32 contractions
are a source/HLO hypothesis; native JAX mapped scalar execution is a future
discussion candidate with correctness/speed unmeasured. No repair is selected.
Keep strict gates, unchanged precision/weights, complete fields/masks and AD.
The one-V100 scope closed at2169.713308/5400 seconds/0.602698 card-hours/CNY0,
exit3; two graphs/24 JVPs, zero VJPs/updates. The saved evidence and
[handoff](research/daily_coarse_graining/daily_speed_handoff_20261010.md) are complete.
Stop as explicitly directed:
“不自动开启下一轮 GPU 诊断或长训练，不自行移动目录/切根分支/清理 runtime。”
No automatic next graph, scope question, precision/reference choice or adoption.

## Completed scope and assets

The [first-day scope](research/daily_coarse_graining/T7_first_day_JVP_localization_results_20261010.md)
closed at 2124.572662062943/5400 GPU seconds/0.5901590727952619 card-hours/CNY 0,
actual exit 3, GPU released. Two graphs compiled once each, 24 JVPs,
zero VJPs/updates; no retry. Unused 3275.427337937057 seconds grant no new run.
Independent NumPy analysis and seven target guards preserve failures; no
full-T7 loss/parameter VJP/fourteen-hard-gate or update-speed acceptance.
The [original/B4 recovery](research/daily_coarse_graining/T7_state_JVP_results_20261009.md)
closed at 2809.490106/3600 free GPU seconds/CNY 0. The separately approved
[B1 completion](research/daily_coarse_graining/T7_B1_state_JVP_completion_proposal_20261009.md)
closed at 1780.7436698693782/3600 seconds/0.4946510194081606 card-hours/CNY 0,
zero updates. Numeric status is `numerical_failed`, ledger child exit 3;
[exit/release receipt](../outputs/diagnostics/shared-latent-reader-20261004/T7-B1-state-jvp-completion-20261009.exit-release.json)
confirms GPU 1 at 0 MiB/0%, no compute processes and completed PIDs absent. Independent
test01 NumPy analysis completed in 4.87128541059792s, exit 0, six guards passed.
Exact measurements/commands/receipts and native/scaled comparisons are in the
[completion report](research/daily_coarse_graining/T7_B1_state_JVP_completion_results_20261009.md).

Preserve both successful JVP graphs, saved VJPs and full arrays remotely under
`runtime/outputs/diagnostics/shared-latent-reader-20261004/`. Compact analysis
and CPU receipts are local under the same project-relative output directory.
Keep the accepted source archive and false/true approval records. Completed
commands must not be rerun; no Teacher regeneration.

All earlier windows and unused balances remain closed/separately accounted.
New paid work requires approved resources, time, directory, command and
worst-case CPU+GPU cost. Architecture/capacity, precision/reference, loss,
sampling, tolerance or recurrence choices need an explicit scientific decision.
Work only in jcwu's remote tree; never compute on login nodes.
