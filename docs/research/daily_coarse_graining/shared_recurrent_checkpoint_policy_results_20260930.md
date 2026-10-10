# T7 Selective Checkpoint Policy: Real-Model Comparison

Date: 2026-09-30. The initial bounded compile screen below was inconclusive;
its repaired free-node continuation reached a real T7 result. The candidate
is not admitted under the current complete-derivative gate. No training or
production AD-policy change occurred.

## Question and prior evidence

The candidate retained the public T7 loss, saved artifact 742/window,
update-0 parameters, expert128/backbone source-reuse implementation and
native `jax.checkpoint` around the same scan transition. Only the checkpoint
policy changed from JAX's default to native
`dots_with_no_batch_dims_saveable`, which may retain selected dot outputs
instead of recomputing them. No model default, objective, precision, masks,
sampling, optimizer or physical process changed.

The historical [conditioned-attention comparison](conditioned_attention_derivative_modes_20260907.md)
reported 43.12 GiB temporary allocation for a different whole-window
no-remat reverse graph on a 32 GiB V100. It is not a memory estimate for this
candidate. The whole-window no-remat configuration was not rerun. The current
[forward/AD result](shared_forward_reverse_results_20260930.md) found that
checkpointed native AD plus possible recomputation accounts for most of the
T7 evaluator's marginal cost, but did not identify pure reverse or neural
versus physical shares.

## Initial bounded attempt

On free `gln01` (`ibc13b02n04`), GPU0 was empty at each start; GPU1 was also
empty. Initial host load was 0.81/0.70/0.87 for the reference and
0.00/0.12/0.59 for the screen. CPU affinity was 0-111; this was no exclusive
node allocation. The JAX 0.4.38 deployed synthetic AD/runtime preflight
passed in 58.119 s before real-model compilation. One GPU was visible.

| T7 stage | Intended graph | Lowering | Last durable stage | Result |
| --- | --- | ---: | --- | --- |
| Reference | Default checkpoint, full parameter and initial-state gradients | 173.884 s | Backend compile | Remote process status unrecorded; no executable/result |
| Screen | Selective policy, admitted parameter-gradient interface | 257.144 s | Backend compile | Remote process status unrecorded; no executable/result |

The reference stage had a 540 s Python-process cap after preflight; the
smaller screen had a 420 s cap. The local SSH tool reported exit 1 for both while
their incremental JSON remained `running` at `compiling`. The preserved logs
contain launch/preflight output but no Python exception or compiler diagnostic.
There was no completed `memory_analysis`, GPU result, all-parameter comparison,
initial-state gradient, physical gate, or warm timing from either real-model
graph. No active GPU or compiler process remained afterward; `nvidia-smi`
reported both GPUs empty. The available kernel log showed no contemporaneous
OOM or segfault entry; system journal access was denied. The exact exit
mechanism is unresolved. These outcomes must not be called OOM, numerical
failure, or proof that the selective policy is slower at steady state.
The [exit diagnosis](shared_recurrent_compiler_exit_diagnosis_20260930.md)
shows that this SSH tool also reports 1 for controlled remote 124 and 137 exits;
neither historical remote process status was captured.

The reference dual-gradient graph lowered in 173.884 s and then did not
complete compilation within this bounded attempt. The parameter-only screen
lowered in 257.144 s and likewise produced no compiled result. The screen
did not proceed to T30, a changed full update, or any training. The prior
admitted parameter-gradient executable remained the only completed reference
at that point; no candidate policy had been accepted.

## Completed free-node continuation

The launcher now persists `*.exit` with the remote phase and actual process
status, and allows 7,200 s for each real-model stage after a 180 s deployed
synthetic preflight. The reference preflight, model and launcher returned 0.
The default-policy dual-gradient graph lowered in 174.487 s and compiled in
2,544.641 s. It reproduced the admitted parameter-gradient executable:
20 combined loss/auxiliary coordinates and all 64,884,481 parameter gradients passed
`rtol=1e-5`, `atol=1e-6`; all 3,854 initial-state gradients were finite and
14 physical hard counts were zero. The reference compiled estimated-live
memory was 3,488,289,277 bytes (3.249 GiB).

On the same saved T7 input, the selective-policy dual-gradient graph lowered
in 185.996 s and compiled in 2,649.163 s. A subsequent cache-reused process
reported 154.352 s in its compile call. Loss/auxiliary and all parameter
gradients passed the strict gate against the saved reference. All 14 physical
hard counts were zero. The initial-state gradient failed at **5/3,854**
coordinates (indices 33, 34, 199, 385, 1024); the largest absolute difference
was 0.311649, while relative L2 was 3.188e-8. The failed coordinates and
their individual allowed differences are in the
[state-difference JSON](../../../outputs/diagnostics/shared-recurrent-checkpoint-policy-20260930-v3-T7-state-difference.json).
The candidate's compiled estimated-live memory was 6,769,885,997 bytes
(6.305 GiB), about 1.94 times the reference, within the 24 GiB gate.

Both saved executables were reloaded for three interleaved warm pairs. Median
T7 **dual-gradient diagnostic** times were 1.574541 s reference and
1.502453 s candidate, a 1.048x ratio. The measured gain is about 72 ms for
this interface; it is not a parameter-only training-branch or complete-update
measurement. The replay's cumulative device peak after the pairs was about
6.56 GiB, not an isolated per-arm observed peak. Replays of the saved
reference and candidate results each exceeded the strict coordinate tolerance
at one or two coordinates across processes, so exact per-coordinate numerical
stability is unresolved even within an arm. This observation does not erase
the direct candidate/reference five-coordinate failure.

The selective policy therefore stops at T7: it fails the existing complete
state-derivative gate, increases estimated live memory substantially, and has
only a small measured dual-gradient T7 speed gain. No T30 compilation or
original-mix full-update integration is warranted for this candidate under
the current contract. This decision rests on executed model results; the
initial compilation timeout alone could not have established it. The default
checkpoint policy remains unchanged. This route used free `gln01` GPU time,
approximately 1.878 single-card hours across the bounded attempts, with CNY 0
paid charge. The retained reference/candidate executables and reports remain
under the project `runtime/outputs/diagnostics/`.

Evidence: [default reference](../../../outputs/diagnostics/shared-recurrent-checkpoint-policy-20260930-v2-T7-reference.json),
[first candidate](../../../outputs/diagnostics/shared-recurrent-checkpoint-policy-20260930-v2-T7-candidate.json),
[cache-reused candidate](../../../outputs/diagnostics/shared-recurrent-checkpoint-policy-20260930-v3-T7-candidate.json),
and [interleaved replay](../../../outputs/diagnostics/shared-recurrent-checkpoint-policy-20260930-v5-T7-replay.json).

## Initial resource and repair record

Paid cost was CNY 0; no Slurm allocation was submitted. At most one V100 was
used at a time. The declared process limits bound the two real-model stages
to 540+420 GPU wall seconds, plus the observed 58.119 s preflight, or less
than 0.283 one-GPU card-hours excluding small launcher overhead. This is a
cap-derived exposure bound, not measured kernel time or an exact elapsed
accounting. Startup and compilation did not yield warm performance. Neither
actual candidate peak nor an isolated memory comparison was measured.

The initial screen had no measured model outcome, so it could not establish
whether the policy helped or hurt execution. The admitted combined T7 graph
had previously taken 222.17 s lowering and 2548.82 s backend compilation;
an earlier graph took 5406.94 s. Those observations motivated the repaired
7,200 s stage cap. Its completed measurements and stop decision are above.

Evidence: [reference JSON](../../../outputs/diagnostics/shared-recurrent-checkpoint-policy-20260930-T7-reference.json),
[screen JSON](../../../outputs/diagnostics/shared-recurrent-checkpoint-policy-20260930-T7-screen.json),
[preflight](../../../outputs/diagnostics/shared-recurrent-checkpoint-policy-20260930-T7-preflight.json),
[probe](../../../scripts/hpc/probe_shared_recurrent_checkpoint_policy.py),
[launcher](../../../scripts/hpc/run_shared_recurrent_checkpoint_policy.sh),
and adjacent `.log` copies under `outputs/diagnostics/`. The original overlay
and originals remain under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-recurrent-checkpoint-policy-20260930`
and `runtime/outputs/diagnostics/`. Two local policy/validation tests passed;
Python syntax, target Bash syntax, and actual reference/candidate/screen
`--dry-run` expansion passed. No files in the immutable model checkout changed.
