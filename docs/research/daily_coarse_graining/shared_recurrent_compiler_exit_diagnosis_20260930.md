# T7 Compiler Exit Diagnosis

Date: 2026-09-30. This is a read-only diagnosis of the two bounded free-`gln01`
attempts, plus a small non-model shell test. No T7/T30 graph was rerun.

The reference launch began at 13:29:48 local cluster time. Its synthetic GPU
preflight passed in 58.119 s; the reference JSON was last written at 13:33:48
with `stage=compiling`, after 173.884 s of lowering. The model-process cap was
540 s. The screen began at 13:52:54; its JSON was last written at 13:57:17
with `stage=compiling`, after 257.144 s of lowering. Its cap was 420 s. Neither
JSON contains a compiled executable, result, or final status. The logs have no
Python exception or compiler diagnostic and were last written near launch or
preflight. These timestamps are compatible with several exit mechanisms; they
do not establish that either cap fired.

The original launcher ran with `set -e` and redirected output through
`exec > >(tee ...) 2>&1`. Its later `model_process_exit_status` print was added
after these attempts. The local PowerShell SSH tool reported `exit_code: 1` for
controlled remote `exit 124`, `exit 137`, and `exit 1`. A small test on `gln01`
using the same `set -e`, `tee` process substitution, and `timeout` pattern ran
`timeout --signal=TERM --kill-after=1s 2s sleep 4`: an EXIT trap printed
`inner_exit_status=124`, while the local SSH tool again reported `exit_code: 1`.
Thus the prior `1` is only the local tool result, not the remote model exit.

The node uses cgroup v1. A later SSH session belonged to a different
`session-28314.scope`; its accessible root memory cgroup reported
`memory.failcnt=0`, but that is not a historical process counter. Account
process accounting and system journal were unavailable; available kernel log
had no contemporaneous OOM or segfault. No T7 process or GPU use remained when
checked, and the cache directory held only small `jit_iota` entries. Neither
timeout, OOM, compiler failure, nor external termination is established.

The launcher now writes a `*.exit` sidecar at launch and records timestamp,
phase, and true shell status from an EXIT trap. For a normally exiting model
command, it also prints `model_process_exit_status`. A hard kill may leave only
the start record, which is itself evidence that a normal EXIT trap did not run.
This instrumentation does not repair or validate the candidate checkpoint
policy. Any future real-model retry requires a separate bounded reason; no
paid job or model recompile was part of this diagnosis.

Evidence: [reference JSON](../../../outputs/diagnostics/shared-recurrent-checkpoint-policy-20260930-T7-reference.json),
[screen JSON](../../../outputs/diagnostics/shared-recurrent-checkpoint-policy-20260930-T7-screen.json),
[prior result](shared_recurrent_checkpoint_policy_results_20260930.md), and
[launcher](../../../scripts/hpc/run_shared_recurrent_checkpoint_policy.sh).
