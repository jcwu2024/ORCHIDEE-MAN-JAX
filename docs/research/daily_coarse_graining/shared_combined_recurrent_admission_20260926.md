# Combined execution: bounded recurrent admission

Status: T7 completed0/passed; T30 launcher29812 running on freeGPU0 since
approximately11:29CST September26. Same immutable overlay/code and60minute
main ceiling; actual T30 CPU inspection passed52.90s, target GPU shell dry-run
passed and both GPUs were idle before launch. T30 report/log use the stem
`runtime/outputs/diagnostics/shared-combined-recurrent-T30-20260926`.
No paid allocation.

## Completed T7 result

Launcher97821 completed0, all64,884,481 parameter-gradient coordinates pass
unchanged rtol1e-5/atol1e-6, maximum difference7.450580597e-8 and relativeL2
3.824144558e-7. Loss and19auxiliary coordinates pass; all14physical counters
are zero. Warm median original1.815831535s, combined1.588351592s:
1.143218x throughput,12.53%less time. This is a same-process sequential
comparison with candidate compilation between the two measurements.
Compiler live estimate1.835856->3.248723GiB, still below24GiB. Lower222.17s,
backend compile2548.82s(42.48min), main probe2971.43s(49.52min); startup
before its internal timer is additional. Saved original/candidate results
and candidate executable remain available. Local result/runtime JSONs are
in `outputs/`. The gain is useful but well below the user's throughput target.

## Historical T7 launch

T7 started on freeGPU0, launcher97821, at00:33:49CST September26.
Local implementation1f3622b has7focused tests passing, Ruff/shell syntax clean.
Actual test02 inspection passed in57.80s: artifact742 case/objective binding
and both staged adapters loaded, no gradient compile or Teacher preparation.
Actual CPU/GPU target-shell dry-run arguments pass. Both GPUs were idle and
host load0.01before dispatch. Remote model checkout remains immutable c599236;
new scripts/adapters are staged in `runtime/transfer/shared-combined-recurrent-20260926`.
Inspect `runtime/outputs/diagnostics/shared-combined-recurrent-T7-20260926.{json,log}`
for the completed T7 result. T30 follows the active status above.

Actual daily
combined execution has passed all loss/gradient/physical comparisons and
matched replay measures1.339x throughput; no recurrent or full-update claim.

## Purpose and unchanged method

Test whether expert128 decoder scheduling and common-source backbone K/V
reuse preserve the existing native checkpointed seven-day calculation and
improve its measured execution time. Only after T7 admission, test T30 in a
separate process. Network weights/capacity, masks, retained exact physics,
all active outputs and the fixed `(flux + stock) / 9` horizon contribution
remain unchanged. Do not shorten or detach state gradients.

Use immutable c599236, the saved fresh update0, artifact742/window0, the
existing `shared-split-dual-inputs-c599236.pkl` cases and the compact daily
context's head/template/state/science scales. No Teacher reads/preparation
or new dataset generation. Original shape-specific executables are in
`shared-split-profile-c599236.assets/{T7,T30}.executable.pkl`.

## Execution and checks

Load only the selected original horizon executable. Evaluate and persist its
full loss/auxiliary/parameter-gradient result before compiling the candidate.
Reuse that saved reference on recovery. Scope both candidate process-head
bindings through lowering/compilation; restore them on exceptions. Candidate
uses the shared `make_shared_recurrent_objective` and native JAX AD. Save its
serialized executable and full result before numerical comparison.

Require unchanged1e-5/1e-6 loss/auxiliary/all-parameter-gradient comparisons,
zero physical hard counts, finite values and compiler estimated live memory
within24GiB. Report complete synchronized warm horizon times separately
from source imports, reference loading, lowering and backend compilation.
This tests a recurrent parameter gradient with full native state feedback;
it is not independent initial-state cotangent certification.

One free GPU, at most60minutes after a3minute runtime preflight. Original
T7/T30 backend compilations historically took about90minutes each, so the
new graph may exceed this bound. This is an explicit resource risk, not a
numerical failure. The candidate's reduced daily compilation/work does not
guarantee equal improvement in recurrent AD. Preserve completed references,
logs and binaries if the ceiling is reached; investigate the actual stage
before any retry. No new paid allocation is approved.

## Next decision

If T7 passes, reuse the same code and inputs for T30 with its own reference
and candidate artifacts. If either fails numerical or physical checks, retain
the discrepancy and localize it without relaxing the gate. After both pass,
compare complete same-data updates, including gradient accumulation, AdamW,
parameter/optimizer state, transport and one/two-GPU wall time. These
component measurements alone cannot certify training throughput, learning,
long-term scientific accuracy or the10x research goal.
