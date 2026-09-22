# Dynamic projection reuse: bounded execution admission

2026-09-23. User authorized continuing the execution optimization after the
cost/memory audit. This is not a new architecture or training experiment.
No paid submission is authorized or requested by this protocol.

## Single change

Add opt-in `gather_dynamic_reuse` to the shared Transformer. Compute identical
axis and local-state K/V projections once per invocation and decoder block,
then gather them inside the existing coordinate chunks. Cache only within the
differentiable invocation. Preserve all weights, initialization, widths, masks,
normalizations, output shapes, state/parameter gradients and physical processes.
Keep `gather_reuse` as the default and comparison path.

Do not combine this with query packing, backbone source projection reuse,
decoder rematerialization, precision changes, or loss changes. Local-state
reuse keys canonical indices/sentinel; axis reuse keys expert/slot/role/position.
Inactive queries must preserve zero-token/bias/mask semantics. Shared projection
does not authorize information transfer across process ownership masks.

## Local checks

Use a bounded two-block model to compare each output and its parameter/state
derivatives separately, including carbon/GPP routing, local-state mappings,
undefined/sentinel and inactive PFT cases, permutation and padded chunks.
Reuse the previously declared projection-execution tolerances: forward
rtol2e-5/atol2e-6, derivatives rtol5e-4/atol2e-5. These are fixed before testing.
No full production parameter initialization or Teacher regeneration is needed.

## Deployed checks

Run `check_daily_training_runtime.py` once on the deployed JAX/CUDA environment
before changed real-model gradients. Reuse that runtime evidence across this
same candidate's component runs. Exercise actual shell arguments with dry-run.

Source: saved shared update100; artifact742; same32 daily samples and first
T7/T30 windows as the completed microbenchmark. Original reference gradients,
losses/hard counters and discarded AdamW proposal are already preserved in
`runtime/outputs/diagnostics/shared-staged-source100-v1`. Read and validate their
existing source identities. Do not rebuild costly references or run cache-only
diagnosis as a prerequisite.

`probe_shared_projection_reuse.py` supports independent daily(micro4), T7 and
T30 components, then a joint replay. Each component compiles only the candidate,
checks the existing24GiB compiled-live limit, compares loss/hard counters and
all75,578,497 gradient coordinates to its saved reference, and immediately saves
the completed result before later work. Gradient/forward tolerances are those
above; hard counters must agree and be zero. Record exact maximum and relative
gradient errors, not just pass/fail. One warm execution and three timed replays
retain no optimizer updates. Report temporary/argument/output memory separately
from cumulative allocator peaks and pool reservation.

Save each compiled executable with its source/microbatch/JAX identity using the
installed JAX serialization API. This is a same-runtime diagnostic artifact,
not a portable model release. A tiny roundtrip check precedes relying on it.
The joint replay loads all three admitted executables, computes all three
components on every replay and makes exactly one full-network AdamW proposal.
Compare every parameter and optimizer coordinate to the saved proposal at
rtol1e-5/atol1e-6; discard it. Saved gradients are numerical references, never a
substitute for computation inside full-update timing.

The first deployment uses free gln01 only after checking GPU availability:
one daily component process with90min ceiling; a second process may run T7/T30
sequentially with120min ceiling after local tests pass. They can run on separate
free GPUs and write separate reports/component files. Joint replay follows only
after all component gates pass; its initial ceiling is25min. No paid fallback
or automatic long training. Preserve completed files if a later stage fails.

## Decision and limits

Require numerical/physical agreement and bounded live memory before adopting
the execution path. Both memory and full-update time matter; a memory reduction
alone is not speed acceptance. Free-node timings are diagnostic and prior paid
timings are cross-process context, not a matched-allocation speedup claim.
Default execution and the500-step continuation stay unchanged until results are
reviewed. Physical equivalence and speed do not demonstrate improved learning.

## Larger-microbatch closure

Both earlier paid candidates completed their probes with `resource_limited`:
micro8 final compiled-live40.33GiB, micro16 final78.01GiB. Neither candidate was
executed. micro16 job15329520 elapsed1:18:57, compilation3226.34s(53.77min),
tariff estimate3.63yuan. This closes that sweep without an OOM execution or a
new speed result. micro4 remains the fastest admitted current execution.

## Local implementation evidence

ORCJAX JAX0.10.1: all4 focused decoder tests passed (forward/static56.08s,
parameter/state gradients168.91s), including nonzero projection biases and
all-PFT-inactive gradients. Existing shared Transformer/microbatch suite19passed
in80.50s; component/reloaded-executable/complete-AdamW plumbing3passed in2.72s.
These bounded tests do not substitute for the deployed JAX0.4.38 CUDA check or
full75,578,497-parameter gradients. Default execution remains unchanged.

Deploy one shared-object checkout under`runtime/checkouts/shared-dynamic-reuse-v1`.
It depends on`runtime/checkouts/shared-staged-large-v1/.git/objects` and that
repository's existing alternates. Preserve those object stores while the new
checkout is active. Store candidate components in
`runtime/outputs/diagnostics/shared-dynamic-reuse-v1`; do not overwrite references.
