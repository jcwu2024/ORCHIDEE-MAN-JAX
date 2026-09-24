# Direct executable loading: two-update integration

Status: completed; unchanged two-update execution and full checkpoint parity
pass, 2026-09-25. Original wrapper exited1 solely because its scalar-report
comparison mistook NumPy-versus-JSON dtype differences for a numerical failure.
Offline correction of that check passes without another model run or changed
tolerance. Preserve both original and `.corrected.json` reports. No paid work
or long learning. This follows deserialize-only49083 and profile31906.

## Result

- Graphs ready including runtime/data preparation:530.72s (8.85min), versus
  1727.4s (28.79min) in the prior cache-hit profile: about69.3% less startup.
- Both complete changing-data updates pass the existing numerical/physical
  checks. Warm second wall15.63s, versus16.11s previously; one observation does
  not establish a sustained update-speed improvement.
- All64,884,481 parameter coordinates pass comparison; max absolute5.96e-8.
- All129,768,963 AdamW-state coordinates pass; max absolute1.86e-9.
- All48 scalar update-record values pass after host dtype normalization;
  max absolute4.77e-7, same rtol1e-5/atol1e-6. Checkpoint tensor dtype checks
  remain strict. No bitwise-equality requirement was added.
- Actual allocator peak12,323,709,184bytes (11.48GiB).
- Original learning subrun completes at568.2s; wrapper including final
  checkpoint loading/comparison completes at587.8s. These are different timers
  and phases, not time-zone differences.

Direct loading is now verified for this bounded same-source/same-method
technical route. It does not lower first-ever compile cost or solve throughput.
The next execution candidate is exact-shape scalar-field grouping, screened
on a small helper before any full-model compile; simple K/V concatenation
yielded only8.36% helper gain with73.4% more temporary memory and is not promoted.

## Implementation and evidence

`probe_shared_serialized_updates.py` replaces only the existing learning
runner's `compile_gradient_evaluators` factory. It loads the three trusted
saved native JAX executables and returns them to the same microbatch
accumulator, physical checks, joint-gradient composition and AdamW updater.
It does not reconstruct reverse passes, lower the three model graphs, export
HLO or inherit trained weights. Existing initialization and the two changing
TRAIN artifacts742/275 are reused for a matched replay.

This initial probe is deliberately scoped to the original c599236 checkout,
the same full method record and one visible V100/JAX0.4.38. It is not yet a
general cross-version/device cache or a default for formal500-update learning.
The new entry script lives in runtime/transfer; the original checkout and
saved reference reports/checkpoints remain unchanged.

Admission compares both updates' loss terms, component gradient norms and
clipping, all64,884,481 resulting parameters and the full AdamW state with
profile31906. Tolerances remain rtol1e-5/atol1e-6 from the existing execution
benchmark. Parameter/moment comparison is broad update evidence, not an
independent new directional derivative test or scientific fit acceptance.
The original numerical/physical checks still execute for every component.

Four local tests pass, including native serialization of a small reverse
function and its use across changed parameters/data. The same tests pass on
test02 for the initial three-case suite using the deployed JAX0.4.38 GPU
environment with CPU backend in30.37s. The added fourth test catches the
NumPy/JSON scalar-type mismatch and still rejects actual value changes.
One pytest cache-configuration warning is unrelated to numerical execution.
Ruff and target-shell syntax/actual --dry-run argument expansion pass.

Run details:

- Launcher `scripts/hpc/run_shared_serialized_updates.sh`, copied to runtime/transfer.
- Free gln01 GPU0, wrapper50963, started02:31:05CST; maximum30minutes.
- At launch both GPUs empty; host load0.01/0.06/0.14, affinity0-111.
- Original checkout `runtime/checkouts/shared-split-width-v1`, c599236.
- Reports/log under runtime/outputs/diagnostics:
  `shared-split-serialized-updates-20260925.json`, `.updates.json`, `.log`.
- A separate bounded attention-helper probe52409 uses freeGPU1 while this
  run loads its executables. Its small test is not long model training.

Graph-ready comparison excludes profile31906's extra1527s diagnostic export.
The full integration and earlier6.45min deserialize-only result have different
preparation scopes and should not be compared as identical startup measures.
