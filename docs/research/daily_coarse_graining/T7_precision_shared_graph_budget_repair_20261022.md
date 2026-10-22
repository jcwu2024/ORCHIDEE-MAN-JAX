# T7 shared graph budget repair — 2026-10-22

New implementation source **96978caf04c9a56273ea7e9b9031f57aebf999ac**. Prepared and locally tested only: no GPU, sbatch, redeployment, fresh execution approval or new paid allocation. Historical execution remains **f2242fc5a7657a3f3061d85a5e1b998663c8f944**, with its checkout and borrowed Git objects untouched. Teacher maintenance changes in the shared worktree are outside this commit.

Job15941021 ended FAILED124:0 after459s, with the sole lower attempt interrupted at the old300s cap; zero completed traces/compiles/calls. [Execution record](T7_current_model_precision_input_ready_execution_20261018.md) and `outputs/T7-current-model-precision-input-ready-execution-20261018/terminal/` retain original JSON/logs/sacct and byte-hash-verified recovery. This is a runtime budget failure, not a numerical verdict. Original input consumption passed; state20 rawNaN remains a separate strict failure. Release verified; estimated¥0.3162, actual bill unknown; one-shot consumed.

## Actual deadline chain and repair

`run_live` reads the real allocation start and fixes the outer plan clock to it. Generic `Supervisor` subtracts elapsed allocation time from5400s and sets work_deadline=total_deadline−180s. `PaidSupervisor` shares240s for startup/discovery/preflight. `CurrentSupervisor.phase(probe)` has a4980s ceiling (`5220−240`), itself clipped to remaining outer work time. The shell has an independent allocation-start+5220 work fallback and+5380 kill fallback, leaving time inside5400 for release.

Previously, the probe constructed base `Journal` after imports/GPU initialization with a fresh5220s timer. Base initialization also captured resources before starting that timer. Separate `deadline(300, lower)` and `deadline(3000, compile)` prevented a slow lower from borrowing compile time. The FP64 context/guard builds source and widens inputs but owns no deadline; its scientific logic and the common base Journal are unchanged.

The repair is scoped to the current-T7 subclass/entry:

- One CAPS owner in probe, reused by outer runner: `load300 + preflight240 + graph3300 + calls1380 + reserve180 = 5400`. Approval still binds exact argv, probe bytes, cwd/output/resources and complete stage_caps; old lower/compile approval fails against new graph caps. No existing approval or historical proposal was edited.
- At probe launch the outer owner passes `CURRENT_T7_PROBE_DEADLINE_EPOCH = now + min(4980, outer_work_deadline−monotonic_now)` through the wrapper and explicit Singularity environment. Probe remaining wall is the minimum of this phase end and allocation_start+5220, minus actual current epoch. Import/container/GPU-bootstrap time is therefore charged, not renewed at Journal creation. Missing/nonfinite original timing is rejected.
- StrictJournal keeps its work deadline across base resource_snapshot time, rearms only the remaining wall time, and clamps every phase to that work deadline. The300s load cap now includes source-cache setup, retained-input consumption, binding and prepared report/preflight admission as well as asset loading; this closes formerly untimed preparation work without changing consumers.
- Immediately before lower, `graph_deadline = min(now+3300, work_deadline−1380)`. It can be created only once. Both lower and compile use this same absolute monotonic deadline, including lower/HLO serialization, executable saving and memory/precision inspection. Phase names remain lower/compile so timeout evidence identifies the interrupted work. Entering compile cannot renew3300s. If lower takes700s, compile has at most2600s (possibly less after intervening overhead).
- Calls have at most1380s and are clamped to the same work deadline. The graph cannot borrow their reserved interval; neither inner phase nor startup can borrow the outer180s save/release reserve or exceed allocation5400s. Delayed preparation/bootstrap can reduce usable graph time. This does not promise all caps can be consumed after arbitrary overhead.

The generic base Journal, physical guard, objective, parameter/state ABI, four directions, six FD steps,52-call schedule, tolerance and scientific gates are unchanged. No capacity, precision policy, masking, Teacher or training change.

## Lightweight validation

Run from the fixed daily checkout, Python `C:/Users/admin/.conda/envs/ORCJAX/python.exe`:

```powershell
& C:/Users/admin/.conda/envs/ORCJAX/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/test_current_T7_precision_deadlines.py tests/unit/test_run_shared_current_T7_precision_admission.py tests/unit/test_shared_current_T7_precision_admission.py tests/unit/test_shared_current_T7_precision_shell.py tests/unit/test_prepare_shared_current_T7_retained_inputs.py --tb=short
& 'C:/Program Files/Git/bin/bash.exe' -n scripts/hpc/run_shared_current_T7_precision_admission.sh
```

**59 passed in5.08s**; existing pytest config warning `Unknown config option: cache_dir` arises because cacheprovider is deliberately disabled. Bash syntax and AST parse of both modified Python entry files passed. Tests use controlled clocks and two subsecond deadline subprocesses, no large graph or GPU. Coverage includes lower700s→compile2600s, nonrenewable graph start, late graph admission/call reserve, lower/compile stage-specific timeout with persisted partial JSON/exit124, exhausted phase rejected before work, calls clamped to remaining wall, allocation/outer phase/import/resource-snapshot accounting, actual Bash deadline export acceptance/rejection, exact argv/new caps approval acceptance and old caps rejection. Existing runner fixture was updated for the already-required retained-data-root argument; production parser requirements were not relaxed.

Evidence/validation receipt: `outputs/T7-current-model-precision-input-ready-execution-20261018/terminal/repair-validation.json`. Source self-review and scoped diff whitespace check passed; raw terminal output retains original whitespace/bytes. Existing input/scientific evidence reused, not rerun against a real model.

## Remaining limits

The interrupted lower provides no completed lowering time, compiled memory, full-model timing or numerical comparisons. A shared3300s budget may still expire or expose another issue; no success prediction is justified. Deployed GPU-runtime verification for this changed timing boundary has not run and requires a separately authorized future allocation/identity/approval. This repair is ready for actual-artifact review only; it does not authorize another job.
