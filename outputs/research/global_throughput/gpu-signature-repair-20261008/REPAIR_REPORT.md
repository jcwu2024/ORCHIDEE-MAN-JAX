# B256 AOT boundary repair and B256/B512 continuation

2026-10-08. No paid submission, requeue, training, real model compilation or model execution was performed. Work belongs to the global-throughput worktree; the old paid checkout, B128 stage and reference assets remain read-only.

## Established failure and remaining uncertainty

Job15600405 completed B128, then B256 lowering63.1790117s and compilation752.6141295s. Its first B256 call failed at JAX0.4.38 `MeshExecutable.call` with `Computation compiled for 673 inputs but called with 536`; it produced no B256 first-call output, gates or timings, and B512 never ran. The public compiled PyTree check did not reject the call. The confirmed proximate failure is disagreement between internal expected input aval metadata and the536 flattened caller leaves. The precise historical trigger that supplied673 metadata is unresolved; lazy-call misbinding is a tested possible mechanism, not a demonstrated historical cause.

A single fresh actual B256 lower-only inspection on canonical CPU JAX0.4.38 (`ibtest02`,180s cap,151518670848B available host memory) completed in123.37698850780725s. The actual nine groups had leaf counts350/14/1/4/10/20/60/38/39. Caller/public/all-argument lowered metadata each contained536 leaves, the public tree matched, and all536 native shape/dtype signatures matched. DCE retained417 inputs (maximum original index535). No `.compile()` or real transition call occurred. This rules out an inherently malformed fresh B256 argument builder; it does not reproduce the previous GPU/sequential-shape trigger. Canonical CPU/GPU JAX `pxla.py` have the sameSHA b4eb972a11e6e6195e7f43eb27968368d6b92f3045ba4140c4247384dc80ab2d.

## Implemented boundary

`benchmark_global_batch.lower_compile` audits public tree and all leaf shape/dtype/weak-type signatures plus available internal all-argument metadata before compilation, then again after compilation. It records admitted leaf/pruning counts. Internal metadata is never rewritten.

The returned `BoundAOTExecutable` retains that specific compiled stage and its own `CompiledCallParams`, calls JAX's standard `Compiled.call` directly, and delegates compiler-memory and other inspection methods. This bypasses only creation/use of the lazy C++ `Compiled.__call__` callable. Standard PyTree, aval, sharding/layout and tracer checks remain. No `keep_unused`, precision, process, physics, static identity, output, mask or tolerance changes were made. `run_two` still executes complete first block, feeds all resulting state into the second block, and synchronizes both outputs.

A canonical tiny sequential128/256/512 nested-AOT surrogate used536 caller leaves,417 active leaves and a separate673-input nested executable, with cache-clearing boundaries. Natural dispatch never failed. Fault injection deliberately attached the unrelated673-input lazy callable to the536-input public stage: it reproduced the exact historical error. The explicit bound call returned all417 outputs exactly at each shape despite that injected lazy-call fault. This verifies the defensive boundary, not the unknown historical cause or real-model GPU admission.

The Python dispatch overhead is included in new core and wrapper collection timings. Historical B128 used the old lazy C++ path: oldB128/newB256/newB512 curve timings are complete-workload observations under these recorded dispatch policies, not a pure isolated batch-size effect. No speedup is claimed from this repair.

## Exact continuation

`--continue-256-512 --b128-baseline-dir ...` constructs a distinct continuation identity/output directory and measures B256 then B512 only. It imports historical B128 timing/provenance as an explicit report baseline, without creating or relabeling a newB128 stage or compiling B128. The original manifest/envelope identity, full220429366B stageSHA e91db1391ccb5f525f9ec29f1cba24235c4ce9cd4153b9297fe2ed014d0c9cb5, complete515 strict gates,760 output leaves with512 lanes, both raw five-sample timing summaries and exact workload/execution provenance are verified. Reuse admission compares all declared source science/input/runtime/compiler/precision/workload fields. Only the already established loadavg tuple/list JSON telemetry equivalence is allowed; scientific arrays are never normalized. Original node/commit/environment/timing origin remains in `imported_B128_baseline`, separate from new stage records.

Canonical read of the actual old remote snapshot passed. B128 medians remain core29.617320833727717s/wrapper33.83803023956716s. Deliberately changed science and lane identities were rejected before any model compilation.

The fixed ordered512 lanes remain FULL64 repeated8, Days2–8 and9–15 at48 half-hours/day. Each new shape has a first complete synchronized warmup,5 complete core collections,5 complete wrapper collections, full outputs, every lane's reference comparison, strict repeat/permutation/replacement gates, and atomic stage sealing before the next compile. Finite cross-reference tolerance failures retain the existing exploratory labeling; hard/schema/nonfinite/internal failures still stop. No scientific acceptance is inferred.

B512 no longer requires a5% B256 speedup in continuation mode, but keeps >=1800s remaining, actual B256 compiler memory available, conservative device scaling with2GiB physical slack, and old actualB64 RSS scaling with8GiB host slack. Completed-stage reuse restores actual compiler-memory evidence for the guard. Old mode keeps its speedup condition. Optional B64 is forbidden for continuation. `slurm_global_gpu_capacity_continue.sh` is a distinct90min,4CPU,1GPU launcher with a new immutable checkout/output identity; its actual target Bash `--dry-run` expansion passed ontest02. No `sbatch` was issued.

## Checks and reproduction

The related local suite passed130 tests in15.06s; after adding two extra early-guard/scalar regressions, the final capacity suite passed20 tests in4.05s. The existing pytest cache_dir configuration warning is unchanged. `git diff --check` passed, and `git diff --name-only a33b82d -- jax_orchidee` was empty. These local checks use JAX0.10.1 and do not replace canonical-runtime evidence.

- `canonical-lower-only.jsonl` and `lower_only_diagnostic.py`: actual source-backed lower-only evidence, no compile.
- `canonical-diagnostic.jsonl` and `canonical_diagnostic.py`: canonical tiny sequential/fault-injection calls plus actual sealedbaseline read. The initial run executes copied helper definitions; it is explicitly not proof that final deployed files are identical.
- `canonical_diagnostic.py --code-dir /absolute/final/immutable/checkout`: reusable final-module diagnostic, imports actual repaired base/capacity helpers and `lower_compile`, records checkout commit plus exact module paths/SHA, and runs the same tiny and sealedbaseline checks. No pytest install is required. Run only on a free compute/test node with canonicalCPU Python, OMP/BLAS1, no caches, and a90s outer timeout. Its defaultbaseline path is the original readonlyjob15600405 capacitydirectory.
- `canonical-launcher-dry-run.txt`: actual target Bash argument expansion.
- `checks.json`: compact executed evidence and limits.

Final immutable code commit `9d19f0760b93e1186eaf796db4903ddfd16a806f` is deployed cleanly at `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v2`. Root ran `canonical_diagnostic.py --code-dir` ontest02 using the actual deployed helpers: all tiny sequential/fault checks and the sealedbaseline read/identity rejection passed onJAX0.4.38. [Final evidence](final-module-canonical-diagnostic.jsonl) records the commit and exact module paths/SHA. [Target argument verification](final-launcher-verification.txt) passed Bash syntax and actual expansion through Python dry-run, selected only256/512, imported noJAX and executed no model. The checkout stayed clean, olda51efcd checkout was unchanged, and the new runroot remains unused. [Deployment record](deployment-verification.json) preserves bundle identity and these checks.

Independent read-only review found no blocking implementation flaw and accepted preparation only. Remaining acceptance is a separately approved real GPU B256/B512 measurement; the original internal-metadata trigger remains unresolved. The [concrete resource/command proposal](PAID_RUN_REVIEW.md) awaits new approval, with no automation.
