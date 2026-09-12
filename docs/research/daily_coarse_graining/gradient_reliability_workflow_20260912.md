# Reduce avoidable training and gradient-test failures

User requested a process-level remedy while continuing the numerical diagnosis.
This is a working procedure, not a claim that all remaining errors are solved.

## What the recent failures actually show

- Local ORCJAX currently has JAX0.10.1/NumPy2.4.6; the accepted GPU environment
  has JAX0.4.38. The scalar float0/np.void AOT failure passed local tests but
  was reproducible in seconds on the actual GPU runtime. Local success was
  incorrectly treated as sufficient API/runtime coverage before a long compile.
- Prepared retained inputs have a window axis, not a window/day axis. The
  actual-data CPU trace caught this runner layout mistake. Synthetic arrays
  alone were not enough to validate the data boundary.
- Putting direct, checkpointed and streamed compilations in one90-minute task
  without saving full intermediate gradients turned a time limit into lost
  useful work. Durable stages now prevent that loss; separate elapsed/compile/
  evaluation times prevent misleading duration estimates.
- Teacher validity and PFT-derived state routes need one owner. Repeating loss
  extraction or cotangent assembly in every runner makes omissions more likely.
  The public rollout component and shared science extractor now own those paths.
- The current finite, small native/streamed discrepancy is a different category.
  It is not yet a proven implementation defect or harmless rounding. Initial
  biomass reserve sensitivity dominates it. More file-integrity checks cannot
  answer that numerical question.

## Concrete execution rule

Before an expensive new derivative graph, run
`scripts/hpc/check_daily_training_runtime.py` on the same Python/JAX/CUDA
runtime used by that graph. The small synthetic check covers scalar bool/int
float0 AOT arguments, T1/2/3/7, float32/64, state-dependent fraction, changing
same-shaped targets/context, public adjoint cache reuse, weighted microbatches
and the real AdamW routine. It reads no Teacher arrays or bulk hashes.
Run it once for a relevant code/environment change or new diagnostic chain,
not once per training batch. This takes seconds to tens of seconds and does
not imply real-model or scientific acceptance.

Then validate any changed actual-data boundary with a real-row shape/forward
check. Changes only to logging/order/persistence do not justify recompiling an
unchanged billion-operation graph. Keep the numerical graph separate from its
orchestration and compare the cheapest meaningful real case before scaling.

Persist a completed expensive reference before compiling another method.
Use independent result files, bounded stages and reproducible case identity.
Stop/restart from complete stages; never overwrite failed evidence as if it had
passed. Report exception, resource limit, numerical mismatch and scientific
fit failure separately. A strict comparison threshold is a diagnostic; use
independent JVP/FD, conditioning and optimization relevance before diagnosing
its cause or justifying a new tolerance. Do not merely loosen a threshold to
turn a red status green.

Continue independent one-day learning while multi-day correctness is unresolved.
Evaluate whether native JAX checkpointed AD supplies the needed bounded memory
without host adjoint maintenance; choose from measurements rather than adding
more manual derivative machinery by default. Architecture, training-data and
loss changes remain scientific decisions, not automatic responses to a runner
exception. Final paper training still follows the fresh, executable protocol.

## Scope and accountability

The runtime check cannot cover every physical regime. Real-model numerical
conditioning and long-horizon science still require work. The practical aim is
to catch cheap interface mistakes before long computation, preserve partial
evidence when something fails, and spend substantial runs on scientific
questions. This adds no user approval round for already authorized free checks.
Existing explicit paid-resource approval rules remain unchanged.
