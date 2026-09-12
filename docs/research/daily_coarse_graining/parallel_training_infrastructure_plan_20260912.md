# Parallel next work: broader one-day data and a public multi-day derivative boundary

The user approved these two tracks after reviewing the current streamed
implementation. Existing fresh-start model choices remain:109,464,375 neural
parameters, all trainable, fresh seed rather than inherited exploratory
weights, immutable v6 data, DOC input conditioning and gsmean applicability.
No new paid job is authorized. Job15005758 remains canceled.

## A. Broader one-day sampling

Reuse the accepted prepared artifact pools, including the2000-package
1001-3000 extension. Implement seeded epoch shuffling of artifacts and row
positions, with bounded artifact mixing per batch to avoid decompressing a
different full package for every sample. Each eligible package-row is used
once per prepared-pool epoch. Report distinct Teacher point/year/day identities
separately from row draws, because different packages can contain the same
Teacher day. Preserve remainder rows and reproducible resume positions.

Fixed train-only monitors must be excluded by row identity and audited for
physical-day overlap before any claim of held-out behavior. This is not a full
v6 epoch or a sealed validation set. Sampling itself creates no Teacher data.
Root will measure actual one-day batch memory/read time before freezing a
larger training budget. Admission, useful training and monitoring should share
one process/allocation when dependencies permit.

## B. Public rollout differentiation component

The existing `StreamedAdjoint` already owns daily VJP accumulation, but callers
still assemble observation cotangents and the legacy step injects only15.96M
parameters. A public component should own forward trajectory, loss reduction,
observation cotangents, direct parameter/initial-state loss derivatives and
the backward recurrence. Reuse the existing reverse core; keep its legacy
parameter-only return compatible while exposing the initial-state adjoint.

The generic daily step takes params, continuous state, discrete state and a
dynamic day input, and returns next state/discrete/observations. The continuous
state uses the canonical flat array. The component supplies its state
observation, and the model adapter supplies required science observations,
including the current state-derived PFT fraction. Loss callback inputs remain
dynamic; no per-batch arrays embedded as new compiled constants.

Provide ordinary scan and checkpointed-scan reference losses with standard
JAX AD/JVP, plus a public streamed value-and-gradient operation. Its supported
derivatives are parameters and initial continuous state; discrete/day/target/
context arguments are explicitly fixed for that streamed method. Physical
parameters can be included in the parameter tree when the model adapter
declares their routing. Do not silently claim arbitrary forcing/context
derivatives. Defer `custom_vjp`: the host execution strategy is not directly
JIT-transformable and a custom VJP prevents direct forward-mode JVP through
that wrapper. Preserve an independent native reference regardless of later
packaging. Segmented checkpointing is a measured option, not a30-day cutoff.

Local tests cover T1/2/3/7 (and cheap longer toy recurrence), parameters and
initial state, direct parameter loss terms, terminal loss, state-dependent
fraction, masks, discrete propagation, changing targets and cache reuse.
Compare direct/checkpointed/streamed gradients, JVP duality and central FD.
These tests verify algorithm/plumbing, not full-model multi-day admission.

Root then binds the current canonical head/updater/retained tail with the
entire neural tree and freezes a short real-v6 resource/AD probe. Start with
T1/2 before T3/7. Compile memory must be measured before executing an oversized
reference. No full-network7/30-day training before real-model admission.

## Ownership and delivery

Two explicitly authorized GPT-5.6 Sol/max agents implement the isolated sampler
and public derivative component/tests. Root owns canonical-model binding,
data IO/protocols, CPU/GPU execution, global docs and integration. No agent
submits jobs independently. Both tracks can progress without waiting for the
other; larger paid budgets remain a separate concrete approval.
