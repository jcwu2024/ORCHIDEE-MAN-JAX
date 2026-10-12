# Execution-chat bounded acceptance

This delivery repairs shared AOT metadata handling and supplies reproducible
input/interface evidence for supervisor review. The checked source is
`scripts/dev/benchmark_global_batch.py`, SHA256
`bc8ce1d62f6947228c8c8438344987345bf18095f307c90a2f45dc4655c6891e`.
The exact file was imported on Linux, with `scale.base` bound to that candidate.

The frozen payload SHA256 is
`be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`.
The old helper reproduces the float `.shape` error on the complete first
scalar argument leaf sequence, at `.values_by_component[4][67]`. The repair
uses JAX `shaped_abstractify` consistently for caller values and abstract avals;
it preserves tree/shape/dtype/weak-type construction admission.

The [exact-candidate diagnostic](linux-jax-0.4.38-candidate-full-metadata.json) and its
[harness](aot_signature_diagnostic.py) record JAX/jaxlib 0.4.38, x64,
ibtest01 and CPU affinity `[0]`. Raw/device scalar calls agree for the same
value, and native tree/shape/dtype mismatch rejection remains. Actual
SingleRunner scalar and B2 core/wrapper paths execute both continuous blocks
for frozen lanes 0 and 4 with the full state carry. Their shape/dtype match;
42 weak-type differences are observed. The strict audit rejects a weak-only
mismatch while native `Compiled.call` accepts it. No new runtime weak gate was
added. Only the physical transition factories were replaced by small functions.

Both first-block and second-block groups have all 64 distinct indices checked:
536 leaves per lane, matching raw/device trees, signatures and weak counts.
The metadata traversal device-puts raw scalar leaves only. The final JSON's
SHA256 is `976b3cf45c3621249b212a9aad3f11c95c297244bdd9876523905b1a49b6ff0c`;
its recorded interpreter and supporting commit match the declared CPU runtime
and immutable v2. The diagnostic schema v3 describes this evidence extension;
the paid execution version remains v2.

The [local checks](LOCAL_CHECKS.md) record six passing AOT-subset tests,
passing targeted Ruff, and the full local module's 50 passes / 10 failures at
the existing science-source identity guard before AOT. All four added
regressions pass; the full-module run includes the raw-scalar regression.
The guard was retained.
Independent source/artifact review found no blocking implementation issue.
The earlier helper-injection JSON is preserved separately as preliminary
evidence; acceptance relies on the exact-candidate result.

The working diff contains no physical model, Fortran, reference or trace
changes. Old v1/v2 execution evidence and immutable v2 remain preserved.
Target diagnostic imports use immutable v2 support code at
`b1b5cf0835da298bddb388e61d754feeb6c14cf2`; the research branch's current science
is not substituted into that historical CPU comparison.

Full physical-model lowering/compilation, scientific acceptance, B64/48-core
execution and throughput remain unverified. This delivery performs no GPU or
Slurm submission, installation, production deployment or paid retry. A future
execution pin/proposal has not been prepared; the consumed v2 grant stays closed.
