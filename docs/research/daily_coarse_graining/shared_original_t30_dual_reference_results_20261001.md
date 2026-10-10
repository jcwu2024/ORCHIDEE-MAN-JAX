# Original T30 Dual-AD Reference

Status: passed. One original-field public T30 loss executable was compiled
with native parameter and initial-state derivatives on 2026-10-01. Its loss,
auxiliary values, and complete parameter gradient strictly bridge to the
accepted parameter-only T30 executable. The initial-state gradient was
serialized and validated as finite and nonzero. This creates the original T30
dual reference needed by later candidate comparisons; it does not compare the
state gradient against another dual-AD executable.

## Scope and identity

The run used artifact 742, the declared update-0 checkpoint, the prepared T30
case, unchanged FP32 neural / FP64 physical precision, and the existing public
recurrent objective. Native `jax.value_and_grad` differentiated arguments 0
and 1: all model parameters and the explicit initial state. The original field
decoder/backbone route and native recurrent checkpoint policy were retained.
The state derivative is an output of this reference and is never an optimizer
parameter. No optimizer update or HLO export ran.

Source checkout: `c599236bcf57b745e9bafed678298010f4c3b58f`; JAX `0.4.38`.
The accepted parameter-only bridge executable was
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-combined-recurrent-T30-20260926.candidate.executable.pkl`.
Its runtime identity and the update-0 checkpoint path are recorded in the
machine report.

The reference-only wrapper loads a private snapshot of the reviewed recurrent
probe and adapter. The probe snapshot SHA256 is
`7262061eaf3a8a687ac93acc4af625ee4eaaa54c936b2c5a34f33097103b01b4`; the
adapter SHA256 is
`0994034a976b2b60023ee477f62f6309975fcbc92fcbfd5653162640bccecc1c`. The
wrapper fixes the invocation to `T30/reference`; the original field path is
selected, so the adjacent normalized adapter is loaded for compatibility but
its candidate patch is not applied. The wrapper SHA256 is
`38eee2bc8b228d0fbe6afcf88ca3b878e22ce72bbb7a04b14e1e6c3236be224`, and the
launcher SHA256 is
`cd4af811cb315e7b59f20047df2f2705dae93ac0e1951089a783093086f2ac1d`.

## Numerical and resource result

| Check | Result |
| --- | ---: |
| Loss and auxiliary bridge | Pass; 20 coordinates, max abs `4.3667e-8`, relative L2 `1.2120e-9` |
| Parameter-gradient bridge | Pass; all 64,884,481 coordinates, max abs `2.6077e-8`, relative L2 `2.9648e-7` |
| Parameter-gradient validity | Pass; 300 leaves, 64,820,353 nonzero coordinates, norm `2.23017` |
| Initial-state gradient | Shape `(3854,)`; finite; 3,833 nonzero coordinates; norm `806824.03` |
| Physical hard counts | Pass; all counters zero |
| Estimated live executable memory | `3,490,683,231` bytes |
| Observed peak device memory | `3,514,478,592` bytes; below the 24-GiB gate |

The strict bridge covers the loss/auxiliary tree and all parameter gradients.
The parameter-only executable has no state-gradient output, so state-gradient
parity is not claimed. No repeated self or paired comparisons were requested
for this reference build.

Compilation took `172.664` seconds to lower and `2545.616` seconds in the
backend; the process report records `2858.282` seconds total. The launcher ran
from 09:34:47 to 10:22:39 on free GPU 1 (`Tesla V100-SXM2-32GB`) of
`ibc13b02n04`, for `0.798` observed GPU card-hours and no paid Slurm
allocation. At launch, CPU affinity was `0-111`, host load was `0.05/0.07/0.30`,
and both visible GPUs were unused. XLA emitted a slow-operation alarm for
`jit_checkpointed_loss_with_aux` after `29m4s`; compilation completed within
the 5,400-second backend cap. The launcher exit status was zero.

## Artifacts

The remote canonical artifacts are under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/` with
stem `shared-original-t30-dual-reference-20261001`. Local copies and the
private probe-source snapshot are retained in
[outputs/diagnostics/shared-original-t30-dual-reference-20261001](/E:/ORCHIDEE-MAN-JAX/outputs/diagnostics/shared-original-t30-dual-reference-20261001).

- [Machine report JSON](/E:/ORCHIDEE-MAN-JAX/outputs/diagnostics/shared-original-t30-dual-reference-20261001/shared-original-t30-dual-reference-20261001.json): SHA256 `befa6c26969d9d52ba67a882d65d6a14a44066bfe8d68d975802258526835828`.
- [ABI and state validation JSON](/E:/ORCHIDEE-MAN-JAX/outputs/diagnostics/shared-original-t30-dual-reference-20261001/shared-original-t30-dual-reference-20261001.validation.json): independently checks the saved identity, all parameter coordinates, and the `(3854,)` state gradient.
- [Serialized result](/E:/ORCHIDEE-MAN-JAX/outputs/diagnostics/shared-original-t30-dual-reference-20261001/shared-original-t30-dual-reference-20261001.reference.pkl): SHA256 `c141381741c4ccabb4f77ebeffebed114e1dda93e7383d3af85a01ae6b4c066`.
- [Serialized executable](/E:/ORCHIDEE-MAN-JAX/outputs/diagnostics/shared-original-t30-dual-reference-20261001/shared-original-t30-dual-reference-20261001.reference.executable.pkl): SHA256 `f3282a3935cfcf99feaf2e427b076c443284b28877d98f67cc95708d96c29f76`.
- [Launcher record](/E:/ORCHIDEE-MAN-JAX/outputs/diagnostics/shared-original-t30-dual-reference-20261001/shared-original-t30-dual-reference-20261001.launcher.txt): model process exit status 0.

The initial wrapper API mismatch exited before model compilation and is
preserved as [setup attempt 1](/E:/ORCHIDEE-MAN-JAX/outputs/diagnostics/shared-original-t30-dual-reference-20261001/shared-original-t30-dual-reference-20261001.setup-attempt1.txt).
