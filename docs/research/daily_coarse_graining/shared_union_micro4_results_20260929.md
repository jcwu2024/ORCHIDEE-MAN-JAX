# Original-four union micro4 comparison

Approved paid job **15461046** completed `0:0` in **13:39** (819s).
The original-four production daily parameter-AD interface passes bounded
numerical/physical admission. The small measured warm difference and increased
memory do not justify production adoption or another paid union full-update run.

## Fixed scope and execution

- Artifact742's original rows0/1/2/3: all590 prepared sample leaves match the
  compact32-row bundle, with four distinct row fingerprints; update0 checkpoint.
- Unchanged 64,884,481-parameter 768/eight +384/two candidate, FP32 neural/FP64
  physical, public mean daily objective, original auxiliary and all300 parameter
  gradients. Native state differentiability remains; state gradients are not
  returned or independently validated by this measurement.
- Matched baseline: admitted source-reuse +expert128 **field** serialized
  binary; zero new field graph compiles. Candidate: scoped opt-in union with
  the same source-reuse backbone. Canonical saved reference is additional truth.
- `gnall`, one node/task,8 allocated CPUs, one V100,64GiB requested host memory,
  nonexclusive,30min cap. Actual node `ibc13b03n05`, affinity `24-27,80-83`,
  visible GPU0, launch load `96.00/96.01/93.00`; launch GPU-process snapshot was
  empty. No fully idle eligible node had been available. This shared-host context
  does not establish a cause for compilation cost.
- JAX/JAXLIB0.4.38; deployed synthetic GPU runtime check passed in76.157s.
  One GPU **0.2275 allocation card-hours**; estimated additive charge
  `(8*0.07+2.20)*819/3600 = CNY0.6279`, not a bill (approved maximum CNY1.38).
  Slurm step MaxRSS7.738GiB; accounting omits memory from AllocTRES, while
  ReqMem/MinMemory reported64G.

## Gates and matched timing

Strict `rtol=1e-5`, `atol=1e-6` passes for scalar loss, all160 auxiliary
coordinates and all64,884,481 parameter-gradient coordinates against both
combined field replay and canonical saved truth. Max gradient delta is
`1.1921e-7` versus combined and `3.5763e-7` versus canonical. Both routes have
all11 hard counts zero, finite meaningful gradients, norm approximately10.3940
and64,862,209 nonzero gradient coordinates. The legacy untagged canonical
reference was accepted only at the admission-linked path with exact identity.

After these gates and explicit warmup, three synchronized interleaved pairs:

| Repeat / order | Field seconds | Union seconds |
| --- | ---: | ---: |
| 1, field then union | 0.374004 | 0.369003 |
| 2, union then field | 0.374802 | 0.369663 |
| 3, field then union | 0.373125 | 0.368760 |
| Median | 0.374004 | 0.369003 |

Measured median ratio **1.01355x**, or **1.337% lower wall time**. This is one
bounded daily micro4 comparison, not a robust general speedup, optimizer/T7/T30,
complete-update throughput, scientific accuracy, or the10x target.

Both24GiB gates pass, but estimated live memory rises10.565 ->11.916GiB
(+12.782%). JAX `peak_bytes_in_use` (active buffer peak) rises
11,362,754,816 ->12,812,827,648B, or10.582 ->11.933GiB (+12.762%).
`pool_bytes`/`peak_pool_bytes` remain18,251,513,856B (17GiB) in both routes;
the active-buffer increase is not a measured total-VRAM residency increase.
Cold costs are separate: Python imports/device2.714s; inputs/references2.215s;
field binary read0.303s/deserialization12.638s/first call13.817s; union
lower100.252s/backend579.554s/first call0.535s. Explicit field/union warmup
0.498/0.372s. Time to valid measurement721.854s; probe725.082s. Historical
combined compile timing is a separate run, not a matched compile-speed ratio.

## Artifacts and decision

[Full small report](../../../outputs/diagnostics/shared-union-micro4-paid-20260929-15461046.json),
[runtime report](../../../outputs/diagnostics/shared-union-micro4-paid-20260929-15461046.runtime.json),
and [launch log](../../../outputs/diagnostics/shared-union-micro4-paid-15461046.log)
are local. Preserve remote full assets under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/` with stem
`shared-union-micro4-paid-20260929-15461046`: `.candidate.executable.pkl`about91MiB,
`.candidate.pkl`about248MiB and `.field-replay.pkl`about248MiB. Earlier references,
row0 full-state admission and caches remain intact.

Close this micro4 round. Keep union opt-in and field default. Return to larger
structural execution costs in the training bottleneck, reusing existing
full-update/recurrent evidence. No further paid job, training, tolerance,
precision or architecture change follows automatically. The independent
component27-coordinate failure remains separate. A small saved-field
directional state-gradient gate remains required before production adoption;
micro4 parameter parity does not cancel it or establish full state parity.
