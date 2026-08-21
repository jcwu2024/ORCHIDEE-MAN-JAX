# Target-Oriented Attention GPU Admission

Status: **accepted zero-update real-v6 GPU admission**.

Date: 2026-08-21.

## Decision

`target_oriented_hierarchical_coordinate_attention_v1` is admitted to a
bounded training design. This result proves that the complete 83,755,173-
parameter candidate can execute and differentiate on one accepted Teacher-v6
sample through the canonical daily-operator boundary. It is not evidence that
the randomly initialized model is scientifically accurate, and it does not
promote a checkpoint.

The run used one free `gln01` V100 32 GB, GPU 0, one process, and a 90-minute
timeout. It performed zero optimizer updates and wrote no checkpoint. It did
not regenerate Teacher data, access historical v5 data or sealed-test rows,
reconstruct 48 half-hour forcing records, or execute a 48-step state scan.

## Identity

- Repository commit: `7c6371c6b5e29e34816cd122944b6546b7c1775c`.
- Dataset manifest SHA256:
  `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`.
- Target-oriented objective contract SHA256:
  `7c9857c0ca57ee65cfcf7514a216a217c3b1661f6cc869c6dfb85d3786bfce5c`.
- Architecture manifest SHA256:
  `9ea64a449f51dd384555978a80816f8f0895b601dd1d0d10c717095119115265`.
- Admission report SHA256:
  `a92e9415110d16b3a0a24ce3278514c694441b43d6ccf3d5245a5e63ad13803b`.

The canonical server report is
`runtime/outputs/training/target-oriented-attention-gpu-admission-7c6371c/report.json`.
The repository evidence copy is
`outputs/target-oriented-attention-gpu-admission-7c6371c-report.json`.

## Result

- Backend and device: `gpu`, `cuda:0`.
- Complete output: 181 fields and 39,371 scalar coordinates.
- All 6,674 declared-defined output values were finite in eager and JIT
  execution.
- Eager forward took `224.77 s`; JIT compile plus first forward took
  `176.43 s`; hot JIT replay took `0.06197 s`.
- Full loss/gradient compile plus execution took `624.99 s`.
- All 83,755,173 gradient elements were finite. The base encoder, domain
  refinement, science cross-Attention, science-token mixer, primary heads,
  state heads, and auxiliary heads all had finite nonzero gradients.
- The bound canonical transition compiled and ran in `218.94 s`.
- The next state had 5,736 declared-defined finite values.
- Water and carbon budget residuals were both exactly `0.0`.

The admission proxy loss was `1152.7393`. This value is expected to be large
at random initialization and is deliberately not an acceptance metric. The
test only required a finite, differentiable, complete real-v6 path.

## GPU Memory

JAX reported a peak `bytes_in_use` of 2,676,487,168 bytes, approximately
2.49 GiB. `nvidia-smi` showed approximately 25 GiB because JAX reserved most
of the visible device memory in its allocator pool. The larger number was
reserved capacity, not live tensors. The GPU returned to 0 MiB after the
process exited.

The admission therefore establishes comfortable single-sample execution
headroom on a 32 GB V100. It does not yet establish the training batch size:
optimizer state, saved activations, and 7/30-day recursive trajectories must
be measured in the training-step preflight.

## Next Gate

The first bounded training design is now frozen and its exact microbatch loss
reductions pass locally. It uses only real Teacher-v6 train/train rows and the
already frozen objective:

```text
L = 1.0 * one_day_state
  + 1.0 * seven_day_science
  + 1.0 * thirty_day_science
  + 0.1 * necessary_process_supervision
```

The next operation is its multi-point-year prepared-batch reader followed by a
free-V100 full-step preflight. Before paid submission, measured training-step
memory and timing must complete the requested resource, wall-time, and
worst-case cost declaration. A 365-day rollout, Gate D2, sealed-test access,
and product acceptance remain closed.
