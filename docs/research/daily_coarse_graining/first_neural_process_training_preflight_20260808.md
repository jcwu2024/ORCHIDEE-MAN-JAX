# First Neural Process Training Preflight

Status: **accepted through the final no-checkpoint training preflight**.

Date: 2026-08-08.

## Bound Identity

- probe source commit: `492f2c049241d482c19c116a8750cc724ce01d75`;
- candidate contract SHA256:
  `59bd5da7ea7eef78e151aef5339167ed640a12c7700d74ef62e744049a8218e6`;
- training protocol SHA256:
  `3196d64033c4204db84eee38e40bd5bbbab5ccf2a6355b33e2cd4583cacce7c3`;
- v6 manifest SHA256:
  `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`;
- v6 acceptance SHA256:
  `8784ed42dc64ef77c9642f613eb2e5940635edc6e744a1af4ea8634293baa8e3`;
- base-statistics JSON/NPZ SHA256:
  `ab7f040a930071bd5604c17355a6493b23f3c76cba1e3084dcd5d5653101a4bf` /
  `e88d7cec504c6051aab6c9feacb619dca94961730ffa18f840563a48fc815a3a`;
- process-statistics JSON/NPZ SHA256:
  `9bc2cab43b3279e39a0fc2876ca298b2b371d6890986d44ef035859ca925eaab` /
  `addf957afbd8338f4aa250155dcd299d568c1070b644e30279f9bf0ea37f9bf8`.

The accepted report is
`runtime/outputs/training/first-neural-process-head-v1-training-probe-streaming-492f2c0/training_probe.json`
on Explore1000. Its SHA256 is
`127ef3289af776d5c5bb994f9129f5d220aa8b8574b18c50e7e8a77aba739277`.

## Protocol And Runner

The protocol status is `ready_unsubmitted`. Both arms use the same
initialization, deterministic train/train sample order, batch size 16,
10,000-update budget, AdamW schedule, and three non-test validation slices.
The host loader performs one vectorized target extraction per annual shard,
keeps canonical assembly inside the JIT graph, and bounds prefetch at two
batches.

The production runner writes atomic latest/best checkpoints and reports every
250 updates. Checkpoint identity binds the source commit, arm, seed, candidate
contract, protocol, dataset, acceptance, and both statistics artifacts.
Resume rejects identity drift. Test splits cannot be selected by the training
or validation iterators.

Local Ruff, format, compilation, E1, training-foundation, and new-runner tests
passed; the combined focused suite reported `47 passed`. The production Slurm
launcher also passes `bash -n`, and the runner-only suite reports `6 passed`.
ShellCheck was unavailable on the workstation and is not claimed.

## Final GPU Probe

The probe ran directly on GPU 0 of `gln01`, an idle 32 GiB V100, through the
accepted project GPU environment and dynamic host-driver bindings. It wrote no
checkpoint. Each arm performed three compile/warmup updates, 20 measured
same-batch updates, 100 real changing train/train batches with prefetch two,
and four batches from each declared validation slice.

| Arm | Same-batch s/update | Streaming s/update | Streaming finite | Validation seconds | Selection score |
| --- | ---: | ---: | --- | ---: | ---: |
| `axial_process_operator_v1` | 0.09603 | 0.16998 | yes | 26.93 | 0.65777 |
| `factorized_process_control_v1` | 0.09129 | 0.17141 | yes | 26.21 | 1.05273 |

All streaming losses, gradient norms, and clip scales were finite. Each
validation slice contained exactly four batches/64 samples, and
`test_split_evaluated=false`. The candidate/control validation values are a
bounded plumbing probe after only 123 diagnostic updates; they are not formal
model-selection evidence and do not satisfy any promotion margin.

## Resource Bound And Active Gate

The streaming measurements project about 28.3-28.6 minutes for 10,000 update
steps alone. Formal runs also include cold compilation, 40 complete
three-slice validation panels, and atomic checkpoint I/O. The conservative
request is therefore one V100, four Slurm CPUs, and a two-hour wall-time limit
for each arm/seed task. No memory directive is required by the measured
32 GiB V100 fit or the current cluster contract.

The active six-task mapping is:

| Array index | Arm | Seed |
| ---: | --- | ---: |
| 0 | `axial_process_operator_v1` | 20260807 |
| 1 | `factorized_process_control_v1` | 20260807 |
| 2 | `axial_process_operator_v1` | 20260808 |
| 3 | `factorized_process_control_v1` | 20260808 |
| 4 | `axial_process_operator_v1` | 20260809 |
| 5 | `factorized_process_control_v1` | 20260809 |

At the two-hour limit, one task costs at most CPU CNY `4 * 2 * 0.07 = 0.56`
plus GPU CNY `1 * 2 * 2.20 = 4.40`, total CNY `4.96`. All six tasks cost at
most CPU CNY `3.36` plus GPU CNY `26.40`, total CNY `29.76`, assuming CPU and
GPU charges are additive.

No paid submission is authorized by this report. After explicit approval,
the array must begin with index 0, then a small batch, before the remaining
indices. Formal training success still requires post-training hard physical
admission and the declared 7/30-day rollout comparison. No 365-day,
long-held-point, sealed-test, Gate D2, or product-acceptance work is open.
