# Teacher Capture GPU Benchmark (2026-07-21)

## Scope

This benchmark measures Teacher label capture for one paper PFT14 landpoint on
an Explore1000 V100. It does not measure neural-network inference and does not
claim scientific readiness for a learned surrogate.

Environment:

- JAX 0.4.38 with float64 enabled;
- one V100;
- 1962 state initialized from the audited 1961 year-end state;
- complete pre-daily training boundary: 19,838 target elements and 350 final
  state leaves;
- one persistent Python process for each cold/hot pair.

## Results

| Capture implementation | Block | Cold | In-process hot | Hot per requested day |
| --- | ---: | ---: | ---: | ---: |
| Per-leaf host slicing | 7 days | not retained as acceptance | 8.681 s / 15 days | 0.579 s/day |
| One device transfer per stacked tree | 7 days | 903.142 s | 4.840 s / 15 days | 0.323 s/day |
| One device transfer per stacked tree | 28 days | 922.590 s | 7.631 s / 29 days | 0.263 s/day |

For both retained optimized runs, repeated targets and final state were
bitwise identical. The local three-day comparison against daily Teacher
capture also passed the existing `atol=1e-8`, `rtol=1e-10` state/target gate.
The normal compiled production block regression remained unchanged.

The deterministic forcing-owned `t2m_daily` field differed by
`5.684341886080802e-14` at Day 4 between GPU compilation layouts. The report
retains `exact=false`; acceptance uses the predeclared float64
`atol=1e-12`, `rtol=0` gate. Discrete state remains exact.

## Decision

The retained capture path:

- returns only the training boundary and day-end state from the capture-only
  executable variant;
- transfers each stacked PyTree to the host once per block;
- slices days from NumPy arrays after transfer;
- keeps production modelout behavior unchanged when capture is disabled.

Increasing the block from 7 to 28 days gives a further 18 percent hot
throughput improvement, but the gain is already diminishing. A single
landpoint does not expose enough parallel work for a V100 to outperform the
established CPU Teacher path by a useful margin. Production Teacher sample
generation should therefore use long-lived CPU shard workers unless a future
implementation batches several independent landpoints into one GPU
executable. GPU resources remain appropriate for batched neural-network
training.

Machine-readable local artifacts:

- `outputs/performance/daily_coarse_graining/compiled_training_capture_in_process_gpu_15d_optimized.json`
- `outputs/performance/daily_coarse_graining/compiled_training_capture_in_process_gpu_29d_block28.json`
- `outputs/performance/daily_coarse_graining/compiled_training_capture_parity_3d_optimized.json`
