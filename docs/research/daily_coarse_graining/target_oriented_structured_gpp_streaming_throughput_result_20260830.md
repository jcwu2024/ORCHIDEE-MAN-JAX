# Structured-GPP Direct-Only Streaming Throughput Result

Date: 2026-08-30

Status: accepted functional and conservative-throughput evidence; broad direct
GPP training and monitor contract is the next gate

Executable commits:

- `fd6a739c35773d4db219ded74e5aa1ff3ed7dc58`: complete-example admission
- `c28a667398fc3c8614645928d1791156a5a8dfe8`: exact direct-GPP-only streamer

Accepted direct-only report SHA256:
`06731a560786a02dd2de5e323a3d3b395eaacc89e51f69ceb89a3a6ed0210761`

## Execution Boundary

The direct-only probe completed on free `gln01` in `191.00 s`. It loaded four
non-monitor train/train parent shards, selected 52 days from each, round-robin
stacked 208 days, and executed exact structured-GPP forward and specialist
gradients with a four-day microbatch. It performed zero optimizer updates and
zero checkpoint writes. It read no typed-sidecar, retained-tail, 7/30-day,
monitor, validation, or sealed-test payload and regenerated no Teacher data.

The two GPUs were already carrying unrelated processes at about 27% reported
utilization each. Therefore the hot GPU result below is a conservative shared-
card measurement, not an uncontended peak-throughput claim.

## Direct-Only Supply Result

The first admission used the complete training-example assembler even though
the device batch retained only GPP. Preparing 208 days took `84.2489 s`. The
accepted direct-only path instead reconstructs only inference-owned inputs and
the exact packed `daily_interface.gpp_daily` leaf:

| Host stage | Seconds for 208 days |
| --- | ---: |
| Four parent-shard reads | 0.1968 |
| Exact direct-GPP assembly | 0.5262 |
| Round-robin and stack | 0.0074 |
| Total preparation | 0.7304 |
| Host-to-device transfer | 0.0033 |

Host preparation is `115.34x` faster than the complete-example admission and
sustains about 284.76 days/s. The result proves that broad direct-GPP training
does not need typed-sidecar reads or construction of the other process targets,
next state, or retained boundary.

## Compute Result

All tested batch sizes had finite loss statistics and finite gradients over all
12,879,363 specialist parameters.

| Effective batch days | Forward, s | Gradient, s | Total, s | Days/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.2152 | 0.4176 | 0.6329 | 50.56 |
| 64 | 0.3806 | 0.7923 | 1.1730 | 54.56 |
| 128 | 0.7495 | 1.5399 | 2.2894 | 55.91 |
| 208 | 1.2157 | 2.4747 | 3.6905 | 56.36 |

The 208-day end-to-end hot path, including one measured host preparation,
sustains 47.05 days/s under contention, or a conservative 5.90 hours per
million days. The accepted earlier joint-training record reached about 103
days/s for the same two-pass compute structure without this observed
contention. New host supply is faster than either GPU rate, so supply is no
longer the primary throughput bottleneck.

JAX reported `1,139,112,192` peak bytes in use and a `25,559,728,128`-byte
allocator limit. The approximately 27 GiB shown by `nvidia-smi` during the run
was predominantly allocator reservation rather than live model/batch storage.

## Decision And Next Gate

Accept the direct-only streamer as the data boundary for broad GPP training.
Use the shared-card end-to-end result as the conservative resource-planning
bound; an idle-card rerun may refine performance but is not required to design
the next contract.

The next gate is to freeze a single broad-data training and train-only monitor
contract. It must reserve the target-blind 128-point-year monitor from the
coverage report, expose roughly one million previously unseen train/train days
without repeating the old 500-artifact schedule, retain the coordinate-aligned
Attention plus structured-GPP mechanism, and define bounded stopping and paid-
job admission before any optimizer update.
