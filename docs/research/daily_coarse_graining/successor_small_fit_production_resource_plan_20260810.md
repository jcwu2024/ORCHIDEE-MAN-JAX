# Successor Small-Fit Production Resource Plan

Date: 2026-08-10.

Status: implementation and bounded partition smoke accepted; CPU and GPU
production submissions remain separately gated by explicit cost approval.

## Bound Implementation

- Partitioned preparation implementation: commit
  `50bc99ad9093223ab44871d6937051f054531668`.
- Runner contract SHA256:
  `8317fc74b6ba921b0ed60e774743884ddc81b233c75ce1823f3246211d1640de`.
- Frozen coefficient contract SHA256:
  `b32f6cd0102d515b6f556e5e56b170229b27d913af8c0c2172b1e44c1f5f07ad`.
- Frozen 500-update schedule ceiling: exactly 500 train/train updates per arm.
- CPU launcher: `scripts/hpc/slurm_prepare_successor_small_fit_500.sh`.
- GPU launcher: `scripts/hpc/slurm_successor_small_fit_500.sh`.

No job in this plan has been submitted. The launchers contain finite resource
requests and fail closed on worktree, Git HEAD, runtime path and existing final
output drift.

## CPU Preparation Design

Four single-core workers build deterministic disjoint update sets by round
robin assignment. Each worker writes only its own atomic partial manifest and
uniquely named update artifacts. An interrupted worker reopens its
identity-bound partial and verifies completed artifact hashes before continuing.

After all workers pass, a separate finalizer reconstructs the frozen schedule
and requires:

- identical source, dataset, runner, candidate, owner and schedule identity;
- the declared worker index/count on every partial;
- exact ordered coverage of each worker's assigned updates;
- exact global coverage `1..500` with no duplicates;
- every artifact hash, schema, trace identity and schedule entry to pass;
- no Teacher regeneration and no historical v5 use.

Only then is `prepared_manifest.json` written with `status=passed`. The GPU
runner already refuses incomplete manifests and has no live batch-preparation
fallback.

### Bounded Smoke

The free `test01` smoke used two workers over the two-update schedule at commit
`50bc99a`. Both workers passed concurrently and produced byte-identical
artifacts to the accepted serial baseline:

```text
update-0001.pkl  d0b0fdaee6275546f5d740c670846a6d8e0849b0fa0c1d25ea36afe2987349ee
update-0002.pkl  2e86a482be394773d91af90cd8114f319ba405cf487f248270dc3bf177ceab71
```

Each worker used about 4.8 GB peak RSS. Its one-time runtime load plus one
artifact took about 203 seconds; the artifact preparation itself took about
70 seconds. The finalizer verified and merged both artifacts in 0.31 seconds
after runtime loading. The accepted smoke manifest is:

```text
runtime/outputs/training/successor-small-fit-partition-smoke-2u-50bc99a/prepared_manifest.json
SHA256 f73c714a7f9e24699f7775b5c23f221a7eda4e1b1fdb81621f927bede26b0f31
```

No Teacher data were regenerated, no v5 data were read and no GPU was used.

## Resource Envelope

The two accepted artifacts total 66,868,764 bytes and take about 70 seconds
each to prepare. Linear projection for 500 updates is approximately 16.72 GB
decimal storage. Four workers receive 125 artifacts each; including the
measured one-time load gives an expected worker wall time near 2.5 hours. The
four-hour CPU limit provides margin for shared-filesystem variability and the
full finalizer audit.

The cold GPU probe took 1,442.46 seconds for two updates. Its second prepared
entry cost 0.227 seconds to load, 1.352 seconds to replay, 1.208 seconds for
attention and 0.894 seconds for control. Extending that measured hot marginal
through update 500 projects about 55 minutes total. The two-hour GPU limit is
intended to cover CPU-allocation-dependent XLA compile variation. Sixteen CPU
cores are requested because both approximately 581-second cold compiles are
host-CPU work that holds the GPU allocation open.

Assuming CPU and GPU charges are additive:

| Stage | Partition and resources | Limit | Worst-case charge |
| --- | --- | ---: | ---: |
| Prepare | `cnmix`, 1 node, 4 tasks x 1 CPU | 4 h | `4 x 4 x 0.07 = CNY 1.12` |
| Fit | `gnall`, 1 node, 1 task, 16 CPUs, 1 GPU | 2 h | CPU `CNY 2.24` + GPU `CNY 4.40` = `CNY 6.64` |
| Both ceilings | staged, not concurrent | - | `CNY 7.76` |

The center's accounting should be confirmed if GPU jobs do not bill CPU and
GPU components additively.

## Approval Boundary

Approval remains staged rather than combined:

1. First approve only the `cnmix` preparation job. Its exact clean execution
   HEAD, output directory, command and CNY 1.12 ceiling must be shown.
2. After completion, independently admit the 500-entry manifest, artifact
   coverage, hashes, disk use, worker timings and peak RSS.
3. Then show a fresh `gnall` approval block binding that exact manifest and
   request separate approval for the CNY 6.64 GPU-stage ceiling.

Do not submit either stage from a test node. Do not start the GPU stage from a
partial manifest. Passing the 500-update fit opens, but does not itself pass,
the separately authorized 7/30-day screen. The 365-day rollout, long-held
points, Gate D2 and product acceptance remain closed.
