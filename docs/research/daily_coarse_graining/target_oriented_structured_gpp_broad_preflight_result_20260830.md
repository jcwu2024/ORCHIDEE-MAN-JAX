# Structured-GPP Broad Schedule And Free-GPU Preflight Result

Date: 2026-08-30

Status: accepted; bounded 1,000-update broad training is the next gate

Executable commit: `853afa3188011f72649fc3b4047bd1869f86a92f`

Schedule report SHA256:
`7de9e840eb8773b4ab9eaeb136cf97c295a13079b6589eebfea970e83679e252`

Schedule SHA256:
`b4295558406fd627837b012897585992ccd4e75042f7e76171b2a286bc7d6b8b`

GPU preflight report SHA256:
`0962b18d30e5f1b718554897064f7ca8d859234d24b93068f24ed034e7a39b0d`

## Target-Blind Schedule

Free `test04` materialized the complete proposed 5,000-update schedule in
`82.73 s` without reading target values. It contains exactly 1,040,000 unique
days in 5,000 batches of 208. Every batch has 52 distinct landpoints and four
days from one point-year per landpoint.

The schedule covers all 535 train landpoints, all 44 years, and all 23,412
non-monitor point-years. Each point-year appears 11 or 12 times and each
landpoint contributes 1,940 to 1,944 days. It overlaps neither the 103,399
prior-exposure days nor any of the 128 reserved monitor point-years.

## Free-GPU Preflight

The free-GPU preflight completed on an idle V100 in `421.14 s`. Sixteen broad
208-day batches all had finite losses and finite gradients over all 12,879,363
specialist parameters. Every one of the 129 specialist leaves had a nonzero
gradient in every batch.

Raw gradient norms ranged from `6.0577` to `23.0190`, with median `12.8578` and
p95 `20.98786557796449`. This p95 becomes the stage-1 clip threshold.

Four independent fresh-AdamW proposals at learning rate `4e-6` reduced their
same-batch losses by `0.4042%`, `0.3593%`, `0.3575%`, and `0.5596%`. Every
proposal was finite and nonzero, all 93,504,164 retained parent parameters were
the same array objects, and all physical hard failures were zero. The 19
declared-dynamic mask differences are the existing reported nonfailure count.

The preflight chained zero updates, wrote no optimizer checkpoint, read no
monitor, validation, or sealed-test values, and used no v5 or regenerated
Teacher data.

## Throughput And Decision

After the first compile, each 208-day gradient took about `3.67-3.76 s` and
preparing 52 random parent shards took about `4.65-4.84 s`. A sequential
1,000-update stage therefore has a measured core time near 2.35 hours before
monitor, checkpoint, recovery, and startup overhead. A five-hour one-V100
envelope is conservative.

Freeze one first broad stage of 1,000 updates and 208,000 unique days. Start
from step-498 parameters with fresh AdamW moments, constant `4e-6` learning
rate, and the p95 clip above. Evaluate the fixed 2,048-day train-only monitor at
updates 0, 250, 500, and 1,000. Paid execution still requires separate resource
approval after the runner and a free-node admission pass.
