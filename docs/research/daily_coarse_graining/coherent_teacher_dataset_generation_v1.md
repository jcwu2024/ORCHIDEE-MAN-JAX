# Coherent Teacher Dataset Generation v1

Status: **accepted v6 production contract; full dataset admitted**.

Contract frozen: 2026-08-05. Pilot accepted: 2026-08-06.

## Decision

The formal daily-surrogate training dataset is generated from one frozen
Teacher release. Each worker advances one continuous landpoint chain and emits
the base Markov shard, 66 typed process fields, explicit defined masks, and the
year-end checkpoint from the same state transitions.

The historical 669-point Markov v5 dataset remains accepted evidence for the
earlier experiments. It is not a parent of the new formal training dataset.
No current-Teacher labels may be joined to its old state rows or represented as
if they came from the same Teacher execution.

## Frozen Identity

The canonical production release is
`manifests/coarse_graining/canonical_teacher_data_release_v2.json`:

- release ID: `pft14-daily-teacher-v6`;
- release SHA256: `1affefd9b3166005cdd868477ba25e68b682c7394438e5a79c4fcf087c8496a8`;
- Teacher Python-tree SHA256:
  `ffae1ec0db2b167fa02dc514b62ae0c8d35932e67e8bc2866565e7b0ae457223`;
- Markov contract SHA256:
  `6813065b51e95a2ec6663d8bc92f5336148704aea502632a05fd51d7e2e9d442`;
- coherent typed-supplement contract SHA256:
  `395c35585ec4fcb04d15f3d189af3f7957646fc38f9f27245c5136ba9a123423`.

`daily_typed_supplement_v3.json` reuses only the frozen 29-label/66-field
schema from historical v1. It replaces the old immutable-parent packaging
semantics with `single_pass_continuous_teacher` and binds the formal dataset
manifest v6. The Markov state contract remains v5. Typed v2 and release v1 are
preserved as the exact identity of the accepted six-point pilot, not as the
formal production product.

Verify the release before planning or running data production:

```bash
python -m research.daily_coarse_graining.teacher_data_release verify \
  --release manifests/coarse_graining/canonical_teacher_data_release_v2.json
```

Any Teacher, config, PFT catalog, contract, or producer-source hash drift is a
new release. Do not edit the frozen manifest to bless drift.

## Transaction And Recovery

A coherent worker completion is one transaction with four required assets:

1. base float64 Markov shard;
2. lossless typed float64 shard plus exact bool masks;
3. complete year-end checkpoint;
4. shared metadata binding all hashes and the preceding checkpoint.

Physical base and typed NPZ files remain separate for streaming and
compression. They are exposed by one `daily_teacher_dataset_manifest_v6` and
one release identity. Missing typed output means the point-year is incomplete.
Workers own complete landpoint chains, so year `Y+1` must name the exact
checkpoint SHA256 from year `Y`. Aggregation reopens all files, verifies hashes
and day inventories, and rejects contract or checkpoint-chain drift.

The ordinary user Teacher keeps capture disabled. Enabling capture changes
only compiled outputs: a three-day 1961 cold-start A/B found elementwise equal
capture-off and capture-on base arrays and final state, exact discrete state,
and all 66 typed fields within the fixed compiled-production gate
`atol=1e-8, rtol=1e-10`. The accepted run completed with `passed=true`; the
102-test v5/v6 data/contract/operator regression and Ruff/compile checks also
pass.

## Six-Point Pilot

The accepted pilot used predecessor release `pft14-daily-teacher-coherent-v1`
and manifest v5 packaging. It validated the same single-pass transaction,
Markov v5 arrays, 66 typed fields, masks, and checkpoint artifacts now bound by
v6. The following command is retained as historical provenance, not the next
production command.
Reproducing that exact command requires commit `2eb6384`; current production
must use release v2 below.

The pilot production spec is
`manifests/coarse_graining/daily_teacher_coherent_pilot_v1.json`. It contains
six 1961 landpoints and uses the already staged raw forcing/restart/reference
assets. It does not read old Teacher output shards.

Build the machine-specific plan on Explore1000 because asset and output paths
are external to Git:

```bash
REPO=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUNTIME=$REPO/runtime
$REPO/.venvs/orcjax_cpu/bin/python \
  -m research.daily_coarse_graining.teacher_production plan \
  --spec manifests/coarse_graining/daily_teacher_coherent_pilot_v1.json \
  --asset-root "$RUNTIME/assets/teacher_669_a403cd3" \
  --teacher-config configs/orchidee_man_250919.yaml \
  --output-root "$RUNTIME/outputs/training/pft14-daily-teacher-coherent-v1-pilot" \
  --plan-path "$RUNTIME/plans/pft14-daily-teacher-coherent-v1-pilot.json" \
  --data-release manifests/coarse_graining/canonical_teacher_data_release_v1.json \
  --production-scope pilot
```

Run `teacher_shards validate` on that plan before `sbatch`. The accepted paid
shape was `cnall`, two nodes, six tasks, three one-CPU workers per node, and a
two-hour limit. Job `14486923` completed in `01:41:21`; all six workers exited
zero. Its worst-case charge was CNY 0.84 and approximate allocation cost was
CNY 0.71.

Pilot acceptance required six complete point-years, one release and Markov
contract, exact discrete/mask/serialization checks, valid checkpoint metadata,
no source drift, and measured wall time, peak RSS, base bytes, and typed bytes.
All gates passed. The formal evidence and measurements are in
[`coherent_teacher_pilot_20260806.md`](coherent_teacher_pilot_20260806.md).

## Production Sizing Gate

Explore1000 job `14490208` tested the formal v6 producer on ten representative
landpoints, two years per point, with five one-CPU workers on one node. All
`20/20` point-years passed complete hash verification and aggregation. The
first cold entry averaged 2148.84 seconds total, while steady hot capture was
about 82 seconds per point-year. Changing landpoint in the same process did
not create another compiled executable. Five workers peaked at about 31.2 GiB
RSS each and completed without swap or OOM.

The accepted full-production topology is 100 long-lived workers on 20 nodes,
five workers per node, with a 180-second local startup stagger. Expected wall
time is 8.6-10 hours and expected CPU charge is about CNY 60. Use a 12-hour
limit for a CNY 84 worst-case cap. Full evidence is in
[`teacher_v6_reuse_benchmark_20260806.md`](teacher_v6_reuse_benchmark_20260806.md).

## Full Production Acceptance

The accepted formal plan was built from the frozen 669-point production spec
with:

```bash
REPO=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUNTIME=$REPO/runtime
$REPO/.venvs/orcjax_cpu/bin/python \
  -m research.daily_coarse_graining.teacher_production plan \
  --spec manifests/coarse_graining/daily_teacher_production_669.json \
  --asset-root "$RUNTIME/assets/teacher_669_a403cd3" \
  --teacher-config configs/orchidee_man_250919.yaml \
  --output-root "$RUNTIME/outputs/training/pft14-daily-teacher-v6" \
  --plan-path "$RUNTIME/plans/pft14-daily-teacher-v6.json" \
  --data-release manifests/coarse_graining/canonical_teacher_data_release_v2.json \
  --production-scope full
```

Full production, aggregation, and admission completed on 2026-08-07. The
accepted aggregate contains 669 landpoints, 33,450 point-years, and 12,208,581
daily transitions. Its manifest SHA256 is `5fc2c069...dc13aa99`; all coherent
assets and checkpoint chains passed, and the canonical training acceptance
report SHA256 is `8784ed42...aa8e3`. See
[`teacher_v6_full_production_acceptance_20260807.md`](teacher_v6_full_production_acceptance_20260807.md).

Gate E2 is closed. Neural candidate training must use the accepted manifest,
statistics, and acceptance report. The old immutable-parent producer and its
job history remain reproducibility assets, not fallback production paths.
