# Coherent Teacher Dataset Generation v1

Status: **active production contract; six-point pilot pending**.

Date: 2026-08-05.

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

The canonical release is
`manifests/coarse_graining/canonical_teacher_data_release_v1.json`:

- release ID: `pft14-daily-teacher-coherent-v1`;
- release SHA256: `843d559b1dd1b5dbc494b22f478ded1b1cfc474716753f43021851c0428b6f7d`;
- Teacher Python-tree SHA256:
  `ffae1ec0db2b167fa02dc514b62ae0c8d35932e67e8bc2866565e7b0ae457223`;
- Markov contract SHA256:
  `6813065b51e95a2ec6663d8bc92f5336148704aea502632a05fd51d7e2e9d442`;
- coherent typed-supplement contract SHA256:
  `fd67deb3fb88a7293ad211945bc742b78318dd692039b38497eb2d587d5afe73`.

`daily_typed_supplement_v2.json` reuses only the frozen 29-label/66-field
schema from historical v1. It replaces the old immutable-parent packaging
semantics with `single_pass_continuous_teacher` and dataset manifest v5.

Verify the release before planning or running data production:

```bash
python -m research.daily_coarse_graining.teacher_data_release verify \
  --release manifests/coarse_graining/canonical_teacher_data_release_v1.json
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
compression. They are exposed by one `daily_teacher_dataset_manifest_v5` and
one release identity. Missing typed output means the point-year is incomplete.
Workers own complete landpoint chains, so year `Y+1` must name the exact
checkpoint SHA256 from year `Y`. Aggregation reopens all files, verifies hashes
and day inventories, and rejects contract or checkpoint-chain drift.

The ordinary user Teacher keeps capture disabled. Enabling capture changes
only compiled outputs: a three-day 1961 cold-start A/B found elementwise equal
capture-off and capture-on base arrays and final state, exact discrete state,
and all 66 typed fields within the fixed compiled-production gate
`atol=1e-8, rtol=1e-10`. The accepted run completed with `passed=true`; the
100-test data/contract/operator regression and Ruff/compile checks also pass.

## Six-Point Pilot

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

Run `teacher_shards validate` on that plan before `sbatch`. The proposed paid
shape is `cnall`, two nodes, six tasks, three one-CPU workers per node, and a
two-hour limit. Worst-case charge is `6 * 2 * CNY 0.07 = CNY 0.84`. Submission
still requires explicit approval under the Explore1000 policy.

Pilot acceptance requires six complete point-years, one release and Markov
contract, exact discrete/mask/serialization checks, valid restart chains, no
source drift, and measured wall time, peak RSS, base bytes, and typed bytes.
Only those measurements may set the full 669-point resource request.

## Next Gate

After the pilot passes, decide whether to authorize the complete coherent
Teacher dataset. Neural architecture selection begins only after enough
coherent training data exists. The old immutable-parent producer and its job
history remain reproducibility assets, not fallback production paths.
