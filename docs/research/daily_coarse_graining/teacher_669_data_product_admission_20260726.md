# 669-Point Teacher Data-Product Admission

Date: 2026-07-26

## Decision

The contract-level production admission passes. Generate the complete frozen
669-landpoint, 1961-2010 baseline after staging and plan verification rather
than creating another architecture-specific spatial pilot.

This decision does not promote the current neural checkpoint and does not
authorize a paid job by itself. It freezes what the Teacher data product must
contain so that future model architectures, objectives, and rollout horizons
can reuse the same trajectories.

## Machine Gate

The versioned policy is
[`../../../manifests/coarse_graining/daily_teacher_669_data_product_policy.json`](../../../manifests/coarse_graining/daily_teacher_669_data_product_policy.json).
Run the pre-generation contract gate with an accepted v5 aggregate:

```bash
python -m research.daily_coarse_graining.teacher_data_product_admission \
  --policy manifests/coarse_graining/daily_teacher_669_data_product_policy.json \
  --contract-manifest /absolute/path/to/v5/dataset_manifest.json \
  --output /absolute/path/to/pre_generation_admission.json
```

The accepted local report has phase `pre_generation_contract`. It binds the
frozen production spec SHA256 `6bb0198f...a6a27e` to Daily Markov Contract v5
SHA256 `6813065b...9d442` and verifies:

- 3,854 continuous and 12 exact discrete cross-day state values;
- 2,855 fast-day target values and all eight target families;
- 90 named daily diagnostics;
- five native six-hour forcing records per day;
- 84 parameter, 735 landpoint-static, and one annual-condition values;
- PFT1/PFT14 compaction, source hashes, and required scientific carbon,
  vegetation, water, thermal, and surface-state inventories;
- 535/67/67 train/validation/test landpoints and the frozen temporal split;
- the explicit Teacher Day 1 cold-start boundary.

After generation and aggregation, repeat the command with the production
manifest and add `--require-production-dataset`. This final phase verifies all
669 landpoints, all 50 years, all 33,450 unique shards, contract hashes, and
every spatial and temporal split. Contract evidence from a smaller dataset
cannot pass that final mode.

## Scale And Use

The frozen data product contains:

- 33,450 point-years;
- 12,208,581 daily transitions;
- 8,591,565 train/train transitions;
- approximately 104-162 GiB compressed output;
- an estimated 67-91 CNY of CPU time at the measured bounds;
- about 10.8 days conservative wall time at the currently accepted five
  concurrent workers.

Training must stream balanced windows from immutable shards. The same assets
support one-step supervision, multiday curricula, free rollout, alternative
architectures and objectives, and spatial/temporal/joint/complete-chain
validation. Validation and sealed-test samples never contribute statistics or
training updates.

## Explicit Boundaries

The baseline stores one archived calibrated parameter tuple per landpoint.
That variation is useful spatial evidence but does not independently identify
the response to each calibrated parameter. A later parameter-response dataset
must be a separate parent-hash-bound extension using training landpoints only;
it must not rewrite or duplicate the baseline shards.

Likewise, a new forcing product is a separate extension distribution. The
baseline does not claim unseen-forcing generalization. Full half-hour internal
state is intentionally excluded because it is unnecessary for the accepted
daily operator and would multiply storage substantially.

The neural operator starts on Day 2. The real Teacher Day 1 bootstrap creates
canonical `S[1]`, stored as `state_trajectory[0]` in the 1961 shard. A fully
neural cold-start initializer remains outside the present surrogate scope.
