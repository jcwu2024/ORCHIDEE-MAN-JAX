# Teacher v6 Full Production Acceptance

Status: **accepted for neural training**.

## Frozen Identity

- dataset ID: `pft14-daily-teacher-669-1961-2010`
- Teacher commit: `19087b171564f44b49eb96fa1217169deca5d57a`
- release ID: `pft14-daily-teacher-v6`
- release SHA256: `1affefd9b3166005cdd868477ba25e68b682c7394438e5a79c4fcf087c8496a8`
- generation-plan SHA256: `90fa26605b6baf02b67da14096c6de4da4c8363f0938a3cc6623f8545f6639dd`
- Markov-contract SHA256: `6813065b51e95a2ec6663d8bc92f5336148704aea502632a05fd51d7e2e9d442`
- dataset-manifest SHA256: `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`

The server dataset root is
`runtime/outputs/training/pft14-daily-teacher-v6`. It occupies
167,563,257,950 bytes, approximately 156.1 GiB. This directory is an external
runtime asset and is not stored in Git.

## Production Inventory

- 669 landpoints
- 50 noleap years, 1961-2010
- 33,450 coherent point-year shards
- 12,208,581 daily transitions
- 100 complete worker manifests
- zero missing entries, active locks, or recovery candidates

The aggregate contains 23,540 train/train shards, 1,605 train/validation,
1,605 train/test, 2,948 validation/train, 201 validation/validation, 201
validation/test, 2,948 test/train, 201 test/validation, and 201 test/test.

The main production allocation was job `14493094`, with worker 98 recovered
by job `14493718` after a node-local compile-time allocation failure. The
combined worker inventory is complete. Aggregation reopened and hash-verified
all base shards, typed shards, metadata, checkpoints, input identities, and
year-to-year checkpoint links before writing the unified manifest.

## Formal Acceptance

The accepted finalizer was Explore1000 job `14500801` at finalizer commit
`b96702b`. It completed in `02:17:57` with exit code zero. Its output root is
`runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2`.

Formal production admission passed against
`daily_teacher_669_data_product_policy_v2.json`. Canonical acceptance then
reopened and hashed every base training shard, audited the target
representation, fitted train/train-only statistics, and ran a real JAX
forward/loss/gradient smoke.

Accepted evidence hashes:

- `production_admission.json`: `a7f70bae96e9eb020993905fbf4d1d7c02fb0108e2144e76a3571a6d89837c43`
- `acceptance_report.json`: `8784ed42dc64ef77c9642f613eb2e5940635edc6e744a1af4ea8634293baa8e3`
- `training_statistics.json`: `ab7f040a930071bd5604c17355a6493b23f3c76cba1e3084dcd5d5653101a4bf`
- `training_statistics.npz`: `e88d7cec504c6051aab6c9feacb619dca94961730ffa18f840563a48fc815a3a`
- `worker_progress.json`: `559574d76afb36e9e5c1d28afcb9df14eba35689aa02537d87d2eaa99061d5a2`

## Scientific And Training Gates

The target representation audit passed all 33,450 shards and 12,208,581
transitions. It found zero persistence mismatches across 2,754
persistence-mapped columns; 101 columns use mean-centered representation. The
three declared undefined-value families remain source-defined and masked.

Statistics use 8,591,565 train/train samples only. Validation and test data do
not contribute normalization values.

The real batch smoke used JAX 0.4.38 on CPU with batch size 8. The canonical
test model had 1,963,369 parameters. Predictions, loss, and all 16 gradient
leaves were finite; loss was `0.019547218456864357` and gradient norm was
`0.1043393010253475`.

## Decision

Gate E2 data production is complete. Do not regenerate the dataset or join it
to historical v5 rows. All subsequent neural experiments must bind the
dataset manifest, acceptance report, and statistics hashes above. Test splits
remain sealed until the architecture and training protocol are frozen.
