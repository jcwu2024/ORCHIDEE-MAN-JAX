# Successor 7/30-day Rollout Screen Package

Status: **machine contract and local implementation accepted; numerical screen
awaits the successful bounded-fit checkpoints**.

Date: 2026-08-10.

## Decision

The post-fit model-selection screen is frozen before job `14517771` produces
any training result. Its canonical machine-contract SHA256 is
`99ec6eb08fc33ca94364de604b262fa29139e2dcfd7c9176d1a5d772a0b6fc07`.
It does not change the training loss, validation gates, model capacity, or
window panel after seeing a checkpoint.

Both arms enter through the same accepted boundary:

```python
bind_daily_operator(
    input_assembler=canonical_assembler,
    process_head=checkpoint_head,
    retained_tail=canonical_retained_tail_adapter,
)
```

The evaluator loads parameters only after the successful fit report and the
atomic checkpoint-set manifest both pass exact SHA256 and identity admission.
It requires both 500-update arms, their accepted promotion result, exact
parameter counts, and zero v5, Teacher-regeneration, or sealed-test access.
Optimizer trees are verified by the checkpoint reader but are not loaded into
the evaluation parameter registry.

## Frozen Panel

The only admitted model-selection slices are:

| Slice | Spatial split | Temporal split |
| --- | --- | --- |
| temporal | train | validation |
| spatial | validation | train |
| joint | validation | validation |

Shard and start selection use a fixed SHA256 rank with seed `20260810`.
Each slice uses distinct landpoints; the four formal windows within a shard
do not overlap. Windows never cross a shard or year, and row zero of every
annual shard is excluded. Landpoint identity controls sampling only and is
never a model feature.

- Free smoke: one shard and one 30-day window per slice, three windows total;
  it is not promotion eligible.
- Formal bounded screen: eight shards and four 30-day windows per slice, 96
  windows total.
- Each 30-day free rollout yields both Day-7 and Day-30 evidence. Candidate
  and control consume the same serialized window artifacts and the same plan
  SHA256.

The formal panel is prepared on free `test01`-`test04` CPU nodes. Preparation
reads existing coherent v6 shards and stores the canonical inputs, Teacher
endpoints, and deterministic retained-tail day inputs in hash-bound artifacts.
It does not regenerate Teacher data. This keeps context construction and host
assembly outside paid GPU time.

## Scientific Gates

At Day 7 and Day 30, the report contains normalized RMSE, normalized signed
bias, normalized mean absolute error, and evaluated-value count for:

- cumulative GPP, NPP, maintenance respiration, growth respiration, and
  heterotrophic respiration;
- terminal AGB, BGB, total live biomass, litter, particulate soil carbon,
  DOC, and ecosystem carbon.

The primary set is cumulative GPP/NPP plus terminal AGB/BGB/total live
biomass/ecosystem carbon. Every slice and horizon must pass all gates:

- the attention candidate's primary composite normalized RMSE is at least 5%
  lower than the graph-MLP control;
- no primary key regresses by more than 5%;
- no other respiration or litter/soil-carbon key regresses by more than 10%;
- every candidate key has absolute normalized signed bias at most 0.10;
- candidate and control each have zero nonfinite, negative-stock,
  defined-mask, discrete-state, and water/carbon-budget failures.

One failed gate rejects promotion. A successful bounded screen only opens a
separate authorization decision for 365-day evaluation; it does not itself
authorize long-held points, Gate D2, or product acceptance.

## Implementation

- Machine contract:
  `manifests/coarse_graining/successor_7_30_rollout_screen_v1.json`
- Inventory, window selection, metrics, checkpoint admission and
  classification:
  `daily_operator/successor_rollout_screen.py`
- CPU artifact producer and reader:
  `scripts/hpc/prepare_successor_rollout_screen.py` and
  `scripts/hpc/successor_rollout_screen_prepared.py`
- GPU evaluator:
  `scripts/hpc/run_successor_rollout_screen.py`
- Explore1000 wrappers:
  `prepare_successor_rollout_screen.sh`,
  `run_successor_rollout_screen.sh`, and
  `slurm_successor_rollout_screen.sh`

The formal Slurm proposal is one `gnall` node, four allocated CPUs, one V100,
and a 45-minute limit. Its additive worst-case ceiling is CPU CNY 0.21 plus
GPU CNY 1.65, totaling CNY 1.86. The file is a proposal only; no screen job is
submitted by this package.

## Local Acceptance

Focused synthetic and failure-path tests cover contract drift, deterministic
split selection, annual-row exclusion, plan tampering, metric accumulation,
candidate/control pass and reject classifications, free-smoke
non-promotability, checkpoint-report admission, optimizer exclusion, artifact
round-trip, and artifact hash drift. The focused suite passes `12/12`, with
Ruff, Python compilation, shell syntax, and `git diff --check` also passing.

Actual checkpoint admission, the free real-v6 GPU smoke, and the formal
numeric classification remain pending. Job `14517771` failed before training
and published no checkpoint; they cannot run until a separately authorized
future fit completes successfully and atomically publishes both arms.
