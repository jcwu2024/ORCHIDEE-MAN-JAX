# Expand effective training coverage in parallel with continuation v3

The user approved coverage expansion preparation while job 14991510 runs.
This does not authorize a second paid GPU training run or all-parameter
unfreezing. The current training checkout stays at `ce7cd79`; independent
preparation reads its existing code and writes separate runtime outputs.

## Why this work is needed now

Exploration justifies bounded budgets, not indefinite reuse of a small sample.
Wider data coverage can be introduced before a final training recipe is frozen.
The currently frozen 93,504,164 parameters contain inherited learned weights,
not random unused capacity. Updating only the 15,960,211 GPP/feedback parameters
was a targeted optimization choice, not proof that this is the best final scope.
Broader/fuller unfreezing remains a separate question: keep it fixed for the
coverage experiment, then assess whether it limits learning. Final training may
be staged, partially frozen or jointly fine-tuned, but must reproduce all
consequential inherited pretraining and stage rules end to end.

## Metadata evidence

Free test02 computed exact unique day keys from the existing schedule and
v6 train/train index. No Teacher target arrays, validation targets, sealed
test targets, or model checkpoints were read. No gradients or model updates.
Script: `scripts/hpc/audit_conditioned_training_coverage.py`.
Local result: `outputs/conditioned-coverage-extension-20260910.json`.

The available train/train split contains 535 landpoints, 44 years (1961-2004),
23,540 point-years and 8,568,025 eligible days after first-row exclusion.

| Package set | Unique point-years | Unique days | Fraction of eligible days |
| --- | ---: | ---: | ---: |
| Current 678-package pool | 13,265 | 140,002 | 1.634% |
| Original 1,000 prepared packages | 16,590 | 205,520 | 2.399% |
| Proposed 2,000 new packages alone | 21,496 | 406,022 | 4.739% |
| Current pool plus proposed extension | 22,653 | 539,171 | 6.293% |

All four sets include all 535 training locations and all 44 years. The
important gaps are combinations, dates and within-year trajectories; merely
counting locations/years would conceal these gaps. The proposed extension adds
399,169 unique days versus the current pool (395,854 versus the original 1,000
packages), increasing current-pool unique day coverage by 3.851 times.

These are sample-availability counts for these named package sets, not a
lifetime exposure census of every inherited pretraining stage. For example,
the historical broad specialist training also has its own sampling path.
Nor does a prepared window prove it was used for recurrent gradients: the
current method evaluates direct GPP on all 40 packages per update but uses the
scheduled first eight packages for state/7/30-day objectives.

## Concrete expansion

Reuse entries 1001-3000 of the existing target-oriented seeded schedule,
already supported by its 3,000-entry maximum. No new data selection algorithm,
normalization, loss, model, or Teacher production is needed. The original
1-1000 entries remain a prefix. Each new package contains 28 distinct
locations from a common training year: 32 one-day samples, eight 7-day windows
and four 30-day windows, totaling 208 direct-GPP day positions. Overlapping
days are deduplicated in the table above.

The extension is a bounded coverage increase, not full-data training. It uses
the same year-balanced random sampling as existing packages, so it does not
claim deliberate oversampling of drought, dormant vegetation, or other rare
states. Target-aware balancing would require a separate train-only design.

Free test03 completed entries 1001-1002 as a two-package pilot using the
unchanged `prepare_target_oriented_training_batches` implementation, with
existing retained-input reuse from the first 250-package manifest. It only
extracts existing arrays and static/forcing context; it does not run Teacher
production or train a network. Pilot output:
`runtime/outputs/training/conditioned-coverage-pilot-ce7cd79/chunks/1001-1002/`.
Log: `runtime/logs/conditioned-coverage-pilot-ce7cd79.log`.

The pilot finished with exit 0 in 14m42.06s including startup and retained-input
preload. Package times were 199.20 and 198.18 seconds, almost entirely batch
preparation; retained-input preparation was zero because all 535 landpoints
could reuse the accepted inputs. Total compressed output is 6,757,426 bytes.
Maximum RSS was about 5.6 GB; observed CPU availability was 47%, so this free
node is not an allocated-core performance benchmark. A separate free test02
process read both artifacts back, confirmed 32 one-day samples / eight seven-day
windows / four thirty-day windows each, finite defined GPP targets, and a common
executable structure. Years are 1964 and 1990. This is data admission; a new-package
neural forward/state/budget test remains part of the later GPU-training preflight.

## Proposed CPU materialization

Prepare 1001-3000 as ten 200-package chunks. Script:
`scripts/hpc/slurm_conditioned_coverage_preparation.sh`.
Proposed cnall array: ten jobs, at most ten concurrent; each one node/task,
one CPU, no GPU, maximum 12h, 710m inner timeout, no requeue. Omit Slurm memory
requests on this CPU partition as required by the current deployment rules.
Maximum total CPU cost: 10 * 1 * 12 * CNY 0.07 = **CNY 8.40**.
This budget is not yet approved; no sbatch has been issued for it.
At the measured free-node wall time, 200 packages take about 11.1h plus startup;
dedicated CPU time may be faster, but no speedup is assumed for the ceiling.
Parallel queue admission is not guaranteed, and the 12h bound is per task.
Each artifact is saved as it completes. No automatic retry is authorized.
Estimated new compressed storage is about 6.8 GB based on the pilot.

The standalone launcher has been placed in the project runtime transfer directory
and invoke the unchanged `ce7cd79` preparer, without changing the running GPU
job's checkout. Working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.
Combine chunk manifests after completion using the existing manifest schema;
retain old package/monitor provenance and measure the materialized coverage.
Ruff/whitespace checks pass locally and the transferred launcher passes bash
syntax checking on free test02. Ready CPU submission from cln01, in that
working directory, after approval:

```bash
EXPECTED_GIT_HEAD=ce7cd7985e28fb5b53ae2d578d11f6b1d44761db /rmprog/slurm/v22.05.7/bin/sbatch runtime/transfer/slurm_conditioned_coverage_preparation_20260910.sh
```

This reads the admitted `ce7cd79` source checkout without updating it. Its new
array output root is `runtime/outputs/training/conditioned-coverage-expanded-ARRAY_JOB_ID/`.
Current training job 14991510 and its files remain separate.

Before later GPU training, complete a small candidate forward/state/budget
check on new packages and freeze its actual selection schedule. To isolate
coverage, compare from the same accepted update-34 weights with 40 updates,
40 packages/update, unchanged loss and trainable parameter scope. Reuse v3
as the fixed-pool comparator; no assumption that it is already a good model.
The reused validation panel remains development validation. Do not select
new train rows from its targets or claim a fresh/sealed test. A separate
GPU resource proposal will be needed; preparing data is not permission to
start that training or unfreeze parameters.
