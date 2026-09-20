# Direct joint dual-GPU learning: one prepared-pool pass

Status2026-09-20: job15279139 ended FAILED124 at950min,4643/5415 updates;
recovery4600, science4000. Not scientifically accepted; see
`direct_joint_dual_pool_results_20260920.md`. The frozen protocol and submission
record below remain unchanged. This was the next
substantial learning run after the short joint/staged pilots and bounded
continuous dual/recovery parity. It supersedes the paused single-GPU epoch
resource proposal; no new architecture, loss, AD or batching study is required.

## Question and method

Determine whether demonstrated early learning extends to substantially more
TRAIN coverage and updates, and reduces production/growth/carbon drift rather
than merely fitting large initial inventories. Direct joint is the main route
because joint600 gave lower developmentT30 GPP/NPP/biomass RMSE. Staged remains
a viable alternative with better canopy/nonliving performance; one pilot did
not prove a final ranking. This run does not train a second comparison arm.

Fresh seed20260911, all109,464,375 parameters jointly optimized, fresh AdamW;
no exploratory, pilot or three-update integration checkpoint inheritance.
Use the existing Attention architecture, canonical daily binding and retained
physical tail, fixed losses/scales, native checkpointed gradients and explicit
FP32 Attention/FP64 physics policy. No subdaily reconstruction in inference.
Learning rate1e-5 constant for this entire declared pass; beta1=.9,beta2=.999,
epsilon1e-8, weight decay0, global clipping1. No ad hoc midway loss/LR changes.

GPU0 computes daily64 as8microbatches of8 and owns AdamW; GPU1 computes oneT7
then oneT30 trajectory. Combine D+.5R7+.5R30 and update once, synchronizing
latest weights after each update. Using two GPUs changes execution, not batch
statistics. Do not increase microbatch or number of windows in this run.

Executable declarations:

- `mixed_learning_epoch_model_v1.json`, `mixed_learning_epoch_v1.json`;
- `mixed_learning_execution_fp32_checked_v1.json`;
- `mixed_learning_joint_pool_v1.json` (one mixed phase, explicit progress counts);
- shared `scripts/hpc/run_mixed_learning.py --dual-device-training` through
  `scripts/hpc/job_mixed_learning_dual_pool.sh`.

## Actual coverage

5415 updates consume346,539 distinct ready TRAIN daily rows across535 sites,
44 years and17,677 point-years, with21 fixed-shape terminal padding occurrences:
346,560 daily draws. This is100% of the ready prepared daily pool, about4.04%
of eligible Teacher TRAIN days, not an epoch of the whole Teacher release.
Daily seed20260912; existing schedule seed20260914. Sampling order is fixed
and replayable, with a different batch at each update.

Also consume5415 uniqueT7 and5415 uniqueT30 windows from the already prepared
schedule; recurrent union198,101 Teacher days across535 sites/44years/8730
point-years. Daily and recurrent days overlap; their counts must not be added
as unique data. Preserve current TRAIN/development exclusions and within-update
T7/T30 day separation. All assets already exist. No Teacher generation,
historicalv5, new per-window files or full-dataset cache expansion.

Compared with joint600, updates/daily draws increase9.025x. This is an explicit
fresh executable method and one meaningful coverage increase, not a promise of
accuracy or the final training recipe. Multiple passes/full-Teacher expansion
may still be needed, guided by this run's learning and validation curves.

## Measurements and interpretation

Retain and evaluate0,200,600,1000,2500,4000,5415. Milestone600 is descriptive
comparison with the prior joint pilot; do not require bitwise agreement across
device layouts. The same process continues between milestones, without
resubmission/permission at each point. Full physical/state/budget checks remain
active every update; failures stop execution. Science misses are reported,
not treated as runtime errors or automatically answered by changing the model.

Each milestone uses existing fixed/excluded TRAIN daily monitors and the same
8T7+8T30 development panel. Report physical-unit per-window RMSE and signed
bias for GPP,NPP,RA,RH,AGB/BGB, canopy and nonliving carbon; include biomass
growth and carbon change from initial state, not just stock relative error.
Record normalized gates alongside actual quantities, gradient/clipping, observed
throughput and per-device live versus reserved memory. Current developmentRH
Teacher totals are zero: low error there does not certify nonzeroRH accuracy.

Keep all milestones and select5415 as this bounded pass's endpoint. No accuracy
early stop and no best-looking checkpoint selection. Interpret train versus
development curves jointly: remaining TRAIN error, transfer gaps and growth/
carbon drift remain distinct diagnoses. A reused small development panel is
not independent validation. After the pass, decide whether to continue declared
training, revisit loss/data coverage, or perform a broader independent7/30day
screen. No365day/50year/product acceptance follows automatically.

## Recovery and remaining evidence

Save a rolling full checkpoint every50updates (~6.3min at current warm rate),
plus7 milestones. Shared checkpoint includes params, AdamW, seed/method/data
identities, both cursors and explicit device layout. Use the same immutable
checkout for recovery. Do not automatically restart on a paid node.
Approximately8*1.23GiB permanent checkpoints plus1.23GiB atomic temporary;
reports/logs add modest bytes. File count is bounded, not per sample or step.

Job15269596's timeout is retained as a failure of full orchestration completion.
Its saved full-state and scalar parity passes. The supplemental read-only saved
forward comparison PASSED on free gln01 GPU0 in516.63s: recovered3 versus
completed dual3 on the same fixed first8 daily inputs, with physical checks
all zero. Loss difference1.697e-7 and largest per-sample term difference3.845e-6
pass unchanged rtol1e-5/atol1e-6. It cannot reconstruct a
lost original live-vs-disk scalar or retroactively mark that job completed.
The new long run retains its ordinary final persisted-forward replay; no costly
repeat of the four-process integration is required.

## Wall time and cost

Two measured warm changing-data updates averaged7.599s:5415updates would take
11.43h before startup/monitors/IO. That small sample is a planning estimate,
not a sustained-throughput guarantee. Budget roughly13–15h including startup,
seven monitors and recovery IO; request16h hard ceiling (950min inner timeout)
to avoid relying on the optimistic warm-only estimate. One process pays one
initial setup, unlike the previous four-process integration. Queue time is
additional. If slower operation exhausts the ceiling, preserve progress and
request any continuation budget explicitly.

gnall1node/2V10032/8allocatedCPUs/64GiB,16h, no requeue. Assuming additive rates:
CPU8*16*.07=8.96yuan;GPU2*16*2.20=70.40yuan; ceiling79.36yuan.
13–15h corresponds to64.48–74.40yuan; these are estimates, not bills.
The measured equal-update dual cost can be modestly higher while wall time
is shorter; we are choosing that tradeoff explicitly.

Working directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-dual-pool-v1`.
Command: `sbatch scripts/hpc/job_mixed_learning_dual_pool.sh`.
Output: `runtime/outputs/training/mixed-learning-dual-pool-v1/report.json`.
Paid submission requires user approval of this specific envelope after preparation.

### Approved submission

User explicitly approved submission after discussing the full-Teacher cost.
Submitted once from gln01 with the declared command; job15279139 started
2026-09-19 16:54:55 server time onibc13b02n02. Slurm confirms2GPUs/8CPUs/64GiB,
16h, no requeue and deadline2026-09-20 08:54:55. Launcher confirmsd1183e6 and
GPU0,1. Slurm log is
`runtime/outputs/training/mixed-learning-dual-pool-v1/slurm-15279139.log`.

The approval covers this run only, not a second paid learning arm or automatic
extension. Keep its code and declared method fixed. Larger-data training cost
remains unresolved: linear extrapolation at the same daily/recurrent ratio is
roughly13-16days per eligible Teacher daily pass, not a measured final-method
runtime. Existing timing/learning records may support a later lower-frequency
recurrent-training design while this substantive learning run proceeds.

## Completed preparation checks

Immutable execution checkout is `d1183e6` at the working directory above.
The actual target Bash launcher passed syntax and argument-expansion dry-run,
including a non-default two-GPU mask `3,5`, the declared schedule and950min
timeout. The Slurm output directory exists. An independent Sol/high source
review found no launch blocker: fresh initialization, phase/cursor arithmetic,
milestones, rolling checkpoints and two-device propagation use the shared paths.

The existing metadata loader passed on free CPU test02, without Teacher payload
reads or optimizer updates. Saved evidence:
`outputs/mixed-learning-dual-pool-v1-metadata.json`. Its base epoch configuration
lists six milestones; the explicit learning schedule adds600, so the effective
run has the seven milestones listed above. Local schedule validation and43
focused tests passed. The supplemental forward script also passes Ruff.
These checks establish preparation and data availability, not trained accuracy.
Supplemental GPU evidence is saved at
`outputs/mixed-learning-15269596-forward-supplement.json`; exit0, zero optimizer
updates and zero gradient evaluations. The job/log directory was confirmed
present after this check. These checks preceded the approved submission above.
