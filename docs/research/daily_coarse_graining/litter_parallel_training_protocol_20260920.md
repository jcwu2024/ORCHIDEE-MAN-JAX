# Four parallel learning arms, frozen before submission

## Question and authorization

Selected100 reduced carbon-stock drift but did not jointly improve T30 GPP,
NPP, respiration and biomass relative to selected20. The user approved four
paid runs after reviewing a maximum CNY33.12 budget. Run the arms concurrently
to distinguish duration, litter weighting, recurrent weighting and learning
rate effects. These are diagnostic continuation experiments, not the final
fresh-initialization training protocol or proof of architecture acceptance.

All arms start from the SAME completed selected100 parameters and AdamW state
(absolute optimizer step4100), preserving the useful carbon corrections.
They do not start from different historical checkpoints or independently
reset optimizer moments. This evaluates behavior near that common state;
it does not establish that the same hyperparameters work from initialization.

## Frozen matrix and data

| Arm | Litter coefficient | Learning rate | D/T7/T30 outer weights | Only change from A |
| --- | ---: | ---: | --- | --- |
| A | 5.221716892123007e-5 | 1e-5 | 1 / 0.5 / 0.5 | None; duration reference |
| B | 1.3054292230307518e-5 | 1e-5 | 1 / 0.5 / 0.5 | Quarter litter coefficient |
| C | 5.221716892123007e-5 | 1e-5 | 1 / 0.5 / 1 | Double T30 outer coefficient |
| D | 5.221716892123007e-5 | 3e-6 | 1 / 0.5 / 0.5 | Lower learning rate |

D contains the unchanged original daily terms plus the weighted litter term.
Apply the litter coefficient once. C changes the scalar T30 weight in BOTH
the reported combined objective and its combined gradient, not its data,
inner science extractor or differentiation boundary. D changes the effective
optimizer learning rate after source-state validation, not historical source
metadata. The original Attention architecture, decoder,109,464,375 trainable
parameters, native JAX AD, FP32 execution policy, AdamW/clip1 remain unchanged.

Each arm performs500 NEW updates: ancestral relative101..600, absolute
4101..4600. Use existing daily/recurrent schedule indices4100..4599 in the
same order across arms, with64 daily draws/microbatch8 +oneT7+oneT30 each
update. Per arm32000 daily draws and500 windows per horizon, not necessarily
unique samples or a full Teacher epoch. Do not regenerate Teacher data.

Executable matrix: `manifests/coarse_graining/litter_parallel_training_v1.json`.
Parent checkpoint:
`runtime/outputs/diagnostics/litter-mixed-continuation-v1.selected-continuation-update-000100.checkpoint.pkl`.
Parent report: `runtime/outputs/diagnostics/litter-mixed-continuation-v1.json`.

## Assessment, stopping and interpretation

Evaluate all arms at100/250/500 NEW updates (ancestral relative200/350/600),
using the unchanged16-window T7/T30 development panel and fixed/excluded daily
TRAIN panels. Reuse the saved selected100 baseline. Report physical-unit
cumulative GPP/NPP/autotrophic respiration errors, AGB/BGB/canopy/nonliving
stock-change errors, signed changes and wrong-sign counts. Preserve all three
milestone checkpoints, saving before evaluation. One rolling recovery every
25 updates; compact per-update logs, full report every10 updates/lifecycle.

Stop on actual nonfinite/physical violations,500 new updates,175min inner
wall limit, or the same physical metric exceeding2x the COMMON selected100
RMSE at two consecutive assessments AND exceeding the absolute increase
floor:1gC/m2 flux/biomass,5canopy,10nonliving. This is a broad diagnostic
ceiling, not scientific acceptance. No automatic job extension or restart.
An incomplete or stopped arm cannot be called a completed500-update arm.

Do not rank arms by total loss because the objectives differ. Report each
physical objective and the full learning trajectory; A is an experimental
reference, not an adequate scientific baseline. Comparisons between arms
with different stopping points must use shared completed milestones.
No criterion here promotes a final model. The development panel has been
repeatedly consulted; independent site/window assessment is required before
claiming generalization. Current all-zero RH targets cannot certify nonzero
RH learning. Endpoint bookkeeping does not certify ecosystem flux closure.

## Runtime, cost and deployment

Measured warm single-GPU updates in the completed continuation: median12.703s
(79 updates excluding first compilation-bearing update).500 warm updates
therefore take105.86min. Startup/compilation/checkpoints/evaluation add time;
plan roughly2–3h per arm, not a completion guarantee. Four independent arms
run concurrently and need no new multi-GPU collective implementation.

gnall array0-3%4, each task1node/1V10032/8allocatedCPUs/64GiB,3h hard limit,
175min inner timeout, no requeue. Use project rate0.07yuan/CPU-hour (newer
deployment documentation),2.20yuan/GPU-hour, assumed additive:
CPU4*8*3*.07=6.72;GPU4*1*3*2.20=26.40;total maximum33.12yuan.
All four arms are paid as explicitly requested; no additional paid probe.

Working directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/litter-parallel-v1`.
Command: `sbatch scripts/hpc/job_litter_parallel_training.sh`.
Outputs: `runtime/outputs/training/litter-parallel-v1/{A,B,C,D}/report.json`;
Slurm logs `slurm-ARRAYID_TASKID.log` in the parent directory.

Before submission, exercise target-shell dry-run for all four array mappings,
validate source report/checkpoint/schedules through the actual deployed CLI,
and test scalar/gradient weighting and recovery offsets locally. Reuse admitted
AD evidence because only scalar weights and optimizer LR change; no new
physical model, state structure or AD boundary. The first complete paid
updates serve as bounded live execution checks before the same process
continues. Do not add bulk hashes or repeat historical gradient investigations.

Three milestone containers and one replaced recovery per arm require roughly
21GB for four arms, plus reports/logs and small preflight outputs, fewer than
100 experiment output entries. Source data/checkpoints are shared read-only;
no per-day/sample files or duplication of the Teacher pool. Account quota
remains a separate limit; existing data is not authorized for deletion.

## Preparation evidence

28 focused local tests pass, covering the new matrix, effective optimizer,
weighted-gradient/direct-scalar agreement, checkpoint-history alignment and
the prior continuation/admission/physical-monitor behavior. Module compilation
and CLI help pass. The actual saved selected100 report validates with all16
development windows and two daily panels. Target-shell array mappings were
exercised with a non-default GPU index; log/output directories exist before
submission. Final deployed source-checkpoint reads and submission are pending.
