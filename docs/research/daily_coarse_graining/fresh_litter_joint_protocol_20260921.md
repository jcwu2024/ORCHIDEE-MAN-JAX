# Fresh joint training with litter supervision from initialization

Status: recipe and launch preparation, pending real-model fresh admission.
No new paid run is submitted or authorized. This is a method-development
candidate, not the accepted final scientific model.

## Scientific question and fixed method

Can the existing Attention learn production, biomass and carbon-stock change
together when litter supervision is present from the first update, rather
than added after 4,000 updates? The completed inherited four-arm experiment
does not answer this question: T7 improved, but all final T30 production
errors exceeded their common start. We retain one minimal-change recipe,
not four more arms and not an architecture replacement.

Initialize all 109,464,375 neural parameters with seed20260911 and fresh
AdamW moments. No exploratory or integration checkpoint initializes training.
Use the existing canonical daily binding, conservative updater, retained tail,
native checkpointed JAX AD and FP32 Attention/FP64 physical execution policy.
No Teacher regeneration, v5 data, or 48-step reconstruction is introduced.

The scalar objective is D + 0.5 R7 + 0.5 R30. D consists of the existing
flux MSE (weight1), state anchor (0.1), necessary process supervision (0.01),
canopy change pseudo-Huber (0.1), litterpart feedback MSE (0.1), and separate
above/below litter change pseudo-Huber (5.221716892123007e-5). The latter uses
TRAIN coordinate-variance scales with floor0.01gC/m2/day and source PFT13.
This coefficient came from an exploratory trained-state calibration; it is
a fixed candidate setting, not proof of balanced gradients at initialization.
Recurrent terms and all their coefficients are recorded in the mixed manifest.

AdamW: LR1e-5 throughout, beta1=.9, beta2=.999, epsilon1e-8, weight decay0,
global clipping norm1. All modules remain trainable throughout. No adaptive
loss changes, learning-rate search, freezing or phase switch during this run.

Executable declarations:

- `manifests/coarse_graining/mixed_learning_fresh_litter_model_v1.json`
- `manifests/coarse_graining/mixed_learning_fresh_litter_v1.json`
- `scripts/hpc/run_mixed_learning_fresh_litter.sh`
- `scripts/hpc/job_mixed_learning_fresh_litter.sh`

The shared batch objective already preserved optional litter supervision in
its three-component mode. Explicit fourth-component wiring makes its weight
and diagnostics visible; it must not multiply the coefficient twice.

## Sampling, duration and resource segmentation

Reuse existing TRAIN data and replayable ordering: daily seed20260912,
recurrent seed20260914. Each update averages64 daily samples using8
microbatches of8, plus one T7 and one T30 trajectory. GPU0 owns the daily
branch and AdamW; GPU1 owns recurrent gradients. Combine once and synchronize
the updated parameters. Batch statistics do not change with device layout.

The complete declared method is one prepared-pool pass of5415 updates:
346,539 distinct daily rows,535sites,44years,17,677point-years, and21 terminal
padding draws. This is about4.04% of eligible Teacher TRAIN days, not a full
Teacher epoch. Recurrent and daily coverage overlap; do not add their counts.

Evaluate and retain updates0/100/250/500/1000/2500/4000/5415. Final endpoint
selection is5415; report earlier checkpoints without selecting the nicest
curve retrospectively. The proposed first resource segment ends at1000:
64,000 daily draws and1,000 windows of each horizon. It is intermediate,
not a complete pass. A resource boundary preserves parameters, AdamW and data
cursors under the same immutable method, and does not permit changing losses.
There is no automatic paid continuation or second-seed run.

Stop on nonfinite loss/gradient/state, failed physical checks or resource
limit. Ordinary scientific inaccuracy is reported, not hidden as a runtime
failure. Save recovery every50updates and milestone checkpoints. An explicitly
bounded segment must complete its monitor and persisted-forward replay, then
report stopped_for_recovery rather than completion of the full method.

## Admission and evaluation

First run the deployed small runtime-interface check, then a separate fresh
two-update joint integration on free GPUs. Check finite/nonzero gradients,
complete parameter updates, daily/recurrent hard counters, checkpoint replay,
memory and wall time. Reuse existing unchanged native-AD evidence and test
the explicit optional-loss composition locally. This is not a new adjoint
implementation or a requirement to match trained accuracy at random weights.

Keep fixed TRAIN daily/recurrent monitors and the old16 development windows
for continuity. These repeatedly consulted windows are not sealed validation.
Report physical-unit GPP/NPP/RA/RH, AGB/BGB growth, canopy/nonliving stock
change, signed bias and per-window errors. Preserve the difficult windows.
All-zero RH targets remain insufficient evidence of nonzero RH fidelity.
Broader evaluation selection and its actual split membership must be recorded
before viewing new predictions. A TRAIN exclusion must not be called a
spatially held-out site. No final accuracy thresholds are inferred from the
best exploratory arm, and normalized gates do not imply scientific acceptance.

## Proposed first paid segment, not submitted

gnall:1node,2V10032GB,8allocatedCPUs,64GiBRAM,5h ceiling,295min inner timeout.
Command: `sbatch scripts/hpc/job_mixed_learning_fresh_litter.sh`.
Working directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/fresh-litter-v1`.
Output directory must exist before sbatch opens its log.

Worst-case charge: CPU8*5*.07=2.80yuan; GPU2*5*2.20=22.00yuan; total24.80yuan.
This is a proposal pending fresh integration timing and explicit approval.
Prior sustained full-run and litter-arm timings justify roughly4–5h planning
for1000updates including startup and monitors; it is not a completion promise.
Keep eight milestone checkpoints plus rolling recovery for the complete pass,
roughly11GiB plus one temporary checkpoint, with bounded file count.
