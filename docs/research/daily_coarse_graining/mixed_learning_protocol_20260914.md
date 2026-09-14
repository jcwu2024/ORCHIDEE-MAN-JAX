# Fresh 200-update mixed learning protocol

User approved the 200-update task on 2026-09-14. This is bounded learning,
not a final model or independent validation. It supersedes the implementation
pending language in `mixed_learning_followon_design_20260914.md`.

## Executable method

`manifests/coarse_graining/mixed_learning_v1.json` and
`mixed_learning_model_v1.json` freeze the method. The entry point is
`scripts/hpc/run_mixed_learning.py`, invoked by `run_mixed_learning.sh`.
The existing Attention architecture has 109,464,375 parameters. All parameters
are initialized from seed20260911 and updated jointly. No exploratory weights
are inherited. AdamW uses constant learning rate1e-5, beta1=.9, beta2=.999,
epsilon1e-8, weight decay0, and global clip norm1.

Each update combines the mean of64 daily samples (microbatch8), one T7 and one
T30 complete trajectory. The objective is `D + .5 R7 + .5 R30`. The admitted
physical-feedback-v2 daily loss and recurrent-science loss/scales are unchanged.
Each component is differentiated with native JAX AD, then means are combined,
clipped once and passed through exactly one AdamW update. There is no manual
observation cotangent, architecture change or new half-hour inference scan.

One-day sampling uses the existing346539-day expanded TRAIN pool, seed20260912,
fixed_replay_pad, batch64/micro8. Metadata inspection on test02 established
max5 artifacts per batch over the whole5415-batch epoch; the first200 batches
contain12800 distinct Teacher days,372 sites and44 years, with zero replay.
The ready recurrent schedule uses seed20260914,400 unique semantic windows,
267 sites,44 years and7387 unique days. The64 fixed monitor and296 development
dates are excluded. No Teacher regeneration or payload duplication is needed.
Source manifests, ordered metadata, both cursor positions and initialization
are bound into recovery. Daily and recurrent sources have separate identities.

## Monitoring and recovery

Retain complete parameter/AdamW checkpoints at0/50/100/200; overwrite a single
rolling recovery every10 updates. Files are written atomically. Resume selects
a newer durable milestone when it is ahead of the rolling recovery. The daily
batch cursor, recurrent pair cursor and AdamW step must all match. Resume is
restricted to the same immutable code and method, without learning-rate reset.
Every update records its scalar components, norms, clip, timing and hard counts.

At each milestone evaluate the fixed first training daily batch, first TRAIN
T7/T30 pair, excluded64-day monitor and unchanged8T7+8T30 development panel.
The shared metrics report physical-unit GPP/NPP, maintenance/growth/total
autotrophic respiration, RH, biomass and carbon stocks; canopy metrics are
reported separately. Zero Teacher RH totals are explicitly identified.
The existing13-key development gates remain T7 NRMSE<=.35/bias<=.10 and
T30<=.25/<=.07. Poor accuracy is reported without stopping at50/100 or changing
weights/data. Nonfinite values/gradients/updates, established physical hard
counts, missing metric populations or resource failures stop technically.
Update200 is the declared final bounded result; report all milestones.

The integration mode uses the same runner with one update and initial/final
monitors plus a full checkpoint replay. It has a separate run identity; its
weights cannot serve as the formal200-update initialization. Reuse completed
daily/T7/T30 gradient evidence: this integration is not another JVP audit.

## Resources and current execution

Implementation and43 relevant local tests pass. Actual test02 metadata and
target-shell launcher expansion pass. Free GPU0 integration is running at
48659ae (Bash65184/Python65235): update0 full checkpoint saved203s, initial
fixed monitors in progress. Localce6e655 contains only recovery-report and
monitor-reference cleanup, to be deployed after that checkout exits.
Free gln01 integration is limited to90 minutes on one idle V10032GiB.
`job_mixed_learning.sh` prepares gnall, one V100,8 allocated CPUs,64GiB host RAM,
6-hour limit. At additive rates its ceiling is CPU8*6*.07=3.36 yuan plus
GPU1*6*2.20=13.20 yuan, total16.56 yuan. This is a proposed upper bound,
including setup/compile, changing-data reads,72 monitored trajectories and
checkpoint IO, not an estimate from warm update time alone. Revise downward
if measured integration supports it; do not silently raise the bound.

Paid workdir: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-v1`.
Submission command: `sbatch scripts/hpc/job_mixed_learning.sh`.
Output: `runtime/outputs/training/mixed-learning-v1/report.json` and sibling
checkpoints. The concrete envelope still requires approval before sbatch.
User's task approval authorizes implementation and free checks immediately.

Four retained checkpoints plus one rolling file, one report and one Slurm log
create a bounded number of files. Each float32 parameter+two-moment snapshot
is approximately1.22GiB; five snapshots approximately6.12GiB plus one atomic
temporary write at a time. The existing data containers are reused.
