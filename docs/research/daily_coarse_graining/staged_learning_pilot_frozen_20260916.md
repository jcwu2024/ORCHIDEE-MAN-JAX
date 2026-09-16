# First matched-compute learning segment

This freezes the first learning segment before seeing either arm's outcomes.
It compares early compute efficiency and learning trends, not converged-model
superiority. A temporarily worse arm must not be rejected merely on this
endpoint. The user explicitly raised the possibility of later crossover.

## Budget evidence and frozen method

Free single-GPU budget profile e3150b9 completed, wrapper exit0,
Python905.5105s. Source: `outputs/mixed-learning-budget-profile-v1.json`, with
the same basename under remote runtime/outputs/diagnostics. Three warmed
complete daily64 updates took6.0247,5.9896,5.9545s; mixed updates took12.0375,
11.7693,11.7351s. Reads, checks, gradients and discarded AdamW proposals are
included. All physical/finite gates pass; no weights were retained. Fixed-data
timing does not establish rotating-pool throughput.

The preregistered calculation yields400*5.9896466/11.7692560+400=603.5693,
rounded to600 joint updates. Staged remains400daily+400mixed. Nominal update
times are1.9615/1.9732h; startup, changing samples, evaluation and checkpoint
IO are additional. The provisional joint600 name is now frozen and its
update100 monitor is added before launch to match early compute landmarks.

Both arms independently initialize109,464,375 parameters from seed20260911.
All neural parameters train throughout; no exploratory checkpoint is inherited.
Use unchanged admitted AttentionFP32/checked split, physicalFP64, daily64/micro8,
AdamW lr1e-5/clip1, dataset/splits, sampling seeds and component losses. Joint
always uses D+.5R7+.5R30. Staged changes only objective inclusion: D for1..400,
then D+.5R7+.5R30 for401..800, retaining AdamW moments/global step. Recurrent
cursor starts0 at staged401 and advances only during mixed updates.

Joint consumes38400 daily draws and600 windows per horizon; staged consumes
51200 daily draws and400 windows per horizon from the existing prepared pool
and5415-row schedule. The draws and windows overlap between arms; these are
not additive unique coverage counts and neither arm is an epoch. No Teacher
generation, extra data expansion or architecture change is part of this job.

## Learning-curve interpretation

Joint checkpoints:0/100/200/400/600; staged:0/200/400/600/800. The corresponding
post-initial pairs have approximately equal nominal update compute. Plot the
actual cumulative update seconds and total elapsed times as well, because
fixed-input estimates are not exact matching of observed rotating-data costs.

Use the same fixed development windows and existing daily TRAIN/excluded
monitors at every checkpoint. Report physical-unit GPP,NPP,RA,RH,AGB,BGB,
canopy and nonliving carbon, recurrent T7/T30 errors/biases, physical validity
and population counts. Compare like-for-like physical metrics; daily-only and
mixed training losses are different objectives and cannot rank arms directly.
The development panel is not an untouched final test set. Zero-target RH cases
do not establish learned nonzero respiration response.

Show all checkpoints, endpoint absolute accuracy and changes over the last two
evaluation intervals. Do not select a favorable intermediate checkpoint and
hide later deterioration. Falling aggregate loss or beating the other weak
arm is not scientific acceptance. If trajectories cross, remain close, or the
lagging arm is still improving, report the ranking as unresolved. If both are
poor but improving, report insufficient evidence about eventual adequacy,
not proof that more training will succeed or that the architecture fails.
Worsening RA/carbon feedback must be reported even if GPP improves.

Retain full checkpoints/optimizer states for diagnosis and a possible declared
extension. There is no automatic training extension or paid retry. Any extension
needs a specified budget, recovery identity and comparison rule before launch;
the current600/800 limits remain binding. One seed cannot establish general
superiority or final long-term accuracy. This is the first curve segment.

## Paid resource proposal, ready for concrete approval

Two independent array tasks on gnall, at most two concurrent. Each requests
1V10032,8 allocated CPUs,64GiB host RAM,4h Slurm limit and220min Python limit.
The array runs joint/staged simultaneously when available; a failure in one
does not prevent the other's start. Neither task performs dual-GPU training.
No automatic requeue. Estimated maximum per task:8*4*.07+1*4*2.20=11.04yuan;
both=22.08yuan assuming additive CPU/GPU billing. Expected each roughly2.5-3.5h
including overhead, excluding queue; the4h cap is not a completion guarantee.

Working directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-stages-pilot-v1`.
Command: `sbatch scripts/hpc/job_mixed_learning_stages_pilot.sh`.
Outputs: `runtime/outputs/training/mixed-learning-stages-pilot-v1/{joint,staged}/report.json`;
Slurm log: `slurm-%A_%a.log` in their parent directory. Create that directory
before submission. Free dual v2 remains independent and its running checkout
must not be changed. User authorized use of paid cards; concrete22.08yuan
two-task envelope is to be confirmed before sbatch under AGENTS.md.
