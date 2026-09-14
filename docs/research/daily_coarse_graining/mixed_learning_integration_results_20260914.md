# Mixed learning integration results

The free GPU integration completed successfully at48659ae: Bash65184 and
Python65235 exited, launcher exit0. Remote report/log are
`runtime/outputs/diagnostics/mixed-learning-v1-integration.{json,log}`;
the JSON is also under local `outputs/`. This is one update of the new
persistent daily/T7/T30 runner, not a trained-model acceptance result.

All109464375 parameters started fresh and received one AdamW update of
`D + .5 R7 + .5 R30`. Component gradient norms were2180.5776/494.7655/275.0101;
combined norm2499.2717, clip scale0.0004001166. All are finite. Every training
daily/T7/T30 hard count was zero. Both initial and updated fixed TRAIN and
development panels passed physical validity and metric-population checks.
The development science screen failed both before and after the update, as
expected for this fresh one-update technical test; do not label it learning
or generalization success. Training daily loss312.456552 ->312.004942;
excluded64-day loss581.752321 ->581.195920. Changes remain small.

On the unchanged T30 development panel, daily GPP RMSE2.103943 ->2.102212,
NPP1.825608 ->1.822899, RA.347214 ->.348057gC/m2/day. AGB change RMSE37.238195
->37.178992 and BGB17.207383 ->17.183978gC/m2. Canopy change RMSE81478.757226
->81478.746796gC/m2 is effectively unchanged. Thus tiny improvements coexist
with slightly worse RA and unresolved canopy error. These fresh one-update
results do not justify any architecture or loss retuning before the declared
200-update learning budget.

Full initial and update1 recovery checkpoints were written, and the restored
update1 checkpoint reproduced the actual daily forward value exactly. Local
tests additionally cover interrupted/resumed AdamW equivalence and both data
cursors. No new native/JVP check was requested or required.

## Runtime and next resource envelope

Total1079.145s (17.99min). Explicit compile stages: T7 215.813s, T30 222.152s,
daily101.230s. Initial full checkpoint saved at203s; initial monitors171.732s,
updated monitors29.823s. First complete update129.053s includes first-use
execution/compilation of helper and forward paths; there is no second warm
update in this integration, so do not claim129s is a measured steady speed.
Earlier actual T30 mixed admission had20.76s warm full-update time but a
different horizon mixture. These give resource-planning evidence, not a
guaranteed200-update finish time.

Peak live device17145761792 bytes (~15.97GiB); reserved28967692544 bytes
(~26.98GiB). Host peak20660785152 bytes (~19.24GiB). Initial checkpoint875821584
bytes (shared zero Adam moments serialize once); updated checkpoint1313707031
bytes, written in2.156s. Reuse data containers and keep bounded checkpoint files.

Prepare one gnall V10032GiB,8 CPUs,64GiB host RAM,6h hard limit (355min inner
timeout), maximum additive charge16.56yuan. This allows startup, changing-data
IO, four complete milestone monitors, checkpoints and variation in step time.
It is a cap, not a promise of six hours of use. Rotating-data throughput remains
to be measured inside the approved learning run; rolling recovery limits loss
of work if the bound is reached. No extra standalone benchmark is required.

After the free processes exited, the execution checkout was fast-forwarded
to12eb22c, including previously tested recovery-report completeness and prompt
monitor-trajectory release. These edits do not change losses or AD. Actual
Slurm-entry `--dry-run` and shell syntax passed on gln01. Output directory is
ready and no training report exists there. Submission command is
`sbatch scripts/hpc/job_mixed_learning.sh` from
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-v1`.
The200-update task is approved, but its newly specified concrete fee envelope
has not yet been approved; no paid job has been submitted.
