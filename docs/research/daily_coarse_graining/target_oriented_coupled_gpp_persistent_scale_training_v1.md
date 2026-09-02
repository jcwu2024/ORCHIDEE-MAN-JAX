# Persistent Scale-Selected Coupled-GPP Training

Date: 2026-09-03

Status: frozen; free `gln01` runner admission accepted, paid authorization pending

Machine contract:
`manifests/coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_v1.json`

Contract SHA256:
`1742d9d19a6b8e8d91331723bf06047a17fa5ce5d31cc528987205a838d8b5e4`

## Purpose

This is the first persistent trial of the accepted scale-selecting coupled-GPP
method. It keeps the same Attention candidate, 15,960,195-parameter trainable
boundary, conservative updater, canonical retained exact tail, four training
sources, and common-direction construction used by the accepted updates 5-7.
It does not open an architecture search or introduce an optimizer.

The trial tests whether repeated bounded updates can retain the independent
7/30-day gains while recovering the direct teacher-state GPP loss that
regressed in the transfer audit.

## Training Method

Each update uses eight train/train artifacts and averages four source losses
and their gradients:

- 1,664 direct teacher-state GPP days;
- 256 reachable one-day state transitions;
- 64 seven-day windows;
- 32 thirty-day windows.

The minimum-norm simplex direction over unit source gradients is recomputed
after every accepted update. The runner evaluates scales `1`, `1/2`, `1/4`,
`1/8`, `1/16`, and `1/32` in descending order and retains the first scale for
which every source stays within the frozen relative tolerance, the weighted
objective strictly decreases, and all finite/physical gates pass. There is no
AdamW state, momentum, weight decay, or learning-rate schedule.

The persistent pool excludes calibration artifacts, monitor artifacts, and
all artifacts consumed by updates 1-7. It contains 678 artifacts. The frozen
PCG64 schedule presents 800 artifacts in 100 eight-artifact updates; repeats
can occur across epochs but not within one update. Its SHA256 is
`c79c5f2e125706d73b306ccd5aa23582a062b6f387d55224edee52666492548d`.

## Monitor And Stop Rules

The independent train/train monitor runs initially and after updates 10, 25,
50, and 100. One rollout evaluation produces both the component-resolved
training objective and absolute per-key reports for GPP, maintenance/growth/
heterotrophic/autotrophic respiration, NPP, NEP, NEE, AGB, BGB, live biomass,
nonliving carbon, and ecosystem carbon.

At update 10, direct GPP may not be worse than the after-update-7 state;
one-day state and 7/30-day aggregate gains must remain better than the
pre-update-5 state; cumulative 7/30-day GPP may not be worse than the initial
persistent monitor; and all physical gates must pass.

A checkpoint is selectable only when all continuation checks pass and direct
GPP has recovered to the pre-update-5 baseline. If this has not happened by
update 25, the run stops without selecting a checkpoint. Once a checkpoint
has been selected, a later milestone that loses this full gate stops the run.
Aggregate improvement cannot compensate for a GPP failure.

## Persistence

After every accepted production update, the runner atomically writes a
non-promotable recovery containing only the trainable parameter tree, update
records, and monitor records. It does not persist fixed parameters or an
optimizer state. A non-promotable checkpoint is written only at a milestone
that passes the full GPP recovery gate. Resume requires the exact experiment
identity and parameter-tree schema.

## Admission

The free `gln01` admission passed at commit `7d57887`. It reconstructed the
selected update-7 state, evaluated the full detailed initial monitor, executed
one discarded persistent update, and round-tripped a temporary parameter-only
recovery. No checkpoint or recovery remained. See
[`target_oriented_coupled_gpp_persistent_scale_training_admission_result_20260903.md`](target_oriented_coupled_gpp_persistent_scale_training_admission_result_20260903.md).

Paid training still requires a separate review and explicit approval of the
exact Slurm resources, wall-time limit, command, working directory, and
worst-case charge. Fresh validation, 365-day rollout, Gate D2, sealed test,
and product acceptance remain closed.
