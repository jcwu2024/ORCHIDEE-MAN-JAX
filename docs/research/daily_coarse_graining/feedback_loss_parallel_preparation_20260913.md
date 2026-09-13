# Work prepared while loss calibration runs

Date: 2026-09-13. Parallel preparation completed as calibration finished.
Trial weights are now recorded in `feedback_loss_calibration_results_20260913.md`;
target-runtime shapes and actual updates remain pending.

## Two-update technical admission

The required next execution has a concrete scope: first two32-row batches of
epoch0 from the existing train-only sampler, microbatch4, fresh seed20260911,
all109,464,375 parameters. It uses the same dynamic combination of the three
shared loss components as calibration. It inherits no checkpoint and writes
no model checkpoint. Running calibration checkout ac5aa51 stays immutable.

The configuration template is
`manifests/coarse_graining/feedback_loss_update_admission_v1.template.json`.
Null supplemental coefficients are intentional placeholders. They must be
replaced with explicitly reviewed finite weights after the completed
calibration is read; they do not default to zero or authorize a launch.
The scope does not require the user to choose technical loss coefficients.
The agent makes and records that engineering choice from the evidence.

`scripts/hpc/probe_feedback_loss_updates.py` implements native reverse AD and
two independent JVP projections of the actual combined first-batch scalar,
then ordinary AdamW with microgradient accumulation and clipping after the
whole32-row mean. Directional tolerance is the existing relative1% plus
absolute1e-6; no tolerance change or custom observation cotangent is involved.
Changing only optional loss weights should reuse a coefficient-dynamic graph.

Before and after each update, evaluate the same64 selected TRAIN rows so
different input batches cannot masquerade as learning. Report each component
loss, GPP/NPP/maintenance/growth/total-autotrophic/heterotrophic respiration,
AGB/BGB changes, canopy stock-change residuals and litterpart input-unit errors.
Also report full-gradient and optimizer finiteness, state/defined/discrete and
budget hard counts, clipping, cold/hot time and actual versus reserved memory.

Numerical/physical failures or failed directional checks stop the probe.
Finite updates are technical admission only; an unfavorable science change
is recorded honestly and is not repaired by silently selecting another weight
or continuing for more steps. Two updates cannot certify long-run learning.

`run_feedback_loss_updates.sh` provides shape and free-GPU modes. Before real
execution, exercise actual Bash expansion and real data shapes on a free CPU
node. The planned GPU cap is60m, preceded by the small runtime check when the
new JVP/optimizer boundary requires it. Reuse unchanged evidence; no paid job
is submitted as part of this preparation.

## Calibration interpretation

`feedback_loss_geometry.py` and `scripts/dev/summarize_feedback_loss_calibration.py`
read the saved scalar gradient summaries without any model execution. For
explicitly supplied weights, the three-component Gram matrix gives the norm
of their combined raw gradient and each component's directional derivative.
This can expose conflicting objectives before spending time on updates.

This is an infinitesimal Euclidean-gradient calculation. Adam's
preconditioning and a finite step can change its implications, so it does not
select coefficients automatically, guarantee descent, or replace the actual
two-update check. A shape-only or incomplete calibration cannot be marked
ready for weight review. Reading existing saved gradients creates no new GPU
or data preparation work.

## Following learning and rollout acceptance

The separate plan `feedback_loss_learning_acceptance_20260913.md` specifies
the later bounded-learning and7/30-day evidence boundary. It keeps physical
productivity, respiration and biomass outcomes as the primary criteria.
Neither fitting litterpart nor reducing canopy loss is sufficient for model
acceptance. The development panel has already informed loss design and must
not be presented as a final independent test. The historical checkpoint loader
must not be silently weakened to accept an arbitrary new checkpoint.
