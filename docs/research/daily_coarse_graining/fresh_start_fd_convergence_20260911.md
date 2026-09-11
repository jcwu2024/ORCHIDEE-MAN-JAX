# Fixed-direction finite-difference convergence and branch localization

The free gsmean-v2 run at 85e6b39 stopped before any update after 2005.78s.
All initial hard counts were zero and the full gradient was finite, norm
0.635383586 (previous invalid gsmean version:758.451725). The gsmean head
norm is now1.06166e-7, retaining vegetation response. Reverse and JVP agree:
0.635381858/0.635381852 at epsilon1e-3, but central FD is0.466128604.
At3e-4, reverse/JVP/FD are0.635379711/0.635379709/0.635347742 and pass.
This does not establish that large-step curvature/branching, rather than an
AD error, caused the larger discrepancy; the following audit distinguishes it.

The user explicitly canceled queued/held paid job15005758; scheduler confirmed
CANCELLED with RunTime00:00:00. No new paid job or training is part of this audit.

## Frozen diagnostic

Keep v2 architecture, input/output rules, seed20260911, full trainable tree,
loss/scales and sample0 (point217.0-109.0,1963,day6). Reproduce the stopped
run's initial scalar and unit-gradient direction. Do not alter weights,
tolerances, loss eligibility or model branches to force agreement.

Evaluate exactly epsilons [1e-3,7e-4,5e-4,3e-4,1e-4,3e-5,1e-5], both signs.
Use the actual rounded float32 central displacement for the reverse/JVP
comparison, as in v2. Record central and one-sided finite differences, actual
rounded direction norms, and weighted flux/state/process finite-difference
contributions. Use the original atol1e-6/rtol1% throughout.

For baseline and each endpoint, export diagnostic state values and defined
masks, group counts/losses and next discrete state. Compare complete state
masks/discrete values against baseline, and identify any changed canonical
leaf. Keep physical hard counts separate from changed-but-declared masks.
Also compare diagnostic-output scalar with the ordinary scalar at baseline
and the two original endpoint pairs, since changing returned outputs can
change numerical fusion. Save compact state diagnostics, not parameter weights.

Classify a local convergence plateau only when at least three consecutive
tested epsilons <=3e-4 have unchanged state/discrete masks, valid hard counts
and passing reverse/JVP/FD comparisons. A single passing epsilon is insufficient.
This is a bounded local-direction classification, not global AD or training
acceptance. Report failures at larger epsilons rather than suppressing them;
use component/mask evidence to interpret their cause before changing admission.

Definition-head logits are included in the state diagnostic export so any
learned defined/undefined switch can be tied to its threshold, rather than
attributed from a mask count alone.

Local classification/mask tests and complete CPU shape tracing precede one
free GPU audit, at most45 minutes, reusing the warm compilation cache. No
optimizer state or checkpoint is written. A later learning run requires an
explicit versioned admission rule supported by the measured convergence and
branch evidence; preserve the original v2 failure record.

## Local and CPU preparation

Five local tests pass (7.37s), covering unchanged tolerances, a genuine
three-scale plateau requirement, mask/discrete-change reporting and the
existing loss-attribution/layout checks. Ruff passes. Free test02 complete
diagnostic shape tracing at f4378e2 passed in194.4s, report
`runtime/outputs/diagnostics/fresh-start-fd-convergence-shapes.json`.
The additional definition-logit export simply returns existing computed
arrays; it changes neither the ordinary loss nor model parameter values.

## Free-GPU execution

Launched late 2026-09-11 on free gln01 GPU1, executable checkout **35e608a**,
Python PID **908**, 45-minute cap. Both GPUs were empty before launch; only
GPU1 is used. Working directory `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.

Command:
`GPU_DEVICE=1 PYTHON_MODULE=scripts.hpc.audit_fresh_start_fd_convergence RUN_TIMEOUT=45m OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/fresh-start-fd-convergence-v1.json bash scripts/hpc/run_fresh_start_one_day.sh --protocol manifests/coarse_graining/fresh_start_one_day_probe_v2.json`.

Report: `runtime/outputs/diagnostics/fresh-start-fd-convergence-v1.json`.
State/logit export: same prefix with `.states.pkl` suffix.
Log: `runtime/logs/fresh-start-fd-convergence-v1.log`.
Bootstrap finished at188.9s and initial-gradient/diagnostic preparation is
running. The report must be read before any admission amendment or new training.
Keep the server checkout stable and do not launch a duplicate. Paid job15005758
remains canceled with zero runtime; no new paid work.
