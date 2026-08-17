# Mechanistic-Output One-Point Qualification Result

Date: 2026-08-17.

Status: not qualified; total loss, primary science, and all hard gates pass in
the first formal run, while four independent expert-reduction gates fail. A
second attribution run confirms the residual families and exposes training
trajectory sensitivity. No checkpoint is retained.

## Boundary And Data

Both runs use the frozen v6 Teacher release, one complete 364-day train/train
point-year at `001.0-071.0` in 1961, batch size 8, fresh seed `20260815`, and
the existing 100-epoch unclipped AdamW schedule. The candidate has 49,744,537
parameters and learns 165 fields/39,153 scalar outputs. It materializes the
complete 181-field/39,371-value process boundary around one conservative
update and then enters the canonical retained exact tail.

No Teacher data is regenerated, no v5 row is read, no checkpoint is loaded or
written, and no 48-record forcing reconstruction or 48-step state scan occurs.

## First Formal Run

Remote report:

`runtime/outputs/training/mechanistic-output-one-point-qualification-local-20260817-a/report.json`

SHA256:

`b72ac45f6821337ded953c93d1856bec33a11207b40a032b66a140d5588daa0b`

The run completes all 4,600 updates in 1,912.76 seconds. All gradients remain
finite. Total process loss falls from `4.5812711` to `0.0410207`, a
`99.1046%` reduction that passes the `99%` total gate. GPP, maintenance
respiration, NPP, AGB tendency, and BGB tendency NRMSE are respectively
`0.01936`, `0.00294`, `0.02333`, `1.04e-5`, and `2.08e-5`; all pass the
`0.05` gate. Every prediction/state/budget/inventory hard count is zero.

Three expert reductions pass:

- daily interface/finalize: `99.4750%`;
- maintenance respiration: `99.9920%`;
- soil thermal: `99.8001%`.

Four expert reductions fail the independent `95%` gate:

- canopy GPP: `93.9165%`;
- hydrology/snow: `83.0539%`;
- litter/POC/DOC: `80.3503%`;
- surface energy: `92.2124%`.

This is a substantive improvement over the previous baseline, whose total
reduction remained `95.4744%` after 100 epochs plus 25 exact-gradient updates.
It is not a qualification pass because aggregate success cannot hide an
unlearned expert.

## Attribution Reproduction

Remote report:

`runtime/outputs/training/mechanistic-output-one-point-attribution-local-20260817-a/report.json`

SHA256:

`86e7f64eec9db75109a8092226d9dbdff5533dc13b26c659e649447c46c7b539`

This run adds read-only initial/final per-expert, per-label, per-field, and
exact-derived diagnostics without changing the training target or retaining
parameters. It completes all 4,600 finite updates and again passes every
primary science and hard gate. Its total reduction is `98.4957%`, lower than
the first run. One epoch reaches a maximum minibatch gradient norm of
`1559.87`, while its median is `1.70`; an earlier batch-loss spike recovers on
the next epoch. The changed read-only evaluation order is sufficient to
produce a different GPU optimization trajectory, so the current recipe has a
training-stability risk in addition to its structural residuals.

The dominant learned-field residuals are not distributed uniformly:

- hydrology endpoints: `humrel=69.22`, `humrelv=11.56`, `temp_hydro=0.834`,
  `ae_ns=0.752`, `nroot=0.570`, and `evap_bare_lim=0.450` normalized field MSE;
- canopy capacity: `qsintveg.debt.incoming_amount=3.276`;
- coupled energy endpoints: `fluxsens=1.260`, `vevapp=1.050`, `evapot=0.766`,
  `evapot_corr=0.688`, and `qsurf=0.279`;
- energy integrals: ground heat `0.222`, net radiation `0.0712`, sensible heat
  `0.0491`, and latent heat `0.0227` label MSE;
- DOC/litter: DOC vertical diffusion `0.1325`, OK_LEAK auxiliary `0.0552`, and
  free/adsorbed equilibration `0.0320` label MSE.

The GPP primitive itself is well fitted (`carbon.fast_gpp=0.000407`); the
canopy expert fails because its fast DIFFUCO endpoint label remains at
`0.0696`. Likewise, maintenance and thermal scientific outputs are no longer
the limiting mechanism.

## Decision

Do not open 64-point-year, full-epoch, rollout, paid, Gate D2, sealed-test, or
product stages. Do not merely repeat the same stochastic recipe and do not
weaken the expert gate after observing the result.

The active bounded task is a source/representation audit of the dominant
hydrology endpoints and canopy-debt coordinate. Determine which hydrology
diagnostics are exact one-boundary functions of predicted state, forcing, and
static parameters; determine the correct capacity/debt coordinate for canopy
water; and quantify the residual after those changes. Surface-energy and DOC
parameterization changes follow only after that audit. A revised neural
architecture must be discussed with the user before implementation.
