# Target-Oriented Flux Gradient-Path Result (2026-08-26)

## Decision

The frozen four-window, zero-training attribution passes its engineering gate
and classifies both extreme-window pairs as
`rollout_recurrence_dominant`. The extreme 7-day and 30-day maintenance-
respiration gradients are amplified primarily through the predicted
`hydrol_previous_step_state.snowheat` state carried into later days. They are
not explained by unusually large maintenance losses or by a uniformly
unstable shared Attention backbone.

This result does not reject the Attention architecture and does not admit
training. The next gate is a narrower zero-training causal intervention: keep
the same model and free rollout, but replace only the predicted snowheat slice
in each cross-day carry with the next day's Teacher day-start snowheat. The
intervention is diagnostic only and cannot enter training or final inference.

No optimizer or parameter step was taken. No checkpoint or recovery was
written. Teacher v6 was not regenerated, historical v5 was not used,
validation values and sealed test were not read, and no paid resource was
used. Unchanged training, architecture changes, 365-day rollout, Gate D2, and
product acceptance remain closed.

## Evidence Identity

- branch: `research/daily-coarse-graining`;
- executable commit:
  `53269afca0a03ce6658359956117c38b162f69b2`;
- report schema: `target_oriented_flux_gradient_path_attribution_report_v1`;
- remote report:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/target-oriented-flux-gradient-path-53269af/report.json`;
- local ignored copy:
  `outputs/diagnostics/target-oriented-flux-gradient-path-53269af/report.json`;
- report SHA256:
  `a2e9a88575563f832bf705827c869dcbe920d15abd656f06b0eb08068ffa870d`;
- gradient-path contract SHA256:
  `d683b87ae2658c4aeb5658cf42ab0b55f6e9cef94cd18af5419a376e3a821900`;
- update-1000 checkpoint SHA256:
  `a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7`;
- prepared-manifest SHA256:
  `8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31`.

The successful free-`gln01` run completed all `24/24` full gradients in
`2366.00 s`. Every gradient covers all `83,755,173` parameters, is finite and
nonzero, and all physical hard-failure counts are zero. JAX peak live device
memory was `2,203,604,224` bytes (`2.05 GiB`).

The first invocation compiled four new diagnostic graphs but exceeded the
outer 120-minute execution window; its persistent compilation cache was
retained. The first continuation encountered a CUDA graph instantiation OOM
after another user occupied about 25.2 GiB on GPU 0. Running on the empty GPU
1 completed. The measured model peak remained about 2.05 GiB; neither event
is scientific evidence about model memory demand.

## Gradient-Path Result

| Window | Maintenance free score | Free gradient | Teacher-state gradient | Free / Teacher-state | Extreme / matched control |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7-day extreme | 0.276417 | 186303.07 | 0.76610 | 243184.83 | 4196.11 |
| 7-day matched control | 0.358221 | 44.40 | 5127.44 | 0.00866 | - |
| 30-day extreme | 0.276487 | 430.96 | 8.7281 | 49.38 | 25.93 |
| 30-day matched control | 0.273844 | 16.62 | 950.59 | 0.01748 | - |

Maintenance respiration is aligned almost exactly with the reconstructed
aggregate gradient in both extreme windows. The 7-day extreme maintenance
gradient is nearly `4200` times its matched control, and the 30-day extreme
gradient is about `26` times its matched control. Removing cross-day predicted
state recurrence by evaluating each day from its Teacher state collapses the
extreme gradients by factors of about `243185` and `49.4`.

The controls prevent a broader but false interpretation. Their Teacher-state
maintenance gradients are larger, not smaller, than their free gradients.
Teacher-state evaluation therefore does not generically reduce loss or
gradient. The measured mechanism is a scenario-dependent recurrent path in
the extreme windows.

## Snowheat Attribution

The largest state-head contribution to both extreme maintenance gradients is:

```text
thermal.hydrol_previous_step_state.snowheat.normalized_tendency
```

Its free-gradient norm is `186301.54` in the 7-day extreme window and
`429.53` in the 30-day extreme window, accounting for nearly the entire
maintenance gradient. The next state-head norms are orders of magnitude
smaller.

In the 7-day extreme window, growth respiration also has a free-gradient norm
of `6390.02` dominated by the same snowheat head. It is almost exactly opposed
to the maintenance gradient (cosine `-0.999999997`), so the reconstructed
aggregate norm is `22489.30` rather than the maintenance norm scaled alone.
This cancellation does not remove the underlying recurrent amplification.

The state owner is the continuous Markov leaf
`hydrol_previous_step_state.snowheat`, sourced from
`src_sechiba/explicitsnow.f90 snowheat/snowtemp`. The next diagnostic must
resolve its exact `start:stop` coordinates from `DailyMarkovContract.state_leaves`;
no hard-coded flat coordinates are allowed.

## Next Gate

Freeze and execute `target_oriented_snowheat_recurrence_intervention_v1` on
the same four windows. Compare the unmodified free rollout with a rollout in
which only the snowheat coordinates of the cross-day continuous-state carry
are replaced by the next Teacher day-start values. Recompute maintenance
score, full gradient, snowheat-head attribution, and free-reference agreement.

If the extreme gradients collapse while the matched controls remain
interpretable, snowheat recurrence is confirmed as the causal path and a
state-Jacobian conditioning design can be discussed. If they do not, retain
the current architecture and inspect the next coupled snow thermal state.
The diagnostic replacement itself must never be proposed as training or
inference behavior.
