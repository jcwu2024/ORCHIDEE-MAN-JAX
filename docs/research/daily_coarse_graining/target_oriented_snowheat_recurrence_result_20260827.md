# Target-Oriented Snowheat Recurrence Result (2026-08-27)

## Decision

The frozen snowheat-only intervention passes its engineering gate and returns
`snowheat_recurrence_partial`. It causally confirms the snowheat cross-day
path for the 7-day extreme window: the maintenance gradient falls by a factor
of `10416.39`, and the snowheat-head gradient falls by `622940.95`.

The 30-day extreme window is not explained by snowheat alone. Its snowheat-
head gradient falls by `78.96`, but the full maintenance gradient falls only
by `1.56`. After the intervention,
`thermal.hydrol_previous_step_state.snowtemp.normalized_tendency` has gradient
norm `272.77` and dominates the remaining full norm `276.37`. This opens one
minimal zero-training coupled intervention on snowheat plus snowtemp for the
two frozen 30-day windows.

The Attention architecture is not rejected and training is not admitted. The
Teacher-state replacement is diagnostic only and cannot enter training or
final inference.

## Evidence Identity

- branch: `research/daily-coarse-graining`;
- executable commit:
  `159f67205d955005f5d988e5ace8ea6de28fc19c`;
- report schema:
  `target_oriented_snowheat_recurrence_intervention_report_v1`;
- remote report:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/target-oriented-snowheat-recurrence-159f672/report.json`;
- local ignored copy:
  `outputs/diagnostics/target-oriented-snowheat-recurrence-159f672/report.json`;
- report SHA256:
  `e2a6934bf05027243dc2df046a194e18821dcc1fb6804fec2126208f98a05e0f`;
- intervention contract SHA256:
  `006e9d3d8ff63253ea2e70c995735856e9c3e93285934934fbbe3f7b16cab122`;
- update-1000 checkpoint SHA256:
  `a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7`;
- prepared-manifest SHA256:
  `8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31`.

The successful free-`gln01` run completed all `8/8` full gradients in
`4624.28 s`. Every gradient covers all `83,755,173` parameters, is finite and
nonzero, free-reference scores reproduce exactly, and all physical hard-
failure counts are zero. JAX peak live device memory was `2,204,376,320`
bytes (`2.05 GiB`).

An earlier invocation at commit `cff2f29` was manually stopped before any
report because it separately compiled intervention forward and reverse graphs.
Commit `159f672` returns the hard counts as `value_and_grad` auxiliary output,
reducing the new graph count from four to two. No scientific value or gate was
changed.

## Window Results

| Window | Free score | Snowheat-carry score | Free gradient | Snowheat-carry gradient | Reduction | Mode cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7-day extreme | 0.276417 | 0.272807 | 186303.07 | 17.89 | 10416.39 | -0.0166 |
| 7-day control | 0.358221 | 0.352932 | 44.40 | 45.40 | 0.978 | 0.8641 |
| 30-day extreme | 0.276487 | 0.276611 | 430.96 | 276.37 | 1.559 | 0.0320 |
| 30-day control | 0.273844 | 0.273841 | 16.62 | 16.28 | 1.020 | 0.9173 |

The intervention barely changes the maintenance scores. It changes the
extreme gradient paths, especially at 7 days, while leaving the controls near
their original gradient magnitudes and directions. This supports a causal
path interpretation rather than a simple loss-rescaling effect.

For the 7-day extreme, the snowheat-head norm changes from `186301.54` to
`0.299`; the remaining gradient is dominated by the direct maintenance output
head (`17.79`). For the 30-day extreme, the snowheat-head norm changes from
`429.53` to `5.44`, but snowtemp rises to the dominant remaining state-head
norm `272.77`; the direct maintenance head is only `34.79`.

The exact v6 state coordinates were resolved by key at runtime:

```text
hydrol_previous_step_state.snowheat: [510:513], shape [1, 3]
```

No flat coordinate was hard-coded in the intervention.

## Next Gate

Freeze `target_oriented_snow_thermal_recurrence_intervention_v1`. Use only the
30-day extreme and matched-control windows. Keep the accepted snowheat Teacher
carry and additionally replace the `hydrol_previous_step_state.snowtemp`
coordinates from each current Teacher day-start state. Compute exactly two
maintenance gradients and compare them with the snowheat-only parent report.

If the extreme full gradient and snowtemp-head gradient both fall by at least
a factor of ten while the control remains finite and physically valid, the
coupled snow thermal recurrence is causally confirmed. The next action would
then be an architecture/training-design discussion about bounded state-
Jacobian conditioning, not another automatic network edit.
