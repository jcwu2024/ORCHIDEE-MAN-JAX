# Target-Oriented Snow Thermal Alias Recurrence Result

Date: 2026-08-27

Status: accepted zero-training diagnostic; mixed causal classification

Report SHA256:
`5714d30db9a905498a1e31203f7d6a622417fc165328ac47bf93e077e4422168`

Contract SHA256:
`5ab495baf5c140d71e841defaa6e799d8c6419717944e6aa38ab37b5ecd77f63`

Git commit: `51bb538d99db15ce15d0aeb803db509a24f03c69`

## Result

Both complete 30-day maintenance gradients are finite over all `83,755,173`
parameters. All four state slices and both canonical/mirror owner relations
resolve, and every physical hard-failure count is zero. No optimizer step,
checkpoint, Teacher regeneration, historical-v5 read, validation-value read,
sealed-test read, or architecture change occurred.

For the extreme window, completing the causal cut through both `diffuco`
mirrors changes the full gradient norm from `52.87295` to `33.69151`, a
`1.5693`-fold reduction. The remaining snowheat-head norm is exactly zero,
compared with `40.70881` in the parent. This proves that the previous residual
snowheat path was carried through the omitted mirror. The frozen twofold full-
gradient threshold is not met because the direct maintenance head remains
`33.46683`; the preregistered classification is
`snow_thermal_alias_path_mixed`.

For the matched control, the full gradient falls from `31.69702` to
`15.67716`, a `2.0219`-fold reduction. Its snowheat head also becomes exactly
zero, while the direct maintenance head remains `15.52365`. The controls were
reported but were not a classification threshold.

Peak active JAX memory was `2,203,553,024` bytes (about `2.05 GiB`). Total
runtime was `2110.56 s`, including the cached-runtime setup and two complete
gradients.

## Source Audit Of The Remaining Path

The next-state heads are now small (`ptn=1.76022`, `temp_sol=1.58936`, and
`tsoil=0.24547` in the extreme window). The residual is therefore no longer a
snow-thermal state-head problem; it is concentrated in the learned
`daily.fast_maintenance_respiration` head.

Fortran establishes two different state semantics:

- `stomate_resp.f90::maint_respiration` lines 122-151 computes current-step
  maintenance from temperature, vegetation structure, root profile, biomass,
  SLA, and maintenance coefficients. It does not consume prior
  `resp_maint` or `resp_maint_part`.
- `stomate.f90::stomate_main` lines 3244-3267 accumulates current-day
  `resp_maint_part`; lines 4297-4315 pass that current-day accumulation to
  `StomateLpj`; and lines 4943-4958 reset it to zero at day end.
- `stomate_npp.f90::npp_calc` lines 279-320 overwrites `resp_maint` and derives
  it from the current-day `resp_maint_part`; line 530 consumes it in current-
  day NPP. It is not a driver of the next maintenance calculation.

The canonical JAX path preserves those semantics:
`jax_orchidee/driver/orchestration.py::_paper_day_daily_reset_fields` resets
`resp_maint_part`, and `_paper_day_first_stomate_state_packet` writes the
current-day `npp_update.resp_maint` into the next Markov state. Consequently,
`resp_maint_part` is a reset accumulator and is not a plausible cross-day
feedback state. `resp_maint` is a persisted diagnostic/output that is
overwritten on the next daily NPP call.

The neural source router is broader than the physical owner. The maintenance
expert admits the complete `state_carbon` family, and the strong-baseline
context constructs one token for every continuous state leaf. Therefore the
persisted previous-day `resp_maint` token can reach the next day's maintenance
Attention expert even though the Fortran maintenance owner does not read it.

## Next Gate

Execute one bounded zero-training `resp_maint` recurrence intervention on the
same extreme and matched-control 30-day windows. Preserve the accepted four-
slice snow-thermal alias Teacher carry, additionally replace only
`slowproc_stomate_previous_step_state.resp_maint` from each current Teacher
day-start state, and recompute the two complete maintenance gradients.

Do not carry `resp_maint_part`: its source-exact day-start value is already the
post-reset zero accumulator. The new intervention is diagnostic only and
cannot enter training or final inference. Any subsequent change to the
maintenance input mask or source routing requires architecture discussion
before implementation.
