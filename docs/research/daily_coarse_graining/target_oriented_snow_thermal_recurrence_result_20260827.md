# Target-Oriented Snow Thermal Recurrence Result

Date: 2026-08-27

Status: accepted zero-training diagnostic; mixed causal classification

Report SHA256:
`c4b80543a3a726819af3d6e658603484b7aa233cb485333e689af47cfb303810`

Contract SHA256:
`959413d639a6bd853b15ca2fc896beea4ea5e690463f4a78e8c3cac1ccaf96ca`

Git commit: `975ae9b66021d763bf83be9cbc343edc31739c81`

## Result

Both complete 30-day maintenance gradients are finite over all `83,755,173`
parameters. All physical hard-failure counts are zero. No optimizer step,
checkpoint, Teacher regeneration, historical-v5 read, validation-value read,
sealed-test read, or architecture change occurred.

For the extreme window, adding Teacher snowtemp carry to the accepted
snowheat-only parent changes the full gradient norm from `276.37034` to
`52.87295`, a `5.2271`-fold reduction. The snowtemp output-head norm falls
from `272.76550` to `0.55118`, a `494.88`-fold reduction. This confirms that
snowtemp was a real part of the 30-day recurrent path, but the frozen tenfold
full-gradient threshold is not met, so the classification is
`coupled_snow_thermal_recurrence_mixed`.

The remaining extreme gradient is led by the snowheat output head at
`40.70881` and the direct maintenance head at `33.42881`. The next thermal
heads are much smaller: `temp_sol=2.40996` and `ptn=2.29640`.

For the matched control, the full gradient increases from `16.28446` to
`31.69702`, while its snowtemp head falls from `4.50371` to `0.16063`. This
reinforces that Teacher carry is a causal probe, not a generally beneficial
rollout mode.

Peak active JAX memory was `2,203,643,392` bytes (about `2.05 GiB`). Total
runtime was `2151.32 s`.

## Post-Result Mechanism Audit

The v6 Markov state contains both canonical `hydrol` snowheat/snowtemp leaves
and `diffuco` copies. The fast-owner contract declares each `diffuco` leaf a
mirror of its matching `hydrol` owner. The completed intervention replaced
only the two `hydrol` slices; it therefore left both mirror slices on the
predicted cross-day carry.

The persistent snowheat-head gradient cannot yet be called a same-day path.
The next bounded gate is the four-slice mirror-complete intervention frozen in
`target_oriented_snow_thermal_alias_recurrence_intervention_v1.md`. Attention
is retained. Training and architecture changes remain closed pending that
result.
