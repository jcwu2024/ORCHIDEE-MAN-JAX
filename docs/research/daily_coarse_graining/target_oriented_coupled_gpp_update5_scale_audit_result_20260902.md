# Coupled GPP Update-5 Scale Audit Result

## Outcome

The free `gln01` audit completed at commit `25ec5bc` in `2,696.660`
seconds. Four bounded scales pass the frozen four-source and physical gate:
`1/4`, `1/8`, `1/16`, and `1/32`. The classification is
`bounded_direct_finite_step`.

The JSON report is
`outputs/research/daily_coarse_graining/target-oriented-coupled-gpp-update5-scale-audit-25ec5bc.json`,
SHA256 `2199a0b7e8c3bad7b8990e9a06e2d13f361c7fec4acf7f37e78bb4befeef5081`.

This establishes that update 5 does not require an architecture change. Its
recalibrated common direction has a finite trust region; the previously used
full displacement was simply outside that region.

## Finite-Scale Results

All values below are relative loss changes from the same post-update-4 state.
Negative values are improvements.

| Scale | GPP | State | 7-day | 30-day | Weighted | Accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `1` | `+58.1508%` | `-28.6954%` | `-9.9005%` | `-10.6082%` | `+9.1685%` | no |
| `1/2` | `+9.3058%` | `-18.9490%` | `-4.9128%` | `-5.3225%` | `-2.4318%` | no |
| `1/4` | `-0.3069%` | `-10.4394%` | `-2.5014%` | `-2.6021%` | `-2.9544%` | yes |
| `1/8` | `-1.3342%` | `-5.4276%` | `-1.3876%` | `-1.4295%` | `-1.9587%` | yes |
| `1/16` | `-0.9540%` | `-2.7681%` | `-0.6948%` | `-0.4812%` | `-1.0334%` | yes |
| `1/32` | `-0.5476%` | `-1.3969%` | `-0.3467%` | `-0.4101%` | `-0.5748%` | yes |

The maximum accepted scale is `1/4`, with parameter displacement norm
`0.00193974331`. Scale `1/2` lowers the weighted objective but is rejected
because direct GPP worsens by `9.3058%`; the aggregate score is not allowed to
hide regression in a protected long-horizon driver.

The realized transition agrees with the prior derivative-based diagnostic:
the diagnostic predicted a GPP zero crossing near scale `0.253654`, and the
actual `1/4` step is the first tested scale with non-worsening GPP. The
diagnostic estimate did not participate in acceptance.

## Integrity And Cost

Every scale passes the physical/nonfinite hard gate. There are no budget,
unexpected-mask, discrete, nonfinite, or negative-stock failures. The parent
recovery, source checkpoint, predecessor result, and `93,504,164` fixed
parameters remain unchanged. No proposal or checkpoint is retained.

No fresh validation or sealed-test value is read; no Teacher data is
regenerated; no v5 data, 48-record forcing reconstruction, or subdaily state
scan is used. Reported JAX allocator peak is `4,098,246,656` bytes. The run
used free `gln01`, so there is no Slurm charge.

## Next Gate

Freeze a no-checkpoint scale-selecting direct-recalibration smoke. At every
scheduled update it must recompute the four-source common direction, evaluate
the frozen scales from largest to smallest, and apply only the largest scale
that passes every source and physical gate. It must stop when no tested scale
passes and must evaluate the established non-test 7/30-day transfer before
any paid continuation is considered.

Architecture changes, paid training, fresh validation, 365-day rollout, Gate
D2, sealed test, and product acceptance remain closed.
