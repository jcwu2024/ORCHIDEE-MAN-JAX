# Coupled GPP Potential Internal Attribution v1 Result (2026-09-05)

Status: engineering-inconclusive for component attribution; the isolated
potential-block jump moved one interpolation interval and requires the frozen
v2 rerun

Report SHA256:
`faf1e4c9a1220a4139372fe59765080cd3979e1fcbfff8395777e3d78d4e5f64`

Executable commit: `aae98e34b9cfba02f9ae37ce22ffeeb4a633e8b0`

## Execution And Engineering Gate

The zero-training free-`gln01` run completed on a V100 in `937.716` seconds
and wrote
`outputs/diagnostics/target-oriented-coupled-gpp-potential-internal-aae98e3.json`.
Direction reconstruction, control-trajectory reproduction, exact component
coverage, float32 shadow reproduction, finite factorization, float64 dtype,
and all finite-value gates pass. The single failed gate is
`isolated_potential_path_reproduced`.

The first launch at commit `102a056` reconstructed the historical direction
but then exposed a report-schema compatibility bug: the child report uses
`reconstructed_treatment_parameter_delta_l2`, while the reused helper only
accepted the predecessor's `reconstructed_parameter_delta_l2`. It wrote no
JSON or state. Commit `aae98e3` accepts both real schemas and adds regression
coverage.

## Failed Frozen Assumption

Both isolated endpoints reproduce exactly, but the maximum adjacent interval
does not:

| Gate | Frozen v1 value | Observed value |
| --- | --- | --- |
| Control endpoint GPP | `2.7673363042` | `2.7673363042` |
| Potential-only endpoint GPP | `4.8698992466` | `4.8698992466` |
| Maximum adjacent interval | `10/16 -> 11/16` | `11/16 -> 12/16` |

The relevant three-point comparison is:

| Fraction | Whole structured tree | Isolated potential blocks | Isolated full-float64 potential |
| ---: | ---: | ---: | ---: |
| `10/16` | `2.7671984108` | `2.7673231588` | `2.7673233050` |
| `11/16` | `4.8699010524` | `2.7673221604` | `2.7673220160` |
| `12/16` | `4.8699009872` | `4.8699012434` | `4.8699009008` |

The accepted parent established endpoint sufficiency and necessity of
`potential_blocks`; it did not establish that interpolating all other small
structured-GPP parameter changes would leave the threshold location fixed.
Those two claims do not commute. The v1 contract incorrectly treated the
whole-tree `10/16 -> 11/16` interval as the isolated-subtree interval.

## Bounded Evidence And Invalid Attribution

The production-equivalent float32 shadow reproduces the isolated path. The
full-float64 potential branch remains sharp and moves over the same observed
`11/16 -> 12/16` interval. Its endpoint jump is `2.1025628812`, versus
`2.1025629425` in production float32, a magnitude ratio of
`0.9999999709`. This supports intrinsic potential-block conditioning rather
than intermediate float32 precision as the sampled mechanism, subject to the
v2 engineering pass.

The v1 block/component replacements used `10/16` and `11/16`, whose isolated
GPP difference is only `-9.9838456e-7`. Their reported materiality ratios do
not explain the `+2.1025791` main jump and must not be used to select a block
or component.

## Decision And Next Gate

The v1 component attribution is closed as engineering-inconclusive. The next
gate is the frozen v2 zero-training rerun over the observed isolated interval
`11/16 -> 12/16`, retaining the same sample, two block controls, 16 component
controls, float32/full-float64 paths, and acceptance thresholds.

No optimizer, parameter update, checkpoint, recovery, architecture change,
Teacher regeneration, historical v5 read, validation read, sealed-test read,
half-hour forcing reconstruction, 48-step scan, paid execution, 365-day
rollout, Gate D2, or product acceptance occurred.
