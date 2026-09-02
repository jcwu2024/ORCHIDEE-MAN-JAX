# Coupled GPP Update-3 Common-Direction Audit Result

## Outcome

The free `gln01` audit completed at commit `e704749` in `4,521.701` seconds.
The classification is `direct_common_finite_step`, so the frozen scientific
gate passes. The JSON report is
`outputs/research/daily_coarse_graining/target-oriented-coupled-gpp-update3-common-direction-e704749.json`,
SHA256 `642086afe65c08909b162153cf08ab8ca72bf31f5041b47e422592165b95ec8d`.

This is positive evidence for the existing coupled-GPP Attention candidate.
The failed fixed-weight update 3 was not a proof that the network lacked a
learnable direction. The same four source gradients contain a broad finite
common-descent step when their direction is solved explicitly.

## First-Order Geometry

The minimum-norm simplex direction has positive cosine with every source:

| Source | Cosine | Normalized simplex weight |
| --- | ---: | ---: |
| Direct GPP | `0.226096` | `0.381487` |
| Reachable one-day state | `0.226096` | `0.246061` |
| 7-day science | `0.226096` | `0.372452` |
| 30-day science | `0.248111` | `0.0` |

The zero 30-day simplex weight does not remove that objective. Its gradient is
strongly aligned with the supported 7-day gradient (cosine `0.817141`), so the
minimum-norm combination already descends the 30-day loss. By contrast, the
direct-GPP gradient conflicts strongly with the 7-day and 30-day gradients
(cosines `-0.695864` and `-0.519276`). Fixed scalar weights cannot be expected
to remain a common direction as scheduled groups change.

## Finite Steps

All five direct common-direction scales pass. Relative loss changes are shown
below; negative values are improvements.

| Scale | GPP | State | 7-day | 30-day | Weighted | Accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `1` | `-0.9890%` | `-6.5908%` | `-3.7807%` | `-6.5117%` | `-3.6710%` | yes |
| `1/2` | `-1.5701%` | `-3.5801%` | `-1.8937%` | `-3.3563%` | `-2.3439%` | yes |
| `1/4` | `-1.0538%` | `-1.8670%` | `-0.9801%` | `-1.5270%` | `-1.2817%` | yes |
| `1/8` | `-0.5935%` | `-0.9524%` | `-0.4887%` | `-0.4918%` | `-0.6309%` | yes |
| `1/16` | `-0.3133%` | `-0.4812%` | `-0.2449%` | `-0.1982%` | `-0.3162%` | yes |

The full direct displacement norm is `0.00775897335105`. The complete
accepted scale interval is substantially stronger evidence than the earlier
isolated reset-AdamW result.

The fresh-AdamW realization retains positive first-order cosines, but rotates
them down to `0.02949`-`0.08313`. Its accepted scales are `1`, `1/8`, and
`1/16`; `1/2` and `1/4` fail because 7-day loss rises `0.06146%` and
`0.22094%`. This nonmonotonic response rejects fresh AdamW as the primary
realization of the common direction. The direct normalized step is the next
method to test.

## Integrity

All one-day, 7-day, and 30-day physical hard gates pass at every evaluated
scale. All `15,960,195` trainable values are finite and all 190 trainable
leaves move in both realizations. The `93,504,164` fixed parameters, source
checkpoint, and parent recovery remain unchanged. No checkpoint or proposal
is retained; no validation or sealed-test value is read; no Teacher data is
regenerated; no v5 data, 48-record forcing reconstruction, or subdaily state
scan is used.

Reported device memory peaks at `4,329,502,720` bytes in the JAX memory
profile. The external allocator reserved approximately 28.9 GiB during the
run; that reservation is not the same as live model memory.

## Next Gate

Freeze and locally qualify a separate three-to-five-update free-`gln01`
smoke. Starting from the exact update-1 recovery, it must reconstruct the
accepted update 2, then recompute the four gradients and the minimum-norm
common direction after every subsequent scheduled macro update. It must use
the direct realization, apply a proposal only when all four finite-step and
physical gates pass, and retain no checkpoint.

This result does not authorize paid continuation, fresh validation, a
365-day rollout, Gate D2, sealed-test access, or product acceptance.
