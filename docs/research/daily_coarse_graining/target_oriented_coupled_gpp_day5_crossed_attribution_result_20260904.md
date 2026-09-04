# Coupled GPP Day-5 Crossed Attribution Result (2026-09-04)

Status: accepted zero-training causal localization; the Day-5 jump is a direct
`structured_gpp` parameter-map effect, not a carried-state effect

Report SHA256:
`c014ac09558f30671bc511cfa9a63690f46f42434da428a414664581987780c5`

Executable commit: `1c7dc9b863e09718a001e54051374a6872795a8a`

## Execution And Integrity

The free-`gln01` GPU execution completed in `915.314 s` with status `passed`.
It reconstructed the historical update-3 `1/64` displacement from the same 26
nonzero sources and reproduced the accepted norm and parameter displacement.
All crossed complete-day states are finite, all water and carbon budget
residuals are zero, all factorization comparisons pass, and every fixed path
endpoint is exact. The discrete Day-5 start states are identical.

The first execution at commit `368cd3d` completed every expensive evaluation
but failed while constructing the report because it read `runtime_year` from
a nonexistent artifact attribute. Commit `1c7dc9b` reads the year from the
canonical schedule before direction reconstruction and adds a direct
regression test. This repair changes no model value, diagnostic intervention,
or acceptance threshold.

## Parameter-State Cross

| Cell | Parameters | Day-5 state | GPP | Potential |
| --- | --- | --- | ---: | ---: |
| `CC` | control | control | `2.7673363042` | `0.9912779331` |
| `TC` | treatment | control | `4.8698997235` | `1.7444095612` |
| `CT` | control | treatment | `2.7673363042` | `0.9912779331` |
| `TT` | treatment | treatment | `4.8698997235` | `1.7444095612` |

The free GPP jump is `+2.1025634193`. Holding the Day-5 state at control
retains exactly `100%` of the jump. The symmetric decomposition assigns
`+2.1025634193` (`100%`) to parameters, exactly zero to carried state, and
exactly zero to parameter-state interaction. The accepted classification is
`state_path_not_explanatory_for_day5_jump`.

The six state-owner groups do contain treatment-control differences: the
largest normalized changes are about `0.0041734` in canopy and water state.
Nevertheless, replacing or transplanting any complete GPP-admitted owner
leaves Day-5 GPP unchanged at report precision. No state owner is selected.

## Parameter-Family Attribution

The result is identical at control and treatment Day-5 state:

| Treatment family inserted alone | GPP | Effect | Sufficiency | Necessity |
| --- | ---: | ---: | ---: | ---: |
| `structured_gpp` | `4.8698997235` | `+2.1025634193` | `1.0` | `1.0` |
| `coupled_gpp_carbon_feedback` | `2.7673363042` | `0` | `0` | `0` |

The two-family interaction is exactly zero. The newer carbon-feedback adapter
and all cross-day state paths are therefore excluded as explanations for this
specific jump. The effect lies inside the existing `12,879,363`-parameter
`structured_gpp` specialist and, within its output factorization, almost
entirely inside the potential branch.

## Bounded Path Result

Both parameter paths are identical because the Day-5 state has no measured
effect:

| Fraction of endpoint parameter displacement | GPP | Potential |
| ---: | ---: | ---: |
| `0` | `2.7673363042` | `0.9912779331` |
| `1/8` | `2.7673096070` | `0.9912669063` |
| `1/4` | `2.7672823063` | `0.9912555814` |
| `1/2` | `2.7672266213` | `0.9912328720` |
| `3/4` | `4.8699009872` | `1.7444151640` |
| `1` | `4.8698997235` | `1.7444095612` |

State interpolation is exactly constant at `2.7673363042` under control
parameters and `4.8698997235` under treatment parameters. Parameter
interpolation instead changes only slightly through `1/2`, then jumps between
`1/2` and `3/4`.

The implemented factor path uses fixed masks with linear layers, softmax
attention, GELU, LayerNorm, and softplus; this audit found no explicit
parameter-dependent hard branch. The sharp path is therefore evidence of a
numerically ill-conditioned or float32-quantized parameter response, but this
run does not distinguish those explanations or identify the responsible
parameter subtree.

## Decision And Next Gate

The parameter/state crossed attribution is accepted and closed. It rules out
further state-owner and snowheat interventions for this Day-5 jump. It also
shows that continuing training without resolving the `structured_gpp`
potential jump would risk accepting another locally improving update that
changes one window catastrophically.

The next action requires user discussion before any architecture change. The
recommended bounded follow-up is a zero-training precision and parameter-
subtree attribution of `structured_gpp`: measure changed float32 coordinates
and ULPs by subtree, transplant its existing top-level subtrees one at a time,
and compare endpoint interpolation with a higher-precision diagnostic path.
That would decide whether the repair belongs in numerical precision/update
realization or in the specialist architecture. It must not train, retain a
parameter update, read fresh validation or sealed test, or modify the daily
operator boundary.

No optimizer, checkpoint, recovery, Teacher regeneration, historical v5 read,
fresh-validation read, sealed-test read, half-hour forcing reconstruction,
48-step scan, architecture change, or paid execution occurred.
