# DOC Diffusion Daily-Boundary Baseline Result

Date: 2026-08-17.

Status: completed non-test mechanism panel; endpoint-trapezoid physical
baseline accepted for a learned-residual output.

## Identity And Scope

- Git commit: `04b0c2ea637a5739c38a5c6e3b30418187393107`.
- Data: 20 accepted v6 point-years, comprising eight spatial-train points and
  two spatial-validation points in 1961 and temporal-validation year 2006.
- Report:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/doc-diffusion-boundary-baseline-04b0c2e/report-v1.json`.
- Report SHA256:
  `e200f2d770f52c9ee113d4b95b47bf94bbac72fbf17d1c5cc775a762f8b69cbd`.

The panel did not regenerate Teacher data, read v5 or a sealed test split,
materialize subdaily forcing, or execute a subdaily state scan. Each proxy
uses one existing source-backed diffusion-kernel call at each daily boundary.

## Correct Source Meaning

`ok_leak.doc_diffusive_interface` is not the signed net transfer through each
interface. The source kernel first computes a nonnegative magnitude from the
absolute adjacent-layer concentration difference. The direction is applied
later from the concentration ordering. The v6 label sums those nonnegative
magnitudes over 48 source-ordered half-hour calls.

Consequently, the label cannot be reproduced exactly by one daily kernel
call. It also cannot be converted into a reliable signed daily sum after the
fact because v6 does not retain the 48 concentration orderings.

The related `ok_leak.doc_advective_interface` label has the same separation
between magnitude and direction: its stored values are nonnegative transport
magnitudes, while the sign of `wat_flux` at each source call owns direction.
The old `daily signed interface sum` descriptions are therefore rejected for
both fields. This semantic correction changes no v6 values and requires no
data regeneration.

## Panel Result

Three daily-boundary proxies were compared with the cumulative Teacher label:

1. one source-kernel call on the day-start DOC and temperature states;
2. one call on the day-end states;
3. the arithmetic mean of the two boundary magnitudes.

Train/train selected the endpoint trapezoid before validation slices were
examined. It passed the preregistered gate in all four non-test slices:

| Slice | Correlation | NRMSE | Residual variance fraction |
| --- | ---: | ---: | ---: |
| Train/train | 0.9814 | 0.2312 | 0.0535 |
| Temporal validation | 0.9842 | 0.2078 | 0.0433 |
| Spatial validation | 0.9148 | 0.4519 | 0.2058 |
| Joint validation | 0.9573 | 0.2933 | 0.0869 |

Across all selected point-years, correlation is `0.9833`, NRMSE is `0.2119`,
and the residual variance fraction is `0.0450`. The boundary calculation thus
explains about 95.5% of the observed variance while leaving a measurable
residual.

## Decision

The revised field mechanism is:

```text
daily DOC diffusion magnitude
  = 0.5 * (day-start source-kernel magnitude
           + day-end source-kernel magnitude)
    + learned correction in a nonnegative baseline-centered coordinate
```

This is a strong physical baseline, not a retained exact process. The final
composition must preserve nonnegativity and handle genuine zero-activity
interfaces explicitly. At inference it uses two daily-boundary calls and no
48-step reconstruction or scan.

Because the Teacher label does not preserve cumulative direction, it remains
process supervision rather than a standalone signed stock-update instruction.
The conservative DOC endpoint update remains the stock owner.
