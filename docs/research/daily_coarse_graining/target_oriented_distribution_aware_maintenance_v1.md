# Target-Oriented Distribution-Aware Maintenance V1

Date: 2026-08-28

Status: local implementation and real-v6 GPU admission accepted

Machine contract:
`manifests/coarse_graining/target_oriented_distribution_aware_maintenance_v1.json`

Contract SHA256: `f167943cfb80902fd968259f7f2a1a852d783b151a7eedd16b3157abd1a37d2d`

## Decision

Retain the accepted update-1000 coordinate-aligned Attention model and every
non-maintenance output. Replace only the direct maintenance projection with a
maintenance-specific Attention adapter that preserves all 24 PFT/part
coordinates and can read the broader canonical daily context.

This is not an Attention-versus-MLP comparison. The four-owner bridge proves
that `biomass`, `lai`, `veget_max`, and `t2m_longterm` repair part of biomass
recurrence but only `4/10` protected cells. The 7/30-day growth-respiration and
NPP reversals require information beyond those four owners. Attention remains
the selected context mechanism.

## Candidate Mechanism

The parent already produces one `768`-wide hidden vector for each of the 24
maintenance coordinates. The rejected predecessor averaged those vectors into
aboveground and belowground groups and returned only two corrections. V1 keeps
all 24 queries separate.

Each maintenance coordinate attends to a context bank containing:

- every canonical source token admitted by the existing input assembler;
- every compact PFT token;
- all 24 maintenance-coordinate hidden states.

One shared `384 -> 768` projection aligns source/PFT tokens with the
maintenance hidden width. Two residual Attention blocks use width `768`, eight
heads, and FFN width `1536`. A shared `768 -> 1` projection returns one bounded
temperature-multiplier correction for every PFT/part coordinate. There is no
coordinate pooling before output.

The adapter contains `9,750,529` parameters. Removing the old 769-parameter
maintenance projection gives a complete candidate of `93,504,933` parameters.
The first training stage may update only the adapter. The parent Attention
encoder and every non-maintenance head remain exact checkpoint values.

The final kernel uses the accepted small `1e-3` initialization and zero bias,
so gradients reach both adapter blocks from the first update while the initial
correction remains near the physical daily baseline.

## Output And Daily Binding

The network output is

```text
delta_temperature_multiplier[pft, plant_part] in [-1, 1]
```

The existing source-backed materializer combines it with day-start biomass,
LAI, exact `coeff_maint_zero`, named `maint_resp_slope` parameters, daily air
temperature, and root temperature. It produces the existing
`daily.fast_maintenance_respiration[1, n_pft, 12]` process field. The
conservative updater and retained exact daily tail remain unchanged through
the canonical `bind_daily_operator(...)` boundary.

Fortran provenance remains
`stomate_resp.f90::maint_respiration:234-375`; the accepted JAX source kernel
is `_maintenance_respiration_core:2470-2529`. The candidate introduces no
cross-day neural hidden state, 48-record forcing reconstruction, or 48-step
scan.

## Training Objective

The candidate must be trained against both distributions that exposed the
predecessor failure:

- the existing one-day/process/state plus free 7/30-day science objective;
- a Teacher-day-start 7/30-day guard over maintenance, growth respiration, and
  NPP.

Before an optimizer is created, the Teacher-guard coefficient is frozen from
the existing eight train/train calibration batches as the median free-gradient
norm divided by the median Teacher-guard gradient norm, clipped to `[0.1,
10.0]`. The exact resulting coefficient requires a separate post-admission
training contract. It cannot be tuned from validation or rollout results.

## Matched Reference And Gates

The unchanged update-1000 checkpoint remains a diagnostic reference, not a
baseline whose small defeat implies success. Promotion requires the absolute
target-oriented 7/30-day gates, at least 20% median maintenance improvement at
both horizons, nonpositive median growth/NPP/AGB/BGB/total-live regression,
and zero physical hard failures.

Local acceptance covers parameter ownership, all 24 addressable corrections,
forward/loss/gradient, exact non-maintenance output preservation, inactive PFT
isolation, PFT permutation equivariance, and one canonical state/budget
transition. The accepted zero-update real-v6 admission covers the complete
forward/loss and all `9,750,529` adapter-gradient coordinates.

This contract authorizes no optimizer update, paid job, validation read,
checkpoint, 365-day rollout, Gate D2, sealed test, or product work.

## Local Acceptance

The implementation preserves the existing Attention parent and replaces only
the 769-parameter maintenance projection. The adapter parameter count is
exactly `9,750,529`, and the complete candidate count is exactly
`93,504,933`.

The focused local candidate and predecessor suites cover 11 unique tests. All
pass after JIT-safe extraction of the shared physical materializer. The new
candidate passes eager/JIT forward, complete masked loss, finite adapter
gradients with nonzero context-projection, Attention, FFN, and final-projection
components, all 24 separately retained corrections, exact equality of all 160
non-maintenance parent outputs, bare/inactive zero behavior, PFT permutation
equivariance, canonical daily binding, and water/carbon residuals within
`2e-12`. Its inference source contains no half-hour reconstruction or scan.

The zero-update real-v6 admission passed on free `gln01` at commit `54ada54`;
the report SHA256 is `376d8016...c66ece5`. All `9,750,529` adapter-gradient
elements are finite, every adapter leaf is nonzero, all 160 non-maintenance
outputs exactly equal the parent, and canonical water/carbon residuals are
zero. The accepted evidence is in
`target_oriented_distribution_aware_maintenance_gpu_admission_20260828.md`.

The active gate is a separate post-admission training contract and frozen
Teacher-guard coefficient. No optimizer or paid training may be created before
that calibration is accepted.
