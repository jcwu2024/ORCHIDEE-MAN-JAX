# Strong-Baseline One-Point Loss-Floor Diagnostic Result

Date: 2026-08-17.

Status: completed; qualification failed; 64-point-year and paid stages remain
closed.

## Identity

- Git commit: `04b0c2ea637a5739c38a5c6e3b30418187393107`.
- Data: accepted `pft14-daily-teacher-v6`, landpoint `001.0-071.0`, year
  `1961`, split `train/train`, 364 valid days.
- Architecture: `axis_aligned_process_expert_operator_v1`, 49,750,681
  parameters, fresh seed `20260815`, graph path disabled.
- Formal report:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/strong-baseline-one-point-loss-floor-04b0c2e/report.json`.
- Report SHA256:
  `2f3383540ef94e5a8e17b982dccab35fb67c84147cb8e18bd982945697ad5661`.
- Diagnostic-only parameter state SHA256:
  `e711cdfe963b0a3bd062bf71a5dee247faca918bff897f4127c76d12c2e115ec`.

The state is about 190 MiB, contains no optimizer state, and is marked
`diagnostic_only=true`, `formal_training_initialization_allowed=false`, and
`promotion_checkpoint=false`.

## Execution

The short real-v6 preflight first passed one epoch and one exact-gradient tail
update, including detailed loss reporting and state serialization. The formal
run then completed:

- 100 complete shuffled epochs and 4,600 finite AdamW updates;
- 25 exact all-364-day updates, each accumulated from 46 size-8
  microbatches;
- exact coverage and unchanged expert/label/field counts on every evaluation;
- zero nonfinite gradients or parameters;
- 2,935.50 seconds elapsed;
- 8,630,748,416-byte JAX peak allocation, about 8.04 GiB.

No Teacher data were generated, no v5 data were read, and no 48-record forcing
reconstruction or 48-step state scan was executed.

## Loss Result

| Stage | Total process loss | Reduction from initialization |
| --- | ---: | ---: |
| Fresh initialization | 4.2262827073 | 0% |
| End of 100 minibatch epochs | 0.1978143451 | 95.3194% |
| End of 25 exact-gradient updates | 0.1912643210 | 95.4744% |

The exact tail reduced its starting loss by only 3.3112%, adding 0.1550
percentage points to the initialization-relative reduction. Its median exact
gradient norm was `0.69373`; the median ratio between the exact gradient norm
and the real-sample-weighted mean microbatch norm was `0.53937`. Batch
gradients therefore cancel materially, but removing that stochastic sampling
did not approach the required 99% reduction.

Final expert reductions were:

| Expert | Reduction | Final loss | Gate |
| --- | ---: | ---: | --- |
| canopy GPP | 31.05% | 0.638255 | fail |
| litter/POC/DOC | 53.13% | 0.069689 | fail |
| hydrology/snow | 75.68% | 0.083872 | fail |
| soil thermal | 91.71% | 0.099864 | fail |
| surface energy | 96.70% | 0.016536 | pass |
| daily interface/finalize | 98.33% | 0.430505 | pass |
| maintenance respiration | 99.98% | 0.000130 | pass |

## Concentrated Floor

The residual is highly concentrated rather than uniformly distributed. The
five largest label contributions account for 91.06% of the final total loss:

| Label | Final label loss | Share of total process loss |
| --- | ---: | ---: |
| `fast_owner.diffuco` | 1.274 | 47.59% |
| `fast_owner.driver` | 1.149 | 28.59% |
| `fast_owner.thermosoil` | 0.399 | 7.41% |
| `carbon.doc_vertical_diffusion` | 0.706 | 4.09% |
| `water.canopy_inventory_endpoints` | 0.815 | 3.38% |

The two largest labels alone account for 76.18%. Their principal field floors
are `diffuco_previous_step_state.z0m` (`38.77` normalized MSE), `evapot`
(`3.36`), `evapot_corr` (`2.25`), `drysoil_frac` (`1.43`),
`temp_sol_pft` (`1.18`), `driver_previous_step_state.fluxsens` (`3.07`), and
`vevapp` (`0.83`). The strongest soil-thermal floors are `soilcap` (`1.37`),
`dgrnd_snow` (`1.33`), and `cgrnd_snow` (`1.10`).

Several of these are derived endpoint or coefficient fields with explicit
source owners: `z0m` is owned by `condveg_roughness`, while `soilcap`,
`dgrnd_snow`, `cgrnd_snow`, `cgrnd`, `dgrnd`, and `soilflx` are owned by
`thermosoil_coef`. This is direct evidence to audit whether the daily operator
should reconstruct them consistently from primitive predicted state instead
of asking independent neural heads to reproduce each derived endpoint.

## Corrected Science Metrics

The accepted train-only one-day AGB/BGB tendency scales make the biomass gate
discriminating: random-initialization NRMSE was `0.66162/0.71647`, not the
near-zero values produced by the old stock normalization.

| Metric | Final NRMSE | Gate `<=0.05` |
| --- | ---: | --- |
| maintenance respiration | 0.006826 | pass |
| AGB tendency | 0.025472 | pass |
| GPP | 0.048135 | pass |
| BGB tendency | 0.050637 | fail |
| NPP | 0.056944 | fail |

All six nonfinite, water/carbon budget, and negative-inventory hard counts are
zero.

## Reproducibility Observation

The earlier run at commit `4b849fb` reached `97.6026%` reduction, whereas the
new minibatch stage reached `95.3194%`. Initialization and epoch-1 behavior
match, then the trajectories separate by epoch 5-10. No architecture, base
loss, batch order, optimizer schedule, or seed changed between those paths;
the new code adds read-only detailed evaluations before training. This is
consistent with sensitivity to GPU numerical/parallel execution, but one pair
does not prove the exact source. It does prove that a single stochastic
one-point outcome is not stable enough to define capability.

The exact-gradient tail did not rescue the poorer trajectory, so rerunning the
same stochastic recipe is not authorized as the next step.

## Decision And Next Gate

The one-point qualification remains failed. Do not open 64-point-year
training, paid compute, 7/30/365-day rollout, long-held points, Gate D2, sealed
test, or product acceptance.

The next bounded task is a source/representation audit of the dominant five
labels, beginning with the `fast_owner.diffuco`, `fast_owner.driver`, and
`fast_owner.thermosoil` endpoint fields. For every high-loss field, determine
whether it is:

1. exactly reconstructable once per day from admitted primitive outputs and
   existing source-backed JAX kernels;
2. a necessary learned primitive that needs a coupled/structured decoder; or
3. a target whose normalization or representation makes the current gate
   poorly conditioned.

This audit may propose a revised process head and reconstruction boundary, but
the architecture must be discussed with the user before implementation.
