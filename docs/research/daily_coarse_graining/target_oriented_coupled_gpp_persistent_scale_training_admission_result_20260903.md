# Persistent Scale-Selected Training Admission Result

Date: 2026-09-03

Status: free runner admission accepted; paid 100-update trial requires separate
resource authorization

Executable commit: `7d578871677740a9351450b239cca72e4f8d3b22`

Contract SHA256:
`1742d9d19a6b8e8d91331723bf06047a17fa5ce5d31cc528987205a838d8b5e4`

Result SHA256:
`f9fa0f5e457afebf803a853953517bb5343b9fd58f8074ed68effa0492285119`

## Result

The repaired free `gln01` admission completed with status `passed` and
classification `persistent_runner_admission_passed` in `4,669.897` seconds.
It reconstructed the selected update-7 state, evaluated the detailed
eight-artifact initial monitor, proposed one new persistent update, selected
scale `1`, round-tripped a temporary parameter-only recovery, and discarded
both the update and the temporary recovery.

The initial four-source means were:

| Source | Mean loss |
| --- | ---: |
| direct GPP, 208 teacher days per artifact | 0.06668465 |
| reachable state, 32 teacher days per artifact | 0.02382732 |
| 7-day component-resolved science | 0.07552690 |
| 30-day component-resolved science | 0.08717707 |

The monitor's cumulative GPP scores were `0.41784010` at 7 days and
`0.49782219` at 30 days. The per-key loss scores below are reported separately
so that good stock behavior cannot hide poor productivity or respiration:

| Science key | 7-day score | 30-day score | Absolute gate at 7/30 days |
| --- | ---: | ---: | --- |
| GPP | 0.41784010 | 0.49782219 | fail / fail |
| maintenance respiration | 0.00570171 | 0.00737495 | pass / pass |
| growth respiration | 0.14559890 | 0.16435917 | fail / fail |
| heterotrophic respiration | 0 | 0 | pass / pass |
| autotrophic respiration | 0.07611733 | 0.08018841 | fail / fail |
| NPP | 0.56010662 | 0.70786003 | fail / fail |
| NEP | 0.28654140 | 0.36212963 | fail / fail |
| NEE | 0.28654140 | 0.36212963 | fail / fail |
| aboveground biomass carbon | 0.00000947 | 0.00021262 | pass / pass |
| belowground biomass carbon | 0.00001384 | 0.00032232 | pass / pass |
| total live biomass carbon | 0.00002103 | 0.00048228 | pass / pass |
| nonliving ecosystem carbon | 0.00002360 | 0.00043583 | pass / pass |
| ecosystem carbon | 0.00002427 | 0.00046206 | pass / pass |

These absolute gates are diagnostic at this admission stage. Their failures
make clear that the current parameters are not a scientifically accepted
surrogate. The admission accepts the execution, monitoring, stopping, and
recovery machinery only.

## Runtime And Persistence Evidence

The first detailed monitor artifact took `560.895` seconds because it included
the two rollout compilations. The remaining seven took approximately
`9.4-11.8` seconds each. The discarded persistent update took `716.909`
seconds and accepted the largest tested scale, `1`.

JAX reported `4,114,680,576` peak bytes in use and `1,056,111,360` bytes in
use after the run. The approximately 25.6 GB allocator limit is not the live
tensor footprint.

The temporary recovery SHA256 was
`723edcb7ad361583d0b57cbe2ca0b71abe57c96acc1c7de150030e19e57707d0`.
The report records `persistent_updates_persisted=0`,
`checkpoint_written=false`, and `recovery_retained=false`. The remote
`target-oriented-coupled-gpp-persistent-admission-7d57887-scratch` directory
was empty after exit. A post-run `nvidia-smi` check showed both `gln01` V100s
at `0 MiB` with no running process.

No Teacher data were regenerated, historical v5 data were not used, fresh or
sealed validation was not read, no 48-record forcing was reconstructed, and
no subdaily state-scan step was executed.

## Decision

The free admission gate is accepted. The next permitted action is to request
explicit authorization for the frozen 100-update `gnall` trial. That trial
must retain the update-10 and update-25 early-stop rules and may select a
checkpoint only after direct GPP recovers to the pre-update-5 baseline while
the accepted state and 7/30-day gains remain intact.

Fresh validation, 365-day rollout, Gate D2, sealed test, and product acceptance
remain closed. The paid trial is not evidence that the architecture has
already met the absolute GPP, NPP, respiration, or carbon-flux gates.
