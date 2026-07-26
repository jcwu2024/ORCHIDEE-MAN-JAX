# Structured Architecture Matched A/B Result

Date: 2026-07-26

## Decision

Reject `structured_process_film_v1` as a promotion candidate. Keep the frozen
`canonical_multistep_v1` checkpoint as the accepted neural baseline. Do not
continue this candidate with more epochs, wider adapters, or FiLM-weight
tuning.

This is an informative negative result, not a failed experiment. Both arms
started from the same accepted checkpoint, used identical samples and update
budgets, completed all expected diagnostics, and left the sealed test split
untouched.

## Execution Evidence

- Slurm job: `14386641`, `COMPLETED`, exit `0:0`;
- node and resources: `ibc13b03n04`, one V100 and four CPU cores;
- elapsed time: `01:03:27`;
- peak RSS: `8,474,888 KiB`;
- accepted initialization checkpoint SHA256:
  `79728593fa78f45f0c77dbe219f983114f76fe39bc0b63443c4af11e612b5bc2`;
- both arms used seed `20260724`, batch size 4, learning rate `1e-4`, and
  `1:64,3:64,7:64` `canonical_multistep_v1` updates;
- every arm received the same number of batches from every training
  landpoint at every horizon;
- all eight rollouts had zero defined-status and discrete-state mismatches;
- no sealed test sample was evaluated.

At the stated additive rates, elapsed resource cost was approximately CNY
`0.30` CPU plus CNY `2.33` GPU, or CNY `2.62` total.

## Seven-Day Results

The flat and structured arms are matched continuation controls. The frozen
column is the pre-experiment accepted checkpoint with the promoted physical
stock projection.

| Split | Frozen | Flat continuation | Structured | Structured vs flat |
|---|---:|---:|---:|---:|
| train/train | 0.144765 | 0.155313 | 0.167237 | +7.68% |
| train/validation-year | 0.133391 | 0.163880 | 0.182427 | +11.32% |
| validation-point/train-year | 0.519286 | 0.534859 | 0.511520 | -4.36% |
| joint validation | 0.428353 | 0.451209 | 0.432085 | -4.24% |

The candidate improves both spatial-validation cases relative to the matched
flat continuation, but materially regresses both cases at the seen
landpoint. It also fails to improve the frozen baseline robustly: the
validation-point/train-year result improves only 1.50%, joint validation is
0.87% worse, and seen-condition results are 15.52% and 36.76% worse.

The named-state result is similarly mixed. On train/train, the structured arm
improves biomass, NPP, growth respiration, and maintenance respiration, but
`litterpart` worsens from `4.074` for the flat continuation to `4.614`; the
frozen value was `3.652`. On joint validation it prevents the flat
continuation's severe `litterpart` regression (`5.806 -> 3.096`), but remains
slightly above the frozen `3.005` and does not improve the global score.

The matched flat arm is important evidence by itself: another 192 updates
worsen all four frozen global scores. Lower training loss or additional
updates therefore cannot be treated as progress without frozen rollout
validation.

## Condition-Use Result

| Arm | Parameter permutation loss change | Static permutation loss change | Classification |
|---|---:|---:|---|
| Flat continuation | -3.56e-6 | 1.33e-4 | detectable static use |
| Structured | -4.25e-5 | 5.26e-5 | condition use insufficient |

The predeclared detectability threshold is `1e-4`. The structured network does
respond numerically to parameter and static perturbations, but those responses
are not useful on validation data. Parameter permutation slightly improves
its loss, which indicates a wrongly learned conditional association rather
than a dead computational channel. Forcing remains strongly used by both arms
with validation permutation loss changes near `0.33`.

The validation point remains outside the training leave-one-out envelope in
the joint 617-dimensional active static-feature group. This is a group-level
distance result, not a claim that all 617 attributes are individually outside
their training ranges. The flat continuation crosses the condition-use
threshold yet still fails spatially, strengthening the evidence that bounded
spatial-data coverage is now the next variable to test.

## Next Gate

Do not promote either continuation checkpoint. Do not inspect the sealed test
split or generate all 669 trajectories.

Prepare a source-selected spatial pilot from the frozen 669-point inventory:

1. select only additional training-split landpoints using parameters, static
   attributes, and forcing climatology, without looking at neural errors;
2. explicitly reduce the static-feature coverage gap around the existing
   validation point while also adding maximin extremes;
3. preserve the current spatial validation and sealed test points;
4. estimate full-trajectory Teacher cost and storage before generation;
5. use `canonical_flat_v1` as the primary data-coverage control. Revisit a
   structured architecture only after expanded data establish that its
   conditional relationships are identifiable.

Server evidence root:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/experiments/
structured-architecture-ab-cce121e
```
