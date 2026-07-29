# Rollout `min_stomate` Threshold Stabilization

Date: 2026-07-30

## Decision

The update-226 nonfinite-gradient failure is closed. The failure was not a
generic architecture or optimizer instability. It was a source-level
floating-point boundary pathology in the retained daily carbon transition,
triggered only after the neural rollout left the normal Teacher state
manifold.

Commit `f9554c6` preserves the source carbon budget but pins a corrected
negative carbon stock exactly to `min_stomate`. Commit `452b743` adds a
resumable, fail-closed screening-prefix gate. The candidate passed a complete
64-sample update audit and 256 sequential mixed-horizon optimizer updates,
including an exact checkpoint restart at update 128.

This is not promotion of the daily surrogate. The formal matched 8,192-update
screen remains outstanding and the sealed test split remains untouched.

## Root Cause

The source constant is `min_stomate = 1e-8` in
`src_parameters/constantes_var.f90:177`. It is consumed by prescribe,
allocation, and NPP/post-NPP owners.

An owner-isolation diagnostic on rollout samples 26 and 29 established:

- prescribe-only was identical to the historical zero-threshold path;
- allocation-only caused finite, small changes;
- post-NPP-chain-only reproduced the full failure;
- the NPP correction itself changed biomass by about `1e-8`;
- the first explosive boundary was `npp_leaf_age_sla_age_update`, where a
  strict `biomass > min_stomate` gate divided old leaf-class mass by a
  near-zero corrected stock.

The source expression is:

```fortran
bm_create = min_stomate - biomass
biomass = biomass + bm_create
```

For a crafted negative stock, both source-extracted `gfortran -O0` and `-O3`
executions produced `1.0000000000000002e-8`, one ULP above the threshold.
That result incorrectly activates the subsequent strict `>` gate. This proves
that the pathology exists in the literal Fortran expression as well as JAX;
it is not a JAX transcription error.

The stabilized JAX owner still computes `bm_create` from the old stock and
uses it unchanged in the maintenance-respiration carbon balance. Only the
corrected stock writeback is pinned exactly to `min_stomate`, matching the
source comment's stated semantic intent and keeping the strict gate inactive.

## Local Evidence

The NPP source-extraction Oracle now uses the real `min_stomate=1e-8` constant
instead of its historical zero-valued stub. It passes with a maximum biomass
difference of `6.08e-17`; all downstream leaf-age and leaf-fraction outputs
match.

Regression results:

```text
7 targeted NPP tests passed
228 carbon/daily/rollout tests passed
68 coupled STOMATE tests passed
18 prefix-gate framework tests passed
Ruff, py_compile, and git diff --check passed
```

The dedicated threshold micro-case verifies exact stock writeback, unchanged
carbon-budget compensation, an inactive strict leaf-fraction gate, and finite
leaf-age state.

## GPU Evidence

All diagnostics used the frozen source execution from training commit
`13f09c9`, current diagnostic code, the admitted 669-point v5 dataset, and no
sealed-test sample.

At commit `f9554c6`:

| Gate | Result |
| --- | --- |
| sample 26 `L_next` | loss `8.57863e-5`, gradient norm `0.0096586`, 0 nonfinite gradients |
| sample 29 `L_next` | loss `0.0902370`, gradient norm `0.00867404`, 0 nonfinite gradients |
| complete update 226 | 64/64 rollout samples finite, 0 bad anchor samples |
| update 226 hard counts | all four exact hard counts zero |
| update 226 candidate | loss `0.0134245`, gradient norm `0.110331`, update applied |

The complete update report also has finite independent gradients for
`L_fast`, `L_next`, `L_rollout`, `L_bias`, and `L_science`.

At commit `452b743`, the sequential prefix gate passed:

| Prefix | Resume point | Horizon counts `1/3/7/30` | Last loss |
| --- | ---: | --- | ---: |
| 128 | 0 | `51/37/29/11` | `0.0210088` |
| 256 | 128 | `105/81/47/23` | `0.00686241` |

Every update was applied. At update 256, all exact hard counts remained zero.
The second invocation accepted and extended the first invocation's
hash-bound checkpoint, proving the resume path rather than only writing a
checkpoint.

Evidence SHA256 values:

```text
sample 26: c1f9965d8e1ee010bb4d3678ed14d722528421855e0f12efed03dc7e4b0b1214
sample 29: 1df6898c767f86423be43da54b2f9062beb3b3ff529f29db1ec8528825160d43
update 226: 66a2f5f89039dde4fa37428cc6d887a7e635cbc709c615ca2df14bee8a1cfbf0
prefix 128: bfdcf0eaff43fa4a90daf842f215af8887b0eaf142a7b5c2bd8a334d052095e0
prefix 256: 8d23f0063eeb943d440172815ff3b429b53566acf11bd68cf4d0ed4093b6748c
```

## Next Gate

Create a new immutable Experiment B output root and regenerate preflight,
train-only coefficient calibration, and execution identity from one clean
current commit. Then rerun both matched arms under the existing six-hour
`gnall` limit. Do not reuse the failed `13f09c9` training root as formal
promotion evidence, do not bypass the matched control, and do not inspect the
sealed test split before the screen passes.
