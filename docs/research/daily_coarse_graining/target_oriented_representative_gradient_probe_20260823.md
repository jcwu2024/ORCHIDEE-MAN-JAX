# Target-Oriented Representative Gradient Probe

Status: **accepted repaired representative gradient gate; all three objective
sources, all combined parameter gradients, the optimizer proposal, and the
physical hard gates pass; the eight-batch preflight is active**.

Date: 2026-08-23 to 2026-08-24.

## Scope

Explore1000 jobs `14644566` and `14650340` each ran the same zero-retained-
update representative batch on one V100. They used the accepted Teacher-v6
release and existing prepared batch. They did not regenerate Teacher data,
read historical v5 or the sealed test split, retain an optimizer update, or
write a checkpoint.

The first report is
`runtime/outputs/training/target-oriented-representative-gradient-probe-d1eda1b/report.json`.
Its SHA256 is
`3575cfaede28cb572f1283a9de50058e1a76c915cb0ba2c3aa846ae9793f2aee`.
The job ran for `01:11:12` and the report recorded approximately `8.196 GB`
peak live JAX device memory.

The second job ran commit
`5a50a65ed8e6a07eb7294705a57075a9ae91a874`. Its report is
`runtime/outputs/training/target-oriented-representative-gradient-probe-5a50a65/report.json`,
with SHA256
`c1bce9ed922ede9713aaa0f552e157c51106e999c822b5bb220bf2523fdbbdde`.
It ran for `01:11:46`.

## Result

The amended state anchor removed the earlier approximately `1e14` DOC loss
domination. The representative batch reported:

| Component | Loss |
| --- | ---: |
| One-day state anchor | `327.2342617` |
| Necessary process supervision | `0.3917095` |
| Seven-day science | `0.9829326` |
| Thirty-day science | `0.3557758` |
| Total | `328.6121410` |

All learned outputs, states, budgets, discrete state, and inventories remained
finite and valid. Undeclared mask mismatches, budget failures, discrete
mismatches, negative stocks, and nonfinite predictions were all zero. The
only mask mismatches were the already declared dynamic `rveget` status: 24 on
the one-day samples, 28 over seven-day windows, and 95 over thirty-day
windows.

Both gates failed because the one-day/process, seven-day, and thirty-day
reverse gradients all contained NaNs. Job `14650340` reproduced the first
job's loss, hard counts, and gradient classifications exactly. The optimizer
proposal was rejected and no state was retained in either execution.

## Correcting The First Diagnosis

`daily_operator/strong_baseline_architecture.py::_masked_summary` computed a
masked standard deviation as:

```python
jnp.sqrt(jnp.maximum(variance, 0.0))
```

Real input groups can contain one selected coordinate or constant values, so
their variance is exactly zero. The forward standard deviation is then the
correct finite value zero, but reverse-mode differentiation through
`sqrt(0)` is singular. A minimal local reproduction returned finite forward
values and a NaN gradient.

The local repair uses the existing canonical
`source_sqrt_with_finite_zero_tangent` primitive. It preserves the exact
forward value and defines the zero-variance tangent as zero. A new regression
covers both a constant two-value group and a one-value group. The focused
zero-variance regression, canonical AD primitive tests, complete
target-oriented Attention suite, Ruff, and `git diff --check` pass.

This is a real derivative defect, but it does not explain the parameter-
gradient failure. The masked summary is computed from fixed model inputs, and
the representative probe differentiates with respect to network parameters.
Job `14650340` confirmed the distinction: applying only this repair left all
three NaN gradient sources unchanged.

## Learned-Output Boundary Localization

The follow-up diagnostic differentiates each of the seven one-day state-group
losses with respect to the 39,006 scalar physical learned-output coordinates.
At commit `3d5a62f`, every group had exactly one nonfinite coordinate:

```text
inventory.water.qsintveg.storage.incoming_amount
```

It was one of the field's two PFT coordinates; all other 39,005 coordinates
were finite. The report SHA256 is
`1c223c2710e37437391ce4b8f31ab2382b588bda991d292abe936f7ef6251a72`.

The source is
`daily_operator/mechanistic_output_reconstruction.py::signed_endpoint_to_transfers`.
The old expression was:

```python
outgoing = jnp.where(source > 0.0, removed / source, 0.0)
```

Bare soil has zero canopy inventory and zero removed amount. The division
therefore evaluates `0/0` before `where` masks the forward branch to zero.
Forward state and budgets remain finite, but reverse mode receives the NaN.

Commit `c8cb497` first selects a denominator of one wherever the source is not
positive, then applies the unchanged forward `where`. This preserves the
physical result. The minimal gradient reproduction changes from NaN to finite
`[[0, -5]]`, and the complete mechanistic-output regression passes 13 tests.

The repaired real-v6 free-`gln01` diagnostic first made every output-boundary
coordinate finite for all seven groups; its report SHA256 is
`4444e23b79a9a08e033b420c7d64bb606f35f5cf0aacc18232322ff8b721ddc4`.
Commit `e8294b0` then propagated the real boundary cotangent through the
complete Attention head. All 83,755,173 parameter-gradient values are finite,
with total norm `419576.1733`. The base encoder, domain refinement, science
cross-attention, science mixer, primary heads, and state heads all have finite
nonzero gradients. Auxiliary heads are correctly zero because this diagnostic
uses only the state anchor. The final report is
`runtime/outputs/diagnostics/target-oriented-output-boundary-vjp-e8294b0/report.json`,
with SHA256
`ee2f1715e1316658fc7ccdaaf6f35a0819c7fdde4de19afead800b076a5fba59`.

No architecture, data, loss coefficient, optimizer, or prepared-batch policy
changed during this diagnosis.

## First Repaired Full-Probe Attempt

Approved paid job `14651381` ran commit
`2a6a5d5676873d715039daf0969f3a49e5ba58ce` with one V100, four CPUs, and a
`01:30:00` limit. Slurm terminated it at `01:30:21` with state `TIMEOUT`. The
runner had not completed the first representative batch and had not reached
its atomic report write, so no report, optimizer proposal, checkpoint, or
training update exists.

The log contains no Python exception, nonfinite report, or physical hard-gate
failure. It shows that the repaired path continued through full XLA work that
the earlier NaN executions did not time reliably. Compilation of one backward
module alone took approximately `20m27s`. The project-scoped persistent JAX
cache is now approximately 262 MiB and contains the completed forward,
one-day, and later backward executables from this attempt. The timeout is
therefore incomplete execution evidence, not a model or gradient rejection.

The rerun changes no numerical operation. The runner now synchronizes and
prints completion time after each of the one-day/process, 7-day, and 30-day
gradient sources. The Slurm ceiling is raised to three hours so the existing
cache and any remaining cold compile can complete without another 90-minute
cutoff.

## Passing Repaired Execution

Approved paid job `14652590` ran commit
`50c2c14435361d62c64b7cc643d568dfa36bd9bc` on one V100 with four CPUs. It
completed normally in `00:53:12`. The report is
`runtime/outputs/training/target-oriented-representative-gradient-probe-50c2c14/report.json`
with SHA256
`aa1954124408e9291aff0afb4da7b26d52019f22369ee66e55e92742733ac861`.

The report status is `passed`:

- all 83,755,173 values in each of the one-day/process, 7-day, and 30-day
  source gradients are finite;
- their norms are `5202018.8299`, `1.4514`, and `0.8519`;
- all 83,755,173 values in the combined gradient are finite, with all seven
  connected parameter families finite and nonzero;
- the optimizer proposal has raw norm `5202019.0`, clip scale `1.0`, finite
  step norm `0.00345624`, and zero nonfinite parameter or optimizer-state
  values;
- peak live device memory is `8,179,523,584` bytes;
- budget, discrete, stock, unexpected-mask, prediction-nonfinite, and state-
  nonfinite hard counts are zero; and
- the 147 observed mask mismatches are entirely the declared dynamic
  `rveget` status.

The runner retained zero updates, wrote no checkpoint, did not regenerate
Teacher data, did not read v5 or the sealed test split, and did not alter the
architecture or objective.

The stage timings were `102.4s` for one-day/process, `216.2s` for 7-day, and
`2083.2s` for 30-day; the last includes a `23m16.7s` cold XLA compile. At
initialization, the long-horizon gradients strongly reach the primary science
heads but are about seven orders of magnitude below the one-day anchor in the
shared representation families. AdamW's per-coordinate normalization means
the raw global ratio is not itself the resulting parameter-step ratio, but it
is a conditioning fact that must be observed across the eight diverse batches
before training.

## Next Gate

Run the frozen eight-batch zero-update preflight. It repeats the same 32
one-day, eight 7-day, and four 30-day composition over eight diverse
point-year batches, establishes the 90th-percentile gradient clipping
threshold, and records whether the source/family conditioning seen here is
stable. Retain all existing finite-gradient, parameter-family coverage,
optimizer, memory, state, budget, discrete, inventory, and mask gates.

Paid training, retained optimizer updates, 365-day rollout, Gate D2, sealed
test access, and product acceptance remain closed.
