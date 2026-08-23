# Target-Oriented Representative Gradient Probe

Status: **completed negative execution evidence; a confirmed zero-variance
reverse-mode defect is repaired locally and awaits real-v6 verification**.

Date: 2026-08-23.

## Scope

Explore1000 job `14644566` ran one zero-retained-update representative batch
on one V100 at commit `d1eda1bdbdf8b9d33df7e4599c9935abfa7d29c0`. It used the
accepted Teacher-v6 release and existing prepared batch. It did not regenerate
Teacher data, read historical v5 or the sealed test split, retain an optimizer
update, or write a checkpoint.

The report is
`runtime/outputs/training/target-oriented-representative-gradient-probe-d1eda1b/report.json`.
Its SHA256 is
`3575cfaede28cb572f1283a9de50058e1a76c915cb0ba2c3aa846ae9793f2aee`.
The job ran for `01:11:12` and the report recorded approximately `8.196 GB`
peak live JAX device memory.

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

The gate nevertheless failed because the one-day/process, seven-day, and
thirty-day reverse gradients all contained NaNs. The auxiliary output heads
had finite gradients, and the one-day direct primary heads were also finite.
NaNs appeared after gradients traversed shared state summaries and propagated
through rollouts. The optimizer proposal was therefore rejected and no state
was retained.

## Confirmed Derivative Defect

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

This is a confirmed defect in the executed derivative graph and explains the
observed failure pattern. It is not yet claimed to be the only possible NaN
source. That claim requires a real-v6 rerun.

## Next Gate

Commit and synchronize the forward-preserving derivative repair, then rerun
the same single-batch, zero-update representative probe. Advance to the full
eight-batch gradient calibration only if all three gradient sources and all
seven parameter families are finite and nonzero where connected.

Paid training, retained optimizer updates, 365-day rollout, Gate D2, sealed
test access, and product acceptance remain closed.
