# Coupled GPP Artifact-Aware Direction Result (2026-09-04)

## Outcome

The bounded free-`gln01` audit completed at commit `77d5a45` in
`3,630.564` seconds. Its classification is
`artifact_aware_direction_transfers_to_monitor`: a 43-constraint common
direction exists, scale `1/2` passes all train-only gates, and fixed-monitor
direct-GPP mean loss improves by `2.7567%`. All `40/40` train-only artifacts
and all `8/8` fixed-monitor artifacts improve.

The server report is
`runtime/outputs/diagnostics/target-oriented-coupled-gpp-artifact-aware-direction-77d5a45.json`,
SHA256
`67bde6d8e57f9c49a31f5a6dbba89f0b24545d92a8a49366054668b62b09cbfe`.
The hash identifies the immutable report; it is not a cross-GPU numerical
equality gate.

## Direction Geometry

The minimum-norm simplex solver finds a common direction over 40 separate
direct-GPP artifact gradients and three scheduled non-GPP gradients. The
minimum constraint cosine is `0.0422923`, above the frozen `1e-4` gate; the
maximum is `0.2692304`.

The simplex support has 21 of 43 constraints: all three scheduled non-GPP
constraints and 18 direct-GPP artifacts. Reachable state has the largest
weight at `34.7335%`; the largest GPP artifact weights are update 393 at
`14.8398%`, update 567 at `10.1353%`, and update 984 at `8.8372%`. This is a
materially different direction from the failed five-group mean construction:
the solver is controlled by difficult individual batches rather than by five
averages.

## Train-Only Step

The first and largest frozen scale, `1/2`, passes, so the early-stop rule
correctly skips the remaining five scales. Scheduled source changes are:

| Source | Relative change |
| --- | ---: |
| Direct GPP | `-4.9839%` |
| Reachable one-day state | `-1.4359%` |
| 7-day component-resolved science | `-0.6419%` |
| 30-day component-resolved science | `-0.6429%` |

All scheduled physical gates pass. Across the 40 separate direct-GPP
artifacts:

- `40/40` improve;
- the combined mean changes from `0.0584647415` to `0.0564356740`, or
  `-3.4706%`;
- the median artifact change is `-3.0474%`;
- the least-improved artifact is update 816 at `-0.4916%`; and
- the five eight-artifact group means improve by `1.6796-4.9839%`.

Thus the finite-step result is not being carried by a mean while a hidden
train-only subgroup regresses.

## Fixed-Monitor Transfer

The monitor was read only after scale `1/2` had been selected using train-only
evidence. Its mean changes from `0.0607357871` to `0.0590614624`, or
`-2.756735%`:

| Artifact update | Relative change |
| ---: | ---: |
| `251` | `-5.3709%` |
| `358` | `-3.2263%` |
| `465` | `-0.3682%` |
| `572` | `-3.2391%` |
| `679` | `-1.2783%` |
| `786` | `-0.0507%` |
| `893` | `-3.6356%` |
| `1000` | `-2.0373%` |

This is qualitatively stronger than both failed mean-direction audits, which
each improved only `4/8` monitor artifacts and worsened the monitor mean.
Separate GPU reconstructions of the logical pre-update-2 state are not
bit-identical, and this run's baseline differs materially from the preceding
ones. The within-run paired result is valid, but its magnitude must not be
ranked directly against those earlier percentages. Independent consecutive
update reconstruction is required before any persistent training decision.

## Integrity And Runtime

The parent recovery, both predecessor results, and all `93,504,164` fixed
parameters remain unchanged. No optimizer update, parameter proposal,
recovery, checkpoint, or trainable parameter is written. The audit uses only
accepted v6 train/train artifacts and does not read fresh validation or sealed
test data, regenerate Teacher data, join v5 data, reconstruct 48 forcing
records, or execute a 48-step state scan.

Reported JAX allocator peak is `12,114,246,144` bytes against a
`25,559,728,128`-byte allocation limit. The 43-gradient matrix is therefore
feasible on one V100. Early stopping after scale `1/2` reduces runtime by
about 21 minutes relative to the preceding six-scale audit.

## Decision

Accept artifact-level GPP direction construction for a bounded continuity
test. This does not promote the model and does not yet show that the direction
remains feasible after its own first update.

The next task is a frozen, no-checkpoint three-update backtracking smoke. Each
update must recompute an artifact-aware direction on a new train-only panel,
select its scale without monitor feedback, and preserve scheduled state plus
7/30-day science. Only after all three train-only steps are fixed may the
fixed monitor trajectory be evaluated. Persistent training, fresh validation,
365-day rollout, Gate D2, sealed test, and product acceptance remain closed.
