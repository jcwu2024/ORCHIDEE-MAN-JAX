# Experiment B Calibration Failure

Date: 2026-07-29

## Outcome

Explore1000 job `14410820` ran on `gnall` node `ibc13b02n01` for `00:07:47`
and exited with code `1:0`. It failed in train-only coefficient calibration
before either matched training arm started. No arm checkpoint was generated,
and the sealed test split was not used.

The failing deterministic schedule entry was:

- calibration ordinal: 4;
- landpoint: `087.0-105.0`;
- year: 1961;
- horizon: 1;
- batch size: 256.

The diagnostic asset is:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/
canonical-669-rollout-stability-14e4b31/
coefficient_calibration_diagnostic_0004.json
```

Its SHA256 is
`f24eeef2f937c2f828b4b7449cdadbaa5d60ef8cf581d186564eb622e6751a2e`.

## Attribution

The batch had exactly three defined/undefined status mismatches. All three
belonged to `diffuco_previous_step_state.rveget`, offset 1, which is the
compact PFT14 column:

| Day | Predicted | Teacher |
| --- | ---: | ---: |
| 2 | undefined | 1786.4529129942741 |
| 294 | 1404.7333118293113 | undefined |
| 358 | undefined | 1136.434452000587 |

There were zero discrete-state mismatches, nonfinite defined values, and
negative source-constrained carbon stocks. The failure is therefore not a
retained STOMATE handoff error, numerical corruption, or physical-domain
projection failure. It is an ordinary error from the explicit learned
`rveget` defined/undefined classifier.

## Protocol Amendment

Protocol v1 supervised dynamic `rveget` status with a binary loss while also
requiring every status prediction to be exact before any optimizer update.
Those requirements are contradictory for a learned classifier that has not
already reached perfect training accuracy.

Protocol v2 keeps two separate counts:

- `declared_dynamic_status_mismatches`: only contract-declared dynamic
  `rveget` owners; supervised, reported, and compared with the matched control,
  but not an optimizer veto;
- `unexpected_defined_status_mismatches`: every other defined-status change;
  exact-zero hard failure.

Discrete-state mismatches, nonfinite defined values, and negative
source-constrained carbon stocks remain exact-zero hard failures. The
candidate/control screening adds a maximum declared-dynamic-status error ratio
of `1.05`; this prevents the candidate from gaining rollout score by damaging
the classifier.

Protocol v2 canonical SHA256:
`a0ecfd4c89ff3c5691f153168f3ba80939582774689b92ff04b0cf77e699cb40`.

## Resubmission Gate

Do not reuse the v1 preflight or output identity. Before another paid job:

1. Commit the validated v2 implementation.
2. Create a clean commit-bound server worktree and a new output root.
3. Regenerate protocol, Git HEAD, preflight, and execution identities.
4. Run calibration ordinal 4 as a free `gln01` smoke.
5. Require three declared dynamic mismatches, zero hard counts, finite loss and
   gradients, and an applied candidate update.
6. Only after that gate passes, request approval for a new paid Experiment B
   run.
