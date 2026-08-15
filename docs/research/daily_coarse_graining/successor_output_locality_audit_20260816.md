# Successor Output-Locality Audit, 2026-08-16

## Motivation

The decoder-leaf audit closes raw Euclidean parameter-space direct steps. The
second decoder kernel and bias are both required for common first-order
descent, while excluding either first-decoder leaf leaves the stopped
`28.72%` finite regression unchanged. Parameter distance therefore does not
provide a usable local metric at this state.

This audit asks where the mismatch between first-order descent and the exact
norm-`0.0005` finite step appears in function space. It does not change the
Attention architecture or authorize another update.

## Frozen Mechanism

Reconstruct the same stopped state and negative unit parameter direction. On
the same ten training groups, run all four 30-day context trajectories per
group and record every named decoded output in its accepted training
representation. Day-start-delta heads are converted relative to their own
trajectory state, and all 181 fields are normalized by the accepted
train/train output scales.

For each group:

1. compute a forward JVP of the 30-day normalized output tape;
2. compute the exact output tape after the norm-`0.0005` parameter step;
3. replay both tapes through the same conservative update and retained exact
   processes;
4. take one three-seed VJP of NPP, AGB tendency, and BGB tendency with respect
   to the base output tape.

The finite-step mismatch is decomposed as:

```text
exact loss change - first-order loss change
  = base-output VJP * (exact output change - JVP output change)
  + nonlinear finite response of the conservative trajectory and loss
```

The first term is ranked by named field. No field is individually perturbed
or rerun, so the audit covers all 181 fields without 181 intervention
rollouts.

## Interpretation Gate

The output tapes must reproduce the ordinary end-to-end losses within
`1e-9`, the output-space chain rule must close within `1e-6`, and all state,
budget, nonfinite, identity, and stopped-reference gates must remain exact.
The BGB regression must be reproduced on at least one group.

For regressing BGB groups, output nonlinearity is called dominant only when it
accounts for at least `60%` of the absolute output-versus-downstream nonlinear
decomposition. It is called localized only when the top 12 fields account for
at least `80%` of absolute field contribution. The outcomes route the next
metric design:

- localized output nonlinearity: constrain the identified normalized fields;
- distributed output nonlinearity: constrain all normalized decoded fields;
- downstream finite-response dominance: constrain NPP/biomass trajectory
  observables rather than decoder fields.

These are metric-scope decisions, not architecture promotion or rejection.

## Boundaries

- accepted v6 data only; no Teacher regeneration and no v5 join;
- ten groups, 40 point-years, 1,200 daily transitions;
- one output JVP and one three-seed output VJP per group;
- no optimizer update, checkpoint, recovery write, retained parameters,
  validation values, sealed test, or paid node;
- one free-`gln01` execution only after local admission.

The machine contract is
`manifests/coarse_graining/successor_output_locality_audit_v1.json`, SHA256
`edaebad5831cfb0fd4ea678ef61e5d4aed23e3b67e8dbcb64dd7451e7f7e9852`.

## Local Admission

The physical/training output representations now round-trip exactly for
day-start-delta and direct fields. Field-level decomposition, aggregation,
ranking, and all three metric-scope classifications are covered by tests.
The original geometry suite plus new locality tests passes `83/83`; another
14 directly related fast-owner/materializer/schema tests pass. Ruff, Python
compilation, Bash syntax, canonical contract hash, and `git diff --check`
also pass.

The next action is one free-`gln01` output-locality audit. It is diagnostic
only and must stop after writing the report.

## Result

The free-`gln01` audit completed at commit `2a29c2c`. The report is
`runtime/outputs/training/successor-output-locality-ca6b865/report.json`,
SHA256
`93ebbff89626f438200b239a5a15745fae066fda567e92b49db74437071cfa85`.
All ten groups completed in `1968.70` seconds. Peak JAX device bytes were
`13,826,391,552`; hard and nonfinite counts are zero. Output-tape replay and
the JVP/VJP chain rule close at numerical precision, and the stopped failed
step is reproduced exactly.

Decoded-output nonlinearity accounts for `96.0844%` of the NPP finite
mismatch, `78.3205%` of AGB tendency, and `79.7907%` of BGB tendency. All ten
groups regress each of these three losses at the exact finite step despite
negative first-order changes.

The loss-relevant field attribution is sharply localized:

- `daily.fast_maintenance_respiration` carries `94.2874%` of absolute NPP,
  `87.4331%` of AGB, and `95.1144%` of BGB output-nonlinearity contribution;
- `daily.fast_gpp` is the second field, but its signed contribution is
  negative and therefore offsets rather than causes the finite regression;
- `daily.temperature_context.tsoil`, `endpoint.daily_summary.soilhum_daily`,
  and `endpoint.daily_summary.humrel_daily` supply the remaining nonzero
  contribution.

For maintenance respiration, the norm-`0.0005` step has normalized exact RMS
`0.0820612` and maximum absolute change `4.71565`, while its first-order RMS
is only `0.000413777`; the normalized linearization cosine is `-0.0357822`.
Thus the parameter direction appears locally quiet in maintenance space but
crosses into a large, oppositely oriented finite functional response.

The formal classification is `decoded_output_nonlinearity_localized`, with
next action `design_normalized_field_local_output_trust_region`. No optimizer
update, checkpoint, recovery, retained parameters, validation read, or sealed
test occurred.

## Decision After Execution

The next metric must protect the realized maintenance-respiration function,
not suppress all carbon outputs uniformly. GPP currently offsets the failed
step, so a joint undifferentiated carbon-output cap would remove useful
movement. The field-local design must use the raw failed proposal as its
matched control, preserve the all-40 common-descent objectives, and freeze an
exact normalized maintenance-output acceptance gate before another GPU run.

This evidence does not reject Attention and does not authorize training. A
safe maintenance-local correction mechanism and its bounded candidate set
must be designed next.
