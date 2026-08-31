# Structured-GPP Fresh Error Attribution V1

Date: 2026-08-31

Status: frozen before attribution execution

Contract SHA256:
`2c93dbe5d31791474c9c02bbf08efd76f831c1f7e99bb5c1458b4cd8f880112a`

## Question

The broad step-1000 checkpoint fits its train-only monitor but fails GPP,
growth respiration, NPP, NEP, and NEE on the fresh 96-window panel. This audit
separates three questions before another training or architecture decision:

1. how much GPP error is already present when the process head receives the
   correct Teacher day-start state;
2. how much additional GPP error is introduced when the model carries its own
   predicted state across days; and
3. how much of the downstream growth-respiration and NPP failure is causally
   removed when only GPP is made exact.

## Frozen Modes

All modes use the exact broad step-1000 parameters, native daily inputs,
prepared fresh panel, conservative updater, and retained exact tail.

- `ordinary_free_rollout` is the legal candidate path and carries predicted
  state across days.
- `teacher_day_start_direct_gpp` calls only the process head independently on
  each Teacher day-start state. It does not execute a replacement rollout.
- `teacher_gpp_bridge` keeps the candidate's free state recurrence and every
  other learned output, but replaces `daily.fast_gpp` with the matched Teacher
  value before the conservative update and retained tail. It is a diagnostic
  intervention and cannot enter training or final inference.

For each matched day, the audit records the exact identity

```text
free GPP error = Teacher-state local GPP error + state-feedback GPP increment
```

It reports normalized RMS, signed bias, the local/feedback cosine, and the
squared-error cross term both daily and after the same 7/30-day sums used by
the science gate. This avoids pretending that two RMSE values can simply be
subtracted.

The bridge reports the existing absolute science statistics for GPP,
maintenance respiration, growth respiration, autotrophic respiration, NPP,
and the protected carbon stocks. It also verifies the raw cumulative identity

```text
NPP error = GPP error - maintenance-respiration error
            - growth-respiration error
```

## Bounded Gates

First run one frozen window in each of the six slice/horizon cells as a free-
GPU admission. It must pass the finite, residual-closure, carbon-identity, and
physical gates before the formal 96-window execution. In formal mode the
ordinary free mode must reproduce every metric in the accepted formal report
within `1e-12`. GPP residual closure must be within `1e-12`, its
squared-error decomposition within relative `1e-12`, and the raw carbon
identity within `1e-10`. All reported values must be finite and nonempty, and
both ordinary and bridged rollouts must have zero physical hard failures.

Scientific classification is descriptive rather than a promotion gate.
Teacher-state cumulative GPP failure identifies a local-mapping problem.
Feedback is called material only when its RMS is at least 25% of total GPP RMS
in at least three of six cells. Teacher GPP is sufficient only if growth
respiration and NPP pass all six bridged cells; it contributes if their median
acceptance-score reduction is at least 50%.

## Boundaries

The audit performs zero optimizer updates, parameter steps, and checkpoint
writes. It regenerates no Teacher data, uses no historical v5 or sealed test,
and performs no 48-step forcing reconstruction or subdaily state scan. It
cannot authorize 365-day rollout, Gate D2, product acceptance, a loss edit, or
an architecture edit. If integrity passes, the evidence must be interpreted
and any architecture change discussed with the user first.
