# Coupled-GPP Scale-Selected 7/30-Day Transfer Audit v1

Date: 2026-09-02

Status: frozen before non-test transfer values are evaluated

Machine contract:
`manifests/coarse_graining/target_oriented_coupled_gpp_scale_selected_transfer_audit_v1.json`

Contract SHA256:
`110d3b4a158d4ab9a76858ce1afab082df97a842874c248a0094f01869ab571d`

## Question

Do the three accepted scale-selected updates improve 7-day and 30-day free
rollout on data that did not determine their gradients or selected scales?

The audit reconstructs updates 5-7 exactly in one process, retains no state,
and compares the parameter state before update 5 with the selected state after
update 7.

## Independent Panel

Use the existing train/train non-test monitor artifacts:

```text
251, 358, 465, 572, 679, 786, 893, 1000
```

They supply 64 seven-day and 32 thirty-day windows. None belongs to the 24
artifacts used by updates 5-7. The panel is not used for a gradient, direction,
scale choice, or parameter update. No new validation or test values are read.

## Acceptance

Evaluate identical before/after panels with the established component-resolved
science objective and conservative daily updater. The mean 7-day loss and mean
30-day loss must each strictly decrease. GPP-direct and one-day reachable-state
changes are reported for attribution but do not add another gate. Physical and
finite-value gates must pass before and after.

Every artifact's change and the number improved per horizon are reported, but
unanimous improvement across all eight artifacts is not required. This avoids
turning normal heterogeneous sampling noise into an artificial rejection.

## Execution And Next Gate

One free `gln01` execution is allowed with a 120-minute timeout. It writes only
the JSON audit report. It writes no checkpoint or recovery and retains no
reconstructed parameter tree. Teacher regeneration, historical v5, fresh
validation, sealed test, 48-record forcing reconstruction, and 48-step state
scans remain forbidden.

Passing opens the design and local qualification of a 100-update persistent
scale-selecting training contract. Failure triggers horizon-level attribution
without changing the Attention architecture. Paid training is not authorized
by this contract alone.
