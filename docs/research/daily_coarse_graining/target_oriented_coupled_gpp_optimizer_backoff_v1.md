# Coupled-GPP Optimizer Backoff V1

Date: 2026-09-01

Status: frozen before one discarded half-rate optimizer proposal

Job `14902356` established a common first-order descent direction but rejected
the finite `4e-6` AdamW step. This follow-up changes only the learning rate to
`2e-6`, the exact one-half backoff. It does not change the model, trainable
families, four objective coefficients, clip threshold, eight artifacts, data,
or physical gates.

The proposal starts from the same structured-GPP checkpoint and exact-zero
coupled adapter with one fresh AdamW state shared by both trainable families.
No inherited moments are used. It is one independent proposal, not a chained
update and not a rate sweep. The optimizer, proposed parameters, and all
moments are discarded after the complete eight-artifact reevaluation.

Acceptance remains strict: all averaged source gradients must align positively
with the realized negative step; direct GPP, reachable state, 7-day science,
30-day science, and the weighted total must each decrease; every trainable leaf
must change; and every physical, fixed-parameter, and checkpoint-integrity gate
must pass. Passing opens only target-blind schedule and bounded-training
contract design. Failure closes this optimizer realization before training.

The machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_optimizer_backoff_v1.json`,
canonical SHA256
`129d2892c384aee8d206b61a93e642020a7106dd436fbe88dbe902f90aadc315`.
