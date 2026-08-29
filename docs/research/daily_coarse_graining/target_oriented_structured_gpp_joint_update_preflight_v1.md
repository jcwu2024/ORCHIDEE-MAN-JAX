# Structured GPP Joint-Update Preflight V1

Date: 2026-08-29

Status: frozen before one free-GPU discarded joint optimizer proposal; local
implementation accepted

Machine contract:
`manifests/coarse_graining/target_oriented_structured_gpp_joint_update_preflight_v1.json`

Contract SHA256:
`5d255e49f7add44c4948a3e737fe6edd72999b6fbd6a37439dc787cb38c0c5d5`

## Purpose

The accepted calibration established a finite shared raw gradient for direct
GPP, 7-day science, and 30-day science. This preflight checks the next distinct
question: whether the actual inherited AdamW state transforms that raw
gradient into a useful parameter step. Earlier work in this project showed
that a valid common raw direction can be destroyed by optimizer momentum or
coordinate-wise preconditioning, so raw cosine evidence alone is not an
optimizer admission.

## Frozen Proposal

Train/train artifact 512 supplies 208 direct Teacher-GPP days, eight recursive
7-day windows, and four recursive 30-day windows. The three gradients are
combined with fixed coefficients `1.0`, `5.065197169020285`, and
`4.594433600967992`. Global clipping uses the calibrated threshold
`46.802861793665286`.

The proposal loads both the specialist and AdamW state from the accepted
warm-start checkpoint. Optimizer step 250 is therefore transformed into one
ephemeral step 251 at learning rate `4e-6`. A reset optimizer is not tested:
the purpose is to admit or reject the exact continuation mechanism planned
for joint training.

## Acceptance

All source and combined gradients must be finite, nonzero, cover every
specialist element, and reach every specialist leaf. The cosine between each
source gradient and the negative realized parameter step must be strictly
positive. Direct GPP, 7-day science, and 30-day science losses must each
strictly decrease on the same artifact after the proposal.

Every declared specialist component must change by a finite nonzero amount.
Pre- and post-proposal one-day, 7-day, and 30-day physical hard-failure counts
must be zero. All retained parent parameter objects and the warm-start
checkpoint hash must remain unchanged, and the proposed specialist and
optimizer state must contain no nonfinite value.

The proposal is discarded after inspection. No checkpoint or recovery is
written. Teacher regeneration, v5, validation, sealed test, paid training,
365-day rollout, Gate D2, and product acceptance remain closed.

## Implementation

Contract loading and the frozen three-source objective are in
`daily_operator/target_oriented_structured_gpp_joint_update.py`.
`scripts/hpc/run_target_oriented_structured_gpp_joint_update_preflight.py`
loads the accepted endpoint, evaluates the three source gradients, executes
the inherited AdamW proposal, and evaluates all three post-step losses and
physical gates. The free-node boundary is
`scripts/hpc/run_target_oriented_structured_gpp_joint_update_preflight.sh`.
Focused local tests, Ruff, Python compilation, CLI import, and Bash syntax
pass.

On pass, the next task is to freeze a bounded 250-update joint-training
contract and request separate paid-resource authorization. This preflight does
not itself authorize that training.

