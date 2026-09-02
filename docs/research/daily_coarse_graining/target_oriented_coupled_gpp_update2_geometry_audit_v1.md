# Coupled-GPP Update-2 Geometry Audit

Date: 2026-09-02

Status: completed accepted zero-training audit

Machine contract SHA256:
`7585cc5b46a76fb19d08703c8e88839f8758258888a1f302d5448dbf7e79f198`.

## Question

Paid bounded training retained update 1 and rejected update 2 because direct
GPP regressed `2.63633%` even though the weighted objective and the other
three source losses improved. The raw direct-GPP gradient still had positive
cosine with the combined gradient, but that does not prove that inherited
AdamW moments produced a direct-GPP descent step. This audit distinguishes
optimizer-moment rotation from finite-step curvature before any objective,
architecture, or training-policy change.

## Exact Parent And Data

The audit loads the unchanged recovery from job `14903328`, requiring exactly
one completed update and optimizer step 1. It reproduces the rejected update-2
batch `326, 935, 542, 913, 897, 836, 887, 863`, covering 1,664 direct-GPP
Teacher days, 256 one-day state transitions, 64 seven-day windows, and 32
thirty-day windows. The frozen parent report and rejected-proposal SHA256
values are part of the contract.

No fresh validation target, sealed test, v5 row, regenerated Teacher datum,
48-record forcing reconstruction, or subdaily scan is allowed.

## Directions And Finite Steps

The four source gradients and their existing weighted sum are computed once.
Two optimizer displacements are then compared from the same update-1
parameters:

- inherited AdamW uses the exact first and second moments stored by update 1;
- reset AdamW uses zero moments but the same combined gradient, learning rate
  `2e-6`, clipping threshold `12.808193184117403`, and weight decay.

For each displacement, source-gradient cosines are measured against
`parameters_before - parameters_proposed`. The finite scale grid is exactly
`1, 1/2, 1/4, 1/8, 1/16`. Every scale is evaluated on all eight update-2
artifacts. A scale passes only if the weighted objective strictly decreases,
all four source losses do not increase beyond relative tolerance `1e-6`, and
all physical hard counts remain zero.

The inherited scale-one result must remain rejected by the same frozen
finite-step rule. Its four relative changes are also compared with the parent
report at absolute reference tolerance `1e-8`, but that numeric comparison is
reporting-only: different legal GPU reduction order must not abort the audit.

## Retention And Decision

All parameter proposals and both optimizer proposals are discarded. The
parent recovery must retain the same size and modification time; the audit
writes only one JSON report. No checkpoint is written and no continuation is
authorized by the audit alone.

If an inherited scale passes, the next gate is to freeze a backtracking
training amendment. If only reset AdamW passes, the next gate is to freeze an
optimizer-reset amendment. If neither passes, training remains closed and the
four-source direction or preconditioning must be reviewed. The audit does not
read the 96-window validation panel and cannot promote the candidate.

The first free execution at commit `cc86e44` completed the full update-2
gradient and inherited scale-one forward, then stopped because the original
contract incorrectly made a `1e-10` numeric-reference comparison fatal. No
proposal, recovery, checkpoint, or report was written. The corrected local
contract, classification tests, runner compilation, and shell syntax checks
passed before the bounded free-`gln01` rerun. The corrected execution completed
at commit `b7697f5`; the accepted result is recorded in
[`target_oriented_coupled_gpp_update2_geometry_result_20260902.md`](target_oriented_coupled_gpp_update2_geometry_result_20260902.md).
No paid partition was authorized.
