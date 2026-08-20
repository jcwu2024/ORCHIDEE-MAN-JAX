# DOC Endpoint Target/Loss Geometry Audit Result

Date: 2026-08-20.

Status: the zero-training audit passed its integrity checks and rejects the v4
joint signed DOC endpoint as a training coordinate. No architecture is
promoted. The next gate is user review and freeze of a revised DOC endpoint
mechanism before implementation.

## Execution

The audit ran on free Explore1000 CPU test node `test01` at commit
`143f4b64b225593272e2af271cb386104879f5eb`. It read all 364 valid train/train
days for point `001.0-071.0`, year 1961, from the accepted v6 release. It made
zero training updates, initialized no network, wrote no checkpoint,
regenerated no Teacher data, used no v5 data, read no sealed-test rows,
reconstructed no 48-record forcing, and executed no 48-step state scan.

The report is at
`runtime/outputs/training/doc-endpoint-target-loss-geometry-audit-v1/audit_report.json`
on Explore1000 and
`outputs/daily_coarse_graining/doc-endpoint-target-loss-geometry-audit-v1/audit_report.json`
locally. Both copies have SHA256
`890b678f99be3e5b2479ae2599cd75facc2fbae0bd913389a9633bc1a0d4bba1`.
The frozen protocol manifest has SHA256
`4f5003f3750928b3aa9119b56045d22337bd253ac758c85c690523a935b7d6e7`.

## Target Population

All `326,144` declared DOC coordinates are finite, physically valid, and
admitted. The partition is:

| Status | Count | Fraction of all coordinates |
| --- | ---: | ---: |
| Incoming | `24,412` | `7.4850%` |
| Unchanged | `285,789` | `87.6266%` |
| Outgoing | `15,943` | `4.8883%` |

No coordinate has simultaneous incoming and outgoing activity. Among the
`40,355` active transitions, `60.4931%` are incoming and `39.5069%` are
outgoing. Day-end DOC is exactly zero in `270,088` coordinates (`82.8125%`).

## Findings

The v4 joint transform is algebraically correct. Every active target has a
finite inverse, and finite reconstruction has maximum absolute error
`8.88e-16`. Its typical branch derivatives are also balanced: median
loss-normalized Jacobians are `0.4730` for incoming and `0.5340` for outgoing,
with ratio `0.8857`. The failure is therefore not an inverse bug or a broad
incoming-versus-outgoing Jacobian-scale mismatch.

The joint coordinate still fails the frozen conditioning gate because 110
active targets have normalized Jacobian at or below `1e-6`. More importantly,
the two losses coupled through the same scalar are severely unequal. Under a
zero predictor, incoming contributes `97.6209%` of their combined squared
error and outgoing only `2.3791%`; their per-field normalized MSE values are
`0.0489251` and `0.00119234`. The same imbalance is present around the v4
zero-raw center. This explains the formal run: the shared scalar had a much
larger incentive to repair incoming and could sacrifice outgoing.

Simply restoring the v3 independent softplus/logit heads is not an adequate
repair. Their active inverse targets are finite, but 563 active targets have
normalized Jacobian at or below `1e-6`, compared with 110 for the joint head.
Their median normalized Jacobians are only `0.01826` and `0.01805`. Separate
heads remove direct competition but restore the small-target gradient
starvation already observed in v3.

An explicit incoming/unchanged/outgoing status plus branch-native,
loss-scaled magnitude has finite targets for all `326,144` coordinates. The
active magnitude medians are closely matched at `0.01843` incoming and
`0.01838` outgoing and do not acquire an additional softplus/logit Jacobian.
This representation is a shortlist result, not an accepted network design;
its large unchanged class and exact classification/magnitude weighting still
need a frozen mechanism and loss contract.

## Decision

Retain v4's coordinate-aligned five-token DOC Attention evidence. Reject its
one-scalar signed endpoint head. Do not treat the v3 separate heads as a
sufficient fallback. Before changing code, discuss and freeze one revised
mechanism that specifies:

1. the exact incoming/unchanged/outgoing target and conservative
   materialization;
2. branch-specific magnitude coordinates and boundary behavior;
3. class and branch loss balancing that prevents the unchanged class or
   incoming field from dominating;
4. absolute head-only and field-level early-stop gates.

After local forward/loss/gradient and one-day state/budget tests, any new head
must pass a 5-10 epoch no-checkpoint screen before a final 100-epoch
memorization gate is considered. Multi-point training, rollout, sealed tests,
Gate D2, and product acceptance remain closed.
