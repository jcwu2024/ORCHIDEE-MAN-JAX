# Coupled GPP Direct-Step Transfer Audit V1

## Question

Persistent update 2 improves direct GPP on its scheduled group by `5.8362%`
but worsens the fixed monitor by `2.2197%`. The scheduled and monitor mean GPP
gradients are strongly aligned, while the applied four-source direction is
nearly orthogonal to monitor GPP and its finite scale crosses into regression.

This audit asks one bounded question before changing direction construction:
does a smaller realization of the same update-2 direction protect direct GPP
on several disjoint train/train groups and then transfer to the fixed monitor?
It is not training and cannot promote a model.

The frozen machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_direct_step_transfer_audit_v1.json`,
canonical SHA256
`b6f05589af4e8f985acf32b95853561a59ec68299bbd000ddec2438382913a61`.

## Reconstruction And Direction

The runner reconstructs the accepted update-7 state once, replays persistent
update 1, and recomputes the persistent update-2 four-source minimum-norm
direction exactly once. It does not read the persistent recovery, repeat an
old scale search, or use optimizer or momentum state. The full displacement
norm is `0.007758973351050004`; all tested candidates are in-memory scaled
realizations of that one direction.

The predecessor result is the completed direct-generalization report with
SHA256
`23327c3c4081e283ea934acd03cb2db3c0c04b7ddbf261c430fb17b219e51cd9`.
Its hash identifies immutable accepted evidence; no cross-GPU numerical value
is required to be bit-identical.

## Frozen Train-Only Selection

The scheduled update-2 artifacts are
`317,657,366,681,671,312,993,984`. Four guard groups come from persistent
schedule updates 11 through 14 and were unused by updates 1 through 10:

```text
337,372,678,285,816,469,511,405
567,658,962,267,865,566,496,339
755,923,517,943,430,296,653,400
482,975,601,853,393,845,641,842
```

The descending displacement scales are
`1/2,1/4,1/8,1/16,1/32,1/64`. A scale passes only when:

1. all four scheduled-group source losses pass the existing finite-step rule;
2. the scheduled physical hard gate passes;
3. every guard-group mean direct-GPP relative change is at most `1e-6`; and
4. the combined 32-artifact guard mean decreases strictly.

The largest passing scale is selected. The fixed monitor artifacts
`251,358,465,572,679,786,893,1000` are not read during this selection. Only
after selection, the runner evaluates direct GPP before and after that one
selected scale on the monitor. If no train-only scale passes, the monitor is
not read.

Possible classifications are:

- `train_only_scale_transfers_to_monitor`;
- `train_only_scale_fails_monitor`;
- `no_train_only_shared_finite_scale`.

## Boundary And Next Decision

No parameter proposal, optimizer update, recovery, or checkpoint is retained.
Only accepted v6 train/train data are allowed. Fresh validation, historical
v5, sealed test, Teacher regeneration, 365-day rollout, Gate D2, 48-record
forcing reconstruction, and a 48-step state scan remain forbidden.

One free `gln01` execution is allowed with a 120-minute timeout. Paid
partitions are not authorized. A transferred scale opens a separately frozen
multi-group GPP-protected backtracking smoke. Either failure classification
requires replacing the single scheduled-group GPP constraint with
multi-group direct-GPP direction constraints before any continuation.

## Implementation

- Contract:
  `manifests/coarse_graining/target_oriented_coupled_gpp_direct_step_transfer_audit_v1.json`
- Contract logic:
  `research/daily_coarse_graining/daily_operator/target_oriented_coupled_gpp_direct_step_transfer.py`
- Runner:
  `scripts/hpc/audit_target_oriented_coupled_gpp_direct_step_transfer.py`
- Free-node launcher:
  `scripts/hpc/run_target_oriented_coupled_gpp_direct_step_transfer.sh`
