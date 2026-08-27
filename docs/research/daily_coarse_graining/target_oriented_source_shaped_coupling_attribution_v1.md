# Target-Oriented Source-Shaped Coupling Attribution V1

Date: 2026-08-27

Status: completed on free `gln01`; accepted result is recorded in
`target_oriented_source_shaped_coupling_attribution_result_20260828.md`

Contract SHA256:
`12b327c81851551dfd7ef5e4878b03845290d78b5d51d26168046fe25dce93b2`

## Question

Branch-only training improves 7/30-day maintenance respiration but worsens
several protected GPP/growth/NPP/biomass scores. This audit asks whether the
update-100 versus update-0 regression already exists within independent
Teacher-started days or requires predicted state to recur across days.

It does not train, rank a new architecture, or reopen the Attention backbone.

## Frozen States And Controls

Both evaluated states use the same complete 84,148,646-parameter
source-shaped Attention forward:

- `update_0`: deterministic source-shaped initialization from the fixed
  update-1000 parent;
- `update_100`: the two trained branch subtrees from Slurm job `14701638`,
  with all 83,754,404 retained coordinates unchanged.

The accepted free-rollout update-0 and update-100 screen records are reused
without reexecution. The causal intervention evaluates the same two states
with every day independently initialized from that day's prepared Teacher
day-start continuous and discrete state. The daily neural forward,
conservative updater, and retained exact tail remain unchanged. Only predicted
cross-day carry is removed.

This matched design gives two update-100 versus update-0 comparisons:

| Mode | Same-day model | Cross-day state |
| --- | --- | --- |
| Free rollout | unchanged | predicted |
| Teacher day start | unchanged | reset before every day |

## Frozen Data

The audit uses only the existing train/train evaluation groups
`251, 358, 465, 572, 679, 786, 893, 1000`, with eight 7-day and four 30-day
windows per group. It reads neither validation values nor sealed test. Teacher
v6 is reused unchanged and historical v5 is prohibited.

## Metrics And Decision

For GPP, maintenance respiration, growth respiration, autotrophic respiration,
NPP, AGB, BGB, total live biomass, and ecosystem carbon, the audit reports the
update-100 versus update-0 change in the same acceptance score used by the
frozen screen:

```text
0.5 * (normalized RMSE^2 + normalized signed bias^2)
```

A median score increase greater than `1%` is material. A material regression
that remains under Teacher-day-start execution proves a direct same-day
component. A regression present only in free rollout requires cross-day
recurrence. The report also converts normalized signed biases back to raw
cumulative units and requires the autotrophic-respiration and NPP carbon
identities to close within `1e-10`.

If maintenance improves while NPP still regresses under Teacher day starts and
the identities close, the next gate is an architecture discussion about a
jointly trainable GPP/growth/maintenance flux scope. It is not permission to
implement that architecture.

## Execution Boundary

The run creates no optimizer, applies zero updates, and writes no checkpoint.
It reconstructs no 48-record forcing sequence and executes no 48-step state
scan. A free `gln01` V100 may execute the bounded panel. Paid work, validation,
broader training, 365-day rollout, Gate D2, sealed test, and product work
remain closed.

## Local Qualification

The frozen contract loader, update-state comparison, raw carbon-bias identity,
classification, and group-order rejection tests pass. The complete related
set passes 25 tests, including the existing source-shaped forward, loss,
gradient, PFT permutation, budget, screen, checkpoint, and virtual-step
coverage. Ruff lint, formatting for the new modules, Python compilation, CLI
loading, Bash syntax, and `git diff --check` pass.

The implementation adds Teacher-day-start acceptance statistics to the
existing rollout step factory and a diagnostic runner that loads the accepted
branch-only checkpoint without creating an optimizer. The real-v6 execution
completed at commit `4c22a4c`; report SHA256 is
`e2a0b90f...26a0d6b`. See the result document for the causal classification
and next gate.
