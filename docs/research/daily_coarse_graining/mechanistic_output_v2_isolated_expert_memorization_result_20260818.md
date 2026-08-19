# Mechanistic Output V2 Isolated-Expert Memorization Result

Date: 2026-08-18.

Status: completed bounded diagnostic. Both executions completed without a
runtime failure, but neither selected expert passed the frozen aggregate
memorization gate. No checkpoint is eligible for retention or promotion.

## Question And Boundary

The frozen protocol tested whether the existing 49,743,001-parameter
mechanistic-output v2 mapping could memorize either of its two dominant
residual process families when trained without the other six expert losses:

1. `litter_poc_doc`;
2. `surface_energy`.

Both runs used the same fresh initialization, the same 364 train/train days
from landpoint `001.0-071.0` in 1961, batch size 8, 100 complete shuffled
epochs, 4,600 AdamW updates, and the existing warmup/cosine schedule. The
complete 181-field forward boundary, conservative update, retained exact tail,
science diagnostics, and hard state/budget checks remained active. The v6
Teacher data were not regenerated, v5 data were not used, and no 48-record
forcing reconstruction or 48-step state scan was introduced.

The source checkout remained at `806cffd1293f65ecf19931de5c7d98261987fad1`.
The exact diagnostic runner used for the accepted runs had SHA256
`156fcc38fb8f31fabebdf77b89da57b3f83507e4a96c621b96c3294cf1ad90d7`.
It kept the ordinary training step free of coverage reductions and evaluated
gradient/update coverage only at fixed initialization and final probes.

## Smoke Admission

The corrected one-epoch smokes completed all 46 updates and passed the frozen
execution, loss-decrease, count, coverage, and hard gates:

| Expert | Initial loss | Final loss | Reduction | Epoch seconds | Report SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| `litter_poc_doc` | 0.1449571591 | 0.1019792449 | 29.6487% | 392.84 | `66eff2d0...09ba97` |
| `surface_energy` | 0.3957967121 | 0.3135248298 | 20.7864% | 397.59 | `a19d9831...d359` |

The first epoch includes one-time compilation. For comparison, the earlier
ordinary 100-epoch mechanistic-output v2 run had a 12.54-second median epoch
after compilation. Reusing each smoke's persistent compilation cache kept the
full diagnostics bounded.

## Full Results

| Gate item | `litter_poc_doc` | `surface_energy` |
| --- | ---: | ---: |
| Completed epochs / updates | 100 / 4,600 | 100 / 4,600 |
| Initial selected-expert loss | 0.1449571591 | 0.3957967121 |
| Final selected-expert loss | 0.0118844473 | 0.0082090925 |
| Final reduction | 91.8014% | 97.9259% |
| Required reduction | 99% | 99% |
| Best evaluated reduction | 92.9891% | 97.9259% |
| Dominant fields | 28 | 11 |
| Dominant field gate | 16 pass / 12 fail | 11 pass / 0 fail |
| Selected block/head coverage | pass | pass |
| Count identity | pass | pass |
| Sum of all hard failures | 0 | 0 |
| Checkpoint written | no | no |
| Wall time | 2,535.73 s | 1,965.03 s |

The full report SHA256 values are:

- `litter_poc_doc`:
  `bc9feaabe6c800d6dd369091c470bdaa960e9c090ae07b6125511f7ff334074d`;
- `surface_energy`:
  `f72530053337c82ee74658b613f7bbbfa61a40dfa5a4ee5023e48e52a4400d4f`.

## Field Findings

All 11 dominant surface-energy fields passed the field-local gate. The weakest
was `typed.energy.soilflx`, which fell 95.3364%; the remaining dominant fields
fell 96.9172% to 98.6018%. Surface energy is therefore close to the aggregate
gate and does not show one isolated field that the current coordinates cannot
fit. It nevertheless misses the preregistered 99% aggregate threshold by
1.0741 percentage points within the fixed budget.

The litter/POC/DOC branch is qualitatively different. Twelve of 28 dominant
fields fail:

| Field | Reduction |
| --- | ---: |
| `inventory.carbon.litter_above.incoming_amount` | 5.6326% |
| `endpoint.ok_leak_aux.lignin_struc_below` | 26.4348% |
| `typed.ok_leak.doc_free_to_adsorbed` | 35.2435% |
| `inventory.carbon.DOC.incoming_amount` | 43.0678% |
| `endpoint.ok_leak_aux.litterpart` | 58.0175% |
| `endpoint.ok_leak_aux.dead_leaves` | 59.9693% |
| `typed.ok_leak.doc_advective_interface` | 83.9400% |
| `endpoint.ok_leak_aux.fuel_1hr` | 92.3827% |
| `endpoint.ok_leak_aux.lignin_struc_above` | 92.5892% |
| `endpoint.ok_leak_aux.fuel_100hr` | 94.6135% |
| `endpoint.ok_leak_aux.fuel_1000hr` | 94.6257% |
| `endpoint.ok_leak_aux.fuel_10hr` | 94.6798% |

Other fields in the same expert do fit, including litter respiration,
litter-below outgoing fraction, DOC drainage, and litter-to-DOC transfer.
This is not a disconnected-gradient failure: selected expert blocks and
selected label heads have nonzero gradients and realized AdamW updates at both
coverage probes, while unselected expert blocks and label heads remain exactly
zero as required. The failure is concentrated in the coupled litter endpoint,
incoming-inventory, adsorption, and advective-transfer mappings.

## Decision

Joint equal-expert interference is not the whole explanation. Removing the
other six objectives improves both families substantially, but the existing
mapping still cannot meet the absolute memorization contract within the fixed
budget. The litter/POC/DOC branch provides the decisive architecture evidence;
surface energy is a near-threshold optimization result with all field gates
already passing.

The active gate is now an architecture discussion with the user. The next
candidate should address the litter/POC/DOC branch's coupled endpoint and
transfer structure while preserving the accepted mechanistic-output v2
boundary, conservative update, retained exact tail, and axis-preserving
Attention foundation unless the discussion explicitly changes them. No new
network implementation or training is authorized before that discussion.

Do not repeat these runs, extend their training budget, start a paid complete
run, open 64-point-year training, begin rollout, read sealed test data, or
advance to Gate D2 or product acceptance. No checkpoint exists to continue.
