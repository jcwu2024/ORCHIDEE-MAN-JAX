# Hierarchical Coordinate-Query Attention v3 DOC Audit

Date: 2026-08-20.

Status: the bounded litter/POC/DOC memorization run completed normally but
missed its frozen absolute gate. The follow-up target audit passed. No
checkpoint is eligible for retention or promotion, and the next architecture
change requires discussion with the user.

## Memorization Result

Approved paid job `14619803` trained
`hierarchical_coordinate_query_attention_v3` on the frozen 364-day
train/train point-year at commit
`f40b77d40c392132212ce478365e7971555e59a8`. It used the accepted v6 release
only and did not regenerate Teacher data, join v5 data, reconstruct half-hour
forcing, or scan 48 substeps during candidate inference.

The job completed 100 epochs and 4,600 finite AdamW updates in `05:34:10`.
Selected litter/POC/DOC loss fell from `0.1895335` to `0.00269069`, a
`98.5804%` reduction. All gradient-coverage, count, state, budget, inventory,
and finite-value gates passed. The frozen gate nevertheless required at least
`99%` aggregate reduction and at least `95%` reduction, or final loss at most
`1e-4`, for every initially dominant field. It passed `25/28` dominant fields
and failed these three:

| Field | Reduction |
| --- | ---: |
| `inventory.carbon.DOC.incoming_amount` | `-0.3085%` |
| `typed.ok_leak.doc_free_to_adsorbed` | `83.2158%` |
| `typed.ok_leak.doc_advective_interface` | `93.5753%` |

This is a large improvement over the rejected v2 mapping's `91.8014%` and
`16/28` dominant-field pass count, but it is not an accepted process head.
No checkpoint was written. The formal report is at
`runtime/outputs/training/hierarchical-coordinate-attention-litter-100-epoch-f40b77d/report.json`
on Explore1000 and has SHA256
`12dfe55052477bdffb23c0e49f930a79322e3680d125be0ca5eb5db6617b10bd`.

## Target Audit

The zero-training audit at commit `0f82d1f` read the same frozen point-year on
free node `test02`. It inspected target sparsity, normalized magnitude,
per-coordinate constant baselines, daily activity, daily cross-field
correlation, and the DOC inventory endpoint partition. It wrote no checkpoint
and performed no training.

The three failed targets are sparse across physical coordinates but are not
rare in time:

| Field | Active days | Meaningful cell fraction | Active coordinates |
| --- | ---: | ---: | ---: |
| DOC incoming amount | `364/364` | `9.8769%` | `131/896` |
| DOC free-to-adsorbed | `364/364` | `29.4165%` | `66/224` |
| DOC advective interface | `364/364` | `29.3257%` | `66/224` |

Per-coordinate constant means leave `77.37%`, `91.00%`, and `54.49%` of the
respective zero-predictor squared error. Static coordinate identity therefore
does not explain the targets, especially free-to-adsorbed transfer.

Daily normalized target intensity is strongly related across the failed
fields:

| Pair | Signed daily-mean correlation | Absolute daily-mean correlation |
| --- | ---: | ---: |
| incoming vs free-to-adsorbed | `0.8782` | `0.9597` |
| incoming vs advective | `0.8892` | `0.8892` |
| free-to-adsorbed vs advective | `0.8620` | `0.8403` |

The endpoint encoding is valid: incoming amount and outgoing fraction are
mutually exclusive at every admitted coordinate, incoming amount is never
negative, and outgoing fraction stays in `[0, 1]`.

The audit report is at
`runtime/outputs/training/hierarchical-doc-target-audit-0f82d1f/report.json`
on Explore1000 and has SHA256
`6331b736f97e44a2443da1d17026f3f8711df895d5da316950ae009f7b87587b`.

## Interpretation And Active Gate

The current candidate already gives every output coordinate a local query,
but it sends only field-balanced pooled hidden contexts from DOC reaction and
transport into the inventory stage. It does not send coordinate-aligned
predicted DOC transfers. The audit shows that this missing relationship is a
credible remaining bottleneck: the failed fields vary every day together,
while a static coordinate predictor cannot account for most of their error.

The next gate is an architecture discussion with the user. The discussion
should decide whether to replace the pooled DOC handoff with a bounded,
source-routed coordinate coupling and whether the mutually exclusive DOC
inventory endpoints should use one joint transition representation. Do not
implement that revision, repeat the 100-epoch run, open multi-point training,
start rollout, read sealed test data, or advance to Gate D2 before that
decision.
