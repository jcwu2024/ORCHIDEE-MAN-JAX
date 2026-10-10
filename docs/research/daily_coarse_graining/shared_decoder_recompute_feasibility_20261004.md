# Shared decoder recompute feasibility

**Disposition:** no new decoder execution change is justified from current
source and saved IR. This is a bounded feasibility audit; no model, fixture,
GPU or training execution was performed.

The current priority is the original T30 route, measured at about 6.76s or
roughly 61% of the 11.087s candidate update. The matched T30 forward/combined
component medians are 1.196103s and 6.714903s. Their 5.518800s difference is
incremental native AD cost (82.19% of the combined component), including
reverse work and checkpoint recomputation. It does not isolate pure reverse
or identify a decoder operation that is redundantly recomputed.

## Source findings

At `rollout_loss.py::RolloutLoss._scan` (lines79-91), checkpointed runs wrap
the complete transition in `jax.checkpoint` before `lax.scan`. The public T7
and T30 objectives bind this checkpointed loss in
`shared_recurrent_training.py::make_shared_recurrent_objective` (lines152-246).
There is no decoder-local checkpoint boundary in the current shared decoder
path. Consequently, an inner save/recompute policy cannot be credited with a
speed benefit from a helper-only fixture; its effect must be checked through
the actual outer transition and complete public objective.

The shared decoder already reuses common family, process and carbon K/V
projections per expert and block (`shared_transformer.py::_decoder_projection_cache`,
lines379-422). Dynamic axis and local K/V are cached per field and block
(`_field_dynamic_projection_cache`, lines424-448), then gathered for coordinate
rows. Those dynamic tokens depend on field ownership and day-varying state or
inputs. The two coordinate blocks are sequential: each block's attention,
normalization and FFN update the query consumed by the next block
(`_run_coordinate_blocks_reusing_projections`, lines537-607). This audit found
no identical within-transition projection computation left to hoist without
changing its input identity or existing cache scope.

The current field path maps fixed-size row chunks through both blocks and the
readout (`_decode_field`, lines1141-1323). The separate expert128 diagnostic
groups rows and retains 128-row chunks (`shared_decoder_macro_packed.py`,
lines25-76). Its saved native-VJP HLO confirms that the learned body keeps 128
rows and width384, but the raw-gather-only relocation did not improve forward
or native-VJP medians and failed strict parameter/raw-input derivative checks.
Changing map batching further is not justified as an order-preserving shortcut:
saved chunk-vmap/group results show learned contraction layouts or accumulation
changed, with strict gradient failures. Existing group/vmap repairs and the
closed temporal-unroll screen are not new candidates.

The remaining mapped decoder work includes query-dependent attention query/output
projections and the two FFNs; normalization also repeats on gathered keys.
K/V projection reuse across fields has already been exercised by the U4 union
route: its micro4 warm ratio
was only 1.01355x, with estimated live memory about12.8% higher. That is
recorded prior evidence, not a new candidate or a T30 speed claim.

Key normalization before gather is also excluded as a new candidate: it
overlaps the existing N4/N7 whole-bank normalization family. N7's complete T7
checkpointed AD comparison gave a 1.039489x median but failed seven state
coordinates and retained persistent differences beyond within-arm replay.
The named-dot checkpoint save-policy route is closed as well: its T7 dual-AD
speed ratio was about1.048x, estimated live storage rose about1.94x, and five
of3,854 initial-state derivatives failed the strict gate. The separate
`prevent_cse=False` screen changed compiled structure but showed only a
1.014791x overall representative T7 ratio with reversal by execution order and strict
derivative failures. These results block a renamed continuation of those
families; they do not establish that every future rematerialization policy is
ineffective.

## Decision and next measurement

Do not add an opt-in decoder adapter now. The source does not expose duplicated
same-input work inside the active cache scope; the saved gather relocation had
no speed benefit; the measured batching alternatives change learned reduction
execution; and current T30 timing does not distinguish reverse from recompute.
No synthetic helper comparison could establish that an internal remat policy
survives the public outer checkpoint boundary, so no CPU fixture was run.

The one prepared alternative is an offline census of historic HLO metadata
attributed to `_stable_head_l2_normalize` in the existing optimized native-VJP
HLO for the original 128-row component. Its source-line attribution refers to
the saved runtime source metadata at `strong_baseline_architecture.py` lines
901-911; it is not proof about current-source instruction placement. It reads
no model inputs and compiles nothing. Run from the project root:

```powershell
conda run -n ORCJAX python -B scripts/hpc/census_shared_decoder_saved_hlo.py --hlo outputs/diagnostics/shared-command-buffer-loops-20261001/shared-command-buffer-loops-threshold-component-default-20261001.default.native_vjp.hlo.txt --output outputs/diagnostics/shared-decoder-recompute-feasibility-20261004/original-native-vjp-normalization-census.json
```

The reader records the HLO hash and counts instructions attributed to those
historical source lines by optimized-HLO loop phase. Stop without inference if
the saved input is missing, unreadable, has an unexpected schema, or yields
zero attributed matches. Its limits are explicit in the output: metadata
occurrences are not execution or recompute counts, timings, memory estimates
or proof of loop-invariant execution. This is not a speed implementation. A future checkpoint policy
decision requires residual evidence at the public outer T30 boundary and a
separately authorized actual-runtime comparison; no such run is currently
selected. Any eventual full-update claim still requires daily/T7/T30, full
derivatives, masks and physical gates.

Graph evidence supplied for this audit is generation
`2026-10-04T01:39:49Z`. It covers `shared_transformer.py` and
`shared_decoder_expert_execution.py`, but reports `metadata_changed`; a fresh
coverage query for all five direct-source paths failed with `Transport closed`.
Direct current-source reads and saved local IR/results are the basis here.
No graph-completeness claim is made. Paths, source hashes and checked claims
are recorded in
`outputs/diagnostics/shared-decoder-recompute-feasibility-20261004/source-audit.json`.
