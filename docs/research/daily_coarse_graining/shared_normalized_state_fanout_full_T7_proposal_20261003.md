# Shared normalized-state native fan-in: full public T7 preparation

Status: **prepared diagnostic retained; real T7 precision expansion paused by
the user on 2026-10-03; no GPU execution is authorized**.
The preceding FP32 public-cotangent capture is closed as a strict numerical
bridge failure. Its hidden partials are not admitted reference evidence. This proposal
uses the separately admitted local normalized-state component mechanism; the
actual full public T7 mechanism remains unaccepted. It does not
restart the closed projected four-key/value bank policy.

## Exact candidate and public workload

Use immutable checkout `c599236bcf57b745e9bafed678298010f4c3b58f` at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`.
Retain original update0 weights, all 300 parameter leaves /64,884,481
coordinates, all seven days, all3,854 initial-state gradients, and the shared
public `(flux+stock)/9` objective with native checkpointed AD and
`argnums=(0,1)`. Retain original FP32 neural forward, FP64 physical computation,
expert chunk128, masks, prepared inputs and native public loss/adjoint.

Immediately after the original FP32 state normalization, insert one native
`jnp.asarray(state, dtype=jnp.float64)`. The original one field and seven axis
summary consumers still narrow/reduce FP32. Native reverse widens the eight
FP32 partials, accumulates FP64, then narrows once into the original FP32
normalization. No delta inputs, custom VJP, changed decoder banks, changed
tolerance, reference adoption, optimizer update or training is introduced.

Source pinning checks the original `_masked_summary`957–973,
`_mask_inactive_pft_elements`984–990,
`assemble_strong_baseline_context`1100–1355 and
`input_conditioning.standardize_continuous`12–52 against the declared checkout.
Eight `jax.named_scope` wrappers identify the original consumers in metadata.
Removing those wrappers and the single widen restores the entire original
assembler AST exactly; original calls, arguments and control flow are retained.
The candidate alias is restored after lowering/compilation.

## Run bounds and terminal rules

The proposed command, **only after new user authorization**, is:

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/normalized-state-fanout-full-T7-20261003/run_normalized_state_fanout_full_T7.sh --gpu 1
```

Use one free `gln01` physical GPU1 on `ibc13b02n04`, outside paid Slurm. Whole
dispatch3600s, main3590s, kill grace10s; CNY0, worst-case1 GPU-card-hour. The Bash
entry records the first origin before Python imports. Stage maxima are90s
deployed preflight,300s assets plus original saved-executable replay,3100s one
combined lower/backend attempt,90s candidate first raw/binary/protobuf saves,
and180s three interleaved pairs. Every cap is clipped to the original deadline;
these maxima are not additive promises. The budget uses measured full-graph
lower193.519s/backend2508.717s and original full-AD histories, not the component's
approximately5s compile.

At most one candidate graph/backend attempt is allowed. The original executable
is loaded without reference recompilation. The launcher records node, CPU
affinity/load, visible GPU processes, memory and elapsed card-hours. Original
and candidate must pass ABI, finite values, physical counts, source masks and
24GiB estimated/observed memory. A hard/resource/budget failure terminates.
Ordinary precompile interface repair can resume only within the exact first
origin, with policy bytes unchanged; an attempted graph, saved candidate result
or scientific/hard terminal denies retry. No paid fallback or old bank restart.

## Numerical and timing evidence

Save complete `original-first.pkl` before original hard checks and complete
`candidate-first.pkl` before candidate hard/strict checks. Compare the public
loss, all19 auxiliary outputs, all64,884,481 parameter coordinates and all3,854
state coordinates with existing strict rules and no waiver. Historical saved
result drift is reported, not substituted for the same-window original arm.

If hard/resource checks pass, a strict gradient discrepancy **does not suppress**
the three remaining pairs: reference/candidate, candidate/reference,
reference/candidate. Pair comparisons and within-arm drift distinguish persistent
cross-arm failure from repeat variability. Strict failures still reject numerical
admission. Repeats retain compact loss/aux/state results and full parameter
comparison evidence computed in memory, avoiding repeated full parameter-tree
saves. Insufficient remaining budget reports incomplete measurement.

Report all pair times/ratios and medians. There is no speed pass threshold.
This T7 diagnostic does not prove complete optimizer-update speed, production
adoption or final-method acceptance.

## Native mechanism certificate and its limits

Save the actual first candidate native executable binary and optimized HLO
protobuf before repeat timings, without rendering huge HLO text or recompiling.
Record the public-state output leaf index and ABI. A compact offline reader
binds the selected public state ROOT, follows supported tuple/GTE/call/fusion
bindings, and checks a source-bound target F64→F32 conversion, static3854
vectors, eight distinct FP32→FP64 partials, seven distinct FP64 adds and exactly
the field plus axis0…6 consumer scopes. Identity includes the instruction ID
and complete enclosing call context, so a shared callee SSA in separate calls
is distinguished. The final narrow must have an immediate ROOT-reachable vector
transpose use source-bound to `input_conditioning.standardize_continuous`12–52
in that same call context. Lost/fused-away or call-split normalization reverse
continuations are explicitly uncertified; no generic continuation is inferred.
Native protobuf parsing uses official OpenXLA field schemas;
literals and unrelated full-graph text are not rendered.

Only an explicit small numeric/dataflow opcode allowlist is traversed outside
the supported tuple/call bindings. Unknown opcodes, select/async/side-effect
boundaries, active SSA cycles, unknown tuple/called/custom-call/command-buffer paths, lost source/consumer
metadata, unexpected optimized layouts, duplicate identities or dynamic shapes
fail closed as `local_pattern_only_uncertified`. This is a bounded positive
source/shape/ROOT certificate, not occurrence-exhaustive coverage or proof of
gradient causation. Scoped vector definitions stop at F32 producers/external
dependencies; they do not imply closure of the whole public graph. Numerical
acceptance remains separate. **The actual GPU full-T7 certificate is pending**;
CPU checks cannot guarantee the future optimized GPU layout will be supported.
The separate ROOT reachability walk may encounter upstream neural `dot`,
`custom-call` or `select` operations outside its supported allowlist and leave
the certificate uncertified. The GPU runner only saves the protobuf; this
offline reader is not an execution gate and cannot suppress the bounded
numerical comparisons or three pairs. A numerical pass with an uncertified
mechanism permits no automatic adoption.

## Preparation record and ownership

Implementation: `scripts/hpc/normalized_state_fanout_full_T7.py`,
`probe_normalized_state_fanout_full_T7.py`,
`run_normalized_state_fanout_full_T7.py`,
`run_normalized_state_fanout_full_T7.sh`, and
`normalized_state_fanout_proto.py`; reuse the exact component policy bytes as a
source-pinning dependency. Older bank/forward/loop overlays provide immutable
asset and runtime scaffolding only.

Preparation evidence is retained in
`outputs/diagnostics/normalized-state-fanout-full-T7-preparation-20261003/` and
the matching remote runtime output directory. It contains AST/CLI/source
checks, tiny CPU native AD/protobuf interface plus nine rejected negative
controls (including unknown opcode and active SSA cycle), target Bash syntax and
actual full/resume dry-run expansion. No real model, Teacher data, GPU call,
training or paid job is used by preparation checks. Local newer JAX may omit
traditional source metadata; the reader rejects rather than infers it. The
deployed CPU JAX0.4.38 interface check is the relevant preparation compatibility
record, still not deployed GPU acceptance. `dependency-bytes.json` records all
six launcher sources present and exact component bytes matching the new overlay,
the original closed component transfer file, and the locally pinned source.
`final-source-review.json` records an independent bounded source-review pass,
reusing the completed CPU evidence and retaining the future GPU-certificate
limitations above.

The new candidate output family is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/normalized-state-fanout-full-T7-20261003`.
The root agent reviews this concrete prepared route and requests authorization
for a new finite free-GPU window. This document grants no GPU authorization.
