# Split decoder litter/DOC five-field grouping screen

Status: grouped128 and field-local128 scalar/micro4 GPU parity passed;
256 schedules have mode-specific output failures and are not selected,
2026-09-25. All probes ended. The accepted split model,
parameter tree, masks, output capacity, losses and training runner are
unchanged. This work does not compile or run the full model.

## Audited scope

The static layout audit in `outputs/shared-decoder-packing-audit-20260923.json`
identifies five `litter_poc_doc` fields with the same decoder shape: field
indices 64, 78, 79, 80 and 81 (`typed.ok_leak.litter_to_doc`,
`doc_run_2_peat`, `doc_free_to_adsorbed`, `doc_advective_interface` and
`doc_diffusive_interface`). Each has 448 eligible coordinates, seven
production chunks of 64, four axis slots, 42 field-local axis-bank entries,
one local-bank entry, and no raw-carbon bank. All share expert index 5 and
the 20-family source bank. Field 78 has a different axis order from the
other four; the 448-row probe obtains each field's actual inverse axis/local
gather indices from `_decoder_dynamic_plan`, not a common synthetic order.
They are exported locally to compact JSON and read by the deployed probe;
the remote checkout does not rebuild the static layout.

Every local-bank entry maps to state index `-1`, so its attention mask is
always false. The 448-row fixture enforces this. It still projects the
field-local bank in both paths, matching the current helper boundary; the
masked local payload has no direct attention gradient. Axis attention-mask
values are mixed synthetic booleans because actual runtime context is not
loaded. The source-defined axis gather indices and mask shapes are real;
the synthetic values are a limitation for performance attribution. The
small 73-row fixture deliberately uses mixed local masks to stress native
gradients and padded chunks; it is not claimed to be the paper-run layout.

## Compared executions

`reference64` retains five independent `lax.map` calls, each over seven
64-query chunks. Each chunk invokes the production
`_run_coordinate_blocks_reusing_projections` two-block attention/FFN helper
and the common 256-wide readout. The same-expert candidate joins the five
fields into one map with distinct per-field dynamic tokens, masks and
projected axis/local banks. Offset gather indices preserve each bank's
identity; final slices restore all five 448-coordinate outputs. It compares
`grouped64` (same chunk size, graph organization only), `grouped128` and
`grouped256` (larger row contractions with bounded padding). Inactive and
padded rows stay masked and their outputs are zero. All neural payloads and
parameters are FP32, including nonzero K/V biases. Native JAX AD covers
every parameter and input payload; no custom reverse is used.

The default probe runs only those four variants, writing the JSON report
before and after each compile and after each parity check. It stops on the
first strict numerical failure or a configured compiler-temporary limit.
Optional `fieldlocal128`/`fieldlocal256` modes keep five separate maps at a
larger chunk size; they are reserved for a later controlled attribution if
grouped128/256 look useful, not part of the initial four-variant budget.

## Local admission

Four focused tests pass in 45.12 s under `ORCJAX`, including canonical
static mapping checks, JSON-only loading with the static builder patched
to raise, and scalar/micro4 rows73 with nondivisible chunk
padding. `reference64` and all three grouped variants agree at the existing
rtol `1e-5`/atol `1e-6` pointwise gate: zero failures among 366 scalar or
1,461 micro4 loss/output coordinates, and 16,465 scalar or 45,313 micro4
parameter/input-gradient coordinates. Ruff passes. These small-width
results prove implementation parity under mixed masks and padding, not
deployed performance or full-width numerical acceptance. No expensive
width384 CPU compilation was run.

The overlay files are
`research/daily_coarse_graining/daily_operator/shared_decoder_carbon_group.py`
and `scripts/hpc/probe_shared_decoder_carbon_group.py`; the regression test
is `tests/unit/test_shared_decoder_carbon_group.py`. The candidate module is
loaded from an explicit `--candidate-module-file` path against immutable
checkout `c599236`. `--require-gpu` rejects accidental CPU execution.
Run scalar and micro4 as separate bounded free-GPU tasks, with
`--width 384 --rows 448 --microbatch 1|4 --repeats 20`,
`--layout-metadata-file runtime/transfer/shared-decoder-carbon-layout-20260925.json`, separate report
paths under `runtime/outputs/diagnostics/`, and a 10-minute limit per mode.
The runner should preserve the incremental report on interruption. No paid
allocation or long training is implied.

The actual launcher is `scripts/hpc/run_shared_decoder_carbon_group.sh`,
staged under `runtime/transfer/` alongside the two Python overlays. Free
GPU0 scalar wrapper PID58499 and GPU1 micro4 wrapper PID58551 used the
accepted foundationpose container and immutable c599236 checkout. Both GPUs
were empty immediately before launch; host load and affinity are logged.
Initial wrappers58268/58293 exited before Python because the old Bash
interpreted an empty array as unset under `set -u`. The default argument
array is now nonempty; all three default/control argument branches passed
actual target-shell dry-run before relaunch. Original logs are preserved;
second-launch logs have the `-run2.log` suffix. Those runs also exited before
decoder compilation: the static-layout unit fixture expected an ignored
historical dataset manifest absent from the immutable remote checkout.
The correction exports only the audited coordinate-index metadata locally:

```powershell
conda run -n ORCJAX python scripts/hpc/probe_shared_decoder_carbon_group.py --export-layout-only outputs/shared-decoder-carbon-layout-20260925.json
```

The 32,204-byte artifact is staged as
`runtime/transfer/shared-decoder-carbon-layout-20260925.json`. The reader
checks schema, field identities, expert, signature, index shapes and bounds.
The 448-row GPU mode requires that file, eliminating the remote unit
fixture/dataset dependency. No historical data are transferred or trained,
and no decoder compilation was lost. Corrected scalar59816 completed exit0;
micro4 59873 completed exit1 on the declared grouped256 output gate, after
reference64/grouped64/grouped128 completed. Both final reports and `-run3.log`
logs are retained. This was a numerical-screen rejection, not a runtime crash.

## V100 grouped results

These use JAX0.4.38, width384, two blocks/FFN1536, FP32 payloads and20 warm
repeats per variant. Micro4 averages four distinct sample losses. Initial
GPU0/GPU1 occupancy was empty; the two small probes ran concurrently on
separate devices of gln01. Timing remains a free-node component measurement.

| Microbatch | Execution | Warm ms | Compiler temporary MiB | Compile/load s | Strict parity |
| --- | --- | ---: | ---: | ---: | --- |
|1|Field-local64|51.9780|555.296|30.114|Reference|
|1|Grouped64|51.4009|562.585|7.176|Pass|
|1|Grouped128|32.1520|575.333|8.099|Pass|
|1|Grouped256|22.3698|584.064|8.181|Pass|
|4|Field-local64|80.4546|2113.798|30.140|Reference|
|4|Grouped64|81.6207|2265.782|8.395|Pass|
|4|Grouped128|61.1884|2319.124|9.416|Pass|
|4|Grouped256|48.5349|2325.603|9.678|One output failure; all gradients pass|

Grouped128 saves19.826ms scalar (38.14%) and19.266ms micro4 (23.95%).
Compiler temporary bytes grow3.61% scalar and9.71% micro4. This is temporary
storage for the compiled fragment, not measured full-model peak VRAM.
Grouping alone at64 does not materially improve execution time; the larger
query contraction is the promising change. Grouped compilation is shorter,
but sequential component compile/load timings are not a whole-model cold
compile or cache benchmark.

All grouped128 loss/output coordinates pass (2,241 scalar;8,961 micro4),
as do all4,598,529 scalar and7,450,881 micro4 parameter/payload gradients.
At grouped256/micro4, one of8,961 loss/output coordinates exceeds the
unchanged `rtol=1e-5, atol=1e-6` gate; maximum absolute output difference
is1.40071e-6. All7,450,881 gradients pass, maximum absolute difference2.23517e-8.
No tolerance was relaxed. The small output discrepancy does not demonstrate
a broken AD algorithm or a scientific-accuracy failure, but it prevents
claiming this variant meets the declared execution-equivalence gate.

Reports: `outputs/shared-split-carbon-group-20260925-micro1.json` and
`outputs/shared-split-carbon-group-20260925-micro4.json`, also retained under
remote `runtime/outputs/diagnostics/`. Follow-up field-local128/256 controls
completed as scalar61700/GPU0 (exit1 on fieldlocal256 output gate) and micro4
61729/GPU1 (exit0), within their10-minute caps. No probe remains running.

## Matched field-local controls and selected next candidate

| Microbatch | Execution | Warm ms | Compiler temporary MiB | Compile/load s | Strict parity |
| --- | --- | ---: | ---: | ---: | --- |
|1|Field-local64 repeated reference|51.7190|555.296|23.422|Reference|
|1|Field-local128|35.2435|636.045|22.228|Pass|
|1|Field-local256|24.9662|644.696|22.152|One output failure; all gradients pass|
|4|Field-local64 repeated reference|81.0235|2113.798|23.708|Reference|
|4|Field-local128|67.5515|2421.422|23.077|Pass|
|4|Field-local256|54.5688|2433.287|23.079|Pass|

The repeated reference is within0.8% of the grouped run. Both128 schedules
pass every declared coordinate in both microbatch modes. Field-local128
saves31.86% scalar and16.63% micro4 relative to its own reference. Grouped128
is additionally about3.09ms scalar/6.36ms micro4 faster than field-local128,
and uses less temporary storage. Those cross-run comparisons are descriptive;
the effect is larger than the observed repeated-reference variation, but it
is not a controlled whole-model benchmark. Field-local256 scalar fails one
of2,241 loss/output coordinates, maximum absolute difference1.16229e-6;
all component gradients still pass. A256 default has no common scalar/micro4
acceptance across the two schedules, so it is not selected.

The next candidate is **grouped128 restricted to these five fields**,
alongside the already-screened22-field Q1 group. Preserve the rest of the
field-local decoder and the full parameter/output tree. First implement the
real feature-assembly/output-restore boundary and check masks plus native
parameter/state gradients locally. Inspect graph/lowering growth before
another expensive GPU compilation, then measure real daily micro4 and full
daily32+T7/T30 updates. Reuse saved references and keep the accepted64-chunk
model available. Do not start500 learning based on fragment timing.

Do not globally set output_chunk_size128 from this experiment. In particular,
the two896-coordinate DOC inventory fields (22/23) each have896 local-bank
entries and44 axis entries, unlike this group's one masked local entry.
Their state-gradient/storage behavior is not covered. A global or broader
grouping proposal needs an actual-helper check of that different bank shape;
that extra check is not a prerequisite for integrating only the tested group.

The19.27ms micro4/19.83ms scalar savings are larger than the earlier Q1
fragment savings, but are not a solution to full-TRAIN throughput by
themselves. Do not add isolated-fragment times and label that a measured
joint-update speedup. These screens contain zero optimizer updates and no
new evidence of scientific fit. The network architecture and data are unchanged.

Control reports: `outputs/shared-split-carbon-group-20260925-micro1-fieldlocal.json`
and `outputs/shared-split-carbon-group-20260925-micro4-fieldlocal.json`.

## Decision boundary

Only a variant passing every declared output and full parameter/payload
gradient coordinate at unchanged tolerances can be considered for
performance comparison. Read compiler temporary, argument and output bytes
separately from warm component time and lower/compile time. A helper win
does not establish whole-model compile reduction, complete-update
throughput, recurrent-state parity, physical acceptance, or scientific fit.
Even this larger group excludes tokenizer/feature assembly, coordinate
scatter, 156 other output fields, raw-carbon GPP routing, retained physics,
and T7/T30 recurrence. Full-model adoption would require a targeted
integration plan and saved-reference daily/T7/T30 gradients, masks, physical
and complete-update checks before another long learning run.
