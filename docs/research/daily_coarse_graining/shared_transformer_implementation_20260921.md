# Shared daily Transformer: implementation admission

2026-09-21. User-approved implementation of
`shared_transformer_long_term_design_20260921.md`. Scientific acceptance is
false. No new paid training has been submitted.

## Implemented boundary and method

The canonical assembler, conservative updater and source-backed retained tail
are unchanged. `SharedTransformerProcessHead` supplies the same 161 learned
fields / 39,006 coordinates. All 75,578,497 neural parameters initialize from
seed20260921; no old checkpoint or superseded neural stack is constructed.

The architecture is one shared parameter tree: canonical input tokenizer,
eight width768/8-head/FFN3072 backbone blocks, two shared coordinate-decoder
blocks and a lightweight readout. Seven ownership-masked process streams use
the same weights. There are393 source tokens,140 axis tokens and2386 raw-carbon
coordinates. Output execution uses737 field-local chunks, maximum64 queries
and no output-output attention. Small fields use their actual size. FP32
cosine attention uses `logits=cosine(Q,K)*sqrt(head_width)` throughout the new
tokenizer/backbone/decoder; legacy callers retain their old default behavior.

GPP retains its explicit vcmax25 factor and vegetation applicability.
Maintenance retains its source-shaped physical formula and per-organ
coefficients, with corrections from the shared network. Outgoing inventory
fractions use the existing TRAIN mean as their logit center, interior floor
1e-8, instead of default50% daily removal. Incoming and other typed transforms
reuse the accepted transforms. These are modeling choices, not accuracy claims.

The loss has three equal groups: daily necessary process/state-change terms,
four independent carbon fluxes, and seven carbon stocks. Fluxes penalize both
daily and prefix-cumulative errors; stocks penalize adjacent residual changes
and trajectory drift. Prefix/trajectory denominators grow as sqrt(elapsed
days), with equal living/nonliving stock subgroups. NPP and RA remain derived
scientific evaluation quantities. Budget counters still mean endpoint
bookkeeping, not independently demonstrated ecosystem flux closure.

Calibration is bounded to16 existing TRAIN artifacts / at most512 daily rows.
It measures extracted physical TRAIN RMS with0.01gC/m²/day floors and reports
nonzero RH and unique-reference coverage. It is not global scale qualification.
The daily state anchor specifically uses frozen `state_delta` statistics.
Stock change supervision also requires the initial stock to be source-defined.

## Local evidence

- Shared architecture and attention tests:16 passed; masking, PFT permutation,
  chunk invariance, finite gradients, default-off legacy policy and full shapes.
- Legacy strong-baseline tests:10 passed.
- Loss/runtime/physical tests:23 passed before the extra canonical smoke.
- Final physical-head/training suite:9 passed, including the new actual
  small-width canonical head forward across all161 fields. It allocates no old
  neural parameters. This checks the physical head, not an actual retained-tail
  production-day transition.
- Ruff and whitespace checks pass.

Sol/high workers implemented/reviewed independent topology, loss and runtime
components; the main agent integrated the physical head and admission runner.
Codebase graph tools were unavailable (Transport closed); exact-source fallback
was used, without completeness claims about the knowledge graph.

## Actual-runtime admission

Immutable server code: `4eeb168`, under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-transformer-4eeb168`.
This shared-object clone is separate from prior training checkouts; the old
server Git lacks worktree and `git -C`, so deployment uses explicit `cd`.

Both CPU/GPU shell argument expansions passed target-shell `--dry-run`, plus
Bash syntax. Free test02 CPU process52413 (wrapper52337, timeout20minutes)
COMPLETED0 in416.48seconds. Actual canonical forward and full-gradient shapes
pass; all75,578,497 parameters remain trainable. Reports/logs are under
`runtime/outputs/diagnostics/shared-transformer-4eeb168-cpu.{json,log}`.
Local report: `outputs/shared-transformer-4eeb168-cpu.json`.

Calibration consumed512 rows from256 unique point-years/references. Physical
RMS flux scales (gC/m²/day): GPP2.846919, maintenance1.268240, growth0.525143,
RH0.01 floor. Stock-change scales: AGB0.607756, BGB0.263174, above litter0.133129,
below litter0.225678, canopy7.840437, DOC0.561207, particulate carbon0.01 floor.
There were no nonzero RH examples: this panel cannot certify nonzero RH fit.
Admission update rows are307.0-095.0/year1961/start86 (artifact819) and
137.0-087.0/year1963/start171 (artifact982); these are TRAIN, not held-out scores.

Free gln01 GPU0 admission has STOPPED; wrapperPID107435 and PythonPID108244
are inactive. Its report/log prefix
is `runtime/outputs/diagnostics/shared-transformer-4eeb168-gpu`.
Check the wrapper with `ssh gln01 'ps -p 107435 -o pid,etime,pcpu,comm'` from
the login node, or query PythonPID108244 for actual CPU utilization. JSON status and the
terminal `run_exit_status` in the log distinguish completion from a live shell.

The GPU launcher first runs the small actual-environment AD/optimizer runtime
check, then performs real-model single-day forward, physical checks, complete
gradient, native JVP/reverse projection comparison and two full AdamW updates.
The actual GPU environment check PASSED in76.10seconds (JAX0.4.38); local
report `outputs/shared-transformer-4eeb168-gpu-runtime.json`. This synthetic
check is distinct from the running real-model test.
It persists the first full gradient before another compilation, then checks
saved parameter/optimizer readback. GPU model test timeout60minutes plus a
separate5minute runtime-check bound; no Slurm job or paid resource charge.

### Collected GPU outcome

Exit1 after2822.12seconds (about47minutes), reason `compiled_gradient_memory`.
Both initial canonical forwards completed at666.4seconds and every reported
physical/science hard counter is zero. Compilation finished at2822.1seconds;
estimated live gradient memory38,842,265,019bytes =36.17GiB, including
38,237,190,912bytes =35.61GiB temporaries. Parameter/input arguments are0.28GiB.
The estimate exceeds both the24GiB admission ceiling and physical32GiB V100
capacity. This was a pre-execution resource rejection, not an observed OOM,
nonfinite gradient or failed learning experiment. Completed updates:0.
JVP/reverse agreement and checkpoint replay were not reached.
Local report: `outputs/shared-transformer-4eeb168-gpu.json`.

The initial loss is also badly imbalanced: state-change anchor1.45136e9 and
1.32871e9 on the two rows, versus necessary-process labels1.35565/0.65867,
flux0.04585/0.16122 and stock0.45404/0.85899. The state anchor, not the separate
carbon science terms, dominates the objective. Its responsible fields and
scales remain to be diagnosed; this does not establish a specific scaling bug
or trained-model accuracy. Next reduce decoder gradient intermediates and
attribute this loss before attempting learning. Do not simply raise the memory
ceiling, change network architecture or retune loss weights without evidence.

## Next gates and limitations

Passing two updates is execution evidence, not demonstrated learning. The
actual changed joint daily/T7/T30 loss still needs native AD, memory and bounded
update admission. Recurrent integration must reuse
`initial_component_stock_defined` and `shared_trajectories_with_initial_mask`.
Do not rebuild observation cotangents in a runner.

Then freeze a bounded learning/data-coverage contract and independent7/30day
evaluation. Only scientific short-window success permits365day, long-held,
GateD2 and product acceptance. The old24.80yuan fresh-litter proposal stays on
hold. Do not regenerate v6, mix v5, resume old exploratory weights or add a
48-half-hour inference scan.

Runtime risk still to measure: the coordinate decoder repeats bank projections
per coordinate. Shared weights reduce parameter duplication but do not by
themselves guarantee faster inference or training.
