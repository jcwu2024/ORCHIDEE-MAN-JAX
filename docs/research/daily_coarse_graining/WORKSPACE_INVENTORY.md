# Task-Directed Research Source Inventory

Updated 2026-09-26. This is a navigation and retirement aid, not a model,
training, precision, data-generation or deployment contract. Read the current
entry documents first; do not read every file in a dependency closure.

The September 26 consolidation reduced the five entry documents from 13,846
lines / 1,011,841 bytes to 401 lines / 24,832 bytes. The initial three now total
235 lines. This measures reading volume, not billed AI tokens. On-demand public
exports reduce a fresh package import from 55 daily-operator modules to one;
importing the current `shared_transformer` child loads 15 instead of 56.
Import counts are separate from GPU compilation or training throughput.

Local validation: five inventory tests passed; the targeted E1/shared/API suite
had 64 passes and one pre-existing directory-audit failure. That audit reports
the identical 13 violations with the original and revised initializer, with no
new violations. All seven final public-import checks passed, including fresh
first lookup inside JIT and native AD. The 90 current-document/README relative
links resolve. No server computation or scientific-method change was involved.

## Three Source Groups

The compact, machine-readable seeds are in
`manifests/coarse_graining/research_workspace.json`:

| Group | Read for | Status |
| --- | --- | --- |
| `scientific_route` | Shared Transformer, process binding, daily/recurrent objectives, learning plan and monitoring | Retained; training paused |
| `performance_diagnostics` | Exact opt-in decoder/backbone execution, saved complete updates, FFN precision/algorithm probes | Diagnostic only; precision not adopted |
| `frozen_data_infrastructure` | Loading existing v6/prepared artifacts, calibration and microbatch plumbing | Frozen data; no generation authorization |

The scientific starting files are `shared_transformer.py`,
`shared_process_head.py`, `shared_training.py`, `shared_recurrent_training.py`
under `research/daily_coarse_graining/daily_operator/`, and
`scripts/hpc/run_shared_learning.py`. Follow the relevant imports only when the
task touches their behavior. The current network is the fresh-initialization,
full-sized shared architecture; historical checkpoint provenance does not
authorize resuming its exploratory training chain.

Old names can still be current dependencies: the shared runtime reuses
`run_fresh_start_one_day.py`, canonical retained binding and prepared-artifact
loaders. `run_successor_process_head_canonical_gate.py` supplies `_atomic_json`
to current tools. Do not retire these by filename prefix.

## Regenerate Evidence

```powershell
conda run -n ORCJAX python scripts/dev/research_workspace_inventory.py
conda run -n ORCJAX python scripts/dev/research_workspace_inventory.py --output outputs/research/context-cleanup/inventory-20260926.json
conda run -n ORCJAX python -m pytest tests/unit/test_research_workspace_inventory.py -q -p no:cacheprovider
conda run -n ORCJAX python -m pytest tests/unit/test_daily_operator_public_imports.py tests/unit/test_daily_operator_e1.py tests/unit/test_shared_transformer.py tests/unit/test_shared_process_head.py tests/unit/test_shared_daily_microbatch.py -q -p no:cacheprovider
```

The tool reads tracked text and ASTs without importing research packages,
opening Teacher arrays, compiling JAX or contacting servers. Default output
is a compact summary. Optional detailed output must be under `outputs/`.
It follows local imports/from-imports, containing package initialization,
literal tracked file/config references and shell `python -m` entrypoints.
It reports computed dynamic import/file-loading calls separately.

`daily_operator/__init__.py` public exports require special care. Eager exports
are import dependencies even when a task does not use the exported names.
For explicit public/compatibility lazy maps, requested from-import names enter
the task closure; all defining modules remain a separate compatibility closure.
Lazy loading is not permission to delete an old public implementation.

Conservative static closure before/after lazy initialization and bounded
entrypoint retirement (Python files / all dependency lines):

| Group | Before | After |
| --- | --- | --- |
| Scientific route | 207 / 188,159 | 200 / 185,365 |
| Performance diagnostics | 226 / 195,255 | 219 / 192,461 |
| Frozen data infrastructure | 177 / 178,988 | 170 / 176,194 |

These are not files an agent must read, actual loaded-module counts or
dead-code proof. Conditional imports and reused old helper modules remain.

The report is deliberately conservative: imports inside inactive branches and
colliding literal filenames may over-retain files. Computed module paths,
`getattr`-selected public exports, external configuration, serialized kernels
and pickle-defined classes are not exhaustively resolved. Runtime/deployment
identity checks remain necessary before retiring assets they may require.
Four preserved legacy Python2 paper-run scripts fail Python3 AST parsing;
they are reported, not changed. Only parse failures in a selected task closure
or missing declared seeds fail the inventory command.

## Bounded Retirement Findings

Safety snapshot: annotated tag
`research/daily-before-context-cleanup-20260926`, commit `273157c`.

The three legacy shell wrappers below were retired after lead acceptance.
They are outside the declared task closures.
Their complete source and tracked import/literal/script references were
checked. No tracked inbound references were recorded before retirement:

- `scripts/hpc/run_successor_fast_owner_gpu_probe.sh` (81 lines).
- `scripts/hpc/run_successor_process_head_canonical_gate.sh` (80 lines).
- `scripts/hpc/run_successor_process_head_smoke.sh` (74 lines).

These are historical tag-recoverable entrypoints, not active launchers.
Retain their same-named Python files where current code imports their helpers.
External saved commands are outside the tracked-reference check; recover the
historical shell entrypoint from the tag when reproducing that experiment.

Do **not** remove `probe_successor_parameter_geometry.py` or
`probe_successor_multibatch_direction.py` yet. They are outside the current
seed closures but still have historical Python importers and numerical tests.
The 14 geometry/multibatch shell wrappers also have launcher-text tests in
`test_daily_operator_named_biomass_tendency.py` and
`test_successor_output_locality.py`. Retiring that larger family requires
separating obsolete launcher assertions from reusable helper/science tests.

Historical recovery without changing the current checkout:

```powershell
git show research/daily-before-context-cleanup-20260926:scripts/hpc/run_successor_process_head_smoke.sh
git archive --format=zip --output=outputs/research/context-cleanup/successor-entrypoints.zip research/daily-before-context-cleanup-20260926 scripts/hpc/run_successor_fast_owner_gpu_probe.sh scripts/hpc/run_successor_process_head_canonical_gate.sh scripts/hpc/run_successor_process_head_smoke.sh
```

Graph evidence used project `E-ORCHIDEE-MAN-JAX`, generation
`2026-09-26T10:57:17Z`, Verify tier. Candidate coverage had no recorded parse
gaps but `metadata_changed` freshness, so the conclusions above use direct
source/AST and literal-reference fallback. Graph absence is not dead-code proof.
