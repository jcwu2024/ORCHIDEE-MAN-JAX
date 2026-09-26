# Task-Directed Research Source Inventory

Updated 2026-09-26. This is a navigation and retirement aid, not a model,
training, precision, data-generation or deployment contract. Read the current
entry documents first; do not read every file in a dependency closure.

The first cleanup consolidated entry documents. Source cleanup is now complete:
entire obsolete experiment families retired, keeping scientific reports and
tagged source recoverable. These are repository/context changes, not a claim
of faster training, new scientific accuracy or measured AI-token savings.

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

Current runtime helpers now live under `scripts/hpc/` in
`canonical_training_runtime.py`, `retained_training_runtime.py`,
`training_artifact_io.py` and `runtime_diagnostics.py` (624 lines total).
Fifteen extracted function bodies match their snapshot ASTs, apart from a
declared function rename. They no longer import historical experiment runners.
Keep `target_oriented_training_prepared.PreparedTargetOrientedUpdateArtifact`
and `daily_operator.target_oriented_training.PreparedTargetOriented*Batch` at
their original module identities: existing prepared data uses those pickle paths.

## Regenerate Evidence

```powershell
conda run -n ORCJAX python scripts/dev/research_workspace_inventory.py
conda run -n ORCJAX python scripts/dev/research_workspace_inventory.py --output outputs/research/context-cleanup/inventory-after-source-retirement.json
conda run -n ORCJAX python -m pytest tests/unit/test_research_workspace_inventory.py -q -p no:cacheprovider
conda run -n ORCJAX python -m pytest tests/unit/test_training_runtime_helpers.py tests/unit/test_shared_transformer_runtime.py tests/unit/test_daily_operator_public_imports.py -q -p no:cacheprovider
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
There are 321 retained public exports; 29 exclusively retired experimental
exports and four module aliases were deliberately removed. Retained objects keep
their defining modules. Old experiments should run from their historical revision.

Conservative static closure before/after source retirement
(Python files / all dependency lines, including inherited Teacher code):

| Group | Before | After |
| --- | --- | --- |
| Scientific route | 200 / 185,365 | 184 / 173,476 |
| Performance diagnostics | 219 / 192,461 | 203 / 180,572 |
| Frozen data infrastructure | 170 / 176,194 | 151 / 163,957 |

Current conservative closures contain 226, 248 and 185 total dependency files
respectively. These are not files an agent must read, actual loaded-module
counts or dead-code proof. Conditional imports and reused infrastructure remain.

The report is deliberately conservative: imports inside inactive branches and
colliding literal filenames may over-retain files. Computed module paths,
`getattr`-selected public exports, external configuration, serialized kernels
and pickle-defined classes are not exhaustively resolved. Runtime/deployment
identity checks remain necessary before retiring assets they may require.
Four preserved legacy Python2 paper-run scripts fail Python3 AST parsing;
they are reported, not changed. Only parse failures in a selected task closure
or missing declared seeds fail the inventory command.

## Source Retirement and Recovery

Measured against source snapshot `5682064`, 731 files retired: 95 research
modules, 422 scripts, 97 declarative experiment configurations and 117 exclusive
test files. Selective pruning preserves reusable math/contract tests in mixed
files. Source/configuration/test changes remove 174,238 lines and add 926,
including the four extracted runtime modules and focused tests: a net reduction
of 173,312 lines, excluding entry-document changes. This is checkout size and
context reduction, not training throughput or measured token savings.

Five measured-result JSON records remain unchanged despite retired experiment
names: successor gradient feasibility, named-biomass coefficients, tendency
scales, feedback-update admission and parameter-conditioned learning evidence.
Their exact measurements/provenance are historical evidence, not live protocols.
The final snapshot-overlay dependency check reports zero surviving Python,
shell or JSON owners with edges to retired files. Static analysis still has the
limitations above; it does not resolve arbitrary dynamic or external references.

Validation completed locally under `ORCJAX`:

- All 3,014 remaining unit tests collect without errors; collection is not execution.
- 32 helper/runtime/monitor/preparation tests pass, including baseline-created
  prepared-pickle deserialization and unchanged stored trace signatures.
- The research owner reports 23 public-API/core/fixture checks passing.
- The script owner reports 11 initial fast checks, then six post-configuration
  checks (five inventory tests overlap; do not sum these as unique tests).
- Regenerated task inventory: no missing seeds, zero active parse errors; the
  four preserved Python-2 paper scripts are outside the selected closures.
- Current entry/README relative links, helper undefined-name lint and `git diff --check` pass.

A broad legacy numerical run was interrupted after approximately seven minutes
without results; no full numerical-suite result is claimed. No remote compute,
training or new runtime/gradient experiment was performed for cleanup. Unchanged
Teacher/source truth, scientific datasets, weights and dated reports were not
modified. Deployed numerical/scientific acceptance and all current model gates
remain separate from these source-integration checks.

Source snapshot: annotated tag
`research/daily-before-source-retirement-20260926`, commit `5682064`.
The earlier entry-document snapshot remains `research/daily-before-context-cleanup-20260926`.
Both tags are local until explicitly pushed. Source retirement includes old
canonical/multistep, causal-carbon, successor geometry/multibatch/staged training,
conditioned-GPP continuation/attribution, and superseded optimizer diagnostics.
Exclusive tests retire with those implementations; mixed files retain tests of
the current math/helpers. Teacher/source truth, datasets, checkpoints and dated
scientific reports stay intact. Recover exact historical source without adding
a second loose archive to the current checkout:

```powershell
git diff --name-only --diff-filter=D research/daily-before-source-retirement-20260926 HEAD
git show research/daily-before-source-retirement-20260926:scripts/hpc/probe_successor_parameter_geometry.py
```

Graph evidence used project `E-ORCHIDEE-MAN-JAX`, generation
`2026-09-26T10:57:17Z`, Verify tier. Candidate coverage had no recorded parse
gaps but `metadata_changed` freshness, so the conclusions above use direct
source/AST and literal-reference fallback. Graph absence is not dead-code proof.
Post-retirement graph refresh is unverified: another indexing operation is
active with different options. Final integration evidence is the current
source/AST inventory and snapshot-overlay check, not the old graph counts.
