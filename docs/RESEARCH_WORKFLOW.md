# Research context maintenance

This policy keeps day-to-day research context small while preserving evidence.
It applies to `research/daily-coarse-graining`; scientific and server rules in
the project instructions still apply.

## Read by task

1. Read `START_HERE.md`, `NEXT_STEPS.md`, and `CODE_MAP.md` for orientation.
2. Select the relevant task/code group from the workspace inventory linked in
   `CODE_MAP.md`. Inspect symbols and their actual dependencies, not every file
   in `scripts/hpc` or `research/daily_coarse_graining`.
3. Open only the named contract/result needed for the decision. Use
   `DOCUMENT_STATUS.md` to distinguish active contracts from evidence.
4. Consult `research/daily_coarse_graining/HISTORY_INDEX.md` only for a historical
   question. The long historical handoff and old checkpoints are not onboarding.

Use graph discovery with coverage/source verification as required by project
instructions. A file outside a selected task closure is not automatically dead;
public exports, other tasks, shell entry points and dynamic loading can use it.
Large generated inventories belong in `outputs/`, not default reading context.

## Update current facts rather than accumulate messages

- `NEXT_STEPS.md`: one current work order, outstanding decision and next gate.
  Replace superseded instructions instead of prepending another chronology.
- `current-status.md`: latest measured facts, acceptance limits and evidence links.
  Put experiment detail in its dated report; update that report when it finishes.
- `CODE_MAP.md`: stable responsibilities and a small task-based navigation map.
  Do not add a new paragraph for every diagnostic script.
- `DOCUMENT_STATUS.md`: currently needed contracts and a pointer to history.
- Keep the first three entry documents roughly below 250 lines/20 KiB combined,
  and all five below 500 lines. These are editorial targets, not runtime gates.
- Do not duplicate changing status in the two READMEs. Link the current authority.

## Retire with evidence

The pre-cleanup source/docs are preserved at
`research/daily-before-context-cleanup-20260926` (commit `273157c`). See
`HISTORY_INDEX.md` for read/restore commands; no second loose archive is needed.
The tag is local until explicitly pushed with an authorized repository update.

Retire files in bounded dependency groups, including their exclusive wrappers,
configs and tests. Preserve shared helpers and source/scientific evidence.
When only abandoned experiments depend on each other, retire that whole group;
do not retain it solely because its old tests or launchers still import it.
Extract current shared helpers from historical runners before retiring callers.
Exclusively historical research APIs may retire with their experiments; preserve
current interfaces and the module identities used by retained serialized data.
Record what moved out of the current checkout and the exact recovery reference.
Moving every old file into `archive/` alone does not reduce checkout size.
Do not remove data, weights, results or remote checkouts as part of source cleanup.
Do not exclude live files from search/indexing just to make an inventory look small.

Check the relevant imports/tests and current navigation links after a change.
Documentation-only edits need no model compilation, gradient rerun or paid job.
