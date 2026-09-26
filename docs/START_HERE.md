# Project Start Here

Updated: 2026-09-26. Entry-document and obsolete-source cleanup are complete.
The next work is speed research before training; no GPU job or training is active.

## Two products, separate claims

The **PFT14 Teacher** is the source-backed JAX port of the paper-era ORCHIDEE-MAN
configuration. It retains 48 half-hour transitions and the daily STOMATE tail.
Scoped Fortran/lifecycle validations pass; full 669-point Fortran-equivalence
release acceptance is a separate, still-pending gate.

The **daily surrogate** is research, not a Fortran-equivalent or accepted
user-facing model. Its inference boundary is:

```text
day-start physical state + native six-hour forcing + static/physical parameters
  -> shared masked neural process head
  -> conservative daily updater + retained exact daily processes
  -> next-day physical state
```

No 48-step forcing reconstruction or state scan belongs in final daily
inference. Inter-day memory is explicit physical state, not neural hidden state.
PFT-axis extensibility and physical-parameter gradient validity are separate
claims; neither follows from a finite neural loss or a successful PFT14 run.

## Read only what the task needs

1. [NEXT_STEPS.md](NEXT_STEPS.md): current operation, gates and authorization.
2. [CODE_MAP.md](CODE_MAP.md): where changes belong and what remains live.
3. [current-status.md](current-status.md): accepted facts and latest evidence.
4. [DOCUMENT_STATUS.md](DOCUMENT_STATUS.md): select a controlling contract.

These pages are current-only. For a named old experiment or a decision's
provenance, use [HISTORY_INDEX.md](research/daily_coarse_graining/HISTORY_INDEX.md).
Do not read the whole `HANDOFF.md` or dated report collection for onboarding.

## Stable boundaries

- Source truth: `fortran_source/ORCHIDEE`; paper protocol:
  `fortran_run_scripts/paper_250919`; reference truth: `reference/` and `traces/`.
- Teacher code: `jax_orchidee/`; research: `research/daily_coarse_graining/`.
- Accepted coherent Teacher dataset: **v6 fixed**, 669 points, 1961-2010,
  33,450 point-years, 12,208,581 transitions. Reuse it; do not regenerate or
  join historical v5 rows. Dataset admission is not full Fortran equivalence.
- Current neural candidate: 64,884,481 parameters; shared backbone width 768,
  eight blocks; common coordinate decoder width 384, two blocks; native
  checkpointed AD; canonical binding/updater/retained tail.
- Formal training must specify fresh initialization and full-network joint
  optimization reproducibly. Old checkpoints are diagnostic assets, not the
  default final-method starting point. Scientific and product acceptance remain
  pending.

Return to **speed optimization before training**. The 10x target is a research
goal, not a measured result or promise. See NEXT_STEPS, not old
reports' proposed launches, for what is authorized.
