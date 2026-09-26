# Current Project Status

Updated: 2026-09-26. This is the current facts/evidence authority, not a progress
log. [NEXT_STEPS.md](NEXT_STEPS.md) controls operations.

## Work and authorization

Current-context consolidation is complete: compact entry pages, task-based
source inventory, historical recovery and three retired legacy shell launchers.
Historical Python dependencies remain where needed. No GPU job, new paid
allocation or neural training is active. Next is speed optimization before
training; cleanup does not authorize production precision or architecture
changes. The 10x throughput target is unmet and is not a promise.

## Accepted foundations and scope

| Foundation | Accepted evidence / limitation |
| --- | --- |
| Teacher | Source-backed PFT14 half-hour model and exact daily tail; scoped cold/restart/365-day/cross-landpoint/year-handoff checks pass |
| Teacher branch synchronization | Accepted core `5910ba9` on main / `10030eb` on research; [parity acceptance](teacher-branch-parity-acceptance.md) |
| Teacher release | Seven previously unused points completed 1961-2010 checks, largest 2010 AGB/BGB/GPP/NPP relative error `8.5e-5`; full 669-point Fortran equivalence remains a distinct gate |
| PFT interface | Stable identity/catalog, named channels, mask/permutation/variable-axis structural gates; PFT2-PFT13 unsupported and fail closed |
| Canonical daily boundary | Process prediction -> conservative updater -> retained exact daily tail; label replay/lifecycle/mask/budget evidence accepted within tested scope |
| Teacher gradients D1 | Bounded process, Day-2, two-day, restart and source-extracted finite-difference cases accepted; not surrogate D2 |
| Teacher data | Coherent **v6 fixed**, 669 points x 1961-2010, 33,450 point-years, 12,208,581 transitions accepted; reuse without regeneration or v5 joins |

Dataset evidence is in
[teacher_v6_full_production_acceptance_20260807.md](research/daily_coarse_graining/teacher_v6_full_production_acceptance_20260807.md).
The accepted dataset and successful Teacher checks do not establish all Fortran
branches, all PFTs, or the surrogate's scientific validity. Source-driven
process/branch coverage remains the porting rule.

## Current neural method

The current shared candidate has **64,884,481 parameters**: source-masked
shared Transformer backbone width 768/eight blocks; common coordinate decoder
width 384/two blocks. Every potentially active output retains the shared
decoder; auxiliary labels do not imply scientific dispensability.

Canonical assembler/binding, conservative updater and retained-tail adapter
remain unique. Inference is one true daily transition with native six-hour
forcing, no 48-step state scan or reconstruction. Explicit physical state
supplies recurrent memory. PFT/source/defined masks and physical-parameter
ownership remain intact; `vcmax25` and maintenance-respiration factors retain
explicit routes, while `alloc_min`/`residence_time` remain exact retained logic.

Native checkpointed JAX AD is the preferred multi-day route. Completed trained
T7/T30 directional evidence is reused; shared public rollout losses and science
extractors remain authoritative. A changed loss/state/AD boundary needs relevant
bounded checks, not closure of every abandoned custom-reverse experiment.

Final scientific training must start from declared fresh initialization and
optimize the full network jointly, with any stages justified and transitions
specified beforehand. Exploratory weights/checkpoints remain diagnostics and
resource probes, not a default inherited final-method chain. Fresh-init
admission requires physical/numerical/meaningful-gradient gates followed by
learning; it does not require untrained accuracy to match an old checkpoint.

Architecture roles and source ownership are in [CODE_MAP.md](CODE_MAP.md);
controlling documents are selected in [DOCUMENT_STATUS.md](DOCUMENT_STATUS.md).

## Latest completed execution evidence

All following measurements are bounded diagnostics, not final scientific
acceptance. No corresponding free-GPU task remains active.

| Test | Outcome and scope |
| --- | --- |
| Combined unchanged daily32 + T7 + T30 update | Serial `12.553775s`, dual `10.270496s`; complete changing-update gradients/parameters/AdamW and physical parity pass; zero new model-gradient compiles |
| Daily32/64 dual balance | `9.859359s` / `9.769036s`; daily64 branches about `8.313s` / `8.326s`; all proposal parity/physical checks pass; changed daily count, not same-work speedup |
| Daily64 serial vs dual | `16.879002s` vs `9.769036s`, 42.12% less wall with 15.75% more GPU-seconds; free test, learning at daily64 untested |
| Native FFN `F16_F16_F32`, micro4 | Forward + native VJP `72.415 -> 46.250ms`, `1.566x` |
| Native FFN `F16_F16_F32`, micro1 | Forward + native VJP `114.759 -> 95.827ms`, `1.198x` |

FFN values and VJPs are finite, but strict `1e-5`/`1e-6` comparison fails.
Relative L2 errors: forward `0.04193%`, synthetic-cotangent parameter VJP
`0.27757%`, input VJP `0.04524%`. Tiny deployed capability trace confirms
Tensor-Core kernels. This is neural fragment evidence, not actual rollout
gradients, a production numerical-error contract, scientific accuracy or
full-update speed. Physical precision and production defaults are unchanged.

Read [full update](research/daily_coarse_graining/shared_combined_full_update_20260926.md),
[daily64 balance](research/daily_coarse_graining/shared_combined_daily64_balance_20260926.md),
and [FFN results](research/daily_coarse_graining/shared_ffn_precision_results_20260926.md)
for exact experiment/data/runtime identities and limitations. Preserve prepared
inputs, completed reference gradients and executable provenance; do not repeat
large compilations solely to reconstruct already recorded evidence.

## Still pending

- Effective same-workload 10x complete-update throughput; fragment, startup,
  changed sampling and extra-GPU gains remain separate evidence.
- An independently specified executable final training protocol and its
  end-to-end learning/validation, not exploratory checkpoint replay.
- Held-out scientific accuracy, long-horizon canopy/carbon/productivity/biomass
  trajectories, 365-day/long-site drift and sealed final evaluation.
- Surrogate physical-parameter gradient D2, full Teacher release equivalence,
  and final user-facing product acceptance.

For named past experiments, use
[HISTORY_INDEX.md](research/daily_coarse_graining/HISTORY_INDEX.md).
Earlier successes/negative results remain intact; none is promoted to a final
model by this context cleanup.
