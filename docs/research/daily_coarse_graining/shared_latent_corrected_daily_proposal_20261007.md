# Corrected Daily Latent Reader Diagnostic Proposal

Status: paused at the user's request on2026-10-07; preparation is incomplete.
No further deployment/checks or resource application are active. No grant has been issued, no GPU preflight or
model call has run, and no optimizer update or paid job is authorized. This
proposal describes one new free-node occupation with its own approval and
ledger. It cannot draw from the completed T30 grant or any prior balance.

## Question and Scope

Measure one corrected daily candidate graph against the saved corrected
original daily graph under full native parameter and initial-state AD. This is
a bounded performance diagnostic, not scientific acceptance, architecture
adoption, training, or an accuracy claim. Preserve the existing loss,
initialization, physical and mask checks, 14 hard-count maps, 24 GiB gate, and
strict `rtol=1e-5, atol=1e-6` evidence. A strict state-repeat failure remains a
reported failure and cannot be waived by timing or parameter-gradient success.

The candidate is compiled once. The original corrected daily executable is
loaded from its saved binary and must not compile. The probe makes eight
complete AD calls: one cold call per architecture, followed by three
alternating original/candidate pairs in order original/candidate, candidate/
original, original/candidate. This gives six warm paired observations.
The comparison uses one V100 on `gln01` with eight selected CPUs `8-15`; no
exclusive-node request is involved.

## Inputs and Outputs

The immutable source checkout is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`
at `c599236bcf57b745e9bafed678298010f4c3b58f`. The deployed overlay is
`runtime/transfer/shared-latent-reader-20261004`. The full-update manifest is
`runtime/outputs/diagnostics/shared-latent-reader-20261004/corrected-full-update-manifest.json`.
It binds the original report, executable, identity sidecar, acceptance report,
source profile, artifact 742 inputs, update-0 checkpoint, and fresh candidate
parameters. The original executable is
`runtime/outputs/diagnostics/leaf-full-state-cuda-20261005-full-state.executable.pkl`.

The only candidate report and executable paths are
`runtime/outputs/diagnostics/latent-reader-daily-20261006-corrected-candidate.json`
and its `.executable.pkl` sidecar. The launcher refuses to overwrite either,
their identity/compile receipts, the runtime-preflight report, or NVML logs.
CPU validation writes to a separate validation-output path and does not create
the canonical candidate report or executable.

## Resource and Timing Ceiling

The proposed fresh grant is exactly 7,200 free GPU-seconds for one occupation
on one V100, equivalent to a two-hour card-hour ceiling at CNY 0. It has its
own unapproved receipt template, cumulative ledger, and occupation receipt.
The accepted runtime is only `gln01`, with 64 GiB available host memory at
start, CPUs `8-15`, four OMP threads, and one OpenBLAS thread. Slurm allocations
are refused.

The probe wall cap is 6,840 seconds. Lowering is capped at 600 seconds,
backend compilation at 4,800 seconds, and each of the eight AD calls at 180
seconds. These sub-limits sum to 6,840 seconds; they are ceilings, not a
prediction that the work will finish within them. The runtime preflight has a
separate 120-second timeout, and the occupation has its own 120-second
guardian finalization reserve. All startup, compilation, loading, calls,
preflight and cleanup remain inside the hard 7,200-second occupation cap. The
guardian stops the process group at the cap and records the full reservation
if own-process GPU release cannot be confirmed. A failed or interrupted
occupation consumes the one-use grant; a retry needs a separate fresh grant.

## Guarding and Run Boundary

`scripts/hpc/shared_latent_daily_budget.py` adapts the existing free-node
guardian's file locks, monotonic occupied-time accounting, start/end host and
GPU snapshots, process-group termination, and own-GPU-release checks. It uses
new daily-specific receipt and ledger identities; the fixed T30 approval,
phase names, cumulative balance, and artifact family are rejected. The
receipt must carry a fresh UUID, timezone-aware approval time, nonempty human
approval reference, the exact 7,200-second scope, resource record, and this
operation's fixed limits. The checked-in/template form stays `approved=false`.

The launcher invokes the lightweight GPU runtime check inside the same
charged occupation before initializing the probe. It then verifies the
selected card has no visible compute process, at most 100 MiB in use, and zero
utilization before starting the probe. It does not submit Slurm work. A
`--dry-run` expands and prints both the guardian command and nested preflight/
probe command without reading an approval receipt, running preflight, touching
a ledger, or accessing a GPU.

## Validation Record

To be filled after the target-shell argument dry run. Full asset loading is a
separate CPU-only validation owned by the parent task. This preparation does
not run it automatically.
