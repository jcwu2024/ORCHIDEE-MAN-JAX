# Paid affine startup evidence remediation

2026-10-12. Local evidence remediation only. The accepted paid v3 terminal
result remains [job15806646](T7_affine_paid_GPU_recovery_results_20261012.md):
one submission, frozen `58a2d93885e991fc8e5e751f8a52b27aaade08e0`,
FAILED2:0/7s, scope closed/consumed. This candidate preserves the existing
hard gates and adds evidence to the inputs they already inspect. It has not
been frozen, deployed or executed on a GPU. No new execution is authorized.

## Historical fact and uncertainty

The retained CSV and UUID-bound XML both exited0 on n01, driver525.85.12,
minor0, UUID `GPU-3d780420-3659-2d22-3ffa-c6860e4cad9c`.
The unchanged wildcard-cgroup gate rejected before Supervisor, runtime
preflight, CUDA/PJRT identity or affine science. Numerical status remains
`not_executed`. Actual `/proc/self/cgroup`, `devices.list` bytes and kernel
stat rows were not retained; this patch cannot reconstruct them or establish
the site's exact configuration. No post-exit live GPU cleanup was observed.
Estimated additive allocation cost remains CNY0.0048222222; actual bill unknown.
Closed scope SHA256 is
`5b6b24bbb54c98879d74033e6d1b21ed9bc9d4e9f77dafe5ae14fa7092c077d4`.
Free b676 remains paused/approved/unconsumed; old f4ae and58a deployments,
attempts, false/open/closed scopes, source/input identities and results remain
historical assets. No retry, polling, configuration repair or training follows.

## Retained startup chain

The opt-in schema is `T7_paid_startup_observation_v1`. Early Python rejections
carry `paid_startup_observation`; admitted plans retain it in Supervisor output.
Scheduler-clock evidence is emitted to stderr while stdout keeps the exact
two-token shell protocol. Shell evidence uses explicit `%q` fields. Existing
query counts, timeouts, allocation clock, phase deadlines, signal/release and
one-shot behavior remain intact.

| Boundary | Retained observation and unchanged gate |
|---|---|
| Slurm entry/outer shell | Declared checkout/scope/attempt/argument count, inherited job/GPU tokens, raw clock reply, allocation epochs/current clock, typed rejection/exit; first live filesystem operation remains exclusive attempt creation. |
| Scope/capture/source | Arguments, scope and capture receipt text, bounded file size/hash reads, boundary metadata/shapes/numeric identities, actual source hashes/HEAD/dirty output, destination checks; exact approved scope, inputs, clean commit and unused destinations still required. |
| Allocation/cost | Raw scontrol stdout/stderr, parsed actual job, original clock tokens/current clock and resource/rate/quote expectations; exact n01/4CPUs/1GPU/1200s/no-requeue/cost and launch-receipt bindings still required. |
| Management/mutex | Declared known-free paths, lstat mode/presence/read errors and HEAD; atomic mutex and known-free unopened/unreserved checks still precede GPU observation. |
| CSV/XML/kernel/cgroup | Existing CSV/XML responses, selected data/control kernel modes/rdev/major/minor, raw process cgroup memberships, resolved actual-job devices.list path/raw rules/read errors; current triggering rule number/type/major/minor/permissions, selected/control pairs and rejection rationale. Unsupported layouts and wildcard/other-data-device access still fail closed. |
| Runtime shell | Child Python/entry, phase leader PID/PGID, original runtime bindings and resolved CUDA/NVML driver paths; own-session/deadline/driver-file/container gates remain. Free shell callers have no evidence hook. |
| CUDA/PJRT preflight | Expected and actual CVD, backend/JAX/device properties, existing driver library/API attempts/result codes/count/device/UUID; exact UUID, driver single-device and PJRT GPU/local0 checks remain. No extra driver calls are added. |
| Supervisor/results/release | Original limits/deadlines, already-read raw preflight/result JSON, existing monitor replies/errors and release outcome. Process-list evidence keeps only row field counts/expected-UUID match, redacting PIDs/names even on partial timeout; full original response still reaches existing gates. Unverified release retains mutex. |

Evidence retention is bounded to declared inputs and an explicit environment
allowlist; it is not an environment dump. These records improve diagnosis
when a future separately authorized candidate runs. They do not prove GPU
readiness, isolation, successful cleanup, or scientific correctness now.
An early rejection can mean no Supervisor/child; a later preflight can initialize
GPU runtime while the affine science and optimizer remain unexecuted.

## Public policy interpretation for review

The saved [Slurm GRES documentation](https://slurm.schedmd.com/gres.html)
explains per-step CVD and inside/outside-cgroup ordinal remapping; NVML ordinals
and Linux device minors are separate, potentially nondeterministic mappings.
[CUDA environment documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)
defines UUID selection and visible-device enumeration, not OS access isolation.
[Linux device-controller documentation](https://docs.kernel.org/admin-guide/cgroup-v1/devices.html)
defines `a/c/b` device rules and read/write/mknod permissions in a hierarchy.
[Slurm cgroup.conf](https://slurm.schedmd.com/cgroup.conf.html) describes
ConstrainDevices; the current public default does not establish this site's
Slurm22.05.7 configuration. Saved HTTP200 bytes/text/receipts are in the
package's `official/` subtree (GRES lines308–360, CUDA79–99,
ConstrainDevices120–122; Linux rule syntax/open/mknod section).

Functional binding—approved actual allocation→physical UUID→child CVD UUID→
actual driver UUID/count→PJRT one GPU/local0—is distinct from the stronger OS
security assumption that other GPU data devices cannot be opened. A wildcard
permission is not evidence of computation on another card. The conservative
current predicate also rejects `c *:255 rw` when255 is a recorded control minor,
even if that rule does not intersect the selected data minor; that rule alone
does not identify access to another data GPU. `m`-only wildcards remain accepted.
This distinction is a review recommendation only: the predicate is unchanged.
Any future contract separation requires explicit review and execution authority.

## Bounded verification and package

Final CPU/private-fixture receipts:12 new evidence cases (`tests-new.v5`),
16 affected paid cases (`tests-affected.v3`),18 relevant shell cases
(`tests-shell.v2`),14 shared/free regressions (`tests-shared.v1`):60 passing
selected cases. No Slurm/GPU/driver boundary was called unmocked and no affine
scientific probe ran. Private system-TEMP fixtures were removed. Existing
pytest `cache_dir` warning remains. Initial new-test Windows path assertion
failures (`v1`,3failed/5passed) and the timeout-row missing error-type assertion
(`v4`,1failed/11passed) are retained with corrections; no gate was relaxed.

The [local package](../../../outputs/T7-paid-startup-remediation-20261012/)
contains exact test argv/timing/stdout/stderr, `gate-evidence-matrix.v1.json`,
actual diff/changed-path and syntax/link records, `preservation.v1.json`,
`review-package.v1.json`, and final `versioning.v1.json`/commit/push/clean receipts.
Outputs remain outside Git; code/tests, this report and compact current links
are versioned on `research/daily-coarse-graining`. Candidate sources differ
from frozen58a; old approvals cannot launch this candidate. No deployment,
refreeze, remote action, GPU experiment, training or new cost was performed.
