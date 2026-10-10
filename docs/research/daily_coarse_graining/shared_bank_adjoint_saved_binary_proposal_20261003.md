# Saved bank-adjoint binary: proposed480s free acceptance

Status: user approved this additional480s free-GPU acceptance on2026-10-03;
it was executed and is **closed**. The two attempts used382.475999858s through
the final process exit,0.1062433333 free card-hours/CNY0. Both binary ABIs and
estimated memory passed, but no model values were taken: actual HLO exceeded
the executed64MiB text limit, and a same-deadline artifact repair then hit an
import-time compile guard. The original deadline has elapsed; this command has
no remaining allowance. The precision choice and original T7 diagnostic were already
approved. The previous route used its3600s cap after completing the backend and
saving the binary; the new request is bounded time to accept that exact binary.
It did not build another candidate, change scientific gates or adopt a reference.
See `shared_bank_adjoint_accumulation_results_20261003.md` for actual evidence.
The repaired resident-executable600s route is specified separately in
`shared_bank_adjoint_resident_acceptance_proposal_20261003.md` and remains
unapproved. The sections below preserve the executed480s configuration.

## Exact executable and command

Reuse
`runtime/outputs/diagnostics/shared-bank-adjoint-accumulation-T7-20261002/model/candidate.executable.pkl`,
220,794,649 bytes, SHA256
`f0e0a5eabd924533f94f8a636b173b433d61e82d20ff0cb21c326d7d89b99b01`.
The staged entry points are `scripts/hpc/probe_shared_bank_saved_binary_T7.py`
and `scripts/hpc/run_shared_bank_saved_binary_T7.py`/`.sh`, under the existing
bank-adjoint transfer overlay. After approval, the exact command is:

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-bank-adjoint-accumulation-20261002/run_shared_bank_saved_binary_T7.sh --gpu 1 --candidate-sha256 f0e0a5eabd924533f94f8a636b173b433d61e82d20ff0cb21c326d7d89b99b01
```

Working directory: immutable
`runtime/checkouts/shared-split-width-v1` at
`c599236bcf57b745e9bafed678298010f4c3b58f`. Unique output:
`runtime/outputs/diagnostics/shared-bank-adjoint-saved-binary-T7-20261003`.
Use free gln01 physicalGPU1 V100, canonical JAX0.4.38/foundationpose environment,
OMP4/OpenBLAS1 and the already specified LLVM flag. Check GPU1 is empty and record
node, load, affinity and visible processes; no Slurm paid job or paid fallback.

## Budget and execution boundary

The whole additional family is **480s including setup, deserialize, structural
proof, original/candidate calls, host comparisons, ordinary repairs and10s
TERM/KILL grace**. At most0.1333333333 additional free card-hours/CNY0. The model
subprocess cap is the shared remaining time after setup, at most470s. There are
zero new model lowerings/backends and zero new runtime preflights. Reuse the
completed preflight and original saved result/executable/JVP evidence.

The observed original setup/replay path reached replay after about101s, and its
warm original T7 call took2.775s. Candidate deserialize and compact export have
not been timed;480s is bounded headroom, not a completion guarantee. Explicit
stage caps are90s per binary deserialize,75s for compact actual HLO capture/proof
and30s per model call, intersected with the same remaining cap. The assets stage
has120s headroom; all maxima are alternatives within480s, not extra allowances.

The replay code constructs no objective and calls no `.lower`, `.compile`, `jit`
or `value_and_grad`. Both JAX `compile_or_get_cached` and `backend_compile` are
replaced with hard refusal during deserialize/proof/evaluation; zero model
compilations and any forbidden call are recorded. Source/asset identities,
candidate digest, actual input treedef/shapes/dtypes,300 parameter leaves and
64,884,481 coordinates, T7/artifact742/update0 and3854 initial states are checked.
The `--root` option cannot select a different source family.

Before GPU initialization, also pin the original raw compile report to SHA256
`95357c6d5acaee22fd0440bf629ee9add20aaa4243f16539d175c108bf1bcddf`.
Validate its exact source paths/hashes, transformed clone, four gather rewrites,
scope markers, helper/alias/globals identities, x64 flag and all seven complete
applications with56 valid FP32-to-FP64 bank records. The report's policy status
is `active` because it was captured inside the temporary source-patch context;
every application is `complete`. Pinning the entire report rejects reduced
application inventories before they can weaken the structural proof.

An ordinary runtime/proof-interface repair can use the **same command plus
`--resume`**. It preserves prior full artifacts/exits and the original target-clock
start/deadline; it deducts time already spent, keeps the exact digest and refuses
an exhausted allowance. A hard/scientific terminal result cannot be resumed.
No second binary or compile is allowed.

## Bounded proof and unchanged acceptance

Before candidate values, obtain actual optimized runtime modules through
`runtime_executable().hlo_modules()` and `HloPrintOptions.short_parsable()`.
Retain metadata/named scopes, SSA operand names, percent signs/IDs and result
types; omit large constants, backend configuration and operand-shape duplication.
Save at most64MiB of compact plain text and run the scoped field/FP64-scatter/
outer-carry audit. No `candidate.as_text()` or full-text gzip export is used.
An independent stdlib watchdog process enforces the75s stage limit even if a
native call holds Python's GIL. Its timeout artifact maps termination to124.
Incomplete proof stops before values and remains an implementation-evidence gap,
not a scientific failure. Per-field/deduplicated-shape proof is not exhaustive
for repeated same-shaped block occurrences; source guards/application metadata
and complete arrays remain required.

The original loss/auxiliary/parameter bridge, original self-drift recording,
all strict parameter/state comparisons (`rtol=1e-5, atol=1e-6`), finite values,
all14 hard zeros,112 source/science masks,21 exact-zero undefined gradients and
estimated/observed24GiB admission are unchanged. Save the first complete candidate
and three synchronized interleaved full-result pairs, even if a numerical
comparison fails after hard admission. Hard failure saves and stops immediately.
Reuse original JVP coordinates33/199/428/200 as supplementation only. No candidate
JVP, optimizer update, T30, tolerance waiver or reference adoption is proposed.

## Preparation evidence

`outputs/diagnostics/shared-bank-adjoint-accumulation-preparation-20261002/`
`saved-binary-preparation/` contains the actual test01 dry-run, byte/digest metadata,
canonical CPU compact-format HLO/check JSON, focused interface/signal/resume
checks and final source snapshots/readiness. The canonical JAX0.4.38 CPU real
HLO printer output passes the same structural parser; this validates format,
not the GPU candidate. Seven focused CPU checks cover no graph/compile calls,
dual compile refusal, root refusal, actual watchdog SIGTERM, timeout124 mapping,
same-deadline resume and expired-resume refusal. Preparation has initialized no
new GPU or deserialized/executed the candidate.

Fourteen additional focused CPU provenance cases passed, covering the actual
pinned report and rejection of altered report bytes, reduced applications,
missing/duplicate banks, changed source/rewrite/scope/precision evidence and
invalid counts/elements. The existing independent reviewer closed the provenance
finding in the actual family's `saved-binary-source-review.json`. Final
`readiness.json` and `artifact-manifest.json` record the seven staged source
hashes, actual target dry-run and both CPU evidence families. The original
family's manifest verifies all four executed-source snapshots against
`actual-exit.json`; `summary-cleanup.json` records retained artifacts and no
deletions. These readiness records describe preparation before the subsequent
explicit approval; actual execution evidence is recorded separately.
