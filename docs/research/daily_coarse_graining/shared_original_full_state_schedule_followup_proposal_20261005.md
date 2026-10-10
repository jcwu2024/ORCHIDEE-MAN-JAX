# Original schedule continuation — approved proposal record

Status: subsequently **approved and completed** on 2026-10-05. The user replied
“批准”; four-pair continuation settled at 897.671305 GPU-seconds with full
preservation and release. See
shared_original_full_state_schedule_followup_results_20261005.md for actual
timing and the severe state AD limitation. The remaining text preserves the
reviewed proposal; it does not authorize another window. The
proposal was one free gln01 window using two V100-SXM2-32GB cards for at most
900 occupied seconds, **1,800 additional GPU-seconds (0.5 card-hours)**,
worst-case paid cost **0 CNY**. It does not authorize paid fallback, formal
training, model-gradient compilation or scientific changes.

## Concrete command and unchanged scope

Working directory:
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1,
immutable c599236bcf57b745e9bafed678298010f4c3b58f. The declared overlay is
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-latent-reader-20261004.
Actual free node is ibc13b02n04; launch must recheck both GPUs idle and record
node, CPU affinity, host load and processes. The wrapper contains the entire
startup, load, execution, comparison, preservation and cleanup window.

After explicit new resource approval, the HPC owner records its exact note:

```bash
ORIGINAL_SCHEDULE_EXTRA_APPROVAL_NOTE='<actual new user approval>' \
  bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-latent-reader-20261004/run_shared_original_full_state_schedule_extra.sh
```

The target-shell receipt retains the exact expanded budget/container/Python
command. It requests the same original old/new schedules, full daily32 with
eight distinct micro4 slices, complete T7/T30, full parameter and initial-state
native AD, public AdamW, physical/science counters and unchanged strict
rtol 1e-5 / atol 1e-6. Parent GPU0/worker GPU1. Outer900 seconds, inner880 seconds,
save/exit reserve 30 seconds, four additional real paired changing updates,
zero warmup proposals. New output stem:
shared-original-full-state-schedule-resume-20261005; existing results remain.

Each schedule resumes its own retained step 1 parameters and AdamW moments,
advancing step 1→5. The original update0 checkpoint remains provenance and the
tree template. True artifact IDs are [742,275]; continuation starts with275,
then742/275/742. Cumulative attempts1→5 per chain and the same16-attempt cap
are retained. Inherited strict failure is sticky. The first new pair is
reported separately; successful timing completion requires all four additional
pairs and at least three post-first samples. Partial evidence is preserved.

## Why 900 seconds is a bounded plausible allowance

The completed window measured four executable loads totaling 304.305 seconds
and a startup lower bound 308.685 seconds. The cold first pair, including
comparison/download/save, took76.002 seconds. Four such pair costs are304.009
seconds. Allowing roughly 100 additional seconds for loading two743 MiB retained
chains and other lifecycle overhead, plus 30 seconds preservation reserve and
20 seconds external margin, fits within900 seconds. This is a planning bound,
not a runtime guarantee. The measured first 6.997× ratio is not used as a
steady-state speed prediction. The same conservative next-pair gate may stop
early; extra compilation or resource renewal is not automatic.

New stage_events record actual emitted boundaries, parameter-array download
completion, each chain-save completion and resource-close events. Old missing
compare/save timestamps are not fabricated. Timing, runtime/resource failure,
strict numerical failure and scientific admission remain separate outcomes.

## Separate funding source and verified guard

The one persistent ledger currently has 15 settled sessions, total 21672.648975,
ceiling 25200 and global remaining 3527.351025. Original 18,000 source remaining
555.738955 and fixed original1,000 route remaining 199.181077 cannot fund a
900-second two-card window. The daily-only7,200 source retains 2971.612070 and
cannot be transferred. Global headroom is not route authorization.

New proposed scope original_schedule_additional_1800_20261005 and route
original_full_state_schedule_extra_1800_20261005 require a unique actual future
authorization record 25200→27000, increment 1800, explicit note and timestamp.
All original baseline IDs, historical charges, existing authorization and
source/route caps remain. Only actual authorized wrapper invocation applies
that increment and reserves the new session. No shadow ledger or reset.

Local/remote preparation tested a deep-copy hypothetical authorization only;
the actual ledger remained byte-identical with ceiling 25200. Without the new
record the extra source is rejected, despite global headroom. The real
launcher rejects the PENDING/PROPOSAL note marker before reaching the budget
wrapper. Missing note is also rejected.21 budget CPU tests passed;30 resume
CPU tests passed. Root independently reviewed source semantics.

## Actual external asset and preparation verification

On compute host ibc13b02n04, CPU-only container execution without --nv,
JAX_PLATFORMS=cpu and empty CUDA_VISIBLE_DEVICES admitted both actual retained
778,644,397-byte step 1 pickles. Each has 300 FP32 parameter leaves and 64,884,481
coordinates, matching original full tree/shape/dtype/provenance, finite
parameters/AdamW and nonnegative second moments. Both public AdamW steps are1;
115,661 parameter coordinates differ across chains and remain independent.
Next artifact 275 and inherited strict false were verified. CPU check elapsed
12.015 seconds; no CUDA/model-gradient call or ledger mutation. Actual resumed
GPU execution still repeats its mandatory admission checks.

Deployed parent/worker, budget guard, launcher and resume manifest hashes match
local files. Actual target Bash syntax and --dry-run expansion passed, including
900/880/4pairs/1800 source. Pure metadata CLI passed against actual remote
reports and paths; it does not itself deserialize chains. No GPU launch or new
allocation occurred during preparation.

Receipts under outputs/diagnostics/shared-latent-reader-20261004:

- original-full-state-schedule-resume-cpu-admission.json
- original-full-state-schedule-followup-preparation.remote.json
- original-full-state-schedule-followup-budget-plan.local.json
- original-full-state-schedule-followup-staged-sha256.txt
- original-full-state-schedule-followup-launcher-dry-run.txt
- original-full-state-schedule-followup-metadata-dry-run.json
- original-full-state-schedule-resume.manifest.json

## Preservation and bounded storage

Four new full gradient-pair artifacts are estimated8.305 GB, final independent
chains1.557 GB, state evidence and pickle overhead small. If every pair requires
full failed-update parameter/AdamW snapshots, up to approximately 6.23 GB extra
may be retained; the array estimate is therefore approximately
16.1 GB. The declared storage envelope is **approximately 17 GB**, including
state/pickle/metadata margin, with fewer than 20 consolidated outputs.
Current observed personal soft quota is
already exceeded within its grace; shared filesystem free capacity is not
personal allowance. Files remain consolidated, roughly fewer than20 new
artifacts, rather than per-coordinate/sample files. No unique scientific data
or current outputs will be deleted. Current grace and file-count evidence
support this preservation choice; normally recheck them before actual execution.
No new storage purchase is included.
