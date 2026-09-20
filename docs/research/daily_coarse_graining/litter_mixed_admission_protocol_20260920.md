# Balanced litter supervision: bounded joint learning

Frozen before execution2026-09-20. Implement/test on the free execution path;
no paid job or original-run resumption. Prior daily admission22e7fb9 establishes
finite meaningful new-term gradients and useful local litter improvement,
but weight0.1 dominates daily gradients and production does not improve.

## Question and fixed mechanism

Does directly supervising separate above/below daily litter carbon change,
at a weight balanced against the original joint gradient, reduce recurrent
carbon drift while retaining GPP/NPP/respiration and biomass-growth accuracy?
Keep Attention/decoder,109,464,375 trainable parameters, input scaling, existing
TRAIN data, and all original daily/R7/R30 terms. Keep AdamW LR1e-5/clip1 and
source4000 AdamW moments. This is a diagnostic fork of source4000; no claim
that this checkpoint history is a reproducible final fresh training method.

New objective is D_old + lambda L_litter + 0.5 R7 + 0.5 R30. L_litter is the
already admitted separate-owner pseudo-Huber daily change loss, with existing
TRAIN coordinate-variance scales0.09929366/0.05959626gC/m2/day and floor0.01.
No litter term enters the recurrent objectives in this experiment. Old matched
arm uses lambda0; both arms have identical starting parameters/optimizer and
training draws. They test the supervision intervention, not a claim that the
old model is scientifically adequate.

## TRAIN-only coefficient rule

Before ANY update, calculate four gradient components at fixed source4000
parameters: original daily64, unweighted litter daily64, R7, and R30. Use two
declared calibration batches/pairs, original schedule indices4000 and4001.
These are existing train/train data; no new teacher data or test-set access.
Accumulate daily means in micro8. Reuse native checked recurrent AD; do not
construct a new reverse pass or rerun the already admitted full daily JVP.

Record a4x4 Gram matrix per batch, including signed inner products. The old
mixed gradient is g_old=g_daily+0.5g7+0.5g30. Its norm must include cross terms,
not a sum of norms. Define r_i=norm(g_litter)/norm(g_old), then choose ONCE:

    lambda = min(0.5 / median(r_i), 1.0 / max(r_i), 0.001)

This targets a median added-gradient norm at most half the old mixed norm,
with no calibration batch exceeding the old mixed norm. Record component
norms, selected weight and old/new gradient cosines. Old mixed and litter norms
must be finite and nonzero; an individual recurrent gradient may be zero.
An unresolved calibration is a reported result, not a reason to
invent a coefficient. This norm heuristic does not guarantee accuracy or
absence of gradient conflict. It is limited to two TRAIN batches and must be
judged by the ensuing learning/rollout results. Never select lambda from
development improvements. Freeze it for every update in the experiment.

The manifest retains graph instrumentation weight0.1 to reuse the admitted
four-term executable. Actual daily coefficients are[1,.1,.1,lambda], not the
instrumentation weight: the fourth term is unweighted and multiplied once.

## Learning and assessments

Use single free GPU initially and the canonical shared gradient combination
and optimizer. Each arm uses indices4000..4019, daily64/micro8 plus oneT7 and
oneT30 per update. Thus1280 daily draws +20 windows of each horizon per arm;
the arms share examples, not double unique coverage. Calibration reuses the
first two batches without modifying parameters. This is a short diagnostic,
not a full-pool epoch or final-model training.

First two complete joint updates per arm are the integration milestone:
finite gradients/parameters/AdamW, valid state/mask/physical counters and
correct optimizer progress are required. Save that diagnostic state. Once
it passes, continue in the same process to20 updates to amortize compilation
and obtain more informative learning evidence. Keep source checkpoints
unchanged. Preserve final per-arm checkpoints BEFORE potentially expensive
post-training monitoring. Final models are diagnostic assets, not accepted
production models.

Evaluate one shared initial model and each final arm using existing fixed8
and excluded8 daily TRAIN panels plus the unchanged8T7/8T30 development
windows through `build_mixed_rollout_monitor`. Report physical GPP/NPP/RA/RH,
AGB/BGB growth, canopy/nonliving carbon changes and existing science gates.
No reduced panel may claim the strict16-window screen. These are reused
development windows, not sealed final-test acceptance; all-zero RH Teacher
targets retain their usual limitation. Any stock improvement with worsening
production is reported as a tradeoff, not success. Model selection/scale-up
remains a separate decision after seeing all physical outputs.

## Execution and artifacts

Manifest `manifests/coarse_graining/litter_mixed_admission_v1.json`;
launcher `scripts/hpc/run_litter_mixed_admission.sh`.
One idle free GPU,90min maximum. Existing actual-runtime smallcheck and daily
AD evidence are reused because the graphs and state boundaries are unchanged.
Target-shell dry-run and focused local tests before launch. No automatic paid
fallback, long retry, or extension.

Store two consolidated calibration bundles (one per batch, four gradient trees
each), not one file per parameter. Save each arm at update2 and20: four model/
optimizer containers. Expected data roughly9GiB plus compact report/log,
under10 new output files; no existing data deletion. Completed gradient
components should be preserved before compiling another expensive component
so timeouts do not discard them. Reuse requires exact source method/row
identity and the unchanged loss, not bulk dataset rehashing.
If90min interrupts monitoring, saved trained states support a forward-only
completion later; do not repeat completed updates to get a report.

## Prelaunch validation

Implementation and shared-path checks pass44 focused local tests, Ruff and
diff checks. Coverage includes matched component arithmetic after JAX mapping
reordering, train-only coefficient selection, recoverable gradient bundles,
daily litter masks/PFT derivatives, shared gradient accumulation/checkpoints
and physical monitor classification. This is local integration evidence;
actual full mixed GPU admission remains the purpose of the bounded run.

## Launch record

Started2026-09-20 14:51:12CST on gln01 (ibc13b02n04) idleGPU1 from immutable
commit1391a75 in `runtime/checkouts/litter-mixed-admission-v1`. Target-shell
dry-run and actual GPU-container CLI parsing/import check passed first.
Wrapper101505, timeout101520, container starter101521, Python101558;
90min ceiling about16:21CST.
Outputs `runtime/outputs/diagnostics/litter-mixed-admission-v1.{json,log}`.
Initial launch log confirms checkout/GPU and metadata validation passes;
model loading was in progress at the last check. No paid submission and no
scientific result yet.
