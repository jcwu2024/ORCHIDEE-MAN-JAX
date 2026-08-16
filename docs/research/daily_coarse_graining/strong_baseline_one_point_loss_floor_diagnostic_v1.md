# Strong-Baseline One-Point Loss-Floor Diagnostic V1

Status: frozen for local implementation on 2026-08-17.

## Question

The first complete one-point-year qualification reduced the process objective
by `97.6026%`, but it did not reach the required `99%` total reduction and four
of seven experts did not reach `95%`. This diagnostic asks whether that result
is mainly a stochastic-minibatch optimization floor or whether important
labels remain difficult even under an exact gradient over the complete
point-year.

This is not an architecture comparison and does not authorize a new network,
64-point-year training, validation rollout, or paid compute.

## Frozen Data And Model

- Use the accepted `pft14-daily-teacher-v6` release only.
- Use landpoint `1`, year `1961`, split `train/train`, with all 364 valid days.
- Start the 49,750,681-parameter
  `axis_aligned_process_expert_operator_v1` from the same fresh seed as the
  failed qualification.
- Keep `message_gate_scale=0`; the graph path remains unexecuted.
- Do not regenerate Teacher data, read v5, or initialize from any checkpoint.
- Enter through the canonical `bind_daily_operator(...)` boundary for science
  and hard-gate evaluation.

## Stage A: Reproduce The Bounded Qualification

Run the already frozen one-point recipe:

- 100 complete shuffled epochs;
- batch size 8, with zero-weight padding in the last batch;
- 4,600 maximum AdamW updates;
- 2% linear warmup and cosine decay from `1e-4` to `1e-5`;
- weight decay `1e-4`, with biases and normalization scales excluded;
- no gradient clipping.

If all qualification gates pass before epoch 100, stop without running the
loss-floor tail. Otherwise continue from the Stage-A parameters.

## Stage B: Exact All-Day Gradient Tail

Run at most 25 deterministic full-point updates. One update is the exact mean
gradient of the Stage-A process objective over all 364 valid days:

1. evaluate 46 microbatches of size 8;
2. multiply each microbatch mean gradient by its number of real samples;
3. sum those gradients and divide once by 364;
4. apply one AdamW update with a fresh tail optimizer state and fixed learning
   rate `1e-5`.

The reset prevents stale stochastic second moments from defining the tail.
It means the diagnostic distinguishes a general optimization floor from an
apparent capacity floor; it does not attribute every improvement solely to
batch variance.

Every tail update must report exact sample coverage, microbatch count, the
exact gradient norm, the real-sample-weighted mean microbatch gradient norm,
their ratio, process loss, and all nonfinite counts. Stop early if every
qualification gate passes. Stop immediately on a nonfinite value or coverage
failure.

The 25-update ceiling adds at most 1,150 microbatch reverse passes, equivalent
to 25 more complete passes over this point-year. It does not put all 364 days
in GPU memory at once.

## Loss-Floor Evidence

The machine report must compare fresh initialization, the end of Stage A, and
the final state. For every expert, label, value field, and defined-status BCE
field, report:

- mean daily loss;
- active-day count;
- defined scalar count;
- reduction from initialization;
- reduction during the exact-gradient tail.

Counts must be unchanged between evaluations. The report must keep all fields,
including apparently easy fields, so an aggregate cannot hide a local floor.

## Corrected Science Metrics

GPP, maintenance respiration, and NPP retain their accepted train-only
normalization. AGB and BGB one-day errors are differences of next-day stocks
from a shared day-start state, so they are tendency errors. Normalize them by
the accepted train/train one-day tendency scales:

- AGB: `1.3522777765283391`;
- BGB: `0.9474426729431713`.

The runner must load and validate the accepted scale asset through
`load_named_biomass_tendency_scale_asset`; values copied into this document are
not an alternate runtime source.

## Gates And State Retention

The absolute qualification gates remain unchanged:

- total process-loss reduction at least `99%`;
- every expert reduction at least `95%`;
- GPP, maintenance respiration, NPP, AGB tendency, and BGB tendency NRMSE no
  greater than `0.05`;
- zero nonfinite predictions or defined next-state values;
- zero water/carbon budget failures and negative inventories;
- finite gradients throughout.

Write one final parameter file after a completed diagnostic so later
attribution does not require another 100-epoch run. It must contain parameters
but no optimizer state and be marked `diagnostic_only=true`,
`formal_training_initialization_allowed=false`, and
`promotion_checkpoint=false`. A scientific gate failure is a completed
diagnostic, not a shell/job failure. Execution or identity failures remain
process failures.

Passing opens only the design of the 64-point-year gate. Failing keeps that
gate closed and uses the label/field table to decide whether the next change
belongs in representation, loss construction, or optimization. No 7/30/365
day rollout, long-held point, Gate D2, sealed test, or product acceptance is
opened by this diagnostic alone.
