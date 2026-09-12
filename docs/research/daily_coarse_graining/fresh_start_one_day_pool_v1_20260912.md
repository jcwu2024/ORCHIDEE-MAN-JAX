# Broad one-day learning v1: one reproducible prepared-pool epoch

Frozen implementation/preflight plan. Paid execution is proposed, not submitted
or approved. This proceeds independently of the public multi-day derivative
probe; it uses ordinary direct JAX AD through one canonical daily transition.
Executable settings: `manifests/coarse_graining/fresh_start_one_day_pool_v1.json`.

## Evidence and purpose

The full fresh model passed the12-update v3 learning screen on three repeated
samples. Batch4 numerical/resource comparison at4410c41 then passed all hard
checks and all109,464,375 gradient coordinates. Gradient L2 difference versus
the mean of four single gradients is6.61e-7. Actual peak active allocation is
7.85GiB, versus20.63GiB allocator pool. A hot batch4 gradient takes0.400s;
four single gradients took0.930s including their first cold call. Batch/single
compilations took957.90/953.13s respectively; total probe2144.28s.

This establishes a usable microbatch, not learning on the broader data. Test
whether the unchanged clean method improves fixed train and excluded-day
panels when learning across the available training pool.

## Method

Fresh seed20260911, full109,464,375-parameter Attention tree, DOC log-tail16,
bare-soil gsmean exact zero. Reuse v3 losses, train statistics and AdamW
(learning rate1e-5, clip norm1, zero decay). Average the per-sample v3 losses;
do not replace them by a batch-global bias formula. Use8 microbatches of4,
weighted by actual row count, then clip and perform exactly one update.
If a remainder occurs it is weighted and retained, not padded or dropped.

The existing extension's prepared one-day pool provides63,936 row draws,
63,707 unique Teacher days,535 points,17,677 point-years and44 years after
excluding every occurrence of the64 catalogue monitor days. Seed20260912
selects reproducible shuffled package/row epochs, at most4 packages per batch.
One prepared-pool epoch is1998 updates. It is not an epoch of all v6 data.
No Teacher regeneration, sealed validation/test access, or inherited weights.

## Admission and monitoring

Run local accumulator/AdamW/resume/monitor tests, then real CPU tracing of the
batched gradient, optimizer and monitor. A separate bounded free GPU technical
smoke starts fresh and performs the first2 declared updates. It measures the
complete training loop and checks numerical/physical validity, not two-update
scientific accuracy. Its checkpoints are marked technical_smoke and cannot be
resumed as the formal one-epoch run. The formal run starts fresh by default.

Monitor the fixed first training batch and all64 excluded days at updates
0/100/500/1000/1500/final. The excluded panel is disjoint by Teacher day only,
not by site/year, and is not a generalization acceptance panel. Save raw daily
prediction/Teacher values and summarize RMSE, MAE, bias and absolute-error
quantiles in physical units. Include GPP, NPP, maintenance/growth/autotrophic/
heterotrophic respiration, biomass levels and one-day biomass changes.
Do not infer biomass increment quality from accurate absolute stock levels.

Every sampled forward and monitor must pass existing physical checks; every
aggregate gradient and optimizer proposal must be finite. Stop on failure with
the offending row identities and the latest complete checkpoint. Save parameters,
Adam moments/step and sampler cursor atomically at initialization, every250
updates, monitor milestones and final. Resume uses the original report to
retain initial-panel evidence and checks the declared method, not bulk file
hashes. Checkpoint selection is the final complete update, not the best monitor.
Recovery binds the cursor to the canonical in-memory package/row metadata
identity; it never rehashes data payloads. It also requires Adam step, completed
updates and the bounded epoch cursor to agree. Publish the audit report before
the atomic checkpoint; recovery truncates any newer report entries. Numerical
method changes must update the explicit implementation version. The actual
execution Git revision remains in the report for code reproduction; documentation
commits alone need not invalidate recovery.

The one-epoch learning screen requires both mean total loss and mean independent
flux loss to fall on both fixed panels, with physical gates intact. Report
individual flux/stock regressions even when this screen passes. This screen
does not establish scientific accuracy or long-horizon stability. Full-network
7/30-day training remains gated on independent multi-day derivative admission.

## Resources and executable path

Runner: `scripts/hpc/run_fresh_start_one_day_pool.py`. Use the existing GPU
wrapper with `PYTHON_MODULE=scripts.hpc.run_fresh_start_one_day_pool` and supply
`--catalogue-report runtime/outputs/diagnostics/one-day-catalogue-v1.json`.
The free technical smoke adds `--smoke-updates 2`,45-minute limit.

Measured gradient arithmetic suggests1998*8*0.400=6394s (about107minutes)
per epoch, excluding IO, optimizer, compilation, monitors and checkpoint writes.
This is a planning estimate on a shared node, not accepted throughput. Proposed
formal allocation is one V100,8 CPUs,32GiB host memory,4 hours: CPU2.24+GPU8.80
=CNY11.04 worst case assuming additive rates. Confirm the full-loop smoke and
present the exact Slurm command before requesting paid approval. Do not submit
until approved. The existing cross-day test must not be duplicated or disturbed.

## Local and real-data execution record

Eleven local tests pass, including the real runner with a tiny model interrupted
after a durable update, then resumed: parameters/Adam moments/step exactly match
the uninterrupted run, with no repeated/missing batches. Review additionally
checks cursor bounds/count agreement, row semantics and audit/checkpoint order.
Real CPU gradient/optimizer/monitor shapes passed on test02 at fa05bc1 in277.7s;
report `runtime/outputs/diagnostics/one-day-pool-shapes-v1.json`.
The12bac94 follow-up modifies recovery bookkeeping and host-memory reporting,
not the numerical graph. The free GPU0 two-update smoke started at that commit
in isolated `runtime/checkouts/one-day-pool-v1`, server14:39:55, PID13362,
45-minute limit. Result pending:
`runtime/outputs/diagnostics/one-day-pool-smoke-v1.json` and `.log`.
No paid job has been submitted.
