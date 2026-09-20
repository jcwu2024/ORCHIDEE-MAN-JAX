# Direct-joint prepared-pool run: interrupted result

Job15279139 ended FAILED124 at the declared950-minute inner timeout, not a
recorded numerical failure. Slurm elapsed15:50:03, start2026-09-19 16:54:55,
end2026-09-20 08:44:58;2V100/8CPU, immutable d1183e6. Estimated additive cost
78.54yuan using the approved4.96yuan/hour rate, not an invoice.

## Completion and recoverable work

The final report records4643/5415 updates,297152 daily source draws without
padding (85.75% of the346539-row ready pool), and4643T7 plus4643T30 windows.
This is not a whole-Teacher epoch. Recovery metadata records update4600 and
the1,313,711,368-byte file exists on the server; it was not reloaded this turn.
The43 later recorded updates are not checkpointed. Recovery would need815
updates to reach the declared5415 endpoint. No continuation has been submitted.

The last JSON status remains `running/training` because timeout prevented final
wrap-up; scheduler/log evidence controls actual job status. Monitors exist at
0,200,600,1000,2500,4000. No science inference is made for4600/4643, and final
5415 monitor/persisted-forward replay did not run. All427156 recorded training
hard-counter entries are zero. Recorded development physical gates and daily
monitor hard checks pass; scientific acceptance remains false.

## Scientific result

Same reused development panel:8T7+8T30 windows; not independent validation.
The table reports T30 RMSE in gC/m2, accumulated flux or stock **change**.

| Update | GPP | NPP | RA | AGB growth | BGB growth | Canopy change | Nonliving change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|600|16.77|13.78|4.74|9.07|4.82|2193.68|8736.67|
|1000|11.59|10.07|2.92|6.47|3.50|195.53|5839.38|
|2500|8.20|6.39|2.14|4.26|2.13|60.25|4995.20|
|4000|9.30|7.41|2.13|4.83|2.57|19.00|4588.68|

This is substantial learning, especially canopy carbon, but productivity and
growth errors rebound after2500 while canopy/nonliving continue improving.
Excluded-TRAIN daily loss falls1.031 ->0.0823 ->0.0389 ->0.0317 over these
milestones; its decrease alone cannot select a scientifically good checkpoint.

Update2500 passes the configured development gates only. At4000 both horizons
fail NPP signed-bias gates: normalized+0.13218 exceeds0.10(T7), and+0.08425
exceeds0.07(T30); their RMSE gates pass. These normalized biases are not
percentages of each Teacher window's actual NPP.

At4000, mean T30 prediction versus Teacher (gC/m2): GPP56.78/52.36,
NPP20.90/17.24, RA35.88/35.11, AGB growth3.42/1.17, BGB growth1.54/0.15,
canopy change119.87/136.28, nonliving change-2022.65/+125.27. Mean values may
cancel window errors; the full bundle retains per-window comparisons and RMSE.
Nonliving change has the wrong sign in all8 windows at each horizon at both
2500 and4000. Inventory-based normalization (~47624gC/m2) can pass despite
large change errors. Preserve existing gates, but do not interpret a pass as
accurate growth/carbon dynamics. RH Teacher values are all zero on this panel;
zero RH errors do not establish nonzero RH accuracy.

## Runtime and interpretation

Python last elapsed15.815h; timed updates10.047h, remainder5.768h includes
startup/compilation, monitoring, saving, full-report formatting and bookkeeping.
Do not label all remainder IO or wasted compute. Monitors total370.48s. Daily
artifact read counter690.73s overlaps other timers.

Updates2-4643 median7.558s/mean7.784s. Ordinary adjacent cycles excluding saves
and monitors average11.852s, with4.075s outside the update timer. Earlier13-15h
forecast undercounted growing overhead. Reporting revision16c0bad was tested
separately and never applied to this job; its whole-training speedup is unmeasured.

## Next work

Preserve all milestones and recovery. Do not silently select2500 as the formal
model, change the declared5415 endpoint, or claim that completing815 updates
will cure the science problems. A paid continuation needs a concrete resource
proposal and approval. If using the reporting revision on recovery, explicitly
check compatible provenance/method handling; do not relax identity validation.

The next scientific question is why nonliving carbon changes have the wrong
sign and why production/growth improve to2500 then rebound. Start from saved
per-window trajectories and the current loss terms; distinguish data coverage,
loss scaling and checkpoint variability before proposing another architecture.
Retain2500 and4000 as diagnostic contrasts, with a broader independent7/30day
screen required before any365day or long-term claim. No new training/loss/AD
implementation, Teacher generation, GPU inference, or paid submission this turn.

## Evidence

- `outputs/mixed-learning-dual-pool-15279139-final.json` and `.log`: raw snapshots.
- `outputs/mixed-learning-review-pool-20260920/`: reproducible science/timing
  summaries and10figures from existing tools; source JSON status is stale as
  explained above. Figures reviewed; independent Sol/high science review agrees.
- Remote recovery/milestones remain under
  `runtime/outputs/training/mixed-learning-dual-pool-v1/`.
