# Checkpoint4000 carbon endpoint attribution

Free gln01 GPU0 probe at immutable b2092a4 completed exit0,16/16 daily samples,
elapsed652.59s (~10m53s). Python82727 has exited. No AD, optimizer updates,
checkpoint writes or paid computation. All recorded hard counts zero, all720
compact stock/flow records complete. Arithmetic attribution residual below
9.3e-12gC/m2. These are the first8 fixed daily and first8 excluded-TRAIN daily
rows, covering7 point-years; they are not the prior64-sample panels or the
developmentT7/T30 windows.

## Decisive finding

Large litter outgoing amounts explain the one-day deficit. Panel mean amounts
in gC/m2; outgoing is a pool-endpoint bookkeeping decrease, not a physically
identified respiration/export flux.

| Panel | Owner | Teacher outgoing | Predicted outgoing | Teacher incoming | Predicted incoming | Stock error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fixed8 | Aboveground litter |0.003865|35.33638|0.048071|0.010304|-35.37028|
| Fixed8 | Belowground litter |0|25.88759|0.159191|0.419577|-25.62721|
| Excluded8 | Aboveground litter |0.001339|29.37542|0.037047|0.010436|-29.40069|
| Excluded8 | Belowground litter |0|25.03024|0.095806|0.420058|-24.70599|

Every one of the32 sample/above-below combinations has negative stock error.
The model removes approximately19.07%/19.35% of starting aboveground litter
per day (fixed/excluded means of stock-weighted sample fractions), versus
Teacher0.002099%/0.002880%. Belowground fractions are3.776%/3.821% versus0.
The model's outgoing range is18.91%-19.71% aboveground and3.61%-4.11% belowground
over these16 inputs. Incoming errors are orders smaller and sometimes partly
compensate the removal. Thus increasing incoming to cancel the drain would
hide the wrong flow rather than identify it.

Pre-tail and post-tail predicted carbon owner stocks are exactly equal in
this sample set. Teacher counterparts agree within1.23e-11gC/m2. The retained
tail did not introduce the deficit on these inputs; Teacher target alignment
also reproduces the saved one-day endpoint here. This is bounded evidence,
not a claim that the tail never changes these stocks under other conditions.

DOC also has excessive outgoing: fixed4.9666 versusTeacher0.5035, excluded
4.2276 versus0.7029gC/m2. Net errors are-4.2498/-3.6407. Particulate Teacher
stocks/inputs are near zero; predicted incoming~0.21 adds a small positive
artifact. Canopy incoming largely cancels an erroneous outgoing prediction,
leaving small net errors+0.1550/-0.0723. Accurate net stock alone therefore
does not certify correct individual endpoint directions.

## Interpretation and focused next candidate

The scientific failure is now localized to learned endpoint outgoing amounts,
especially litter. The prediction is numerically finite and positive, and the
existing bookkeeping budget passes, but learned dynamics remove carbon far
too fast. This does not implicate native AD or establish a need for another
Attention backbone. The active recurrent objective has no direct litter/DOC
term, and daily standing-stock/process losses have not suppressed this error.
How much comes from loss scaling, sampling, or output geometry still requires
a learning test; a forward replay cannot distinguish their gradient effects.

The next candidate should keep Attention, decoder, daily64/micro8, data splits,
optimizer and the admitted AD path unchanged initially. Add a focused physical
litter-change objective for aboveground and belowground pools separately,
normalized using TRAIN changes (not total inventory). Use the existing shared
PFT/element/area extraction and defined masks, with an explicit train-derived
scale policy for near-zero change. Retain the old loss and physical checks.
Do not tune scale/weight using this16-sample diagnostic or held-out results.

Before training, freeze the exact weight/scale rule and bounded comparison:
same declared checkpoint and identical TRAIN draws for old versus augmented
objective. First verify finite loss, gradients and a bounded complete update;
then a short learning screen should lower litter outgoing/net-change error
without degrading GPP/NPP/RA or canopy. A changed-loss native AD check should
target the new term, not repeat unchanged historical infrastructure audits.
Report single-day and7/30day transfer; do not infer improved GPP from corrected
litter alone. No changed loss implementation or further training has run yet.
Any decoder/backbone revision needs a separate explained design decision.

Checkpoint4000 can seed this causal diagnostic; it is not retroactively the
paper's selected model. A successful final recipe still needs a consolidated
fresh-initialization executable protocol. Original5415 pass remains incomplete;
do not quietly alter its objective when resuming or claim its endpoint reached.

## Evidence

- `outputs/mixed-carbon-endpoints-4000-v1.json`: complete raw output,80 owner
  records with masks, raw arrays and compact stock/flow summaries.
- `outputs/mixed-learning-review-pool-20260920/endpoint-summary.{json,md}`:
  per-sample additive decomposition and panel summaries.
  Independent Sol/high numerical review agrees with outgoing/incoming/tail
  attribution and all720 complete compact records.
- `scripts/hpc/summarize_carbon_endpoints.py`: read-only reproducible summary;
  Ruff passes, all80 decompositions close, no incomplete compact records.
- Probe source/test/launcher and earlier source audit are in
  `carbon_drift_attribution_20260920.md`. No active process or new paid job.
