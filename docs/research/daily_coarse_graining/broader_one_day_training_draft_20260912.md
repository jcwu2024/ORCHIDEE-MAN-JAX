# Broader fresh-start learning: decision draft after the resource probes

This is not an approved execution contract or a submitted job. The current
batch4 probe determines feasible microbatch size and measured wall time before
the update budget is frozen. Keep the public multi-day derivative admission
independent: one-day learning uses ordinary JAX automatic differentiation.

## Scientific question

Can the clean, jointly trainable Attention model learn productivity,
respiration and next-day state across the existing broad training pool,
instead of repeatedly fitting the three prototype samples? The12-update
prototype establishes only that local learning is possible.

## Fixed method proposed for the next contract

- Fresh seed20260911; all109,464,375 neural parameters trainable. No exploratory
  checkpoint inheritance. Retain DOC log-tail16 and exact bare-soil gsmean.
- Existing v6 statistics and the one-day extension catalogue only. Its63,936
  eligible row draws represent63,707 unique Teacher days across535 points and
  44 years. An epoch here covers this prepared pool, not the full v6 release.
- Reproducible epoch shuffle from seed20260912, bounded mixing of four packages.
  Effective batch32; choose microbatch4 only if the real resource/equivalence
  probe supports it. Otherwise use a smaller microbatch with weighted gradient
  accumulation. Clip and apply AdamW once per effective batch, after averaging
  all its samples. Preserve every remainder row.
- Preserve the declared v3 per-sample objective and average it over the batch:
  four independent daily carbon-flux MSEs, state anchor0.1 and necessary
  process supervision0.01. Do not change the loss while testing data expansion.
- Use the existing AdamW rule (learning rate1e-5, clip norm1, no weight decay)
  for the bounded first run. Clipping frequency is a diagnostic; it is not by
  itself an acceptance/rejection criterion.
- Save parameters, Adam moments, completed-update count, and sampler
  epoch/batch-offset together. Resume must reproduce the uninterrupted next
  batch and optimizer step. Temporary restart checkpoints do not redefine
  the scientific initialization or sampling method.

## Reporting and stopping to freeze before launch

Use the64 excluded training-domain monitor days for fixed before/after
physical-unit metrics. They are disjoint days, not a geographical/year holdout
or final validation. Keep a fixed training panel too: losses on changing update
batches alone cannot establish improvement. Report GPP, maintenance, growth,
autotrophic and heterotrophic respiration, NPP, AGB/BGB levels and one-day
changes; include error distributions and raw units, not only percent loss.

One persistent process should compile, perform bounded admission, train and
evaluate. Stop on nonfinite updates or violated physical gates, retaining the
last complete checkpoint and failing sample identity. Freeze the actual update
count, monitoring cadence, scientific decision rule, checkpoint selection and
wall-time/cost envelope after the resource results. Do not submit several tiny
paid experiments to obtain these same measurements separately.

Improved one-day fit will not establish free-running carbon stability. The
independent public-rollout admission must pass before full-network multi-day
optimization; later7/30-day trajectories assess accumulated productivity,
respiration and biomass error, followed by the already declared longer gates.
