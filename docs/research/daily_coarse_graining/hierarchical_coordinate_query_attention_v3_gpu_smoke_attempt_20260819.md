# Hierarchical Coordinate-Query Attention v3 GPU Smoke Attempt

Date: 2026-08-19.

Status: execution failure before training. This attempt provides no scientific
evidence for or against the candidate.

## Frozen Identity

- Git commit: `7dd9510103574d1a88a579274b03555a199c2aaf`.
- Node: free `gln01`, host `ibc13b02n04`, Tesla V100 32 GiB, GPU 0.
- Candidate: `hierarchical_coordinate_query_attention_v3`.
- Mode: mechanistic output, isolated `litter_poc_doc`, one-epoch training
  smoke, logical batch size 8.
- Inputs: existing accepted v6 manifest and frozen statistics only. No v6
  regeneration, v5 data, or sealed-test data was used.

## Outcome

The process completed data loading and the initial forward evaluations, then
failed while XLA compiled the separate `jit_coverage_probe`. It completed zero
of the required 46 AdamW updates. XLA reported an original 82.03 GiB graph,
71.95 GiB after rematerialization, and a failed 72.57 GiB allocation on the
32 GiB V100.

The runner did not write `report.json` or a checkpoint. The failure therefore
does not measure loss reduction, trainability, field fit, or long-horizon
behavior.

## Cause And Bounded Repair

The local gate proved a one-day official loss and gradient, but the GPU runner
compiled an eight-day vectorized backward pass for the candidate's 39,006
learned scalar coordinates. The separate coverage probe also repeated a full
backward pass solely to inspect parameter participation. Its activation peak,
not the 58.7-million-parameter tree alone, exceeded the V100 memory envelope.

The retry preserves the declared optimization semantics:

- logical batch size remains 8;
- one epoch remains 46 AdamW updates over all 364 real days;
- learning-rate schedule, loss, initialization, and model are unchanged;
- each logical candidate batch is differentiated as at most four size-2
  microbatches and combined by exact real-sample weighting before one update;
- zero-weight final padding chunks are skipped;
- coverage is read from the first and final real training updates instead of
  compiling two extra backward graphs;
- rejected and non-candidate paths retain their existing direct-batch code.

This is a memory execution repair, not an architecture or training-policy
revision.

The updated qualification/launcher suite passes 20/20 tests. Ruff, Python
compilation, and `git diff --check` also pass.

## Active Gate

Version and deploy the locally accepted memory repair, then run one fresh
free-`gln01` smoke. Only 46 finite updates, decreasing litter loss, unchanged
counts, nonzero selected coordinate-block and field-head gradients and
updates, and zero hard failures may open the frozen 100-epoch isolated
diagnostic. No paid job, rollout, Gate D2, sealed test, or product stage is
open.
